"""Train and evaluate one configuration on cached Flickr8k features.

Artifacts in --out:
  config.json, environment.json
  batch_log.jsonl            per-batch numerics: loss parts, energy/exchange stats, modality gradient ratio,
                             grad/param extremes, non-finite counts
  training_history.csv       per-epoch train + validation metrics
  best_model.pt, best_val_metrics.json, model_final.pt
  test_image_embeddings.npy, test_text_embeddings.npy, test_image_ids.json
  test_rerank.npz            exchange models only: top-K candidates and exchange scores
  test_results.json          metrics (primary = re-ranked for exchange models, dual_* always) + checks + provenance
  independent_eval.json      written by evaluate_independent.py (separate process)
  run_summary.json           status COMPLETED only if every check passed
  failure_state.pt           only on a numerical failure (pre-step model/optimizer + batch indices)

Exit codes: 0 ok, 2 numerical failure, 3 learning-sanity failure, 4 verification failure.
Usage: python train.py --variant full_x --seed 42 --epochs 10 --out /content/hedo_work/runs/x
"""

import argparse
import copy
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (PACKAGE_DIR, banner, environment_manifest, provenance, read_json, set_seed, sha256_file,
                    write_json)
from evaluation import CachedSplit, NumericalFailure, evaluate
from metrics import CAPTIONS_PER_IMAGE
from models import RetrievalModel, assert_native_mamba2, build_config, count_trainable, total_loss

IMAGES_PER_BATCH = 8
CHANCE_MEAN_RECALL = float(np.mean([100 * k * 5 / 5000 for k in (1, 5, 10)] + [100 * k / 1000 for k in (1, 5, 10)]))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--kl-weight", type=float, default=1e-4, help="legacy index-KL coupling weight")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hvsc-chunk-size", type=int, default=16)
    ap.add_argument("--rerank-k", type=int, default=16, help="top-K re-ranked by the exchange score")
    ap.add_argument("--instrument-every", type=int, default=1)
    ap.add_argument("--detect-anomaly", action="store_true", help="autograd anomaly detection during epoch 1")
    ap.add_argument("--sanity-min-mr", type=float, default=2.0,
                    help=f"abort if epoch-1 val mean recall is below this (chance ~{CHANCE_MEAN_RECALL:.2f})")
    ap.add_argument("--mmap", action="store_true", help="read training features from disk instead of RAM")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def nonfinite_count(tensors):
    return int(sum((~torch.isfinite(t)).sum().item() for t in tensors))


def modality_grad_norms(out, txt_mask):
    """Mean per-token gradient norm of the loss w.r.t. each modality's projected tokens."""
    gi = out["aux_img"]["proj"].grad
    gt = out["aux_txt"]["proj"].grad
    if gi is None or gt is None:
        return None, None
    img = gi.norm(dim=-1).mean()
    txt = (gt.norm(dim=-1) * txt_mask).sum() / txt_mask.sum().clamp(min=1)
    return img.item(), txt.item()


def train_epoch(model, data, optimizer, scheduler, args, epoch, rng, device, log_file, failure_path):
    model.train()
    n_images = data.img.shape[0]
    perm = rng.permutation(n_images)
    n_batches = n_images // IMAGES_PER_BATCH
    totals = {}
    params = [p for p in model.parameters() if p.requires_grad]
    seen, t0 = 0, time.time()

    for b in range(n_batches):
        rows = perm[b * IMAGES_PER_BATCH:(b + 1) * IMAGES_PER_BATCH]
        img, txt, mask, pair, cap_rows = data.batch(rows, device)
        pre_model = {k: v.detach().clone() for k, v in model.state_dict().items()}
        pre_opt = copy.deepcopy(optimizer.state_dict())
        record = {"epoch": epoch, "batch": b + 1, "lr": scheduler.get_last_lr()[0]}
        failure = None

        optimizer.zero_grad(set_to_none=True)
        out = model(img, txt, mask, pair, sample=True)
        loss, parts = total_loss(model, out, pair, args.kl_weight)
        record.update({k: v.item() for k, v in out["stats"].items()})
        record.update({k: v.item() for k, v in parts.items()})
        record.update(loss=loss.item(), logit_scale=model.scale().item())
        for name, t in (("z_img", out["z_img"]), ("z_txt", out["z_txt"]), ("loss", loss)):
            if not torch.isfinite(t).all():
                failure = f"non-finite {name}"
                break

        if failure is None:
            loss.backward()
            record["grad_img_token"], record["grad_txt_token"] = modality_grad_norms(out, mask)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            grads = [p.grad for p in params if p.grad is not None]
            record["grad_norm_preclip"] = float(grad_norm)
            record["grad_absmax"] = max(float(g.abs().max()) for g in grads)
            record["nonfinite_grads"] = nonfinite_count(grads)
            if record["nonfinite_grads"] or not math.isfinite(record["grad_norm_preclip"]):
                failure = "non-finite gradients"

        if failure is None:
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.logit_scale.clamp_(0.0, math.log(100.0))
                if model.exchange_logit_scale is not None:
                    model.exchange_logit_scale.clamp_(0.0, math.log(100.0))
            record["nonfinite_params"] = nonfinite_count(params)
            if record["nonfinite_params"]:
                failure = "non-finite parameters after step"
            if b % args.instrument_every == 0 or failure:
                record["param_absmax"] = max(float(p.detach().abs().max()) for p in params)
                record["nonfinite_optimizer_state"] = nonfinite_count(
                    [v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v) and v.is_floating_point()])
                if record["nonfinite_optimizer_state"]:
                    failure = "non-finite optimizer state"

        if failure or b % args.instrument_every == 0:
            record["failure"] = failure
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
        if failure:
            torch.save({"epoch": epoch, "batch": b + 1, "failure": failure, "image_rows": rows,
                        "caption_rows": cap_rows, "model_state_before_step": pre_model,
                        "optimizer_state_before_step": pre_opt, "record": record,
                        "config": asdict(model.cfg)}, failure_path)
            raise NumericalFailure(f"{failure} at epoch {epoch} batch {b + 1}; state saved to {failure_path}")

        for k in ["loss", *parts]:
            totals[k] = totals.get(k, 0.0) + record[k]
        seen += txt.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    epoch_out = {f"train_{k}": v / n_batches for k, v in totals.items()}
    epoch_out["logit_scale"] = model.scale().item()
    epoch_out["train_captions_per_sec"] = seen / (time.time() - t0)
    return epoch_out


def metrics_equal(a, b, tol=1e-9):
    return all(abs(a[k] - b[k]) <= tol for k in a)


def main():
    args = parse_args()
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"{out} is not empty; pass --overwrite to clear it")
        for p in out.iterdir():
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(os.environ.get("HEDO_DEVICE", "cuda"))  # override only for CPU debugging
    set_seed(args.seed)
    cfg = build_config(args.variant, hvsc_chunk_size=args.hvsc_chunk_size)
    run_config = {"variant": args.variant, "model": cfg.to_dict(), "seed": args.seed, "epochs": args.epochs,
                  "lr": args.lr, "weight_decay": args.weight_decay, "kl_weight": args.kl_weight,
                  "grad_clip": args.grad_clip, "rerank_k": args.rerank_k, "images_per_batch": IMAGES_PER_BATCH,
                  "captions_per_image": CAPTIONS_PER_IMAGE, "optimizer": "AdamW", "schedule": "cosine per step",
                  "precision": "float32 (trainable stack); backbones cached, float32 compute / float16 storage",
                  "sanity_min_mr": args.sanity_min_mr}
    write_json(out / "config.json", run_config)
    write_json(out / "environment.json", environment_manifest())

    banner(f"RUN {args.variant} seed={args.seed} kl={args.kl_weight} chunk={args.hvsc_chunk_size}")
    model = RetrievalModel(cfg).to(device)
    n_native = assert_native_mamba2(model)
    trainable = count_trainable(model)
    rerank_k = args.rerank_k if cfg.exchanges else 0
    print(f"native Mamba2 modules: {n_native} | trainable params: {trainable:,} | rerank_k: {rerank_k}")
    print(f"model config: {cfg}")

    train = CachedSplit("train", in_memory=not args.mmap)
    val = CachedSplit("val")
    test = CachedSplit("test")
    rng = np.random.default_rng(args.seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = (train.img.shape[0] // IMAGES_PER_BATCH) * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history, best_mr, best_epoch = [], -1.0, 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    summary = {"status": "RUNNING", "variant": args.variant, "seed": args.seed, "kl_weight": args.kl_weight,
               "hvsc_chunk_size": args.hvsc_chunk_size, "trainable_params": trainable}
    t_start = time.time()

    try:
        with open(out / "batch_log.jsonl", "w", encoding="utf-8") as log_file:
            for epoch in range(1, args.epochs + 1):
                t0 = time.time()
                with torch.autograd.set_detect_anomaly(args.detect_anomaly and epoch == 1):
                    tr = train_epoch(model, train, optimizer, scheduler, args, epoch, rng, device, log_file,
                                     out / "failure_state.pt")
                val_metrics = evaluate(model, val, device, rerank_k)[0]
                row = {"epoch": epoch, **tr, **{f"val_{k}": v for k, v in val_metrics.items()},
                       "seconds": time.time() - t0}
                history.append(row)
                pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
                print(f"epoch {epoch:02d}/{args.epochs} loss={tr['train_loss']:.5f} "
                      f"infonce={tr['train_infonce']:.5f} val_MR={val_metrics['mean_recall']:.3f} "
                      f"(dual {val_metrics['dual_mean_recall']:.3f}) ({row['seconds']:.0f}s)", flush=True)

                if epoch == 1 and val_metrics["mean_recall"] < args.sanity_min_mr:
                    summary.update(status="FAILED_LEARNING_SANITY", epoch1_val_mr=val_metrics["mean_recall"],
                                   chance_mean_recall=CHANCE_MEAN_RECALL)
                    write_json(out / "run_summary.json", summary)
                    print(f"LEARNING_SANITY: FAIL (val MR {val_metrics['mean_recall']:.3f} < {args.sanity_min_mr}; "
                          f"chance {CHANCE_MEAN_RECALL:.3f})")
                    sys.exit(3)

                if val_metrics["mean_recall"] > best_mr:
                    best_mr, best_epoch = val_metrics["mean_recall"], epoch
                    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": epoch},
                               out / "best_model.pt")
                    write_json(out / "best_val_metrics.json", {"best_epoch": epoch, **val_metrics})
    except NumericalFailure as exc:
        summary.update(status="FAILED_NUMERICAL", error=str(exc))
        write_json(out / "run_summary.json", summary)
        print("NUMERICAL_FAILURE:", exc)
        sys.exit(2)

    train_seconds = time.time() - t_start
    torch.save({"model": model.state_dict(), "config": cfg.to_dict(), "epoch": args.epochs}, out / "model_final.pt")

    banner("VERIFICATION")
    checks = {}
    ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    reloaded_val = evaluate(model, val, device, rerank_k)[0]
    stored_val = {k: v for k, v in read_json(out / "best_val_metrics.json").items() if k != "best_epoch"}
    checks["checkpoint_reload_matches_best_val"] = metrics_equal(reloaded_val, stored_val)

    test_metrics, img1, txt1, rr1, _ = evaluate(model, test, device, rerank_k)
    _, img2, txt2, rr2, _ = evaluate(model, test, device, rerank_k)
    det_abs = float(max(np.abs(img1 - img2).max(), np.abs(txt1 - txt2).max()))
    if rr1 is not None:
        same_candidates = np.array_equal(rr1["i2t_idx"], rr2["i2t_idx"]) and np.array_equal(rr1["t2i_idx"], rr2["t2i_idx"])
        det_abs = max(det_abs, float(np.abs(rr1["i2t_score"] - rr2["i2t_score"]).max()),
                      float(np.abs(rr1["t2i_score"] - rr2["t2i_score"]).max())) if same_candidates else float("inf")
    checks["deterministic_inference"] = det_abs <= 1e-6
    checks["embeddings_unit_norm"] = bool(np.allclose(np.linalg.norm(img1, axis=1), 1, atol=1e-4)
                                          and np.allclose(np.linalg.norm(txt1, axis=1), 1, atol=1e-4))

    np.save(out / "test_image_embeddings.npy", img1)
    np.save(out / "test_text_embeddings.npy", txt1)
    if rr1 is not None:
        np.savez(out / "test_rerank.npz", **rr1)
    write_json(out / "test_image_ids.json", test.image_ids)

    artifacts = ["test_image_embeddings.npy", "test_text_embeddings.npy"] + (["test_rerank.npz"] if rr1 else [])
    results = {
        **test_metrics,
        "reranked": rr1 is not None,
        "rerank_k": rerank_k,
        "best_epoch": best_epoch,
        "best_val_mean_recall": best_mr,
        "chance_mean_recall": CHANCE_MEAN_RECALL,
        "checks": checks,
        "determinism_max_abs_diff": det_abs,
        "trainable_params": trainable,
        "train_seconds": train_seconds,
        "peak_train_memory_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
        "artifact_sha256": {name: sha256_file(out / name) for name in artifacts},
        "provenance": provenance(run_config),
    }
    write_json(out / "test_results.json", results)
    for k in ("mean_recall", "dual_mean_recall", "i2t_r1", "t2i_r1"):
        print(f"{k:18s} {test_metrics[k]:.4f}")

    indep = subprocess.run([sys.executable, str(PACKAGE_DIR / "evaluate_independent.py"), "--run", str(out)],
                           capture_output=True, text=True)
    print(indep.stdout.strip(), indep.stderr.strip()[-2000:])
    checks["independent_evaluator"] = indep.returncode == 0

    for k, v in checks.items():
        print(f"{k:40s} {'PASS' if v else 'FAIL'}")
    summary.update(test_results_sha256=sha256_file(out / "test_results.json"), checks=checks,
                   mean_recall=test_metrics["mean_recall"], code_sha256=results["provenance"]["code_sha256"],
                   config_sha256=results["provenance"]["config_sha256"],
                   feature_manifest_sha256=results["provenance"].get("feature_manifest_sha256"))
    summary["status"] = "COMPLETED" if all(checks.values()) else "FAILED_VERIFICATION"
    write_json(out / "run_summary.json", summary)
    print("RUN_STATUS:", summary["status"])
    if summary["status"] != "COMPLETED":
        sys.exit(4)


if __name__ == "__main__":
    main()
