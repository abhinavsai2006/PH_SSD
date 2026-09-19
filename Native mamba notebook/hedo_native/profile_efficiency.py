"""End-to-end efficiency profile on real Flickr8k test inputs, same GPU, same batch sizes.

For each variant it reports
  * parameters: trainable head, frozen ViT, frozen RoBERTa, total
  * inference latency (median / p90 ms) and throughput for: ViT, RoBERTa, head, and the
    full pipeline (image + caption -> embeddings), at batch sizes 1 and 32
  * peak inference memory above the loaded weights
  * training throughput of the head on cached features (captions/s), 8 images x 5 captions
  * FLOPs from torch.utils.flop_counter. Custom Triton/CUDA kernels (Mamba-2 selective scan,
    causal conv1d) are NOT counted, so FLOPs for Mamba-2 variants are a lower bound.

Inputs are real preprocessed test images and real tokenised test captions, never random tensors.
If a trained seed-42 checkpoint exists it is loaded (weights do not change timing, but numerics stay realistic).

Usage: python profile_efficiency.py [--variants baseline full ...] [--batch-sizes 1 32]
"""

import argparse
import statistics
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.flop_counter import FlopCounterMode

from backbones import FrozenRoBERTa, FrozenViT, vit_transform
from common import DATA_DIR, FEATURE_DIR, REPORT_DIR, RUNS_DIR, banner, environment_manifest, provenance, read_json, write_json
from models import RetrievalModel, build_config, count_trainable, total_loss

DEFAULT_VARIANTS = ["baseline", "hedo_energy", "hvsc_x", "full_x", "full", "baseline_param_matched_x",
                    "no_mixer_meanpool", "transformer_param_matched", "transformer_x"]


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": statistics.median(times), "p90_ms": times[int(0.9 * (len(times) - 1))],
            "mean_ms": statistics.fmean(times)}


def flops(fn):
    with FlopCounterMode(display=False) as counter:
        fn()
    return int(counter.get_total_flops())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 32])
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    banner("END-TO-END EFFICIENCY PROFILE")

    manifest = read_json(DATA_DIR / "manifest.json")
    table = pd.read_csv(DATA_DIR / "captions_test.csv", keep_default_na=False)
    max_bs = max(args.batch_sizes)
    tf = vit_transform()
    images = []
    for iid in table["image_id"].iloc[::5].tolist()[:max_bs]:
        with Image.open(f"{manifest['image_dir']}/{iid}") as im:
            images.append(tf(im.convert("RGB")))
    images = torch.stack(images).to(device)
    ids = torch.from_numpy(np.load(FEATURE_DIR / "ids_test.npy")[:max_bs]).long().to(device)
    mask = torch.from_numpy(np.load(FEATURE_DIR / "mask_test.npy")[:max_bs]).long().to(device)

    vit, roberta = FrozenViT().to(device), FrozenRoBERTa().to(device)
    frozen = {"vit": sum(p.numel() for p in vit.parameters()), "roberta": sum(p.numel() for p in roberta.parameters())}

    train_img = np.load(FEATURE_DIR / "img_train.npy", mmap_mode="r")
    train_txt = np.load(FEATURE_DIR / "txt_train.npy", mmap_mode="r")
    train_mask = np.load(FEATURE_DIR / "mask_train.npy", mmap_mode="r")
    t_img = torch.from_numpy(np.asarray(train_img[:8], dtype=np.float32)).to(device)
    t_txt = torch.from_numpy(np.asarray(train_txt[:40], dtype=np.float32)).to(device)
    t_mask = torch.from_numpy(np.asarray(train_mask[:40]).astype(bool)).to(device)
    pair = torch.arange(8, device=device).repeat_interleave(5)

    results = []
    for variant in args.variants:
        model = RetrievalModel(build_config(variant)).to(device)
        ckpt = RUNS_DIR / f"{variant}__kl0.0001__chunk16__seed42" / "best_model.pt"
        if ckpt.is_file():
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True)["model"])
        model.eval()
        row = {"variant": variant, "trained_weights": ckpt.is_file(), "trainable_params": count_trainable(model),
               "frozen_vit_params": frozen["vit"], "frozen_roberta_params": frozen["roberta"]}
        row["total_params"] = row["trainable_params"] + frozen["vit"] + frozen["roberta"]

        for bs in args.batch_sizes:
            im, tid, tm = images[:bs], ids[:bs], mask[:bs]
            feats = {}

            def run_vit():
                feats["img"] = vit(im)

            def run_roberta():
                feats["txt"] = roberta(tid, tm)

            run_vit()
            run_roberta()

            @torch.no_grad()
            def head(img_feats, txt_feats):
                # dual encoding; exchange models add one pass-2 score per image-caption pair (re-ranking cost)
                _, ai, _ = model.pass1("img", img_feats, None, sample=False)
                _, at, _ = model.pass1("txt", txt_feats, tm.bool(), sample=False)
                if model.cfg.exchanges:
                    model.pair_similarity(ai, at, tm.bool())

            def run_head():
                head(feats["img"], feats["txt"])

            def run_e2e():
                head(vit(im), roberta(tid, tm))

            for name, fn in (("vit", run_vit), ("roberta", run_roberta), ("head", run_head)):
                t = timed(fn, args.warmup, args.iters)
                row[f"bs{bs}_{name}_median_ms"] = t["median_ms"]
            torch.cuda.empty_cache()
            base_mem = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            t = timed(run_e2e, args.warmup, args.iters)
            row[f"bs{bs}_e2e_median_ms"] = t["median_ms"]
            row[f"bs{bs}_e2e_p90_ms"] = t["p90_ms"]
            row[f"bs{bs}_e2e_pairs_per_sec"] = bs / (t["median_ms"] / 1000)
            row[f"bs{bs}_e2e_peak_mem_above_weights_mb"] = (torch.cuda.max_memory_allocated() - base_mem) / 2**20
            if bs == 1:
                row["flops_head_bs1"] = flops(run_head)
                row["flops_e2e_bs1"] = flops(run_e2e)

        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=0.0)

        def train_step():
            opt.zero_grad(set_to_none=True)
            out = model(t_img, t_txt, t_mask, pair, sample=True)
            total_loss(model, out, pair, 1e-4)[0].backward()
            opt.step()

        torch.cuda.reset_peak_memory_stats()
        t = timed(train_step, args.warmup, args.iters)
        row["train_step_median_ms"] = t["median_ms"]
        row["train_captions_per_sec"] = 40 / (t["median_ms"] / 1000)
        row["train_step_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        results.append(row)
        print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
        del model, opt
        torch.cuda.empty_cache()

    out = REPORT_DIR / "efficiency"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out / "efficiency_profile.csv", index=False)
    write_json(out / "efficiency_profile.json", {
        "rows": results,
        "gpu": torch.cuda.get_device_name(0),
        "precision": "float32 for backbones and head",
        "flops_note": "torch.utils.flop_counter does not count custom Triton/CUDA kernels (Mamba-2 SSD scan, "
                      "causal conv1d); Mamba-2 FLOPs are a lower bound.",
        "inputs": "real Flickr8k test images (official ViT transform) and tokenised test captions",
        "environment": environment_manifest(),
        "provenance": provenance(),
    })

    bs = max(args.batch_sizes)
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{End-to-end efficiency on {torch.cuda.get_device_name(0)}, FP32, real Flickr8k inputs. "
             f"Latency: median ms for batch {bs} image--caption pairs including frozen backbones. "
             "FLOPs exclude custom Mamba-2 kernels (lower bound).}",
             "\\label{tab:efficiency}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lrrrrrr}", "\\hline",
             "Model & Trainable & Total params & Head ms & E2E ms & Pairs/s & Train capt./s \\\\", "\\hline"]
    for r in results:
        name = r["variant"].replace("_", "\\_")
        lines.append(f"{name} & {r['trainable_params']:,} & {r['total_params'] / 1e6:.1f}M & "
                     f"{r[f'bs{bs}_head_median_ms']:.2f} & {r[f'bs{bs}_e2e_median_ms']:.1f} & "
                     f"{r[f'bs{bs}_e2e_pairs_per_sec']:.0f} & {r['train_captions_per_sec']:.0f} \\\\")
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    (out / "table_efficiency.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("EFFICIENCY_PROFILE: PASS")


if __name__ == "__main__":
    main()
