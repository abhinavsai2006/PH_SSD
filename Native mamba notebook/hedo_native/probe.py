"""Mechanism probes for H1 (energy dissipation / background suppression) and H2 (modality dominance).

Per completed run (best checkpoint, test split) writes <run>/probe.json; all rows go to
results/probes.csv.

H1 (all runs; energy terms only for EnergyHEDO):
  saliency   last-block ViT CLS->patch attention, per image. background = tokens at or below the
             image median; foreground = top 25%.
  norm_ratio ||operator(x)|| / ||x|| averaged over bg and fg tokens (identity operator -> 1)
  attenuation / dissipated energy on bg vs fg, Spearman(saliency, attenuation)
  energy_max_increase_float64: largest per-step increase of H over 64 test images in float64
             (Theorem 1 predicts <= 0 up to round-off)
  occlusion  dual test mean recall with background tokens zeroed vs foreground tokens zeroed
H2:
  grad_log10_ratio  mean log10(||dL/dx_img|| / ||dL/dx_txt||) over the last training epoch (|.| = imbalance)
  exchange runs:    message reliance r_img = mean |s(full) - s(no message into image stream)| on the
                    5000 positive test pairs, r_txt likewise; dominance D = |r_img - r_txt| / (r_img + r_txt)
  effective_rank    exp(entropy of normalised eigenvalues) of the test embedding covariance (saturation proxy)

Usage: python probe.py [--runs-glob "*seed*"]
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy import stats

from common import FEATURE_DIR, RUNS_DIR, banner, provenance, read_json, write_json
from evaluation import CachedSplit, evaluate, exchange_scores, pass1_all, load_run_model
from models import EnergyHEDO


def effective_rank(x):
    x = x - x.mean(0, keepdims=True)
    ev = np.clip(np.linalg.eigvalsh(np.cov(x.T)), 0, None)
    p = ev / ev.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def h1_probe(model, test, saliency, device):
    n_images = test.img.shape[0]
    op = model.operator["img"]
    fg_mask = saliency >= np.quantile(saliency, 0.75, axis=1, keepdims=True)
    bg_mask = saliency <= np.median(saliency, axis=1, keepdims=True)
    acc = {"norm_ratio": [], "attenuation": [], "dissipated_fraction": []}
    for s in range(0, n_images, 100):
        rows = np.arange(s, min(n_images, s + 100))
        x = model.proj["img"](test.img_tensor(rows, device))
        if isinstance(op, EnergyHEDO):
            t = op.trajectory(x)
            y = t["out"]
            acc["attenuation"].append(t["attenuation"].cpu().numpy())
            acc["dissipated_fraction"].append(t["dissipated_fraction"].cpu().numpy())
        else:
            y = op(x)
        acc["norm_ratio"].append((y.norm(dim=-1) / x.norm(dim=-1).clamp(min=1e-12)).cpu().numpy())
    res = {}
    fg, bg, sal = fg_mask[:n_images], bg_mask[:n_images], saliency[:n_images]
    for name, chunks in acc.items():
        if not chunks:
            continue
        v = np.concatenate(chunks)
        res[f"{name}_bg"] = float(v[bg].mean())
        res[f"{name}_fg"] = float(v[fg].mean())
        res[f"{name}_spearman_saliency"] = float(stats.spearmanr(sal.reshape(-1), v.reshape(-1)).statistic)
    if isinstance(op, EnergyHEDO):
        op64 = EnergyHEDO(op.U.shape[1], op.U.shape[0], op.steps, op.dt_max, op.adaptive, op.eps).to(device).double()
        op64.load_state_dict({k: v.double() for k, v in op.state_dict().items()})
        e = op64.trajectory(model.proj["img"](test.img_tensor(np.arange(min(64, n_images)), device)).double())["energies"]
        res["energy_max_increase_float64"] = float((e[..., 1:] - e[..., :-1]).max())
        res["energy_mean_initial"] = float(e[..., 0].mean())
        res["energy_mean_final"] = float(e[..., -1].mean())
    return res, fg_mask, bg_mask


def occlusion_metrics(model, test, device, fg_mask, bg_mask):
    res = {}
    for name, m in (("occlude_bg", bg_mask), ("occlude_fg", fg_mask)):
        keep = torch.from_numpy(~m).to(device)

        def transform(feats, rows, keep=keep):
            return feats * keep[torch.as_tensor(rows, device=device)].unsqueeze(-1).to(feats.dtype)

        res[f"{name}_dual_mean_recall"] = evaluate(model, test, device, rerank_k=0, img_transform=transform)[0][
            "dual_mean_recall"]
    return res


@torch.no_grad()
def h2_probe(model, test, device, run_dir):
    res = {}
    log = pd.read_json(run_dir / "batch_log.jsonl", lines=True)
    last = log[log["epoch"] == log["epoch"].max()]
    if {"grad_img_token", "grad_txt_token"} <= set(last.columns):
        r = np.log10(last["grad_img_token"] / last["grad_txt_token"])
        res["grad_log10_ratio"] = float(r.mean())
        res["grad_abs_log10_ratio"] = float(r.abs().mean())
    img, txt, cache = pass1_all(model, test, device)
    res["effective_rank_img"] = effective_rank(img)
    res["effective_rank_txt"] = effective_rank(txt)
    if cache is not None:
        n = txt.shape[0]
        ir, tr = np.arange(n) // 5, np.arange(n)
        full = exchange_scores(model, cache, ir, tr, device)
        no_img = exchange_scores(model, cache, ir, tr, device, message_scale=(0.0, 1.0))
        no_txt = exchange_scores(model, cache, ir, tr, device, message_scale=(1.0, 0.0))
        r_img, r_txt = float(np.abs(full - no_img).mean()), float(np.abs(full - no_txt).mean())
        res.update(reliance_img_on_txt=r_img, reliance_txt_on_img=r_txt,
                   dominance_index=abs(r_img - r_txt) / max(r_img + r_txt, 1e-12),
                   gate_img=float(torch.tanh(model.xhvsc.gate["img"])),
                   gate_txt=float(torch.tanh(model.xhvsc.gate["txt"])))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-glob", default="*")
    args = ap.parse_args()
    device = torch.device(os.environ.get("HEDO_DEVICE", "cuda"))
    banner("MECHANISM PROBES (H1, H2)")
    test = CachedSplit("test")
    saliency = np.load(FEATURE_DIR / "saliency_test.npy")
    rows = []
    for summary_path in sorted(RUNS_DIR.glob(f"{args.runs_glob}/run_summary.json")):
        run_dir = summary_path.parent
        s = read_json(summary_path)
        if s.get("status") != "COMPLETED":
            continue
        model = load_run_model(run_dir, device)
        h1, fg, bg = h1_probe(model, test, saliency, device)
        row = {"run": run_dir.name, "variant": s["variant"], "seed": s["seed"], "kl_weight": s["kl_weight"],
               "hvsc_chunk_size": s["hvsc_chunk_size"], **h1, **occlusion_metrics(model, test, device, fg, bg),
               **h2_probe(model, test, device, run_dir)}
        write_json(run_dir / "probe.json", {**row, "provenance": provenance()})
        rows.append(row)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    out = RUNS_DIR.parent / "results" / "probes.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} runs probed -> {out}")
    print("PROBES: PASS")


if __name__ == "__main__":
    main()
