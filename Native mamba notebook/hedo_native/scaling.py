"""H4: computational overhead and scaling with image sequence length.

For each variant, the trainable head (frozen backbones excluded: they are identical across
variants and fixed at 196 patches) is timed at image lengths L in {196, 784, 3136, 12544}
(patch grids 14^2 .. 112^2) with a 64-token caption:
  * inference: image encoding (pass 1), plus one exchange pair (pass 2) for exchange models
  * training: forward + backward of one 8-image x 40-caption step
  * peak memory for both
Inputs are random tensors with the shape of ViT tokens. This is deliberate: the experiment measures
compute scaling at resolutions the cached ViT features do not exist for, and is labelled as such.
The log-log slope of median latency vs L is fitted on L >= 784 (slope ~1 = linear, ~2 = quadratic).
OOM is recorded, not hidden.

Usage: python scaling.py [--variants baseline full_x transformer_param_matched ...]
"""

import argparse
import os
import statistics
import time

import numpy as np
import pandas as pd
import torch

from common import REPORT_DIR, banner, provenance, write_json
from models import RetrievalModel, build_config, count_trainable, total_loss

LENGTHS = [196, 784, 3136, 12544]
DEFAULT_VARIANTS = ["baseline", "hedo_energy", "hvsc_x", "full_x", "full", "transformer_param_matched", "transformer_x"]


def timed(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--lengths", nargs="+", type=int, default=LENGTHS)
    args = ap.parse_args()
    device = torch.device(os.environ.get("HEDO_DEVICE", "cuda"))
    banner("SCALING WITH SEQUENCE LENGTH (H4)")
    g = torch.Generator(device=device).manual_seed(0)
    rows = []
    for variant in args.variants:
        for L in args.lengths:
            row = {"variant": variant, "img_len": L}
            try:
                cfg = build_config(variant, img_len=L)
                model = RetrievalModel(cfg).to(device)
                row["trainable_params"] = count_trainable(model)
                img1 = torch.randn(1, L, 768, device=device, generator=g)
                img8 = torch.randn(8, L, 768, device=device, generator=g)
                txt = torch.randn(40, 64, 768, device=device, generator=g)
                mask = torch.ones(40, 64, dtype=torch.bool, device=device)
                pair = torch.arange(8, device=device).repeat_interleave(5)

                model.eval()

                @torch.no_grad()
                def infer():
                    _, ai, _ = model.pass1("img", img1, None, sample=False)
                    if cfg.exchanges:
                        _, at, _ = model.pass1("txt", txt[:1], mask[:1], sample=False)
                        model.pair_similarity(ai, at, mask[:1])

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                row["infer_bs1_ms"] = timed(infer)
                row["infer_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20

                model.train()
                opt = torch.optim.AdamW(model.parameters(), lr=0.0)

                def step():
                    opt.zero_grad(set_to_none=True)
                    out = model(img8, txt, mask, pair)
                    total_loss(model, out, pair, 1e-4)[0].backward()
                    opt.step()

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                row["train_step_ms"] = timed(step, warmup=3, iters=10)
                row["train_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
                row["status"] = "ok"
            except torch.cuda.OutOfMemoryError:
                row["status"] = "OOM"
            finally:
                model = opt = img1 = img8 = txt = None  # noqa: F841  release GPU memory before the next size
                torch.cuda.empty_cache()
            rows.append(row)
            print(row, flush=True)

    df = pd.DataFrame(rows)
    slopes = {}
    for variant, grp in df[df["status"] == "ok"].groupby("variant"):
        grp = grp[grp["img_len"] >= 784]
        for col in ("infer_bs1_ms", "train_step_ms"):
            if len(grp) >= 2:
                slopes[f"{variant}|{col}"] = float(np.polyfit(np.log(grp["img_len"]), np.log(grp[col]), 1)[0])
    base = df[(df["variant"] == "baseline") & (df["status"] == "ok")].set_index("img_len")
    if not base.empty:
        for col in ("infer_bs1_ms", "train_step_ms", "train_peak_mb"):
            df[f"{col}_vs_baseline"] = df.apply(
                lambda r: r[col] / base.loc[r["img_len"], col] if r["status"] == "ok" and r["img_len"] in base.index
                else np.nan, axis=1)
    out = REPORT_DIR / "scaling"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "scaling.csv", index=False)
    write_json(out / "scaling.json", {"rows": df.to_dict("records"), "loglog_slopes_L_ge_784": slopes,
                                      "gpu": torch.cuda.get_device_name(0),
                                      "inputs": "random tensors with ViT token shape (compute scaling only)",
                                      "provenance": provenance()})
    print("log-log slopes:", {k: round(v, 3) for k, v in slopes.items()})
    print("SCALING: PASS")


if __name__ == "__main__":
    main()
