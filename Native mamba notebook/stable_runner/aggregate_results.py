"""Aggregate native Mamba-2 HEDO-HVSC runs into paper tables. Reads only raw run artifacts.

Inputs : RESULTS_ROOT/seed{42,43,44}/{baseline,mamba2_hedo,mamba2_hvsc,full_hedo_hvsc}/
Outputs: aggregate_results.csv      one row per run directory (used_in_aggregate says why/why not)
         ablation_results.csv       per configuration mean and sample SD (ddof=1) over seeds
         paired_deltas.csv          per-seed deltas vs baseline, mean, SD, 95% t-interval
         efficiency_results.csv     per configuration mean over seeds
         invalid_results_registry.json   static registry + every non-completed/stale run found
         claim_evidence_map.md      claim -> evidence -> experiment -> artifact, verdict by fixed rules
         plots/                     per-run training curves + mean +- SD validation curves
A run is used only if run_status is COMPLETED and its methodology hash equals final_methodology_hash.txt.
Usage: python aggregate_results.py --results-root /content/HEDO_HVSC_NATIVE_MAMBA2_BENCHMARK_V6/results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEEDS = [42, 43, 44]
CONFIGS = ["baseline", "mamba2_hedo", "mamba2_hvsc", "full_hedo_hvsc"]
METRICS = ["i2t_r1", "i2t_r5", "i2t_r10", "i2t_medr", "i2t_meanr", "t2i_r1", "t2i_r5", "t2i_r10", "t2i_medr",
           "t2i_meanr", "mean_recall"]
EFFICIENCY = ["trainable_params", "total_params", "peak_train_vram_mb", "train_seconds_total",
              "inference_head_ms_per_40_pairs", "inference_e2e_ms_per_40_pairs", "throughput_pairs_per_s_e2e",
              "flops_head_bs1"]
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ci95(values):
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    h = T975.get(len(v) - 1, 1.96) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean() - h), float(v.mean() + h)


def plot_runs(rows, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = root / "plots"
    out.mkdir(exist_ok=True)
    panels = [("train_total_loss", "train loss"), ("infonce", "InfoNCE"), ("kl_raw", "KL raw"),
              ("kl_weighted", "KL weighted"), ("grad_norm_mean", "grad norm (pre-clip)"), ("lr_start", "learning rate"),
              ("logit_scale", "logit scale"), ("val_i2t_r1", "val I2T R@1"), ("val_t2i_r1", "val T2I R@1"),
              ("val_mean_recall", "val mean recall")]
    histories = {}
    for r in rows:
        h = pd.read_csv(Path(r["run_dir"]) / "epoch_metrics.csv")
        histories[(r["config"], r["seed"])] = h
        fig, axes = plt.subplots(2, 5, figsize=(18, 6))
        for ax, (col, title) in zip(axes.flat, panels):
            if col in h:
                ax.plot(h["epoch"], h[col], marker="o")
            ax.set_title(title)
            ax.set_xlabel("epoch")
        fig.suptitle(f"{r['config']} seed {r['seed']} (validation only; test never used for selection)")
        fig.tight_layout()
        fig.savefig(out / f"curves_{r['config']}_seed{r['seed']}.png", dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for cfg in CONFIGS:
        hs = [h for (c, _), h in histories.items() if c == cfg]
        if not hs:
            continue
        m = np.array([h["val_mean_recall"].values for h in hs])
        x = hs[0]["epoch"].values
        ax.plot(x, m.mean(0), label=f"{cfg} (n={len(hs)})")
        if len(hs) > 1:
            ax.fill_between(x, m.mean(0) - m.std(0, ddof=1), m.mean(0) + m.std(0, ddof=1), alpha=0.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation mean recall (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "val_mean_recall_mean_sd.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", required=True)
    args = ap.parse_args()
    root = Path(args.results_root)
    final_hash = (root / "final_methodology_hash.txt").read_text().strip()

    rows, registry = [], load(Path(__file__).resolve().parent / "invalid_results_registry.json")
    for seed_dir in sorted(root.glob("seed*")):
        for run_dir in sorted(p for p in seed_dir.iterdir() if p.is_dir()):
            status_path = run_dir / "run_status.json"
            status = load(status_path) if status_path.is_file() else {"status": "NO_STATUS"}
            config = run_dir.name.split("__")[0]
            row = {"run_dir": str(run_dir), "config": config, "seed": int(seed_dir.name[4:]),
                   "status": status.get("status"), "methodology_hash": status.get("methodology_hash")}
            reasons = []
            if "__superseded_" in run_dir.name:
                reasons.append("superseded attempt")
            if status.get("status") != "COMPLETED":
                reasons.append(f"status {status.get('status')}")
            if status.get("methodology_hash") != final_hash:
                reasons.append("methodology hash differs from final_methodology_hash.txt")
            row["used_in_aggregate"] = not reasons
            row["excluded_reason"] = "; ".join(reasons)
            if not reasons:
                test = load(run_dir / "test_results.json")
                eff = load(run_dir / "efficiency.json")
                row.update({k: test[k] for k in METRICS})
                row.update(best_epoch=test["best_epoch"], best_val_mean_recall=test["best_val_mean_recall"],
                           test_evaluated_on=test["TEST_EVALUATED_ON"])
                row.update({k: eff.get(k) for k in EFFICIENCY})
            else:
                registry.append({"run_id": f"{config}_seed{row['seed']}_{run_dir.name}", "date": None,
                                 "configuration": config, "source": str(run_dir), "status": "FAILED" if
                                 status.get("status", "").startswith("FAILED") else "INVALID",
                                 "reason": row["excluded_reason"], "usable_as_result": False})
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(root / "aggregate_results.csv", index=False)
    with open(root / "invalid_results_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)
    used = df[df["used_in_aggregate"]] if len(df) else df

    ablation = []
    for cfg in CONFIGS:
        sub = used[used["config"] == cfg] if len(used) else used
        rec = {"config": cfg, "n_seeds": len(sub), "seeds": sorted(sub["seed"].tolist()) if len(sub) else [],
               "complete": sorted(sub["seed"].tolist()) == SEEDS if len(sub) else False}
        for m in METRICS + ["best_epoch"]:
            vals = sub[m].astype(float) if len(sub) else pd.Series(dtype=float)
            rec[f"{m}_mean"] = vals.mean() if len(vals) else np.nan
            rec[f"{m}_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
        ablation.append(rec)
    pd.DataFrame(ablation).to_csv(root / "ablation_results.csv", index=False)

    deltas = []
    for cfg in CONFIGS[1:]:
        a = used[used["config"] == cfg].set_index("seed") if len(used) else pd.DataFrame()
        b = used[used["config"] == "baseline"].set_index("seed") if len(used) else pd.DataFrame()
        seeds = sorted(set(a.index) & set(b.index)) if len(a) and len(b) else []
        for m in ["mean_recall", "i2t_r1", "t2i_r1"]:
            d = (a.loc[seeds, m] - b.loc[seeds, m]).astype(float).values if seeds else np.array([])
            lo, hi = ci95(d)
            deltas.append({"comparison": f"{cfg} - baseline", "metric": m, "n_paired_seeds": len(d),
                           "per_seed": json.dumps(dict(zip(map(int, seeds), np.round(d, 4).tolist()))),
                           "mean_delta": float(d.mean()) if len(d) else np.nan,
                           "sd_delta": float(d.std(ddof=1)) if len(d) > 1 else np.nan, "ci95_low": lo, "ci95_high": hi,
                           "ci_excludes_zero": bool(len(d) >= 2 and (lo > 0 or hi < 0))})
    deltas_df = pd.DataFrame(deltas)
    deltas_df.to_csv(root / "paired_deltas.csv", index=False)

    eff_rows = []
    for cfg in CONFIGS:
        sub = used[used["config"] == cfg] if len(used) else used
        rec = {"config": cfg, "n_seeds": len(sub)}
        for k in EFFICIENCY:
            vals = pd.to_numeric(sub[k], errors="coerce") if len(sub) else pd.Series(dtype=float)
            rec[k] = vals.mean() if len(vals) else np.nan
        eff_rows.append(rec)
    pd.DataFrame(eff_rows).to_csv(root / "efficiency_results.csv", index=False)

    lines = ["# Claim -> evidence -> experiment -> artifact", "",
             f"Methodology hash: `{final_hash}`", "",
             "Decision rules (fixed in advance): SUPPORTED only with all 3 paired seeds and a 95% t-interval of "
             "the mean-recall delta entirely above 0; CONTRADICTED if entirely below 0; otherwise NOT SUPPORTED "
             "(no measurable difference) or INSUFFICIENT EVIDENCE (fewer than 3 paired seeds).", ""]
    for cfg, claim in (("mamba2_hedo", "HEDO improves retrieval over the Mamba-2 baseline"),
                       ("mamba2_hvsc", "HVSC improves retrieval over the Mamba-2 baseline"),
                       ("full_hedo_hvsc", "HEDO + HVSC improves retrieval over the Mamba-2 baseline")):
        r = deltas_df[(deltas_df["comparison"] == f"{cfg} - baseline") & (deltas_df["metric"] == "mean_recall")]
        r = r.iloc[0] if len(r) else None
        if r is None or r["n_paired_seeds"] < 3:
            verdict = "INSUFFICIENT EVIDENCE"
        elif r["ci95_low"] > 0:
            verdict = "SUPPORTED"
        elif r["ci95_high"] < 0:
            verdict = "CONTRADICTED"
        else:
            verdict = "NOT SUPPORTED (difference within seed variation)"
        lines += [f"## {claim}", f"- evidence: paired per-seed mean-recall deltas: {None if r is None else r['per_seed']}; "
                  f"mean {None if r is None else round(r['mean_delta'], 4)}, 95% CI "
                  f"[{None if r is None else round(r['ci95_low'], 4)}, {None if r is None else round(r['ci95_high'], 4)}]",
                  f"- experiment: baseline vs {cfg}, seeds {SEEDS}", "- artifact: paired_deltas.csv, ablation_results.csv",
                  f"- verdict: **{verdict}**", ""]
    lines += ["## Efficiency", "- evidence: efficiency_results.csv (trainable/total params, peak VRAM, time, latency, "
              "throughput, FLOPs lower bound). Parameter count alone is not an efficiency claim.",
              "- verdict: report measured overhead only; no 'efficient' claim without a matched comparison.", "",
              "## Terminology constraints",
              "- HEDO: 'Hamiltonian-inspired discrete dissipative coordinate-momentum transformation' "
              "(no symplecticity, no conservation, no guaranteed dissipation).",
              "- HVSC: 'aligns modality-specific boundary-state posterior distributions using symmetric KL regularization' "
              "(not a formal information bottleneck, not hierarchical, no cross-modal state feedback)."]
    (root / "claim_evidence_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if len(used):
        plot_runs(used.to_dict("records"), root)
    print(f"{len(df)} run dirs, {int(df['used_in_aggregate'].sum()) if len(df) else 0} used; outputs in {root}")
    print("AGGREGATION: PASS")


if __name__ == "__main__":
    main()
