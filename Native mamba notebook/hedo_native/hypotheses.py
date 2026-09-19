"""Evaluate hypotheses H1-H4 against criteria fixed in METHOD.md (section 5) *before* the runs.

Inputs (all regenerated from raw artifacts): results/all_runs.csv, results/probes.csv,
results/robustness.csv, report/scaling/scaling.json. Seed-level 95% CIs use the t distribution
over matched seeds. A hypothesis is "supported" only if every criterion passes, "not supported"
if a criterion fails, "insufficient data" if an input is missing or n < 3 seeds.

Output: report/hypotheses.json, report/table_hypotheses.tex
Usage: python hypotheses.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from common import REPORT_DIR, RUNS_DIR, banner, provenance, read_json, write_json
from models import PROPOSED

RESULTS = RUNS_DIR.parent / "results"
NONINFERIORITY_MR = 1.0     # H1(c): ours may not lose more than 1 MR point to the baseline
OVERHEAD_MAX = 1.5          # H4(a): <= 1.5x baseline latency / train step at 196 tokens
SLOPE_MAX = 1.2             # H4(b): log-log slope of latency vs length (1 = linear)
ENERGY_TOL = 1e-9


def ci95(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return [float("nan"), float("nan")]
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def paired(df, col, a, b):
    A = df[df["variant"] == a].set_index("seed")[col]
    B = df[df["variant"] == b].set_index("seed")[col]
    seeds = sorted(set(A.index) & set(B.index))
    d = (A.loc[seeds] - B.loc[seeds]).values if seeds else np.array([])
    return {"a": a, "b": b, "metric": col, "n": len(seeds), "mean_diff": float(d.mean()) if len(d) else None,
            "ci95": ci95(d), "per_seed": dict(zip(map(int, seeds), map(float, d)))}


def verdict(criteria):
    if any(c.get("pass") is None for c in criteria):
        return "insufficient data"
    return "supported" if all(c["pass"] for c in criteria) else "not supported"


def default_runs(df):
    return df[(df["kl_weight"] == 1e-4) & (df["hvsc_chunk_size"] == 16)]


def load_csv(name):
    path = RESULTS / name
    return default_runs(pd.read_csv(path)) if path.is_file() else None


def h1(runs, probes):
    if probes is None:
        return {"criteria": [{"name": "probes available", "pass": None}]}
    ours = probes[probes["variant"] == PROPOSED]
    c = []
    inc = ours["energy_max_increase_float64"].max() if len(ours) else None
    c.append({"name": "(a) energy never increases on real tokens (float64)", "value": inc,
              "pass": None if inc is None or np.isnan(inc) else bool(inc <= ENERGY_TOL)})
    d =(ours["attenuation_fg"] - ours["attenuation_bg"]).values
    lo, hi = ci95(d)
    c.append({"name": "(b) background tokens attenuated more than foreground: mean(att_fg - att_bg) > 0, CI excludes 0",
              "n": len(d), "mean": float(d.mean()) if len(d) else None, "ci95": [lo, hi],
              "spearman_saliency_attenuation": float(ours["attenuation_spearman_saliency"].mean()) if len(d) else None,
              "pass": None if len(d) < 3 else bool(lo > 0)})
    mr = paired(runs, "mean_recall", PROPOSED, "baseline")
    c.append({"name": f"(c1) no loss of semantic information: MR(ours) - MR(baseline) CI lower bound > -{NONINFERIORITY_MR}",
              **mr, "pass": None if mr["n"] < 3 else bool(mr["ci95"][0] > -NONINFERIORITY_MR)})
    gap = (ours["occlude_bg_dual_mean_recall"] - ours["occlude_fg_dual_mean_recall"]).values
    lo, hi = ci95(gap)
    c.append({"name": "(c2) semantics stay in the foreground: MR(background occluded) - MR(foreground occluded) "
                      "CI lower > 0 for ours",
              "n": len(gap), "mean": float(gap.mean()) if len(gap) else None, "ci95": [lo, hi],
              "pass": None if len(gap) < 3 else bool(lo > 0)})
    return {"criteria": c}


def h2(probes):
    if probes is None:
        return {"criteria": [{"name": "probes available", "pass": None}]}
    g = paired(probes, "grad_abs_log10_ratio", PROPOSED, "full_x_noexchange")
    ours = probes[probes["variant"] == PROPOSED]
    return {"criteria": [
        {"name": "(a) exchange reduces gradient imbalance between modalities: |log10 ratio|(ours) - (no exchange) CI upper < 0",
         **g, "pass": None if g["n"] < 3 else bool(g["ci95"][1] < 0)},
        {"name": "(b) exchange is used in both directions: both gates non-zero and dominance index < 0.5 on average",
         "dominance_index_mean": float(ours["dominance_index"].mean()) if "dominance_index" in ours and len(ours) else None,
         "gate_img_mean": float(ours["gate_img"].abs().mean()) if "gate_img" in ours and len(ours) else None,
         "gate_txt_mean": float(ours["gate_txt"].abs().mean()) if "gate_txt" in ours and len(ours) else None,
         "pass": None if len(ours) < 3 or "dominance_index" not in ours else
         bool(ours["dominance_index"].mean() < 0.5 and ours["gate_img"].abs().min() > 1e-3
              and ours["gate_txt"].abs().min() > 1e-3)},
    ]}


def h3(robust):
    if robust is None:
        return {"criteria": [{"name": "robustness results available", "pass": None}]}
    corrupted = robust[robust["corruption"] != "clean"]
    per_run = corrupted.groupby(["variant", "seed", "modality"])["relative_mr"].mean().reset_index()
    crit = []
    for modality in ("image", "text"):
        sub = per_run[per_run["modality"] == modality]
        p = paired(sub, "relative_mr", PROPOSED, "baseline")
        crit.append({"name": f"relative MR under {modality} corruptions: ours - baseline CI lower > 0", **p,
                     "pass": None if p["n"] < 3 else bool(p["ci95"][0] > 0)})
    return {"criteria": crit, "per_corruption_mean": corrupted.groupby(["variant", "corruption"])["relative_mr"]
            .mean().unstack().round(4).to_dict()}


def h4():
    path = REPORT_DIR / "scaling" / "scaling.json"
    if not path.is_file():
        return {"criteria": [{"name": "scaling results available", "pass": None}]}
    sc = read_json(path)
    rows = pd.DataFrame(sc["rows"])
    at196 = rows[(rows["variant"] == PROPOSED) & (rows["img_len"] == 196) & (rows["status"] == "ok")]
    crit = []
    for col in ("infer_bs1_ms_vs_baseline", "train_step_ms_vs_baseline"):
        v = float(at196[col].iloc[0]) if len(at196) and col in at196 else None
        crit.append({"name": f"(a) overhead {col} <= {OVERHEAD_MAX} at 196 tokens", "value": v,
                     "pass": None if v is None else bool(v <= OVERHEAD_MAX)})
    for col in ("infer_bs1_ms", "train_step_ms"):
        slope = sc["loglog_slopes_L_ge_784"].get(f"{PROPOSED}|{col}")
        crit.append({"name": f"(b) near-linear scaling: log-log slope of {col} <= {SLOPE_MAX}", "value": slope,
                     "transformer_slope": sc["loglog_slopes_L_ge_784"].get(f"transformer_x|{col}"),
                     "pass": None if slope is None else bool(slope <= SLOPE_MAX)})
    return {"criteria": crit}


def main():
    banner("HYPOTHESES H1-H4")
    runs = load_csv("all_runs.csv")
    runs = runs[runs["status"] == "COMPLETED"] if runs is not None else None
    probes, robust = load_csv("probes.csv"), load_csv("robustness.csv")
    result = {"H1 energy dissipation suppresses background without losing semantics": h1(runs, probes),
              "H2 exchange reduces modality dominance": h2(probes),
              "H3 robustness to noisy multimodal inputs": h3(robust),
              "H4 small overhead and linear scaling": h4()}
    for name, r in result.items():
        r["verdict"] = verdict(r["criteria"])
        print(f"{r['verdict']:>18s}  {name}")
        for c in r["criteria"]:
            print(f"{'':20s}{'PASS' if c['pass'] else ('n/a' if c['pass'] is None else 'FAIL'):5s} {c['name']}")
    write_json(REPORT_DIR / "hypotheses.json", {"hypotheses": result, "thresholds": {
        "noninferiority_mr": NONINFERIORITY_MR, "overhead_max": OVERHEAD_MAX, "slope_max": SLOPE_MAX,
        "energy_tol": ENERGY_TOL}, "provenance": provenance()})
    lines = ["\\begin{table}[t]", "\\centering",
             "\\caption{Pre-registered hypothesis tests (criteria in Section~5 of the method). CIs: 95\\% over matched seeds.}",
             "\\label{tab:hypotheses}", "\\begin{tabular}{lp{0.62\\linewidth}l}", "\\hline",
             "H & Criterion & Result \\\\", "\\hline"]
    for i, (name, r) in enumerate(result.items(), 1):
        for c in r["criteria"]:
            res = "pass" if c["pass"] else ("n/a" if c["pass"] is None else "fail")
            crit = c["name"].replace("_", "\\_").replace("%", "\\%").replace(">", "$>$").replace("<", "$<$")
            lines.append(f"H{i} & {crit} & {res} \\\\")
        lines.append(f" & \\textbf{{Verdict}} & \\textbf{{{r['verdict']}}} \\\\ \\hline")
    lines += ["\\end{tabular}", "\\end{table}"]
    (REPORT_DIR / "table_hypotheses.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("HYPOTHESES: PASS")


if __name__ == "__main__":
    main()
