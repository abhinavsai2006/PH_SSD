"""Aggregate completed runs into tables, figures and statistics.

Uses only runs that are COMPLETED, produced by the current code, and whose
independent evaluation passed. Everything is regenerated from raw run
artifacts; nothing is typed in by hand.

Statistics (see statistical_analysis.json):
  * per-seed values, mean +- SD across seeds
  * paired differences on common seeds: mean, SD, d_z (= mean diff / SD diff)
  * paired t-test only for n >= 3 and labelled indicative; Holm-adjusted across the family
  * Wilcoxon signed-rank only for n >= 6 (with n <= 5 it cannot reach p < 0.05)
  * paired bootstrap over test *images* (their 5 captions move with them): test-set
    sampling uncertainty, complementary to seed variance

Usage: python analyze.py [--bootstrap 2000]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from common import REPORT_DIR, RUNS_DIR, banner, code_sha256, provenance, read_json, write_json
from metrics import CAPTIONS_PER_IMAGE, RECALL_KS, reranked_ranks, retrieval_ranks

METRICS = ["i2t_r1", "i2t_r5", "i2t_r10", "t2i_r1", "t2i_r5", "t2i_r10", "mean_recall", "dual_mean_recall"]
CORE = ["baseline", "hedo_energy", "hvsc_x", "full_x"]
LEGACY_CORE = ["baseline", "hedo", "hvsc", "full"]
LABELS = {
    "baseline": "Mamba-2", "hedo": "Mamba-2 + HEDO", "hvsc": "Mamba-2 + HVSC", "full": "Mamba-2 + HEDO + HVSC",
    "full_linear_operator": "Full, HEDO $\\rightarrow$ linear stack",
    "full_hvsc_deterministic": "Full, HVSC w/o sampling/KL",
    "baseline_param_matched": "Mamba-2, param-matched head",
    "no_mixer_meanpool": "No mixer (mean-pool)",
    "transformer_param_matched": "Transformer, param-matched",
    "hedo_energy": "Mamba-2 + E-HEDO", "hvsc_x": "Mamba-2 + X-HVSC", "full_x": "Mamba-2 + E-HEDO + X-HVSC (ours)",
    "full_x_affine": "Ours, E-HEDO $\\rightarrow$ affine HEDO", "full_x_constdamp": "Ours, constant damping",
    "full_x_noexchange": "Ours, no exchange (bottleneck only)", "full_x_noprior": "Ours, no prior KL",
    "baseline_param_matched_x": "Mamba-2, param-matched to ours", "transformer_x": "Ours with Transformer mixer",
}
COMPARISONS = [
    ("full_x", "baseline"), ("hedo_energy", "baseline"), ("hvsc_x", "baseline"), ("full_x", "hedo_energy"),
    ("full_x", "hvsc_x"), ("full_x", "full_x_affine"), ("full_x", "full_x_constdamp"), ("full_x", "full_x_noexchange"),
    ("full_x", "full_x_noprior"), ("full_x", "baseline_param_matched_x"), ("full_x", "transformer_x"),
    ("full_x", "full"),
    ("full", "baseline"), ("hedo", "baseline"), ("hvsc", "baseline"), ("full", "hedo"), ("full", "hvsc"),
    ("full", "full_linear_operator"), ("full", "full_hvsc_deterministic"), ("full", "baseline_param_matched"),
    ("baseline", "no_mixer_meanpool"), ("baseline", "transformer_param_matched"),
]


def key(row):
    return f"{row['variant']}|kl{row['kl_weight']:g}|c{int(row['hvsc_chunk_size'])}"


def load_runs(allow_stale):
    df = pd.read_csv(RUNS_DIR.parent / "results" / "all_runs.csv")
    df = df[df["status"] == "COMPLETED"].copy()
    if not allow_stale:
        df = df[df["current_code"] == True]  # noqa: E712
    keep = []
    for _, r in df.iterrows():
        ind = RUNS_DIR / r["run"] / "independent_eval.json"
        keep.append(ind.is_file() and read_json(ind)["pass"])
    df = df[keep].copy()
    df["group"] = df.apply(key, axis=1)
    return df


def default_group(variant):
    return f"{variant}|kl0.0001|c16"


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[idx]))
        adj[idx] = running
    return adj


def mean_recall_from_ranks(i2t, t2i):
    return np.mean([np.mean(r < k) * 100 for r in (i2t, t2i) for k in RECALL_KS])


def paired_stats(df, a, b, n_boot, rng_seed=0):
    A = df[df["group"] == a].set_index("seed")
    B = df[df["group"] == b].set_index("seed")
    seeds = sorted(set(A.index) & set(B.index))
    res = {"a": a, "b": b, "seeds": seeds, "n": len(seeds)}
    if not seeds:
        return res
    d = A.loc[seeds, "mean_recall"].values - B.loc[seeds, "mean_recall"].values
    res.update(mean_diff=float(d.mean()), sd_diff=float(d.std(ddof=1)) if len(d) > 1 else None,
               per_seed_diff=dict(zip(map(int, seeds), map(float, d))))
    if len(d) > 1 and d.std(ddof=1) > 0:
        res["d_z"] = float(d.mean() / d.std(ddof=1))
    if len(d) >= 3:
        res["t_test_p"] = float(stats.ttest_rel(A.loc[seeds, "mean_recall"], B.loc[seeds, "mean_recall"]).pvalue)
        res["t_test_note"] = f"indicative only: n={len(d)} seeds, df={len(d) - 1}"
    if len(d) >= 6:
        res["wilcoxon_p"] = float(stats.wilcoxon(d).pvalue)

    if n_boot:
        # paired bootstrap over test images, same resamples for every seed and both groups
        ranks = {}
        for s in seeds:
            for g, frame in (("a", A), ("b", B)):
                run = RUNS_DIR / frame.loc[s, "run"]
                emb = (np.load(run / "test_image_embeddings.npy"), np.load(run / "test_text_embeddings.npy"))
                rr_path = run / "test_rerank.npz"
                ranks[(g, s)] = (reranked_ranks(*emb, dict(np.load(rr_path))) if rr_path.is_file()
                                 else retrieval_ranks(*emb))
        n_img = len(ranks[("a", seeds[0])][0])
        rng = np.random.default_rng(rng_seed)
        diffs = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.integers(0, n_img, n_img)
            cap = (idx[:, None] * CAPTIONS_PER_IMAGE + np.arange(CAPTIONS_PER_IMAGE)).reshape(-1)
            per_seed = [mean_recall_from_ranks(ranks[("a", s)][0][idx], ranks[("a", s)][1][cap])
                        - mean_recall_from_ranks(ranks[("b", s)][0][idx], ranks[("b", s)][1][cap]) for s in seeds]
            diffs[i] = np.mean(per_seed)
        res["bootstrap_test_images"] = {"n_resamples": n_boot, "mean_diff": float(diffs.mean()),
                                        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                                        "frac_le_0": float(np.mean(diffs <= 0))}
    return res


def fmt(mean, sd):
    return f"{mean:.2f} $\\pm$ {sd:.2f}" if not np.isnan(sd) else f"{mean:.2f}"


def latex_table(summary, groups, caption, label):
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{lccccccccc}", "\\hline",
             "Model & $n$ & Params & I2T R@1 & I2T R@5 & I2T R@10 & T2I R@1 & T2I R@5 & T2I R@10 & MR \\\\",
             "\\hline"]
    for g in groups:
        if g not in summary.index:
            continue
        r = summary.loc[g]
        variant, kl, chunk = g.split("|")
        name = LABELS.get(variant, variant)
        if kl != "kl0.0001" or chunk != "c16":
            name += f" ({kl.replace('kl', 'KL=')}, {chunk.replace('c', 'chunk=')})"
        cells = [fmt(r[(m, "mean")], r[(m, "std")]) for m in METRICS[:7]]
        lines.append(f"{name} & {int(r[('seed', 'count')])} & {int(r[('trainable_params', 'mean')]):,} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines) + "\n"


def figures(df, summary, out):
    groups = [default_group(v) for v in CORE if default_group(v) in summary.index]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, g in enumerate(groups):
        vals = df[df["group"] == g]["mean_recall"].values
        ax.bar(i, vals.mean(), yerr=vals.std(ddof=1) if len(vals) > 1 else 0, capsize=4, color="#4c72b0", alpha=0.6)
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.15, 0.15, len(vals)), vals, color="black", s=12, zorder=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([LABELS[g.split("|")[0]] for g in groups], rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Test mean recall (%)")
    ax.set_title("Flickr8k test, mean $\\pm$ SD over seeds (dots = seeds)", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_mean_recall.{ext}", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for g in groups:
        hist = [pd.read_csv(RUNS_DIR / r / "training_history.csv") for r in df[df["group"] == g]["run"]]
        for ax, col, ylabel in ((axes[0], "train_infonce", "Train InfoNCE (pass 1)"), (axes[1], "val_mean_recall", "Val mean recall (%)")):
            m = np.array([h[col].values for h in hist])
            x = hist[0]["epoch"].values
            ax.plot(x, m.mean(0), label=LABELS[g.split("|")[0]])
            if len(m) > 1:
                ax.fill_between(x, m.mean(0) - m.std(0, ddof=1), m.mean(0) + m.std(0, ddof=1), alpha=0.2)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"fig_training_curves.{ext}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--allow-stale-code", action="store_true")
    args = ap.parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_runs(args.allow_stale_code)
    banner(f"ANALYSIS over {len(df)} verified runs (code {code_sha256()[:12]})")
    if df.empty:
        raise SystemExit("no verified runs to analyse")
    df.sort_values(["group", "seed"]).to_csv(REPORT_DIR / "per_seed_results.csv", index=False)

    summary = df.groupby("group").agg({**{m: ["mean", "std"] for m in METRICS},
                                        "seed": "count", "trainable_params": "mean"})
    flat = summary.copy()
    flat.columns = ["_".join(c) for c in flat.columns]
    flat.to_csv(REPORT_DIR / "summary_mean_sd.csv")
    print(flat[["mean_recall_mean", "mean_recall_std", "seed_count"]].to_string())

    comparisons = [(default_group(a), default_group(b)) for a, b in COMPARISONS]
    comparisons += [(g, default_group("full")) for g in summary.index
                    if g.startswith("full|") and g != default_group("full")]
    results = [paired_stats(df, a, b, args.bootstrap) for a, b in comparisons
               if a in summary.index and b in summary.index]
    tested = [r for r in results if "t_test_p" in r]
    if tested:
        for r, p in zip(tested, holm([r["t_test_p"] for r in tested])):
            r["t_test_p_holm"] = float(p)
    write_json(REPORT_DIR / "statistical_analysis.json", {
        "comparisons": results,
        "notes": ["d_z is the paired effect size (mean difference / SD of differences), not pooled Cohen's d.",
                  "With few seeds, t-test p-values are indicative only; do not claim significance from n=3.",
                  "Bootstrap CI reflects test-set sampling only, averaged over matched seeds."],
        "provenance": provenance()})
    for r in results:
        boot = r.get("bootstrap_test_images", {})
        print(f"{r['a']:28s} - {r['b']:28s} n={r['n']} diff={r.get('mean_diff', float('nan')):+.2f} "
              f"d_z={r.get('d_z', float('nan')):.2f} p={r.get('t_test_p', float('nan')):.3f} "
              f"p_holm={r.get('t_test_p_holm', float('nan')):.3f} boot95={boot.get('ci95')}")

    core = [default_group(v) for v in CORE]
    ablation = [default_group(v) for v in ("full_x", "full_x_affine", "full_x_constdamp", "full_x_noexchange",
                                           "full_x_noprior", "baseline", "baseline_param_matched_x",
                                           "transformer_param_matched", "transformer_x", "no_mixer_meanpool",
                                           "full", "full_linear_operator", "full_hvsc_deterministic",
                                           "baseline_param_matched", *LEGACY_CORE[1:3])]
    ablation += sorted(g for g in summary.index if g.startswith("full|") and g != default_group("full"))
    (REPORT_DIR / "table_main.tex").write_text(latex_table(
        summary, core, "Flickr8k 1K test retrieval (mean $\\pm$ SD over seeds). Params: trainable parameters.",
        "tab:main"), encoding="utf-8")
    (REPORT_DIR / "table_ablations.tex").write_text(latex_table(
        summary, ablation, "Ablations and controls on Flickr8k 1K test (mean $\\pm$ SD over seeds).",
        "tab:ablations"), encoding="utf-8")

    stat_lines = ["\\begin{table}[t]", "\\centering",
                  "\\caption{Paired comparisons of test mean recall on matched seeds. $d_z$: paired effect size. "
                  "$p$: paired $t$-test, Holm-adjusted, indicative only. CI: 95\\% bootstrap over test images.}",
                  "\\label{tab:stats}", "\\resizebox{\\linewidth}{!}{%", "\\begin{tabular}{llccccc}", "\\hline",
                  "A & B & $n$ & $\\Delta$MR & $d_z$ & $p_{\\text{Holm}}$ & 95\\% CI \\\\", "\\hline"]
    for r in results:
        if not r["n"]:
            continue
        ci = r.get("bootstrap_test_images", {}).get("ci95")
        stat_lines.append(
            f"{LABELS.get(r['a'].split('|')[0], r['a'])} ({r['a'].split('|', 1)[1]}) & "
            f"{LABELS.get(r['b'].split('|')[0], r['b'])} & {r['n']} & {r['mean_diff']:+.2f} & "
            f"{r.get('d_z', float('nan')):.2f} & {r.get('t_test_p_holm', float('nan')):.3f} & "
            + (f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "--") + " \\\\")
    stat_lines += ["\\hline", "\\end{tabular}}", "\\end{table}"]
    (REPORT_DIR / "table_stats.tex").write_text("\n".join(stat_lines).replace("|", ", ") + "\n", encoding="utf-8")

    figures(df, summary, REPORT_DIR)
    print(f"report written to {REPORT_DIR}")
    print("ANALYSIS: PASS")


if __name__ == "__main__":
    main()
