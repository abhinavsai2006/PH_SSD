"""
LaTeX and CSV Publication Table Generator.
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import csv
from typing import Dict, Any, List


def export_results_to_latex(
    results: Dict[str, Dict[str, float]],
    output_tex: str = "docs/results_table.tex",
    output_csv: str = "docs/results_table.csv",
) -> None:
    """
    Export benchmark evaluation dictionary to publication-ready LaTeX table and CSV.
    """
    os.makedirs(os.path.dirname(output_tex), exist_ok=True)

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model / Variant", "Accuracy (%)", "R@1", "R@5", "R@10", "Avg KL Loss"])
        for model_name, metrics in results.items():
            writer.writerow([
                model_name,
                f"{metrics.get('eval/accuracy', 0.0):.2f}",
                f"{metrics.get('retrieval/i2t_r1', 0.0):.2f}",
                f"{metrics.get('retrieval/i2t_r5', 0.0):.2f}",
                f"{metrics.get('retrieval/i2t_r10', 0.0):.2f}",
                f"{metrics.get('eval/avg_kl_loss', 0.0):.4f}",
            ])

    # Write LaTeX Table
    with open(output_tex, "w") as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Benchmark Comparison of Proposed PH-SSD against Baselines.}\n")
        f.write("\\label{tab:main_results}\n")
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Model / Architecture & Acc (\\%) & R@1 & R@5 & R@10 & KL Loss \\\\\n")
        f.write("\\midrule\n")

        for model_name, metrics in results.items():
            acc = f"{metrics.get('eval/accuracy', 0.0):.2f}"
            r1 = f"{metrics.get('retrieval/i2t_r1', 0.0):.2f}"
            r5 = f"{metrics.get('retrieval/i2t_r5', 0.0):.2f}"
            r10 = f"{metrics.get('retrieval/i2t_r10', 0.0):.2f}"
            kl = f"{metrics.get('eval/avg_kl_loss', 0.0):.4f}"

            if "PH-SSD" in model_name:
                f.write(f"\\textbf{{{model_name}}} & \\textbf{{{acc}}} & \\textbf{{{r1}}} & \\textbf{{{r5}}} & \\textbf{{{r10}}} & \\textbf{{{kl}}} \\\\\n")
            else:
                f.write(f"{model_name} & {acc} & {r1} & {r5} & {r10} & {kl} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
