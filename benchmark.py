"""
Automated Publication Benchmark Suite Entrypoint.
Author: Lead Research Engineer
License: Apache 2.0
"""

import torch
from ph_ssd.baselines.baseline_runner import BaselineRunner
from ph_ssd.experiments.ablation_runner import AblationRunner
from ph_ssd.utils.latex_exporter import export_results_to_latex
from ph_ssd.visualization.plot_results import plot_ablation_bar_chart
from ph_ssd.utils.logger import MetricLogger


def main() -> None:
    logger = MetricLogger("PH-SSD-Benchmark")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.logger.info("Step 1: Running Baseline Comparisons (Mamba-2, VL-Mamba, CLIP, PH-SSD)...")
    base_runner = BaselineRunner(device=device)
    baseline_results = base_runner.run_all_baselines()

    logger.logger.info("Step 2: Exporting LaTeX and CSV Tables...")
    export_results_to_latex(baseline_results, output_tex="docs/baseline_results.tex", output_csv="docs/baseline_results.csv")

    logger.logger.info("Step 3: Running Component Ablation Suite...")
    ablation_runner = AblationRunner(device=device)
    ablation_results = ablation_runner.run_all_ablations(num_epochs=1)

    export_results_to_latex(ablation_results, output_tex="docs/ablation_results.tex", output_csv="docs/ablation_results.csv")
    plot_path = plot_ablation_bar_chart(ablation_results, save_path="docs/ablation_chart.png")

    logger.logger.info(f"Benchmark suite complete! LaTeX tables and figures exported to docs/ ({plot_path}).")


if __name__ == "__main__":
    main()
