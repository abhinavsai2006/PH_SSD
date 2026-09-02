"""
Master Evaluation & Profiling Entrypoint for PH-SSD Model.
Author: Lead Research Engineer
License: Apache 2.0
"""

import json
import os
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from ph_ssd.datasets.dataset_factory import build_dataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.evaluation.evaluator import PHSSDEvaluator
from ph_ssd.utils.logger import MetricLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PH-SSD Multimodal Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test' or 'val')")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = MetricLogger("PH-SSD-Eval")

    val_dataset = build_dataset(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        split=args.split,
        seq_len=config["dataset"]["seq_len"],
        image_size=config["dataset"]["image_size"],
        max_samples=config["dataset"].get("max_samples", 0),
    )
    dataloader = DataLoader(val_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)

    model = PHSSDTaskModel(
        input_dim_A=config["model"]["input_dim_A"],
        input_dim_B=config["model"]["input_dim_B"],
        d_model=config["model"]["d_model"],
        d_embed=config["model"].get("d_embed", 128),
        d_state=config["model"]["d_state"],
        z_dim=config["model"]["z_dim"],
        n_layers=config["model"]["n_layers"],
        vision_encoder=config["model"]["vision_encoder"],
        text_encoder=config["model"]["text_encoder"],
        pretrained=config["model"]["pretrained"],
        use_sd_npf=config["model"].get("use_sd_npf", True),
        use_vcm_ssd=config["model"].get("use_vcm_ssd", True),
    )

    ckpt_path = os.path.join(config["checkpoint"]["dir"], "ph_ssd_best.pt")
    if os.path.exists(ckpt_path):
        logger.logger.info(f"Loading best checkpoint from '{ckpt_path}'...")
        model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)

    evaluator = PHSSDEvaluator(model, device=device)

    logger.logger.info("Running evaluation suite...")
    eval_metrics = evaluator.evaluate(dataloader)

    sample_batch = next(iter(dataloader))
    eff_metrics = evaluator.profile_efficiency(sample_batch["raw_A"], sample_batch["raw_B"])
    eval_metrics.update(eff_metrics)

    for k, v in eval_metrics.items():
        if isinstance(v, float):
            logger.logger.info(f"{k}: {v:.4f}")
        else:
            logger.logger.info(f"{k}: {v}")

    # Export eval metrics to paper_results
    os.makedirs("paper_results", exist_ok=True)
    with open("paper_results/eval_results.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)


if __name__ == "__main__":
    main()
