"""
Master Experiment Runner & Reproducibility Suite for PH-SSD.
Executes Experiments A, B, C, D on Flickr8k and generates machine-readable results.
Author: Reproducibility Auditor & Lead ML Research Engineer
License: Apache 2.0
"""

import os
import sys
import time
import json
import csv
import yaml
import torch
from torch.utils.data import DataLoader

from ph_ssd.datasets.dataset_factory import build_dataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.training.trainer import PHSSDTrainer
from ph_ssd.evaluation.evaluator import PHSSDEvaluator
from ph_ssd.utils.logger import MetricLogger


def run_single_experiment(
    exp_name: str,
    use_sd_npf: bool,
    use_vcm_ssd: bool,
    config: dict,
    epochs: int = 2,
    seed: int = 42,
) -> dict:
    print(f"\n=======================================================")
    print(f"Running Experiment [{exp_name}] (SD-NPF={use_sd_npf}, VCM-SSD={use_vcm_ssd})")
    print(f"=======================================================")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    train_dataset = build_dataset(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        split="train",
        seq_len=config["dataset"]["seq_len"],
        image_size=config["dataset"]["image_size"],
        max_samples=config["dataset"].get("max_samples", 0),
    )
    test_dataset = build_dataset(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        split="test",
        seq_len=config["dataset"]["seq_len"],
        image_size=config["dataset"]["image_size"],
        max_samples=config["dataset"].get("max_samples", 0),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)

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
        use_sd_npf=use_sd_npf,
        use_vcm_ssd=use_vcm_ssd,
    )

    criterion = PHSSDLoss(contrastive_weight=1.0, kl_weight=config["training"].get("kl_weight", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    trainer = PHSSDTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=config["training"]["use_amp"],
        use_ema=False,
        max_grad_norm=1.0,
        warmup_epochs=1,
        total_epochs=epochs,
    )

    evaluator = PHSSDEvaluator(model, device=device)

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        print(f"  Epoch [{epoch}/{epochs}] -> Loss: {train_metrics['epoch/loss']:.4f} | Contrastive Loss: {train_metrics['epoch/contrastive_loss']:.4f}")

    total_train_time = time.time() - start_time

    # Evaluate on test set
    eval_metrics = evaluator.evaluate(test_loader)

    # Efficiency profiling
    sample_batch = next(iter(test_loader))
    eff_metrics = evaluator.profile_efficiency(sample_batch["raw_A"], sample_batch["raw_B"])

    result_entry = {
        "experiment": exp_name,
        "model": "PH-SSD" if (use_sd_npf and use_vcm_ssd) else ("Mamba-SSD+SDNPF" if use_sd_npf else ("Mamba-SSD+VCMSSD" if use_vcm_ssd else "Baseline-MambaSSD")),
        "seed": seed,
        "dataset": config["dataset"]["name"],
        "split": "test",
        "i2t_r1": eval_metrics.get("retrieval/i2t_r1", 0.0),
        "i2t_r5": eval_metrics.get("retrieval/i2t_r5", 0.0),
        "i2t_r10": eval_metrics.get("retrieval/i2t_r10", 0.0),
        "t2i_r1": eval_metrics.get("retrieval/t2i_r1", 0.0),
        "t2i_r5": eval_metrics.get("retrieval/t2i_r5", 0.0),
        "t2i_r10": eval_metrics.get("retrieval/t2i_r10", 0.0),
        "mean_rank": eval_metrics.get("retrieval/i2t_mean_rank", 0.0),
        "median_rank": eval_metrics.get("retrieval/i2t_median_rank", 0.0),
        "mean_recall": eval_metrics.get("retrieval/mean_recall", 0.0),
        "parameters": eff_metrics.get("total_params", 0),
        "trainable_parameters": eff_metrics.get("trainable_params", 0),
        "latency_ms": eff_metrics.get("latency_ms", 0.0),
        "throughput": eff_metrics.get("throughput_fps", 0.0),
        "peak_memory_mb": eff_metrics.get("peak_memory_mb", 0.0),
        "flops": eff_metrics.get("estimated_flops", 0),
        "training_time": total_train_time,
        "status": "COMPLETED",
    }

    return result_entry


def main():
    config_path = "configs/retrieval_flickr8k.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Force 320 samples for quick execution verification
    config["dataset"]["max_samples"] = 320
    epochs = 2

    experiments = [
        ("Exp_A_Baseline", False, False),
        ("Exp_B_SDNPF", True, False),
        ("Exp_C_VCMSSD", False, True),
        ("Exp_D_Proposed_PHSSD", True, True),
    ]

    all_results = []
    for exp_name, use_sd_npf, use_vcm_ssd in experiments:
        res = run_single_experiment(exp_name, use_sd_npf, use_vcm_ssd, config, epochs=epochs, seed=42)
        all_results.append(res)

    os.makedirs("paper_results", exist_ok=True)

    # Save JSON database
    with open("paper_results/all_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # Save CSV database
    fieldnames = list(all_results[0].keys())
    with open("paper_results/all_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print("\n[SUCCESS] All 4 Experiments A, B, C, D Executed Successfully!")
    print("[SAVED] Saved paper_results/all_results.json & paper_results/all_results.csv")


if __name__ == "__main__":
    main()
