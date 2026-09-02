"""
Master Training Entrypoint for PH-SSD Model (Cross-Modal Contrastive Retrieval).
Author: Lead Research Engineer
License: Apache 2.0
"""

import os
import json
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

from ph_ssd.datasets.dataset_factory import build_dataset
from ph_ssd.models.ph_ssd_model import PHSSDTaskModel
from ph_ssd.losses.ph_ssd_loss import PHSSDLoss
from ph_ssd.training.trainer import PHSSDTrainer
from ph_ssd.evaluation.evaluator import PHSSDEvaluator
from ph_ssd.utils.logger import MetricLogger
from ph_ssd.utils.checkpoint import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PH-SSD Multimodal Contrastive Retrieval Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--use_sd_npf", type=bool, default=True, help="Enable SD-NPF pre-filter")
    parser.add_argument("--use_vcm_ssd", type=bool, default=True, help="Enable VCM-SSD coupler")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    epochs = args.epochs or config["training"]["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = MetricLogger("PH-SSD-Train")
    ckpt_dir = config["checkpoint"]["dir"]
    ckpt_mgr = CheckpointManager(ckpt_dir)

    logger.logger.info(f"Using compute device: {device}")
    logger.logger.info(f"Reproducibility Seed: {args.seed}")

    # Build datasets & loaders
    train_dataset = build_dataset(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        split="train",
        seq_len=config["dataset"]["seq_len"],
        image_size=config["dataset"]["image_size"],
        max_samples=config["dataset"].get("max_samples", 0),
    )
    val_dataset = build_dataset(
        dataset_name=config["dataset"]["name"],
        data_dir=config["dataset"]["data_dir"],
        split="val",
        seq_len=config["dataset"]["seq_len"],
        image_size=config["dataset"]["image_size"],
        max_samples=config["dataset"].get("max_samples", 0),
    )

    train_loader = DataLoader(train_dataset, batch_size=config["dataset"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["dataset"]["batch_size"], shuffle=False)

    # Build model with contrastive projections & ablation toggles
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
        use_sd_npf=config["model"].get("use_sd_npf", args.use_sd_npf),
        use_vcm_ssd=config["model"].get("use_vcm_ssd", args.use_vcm_ssd),
    )

    criterion = PHSSDLoss(contrastive_weight=1.0, kl_weight=config["training"].get("kl_weight", 1e-3))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"], weight_decay=config["training"]["weight_decay"])

    trainer = PHSSDTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        use_amp=config["training"]["use_amp"],
        use_ema=config["training"]["use_ema"],
        max_grad_norm=config["training"]["max_grad_norm"],
        warmup_epochs=config["training"].get("warmup_epochs", 1),
        total_epochs=epochs,
    )

    evaluator = PHSSDEvaluator(model, device=device)

    logger.logger.info("Starting PH-SSD Contrastive Training Loop...")
    best_r1 = 0.0
    history = []

    for epoch in range(1, epochs + 1):
        train_metrics = trainer.train_epoch(train_loader)
        eval_metrics = evaluator.evaluate(val_loader)
        train_metrics.update(eval_metrics)

        logger.log_metrics(epoch, train_metrics)
        ckpt_mgr.save_checkpoint(model, optimizer, epoch)

        current_r1 = eval_metrics.get("retrieval/i2t_r1", 0.0)
        if current_r1 >= best_r1:
            best_r1 = current_r1
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "ph_ssd_best.pt"))

        history.append(train_metrics)

    # Save manifest
    os.makedirs("paper_results", exist_ok=True)
    manifest = {
        "seed": args.seed,
        "dataset": config["dataset"]["name"],
        "epochs": epochs,
        "batch_size": config["dataset"]["batch_size"],
        "d_model": config["model"]["d_model"],
        "d_state": config["model"]["d_state"],
        "use_sd_npf": config["model"].get("use_sd_npf", args.use_sd_npf),
        "use_vcm_ssd": config["model"].get("use_vcm_ssd", args.use_vcm_ssd),
        "best_i2t_r1": float(best_r1),
        "device": str(device),
    }
    with open("paper_results/run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.logger.info("Training complete.")


if __name__ == "__main__":
    main()
