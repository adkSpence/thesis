"""
train.py — Main entry point for teacher model training.

Smoke test (local Mac, ~5 mins):
    python train.py --smoke_test

Full training (Kaggle / GPU):
    python train.py --epochs 100 --data_dir /kaggle/working/data
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore")

# Suppress wandb verbose output
os.environ["WANDB_SILENT"] = "true"

import torch
import wandb

from src.data.dataset   import get_dataloaders
from src.training.trainer import Trainer, build_teacher_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",       type=str,   default="./data")
    parser.add_argument("--checkpoint_dir", type=str,   default="./checkpoints")
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--batch_size",     type=int,   default=1)
    parser.add_argument("--val_interval",   type=int,   default=2)
    parser.add_argument("--val_frac",       type=float, default=0.2)
    parser.add_argument("--num_workers",    type=int,   default=0)
    parser.add_argument("--run_name",       type=str,   default="teacher_baseline")
    parser.add_argument("--smoke_test",     action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device : {device}")

    # ── WandB — minimal output ─────────────────────────────────────────────────
    wandb.init(
        project  = "brats-teacher",
        name     = args.run_name,
        config   = vars(args),
        settings = wandb.Settings(silent=True),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    num_samples = 20 if args.smoke_test else None
    train_loader, val_loader = get_dataloaders(
        root_dir    = args.data_dir,
        val_frac    = args.val_frac,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        num_samples = num_samples,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model        = build_teacher_model(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters : {total_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(
        model          = model,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        lr             = args.lr,
        max_epochs     = 2 if args.smoke_test else args.epochs,
        val_interval   = 1 if args.smoke_test else args.val_interval,
        checkpoint_dir = args.checkpoint_dir,
        run_name       = args.run_name,
        smoke_test     = args.smoke_test,
    )

    trainer.train()
    wandb.finish()
    print("WandB run saved.")


if __name__ == "__main__":
    main()