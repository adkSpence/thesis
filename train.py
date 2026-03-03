"""
train.py — Main entry point for teacher model training.

Smoke test (local Mac, ~5 mins):
    python train.py --smoke_test

Full training (Kaggle / GPU):
    python train.py --epochs 100 --data_dir /kaggle/input/brats2021

"""
import argparse
import torch
import wandb

from src.data.dataset  import get_dataloaders
from src.training.trainer import Trainer, build_teacher_model


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS Teacher Model Training")

    parser.add_argument("--data_dir",     type=str,   default="./datasource",
                        help="Root directory of BraTS / Decathlon datasource")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints",
                        help="Where to save model checkpoints")
    parser.add_argument("--epochs",       type=int,   default=100,
                        help="Number of training epochs")
    parser.add_argument("--lr",           type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size",   type=int,   default=1,
                        help="Training batch size")
    parser.add_argument("--val_interval", type=int,   default=2,
                        help="Run validation every N epochs")
    parser.add_argument("--val_frac",     type=float, default=0.2,
                        help="Fraction of datasource held out for validation")
    parser.add_argument("--num_workers",  type=int,   default=0,
                        help="DataLoader workers (0 = main process)")
    parser.add_argument("--run_name",     type=str,   default="teacher_baseline",
                        help="WandB run name")
    parser.add_argument("--smoke_test",   action="store_true",
                        help="Run a quick smoke test on 20 cases, 2 epochs")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available()          else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── WandB ─────────────────────────────────────────────────────────────────
    wandb.init(
        project="brats-teacher",
        name=args.run_name,
        config=vars(args),
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    num_samples = 20 if args.smoke_test else None

    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_samples=num_samples,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_teacher_model(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Teacher model parameters: {total_params:,}")
    wandb.config.update({"model_params": total_params})

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        max_epochs=2 if args.smoke_test else args.epochs,
        val_interval=1 if args.smoke_test else args.val_interval,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        smoke_test=args.smoke_test,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()