import os
import torch
import wandb
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference


def build_teacher_model(device: torch.device) -> torch.nn.Module:
    """
    Teacher 3D U-Net — large capacity model for BraTS segmentation.

    Input : (B, 4, H, W, D)  — 4 MRI modalities
    Output: (B, 4, H, W, D)  — 4 classes (background + NCR + ED + ET)
    """
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(32, 64, 128, 256, 512),   # full-size teacher
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.1,
    ).to(device)
    return model


class Trainer:
    def __init__(
        self,
        model:        torch.nn.Module,
        train_loader,
        val_loader,
        device:       torch.device,
        lr:           float = 1e-4,
        max_epochs:   int   = 100,
        val_interval: int   = 2,
        checkpoint_dir: str = "./checkpoints",
        run_name:     str   = "teacher_baseline",
        smoke_test:   bool  = False,
    ):
        self.model          = model
        self.train_loader   = train_loader
        self.val_loader     = val_loader
        self.device         = device
        self.max_epochs     = max_epochs
        self.val_interval   = val_interval
        self.checkpoint_dir = checkpoint_dir
        self.run_name       = run_name
        self.smoke_test     = smoke_test

        os.makedirs(checkpoint_dir, exist_ok=True)

        # ── Loss & metrics ─────────────────────────────────────────────────
        self.loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

        self.dice_metric = DiceMetric(
            include_background=False,   # ignore background class
            reduction="mean_batch",
        )

        # Post-processing: argmax prediction, one-hot label
        self.post_pred  = AsDiscrete(argmax=True, to_onehot=4)
        self.post_label = AsDiscrete(to_onehot=4)

        # ── Optimiser & scheduler ──────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_epochs
        )

        # ── Tracking ───────────────────────────────────────────────────────
        self.best_val_dice  = -1.0
        self.best_epoch     = 0

    # ──────────────────────────────────────────────────────────────────────────
    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            # batch is a list when RandCropByPosNegLabeld num_samples > 1
            if isinstance(batch, list):
                batch = batch[0]

            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss    = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(f"  Epoch {epoch} | step {step}/{len(self.train_loader)} "
                      f"| loss {loss.item():.4f}")

            # Smoke test — stop after 3 steps
            if self.smoke_test and step >= 2:
                break

        avg_loss = epoch_loss / (step + 1)
        return avg_loss

    # ──────────────────────────────────────────────────────────────────────────
    def _val_epoch(self) -> dict:
        self.model.eval()
        self.dice_metric.reset()

        with torch.no_grad():
            for step, batch in enumerate(self.val_loader):
                if isinstance(batch, list):
                    batch = batch[0]

                inputs = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                # Sliding window inference handles large volumes
                outputs = sliding_window_inference(
                    inputs,
                    roi_size=(128, 128, 128),
                    sw_batch_size=1,
                    predictor=self.model,
                    overlap=0.5,
                )

                # Post-process
                outputs_post = [self.post_pred(i)  for i in outputs]
                labels_post  = [self.post_label(i) for i in labels]

                self.dice_metric(y_pred=outputs_post, y=labels_post)

                if self.smoke_test and step >= 1:
                    break

        # Per-class Dice: [NCR, ED, ET]
        dice_per_class = self.dice_metric.aggregate()   # tensor shape (3,)
        mean_dice      = dice_per_class.mean().item()

        return {
            "mean_dice": mean_dice,
            "dice_NCR":  dice_per_class[0].item(),
            "dice_ED":   dice_per_class[1].item(),
            "dice_ET":   dice_per_class[2].item(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        state = {
            "epoch":      epoch,
            "model":      self.model.state_dict(),
            "optimizer":  self.optimizer.state_dict(),
            "best_dice":  self.best_val_dice,
            "metrics":    metrics,
        }
        # Always save latest
        torch.save(state, os.path.join(self.checkpoint_dir, "latest.pth"))

        # Save best separately
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, "best_model.pth"))
            print(f"  ✅ New best model saved — Dice {self.best_val_dice:.4f}")

    # ──────────────────────────────────────────────────────────────────────────
    def train(self):
        print(f"\n{'='*60}")
        print(f"  Starting training: {self.run_name}")
        print(f"  Epochs     : {self.max_epochs}")
        print(f"  Val every  : {self.val_interval} epochs")
        print(f"  Smoke test : {self.smoke_test}")
        print(f"{'='*60}\n")

        wandb.watch(self.model, log="gradients", log_freq=50)

        for epoch in range(1, self.max_epochs + 1):
            print(f"\n── Epoch {epoch}/{self.max_epochs} ──")

            # ── Train ──────────────────────────────────────────────────────
            train_loss = self._train_epoch(epoch)
            self.scheduler.step()

            wandb.log({"train/loss": train_loss, "epoch": epoch})
            print(f"  Train loss: {train_loss:.4f}")

            # ── Validate ───────────────────────────────────────────────────
            if epoch % self.val_interval == 0:
                metrics = self._val_epoch()

                is_best = metrics["mean_dice"] > self.best_val_dice
                if is_best:
                    self.best_val_dice = metrics["mean_dice"]
                    self.best_epoch    = epoch

                self._save_checkpoint(epoch, metrics, is_best)

                wandb.log({
                    "val/mean_dice": metrics["mean_dice"],
                    "val/dice_NCR":  metrics["dice_NCR"],
                    "val/dice_ED":   metrics["dice_ED"],
                    "val/dice_ET":   metrics["dice_ET"],
                    "epoch":         epoch,
                })

                print(f"  Val Dice  — Mean: {metrics['mean_dice']:.4f} | "
                      f"NCR: {metrics['dice_NCR']:.4f} | "
                      f"ED:  {metrics['dice_ED']:.4f}  | "
                      f"ET:  {metrics['dice_ET']:.4f}")
                print(f"  Best so far: {self.best_val_dice:.4f} @ epoch {self.best_epoch}")

            if self.smoke_test and epoch >= 2:
                print("\n  Smoke test complete ✅")
                break

        print(f"\n{'='*60}")
        print(f"  Training complete.")
        print(f"  Best Val Dice: {self.best_val_dice:.4f} @ epoch {self.best_epoch}")
        print(f"  Checkpoints saved to: {self.checkpoint_dir}")
        print(f"{'='*60}\n")