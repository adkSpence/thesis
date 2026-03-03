import os
from monai.apps import DecathlonDataset
from monai.data import DataLoader
from src.data.transforms import get_base_transformations, get_val_transformations  # noqa


def get_dataloaders(root_dir: str, val_frac: float = 0.2, batch_size: int = 1, num_workers: int = 0, num_samples: int = None):
    """
    Load BraTS from Decathlon and split into train/val loaders.

    Args:
        root_dir:    Path to datasource directory
        val_frac:    Fraction of datasource to use for validation (default 0.2)
        batch_size:  Batch size for DataLoader
        num_workers: Number of DataLoader workers
        num_samples: If set, only use this many cases (useful for smoke tests)

    Returns:
        train_loader, val_loader
    """

    train_transforms = get_base_transformations()
    val_transforms   = get_val_transformations()

    # Full training dataset (388 cases in Decathlon)
    full_dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="training",
        transform=train_transforms,
        download=False,
        val_frac=val_frac,   # MONAI handles the split internally
    )

    val_dataset = DecathlonDataset(
        root_dir=root_dir,
        task="Task01_BrainTumour",
        section="validation",
        transform=val_transforms,
        download=False,
        val_frac=val_frac,
    )

    # Smoke test — restrict to num_samples if provided
    if num_samples is not None:
        from torch.utils.data import Subset
        train_indices = list(range(min(num_samples, len(full_dataset))))
        val_indices   = list(range(min(max(1, num_samples // 5), len(val_dataset))))
        full_dataset  = Subset(full_dataset, train_indices)
        val_dataset   = Subset(val_dataset,  val_indices)

    train_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,         # always 1 for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    print(f"Train cases : {len(full_dataset)}")
    print(f"Val   cases : {len(val_dataset)}")

    return train_loader, val_loader