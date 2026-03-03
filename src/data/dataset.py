import os
import random
import torch
from monai.data import Dataset, DataLoader
from src.data.transforms import get_base_transformations, get_val_transformations


def get_brats_datalist(data_dir: str):
    """
    Scan the BraTS2021 raw folder structure and build a list of dicts.
    Each dict has keys: image (list of 4 modality paths) and label.

    Expected structure:
        data_dir/
            BraTS2021_00001/
                BraTS2021_00001_t1.nii.gz
                BraTS2021_00001_t1ce.nii.gz
                BraTS2021_00001_t2.nii.gz
                BraTS2021_00001_flair.nii.gz
                BraTS2021_00001_seg.nii.gz
    """
    datalist = []

    cases = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
        and d.startswith("BraTS2021_")
    ])

    for case in cases:
        case_dir = os.path.join(data_dir, case)
        t1    = os.path.join(case_dir, f"{case}_t1.nii.gz")
        t1ce  = os.path.join(case_dir, f"{case}_t1ce.nii.gz")
        t2    = os.path.join(case_dir, f"{case}_t2.nii.gz")
        flair = os.path.join(case_dir, f"{case}_flair.nii.gz")
        seg   = os.path.join(case_dir, f"{case}_seg.nii.gz")

        # Only include case if all 5 files exist
        if all(os.path.exists(p) for p in [t1, t1ce, t2, flair, seg]):
            datalist.append({
                "image": [t1, t1ce, t2, flair],   # 4 channels → stacked by MONAI
                "label": seg,
            })
        else:
            print(f"  ⚠️  Skipping {case} — missing files")

    return datalist


def get_dataloaders(
    root_dir:    str,
    val_frac:    float = 0.2,
    batch_size:  int   = 1,
    num_workers: int   = 0,
    num_samples: int   = None,
):
    """
    Build train and val DataLoaders from raw BraTS2021 folder structure.

    Args:
        root_dir:    Path containing BraTS2021_XXXXX case folders
        val_frac:    Fraction held out for validation
        batch_size:  Training batch size
        num_workers: DataLoader workers
        num_samples: If set, only use this many total cases (smoke test)

    Returns:
        train_loader, val_loader
    """

    datalist = get_brats_datalist(root_dir)
    print(f"Total cases found: {len(datalist)}")

    # Restrict for smoke test
    if num_samples is not None:
        random.seed(42)
        datalist = random.sample(datalist, min(num_samples, len(datalist)))
        print(f"Smoke test: using {len(datalist)} cases")

    # Train / val split
    random.seed(42)
    random.shuffle(datalist)

    val_count   = max(1, int(len(datalist) * val_frac))
    train_count = len(datalist) - val_count

    train_list = datalist[:train_count]
    val_list   = datalist[train_count:]

    print(f"Train cases : {len(train_list)}")
    print(f"Val   cases : {len(val_list)}")

    train_transforms = get_base_transformations()
    val_transforms   = get_val_transformations()

    train_ds = Dataset(data=train_list, transform=train_transforms)
    val_ds   = Dataset(data=val_list,   transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader