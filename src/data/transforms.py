import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    EnsureTyped,
    CropForegroundd,
    RandCropByPosNegLabeld,
    MapLabelValued,
)


def get_base_transformations():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # ── CRITICAL: remap label 4 → 3 so classes are 0,1,2,3 ───────────
        # BraTS2021 uses: 0=bg, 1=NCR, 2=ED, 4=ET  (no label 3)
        # Our model outputs 4 channels so ET must be remapped to 3
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])


def get_val_transformations():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # Same label remap for validation
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])