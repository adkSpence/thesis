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
    SpatialPadd,
)

PATCH_SIZE = (128, 128, 128)


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

        # Remap BraTS2021 label 4 → 3  (dataset uses 0,1,2,4 — no label 3)
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),

        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Pad any volume smaller than patch size so crop never crashes
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=PATCH_SIZE,
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
        MapLabelValued(
            keys=["label"],
            orig_labels=[0, 1, 2, 4],
            target_labels=[0, 1, 2, 3],
        ),
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # Same pad for val — sliding window handles large volumes fine
        SpatialPadd(keys=["image", "label"], spatial_size=PATCH_SIZE),

        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])