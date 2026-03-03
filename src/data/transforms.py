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
    ConcatItemsd,
    DeleteItemsd,
)


def get_base_transformations():
    """
    Training transforms for BraTS2021 raw format.
    Input: dict with image = [t1, t1ce, t2, flair paths] and label = seg path
    Output: dict with image = (4, H, W, D) tensor and label = (1, H, W, D) tensor
    """
    return Compose([
        # ── Load all 4 modalities + label ──────────────────────────────────
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        # ── Orient to RAS ──────────────────────────────────────────────────
        Orientationd(keys=["image", "label"], axcodes="RAS"),

        # ── Resample to 1mm isotropic ──────────────────────────────────────
        Spacingd(
            keys=["image", "label"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "nearest"),
        ),

        # ── Crop away empty background ─────────────────────────────────────
        CropForegroundd(keys=["image", "label"], source_key="image"),

        # ── Normalise each modality independently ──────────────────────────
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),

        # ── Random patch crop biased toward tumour ─────────────────────────
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            pos=1,
            neg=1,
            num_samples=1,
        ),

        # ── Spatial augmentation ───────────────────────────────────────────
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),

        # ── Intensity augmentation ─────────────────────────────────────────
        RandShiftIntensityd(keys=["image"], offsets=0.1, prob=0.5),

        # ── Final type cast ────────────────────────────────────────────────
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])


def get_val_transformations():
    """
    Validation transforms — no augmentation.
    """
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
        NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        EnsureTyped(keys=["image"], dtype=torch.float32),
        EnsureTyped(keys=["label"], dtype=torch.long),
    ])