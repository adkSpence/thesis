import torch
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# Use the same root_dir from before
root_dir = "./data"

# Assembly line (Transforms)
my_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
])

# Load the dataset (it won't download again, just verify)
dataset = DecathlonDataset(root_dir=root_dir, task="Task01_BrainTumour", section="training", transform=my_transforms, download=False)

# Get the first patient
data_item = dataset[0]
image = data_item["image"] # Shape (4, 240, 240, 155)
label = data_item["label"] # Shape (1, 240, 240, 155)

# Pick a slice in the middle of the brain (Depth = 155)
slice_idx = 80

plt.figure("Brain Scan Slices", (12, 6))

# Subplot 1: The MRI Scan (FLAIR Modality)
plt.subplot(1, 2, 1)
plt.title(f"MRI Slice {slice_idx} (FLAIR)")
plt.imshow(image[0, :, :, slice_idx], cmap="gray")
plt.axis("off")

# Subplot 2: The Tumor Label (Ground Truth)
plt.subplot(1, 2, 2)
plt.title(f"Tumor Mask Slice {slice_idx}")
plt.imshow(label[0, :, :, slice_idx]) # Note: labels are 0, 1, 2, or 3
plt.axis("off")

plt.show()