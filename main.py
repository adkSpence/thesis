import os
import torch
import matplotlib.pyplot as plt
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# 1. NEW: Ensure the directory exists!
root_dir = "./data"
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
    print(f"Created directory at {root_dir}")

# 2. Setup the 'Assembly Line' (Transforms)
my_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
])

# 3. Start the Download/Load process
print("Fetching Brain Tumor Data (this might take a while)...")
dataset = DecathlonDataset(
    root_dir=root_dir, # Using the variable we checked above
    task="Task01_BrainTumour",
    section="training",
    transform=my_transforms,
    download=True
)

# 4. Inspect the first brain
first_patient = dataset[0]
print(f"Success! Image shape is: {first_patient['image'].shape}")