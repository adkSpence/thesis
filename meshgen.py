import numpy as np
import matplotlib.pyplot as plt
from skimage import measure
from monai.apps import DecathlonDataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd

# ── Label definitions ──────────────────────────────────────────────────────────
# BraTS labels:  0 = Background, 1 = NCR (Necrotic Core),
#                2 = ED (Edema/Invasion), 3 = ET (Enhancing Tumour)
LABEL_CONFIG = {
    1: {"name": "Necrotic Core (NCR)",      "color": [255, 0,   0  ]},   # Red
    2: {"name": "Edema / Invasion (ED)",    "color": [0,   255, 0  ]},   # Green
    3: {"name": "Enhancing Tumour (ET)",    "color": [255, 255, 0  ]},   # Yellow
}

# ── Data setup ─────────────────────────────────────────────────────────────────
root_dir = "datasource"
my_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
])

dataset = DecathlonDataset(
    root_dir=root_dir,
    task="Task01_BrainTumour",
    section="training",
    transform=my_transforms,
    download=False
)

data_item   = dataset[0]
label_volume = data_item["label"][0].numpy()   # shape: (240, 240, 155)

print(f"Label volume shape : {label_volume.shape}")
print(f"Unique label values: {np.unique(label_volume)}")


# ── Per-label mesh generation ──────────────────────────────────────────────────
def generate_mesh_for_label(volume, label_value):
    """Extract a surface mesh for a single label using Marching Cubes."""
    mask = (volume == label_value).astype(float)
    if mask.sum() == 0:
        print(f"  ⚠️  Label {label_value} not found in this volume — skipping.")
        return None, None
    verts, faces, _, _ = measure.marching_cubes(mask, level=0.5)
    return verts, faces


# ── Save coloured OBJ with MTL sidecar ────────────────────────────────────────
def save_colored_obj(base_filename, meshes_by_label):
    """
    Write a single .obj that references per-label materials,
    plus a .mtl file that defines those materials.
    """
    obj_path = f"{base_filename}.obj"
    mtl_path = f"{base_filename}.mtl"
    mtl_name = base_filename.split("/")[-1] + ".mtl"

    # ── Write MTL ──
    with open(mtl_path, "w") as mtl:
        for label_id, cfg in LABEL_CONFIG.items():
            r, g, b = [c / 255.0 for c in cfg["color"]]
            mtl.write(f"newmtl material_label_{label_id}\n")
            mtl.write(f"Kd {r:.4f} {g:.4f} {b:.4f}\n")   # diffuse colour
            mtl.write(f"Ka 0.1 0.1 0.1\n")                # ambient
            mtl.write(f"Ks 0.0 0.0 0.0\n\n")              # no specular

    # ── Write OBJ ──
    with open(obj_path, "w") as obj:
        obj.write(f"mtllib {mtl_name}\n\n")
        vertex_offset = 0

        for label_id, (verts, faces) in meshes_by_label.items():
            if verts is None:
                continue
            cfg = LABEL_CONFIG[label_id]
            obj.write(f"# {cfg['name']}\n")
            obj.write(f"usemtl material_label_{label_id}\n")
            obj.write(f"o label_{label_id}_{cfg['name'].split()[0]}\n")

            for v in verts:
                obj.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            for face in faces:
                f0, f1, f2 = face + vertex_offset + 1   # OBJ is 1-indexed
                obj.write(f"f {f0} {f1} {f2}\n")

            vertex_offset += len(verts)
            print(f"  ✅ Label {label_id} ({cfg['name']}): "
                  f"{len(verts):,} vertices, {len(faces):,} faces")

    print(f"\nSaved OBJ → {obj_path}")
    print(f"Saved MTL → {mtl_path}")


# ── Quick 2-D colour overlay for sanity check ──────────────────────────────────
def visualize_colored_slice(label_volume, slice_idx=80):
    """Show a 2D axial slice with colour-coded tumour regions."""
    slice_2d = label_volume[:, :, slice_idx]

    # Build an RGB image
    h, w = slice_2d.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for label_id, cfg in LABEL_CONFIG.items():
        mask = slice_2d == label_id
        rgb[mask] = cfg["color"]

    plt.figure("Colour-mapped tumour slice", figsize=(6, 6))
    plt.title(f"Axial slice {slice_idx} — colour-coded labels")
    plt.imshow(rgb)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=[c/255 for c in cfg["color"]], label=cfg["name"])
        for _, cfg in LABEL_CONFIG.items()
    ]
    plt.legend(handles=legend_elements, loc="lower right", fontsize=8)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("outputs/colored_slice.png", dpi=150)
    plt.show()
    print("Slice preview saved → outputs/colored_slice.png")


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.makedirs("outputs", exist_ok=True)

    print("\n── Generating per-label meshes ──")
    meshes = {}
    for label_id in LABEL_CONFIG:
        print(f"Processing label {label_id}…")
        verts, faces = generate_mesh_for_label(label_volume, label_id)
        meshes[label_id] = (verts, faces)

    print("\n── Saving coloured OBJ ──")
    save_colored_obj("outputs/brain_tumor_colored", meshes)

    print("\n── 2D slice preview ──")
    visualize_colored_slice(label_volume, slice_idx=80)