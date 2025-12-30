import os
import shutil
import warnings

import fiftyone as fo
import fiftyone.zoo as foz
import yaml

with open("scripts/openimage.yaml", "r") as f:
    data = yaml.safe_load(f)

fo.config.dataset_zoo_dir = data["path"]

map_amount = {"train": 1000, "validation": 100, "test": 1}
name = "open-images-v7"

OUTPUT_ROOT = "./dataset"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

print("Starting download...")

for split, amount in map_amount.items():

    print(f"\n📥 Downloading {split} ({amount} images)...")

    dataset = foz.load_zoo_dataset(
        name,
        split=split,
        max_samples=amount,
        shuffle=True,
        label_types=[],
    )

    fo_split_dir = os.path.join(data["path"], name, split, "data")

    final_split = "valid" if split == "validation" else split
    out_dir = os.path.join(OUTPUT_ROOT, final_split)
    os.makedirs(out_dir, exist_ok=True)

    for img in os.listdir(fo_split_dir):
        if img.endswith(".jpg"):
            shutil.copy(os.path.join(fo_split_dir, img), out_dir)

    print(f"✔ Done: saved images into {out_dir}")

REMOVE_METADATA = True

if REMOVE_METADATA:
    shutil.rmtree(os.path.join(OUTPUT_ROOT, name), ignore_errors=True)
    print("\nRemoved metadata folders.")

print("\nAll splits downloaded and cleaned.")
