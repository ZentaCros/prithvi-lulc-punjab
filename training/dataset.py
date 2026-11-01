"""
LULC Lahore — PyTorch Dataset
Loads preprocessed 256×256 patches with per-band normalization.
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class LULCDataset(Dataset):
    """
    Loads preprocessed patches from data/patches/{split}/images and labels.
    Applies per-band normalization using training set statistics.
    Optionally applies data augmentation (random flip/rotate).
    """

    def __init__(self, patches_dir, split="train", metadata=None, augment=False):
        """
        Args:
            patches_dir: Path to data/patches/ directory
            split: One of 'train', 'val', 'test'
            metadata: Dict from metadata.json (band_means, band_stds, etc.)
            augment: Whether to apply data augmentation
        """
        self.patches_dir = Path(patches_dir)
        self.split = split
        self.augment = augment and (split == "train")  # Only augment training set

        # Load metadata
        if metadata is None:
            meta_path = self.patches_dir / "metadata.json"
            with open(meta_path) as f:
                metadata = json.load(f)

        self.band_means = np.array(metadata["band_means"], dtype=np.float32).reshape(-1, 1, 1)
        self.band_stds = np.array(metadata["band_stds"], dtype=np.float32).reshape(-1, 1, 1)
        self.num_classes = metadata["num_classes"]
        self.class_names = metadata["class_names"]

        # Find all patch files
        img_dir = self.patches_dir / split / "images"
        lbl_dir = self.patches_dir / split / "labels"

        self.image_files = sorted(list(img_dir.glob("*.npy")))
        self.label_files = sorted(list(lbl_dir.glob("*.npy")))

        assert len(self.image_files) == len(self.label_files), \
            f"Mismatch: {len(self.image_files)} images vs {len(self.label_files)} labels"
        assert len(self.image_files) > 0, f"No patches found in {img_dir}"

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load patch
        image = np.load(self.image_files[idx]).astype(np.float32)  # [C, H, W]
        label = np.load(self.label_files[idx]).astype(np.int64)    # [H, W]

        # Normalize: (x - mean) / std
        image = (image - self.band_means) / (self.band_stds + 1e-8)

        # Data augmentation
        if self.augment:
            image, label = self._augment(image, label)

        return torch.from_numpy(image.copy()), torch.from_numpy(label.copy())

    def _augment(self, image, label):
        """Random horizontal/vertical flip and 90-degree rotation."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            image = image[:, :, ::-1]
            label = label[:, ::-1]

        # Random vertical flip
        if np.random.random() > 0.5:
            image = image[:, ::-1, :]
            label = label[::-1, :]

        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k, axes=(1, 2))
            label = np.rot90(label, k, axes=(0, 1))

        return image, label


def get_dataloaders(patches_dir, batch_size=8, num_workers=4):
    """
    Create train/val/test DataLoaders with proper settings.
    Returns (train_loader, val_loader, test_loader, metadata).
    """
    patches_dir = Path(patches_dir)
    meta_path = patches_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    train_ds = LULCDataset(patches_dir, "train", metadata, augment=True)
    val_ds = LULCDataset(patches_dir, "val", metadata, augment=False)
    test_ds = LULCDataset(patches_dir, "test", metadata, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"  Train: {len(train_ds)} patches ({len(train_loader)} batches)")
    print(f"  Val:   {len(val_ds)} patches ({len(val_loader)} batches)")
    print(f"  Test:  {len(test_ds)} patches ({len(test_loader)} batches)")

    return train_loader, val_loader, test_loader, metadata
