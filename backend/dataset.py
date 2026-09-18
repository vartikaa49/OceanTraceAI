import random
from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    Dataset for Sentinel-1 SAR oil-spill segmentation.

    Each original image:
        2048 x 2048
        2 bands: VV and VH

    Each mask:
        2048 x 2048
        0 = background
        1 = oil spill

    Instead of loading the entire image into the model,
    we randomly extract 256 x 256 patches.
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        image_ids,
        patch_size=256,
        patches_per_image=8,
        positive_probability=0.7,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        self.image_ids = image_ids
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.positive_probability = positive_probability

    def __len__(self):
        return len(self.image_ids) * self.patches_per_image

    def _load_image_and_mask(self, image_id):
        image_path = self.image_dir / f"{image_id}.tif"
        mask_path = self.mask_dir / f"{image_id}.tif"

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)

        with rasterio.open(mask_path) as src:
            mask = src.read(1).astype(np.float32)

        return image, mask

    def _get_patch(self, image, mask):
        _, height, width = image.shape

        max_y = height - self.patch_size
        max_x = width - self.patch_size

        # Try to sample an oil-containing patch
        if random.random() < self.positive_probability:

            oil_y, oil_x = np.where(mask > 0)

            if len(oil_y) > 0:
                index = random.randint(0, len(oil_y) - 1)

                center_y = oil_y[index]
                center_x = oil_x[index]

                y = np.clip(
                    center_y - self.patch_size // 2,
                    0,
                    max_y
                )

                x = np.clip(
                    center_x - self.patch_size // 2,
                    0,
                    max_x
                )

            else:
                y = random.randint(0, max_y)
                x = random.randint(0, max_x)

        else:
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)

        image_patch = image[
            :,
            y:y + self.patch_size,
            x:x + self.patch_size
        ]

        mask_patch = mask[
            y:y + self.patch_size,
            x:x + self.patch_size
        ]

        return image_patch, mask_patch

    def __getitem__(self, index):

        image_index = index // self.patches_per_image

        image_id = self.image_ids[image_index]

        image, mask = self._load_image_and_mask(image_id)

        image_patch, mask_patch = self._get_patch(image, mask)

        # Replace invalid values
        image_patch = np.nan_to_num(
            image_patch,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Normalize SAR values approximately to 0-1
        image_patch = np.clip(
            image_patch,
            -50,
            5
        )

        image_patch = (image_patch + 50) / 55

        mask_patch = (mask_patch > 0).astype(np.float32)

        # Convert NumPy arrays to PyTorch tensors
        image_tensor = torch.tensor(
            image_patch,
            dtype=torch.float32
        )

        mask_tensor = torch.tensor(
            mask_patch,
            dtype=torch.float32
        ).unsqueeze(0)

        return image_tensor, mask_tensor


def get_image_ids(image_dir, mask_dir):
    """
    Return image IDs that have both
    an image and a corresponding mask.
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    image_ids = []

    for image_file in sorted(image_dir.glob("*.tif")):

        image_id = image_file.stem

        mask_file = mask_dir / f"{image_id}.tif"

        if mask_file.exists():
            image_ids.append(image_id)

    return image_ids


def split_dataset(image_ids, validation_ratio=0.2, seed=42):
    """
    Split by original image, NOT by patches.

    This prevents patches from the same image
    appearing in both training and validation sets.
    """

    image_ids = list(image_ids)

    random.Random(seed).shuffle(image_ids)

    split_index = int(
        len(image_ids) * (1 - validation_ratio)
    )

    train_ids = image_ids[:split_index]
    validation_ids = image_ids[split_index:]

    return train_ids, validation_ids