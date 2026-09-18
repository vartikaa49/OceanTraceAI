import os
import sys
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from backend
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from backend.dataset import (
    OilSpillDataset,
    get_image_ids,
    split_dataset
)

from backend.model import UNet


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_DIR = "data/training/Oil"
MASK_DIR = "data/training/Mask_oil"

MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "oil_spill_unet.pth"
)

PATCH_SIZE = 256

# Reduced from 4 to make CPU training lighter
PATCHES_PER_IMAGE = 2

# Small batch for CPU
BATCH_SIZE = 2

# Total epochs
EPOCHS = 10

LEARNING_RATE = 0.001

VALIDATION_RATIO = 0.2

SEED = 42


# ==================================================
# REPRODUCIBILITY
# ==================================================

random.seed(SEED)
torch.manual_seed(SEED)


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ==================================================
# DATASET
# ==================================================

print("\nLoading dataset...")

image_ids = get_image_ids(
    IMAGE_DIR,
    MASK_DIR
)

train_ids, validation_ids = split_dataset(
    image_ids,
    validation_ratio=VALIDATION_RATIO,
    seed=SEED
)

print("Total images:", len(image_ids))
print("Training images:", len(train_ids))
print("Validation images:", len(validation_ids))


train_dataset = OilSpillDataset(
    IMAGE_DIR,
    MASK_DIR,
    train_ids,
    patch_size=PATCH_SIZE,
    patches_per_image=PATCHES_PER_IMAGE,
    positive_probability=0.7
)

validation_dataset = OilSpillDataset(
    IMAGE_DIR,
    MASK_DIR,
    validation_ids,
    patch_size=PATCH_SIZE,
    patches_per_image=PATCHES_PER_IMAGE,
    positive_probability=0.7
)


# ==================================================
# DATALOADERS
# ==================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Training patches:", len(train_dataset))
print("Validation patches:", len(validation_dataset))


# ==================================================
# MODEL
# ==================================================

model = UNet(
    in_channels=2,
    out_channels=1
)

model = model.to(device)


# ==================================================
# RESUME FROM SAVED MODEL
# ==================================================

start_epoch = 0

if os.path.exists(MODEL_PATH):

    print("\nSaved model found.")
    print("Loading:", MODEL_PATH)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    # Epoch 1 was already completed
    start_epoch = 1

    print("✓ Previous model loaded.")
    print("✓ Continuing from Epoch 2.")

else:

    print("\nNo saved model found.")
    print("Starting training from Epoch 1.")


# ==================================================
# LOSS FUNCTIONS
# ==================================================

bce_loss = nn.BCEWithLogitsLoss()


def dice_loss(prediction, target):

    prediction = torch.sigmoid(prediction)

    smooth = 1e-6

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return 1 - dice


def combined_loss(prediction, target):

    bce = bce_loss(
        prediction,
        target
    )

    dice = dice_loss(
        prediction,
        target
    )

    return bce + dice


# ==================================================
# METRICS
# ==================================================

def calculate_dice(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    smooth = 1e-6

    intersection = (
        prediction * target
    ).sum()

    dice = (
        2 * intersection + smooth
    ) / (
        prediction.sum()
        + target.sum()
        + smooth
    )

    return dice.item()


def calculate_iou(prediction, target):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > 0.5
    ).float()

    intersection = (
        prediction * target
    ).sum()

    union = (
        prediction
        + target
        - prediction * target
    ).sum()

    smooth = 1e-6

    iou = (
        intersection + smooth
    ) / (
        union + smooth
    )

    return iou.item()


# ==================================================
# OPTIMIZER
# ==================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==================================================
# TRAINING
# ==================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

best_validation_loss = float("inf")


print("\nStarting training...\n")


for epoch in range(
    start_epoch,
    EPOCHS
):

    # ----------------------------------------------
    # TRAINING
    # ----------------------------------------------

    model.train()

    training_loss = 0.0

    for images, masks in train_loader:

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        predictions = model(images)

        loss = combined_loss(
            predictions,
            masks
        )

        loss.backward()

        optimizer.step()

        training_loss += loss.item()

    training_loss /= len(train_loader)


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    model.eval()

    validation_loss = 0.0
    validation_dice = 0.0
    validation_iou = 0.0

    with torch.no_grad():

        for images, masks in validation_loader:

            images = images.to(device)
            masks = masks.to(device)

            predictions = model(images)

            loss = combined_loss(
                predictions,
                masks
            )

            validation_loss += loss.item()

            validation_dice += calculate_dice(
                predictions,
                masks
            )

            validation_iou += calculate_iou(
                predictions,
                masks
            )


    validation_loss /= len(validation_loader)

    validation_dice /= len(validation_loader)

    validation_iou /= len(validation_loader)


    # ----------------------------------------------
    # RESULTS
    # ----------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Train Loss: {training_loss:.4f} "
        f"Val Loss: {validation_loss:.4f} "
        f"Dice: {validation_dice:.4f} "
        f"IoU: {validation_iou:.4f}"
    )


    # ----------------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )

        print(
            "  ✓ Best model saved!"
        )


print("\nTraining complete.")

print(
    "Best model:",
    MODEL_PATH
)