import os
import sys

import cv2
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.transform import xy


# ==================================================
# ALLOW IMPORTS FROM PROJECT ROOT
# ==================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from backend.model import UNet
from backend.dataset import (
    get_image_ids,
    split_dataset
)


# ==================================================
# CONFIGURATION
# ==================================================

IMAGE_DIR = "data/training/Oil"
MASK_DIR = "data/training/Mask_oil"

MODEL_PATH = "models/oil_spill_unet.pth"

OUTPUT_DIR = "outputs"

ENVIRONMENT_DIR = "data/environment"

PATCH_SIZE = 256

THRESHOLD = 0.5

SEED = 42


# ==================================================
# DEVICE
# ==================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ==================================================
# FIND VALIDATION IMAGE
# ==================================================

image_ids = get_image_ids(
    IMAGE_DIR,
    MASK_DIR
)

train_ids, validation_ids = split_dataset(
    image_ids,
    validation_ratio=0.2,
    seed=SEED
)

# Select first validation image
image_id = validation_ids[0]

print(
    "Selected unseen validation image:",
    image_id
)


# ==================================================
# LOAD MODEL
# ==================================================

print("\nLoading trained model...")

model = UNet(
    in_channels=2,
    out_channels=1
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("✓ Model loaded successfully.")


# ==================================================
# LOAD SENTINEL-1 IMAGE
# ==================================================

image_path = os.path.join(
    IMAGE_DIR,
    f"{image_id}.tif"
)

print("\nLoading image:")
print(image_path)

with rasterio.open(image_path) as src:

    image = src.read().astype(
        np.float32
    )

    profile = src.profile.copy()

    transform = src.transform

    crs = src.crs


print(
    "Image shape:",
    image.shape
)

print(
    "CRS:",
    crs
)


# ==================================================
# NORMALIZATION
# ==================================================

image = np.nan_to_num(
    image,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

image = np.clip(
    image,
    -50,
    5
)

image = (
    image + 50
) / 55


# ==================================================
# CREATE PREDICTION MASK
# ==================================================

height = image.shape[1]
width = image.shape[2]

prediction_mask = np.zeros(
    (height, width),
    dtype=np.uint8
)


# ==================================================
# RUN U-NET PREDICTION
# ==================================================

print(
    "\nRunning U-Net prediction..."
)

with torch.no_grad():

    for y in range(
        0,
        height,
        PATCH_SIZE
    ):

        for x in range(
            0,
            width,
            PATCH_SIZE
        ):

            patch = image[
                :,
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ]

            # Ignore incomplete edge patches
            if (
                patch.shape[1] != PATCH_SIZE
                or
                patch.shape[2] != PATCH_SIZE
            ):
                continue

            patch_tensor = torch.tensor(
                patch,
                dtype=torch.float32
            ).unsqueeze(0)

            patch_tensor = patch_tensor.to(
                device
            )

            output = model(
                patch_tensor
            )

            probability = torch.sigmoid(
                output
            )

            prediction = (
                probability > THRESHOLD
            ).float()

            prediction = (
                prediction
                .squeeze()
                .cpu()
                .numpy()
            )

            prediction_mask[
                y:y + PATCH_SIZE,
                x:x + PATCH_SIZE
            ] = prediction.astype(
                np.uint8
            )


print(
    "✓ Prediction completed."
)


# ==================================================
# CALCULATE SPILL PIXELS
# ==================================================

spill_pixels = np.sum(
    prediction_mask > 0
)

total_pixels = (
    height * width
)

spill_percentage = (
    spill_pixels
    /
    total_pixels
) * 100


print(
    "\n========== PREDICTION RESULT =========="
)

print(
    "Image ID:",
    image_id
)

print(
    "Spill pixels:",
    spill_pixels
)

print(
    "Spill percentage:",
    f"{spill_percentage:.4f}%"
)


# ==================================================
# AUTOMATIC SPILL CENTROID
# ==================================================

print(
    "\n========== SPILL LOCATION =========="
)

spill_y, spill_x = np.where(
    prediction_mask > 0
)


if len(spill_y) > 0:

    # Pixel-space centroid
    centroid_row = float(
        np.mean(spill_y)
    )

    centroid_col = float(
        np.mean(spill_x)
    )

    # Convert pixel coordinates
    # to geographic coordinates
    centroid_lon, centroid_lat = xy(
        transform,
        centroid_row,
        centroid_col
    )

    print(
        "Centroid row:",
        f"{centroid_row:.2f}"
    )

    print(
        "Centroid column:",
        f"{centroid_col:.2f}"
    )

    print(
        "Centroid latitude:",
        f"{centroid_lat:.6f}"
    )

    print(
        "Centroid longitude:",
        f"{centroid_lon:.6f}"
    )

else:

    centroid_lat = None
    centroid_lon = None

    print(
        "WARNING: No oil spill detected."
    )


# ==================================================
# SAVE AUTOMATIC SPILL LOCATION
# ==================================================

if centroid_lat is not None:

    os.makedirs(
        ENVIRONMENT_DIR,
        exist_ok=True
    )

    centroid_file = os.path.join(
        ENVIRONMENT_DIR,
        "detected_spill_location.csv"
    )

    centroid_data = pd.DataFrame([
        {
            "image_id": image_id,
            "latitude": centroid_lat,
            "longitude": centroid_lon
        }
    ])

    centroid_data.to_csv(
        centroid_file,
        index=False
    )

    print(
        "\n✓ Automatic spill location saved:"
    )

    print(
        centroid_file
    )

else:

    print(
        "\nNo spill location saved because"
        " no spill was detected."
    )


# ==================================================
# COMPARE WITH GROUND TRUTH
# ==================================================

mask_path = os.path.join(
    MASK_DIR,
    f"{image_id}.tif"
)

with rasterio.open(mask_path) as src:

    true_mask = src.read(1)


true_mask = (
    true_mask > 0
).astype(np.uint8)


# ==================================================
# DICE SCORE
# ==================================================

intersection = np.logical_and(
    prediction_mask,
    true_mask
).sum()

union = np.logical_or(
    prediction_mask,
    true_mask
).sum()

pred_sum = prediction_mask.sum()

true_sum = true_mask.sum()


if (
    pred_sum + true_sum
) > 0:

    dice = (
        2 * intersection
        /
        (pred_sum + true_sum)
    )

else:

    dice = 1.0


# ==================================================
# IOU SCORE
# ==================================================

if union > 0:

    iou = (
        intersection
        /
        union
    )

else:

    iou = 1.0


print(
    "\n========== MODEL PERFORMANCE =========="
)

print(
    "Dice:",
    f"{dice:.4f}"
)

print(
    "IoU:",
    f"{iou:.4f}"
)


# ==================================================
# SAVE PREDICTION MASK
# ==================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

output_mask_path = os.path.join(
    OUTPUT_DIR,
    f"{image_id}_prediction.tif"
)

output_profile = profile.copy()

output_profile.update(
    dtype=rasterio.uint8,
    count=1,
    compress="lzw"
)

with rasterio.open(
    output_mask_path,
    "w",
    **output_profile
) as dst:

    dst.write(
        prediction_mask,
        1
    )


print(
    "\n✓ Prediction mask saved:"
)

print(
    output_mask_path
)


# ==================================================
# CREATE VISUALIZATION
# ==================================================

# Use original normalized VV band
vv = image[0]


# Convert normalized VV to 0-255
vv_display = (
    vv * 255
).astype(np.uint8)


vv_display = cv2.cvtColor(
    vv_display,
    cv2.COLOR_GRAY2BGR
)


# ==================================================
# CREATE OIL OVERLAY
# ==================================================

overlay = vv_display.copy()

overlay[
    prediction_mask > 0
] = [0, 0, 255]


# Blend original and prediction
result = cv2.addWeighted(
    vv_display,
    0.65,
    overlay,
    0.35,
    0
)


# ==================================================
# ADD INFORMATION
# ==================================================

cv2.putText(
    result,
    f"Oil pixels: {spill_pixels}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)

cv2.putText(
    result,
    f"Dice: {dice:.3f}  IoU: {iou:.3f}",
    (20, 75),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)


# ==================================================
# MARK SPILL CENTROID
# ==================================================

if centroid_lat is not None:

    center_x = int(
        round(centroid_col)
    )

    center_y = int(
        round(centroid_row)
    )

    cv2.circle(
        result,
        (center_x, center_y),
        12,
        (0, 255, 0),
        3
    )

    cv2.putText(
        result,
        "SPILL CENTROID",
        (
            center_x + 15,
            center_y
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        result,
        f"Lat: {centroid_lat:.5f}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        result,
        f"Lon: {centroid_lon:.5f}",
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


# ==================================================
# SAVE VISUALIZATION
# ==================================================

output_image_path = os.path.join(
    OUTPUT_DIR,
    f"{image_id}_result.png"
)

cv2.imwrite(
    output_image_path,
    result
)


print(
    "\n✓ Visualization saved:"
)

print(
    output_image_path
)


# ==================================================
# FINAL SUMMARY
# ==================================================

print(
    "\n======================================"
)

print(
    "      OCEANTRACEAI PREDICTION"
)

print(
    "======================================"
)

print(
    "Image ID:",
    image_id
)

print(
    "Oil pixels:",
    spill_pixels
)

print(
    "Spill percentage:",
    f"{spill_percentage:.4f}%"
)

if centroid_lat is not None:

    print(
        "Spill centroid:",
        f"{centroid_lat:.6f}, "
        f"{centroid_lon:.6f}"
    )

    print(
        "Centroid file:",
        "data/environment/"
        "detected_spill_location.csv"
    )

else:

    print(
        "Spill centroid: NOT DETECTED"
    )

print(
    "Dice:",
    f"{dice:.4f}"
)

print(
    "IoU:",
    f"{iou:.4f}"
)

print(
    "\nPrediction finished successfully!"
)

print(
    "======================================"
)