from pathlib import Path

import cv2
import numpy as np
import rasterio
import torch

from model import UNet


# ============================================================
# CONFIGURATION
# ============================================================

# IMPORTANT:
# Training TIFF order was determined to be:
#   Band 1 = VH
#   Band 2 = VV
#
# Therefore the U-Net MUST receive:
#   Channel 1 = VH
#   Channel 2 = VV

VH_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VH_preprocessed.tif"
)

VV_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VV_preprocessed.tif"
)

MODEL_PATH = Path(
    "models/oil_spill_unet.pth"
)

OUTPUT_DIR = Path(
    "outputs/sentinel1_real"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_PATH = (
    OUTPUT_DIR /
    "real_sentinel1_prediction.tif"
)

VISUALIZATION_PATH = (
    OUTPUT_DIR /
    "real_sentinel1_prediction.png"
)

PATCH_SIZE = 256

THRESHOLD = 0.50

# Same normalization used during training.
TRAIN_MIN = -50.0
TRAIN_MAX = 5.0

# IMPORTANT:
# Border padding must NOT use 0 dB.
# 0 dB becomes a very bright normalized input.
# Use the lower training bound instead.
PAD_VALUE = TRAIN_MIN

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 70)
    print("LOADING OIL-SPILL U-NET")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Device: {DEVICE}")

    model = UNet(
        in_channels=2,
        out_channels=1
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint[
                "model_state_dict"
            ]
        else:
            state_dict = checkpoint

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(DEVICE)
    model.eval()

    print(
        "✓ Model loaded successfully."
    )

    return model


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_channel(data):

    data = data.astype(
        np.float32
    )

    valid = (
        np.isfinite(data)
        & (data != -9999)
    )

    output = np.zeros_like(
        data,
        dtype=np.float32
    )

    clipped = np.clip(
        data[valid],
        TRAIN_MIN,
        TRAIN_MAX
    )

    output[valid] = (
        (clipped - TRAIN_MIN)
        /
        (TRAIN_MAX - TRAIN_MIN)
    )

    return output


# ============================================================
# PREDICTION
# ============================================================

def predict_scene(model):

    print("\n" + "=" * 70)
    print("REAL SENTINEL-1 U-NET INFERENCE")
    print("=" * 70)

    print(
        "\nINPUT CHANNEL ORDER"
    )

    print(
        "Channel 1 → VH"
    )

    print(
        "Channel 2 → VV"
    )

    print(
        f"\nVH: {VH_PATH}"
    )

    print(
        f"VV: {VV_PATH}"
    )

    with rasterio.open(VH_PATH) as vh_src, \
         rasterio.open(VV_PATH) as vv_src:

        width = vh_src.width
        height = vh_src.height

        print(
            f"\nScene size: "
            f"{width} × {height}"
        )

        if (
            vh_src.width != vv_src.width
            or
            vh_src.height != vv_src.height
        ):

            raise ValueError(
                "VH and VV dimensions do not match."
            )

        prediction = np.zeros(
            (height, width),
            dtype=np.float32
        )

        count_map = np.zeros(
            (height, width),
            dtype=np.float32
        )

        total_tiles = (
            (
                height +
                PATCH_SIZE -
                1
            )
            //
            PATCH_SIZE
        ) * (
            (
                width +
                PATCH_SIZE -
                1
            )
            //
            PATCH_SIZE
        )

        tile_number = 0

        # ----------------------------------------------------
        # Track valid pixels separately.
        # ----------------------------------------------------

        valid_scene = np.zeros(
            (height, width),
            dtype=bool
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

                    tile_width = min(
                        PATCH_SIZE,
                        width - x
                    )

                    tile_height = min(
                        PATCH_SIZE,
                        height - y
                    )

                    window = (
                        rasterio.windows.Window(
                            x,
                            y,
                            tile_width,
                            tile_height
                        )
                    )

                    # ------------------------------------------------
                    # Read VH and VV.
                    # ------------------------------------------------

                    vh = vh_src.read(
                        1,
                        window=window
                    )

                    vv = vv_src.read(
                        1,
                        window=window
                    )

                    valid = (
                        np.isfinite(vh)
                        &
                        np.isfinite(vv)
                        &
                        (vh != -9999)
                        &
                        (vv != -9999)
                    )

                    if valid.sum() == 0:

                        tile_number += 1
                        continue

                    valid_scene[
                        y:y + tile_height,
                        x:x + tile_width
                    ] |= valid

                    # ------------------------------------------------
                    # Create padded tiles.
                    #
                    # IMPORTANT:
                    # Padding uses -50 dB rather than 0 dB.
                    # ------------------------------------------------

                    vh_full = np.full(
                        (
                            PATCH_SIZE,
                            PATCH_SIZE
                        ),
                        PAD_VALUE,
                        dtype=np.float32
                    )

                    vv_full = np.full(
                        (
                            PATCH_SIZE,
                            PATCH_SIZE
                        ),
                        PAD_VALUE,
                        dtype=np.float32
                    )

                    vh_full[
                        :tile_height,
                        :tile_width
                    ] = vh

                    vv_full[
                        :tile_height,
                        :tile_width
                    ] = vv

                    # ------------------------------------------------
                    # Normalize.
                    # ------------------------------------------------

                    vh_norm = normalize_channel(
                        vh_full
                    )

                    vv_norm = normalize_channel(
                        vv_full
                    )

                    # ------------------------------------------------
                    # CRITICAL:
                    #
                    # Channel 1 = VH
                    # Channel 2 = VV
                    # ------------------------------------------------

                    image = np.stack(
                        [
                            vh_norm,
                            vv_norm
                        ],
                        axis=0
                    )

                    tensor = torch.tensor(
                        image,
                        dtype=torch.float32
                    ).unsqueeze(
                        0
                    ).to(
                        DEVICE
                    )

                    # ------------------------------------------------
                    # U-NET
                    # ------------------------------------------------

                    logits = model(
                        tensor
                    )

                    probabilities = (
                        torch.sigmoid(
                            logits
                        )
                    )

                    probability = (
                        probabilities[
                            0,
                            0
                        ]
                        .cpu()
                        .numpy()
                    )

                    probability = (
                        probability[
                            :tile_height,
                            :tile_width
                        ]
                    )

                    # ------------------------------------------------
                    # Only accumulate genuinely valid pixels.
                    # ------------------------------------------------

                    probability_masked = (
                        probability *
                        valid.astype(
                            np.float32
                        )
                    )

                    prediction[
                        y:y + tile_height,
                        x:x + tile_width
                    ] += probability_masked

                    count_map[
                        y:y + tile_height,
                        x:x + tile_width
                    ] += valid.astype(
                        np.float32
                    )

                    tile_number += 1

                    if (
                        tile_number % 250 == 0
                        or
                        tile_number ==
                        total_tiles
                    ):

                        percent = (
                            tile_number
                            /
                            total_tiles
                            *
                            100
                        )

                        print(
                            f"Tiles: "
                            f"{tile_number:,}/"
                            f"{total_tiles:,} "
                            f"({percent:.1f}%)"
                        )

        # --------------------------------------------------------
        # Average predictions.
        # --------------------------------------------------------

        valid_count = (
            count_map > 0
        )

        prediction[
            valid_count
        ] /= (
            count_map[
                valid_count
            ]
        )

        # Pixels outside the actual scene footprint
        # remain zero.
        prediction[
            ~valid_scene
        ] = 0.0

    return prediction, valid_scene


# ============================================================
# SAVE PREDICTION
# ============================================================

def save_prediction(prediction):

    print("\n" + "=" * 70)
    print("SAVING PREDICTION")
    print("=" * 70)

    with rasterio.open(
        VH_PATH
    ) as src:

        profile = (
            src.profile.copy()
        )

        profile.update(
            dtype="float32",
            count=1,
            nodata=0.0,
            compress="deflate",
            predictor=3,
            BIGTIFF="YES"
        )

        with rasterio.open(
            PREDICTION_PATH,
            "w",
            **profile
        ) as dst:

            dst.write(
                prediction.astype(
                    np.float32
                ),
                1
            )

    print(
        "✓ Prediction saved:"
    )

    print(
        PREDICTION_PATH
    )


# ============================================================
# STATISTICS
# ============================================================

def print_statistics(
    prediction,
    valid_scene
):

    # --------------------------------------------------------
    # Only evaluate pixels inside the actual scene footprint.
    # --------------------------------------------------------

    valid_probabilities = (
        prediction[
            valid_scene
        ]
    )

    if valid_probabilities.size == 0:

        raise RuntimeError(
            "No valid Sentinel-1 pixels found."
        )

    binary = (
        prediction >= THRESHOLD
    )

    binary[
        ~valid_scene
    ] = False

    spill_pixels = int(
        binary.sum()
    )

    valid_pixels = int(
        valid_scene.sum()
    )

    spill_percentage = (
        spill_pixels
        /
        valid_pixels
        *
        100
    )

    print("\n" + "=" * 70)
    print("AI PREDICTION RESULT")
    print("=" * 70)

    print(
        f"Threshold        : "
        f"{THRESHOLD}"
    )

    print(
        f"Valid scene pixels: "
        f"{valid_pixels:,}"
    )

    print(
        f"Predicted pixels : "
        f"{spill_pixels:,}"
    )

    print(
        f"Predicted area   : "
        f"{spill_percentage:.4f}%"
    )

    print(
        f"Probability min  : "
        f"{valid_probabilities.min():.4f}"
    )

    print(
        f"Probability max  : "
        f"{valid_probabilities.max():.4f}"
    )

    print(
        f"Probability mean : "
        f"{valid_probabilities.mean():.4f}"
    )

    # --------------------------------------------------------
    # Probability percentiles.
    # --------------------------------------------------------

    percentiles = np.percentile(
        valid_probabilities,
        [
            1,
            5,
            25,
            50,
            75,
            95,
            99
        ]
    )

    print(
        "\nProbability percentiles:"
    )

    print(
        f"P01 : {percentiles[0]:.4f}"
    )

    print(
        f"P05 : {percentiles[1]:.4f}"
    )

    print(
        f"P25 : {percentiles[2]:.4f}"
    )

    print(
        f"P50 : {percentiles[3]:.4f}"
    )

    print(
        f"P75 : {percentiles[4]:.4f}"
    )

    print(
        f"P95 : {percentiles[5]:.4f}"
    )

    print(
        f"P99 : {percentiles[6]:.4f}"
    )

    # --------------------------------------------------------
    # Count predictions at several thresholds.
    # --------------------------------------------------------

    print(
        "\nThreshold sensitivity:"
    )

    for threshold in [
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]:

        count = int(
            (
                prediction[
                    valid_scene
                ]
                >= threshold
            ).sum()
        )

        percentage = (
            count
            /
            valid_pixels
            *
            100
        )

        print(
            f"Threshold {threshold:.2f}"
            f" → "
            f"{percentage:.4f}%"
        )

    return binary, spill_percentage


# ============================================================
# VISUALIZATION
# ============================================================

def save_visualization(
    prediction,
    binary,
    valid_scene
):

    print(
        "\nCreating visualization..."
    )

    scale = 4

    new_width = max(
        1,
        prediction.shape[1] // scale
    )

    new_height = max(
        1,
        prediction.shape[0] // scale
    )

    small_probability = cv2.resize(
        prediction,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )

    small_valid = cv2.resize(
        valid_scene.astype(
            np.uint8
        ),
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    small_mask = (
        small_probability >=
        THRESHOLD
    )

    small_mask[
        ~small_valid
    ] = False

    # --------------------------------------------------------
    # Probability grayscale.
    # --------------------------------------------------------

    image = (
        np.clip(
            small_probability,
            0,
            1
        )
        * 255
    ).astype(
        np.uint8
    )

    visualization = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR
    )

    # --------------------------------------------------------
    # Make invalid pixels black.
    # --------------------------------------------------------

    visualization[
        ~small_valid
    ] = 0

    # --------------------------------------------------------
    # Highlight predicted regions.
    # --------------------------------------------------------

    visualization[
        small_mask
    ] = (
        0,
        0,
        255
    )

    # --------------------------------------------------------
    # Find contours.
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        small_mask.astype(
            np.uint8
        ),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        visualization,
        contours,
        -1,
        (
            0,
            255,
            0
        ),
        1
    )

    cv2.imwrite(
        str(
            VISUALIZATION_PATH
        ),
        visualization
    )

    print(
        "✓ Visualization saved:"
    )

    print(
        VISUALIZATION_PATH
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "OCEANTRACEAI"
    )

    print(
        "REAL SENTINEL-1 → U-NET PIPELINE"
    )

    print(
        "=" * 70
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This is a real Sentinel-1 scene."
    )

    print(
        "The current scene is from the Gulf of Mexico."
    )

    print(
        "It is NOT the Indian Ocean demonstration."
    )

    print(
        "\nU-NET INPUT ORDER:"
    )

    print(
        "Channel 1 = VH"
    )

    print(
        "Channel 2 = VV"
    )

    print(
        f"\nPadding value = "
        f"{PAD_VALUE} dB"
    )

    model = load_model()

    prediction, valid_scene = (
        predict_scene(
            model
        )
    )

    save_prediction(
        prediction
    )

    binary, spill_percentage = (
        print_statistics(
            prediction,
            valid_scene
        )
    )

    save_visualization(
        prediction,
        binary,
        valid_scene
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "REAL SENTINEL-1 INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nPredicted spill percentage: "
        f"{spill_percentage:.4f}%"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This prediction must pass "
        "spatial/visual validation before "
        "being treated as an oil spill."
    )

    print(
        "\nNext diagnostic:"
    )

    print(
        "Check prediction distribution "
        "and connected components."
    )


if __name__ == "__main__":
    main()