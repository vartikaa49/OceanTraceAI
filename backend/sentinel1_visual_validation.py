from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

VV_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VV_preprocessed.tif"
)

VH_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VH_preprocessed.tif"
)

PREDICTION_PATH = Path(
    "outputs/sentinel1_real/"
    "real_sentinel1_prediction.tif"
)

GEOJSON_PATH = Path(
    "outputs/sentinel1_real/"
    "real_sentinel1_spill.geojson"
)

OUTPUT_PATH = Path(
    "outputs/sentinel1_real/"
    "real_sentinel1_validation.png"
)


# ============================================================
# LOAD GEOLOCATION GRID
# ============================================================

def load_geolocation():

    path = Path(
        "data/processed/sentinel1/"
        "sentinel1_geolocation_grid.json"
    )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return (
        np.asarray(data["lines"], dtype=float),
        np.asarray(data["pixels"], dtype=float),
        np.asarray(data["latitude"], dtype=float),
        np.asarray(data["longitude"], dtype=float)
    )


# ============================================================
# GEOLOCATION INTERPOLATION
# ============================================================

def interpolate(
    row,
    col,
    lines,
    pixels,
    latitude,
    longitude
):

    row = np.clip(
        row,
        lines[0],
        lines[-1]
    )

    col = np.clip(
        col,
        pixels[0],
        pixels[-1]
    )

    r = np.searchsorted(
        lines,
        row
    )

    r = np.clip(
        r,
        1,
        len(lines) - 1
    )

    c = np.searchsorted(
        pixels,
        col
    )

    c = np.clip(
        c,
        1,
        len(pixels) - 1
    )

    r0 = r - 1
    r1 = r

    c0 = c - 1
    c1 = c

    if lines[r1] == lines[r0]:

        wr = 0.0

    else:

        wr = (
            row - lines[r0]
        ) / (
            lines[r1] - lines[r0]
        )

    if pixels[c1] == pixels[c0]:

        wc = 0.0

    else:

        wc = (
            col - pixels[c0]
        ) / (
            pixels[c1] - pixels[c0]
        )

    lat = (
        latitude[r0, c0] * (1 - wr) * (1 - wc)
        + latitude[r0, c1] * (1 - wr) * wc
        + latitude[r1, c0] * wr * (1 - wc)
        + latitude[r1, c1] * wr * wc
    )

    lon = (
        longitude[r0, c0] * (1 - wr) * (1 - wc)
        + longitude[r0, c1] * (1 - wr) * wc
        + longitude[r1, c0] * wr * (1 - wc)
        + longitude[r1, c1] * wr * wc
    )

    return float(lat), float(lon)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCEANTRACEAI")
    print("REAL SENTINEL-1 VISUAL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    print("\nLoading AI prediction...")

    with rasterio.open(
        PREDICTION_PATH
    ) as src:

        prediction = src.read(1)

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
    )

    # --------------------------------------------------------
    # Load VV
    # --------------------------------------------------------

    print("\nLoading VV...")

    with rasterio.open(
        VV_PATH
    ) as src:

        vv = src.read(1)

    # --------------------------------------------------------
    # Load VH
    # --------------------------------------------------------

    print("Loading VH...")

    with rasterio.open(
        VH_PATH
    ) as src:

        vh = src.read(1)

    print(
        f"SAR shape: "
        f"{vv.shape}"
    )

    # --------------------------------------------------------
    # Load GeoJSON
    # --------------------------------------------------------

    print("\nLoading spill geometry...")

    with open(
        GEOJSON_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        geojson = json.load(f)

    if not geojson["features"]:

        raise RuntimeError(
            "No spill geometry found."
        )

    feature = (
        geojson["features"][0]
    )

    polygon = np.asarray(
        feature["geometry"]["coordinates"][0],
        dtype=float
    )

    properties = feature["properties"]

    print(
        f"Area: "
        f"{properties['area_km2']:.4f} km²"
    )

    print(
        f"Centroid: "
        f"{properties['centroid_latitude']:.6f}, "
        f"{properties['centroid_longitude']:.6f}"
    )

    # --------------------------------------------------------
    # Load geolocation grid
    # --------------------------------------------------------

    print("\nLoading geolocation grid...")

    (
        lines,
        pixels,
        latitude,
        longitude
    ) = load_geolocation()

    # --------------------------------------------------------
    # Convert polygon coordinates back to approximate
    # image pixel coordinates.
    #
    # We use nearest geolocation-grid point.
    # --------------------------------------------------------

    polygon_pixels = []

    for lon, lat in polygon:

        distance = (
            (latitude - lat) ** 2
            + (longitude - lon) ** 2
        )

        index = np.unravel_index(
            np.argmin(distance),
            distance.shape
        )

        row_index = index[0]
        col_index = index[1]

        row = lines[row_index]
        col = pixels[col_index]

        polygon_pixels.append(
            [col, row]
        )

    polygon_pixels = np.asarray(
        polygon_pixels,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Create mask for polygon.
    # --------------------------------------------------------

    polygon_mask = np.zeros(
        prediction.shape,
        dtype=np.uint8
    )

    cv2.fillPoly(
        polygon_mask,
        [
            polygon_pixels.astype(
                np.int32
            )
        ],
        1
    )

    # --------------------------------------------------------
    # Downsample for visualization.
    # --------------------------------------------------------

    scale = 8

    height, width = prediction.shape

    small_width = width // scale
    small_height = height // scale

    vv_small = cv2.resize(
        vv,
        (
            small_width,
            small_height
        ),
        interpolation=cv2.INTER_AREA
    )

    prediction_small = cv2.resize(
        prediction,
        (
            small_width,
            small_height
        ),
        interpolation=cv2.INTER_AREA
    )

    polygon_small = cv2.resize(
        polygon_mask,
        (
            small_width,
            small_height
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Normalize VV for display.
    # --------------------------------------------------------

    vv_display = np.clip(
        vv_small,
        -15,
        15
    )

    vv_display = (
        vv_display + 15
    ) / 30

    # --------------------------------------------------------
    # Create RGB-style visualization.
    # --------------------------------------------------------

    rgb = np.zeros(
        (
            small_height,
            small_width,
            3
        ),
        dtype=np.float32
    )

    rgb[:, :, 0] = vv_display

    rgb[:, :, 1] = vv_display

    rgb[:, :, 2] = vv_display

    # --------------------------------------------------------
    # Overlay AI probability.
    # --------------------------------------------------------

    probability = np.clip(
        prediction_small,
        0,
        1
    )

    rgb[:, :, 0] = np.maximum(
        rgb[:, :, 0],
        probability * 0.9
    )

    # --------------------------------------------------------
    # Draw polygon boundary.
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        polygon_small.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:

        cv2.drawContours(
            rgb,
            [contour],
            -1,
            (0, 1, 0),
            2
        )

    # --------------------------------------------------------
    # Create figure.
    # --------------------------------------------------------

    plt.figure(
        figsize=(12, 8)
    )

    plt.imshow(
        rgb,
        origin="upper"
    )

    plt.title(
        "OceanTraceAI — Real Sentinel-1 AI Spill Validation\n"
        f"Predicted Area: "
        f"{properties['area_km2']:.2f} km² | "
        f"Centroid: "
        f"{properties['centroid_latitude']:.4f}, "
        f"{properties['centroid_longitude']:.4f}"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_PATH,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\n✓ Visualization saved:"
        f"\n{OUTPUT_PATH}"
    )

    print("\n" + "=" * 70)
    print("VISUAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()