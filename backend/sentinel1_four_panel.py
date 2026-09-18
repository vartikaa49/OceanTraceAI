from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
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
    "sentinel1_four_panel_validation.png"
)

GEOLOCATION_PATH = Path(
    "data/processed/sentinel1/"
    "sentinel1_geolocation_grid.json"
)

THRESHOLD = 0.50

# Visualization downsampling.
DISPLAY_SCALE = 16


# ============================================================
# LOAD GEOLOCATION GRID
# ============================================================

def load_geolocation_grid():

    with open(
        GEOLOCATION_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    lines = np.asarray(
        data["lines"],
        dtype=np.float64
    )

    pixels = np.asarray(
        data["pixels"],
        dtype=np.float64
    )

    latitude = np.asarray(
        data["latitude"],
        dtype=np.float64
    )

    longitude = np.asarray(
        data["longitude"],
        dtype=np.float64
    )

    return (
        lines,
        pixels,
        latitude,
        longitude
    )


# ============================================================
# GEOLOCATION INTERPOLATION
# ============================================================

def interpolate_geolocation(
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
# GEOJSON POLYGON → IMAGE PIXELS
# ============================================================

def polygon_geo_to_pixels(
    polygon,
    lines,
    pixels,
    latitude,
    longitude
):

    result = []

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

        result.append(
            [col, row]
        )

    return np.asarray(
        result,
        dtype=np.float32
    )


# ============================================================
# RESIZE FOR DISPLAY
# ============================================================

def resize_for_display(
    image,
    scale,
    interpolation=cv2.INTER_AREA
):

    height, width = image.shape[:2]

    new_width = max(
        1,
        width // scale
    )

    new_height = max(
        1,
        height // scale
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=interpolation
    )


# ============================================================
# NORMALIZE BACKSCATTER
# ============================================================

def normalize_backscatter(
    image,
    lower,
    upper
):

    image = np.clip(
        image,
        lower,
        upper
    )

    image = (
        image - lower
    ) / (
        upper - lower
    )

    return image


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCEANTRACEAI")
    print("SENTINEL-1 FOUR-PANEL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load VV
    # --------------------------------------------------------

    print("\nLoading VV...")

    with rasterio.open(
        VV_PATH
    ) as src:

        vv = src.read(1).astype(
            np.float32
        )

    print(
        f"VV shape: {vv.shape}"
    )

    # --------------------------------------------------------
    # Load VH
    # --------------------------------------------------------

    print("\nLoading VH...")

    with rasterio.open(
        VH_PATH
    ) as src:

        vh = src.read(1).astype(
            np.float32
        )

    print(
        f"VH shape: {vh.shape}"
    )

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    print("\nLoading U-Net prediction...")

    with rasterio.open(
        PREDICTION_PATH
    ) as src:

        prediction = src.read(1).astype(
            np.float32
        )

    print(
        f"Prediction shape: "
        f"{prediction.shape}"
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

    features = geojson.get(
        "features",
        []
    )

    if not features:

        raise RuntimeError(
            "No spill polygon found in GeoJSON."
        )

    feature = features[0]

    properties = feature[
        "properties"
    ]

    polygon = np.asarray(
        feature[
            "geometry"
        ][
            "coordinates"
        ][0],
        dtype=np.float64
    )

    area_km2 = float(
        properties["area_km2"]
    )

    perimeter_km = float(
        properties["perimeter_km"]
    )

    centroid_lat = float(
        properties["centroid_latitude"]
    )

    centroid_lon = float(
        properties["centroid_longitude"]
    )

    print(
        f"Area: {area_km2:.4f} km²"
    )

    print(
        f"Perimeter: "
        f"{perimeter_km:.4f} km"
    )

    print(
        f"Centroid: "
        f"{centroid_lat:.6f}, "
        f"{centroid_lon:.6f}"
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
    ) = load_geolocation_grid()

    # --------------------------------------------------------
    # Convert geographic polygon to image coordinates
    # --------------------------------------------------------

    polygon_pixels = polygon_geo_to_pixels(
        polygon,
        lines,
        pixels,
        latitude,
        longitude
    )

    # --------------------------------------------------------
    # Create full-resolution polygon mask
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
    # AI binary mask
    # --------------------------------------------------------

    ai_mask = (
        prediction >= THRESHOLD
    ).astype(
        np.uint8
    )

    # --------------------------------------------------------
    # Downsample all layers
    # --------------------------------------------------------

    vv_small = resize_for_display(
        vv,
        DISPLAY_SCALE
    )

    vh_small = resize_for_display(
        vh,
        DISPLAY_SCALE
    )

    probability_small = resize_for_display(
        prediction,
        DISPLAY_SCALE
    )

    ai_mask_small = resize_for_display(
        ai_mask,
        DISPLAY_SCALE,
        cv2.INTER_NEAREST
    )

    polygon_small = resize_for_display(
        polygon_mask,
        DISPLAY_SCALE,
        cv2.INTER_NEAREST
    )

    # --------------------------------------------------------
    # Normalize VV and VH for display
    # --------------------------------------------------------

    vv_display = normalize_backscatter(
        vv_small,
        -15,
        15
    )

    vh_display = normalize_backscatter(
        vh_small,
        -20,
        10
    )

    # --------------------------------------------------------
    # Find polygon contour
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        polygon_small,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # --------------------------------------------------------
    # Convert centroid to display pixel coordinates
    # --------------------------------------------------------

    centroid_distance = (
        (latitude - centroid_lat) ** 2
        + (longitude - centroid_lon) ** 2
    )

    centroid_index = np.unravel_index(
        np.argmin(
            centroid_distance
        ),
        centroid_distance.shape
    )

    centroid_row = (
        lines[centroid_index[0]]
        / DISPLAY_SCALE
    )

    centroid_col = (
        pixels[centroid_index[1]]
        / DISPLAY_SCALE
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12)
    )

    # ========================================================
    # PANEL A — VV
    # ========================================================

    ax = axes[0, 0]

    ax.imshow(
        vv_display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        "(A) Sentinel-1 VV Backscatter",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Range / Pixel"
    )

    ax.set_ylabel(
        "Azimuth / Line"
    )

    # Polygon boundary
    for contour in contours:

        contour_display = (
            contour[:, 0, :]
        )

        ax.plot(
            contour_display[:, 0],
            contour_display[:, 1],
            linewidth=2
        )

    # ========================================================
    # PANEL B — VH
    # ========================================================

    ax = axes[0, 1]

    ax.imshow(
        vh_display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        "(B) Sentinel-1 VH Backscatter",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Range / Pixel"
    )

    ax.set_ylabel(
        "Azimuth / Line"
    )

    for contour in contours:

        contour_display = (
            contour[:, 0, :]
        )

        ax.plot(
            contour_display[:, 0],
            contour_display[:, 1],
            linewidth=2
        )

    # ========================================================
    # PANEL C — U-NET PROBABILITY
    # ========================================================

    ax = axes[1, 0]

    image = ax.imshow(
        probability_small,
        cmap="inferno",
        vmin=0,
        vmax=1
    )

    ax.set_title(
        "(C) U-Net Oil Spill Probability",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Range / Pixel"
    )

    ax.set_ylabel(
        "Azimuth / Line"
    )

    plt.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="Probability"
    )

    # Polygon boundary

    for contour in contours:

        contour_display = (
            contour[:, 0, :]
        )

        ax.plot(
            contour_display[:, 0],
            contour_display[:, 1],
            linewidth=2
        )

    # ========================================================
    # PANEL D — FINAL AI MASK
    # ========================================================

    ax = axes[1, 1]

    ax.imshow(
        vv_display,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    ax.imshow(
        np.ma.masked_where(
            ai_mask_small == 0,
            ai_mask_small
        ),
        cmap="Reds",
        alpha=0.55,
        vmin=0,
        vmax=1
    )

    # Polygon

    for contour in contours:

        contour_display = (
            contour[:, 0, :]
        )

        ax.plot(
            contour_display[:, 0],
            contour_display[:, 1],
            linewidth=2
        )

    # Centroid

    ax.scatter(
        centroid_col,
        centroid_row,
        s=100,
        marker="*",
        edgecolors="black",
        linewidths=1.5
    )

    ax.set_title(
        "(D) Final AI Spill Mask + Geometry",
        fontsize=15,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Range / Pixel"
    )

    ax.set_ylabel(
        "Azimuth / Line"
    )

    # --------------------------------------------------------
    # Overall title
    # --------------------------------------------------------

    fig.suptitle(
        "OceanTraceAI — Real Sentinel-1D Oil Spill Analysis",
        fontsize=20,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Information box
    # --------------------------------------------------------

    info = (
        f"AI predicted area: {area_km2:.2f} km²\n"
        f"Perimeter: {perimeter_km:.2f} km\n"
        f"Centroid: "
        f"{centroid_lat:.6f}°, "
        f"{centroid_lon:.6f}°\n"
        f"Threshold: {THRESHOLD:.2f}\n"
        f"Scene: Gulf of Mexico"
    )

    fig.text(
        0.5,
        0.015,
        info,
        ha="center",
        va="bottom",
        fontsize=11
    )

    plt.tight_layout(
        rect=[
            0,
            0.06,
            1,
            0.95
        ]
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\n✓ Four-panel visualization saved:"
    )

    print(
        OUTPUT_PATH
    )

    print("\n" + "=" * 70)
    print("FOUR-PANEL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()