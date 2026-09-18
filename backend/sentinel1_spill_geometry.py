from pathlib import Path
import json

import cv2
import numpy as np
import rasterio
from pyproj import Geod


# ============================================================
# CONFIG
# ============================================================

PREDICTION_PATH = Path(
    "outputs/sentinel1_real/real_sentinel1_prediction.tif"
)

VV_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VV_preprocessed.tif"
)

VH_PATH = Path(
    "data/processed/sentinel1/preprocessed/"
    "sentinel1_VH_preprocessed.tif"
)

GEOLOCATION_PATH = Path(
    "data/processed/sentinel1/sentinel1_geolocation_grid.json"
)

OUTPUT_DIR = Path(
    "outputs/sentinel1_real"
)

GEOJSON_OUTPUT = (
    OUTPUT_DIR / "real_sentinel1_spill.geojson"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "real_sentinel1_spill_summary.csv"
)

THRESHOLD = 0.50

# Reduce resolution before connected-component analysis.
DOWNSAMPLE = 4

# Remove tiny detections.
MIN_COMPONENT_PIXELS = 500

# Conservative water-like SAR filtering.
# Oil slicks are generally dark in SAR.
VV_WATER_MAX = 2.0
VH_WATER_MAX = 1.5


# ============================================================
# GEOLOCATION GRID
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
# BILINEAR GEOLOCATION
# ============================================================

def pixel_to_latlon(
    row,
    col,
    lines,
    pixels,
    latitude,
    longitude
):

    row = float(
        np.clip(
            row,
            lines[0],
            lines[-1]
        )
    )

    col = float(
        np.clip(
            col,
            pixels[0],
            pixels[-1]
        )
    )

    r = np.searchsorted(
        lines,
        row
    )

    c = np.searchsorted(
        pixels,
        col
    )

    r = np.clip(
        r,
        1,
        len(lines) - 1
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
# GEOGRAPHIC AREA
# ============================================================

def calculate_polygon_area(
    polygon
):

    geod = Geod(
        ellps="WGS84"
    )

    lons = polygon[:, 0]
    lats = polygon[:, 1]

    area, perimeter = geod.polygon_area_perimeter(
        lons,
        lats
    )

    return (
        abs(area) / 1_000_000.0,
        perimeter / 1000.0
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCEANTRACEAI")
    print("REAL SENTINEL-1 SPILL GEOMETRY")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD PREDICTION
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING AI PREDICTION")
    print("=" * 70)

    with rasterio.open(
        PREDICTION_PATH
    ) as src:

        prediction = src.read(
            1
        ).astype(
            np.float32
        )

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Probability range:",
        f"{prediction.min():.4f}",
        "->",
        f"{prediction.max():.4f}"
    )

    # ========================================================
    # LOAD VV
    # ========================================================

    print("\nLoading VV...")

    with rasterio.open(
        VV_PATH
    ) as src:

        vv = src.read(
            1
        ).astype(
            np.float32
        )

    print(
        "VV shape:",
        vv.shape
    )

    # ========================================================
    # LOAD VH
    # ========================================================

    print("\nLoading VH...")

    with rasterio.open(
        VH_PATH
    ) as src:

        vh = src.read(
            1
        ).astype(
            np.float32
        )

    print(
        "VH shape:",
        vh.shape
    )

    # ========================================================
    # GEOLOCATION
    # ========================================================

    print("\n" + "=" * 70)
    print("LOADING GEOLOCATION GRID")
    print("=" * 70)

    (
        lines,
        pixels,
        latitude,
        longitude
    ) = load_geolocation_grid()

    print(
        "Grid:",
        latitude.shape
    )

    print(
        "Latitude:",
        f"{latitude.min():.6f}",
        "->",
        f"{latitude.max():.6f}"
    )

    print(
        "Longitude:",
        f"{longitude.min():.6f}",
        "->",
        f"{longitude.max():.6f}"
    )

    # ========================================================
    # AI MASK
    # ========================================================

    ai_mask = (
        prediction >= THRESHOLD
    ).astype(
        np.uint8
    )

    original_pixels = int(
        ai_mask.sum()
    )

    original_percentage = (
        original_pixels
        / ai_mask.size
        * 100.0
    )

    print("\n" + "=" * 70)
    print("AI MASK")
    print("=" * 70)

    print(
        "AI predicted pixels:",
        f"{original_pixels:,}"
    )

    print(
        "AI coverage:",
        f"{original_percentage:.4f}%"
    )

    # ========================================================
    # WATER-LIKE FILTER
    # ========================================================
    #
    # The model can confuse bright land/coastal structures
    # with oil. We therefore keep only AI-positive pixels
    # whose SAR backscatter is compatible with dark water.
    #
    # This is a heuristic water constraint, NOT a certified
    # land/water classification product.
    # ========================================================

    valid_vv = np.isfinite(vv)
    valid_vh = np.isfinite(vh)

    water_like = (
        valid_vv
        & valid_vh
        & (vv <= VV_WATER_MAX)
        & (vh <= VH_WATER_MAX)
    )

    filtered_mask = (
        ai_mask
        & water_like.astype(np.uint8)
    )

    filtered_pixels = int(
        filtered_mask.sum()
    )

    filtered_percentage = (
        filtered_pixels
        / filtered_mask.size
        * 100.0
    )

    print("\n" + "=" * 70)
    print("WATER-CONSTRAINED AI MASK")
    print("=" * 70)

    print(
        "Water-like pixels:",
        f"{water_like.sum():,}"
    )

    print(
        "AI pixels after water constraint:",
        f"{filtered_pixels:,}"
    )

    print(
        "Filtered coverage:",
        f"{filtered_percentage:.4f}%"
    )

    # ========================================================
    # DOWNSAMPLE
    # ========================================================

    small_mask = cv2.resize(
        filtered_mask,
        (
            filtered_mask.shape[1] // DOWNSAMPLE,
            filtered_mask.shape[0] // DOWNSAMPLE
        ),
        interpolation=cv2.INTER_NEAREST
    )

    # ========================================================
    # MORPHOLOGY
    # ========================================================

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    small_mask = cv2.morphologyEx(
        small_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    small_mask = cv2.morphologyEx(
        small_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # ========================================================
    # CONNECTED COMPONENTS
    # ========================================================

    print("\n" + "=" * 70)
    print("EXTRACTING CONNECTED COMPONENTS")
    print("=" * 70)

    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            small_mask,
            connectivity=8
        )
    )

    components = []

    for label in range(
        1,
        number_labels
    ):

        area = int(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < MIN_COMPONENT_PIXELS:

            continue

        components.append(
            (
                label,
                area
            )
        )

    components.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(
        "Components retained:",
        len(components)
    )

    if not components:

        raise RuntimeError(
            "No sufficiently large water-constrained "
            "AI components were found."
        )

    # ========================================================
    # BUILD OUTPUT GEOJSON
    # ========================================================

    features = []

    summary_rows = []

    for rank, (
        label,
        component_area
    ) in enumerate(
        components,
        start=1
    ):

        component_mask = (
            labels == label
        ).astype(
            np.uint8
        )

        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:

            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        contour = contour[:, 0, :]

        geographic_polygon = []

        for x, y in contour:

            image_col = (
                float(x)
                * DOWNSAMPLE
            )

            image_row = (
                float(y)
                * DOWNSAMPLE
            )

            lat, lon = pixel_to_latlon(
                image_row,
                image_col,
                lines,
                pixels,
                latitude,
                longitude
            )

            geographic_polygon.append(
                [
                    lon,
                    lat
                ]
            )

        geographic_polygon = np.asarray(
            geographic_polygon,
            dtype=np.float64
        )

        if len(
            geographic_polygon
        ) < 3:

            continue

        # Close polygon.
        if not np.allclose(
            geographic_polygon[0],
            geographic_polygon[-1]
        ):

            geographic_polygon = np.vstack(
                [
                    geographic_polygon,
                    geographic_polygon[0]
                ]
            )

        area_km2, perimeter_km = (
            calculate_polygon_area(
                geographic_polygon
            )
        )

        # Geographic centroid from component pixel centroid.
        component_yx = np.where(
            component_mask > 0
        )

        mean_y = float(
            component_yx[0].mean()
            * DOWNSAMPLE
        )

        mean_x = float(
            component_yx[1].mean()
            * DOWNSAMPLE
        )

        centroid_lat, centroid_lon = (
            pixel_to_latlon(
                mean_y,
                mean_x,
                lines,
                pixels,
                latitude,
                longitude
            )
        )

        print(
            f"\nComponent #{rank}"
        )

        print(
            "Component pixels:",
            f"{component_area:,}"
        )

        print(
            "Area:",
            f"{area_km2:.4f} km²"
        )

        print(
            "Perimeter:",
            f"{perimeter_km:.4f} km"
        )

        print(
            "Centroid:",
            f"{centroid_lat:.6f}, "
            f"{centroid_lon:.6f}"
        )

        feature = {
            "type": "Feature",
            "properties": {
                "rank": rank,
                "component_pixels": component_area,
                "area_km2": round(
                    area_km2,
                    6
                ),
                "perimeter_km": round(
                    perimeter_km,
                    6
                ),
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon,
                "source": "OceanTraceAI U-Net",
                "water_constraint": True,
                "vv_water_max_db": VV_WATER_MAX,
                "vh_water_max_db": VH_WATER_MAX
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    geographic_polygon.tolist()
                ]
            }
        }

        features.append(
            feature
        )

        summary_rows.append(
            {
                "rank": rank,
                "component_pixels": component_area,
                "area_km2": area_km2,
                "perimeter_km": perimeter_km,
                "centroid_latitude": centroid_lat,
                "centroid_longitude": centroid_lon
            }
        )

    # ========================================================
    # SAVE GEOJSON
    # ========================================================

    output_geojson = {
        "type": "FeatureCollection",
        "name": "OceanTraceAI_Real_Sentinel1_Spill",
        "scene": "Gulf of Mexico",
        "features": features
    }

    with open(
        GEOJSON_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output_geojson,
            f,
            indent=2
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "rank,component_pixels,area_km2,"
            "perimeter_km,centroid_latitude,"
            "centroid_longitude\n"
        )

        for row in summary_rows:

            f.write(
                f"{row['rank']},"
                f"{row['component_pixels']},"
                f"{row['area_km2']:.6f},"
                f"{row['perimeter_km']:.6f},"
                f"{row['centroid_latitude']:.8f},"
                f"{row['centroid_longitude']:.8f}\n"
            )

    print("\n" + "=" * 70)
    print("GEOSPATIAL ANALYSIS COMPLETE")
    print("=" * 70)

    print(
        "\n✓ GeoJSON saved:"
    )

    print(
        GEOJSON_OUTPUT
    )

    print(
        "\n✓ Summary saved:"
    )

    print(
        SUMMARY_OUTPUT
    )


if __name__ == "__main__":
    main()