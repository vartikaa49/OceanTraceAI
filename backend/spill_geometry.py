import os

import cv2
import numpy as np
import rasterio
from pyproj import Geod


# ==================================================
# CONFIGURATION
# ==================================================

PREDICTION_PATH = "outputs/01338_prediction.tif"

OUTPUT_DIR = "outputs"

MIN_COMPONENT_AREA = 100


# ==================================================
# LOAD PREDICTION MASK
# ==================================================

print("Loading prediction mask...")

with rasterio.open(PREDICTION_PATH) as src:

    mask = src.read(1)

    transform = src.transform

    crs = src.crs

    width = src.width
    height = src.height


print("Mask shape:", mask.shape)
print("CRS:", crs)


# ==================================================
# FIND CONNECTED COMPONENTS
# ==================================================

binary_mask = (
    mask > 0
).astype(np.uint8)


num_labels, labels, stats, centroids = (
    cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )
)


components = []


for label in range(1, num_labels):

    area_pixels = stats[
        label,
        cv2.CC_STAT_AREA
    ]

    if area_pixels < MIN_COMPONENT_AREA:
        continue

    x = stats[
        label,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        label,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        label,
        cv2.CC_STAT_WIDTH
    ]

    h = stats[
        label,
        cv2.CC_STAT_HEIGHT
    ]

    center_x, center_y = centroids[label]

    components.append({
        "label": label,
        "pixels": int(area_pixels),
        "x": int(x),
        "y": int(y),
        "width": int(w),
        "height": int(h),
        "center_x": center_x,
        "center_y": center_y
    })


# ==================================================
# CHECK COMPONENTS
# ==================================================

print(
    "\nDetected connected regions:",
    len(components)
)


if len(components) == 0:

    print(
        "No significant spill region detected."
    )

    raise SystemExit


# Sort largest first
components.sort(
    key=lambda item: item["pixels"],
    reverse=True
)


largest = components[0]


print("\nLargest spill region:")
print(
    "Pixels:",
    largest["pixels"]
)

print(
    "Bounding box:",
    largest["width"],
    "x",
    largest["height"],
    "pixels"
)


# ==================================================
# PIXEL → GEOGRAPHIC COORDINATES
# ==================================================

center_x = largest["center_x"]
center_y = largest["center_y"]


longitude, latitude = rasterio.transform.xy(
    transform,
    center_y,
    center_x,
    offset="center"
)


print("\n========== SPILL LOCATION ==========")

print(
    "Latitude:",
    latitude
)

print(
    "Longitude:",
    longitude
)


# ==================================================
# BOUNDING BOX COORDINATES
# ==================================================

x_min = largest["x"]
y_min = largest["y"]

x_max = (
    x_min
    + largest["width"]
)

y_max = (
    y_min
    + largest["height"]
)


lon_min, lat_max = rasterio.transform.xy(
    transform,
    y_min,
    x_min,
    offset="center"
)

lon_max, lat_min = rasterio.transform.xy(
    transform,
    y_max - 1,
    x_max - 1,
    offset="center"
)


print("\n========== BOUNDING BOX ==========")

print(
    "Minimum latitude:",
    lat_min
)

print(
    "Maximum latitude:",
    lat_max
)

print(
    "Minimum longitude:",
    lon_min
)

print(
    "Maximum longitude:",
    lon_max
)


# ==================================================
# APPROXIMATE AREA
# ==================================================

# Calculate geographic size of one pixel
# using the latitude of the spill centroid.

geod = Geod(
    ellps="WGS84"
)


lon1, lat1 = rasterio.transform.xy(
    transform,
    center_y,
    center_x,
    offset="center"
)

lon2, lat2 = rasterio.transform.xy(
    transform,
    center_y,
    center_x + 1,
    offset="center"
)

lon3, lat3 = rasterio.transform.xy(
    transform,
    center_y + 1,
    center_x,
    offset="center"
)


_, _, pixel_width = geod.inv(
    lon1,
    lat1,
    lon2,
    lat2
)

_, _, pixel_height = geod.inv(
    lon1,
    lat1,
    lon3,
    lat3
)


pixel_area_m2 = (
    pixel_width
    * pixel_height
)


spill_area_m2 = (
    largest["pixels"]
    * pixel_area_m2
)


spill_area_km2 = (
    spill_area_m2
    / 1_000_000
)


print("\n========== SPILL AREA ==========")

print(
    "Approximate pixel size:",
    f"{pixel_width:.2f} m x {pixel_height:.2f} m"
)

print(
    "Approximate spill area:",
    f"{spill_area_km2:.4f} km²"
)


# ==================================================
# SAVE RESULTS
# ==================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


result_file = os.path.join(
    OUTPUT_DIR,
    "spill_geometry.txt"
)


with open(
    result_file,
    "w"
) as file:

    file.write(
        "OCEANTRACEAI SPILL GEOMETRY\n"
    )

    file.write(
        "============================\n\n"
    )

    file.write(
        f"Spill pixels: {largest['pixels']}\n"
    )

    file.write(
        f"Latitude: {latitude}\n"
    )

    file.write(
        f"Longitude: {longitude}\n"
    )

    file.write(
        f"Area km2: {spill_area_km2:.4f}\n"
    )

    file.write(
        f"Min latitude: {lat_min}\n"
    )

    file.write(
        f"Max latitude: {lat_max}\n"
    )

    file.write(
        f"Min longitude: {lon_min}\n"
    )

    file.write(
        f"Max longitude: {lon_max}\n"
    )


print(
    "\n✓ Geometry saved:"
)

print(
    result_file
)

print(
    "\nSpill geometry extraction complete!"
)