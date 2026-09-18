import glob
import xml.etree.ElementTree as ET

import numpy as np
import rasterio


# ---------------------------------------------------------
# Find files
# ---------------------------------------------------------

measurement = glob.glob(
    r"data/sentinel1/**/*.SAFE/measurement/*vv*.tiff",
    recursive=True
)[0]

calibration = glob.glob(
    r"data/sentinel1/**/*.SAFE/annotation/calibration/calibration-*vv*.xml",
    recursive=True
)[0]


# ---------------------------------------------------------
# Read calibration XML
# ---------------------------------------------------------

root = ET.parse(calibration).getroot()

vectors = []

for element in root.iter():

    if element.tag.split("}")[-1] != "calibrationVector":
        continue

    data = {}

    for child in element:

        tag = child.tag.split("}")[-1]

        if tag in ["line", "pixel", "dn", "sigmaNought"]:

            if child.text:

                data[tag] = np.array(
                    [float(x) for x in child.text.split()],
                    dtype=np.float64
                )

    if all(
        key in data
        for key in ["line", "pixel", "dn", "sigmaNought"]
    ):
        vectors.append(data)


# ---------------------------------------------------------
# Read middle image window
# ---------------------------------------------------------

with rasterio.open(measurement) as src:

    height, width = src.shape

    row_start = height // 2
    col_start = width // 2

    window = (
        (row_start, row_start + 1000),
        (col_start, col_start + 1000)
    )

    dn_image = src.read(
        1,
        window=window
    ).astype(np.float64)


print("========== RAW IMAGE WINDOW ==========")
print("Image size:", height, "x", width)
print("Window start:", row_start, col_start)
print("Window shape:", dn_image.shape)

print("DN min:", dn_image.min())
print("DN max:", dn_image.max())
print("DN mean:", dn_image.mean())


# ---------------------------------------------------------
# Calibration
# ---------------------------------------------------------

image_height, image_width = dn_image.shape

sigma_lut = np.zeros_like(dn_image)

lines = np.array(
    [v["line"][0] for v in vectors]
)


for row_index in range(image_height):

    actual_line = row_start + row_index

    if actual_line <= lines[0]:

        v1 = v2 = vectors[0]

    elif actual_line >= lines[-1]:

        v1 = v2 = vectors[-1]

    else:

        idx = np.searchsorted(
            lines,
            actual_line
        )

        v1 = vectors[idx - 1]
        v2 = vectors[idx]

    sigma1 = np.interp(
        np.arange(
            col_start,
            col_start + image_width
        ),
        v1["pixel"],
        v1["sigmaNought"]
    )

    sigma2 = np.interp(
        np.arange(
            col_start,
            col_start + image_width
        ),
        v2["pixel"],
        v2["sigmaNought"]
    )

    if v1 is v2:

        sigma_lut[row_index] = sigma1

    else:

        weight = (
            actual_line - v1["line"][0]
        ) / (
            v2["line"][0] - v1["line"][0]
        )

        sigma_lut[row_index] = (
            sigma1
            + weight * (sigma2 - sigma1)
        )


# ---------------------------------------------------------
# Convert DN to Sigma0
# ---------------------------------------------------------

valid = dn_image > 0

sigma0 = np.zeros_like(dn_image)

sigma0[valid] = (
    dn_image[valid] ** 2
    / sigma_lut[valid]
)


sigma0_db = np.full_like(
    sigma0,
    -50.0
)

valid_db = sigma0 > 0

sigma0_db[valid_db] = (
    10 * np.log10(
        sigma0[valid_db]
    )
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

print("\n========== CALIBRATED RESULT ==========")

print(
    "Valid pixels:",
    np.sum(valid)
)

print(
    "Valid percentage:",
    100 * np.sum(valid) / valid.size,
    "%"
)

print(
    "Sigma0 dB min:",
    sigma0_db[valid_db].min()
)

print(
    "Sigma0 dB max:",
    sigma0_db[valid_db].max()
)

print(
    "Sigma0 dB mean:",
    sigma0_db[valid_db].mean()
)

print("========================================")