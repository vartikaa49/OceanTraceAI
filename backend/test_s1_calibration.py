import glob
import xml.etree.ElementTree as ET

import numpy as np
import rasterio


# ---------------------------------------------------------
# Find Sentinel-1 VV files
# ---------------------------------------------------------

measurement = glob.glob(
    r"data/sentinel1/**/*.SAFE/measurement/*vv*.tiff",
    recursive=True
)[0]

calibration = glob.glob(
    r"data/sentinel1/**/*.SAFE/annotation/calibration/calibration-*vv*.xml",
    recursive=True
)[0]

print("Measurement:")
print(measurement)

print("\nCalibration:")
print(calibration)


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

        if tag in [
            "line",
            "pixel",
            "dn",
            "sigmaNought"
        ]:

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


print("\nCalibration vectors:", len(vectors))

print(
    "Pixels per vector:",
    len(vectors[0]["pixel"])
)

print(
    "First calibration line:",
    vectors[0]["line"]
)

print(
    "First DN:",
    vectors[0]["dn"][:5]
)

print(
    "First sigmaNought:",
    vectors[0]["sigmaNought"][:5]
)


# ---------------------------------------------------------
# Read small VV window
# ---------------------------------------------------------

with rasterio.open(measurement) as src:

    window = ((0, 2000), (0, 2000))

    dn_image = src.read(
        1,
        window=window
    ).astype(np.float64)

print("\nRaw window:")
print("Shape:", dn_image.shape)
print("Min:", dn_image.min())
print("Max:", dn_image.max())
print("Mean:", dn_image.mean())


# ---------------------------------------------------------
# Interpolate calibration LUT
# ---------------------------------------------------------

image_height, image_width = dn_image.shape

sigma_lut = np.zeros_like(dn_image)


for row in range(image_height):

    # Find nearest calibration vectors
    lines = np.array(
        [v["line"][0] for v in vectors]
    )

    if row <= lines[0]:
        v1 = v2 = vectors[0]

    elif row >= lines[-1]:
        v1 = v2 = vectors[-1]

    else:

        idx = np.searchsorted(
            lines,
            row
        )

        v1 = vectors[idx - 1]
        v2 = vectors[idx]

    # Interpolate sigmaNought along range
    sigma1 = np.interp(
        np.arange(image_width),
        v1["pixel"],
        v1["sigmaNought"]
    )

    sigma2 = np.interp(
        np.arange(image_width),
        v2["pixel"],
        v2["sigmaNought"]
    )

    # Interpolate between azimuth calibration lines
    if v1 is v2:

        sigma_lut[row] = sigma1

    else:

        weight = (
            row - v1["line"][0]
        ) / (
            v2["line"][0] - v1["line"][0]
        )

        sigma_lut[row] = (
            sigma1
            + weight * (sigma2 - sigma1)
        )


# ---------------------------------------------------------
# Convert DN to Sigma0
# ---------------------------------------------------------

sigma0 = np.zeros_like(dn_image)

valid = dn_image > 0

sigma0[valid] = (
    dn_image[valid] ** 2
    / sigma_lut[valid]
)


# ---------------------------------------------------------
# Convert Sigma0 to dB
# ---------------------------------------------------------

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

print("\n========== CALIBRATION RESULT ==========")

print(
    "Sigma0 linear min:",
    np.min(sigma0[valid])
)

print(
    "Sigma0 linear max:",
    np.max(sigma0[valid])
)

print(
    "Sigma0 dB min:",
    np.min(sigma0_db)
)

print(
    "Sigma0 dB max:",
    np.max(sigma0_db)
)

print(
    "Sigma0 dB mean:",
    np.mean(sigma0_db)
)

print("========================================")