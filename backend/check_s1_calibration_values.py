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
# Show first calibration vector
# ---------------------------------------------------------

v = vectors[0]

print("========== FIRST CALIBRATION VECTOR ==========")

print("Line:", v["line"][0])

print("\nPixel:")
print(v["pixel"][:10])

print("\nDN:")
print(v["dn"][:10])

print("\nSigmaNought:")
print(v["sigmaNought"][:10])


# ---------------------------------------------------------
# Read first 1000 pixels of first image line
# ---------------------------------------------------------

with rasterio.open(measurement) as src:

    image = src.read(
        1,
        window=((0, 1), (0, 1000))
    ).astype(np.float64)

dn = image[0]


# ---------------------------------------------------------
# Interpolate SigmaNought
# ---------------------------------------------------------

sigma_lut = np.interp(
    np.arange(1000),
    v["pixel"],
    v["sigmaNought"]
)


# ---------------------------------------------------------
# Calculate Sigma0
# ---------------------------------------------------------

valid = dn > 0

sigma0 = np.zeros_like(dn)

sigma0[valid] = (
    dn[valid] ** 2
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
# Print sample values
# ---------------------------------------------------------

print("\n========== SAMPLE CALIBRATION ==========")

print("First 20 raw DN:")
print(dn[:20])

print("\nFirst 20 SigmaNought:")
print(sigma_lut[:20])

print("\nFirst 20 Sigma0 dB:")
print(sigma0_db[:20])

print("\nValid pixels:", np.sum(valid))

print("Sigma0 dB minimum:", sigma0_db[valid_db].min())

print("Sigma0 dB maximum:", sigma0_db[valid_db].max())

print("Sigma0 dB mean:", sigma0_db[valid_db].mean())

print("\n========================================")