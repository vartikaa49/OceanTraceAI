from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio
from rasterio.windows import Window


# ============================================================
# CONFIGURATION
# ============================================================

SAFE_DIR = Path(
    "data/sentinel1/"
    "S1D_IW_GRDH_1SDV_20260908T000923_20260908T000948_004479_0084FE_93F3.SAFE"
)

MEASUREMENT_DIR = SAFE_DIR / "measurement"
CALIBRATION_DIR = SAFE_DIR / "annotation" / "calibration"

OUTPUT_DIR = Path("data/processed/sentinel1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 512

POLARIZATIONS = ["vv", "vh"]


# ============================================================
# XML HELPERS
# ============================================================

def find_xml(pattern):
    files = sorted(CALIBRATION_DIR.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No XML found for pattern: {pattern}"
        )

    return files[0]


def parse_numbers(element):
    if element is None or element.text is None:
        return np.array([], dtype=np.float64)

    return np.asarray(
        [float(x) for x in element.text.split()],
        dtype=np.float64
    )


# ============================================================
# CALIBRATION LUT
# ============================================================

def load_calibration(calibration_xml):
    print(f"Reading calibration XML:")
    print(f"  {calibration_xml}")

    root = ET.parse(calibration_xml).getroot()

    vectors = root.findall(".//calibrationVector")

    if not vectors:
        raise RuntimeError(
            "No calibrationVector elements found."
        )

    lines = []
    pixels = []
    sigma_luts = []
    beta_luts = []
    gamma_luts = []
    dn_luts = []

    for vector in vectors:

        line = int(vector.findtext("line"))

        pixel = parse_numbers(
            vector.find("pixel")
        )

        sigma = parse_numbers(
            vector.find("sigmaNought")
        )

        beta = parse_numbers(
            vector.find("betaNought")
        )

        gamma = parse_numbers(
            vector.find("gamma")
        )

        dn = parse_numbers(
            vector.find("dn")
        )

        lines.append(line)
        pixels.append(pixel)
        sigma_luts.append(sigma)
        beta_luts.append(beta)
        gamma_luts.append(gamma)
        dn_luts.append(dn)

    print(f"  Calibration vectors: {len(lines)}")
    print(f"  Calibration pixels: {len(pixels[0])}")

    return {
        "lines": np.asarray(lines, dtype=np.float64),
        "pixels": np.asarray(pixels[0], dtype=np.float64),
        "sigma": np.asarray(sigma_luts, dtype=np.float64),
        "beta": np.asarray(beta_luts, dtype=np.float64),
        "gamma": np.asarray(gamma_luts, dtype=np.float64),
        "dn": np.asarray(dn_luts, dtype=np.float64),
    }


# ============================================================
# NOISE LUT
# ============================================================

def load_noise(noise_xml):
    print(f"Reading noise XML:")
    print(f"  {noise_xml}")

    root = ET.parse(noise_xml).getroot()

    # --------------------------------------------------------
    # RANGE NOISE
    # --------------------------------------------------------

    range_vectors = root.findall(
        ".//noiseRangeVector"
    )

    if not range_vectors:
        raise RuntimeError(
            "No noiseRangeVector elements found."
        )

    range_lines = []
    range_pixels = []
    range_luts = []

    for vector in range_vectors:

        line = int(vector.findtext("line"))

        pixel = parse_numbers(
            vector.find("pixel")
        )

        lut = parse_numbers(
            vector.find("noiseRangeLut")
        )

        range_lines.append(line)
        range_pixels.append(pixel)
        range_luts.append(lut)

    range_lines = np.asarray(
        range_lines,
        dtype=np.float64
    )

    range_pixels = np.asarray(
        range_pixels[0],
        dtype=np.float64
    )

    range_luts = np.asarray(
        range_luts,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # AZIMUTH NOISE
    # --------------------------------------------------------

    azimuth_vectors = root.findall(
        ".//noiseAzimuthVector"
    )

    azimuth_blocks = []

    for vector in azimuth_vectors:

        swath = vector.findtext("swath")

        first_line = vector.findtext(
            "firstAzimuthLine"
        )

        first_pixel = vector.findtext(
            "firstRangeSample"
        )

        last_line = vector.findtext(
            "lastAzimuthLine"
        )

        last_pixel = vector.findtext(
            "lastRangeSample"
        )

        line_element = vector.find("line")
        lut_element = vector.find("noiseAzimuthLut")

        if (
            first_line is None
            or first_pixel is None
            or last_line is None
            or last_pixel is None
            or line_element is None
            or lut_element is None
        ):
            continue

        lines = parse_numbers(line_element)
        lut = parse_numbers(lut_element)

        azimuth_blocks.append(
            {
                "swath": swath,
                "first_line": int(first_line),
                "first_pixel": int(first_pixel),
                "last_line": int(last_line),
                "last_pixel": int(last_pixel),
                "lines": lines,
                "lut": lut,
            }
        )

    print(
        f"  Range-noise vectors: "
        f"{len(range_lines)}"
    )

    print(
        f"  Azimuth-noise blocks: "
        f"{len(azimuth_blocks)}"
    )

    return {
        "range_lines": range_lines,
        "range_pixels": range_pixels,
        "range_luts": range_luts,
        "azimuth_blocks": azimuth_blocks,
    }


# ============================================================
# 2-D LUT INTERPOLATION
# ============================================================

def interpolate_range_noise(
    row,
    col_start,
    col_end,
    noise_data
):
    """
    Interpolate range-noise LUT for one image row.

    First interpolate along range (pixel).
    Then interpolate between the two surrounding
    azimuth/range vectors.
    """

    lines = noise_data["range_lines"]
    pixels = noise_data["range_pixels"]
    luts = noise_data["range_luts"]

    # --------------------------------------------------------
    # Find surrounding calibration/noise lines
    # --------------------------------------------------------

    if row <= lines[0]:

        upper = 0
        lower = 0

    elif row >= lines[-1]:

        upper = len(lines) - 1
        lower = upper

    else:

        upper = np.searchsorted(
            lines,
            row
        )

        lower = upper - 1

    line1 = lines[lower]
    line2 = lines[upper]

    # --------------------------------------------------------
    # Interpolate each noise vector in range direction
    # --------------------------------------------------------

    cols = np.arange(
        col_start,
        col_end,
        dtype=np.float64
    )

    noise1 = np.interp(
        cols,
        pixels,
        luts[lower]
    )

    noise2 = np.interp(
        cols,
        pixels,
        luts[upper]
    )

    # --------------------------------------------------------
    # Interpolate between azimuth lines
    # --------------------------------------------------------

    if upper == lower:

        return noise1.astype(
            np.float32
        )

    weight = (
        row - line1
    ) / (
        line2 - line1
    )

    noise = (
        noise1
        + weight * (noise2 - noise1)
    )

    return noise.astype(
        np.float32
    )


# ============================================================
# AZIMUTH NOISE INTERPOLATION
# ============================================================

def interpolate_azimuth_noise(
    row,
    col_start,
    col_end,
    noise_data
):
    """
    Interpolate azimuth-noise scaling factor.

    Modern Sentinel-1 noise annotations contain
    azimuth LUTs that scale the range-noise profile.
    """

    width = col_end - col_start

    result = np.ones(
        width,
        dtype=np.float32
    )

    for block in noise_data["azimuth_blocks"]:

        # Does this row belong to this block?
        if row < block["first_line"]:
            continue

        if row > block["last_line"]:
            continue

        # Does this block overlap our range?
        if col_end - 1 < block["first_pixel"]:
            continue

        if col_start > block["last_pixel"]:
            continue

        lines = block["lines"]
        lut = block["lut"]

        if len(lines) == 0:
            continue

        value = np.interp(
            row,
            lines,
            lut
        )

        start = max(
            col_start,
            block["first_pixel"]
        )

        end = min(
            col_end,
            block["last_pixel"] + 1
        )

        if end <= start:
            continue

        result[
            start - col_start:
            end - col_start
        ] = value

    return result


# ============================================================
# CALIBRATION INTERPOLATION
# ============================================================

def interpolate_calibration(
    row,
    col_start,
    col_end,
    calibration
):
    """
    Interpolate sigmaNought calibration LUT
    for a single image row.
    """

    lines = calibration["lines"]
    pixels = calibration["pixels"]
    lut = calibration["sigma"]

    cols = np.arange(
        col_start,
        col_end,
        dtype=np.float64
    )

    # --------------------------------------------------------
    # Find surrounding calibration lines
    # --------------------------------------------------------

    if row <= lines[0]:

        upper = 0
        lower = 0

    elif row >= lines[-1]:

        upper = len(lines) - 1
        lower = upper

    else:

        upper = np.searchsorted(
            lines,
            row
        )

        lower = upper - 1

    # --------------------------------------------------------
    # Range interpolation
    # --------------------------------------------------------

    cal1 = np.interp(
        cols,
        pixels,
        lut[lower]
    )

    cal2 = np.interp(
        cols,
        pixels,
        lut[upper]
    )

    # --------------------------------------------------------
    # Azimuth interpolation
    # --------------------------------------------------------

    if upper == lower:

        return cal1.astype(
            np.float32
        )

    weight = (
        row - lines[lower]
    ) / (
        lines[upper] - lines[lower]
    )

    result = (
        cal1
        + weight * (cal2 - cal1)
    )

    return result.astype(
        np.float32
    )


# ============================================================
# PROCESS ONE POLARIZATION
# ============================================================

def process_polarization(pol):
    print()
    print("=" * 70)
    print(f"PROCESSING {pol.upper()}")
    print("=" * 70)

    # --------------------------------------------------------
    # Locate files
    # --------------------------------------------------------

    measurement_files = sorted(
        MEASUREMENT_DIR.glob(
            f"*grd-{pol}*.tiff"
        )
    )

    if not measurement_files:

        measurement_files = sorted(
            MEASUREMENT_DIR.glob(
                f"*grd-{pol}*.tif"
            )
        )

    if not measurement_files:

        raise FileNotFoundError(
            f"Measurement TIFF not found for {pol}"
        )

    measurement_path = measurement_files[0]

    calibration_xml = find_xml(
        f"calibration-s1d-iw-grd-{pol}-*.xml"
    )

    noise_xml = find_xml(
        f"noise-s1d-iw-grd-{pol}-*.xml"
    )

    output_path = (
        OUTPUT_DIR
        / f"sentinel1_{pol.upper()}_sigma0_db.tif"
    )

    print("Measurement:")
    print(f"  {measurement_path}")

    print("Calibration:")
    print(f"  {calibration_xml}")

    print("Noise:")
    print(f"  {noise_xml}")

    # --------------------------------------------------------
    # Load LUTs
    # --------------------------------------------------------

    calibration = load_calibration(
        calibration_xml
    )

    noise = load_noise(
        noise_xml
    )

    # --------------------------------------------------------
    # Open measurement
    # --------------------------------------------------------

    with rasterio.open(
        measurement_path
    ) as src:

        height = src.height
        width = src.width

        profile = src.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES",
            nodata=-9999.0,
        )

        print()
        print(
            f"Input shape: "
            f"{height} x {width}"
        )

        with rasterio.open(
            output_path,
            "w",
            **profile
        ) as dst:

            total_valid = 0
            total_pixels = 0

            # ------------------------------------------------
            # Process rows in blocks
            # ------------------------------------------------

            for row_start in range(
                0,
                height,
                BLOCK_SIZE
            ):

                row_end = min(
                    row_start + BLOCK_SIZE,
                    height
                )

                block_height = (
                    row_end - row_start
                )

                block = src.read(
                    1,
                    window=Window(
                        0,
                        row_start,
                        width,
                        block_height
                    )
                ).astype(
                    np.float32
                )

                output = np.full(
                    (
                        block_height,
                        width
                    ),
                    -9999.0,
                    dtype=np.float32
                )

                for local_row in range(
                    block_height
                ):

                    global_row = (
                        row_start
                        + local_row
                    )

                    dn = block[
                        local_row
                    ]

                    # ----------------------------------------
                    # Valid measurement pixels
                    # ----------------------------------------

                    valid = (
                        np.isfinite(dn)
                        & (dn > 0)
                    )

                    if not np.any(valid):
                        continue

                    # ----------------------------------------
                    # Calibration LUT
                    # ----------------------------------------

                    sigma_cal = (
                        interpolate_calibration(
                            global_row,
                            0,
                            width,
                            calibration
                        )
                    )

                    # ----------------------------------------
                    # Thermal noise LUT
                    # ----------------------------------------

                    range_noise = (
                        interpolate_range_noise(
                            global_row,
                            0,
                            width,
                            noise
                        )
                    )

                    azimuth_noise = (
                        interpolate_azimuth_noise(
                            global_row,
                            0,
                            width,
                            noise
                        )
                    )

                    # Full noise contribution:
                    #
                    # eta =
                    # noiseRangeLut *
                    # noiseAzimuthLut
                    #
                    eta = (
                        range_noise
                        * azimuth_noise
                    )

                    # ----------------------------------------
                    # Sentinel-1 calibration + noise removal
                    #
                    # sigma0 =
                    # (DN² - eta) / sigmaNought²
                    # ----------------------------------------

                    dn_squared = (
                        dn.astype(np.float32)
                        ** 2
                    )

                    numerator = (
                        dn_squared
                        - eta
                    )

                    numerator = np.maximum(
                        numerator,
                        0.0
                    )

                    denominator = (
                        sigma_cal
                        ** 2
                    )

                    sigma0 = (
                        numerator
                        / denominator
                    )

                    # ----------------------------------------
                    # Convert linear power → dB
                    # ----------------------------------------

                    valid_sigma = (
                        valid
                        & np.isfinite(sigma0)
                        & (sigma0 > 0)
                    )

                    db = np.full(
                        width,
                        -9999.0,
                        dtype=np.float32
                    )

                    db[valid_sigma] = (
                        10.0
                        * np.log10(
                            sigma0[
                                valid_sigma
                            ]
                        )
                    )

                    # ----------------------------------------
                    # Basic sanity range
                    #
                    # Sentinel-1 extreme numerical values
                    # are not useful for the ML pipeline.
                    # ----------------------------------------

                    finite_db = (
                        np.isfinite(db)
                        & (db > -60.0)
                        & (db < 60.0)
                    )

                    db[
                        ~finite_db
                    ] = -9999.0

                    output[
                        local_row
                    ] = db

                    total_valid += int(
                        np.sum(finite_db)
                    )

                    total_pixels += width

                dst.write(
                    output,
                    1,
                    window=Window(
                        0,
                        row_start,
                        width,
                        block_height
                    )
                )

                print(
                    f"  Rows "
                    f"{row_start:5d} - "
                    f"{row_end - 1:5d}"
                )

    print()
    print(f"✓ Output saved:")
    print(f"  {output_path}")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    with rasterio.open(
        output_path
    ) as src:

        values = src.read(
            1,
            masked=True
        )

        values = values.compressed()

    if len(values) > 0:

        print()
        print("Statistics:")
        print(
            f"  Valid pixels: "
            f"{len(values):,}"
        )

        print(
            f"  Min: "
            f"{values.min():.4f} dB"
        )

        print(
            f"  Max: "
            f"{values.max():.4f} dB"
        )

        print(
            f"  Mean: "
            f"{values.mean():.4f} dB"
        )

        print(
            f"  Median: "
            f"{np.median(values):.4f} dB"
        )

        print(
            f"  P5: "
            f"{np.percentile(values, 5):.4f} dB"
        )

        print(
            f"  P95: "
            f"{np.percentile(values, 95):.4f} dB"
        )

    return output_path


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 70)
    print("OCEANTRACEAI — SENTINEL-1 CALIBRATION + THERMAL NOISE REMOVAL")
    print("=" * 70)

    print()
    print(f"SAFE directory:")
    print(f"  {SAFE_DIR}")

    print()
    print(
        "Processing both VV and VH..."
    )

    outputs = []

    for polarization in POLARIZATIONS:

        output = process_polarization(
            polarization
        )

        outputs.append(output)

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    for output in outputs:
        print(f"✓ {output}")