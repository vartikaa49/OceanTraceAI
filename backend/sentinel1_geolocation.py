from pathlib import Path
import xml.etree.ElementTree as ET
import json
import numpy as np


SAFE_DIR = Path(
    "data/sentinel1/"
    "S1D_IW_GRDH_1SDV_20260908T000923_20260908T000948_004479_0084FE_93F3.SAFE"
)

ANNOTATION_DIR = SAFE_DIR / "annotation"

OUTPUT_DIR = Path("data/processed/sentinel1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "sentinel1_geolocation_grid.json"


def find_vv_annotation():
    files = sorted(ANNOTATION_DIR.glob("s1d-iw-grd-vv-*.xml"))

    if not files:
        raise FileNotFoundError(
            "VV Sentinel-1 annotation XML not found."
        )

    return files[0]


def local_name(tag):
    """
    Remove XML namespace from a tag.
    """
    return tag.split("}")[-1]


def extract_geolocation_points(annotation_file):

    print("=" * 70)
    print("SENTINEL-1 GEOLOCATION GRID EXTRACTION")
    print("=" * 70)

    print(f"\nAnnotation:")
    print(annotation_file)

    tree = ET.parse(annotation_file)
    root = tree.getroot()

    points = []

    for element in root.iter():

        if local_name(element.tag) != "geolocationGridPoint":
            continue

        values = {}

        for child in element:

            name = local_name(child.tag)

            if child.text is not None:
                values[name] = child.text.strip()

        required = [
            "line",
            "pixel",
            "latitude",
            "longitude"
        ]

        if all(key in values for key in required):

            points.append(
                {
                    "line": int(values["line"]),
                    "pixel": int(values["pixel"]),
                    "latitude": float(values["latitude"]),
                    "longitude": float(values["longitude"]),
                    "incidenceAngle": float(
                        values.get("incidenceAngle", "nan")
                    ),
                    "elevationAngle": float(
                        values.get("elevationAngle", "nan")
                    ),
                }
            )

    if not points:
        raise RuntimeError(
            "No geolocation points were found."
        )

    print(f"\nGeolocation points found: {len(points)}")

    return points


def organize_grid(points):

    lines = sorted(
        set(point["line"] for point in points)
    )

    pixels = sorted(
        set(point["pixel"] for point in points)
    )

    print(f"Unique lines:  {len(lines)}")
    print(f"Unique pixels: {len(pixels)}")

    print(
        f"Line range:   {min(lines)} -> {max(lines)}"
    )

    print(
        f"Pixel range:  {min(pixels)} -> {max(pixels)}"
    )

    latitude = np.full(
        (len(lines), len(pixels)),
        np.nan,
        dtype=np.float64
    )

    longitude = np.full(
        (len(lines), len(pixels)),
        np.nan,
        dtype=np.float64
    )

    incidence = np.full(
        (len(lines), len(pixels)),
        np.nan,
        dtype=np.float64
    )

    elevation = np.full(
        (len(lines), len(pixels)),
        np.nan,
        dtype=np.float64
    )

    line_index = {
        value: index
        for index, value in enumerate(lines)
    }

    pixel_index = {
        value: index
        for index, value in enumerate(pixels)
    }

    for point in points:

        i = line_index[point["line"]]
        j = pixel_index[point["pixel"]]

        latitude[i, j] = point["latitude"]
        longitude[i, j] = point["longitude"]

        incidence[i, j] = point["incidenceAngle"]
        elevation[i, j] = point["elevationAngle"]

    return (
        lines,
        pixels,
        latitude,
        longitude,
        incidence,
        elevation,
    )


def bilinear_interpolate(
    line,
    pixel,
    lines,
    pixels,
    values
):

    lines = np.asarray(lines, dtype=np.float64)
    pixels = np.asarray(pixels, dtype=np.float64)

    # Find surrounding line indices
    i1 = np.searchsorted(lines, line)

    if i1 == 0:
        i0 = 0
        i1 = 1

    elif i1 >= len(lines):
        i1 = len(lines) - 1
        i0 = i1 - 1

    else:
        i0 = i1 - 1

    # Find surrounding pixel indices
    j1 = np.searchsorted(pixels, pixel)

    if j1 == 0:
        j0 = 0
        j1 = 1

    elif j1 >= len(pixels):
        j1 = len(pixels) - 1
        j0 = j1 - 1

    else:
        j0 = j1 - 1

    # Coordinates
    l0 = lines[i0]
    l1 = lines[i1]

    p0 = pixels[j0]
    p1 = pixels[j1]

    # Avoid division by zero
    if l1 == l0:
        line_weight = 0.0
    else:
        line_weight = (line - l0) / (l1 - l0)

    if p1 == p0:
        pixel_weight = 0.0
    else:
        pixel_weight = (pixel - p0) / (p1 - p0)

    # Four surrounding values
    v00 = values[i0, j0]
    v01 = values[i0, j1]
    v10 = values[i1, j0]
    v11 = values[i1, j1]

    # Bilinear interpolation
    top = (
        v00 * (1 - pixel_weight)
        + v01 * pixel_weight
    )

    bottom = (
        v10 * (1 - pixel_weight)
        + v11 * pixel_weight
    )

    result = (
        top * (1 - line_weight)
        + bottom * line_weight
    )

    return float(result)


def pixel_to_latlon(
    line,
    pixel,
    lines,
    pixels,
    latitude,
    longitude
):

    lat = bilinear_interpolate(
        line,
        pixel,
        lines,
        pixels,
        latitude
    )

    lon = bilinear_interpolate(
        line,
        pixel,
        lines,
        pixels,
        longitude
    )

    return lat, lon


def main():

    annotation_file = find_vv_annotation()

    points = extract_geolocation_points(
        annotation_file
    )

    (
        lines,
        pixels,
        latitude,
        longitude,
        incidence,
        elevation,
    ) = organize_grid(points)

    # ---------------------------------------------------------
    # Geographic coverage
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("GEOGRAPHIC COVERAGE")
    print("=" * 70)

    print(
        f"Latitude:  "
        f"{np.nanmin(latitude):.6f} "
        f"to "
        f"{np.nanmax(latitude):.6f}"
    )

    print(
        f"Longitude: "
        f"{np.nanmin(longitude):.6f} "
        f"to "
        f"{np.nanmax(longitude):.6f}"
    )

    print(
        f"Incidence angle: "
        f"{np.nanmin(incidence):.2f}° "
        f"to "
        f"{np.nanmax(incidence):.2f}°"
    )

    # ---------------------------------------------------------
    # Test points
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("PIXEL → GPS TEST")
    print("=" * 70)

    test_points = [
        (0, 0),
        (0, 1285),
        (0, 5140),
        (1000, 5000),
        (5000, 10000),
        (10000, 15000),
        (16000, 25000),
    ]

    for line, pixel in test_points:

        lat, lon = pixel_to_latlon(
            line,
            pixel,
            lines,
            pixels,
            latitude,
            longitude
        )

        print(
            f"Pixel "
            f"(line={line}, pixel={pixel}) "
            f"→ "
            f"Lat={lat:.6f}, "
            f"Lon={lon:.6f}"
        )

    # ---------------------------------------------------------
    # Save grid
    # ---------------------------------------------------------

    output = {
        "satellite": "Sentinel-1D",
        "annotation_file": str(annotation_file),
        "number_of_points": len(points),
        "lines": lines,
        "pixels": pixels,
        "latitude": latitude.tolist(),
        "longitude": longitude.tolist(),
        "incidence_angle": incidence.tolist(),
        "elevation_angle": elevation.tolist(),
        "coverage": {
            "min_latitude": float(np.nanmin(latitude)),
            "max_latitude": float(np.nanmax(latitude)),
            "min_longitude": float(np.nanmin(longitude)),
            "max_longitude": float(np.nanmax(longitude)),
        }
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("GEOLOCATION GRID SAVED")
    print("=" * 70)

    print(
        f"\nOutput:"
        f"\n{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()