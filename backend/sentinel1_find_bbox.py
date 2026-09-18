from pathlib import Path

import numpy as np
import rasterio


INPUT_DIR = Path("data/processed/sentinel1")

VV_FILE = INPUT_DIR / "sentinel1_VV_sigma0_db.tif"
VH_FILE = INPUT_DIR / "sentinel1_VH_sigma0_db.tif"

NODATA = -9999.0


def find_bbox(path):

    print(f"\nScanning: {path.name}")

    min_row = None
    max_row = None
    min_col = None
    max_col = None

    with rasterio.open(path) as src:

        for row in range(0, src.height, 512):

            row_end = min(row + 512, src.height)

            data = src.read(
                1,
                window=((row, row_end), (0, src.width)),
                out_dtype="float32"
            )

            valid = (
                (data != NODATA)
                & np.isfinite(data)
            )

            if not np.any(valid):
                continue

            rows, cols = np.where(valid)

            current_min_row = row + int(rows.min())
            current_max_row = row + int(rows.max())

            current_min_col = int(cols.min())
            current_max_col = int(cols.max())

            if min_row is None:
                min_row = current_min_row
                max_row = current_max_row
                min_col = current_min_col
                max_col = current_max_col
            else:
                min_row = min(min_row, current_min_row)
                max_row = max(max_row, current_max_row)
                min_col = min(min_col, current_min_col)
                max_col = max(max_col, current_max_col)

    return (
        min_row,
        max_row,
        min_col,
        max_col
    )


def main():

    print("=" * 70)
    print("SENTINEL-1 VALID SAR FOOTPRINT")
    print("=" * 70)

    vv_bbox = find_bbox(VV_FILE)

    print("\nVV valid bounding box:")
    print(f"Rows:    {vv_bbox[0]} -> {vv_bbox[1]}")
    print(f"Columns: {vv_bbox[2]} -> {vv_bbox[3]}")

    vh_bbox = find_bbox(VH_FILE)

    print("\nVH valid bounding box:")
    print(f"Rows:    {vh_bbox[0]} -> {vh_bbox[1]}")
    print(f"Columns: {vh_bbox[2]} -> {vh_bbox[3]}")

    # Combined bounding box
    min_row = min(vv_bbox[0], vh_bbox[0])
    max_row = max(vv_bbox[1], vh_bbox[1])
    min_col = min(vv_bbox[2], vh_bbox[2])
    max_col = max(vv_bbox[3], vh_bbox[3])

    print("\n" + "=" * 70)
    print("COMBINED VV + VH FOOTPRINT")
    print("=" * 70)

    print(f"Rows:    {min_row} -> {max_row}")
    print(f"Columns: {min_col} -> {max_col}")

    height = max_row - min_row + 1
    width = max_col - min_col + 1

    print(f"\nFootprint height: {height}")
    print(f"Footprint width:  {width}")
    print(
        f"Footprint pixels: "
        f"{height * width:,}"
    )

    # Save for next stage
    output_file = INPUT_DIR / "sentinel1_valid_bbox.txt"

    with open(output_file, "w") as f:

        f.write(f"min_row={min_row}\n")
        f.write(f"max_row={max_row}\n")
        f.write(f"min_col={min_col}\n")
        f.write(f"max_col={max_col}\n")
        f.write(f"height={height}\n")
        f.write(f"width={width}\n")

    print(
        f"\n✓ Saved:"
        f"\n{output_file}"
    )

    print("\n" + "=" * 70)
    print("FOOTPRINT ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()