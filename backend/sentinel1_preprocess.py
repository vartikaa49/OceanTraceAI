from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window


INPUT_DIR = Path("data/processed/sentinel1")
OUTPUT_DIR = Path("data/processed/sentinel1/preprocessed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VV_INPUT = INPUT_DIR / "sentinel1_VV_sigma0_db.tif"
VH_INPUT = INPUT_DIR / "sentinel1_VH_sigma0_db.tif"

VV_OUTPUT = OUTPUT_DIR / "sentinel1_VV_preprocessed.tif"
VH_OUTPUT = OUTPUT_DIR / "sentinel1_VH_preprocessed.tif"

NODATA = -9999.0

# Keep the processing memory-safe.
TILE_SIZE = 1024

# Light speckle reduction.
# We use a small Gaussian filter rather than aggressively smoothing
# the SAR image and potentially destroying small spill structures.
GAUSSIAN_KERNEL = 3
GAUSSIAN_SIGMA = 0.8


def process_tile(data):
    """
    Clean and lightly smooth one SAR tile.

    Input:
        float32 Sigma0 dB tile

    Output:
        float32 preprocessed tile
    """

    data = data.astype(np.float32)

    valid = (
        (data != NODATA)
        & np.isfinite(data)
    )

    # Replace NoData temporarily so OpenCV can process the tile.
    clean = data.copy()
    clean[~valid] = 0.0

    # Light smoothing for speckle reduction.
    filtered = cv2.GaussianBlur(
        clean,
        (GAUSSIAN_KERNEL, GAUSSIAN_KERNEL),
        GAUSSIAN_SIGMA
    )

    # Restore NoData.
    filtered[~valid] = NODATA

    return filtered.astype(np.float32)


def preprocess_file(input_file, output_file):

    print("\n" + "=" * 70)
    print("PROCESSING")
    print("=" * 70)

    print(f"Input : {input_file}")
    print(f"Output: {output_file}")

    with rasterio.open(input_file) as src:

        profile = src.profile.copy()

        profile.update(
            dtype="float32",
            count=1,
            nodata=NODATA,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
            BIGTIFF="YES"
        )

        print(f"\nWidth : {src.width}")
        print(f"Height: {src.height}")
        print(f"NoData: {src.nodata}")
        print(f"CRS  : {src.crs}")

        with rasterio.open(
            output_file,
            "w",
            **profile
        ) as dst:

            total_tiles = (
                (src.height + TILE_SIZE - 1) // TILE_SIZE
            ) * (
                (src.width + TILE_SIZE - 1) // TILE_SIZE
            )

            tile_number = 0

            for y in range(
                0,
                src.height,
                TILE_SIZE
            ):

                for x in range(
                    0,
                    src.width,
                    TILE_SIZE
                ):

                    width = min(
                        TILE_SIZE,
                        src.width - x
                    )

                    height = min(
                        TILE_SIZE,
                        src.height - y
                    )

                    window = Window(
                        x,
                        y,
                        width,
                        height
                    )

                    data = src.read(
                        1,
                        window=window,
                        out_dtype="float32"
                    )

                    processed = process_tile(data)

                    dst.write(
                        processed,
                        1,
                        window=window
                    )

                    tile_number += 1

                    if (
                        tile_number % 25 == 0
                        or tile_number == total_tiles
                    ):
                        percent = (
                            tile_number
                            / total_tiles
                            * 100
                        )

                        print(
                            f"Tiles: "
                            f"{tile_number}/{total_tiles} "
                            f"({percent:.1f}%)"
                        )

    print(f"\n✓ Saved: {output_file}")


def main():

    print("=" * 70)
    print("OCEANTRACEAI SENTINEL-1 PREPROCESSING")
    print("=" * 70)

    print("\nProcessing VV...")
    preprocess_file(
        VV_INPUT,
        VV_OUTPUT
    )

    print("\nProcessing VH...")
    preprocess_file(
        VH_INPUT,
        VH_OUTPUT
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print("\nOutputs:")

    print(
        f"VV: {VV_OUTPUT}"
    )

    print(
        f"VH: {VH_OUTPUT}"
    )


if __name__ == "__main__":
    main()