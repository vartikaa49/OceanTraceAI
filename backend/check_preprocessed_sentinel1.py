from pathlib import Path

import numpy as np
import rasterio


DATA_DIR = Path("data/processed/sentinel1/preprocessed")

FILES = {
    "VV": DATA_DIR / "sentinel1_VV_preprocessed.tif",
    "VH": DATA_DIR / "sentinel1_VH_preprocessed.tif",
}


def check_file(name, path):

    print("\n" + "=" * 70)
    print(f"{name} PREPROCESSED SAR CHECK")
    print("=" * 70)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return

    with rasterio.open(path) as src:

        print(f"File   : {path}")
        print(f"Width  : {src.width}")
        print(f"Height : {src.height}")
        print(f"Dtype  : {src.dtypes[0]}")
        print(f"NoData : {src.nodata}")
        print(f"CRS    : {src.crs}")

        # Read a representative window rather than the whole
        # 430-million-pixel scene.
        window_size = 1024

        row = min(5000, src.height - window_size)
        col = min(5000, src.width - window_size)

        window = rasterio.windows.Window(
            col,
            row,
            window_size,
            window_size
        )

        data = src.read(
            1,
            window=window
        ).astype(np.float32)

        valid = (
            np.isfinite(data)
            & (data != src.nodata)
        )

        valid_data = data[valid]

        print("\nRepresentative window:")
        print(f"Rows   : {row} -> {row + window_size}")
        print(f"Columns: {col} -> {col + window_size}")

        print("\nPixel statistics:")

        if len(valid_data) == 0:

            print("ERROR: No valid pixels found!")

        else:

            print(f"Valid pixels : {len(valid_data):,}")
            print(
                f"Valid ratio  : "
                f"{len(valid_data) / data.size * 100:.2f}%"
            )

            print(
                f"Minimum      : "
                f"{np.min(valid_data):.4f} dB"
            )

            print(
                f"Maximum      : "
                f"{np.max(valid_data):.4f} dB"
            )

            print(
                f"Mean         : "
                f"{np.mean(valid_data):.4f} dB"
            )

            print(
                f"Median       : "
                f"{np.median(valid_data):.4f} dB"
            )

            print(
                f"P1           : "
                f"{np.percentile(valid_data, 1):.4f} dB"
            )

            print(
                f"P99          : "
                f"{np.percentile(valid_data, 99):.4f} dB"
            )


def main():

    print("=" * 70)
    print("OCEANTRACEAI - PREPROCESSED SENTINEL-1 VALIDATION")
    print("=" * 70)

    for name, path in FILES.items():
        check_file(name, path)

    print("\n" + "=" * 70)
    print("CHECK COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()