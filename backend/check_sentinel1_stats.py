import rasterio
import numpy as np
from pathlib import Path

INPUT_DIR = Path("data/processed/sentinel1")
NODATA = -9999.0

print("=" * 70)
print("SENTINEL-1 CALIBRATED DATA QUALITY CHECK")
print("=" * 70)

for path in sorted(INPUT_DIR.glob("*_sigma0_db.tif")):

    print(f"\nFile: {path.name}")

    with rasterio.open(path) as src:

        valid_values = []

        for y in range(0, src.height, 1024):

            height = min(1024, src.height - y)

            data = src.read(
                1,
                window=((y, y + height), (0, src.width)),
                out_dtype="float32"
            )

            data = data[data != NODATA]
            data = data[np.isfinite(data)]

            if data.size > 0:

                # Sample to keep memory usage low
                step = max(1, data.size // 200000)
                valid_values.append(data[::step])

        values = np.concatenate(valid_values)

        print(f"Width:       {src.width}")
        print(f"Height:      {src.height}")
        print(f"CRS:         {src.crs}")
        print(f"NoData:      {src.nodata}")
        print(f"Valid pixels: {values.size:,}")

        print("\nStatistics excluding NoData:")
        print(f"Minimum:     {np.min(values):.2f} dB")
        print(f"Maximum:     {np.max(values):.2f} dB")
        print(f"Mean:        {np.mean(values):.2f} dB")
        print(f"Median:      {np.median(values):.2f} dB")
        print(f"5th percentile:  {np.percentile(values, 5):.2f} dB")
        print(f"95th percentile: {np.percentile(values, 95):.2f} dB")

print("\n" + "=" * 70)
print("QUALITY CHECK COMPLETE")
print("=" * 70)