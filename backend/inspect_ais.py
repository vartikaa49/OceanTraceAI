import zstandard as zstd
from pathlib import Path

ais_file = Path("data/ais/ais-2026-03-31.csv.zst")

print("Opening AIS file...")
print(f"File: {ais_file}")
print(f"Size: {ais_file.stat().st_size / (1024 * 1024):.2f} MB")

with open(ais_file, "rb") as compressed_file:
    dctx = zstd.ZstdDecompressor()

    with dctx.stream_reader(compressed_file) as reader:
        data = reader.read(10000)

text = data.decode("utf-8", errors="replace")

print("\n========== FIRST AIS DATA ==========")
print(text[:10000])
print("====================================")