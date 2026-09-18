from pathlib import Path
from datetime import datetime, timedelta
import math
import pandas as pd

ORIGIN_FILE = Path("data/environment/indian_backtracked_origin.csv")
OUTPUT_FILE = Path("data/ais/synthetic_indian_ocean_ais.csv")

VESSELS = [
    {
        "mmsi": "419000001",
        "vessel_name": "ARABIAN TRADER",
        "flag": "India",
        "start": (-45, -20),
        "end": (5, 2),
        "speed": 12,
        "gap": False
    },
    {
        "mmsi": "563000002",
        "vessel_name": "STRAIT NAVIGATOR",
        "flag": "Singapore",
        "start": (35, 30),
        "end": (15, 20),
        "speed": 10,
        "gap": False
    },
    {
        "mmsi": "470000003",
        "vessel_name": "GULF HORIZON",
        "flag": "UAE",
        "start": (-60, 35),
        "end": (-25, 15),
        "speed": 14,
        "gap": False
    },
    {
        "mmsi": "417000004",
        "vessel_name": "CEYLON STAR",
        "flag": "Sri Lanka",
        "start": (55, -35),
        "end": (30, -15),
        "speed": 11,
        "gap": False
    },
    {
        "mmsi": "533000005",
        "vessel_name": "MALACCA VOYAGER",
        "flag": "Malaysia",
        "start": (-30, 50),
        "end": (0, 5),
        "speed": 9,
        "gap": True
    }
]


def load_origin():
    if not ORIGIN_FILE.exists():
        raise FileNotFoundError(
            f"Indian origin file not found: {ORIGIN_FILE}\n"
            "Run: python backend\\drift.py"
        )

    df = pd.read_csv(ORIGIN_FILE)
    row = df.iloc[0]

    latitude = float(row["origin_latitude"])
    longitude = float(row["origin_longitude"])

    source_time = datetime.fromisoformat(
        str(row["estimated_source_time_utc"])
    )

    return latitude, longitude, source_time


def offset_position(
    latitude,
    longitude,
    east_km,
    north_km
):
    lat = latitude + north_km / 111.0

    lon = longitude + (
        east_km /
        (
            111.0 *
            math.cos(math.radians(latitude))
        )
    )

    return lat, lon


def generate_track(
    vessel,
    origin_latitude,
    origin_longitude,
    source_time,
    timestamps
):
    start_east, start_north = vessel["start"]
    end_east, end_north = vessel["end"]

    total_steps = len(timestamps)
    records = []

    for i, timestamp in enumerate(timestamps):

        progress = (
            i / (total_steps - 1)
            if total_steps > 1
            else 0
        )

        east = (
            start_east +
            (end_east - start_east) * progress
        )

        north = (
            start_north +
            (end_north - start_north) * progress
        )

        latitude, longitude = offset_position(
            origin_latitude,
            origin_longitude,
            east,
            north
        )

        # Intentional 30-minute AIS gap
        # for MALACCA VOYAGER.
        if vessel["gap"]:
            gap_start = (
                source_time -
                timedelta(minutes=15)
            )

            gap_end = (
                source_time +
                timedelta(minutes=15)
            )

            if gap_start <= timestamp <= gap_end:
                continue

        if i < total_steps - 1:

            next_progress = (
                (i + 1) / (total_steps - 1)
            )

            next_east = (
                start_east +
                (end_east - start_east) *
                next_progress
            )

            next_north = (
                start_north +
                (end_north - start_north) *
                next_progress
            )

            delta_east = next_east - east
            delta_north = next_north - north

            cog = (
                math.degrees(
                    math.atan2(
                        delta_east,
                        delta_north
                    )
                ) + 360
            ) % 360

        else:
            cog = 0

        records.append({
            "mmsi": vessel["mmsi"],
            "base_date_time": timestamp.isoformat(),
            "longitude": round(longitude, 6),
            "latitude": round(latitude, 6),
            "sog": vessel["speed"],
            "cog": round(cog, 2),
            "heading": round(cog),
            "vessel_name": vessel["vessel_name"],
            "flag": vessel["flag"],
            "vessel_type": "Cargo",
            "status": "Under way",
            "length": 180,
            "width": 30
        })

    return records


def main():

    print("=" * 68)
    print("OCEANTRACEAI - INDIAN OCEAN SYNTHETIC AIS GENERATOR")
    print("=" * 68)

    origin_latitude, origin_longitude, source_time = load_origin()

    print("\nUsing OceanTraceAI Indian backtracked origin:")
    print(f"Latitude  : {origin_latitude:.6f}")
    print(f"Longitude : {origin_longitude:.6f}")
    print(f"Source time : {source_time} UTC")

    start_time = (
        source_time -
        timedelta(hours=2)
    )

    end_time = (
        source_time +
        timedelta(hours=8)
    )

    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq="1min"
    ).to_pydatetime()

    print("\nAIS time window:")
    print(f"Start : {start_time} UTC")
    print(f"End   : {end_time} UTC")

    all_records = []

    print("\nGenerating vessel tracks:")
    print("-" * 68)

    for vessel in VESSELS:

        records = generate_track(
            vessel,
            origin_latitude,
            origin_longitude,
            source_time,
            timestamps
        )

        all_records.extend(records)

        print(
            f"{vessel['vessel_name']:<24}"
            f"{len(records):>6} records"
        )

    ais_df = pd.DataFrame(all_records)

    ais_df = ais_df.sort_values(
        ["base_date_time", "mmsi"]
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ais_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 68)
    print("INDIAN OCEAN AIS GENERATION COMPLETE")
    print("=" * 68)

    print(f"\nOrigin latitude  : {origin_latitude:.6f}")
    print(f"Origin longitude : {origin_longitude:.6f}")
    print(f"Source time      : {source_time} UTC")
    print(f"Records          : {len(ais_df)}")
    print(f"Vessels          : {ais_df['mmsi'].nunique()}")

    print(
        f"Time range       : "
        f"{ais_df['base_date_time'].min()} "
        f"to "
        f"{ais_df['base_date_time'].max()}"
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    print(
        "\nIMPORTANT: This AIS dataset is SYNTHETIC "
        "and is used only for demonstration/testing."
    )


if __name__ == "__main__":
    main()
