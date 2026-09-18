import math
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# ============================================================
# OCEANTRACEAI - INDIAN OCEAN SYNTHETIC AIS GENERATOR
# ============================================================

# IMPORTANT:
# This MUST use the new Indian backtracked origin.
ORIGIN_FILE = Path(
    "data/environment/indian_backtracked_origin.csv"
)

OUTPUT_FILE = Path(
    "data/ais/synthetic_indian_ocean_ais.csv"
)


# ============================================================
# AIS TIME WINDOW
# ============================================================

# The source time comes from the backward-drift calculation.
# We generate AIS around that time instead of using a
# hard-coded Gulf date/time.

HOURS_BEFORE_SOURCE = 2
HOURS_AFTER_SOURCE = 8

INTERVAL_MINUTES = 1


# ============================================================
# VESSEL DEFINITIONS
# ============================================================

VESSELS = [
    {
        "mmsi": "419000001",
        "vessel_name": "ARABIAN TRADER",
        "flag": "India",
        "start_offset": (-45, -20),
        "end_offset": (5, 2),
        "speed": 12,
        "ais_gap": False,
    },
    {
        "mmsi": "563000002",
        "vessel_name": "STRAIT NAVIGATOR",
        "flag": "Singapore",
        "start_offset": (35, 30),
        "end_offset": (15, 20),
        "speed": 10,
        "ais_gap": False,
    },
    {
        "mmsi": "470000003",
        "vessel_name": "GULF HORIZON",
        "flag": "UAE",
        "start_offset": (-60, 35),
        "end_offset": (-25, 15),
        "speed": 14,
        "ais_gap": False,
    },
    {
        "mmsi": "417000004",
        "vessel_name": "CEYLON STAR",
        "flag": "Sri Lanka",
        "start_offset": (55, -35),
        "end_offset": (30, -15),
        "speed": 11,
        "ais_gap": False,
    },
    {
        "mmsi": "533000005",
        "vessel_name": "MALACCA VOYAGER",
        "flag": "Malaysia",
        "start_offset": (-30, 50),
        "end_offset": (0, 5),
        "speed": 9,
        "ais_gap": True,
    },
]


# ============================================================
# LOAD INDIAN BACKTRACKED ORIGIN
# ============================================================

def load_origin():

    if not ORIGIN_FILE.exists():

        raise FileNotFoundError(
            f"\nIndian origin file not found:\n"
            f"{ORIGIN_FILE}\n\n"
            f"Run this first:\n"
            f"python backend\\drift.py"
        )

    df = pd.read_csv(
        ORIGIN_FILE
    )

    if df.empty:
        raise ValueError(
            "Indian origin file is empty."
        )

    row = df.iloc[0]

    latitude = float(
        row["origin_latitude"]
    )

    longitude = float(
        row["origin_longitude"]
    )

    source_time = datetime.fromisoformat(
        str(
            row["estimated_source_time_utc"]
        )
    )

    return (
        latitude,
        longitude,
        source_time
    )


# ============================================================
# CONVERT EAST/NORTH OFFSET TO LAT/LON
# ============================================================

def offset_position(
    origin_latitude,
    origin_longitude,
    east_km,
    north_km
):

    latitude_change = (
        north_km / 111.0
    )

    longitude_change = (
        east_km
        / (
            111.0
            * math.cos(
                math.radians(
                    origin_latitude
                )
            )
        )
    )

    latitude = (
        origin_latitude
        + latitude_change
    )

    longitude = (
        origin_longitude
        + longitude_change
    )

    return (
        latitude,
        longitude
    )


# ============================================================
# GENERATE ONE VESSEL TRACK
# ============================================================

def generate_vessel_track(
    vessel,
    origin_latitude,
    origin_longitude,
    source_time,
    timestamps
):

    start_east, start_north = (
        vessel["start_offset"]
    )

    end_east, end_north = (
        vessel["end_offset"]
    )

    total_steps = len(
        timestamps
    )

    records = []

    for i, timestamp in enumerate(
        timestamps
    ):

        # ----------------------------------------------------
        # Position along route
        # ----------------------------------------------------

        if total_steps <= 1:
            progress = 0.0
        else:
            progress = (
                i / (total_steps - 1)
            )

        east_km = (
            start_east
            + (
                end_east
                - start_east
            )
            * progress
        )

        north_km = (
            start_north
            + (
                end_north
                - start_north
            )
            * progress
        )

        latitude, longitude = (
            offset_position(
                origin_latitude,
                origin_longitude,
                east_km,
                north_km
            )
        )

        # ----------------------------------------------------
        # Intentional AIS gap
        #
        # MALACCA VOYAGER:
        # 30-minute AIS transmission gap centered
        # around estimated source time.
        # ----------------------------------------------------

        if vessel["ais_gap"]:

            gap_start = (
                source_time
                - timedelta(minutes=15)
            )

            gap_end = (
                source_time
                + timedelta(minutes=15)
            )

            if (
                gap_start
                <= timestamp
                <= gap_end
            ):
                continue

        # ----------------------------------------------------
        # Calculate course
        # ----------------------------------------------------

        if i < total_steps - 1:

            next_progress = (
                (i + 1)
                / (total_steps - 1)
            )

            next_east = (
                start_east
                + (
                    end_east
                    - start_east
                )
                * next_progress
            )

            next_north = (
                start_north
                + (
                    end_north
                    - start_north
                )
                * next_progress
            )

            delta_east = (
                next_east
                - east_km
            )

            delta_north = (
                next_north
                - north_km
            )

            cog = (
                math.degrees(
                    math.atan2(
                        delta_east,
                        delta_north
                    )
                )
                + 360
            ) % 360

        else:
            cog = 0.0

        # ----------------------------------------------------
        # Create AIS record
        # ----------------------------------------------------

        records.append(
            {
                "mmsi": vessel["mmsi"],
                "base_date_time":
                    timestamp.isoformat(),
                "longitude":
                    round(
                        longitude,
                        6
                    ),
                "latitude":
                    round(
                        latitude,
                        6
                    ),
                "sog":
                    vessel["speed"],
                "cog":
                    round(
                        cog,
                        2
                    ),
                "heading":
                    round(cog),
                "vessel_name":
                    vessel["vessel_name"],
                "flag":
                    vessel["flag"],
                "vessel_type":
                    "Cargo",
                "status":
                    "Under way",
                "length":
                    180,
                "width":
                    30,
            }
        )

    return records


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 68)
    print(
        "OCEANTRACEAI - INDIAN OCEAN "
        "SYNTHETIC AIS GENERATOR"
    )
    print("=" * 68)

    # --------------------------------------------------------
    # 1. Load Indian estimated origin
    # --------------------------------------------------------

    (
        origin_latitude,
        origin_longitude,
        source_time
    ) = load_origin()

    print("\nUsing OceanTraceAI Indian backtracked origin:")
    print(
        f"Latitude  : "
        f"{origin_latitude:.6f}"
    )
    print(
        f"Longitude : "
        f"{origin_longitude:.6f}"
    )
    print(
        f"Source time : "
        f"{source_time} UTC"
    )

    # --------------------------------------------------------
    # 2. Generate time window
    # --------------------------------------------------------

    start_time = (
        source_time
        - timedelta(
            hours=HOURS_BEFORE_SOURCE
        )
    )

    end_time = (
        source_time
        + timedelta(
            hours=HOURS_AFTER_SOURCE
        )
    )

    timestamps = pd.date_range(
        start=start_time,
        end=end_time,
        freq=f"{INTERVAL_MINUTES}min"
    ).to_pydatetime()

    print("\nAIS time window:")
    print(
        f"Start     : "
        f"{start_time} UTC"
    )
    print(
        f"End       : "
        f"{end_time} UTC"
    )
    print(
        f"Interval  : "
        f"{INTERVAL_MINUTES} minute"
    )

    # --------------------------------------------------------
    # 3. Generate vessels
    # --------------------------------------------------------

    all_records = []

    print("\nGenerating vessel tracks:")
    print("-" * 68)

    for vessel in VESSELS:

        records = (
            generate_vessel_track(
                vessel,
                origin_latitude,
                origin_longitude,
                source_time,
                timestamps
            )
        )

        all_records.extend(
            records
        )

        print(
            f"{vessel['vessel_name']:<24}"
            f"{len(records):>6} records"
        )

    # --------------------------------------------------------
    # 4. Create dataframe
    # --------------------------------------------------------

    ais_df = pd.DataFrame(
        all_records
    )

    # --------------------------------------------------------
    # 5. Sort chronologically
    # --------------------------------------------------------

    ais_df = ais_df.sort_values(
        [
            "base_date_time",
            "mmsi"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    ais_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # 7. Final statistics
    # --------------------------------------------------------

    print("\n" + "=" * 68)
    print(
        "INDIAN OCEAN AIS GENERATION COMPLETE"
    )
    print("=" * 68)

    print(
        f"\nOrigin latitude  : "
        f"{origin_latitude:.6f}"
    )

    print(
        f"Origin longitude : "
        f"{origin_longitude:.6f}"
    )

    print(
        f"Source time      : "
        f"{source_time} UTC"
    )

    print(
        f"Records          : "
        f"{len(ais_df)}"
    )

    print(
        f"Vessels          : "
        f"{ais_df['mmsi'].nunique()}"
    )

    print(
        f"Time range       : "
        f"{ais_df['base_date_time'].min()} "
        f"to "
        f"{ais_df['base_date_time'].max()}"
    )

    print(
        "\nSaved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "This AIS dataset is SYNTHETIC and is "
        "used only for algorithm demonstration/testing."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()