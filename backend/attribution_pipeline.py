import pandas as pd
from pathlib import Path
from datetime import datetime

from drift import calculate_backward_drift
from environment import get_historical_wind
from current import get_ocean_current


# ============================================================
# FILE PATHS
# ============================================================

SPILL_LOCATION_FILE = Path(
    "data/environment/detected_spill_location.csv"
)

OUTPUT_FILE = Path(
    "data/environment/backtracked_origin.csv"
)


# ============================================================
# SENTINEL-1 OBSERVATION TIME
# ============================================================

# Acquisition time from the Sentinel-1 product
# currently being used for testing.
OBSERVATION_TIME = datetime(
    2026,
    9,
    8,
    0,
    9,
    23
)


# ============================================================
# BACKWARD DRIFT PERIOD
# ============================================================

BACKTRACK_HOURS = 12


# ============================================================
# LOAD DETECTED SPILL LOCATION
# ============================================================

def load_detected_spill_location():

    if not SPILL_LOCATION_FILE.exists():
        raise FileNotFoundError(
            f"Spill location file not found:\n"
            f"{SPILL_LOCATION_FILE}"
        )

    # Keep image ID as a string so 01338 remains 01338.
    df = pd.read_csv(
        SPILL_LOCATION_FILE,
        dtype={"image_id": str}
    )

    if df.empty:
        raise ValueError(
            "Spill location CSV is empty."
        )

    row = df.iloc[0]

    image_id = str(row["image_id"])

    spill_lat = float(
        row["latitude"]
    )

    spill_lon = float(
        row["longitude"]
    )

    return (
        image_id,
        spill_lat,
        spill_lon
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    print("=" * 48)
    print("       OCEANTRACEAI DRIFT PIPELINE")
    print("=" * 48)


    # --------------------------------------------------------
    # STEP 1 — DETECTED SPILL LOCATION
    # --------------------------------------------------------

    print()
    print("STEP 1: DETECTED SPILL LOCATION")
    print("-" * 47)

    (
        image_id,
        spill_lat,
        spill_lon
    ) = load_detected_spill_location()

    print(
        f"Image ID       : {image_id}"
    )

    print(
        f"Spill latitude  : {spill_lat:.6f}"
    )

    print(
        f"Spill longitude : {spill_lon:.6f}"
    )

    print(
        f"Observation UTC: "
        f"{OBSERVATION_TIME}"
    )


    # --------------------------------------------------------
    # STEP 2 — HISTORICAL WIND
    # --------------------------------------------------------

    print()
    print("STEP 2: HISTORICAL WIND")
    print("-" * 47)

    wind_speed, wind_direction = (
        get_historical_wind(
            spill_lat,
            spill_lon,
            OBSERVATION_TIME
        )
    )

    print(
        f"Wind speed     : "
        f"{wind_speed} m/s"
    )

    print(
        f"Wind direction : "
        f"{wind_direction}°"
    )


    # --------------------------------------------------------
    # STEP 3 — DYNAMIC COPERNICUS OCEAN CURRENT
    # --------------------------------------------------------

    print()
    print("STEP 3: OCEAN CURRENT")
    print("-" * 47)

    current_speed, current_direction = (
        get_ocean_current(
            latitude=spill_lat,
            longitude=spill_lon,
            observation_time=OBSERVATION_TIME
        )
    )

    print(
        f"Current speed     : "
        f"{current_speed:.6f} m/s"
    )

    print(
        f"Current direction : "
        f"{current_direction:.2f}°"
    )

    print(
        "Source            : "
        "Copernicus Marine"
    )


    # --------------------------------------------------------
    # STEP 4 — BACKWARD DRIFT
    # --------------------------------------------------------

    print()
    print("STEP 4: BACKWARD DRIFT")
    print("-" * 47)

    (
        origin_lat,
        origin_lon
    ) = calculate_backward_drift(

        latitude=spill_lat,

        longitude=spill_lon,

        observation_time_hours=BACKTRACK_HOURS,

        wind_speed=wind_speed,

        wind_direction=wind_direction,

        current_speed=current_speed,

        current_direction=current_direction
    )

    print(
        f"Estimated source latitude : "
        f"{origin_lat:.6f}"
    )

    print(
        f"Estimated source longitude: "
        f"{origin_lon:.6f}"
    )


    # --------------------------------------------------------
    # STEP 5 — SAVE BACKTRACKED ORIGIN
    # --------------------------------------------------------

    print()
    print("STEP 5: SAVE BACKTRACKED ORIGIN")
    print("-" * 47)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = pd.DataFrame([
        {
            "image_id": image_id,

            "observation_time_utc":
                OBSERVATION_TIME.isoformat(),

            "spill_latitude":
                spill_lat,

            "spill_longitude":
                spill_lon,

            "wind_speed_ms":
                wind_speed,

            "wind_direction_deg":
                wind_direction,

            "current_speed_ms":
                current_speed,

            "current_direction_deg":
                current_direction,

            "backtrack_hours":
                BACKTRACK_HOURS,

            "origin_latitude":
                origin_lat,

            "origin_longitude":
                origin_lon
        }
    ])

    result.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("✓ Result saved to:")

    print(
        OUTPUT_FILE
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 48)
    print("       DRIFT PIPELINE COMPLETE")
    print("=" * 48)

    print()
    print("Automatic flow:")

    print("U-Net detection")

    print("      ↓")

    print(
        f"Spill centroid "
        f"({spill_lat:.6f}, {spill_lon:.6f})"
    )

    print("      ↓")

    print("Historical wind")

    print("      ↓")

    print("Copernicus ocean current")

    print("      ↓")

    print(
        f"Backtracked origin "
        f"({origin_lat:.6f}, {origin_lon:.6f})"
    )

    print("=" * 48)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()