from pathlib import Path
from datetime import datetime, timedelta
import math
import csv
import sys

import requests

# Make sure backend modules can be imported
sys.path.append(str(Path(__file__).resolve().parent))

from current import get_ocean_current


# ============================================================
# OCEANTRACEAI - GULF OF MEXICO BACKWARD DRIFT
# ============================================================

# Centroid obtained from spill_geometry.py
SPILL_LATITUDE = 28.9081354590571
SPILL_LONGITUDE = -89.02388121183533

# Gulf investigation observation time
OBSERVATION_TIME = datetime(
    2026,
    9,
    8,
    0,
    9,
    23
)

BACKTRACK_HOURS = 12

OUTPUT_FILE = Path(
    "data/environment/gulf_backtracked_origin.csv"
)


# ============================================================
# HISTORICAL WIND
# ============================================================

def get_historical_wind(
    latitude,
    longitude,
    observation_time
):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": observation_time.strftime(
            "%Y-%m-%d"
        ),
        "end_date": observation_time.strftime(
            "%Y-%m-%d"
        ),
        "hourly":
            "wind_speed_10m,wind_direction_10m",
        "timezone": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    times = data["hourly"]["time"]
    speeds = data["hourly"]["wind_speed_10m"]
    directions = data["hourly"]["wind_direction_10m"]

    closest_index = min(
        range(len(times)),
        key=lambda i: abs(
            datetime.fromisoformat(
                times[i]
            ) - observation_time
        )
    )

    return (
        float(speeds[closest_index]),
        float(directions[closest_index])
    )


# ============================================================
# WIND VECTOR
# ============================================================

def direction_to_vector(
    speed,
    direction_from
):

    # Meteorological wind direction tells us
    # where the wind comes FROM.

    direction_toward = (
        direction_from + 180
    ) % 360

    radians = math.radians(
        direction_toward
    )

    east = (
        speed *
        math.sin(radians)
    )

    north = (
        speed *
        math.cos(radians)
    )

    return (
        east,
        north,
        direction_toward
    )


# ============================================================
# VECTOR DIRECTION
# ============================================================

def vector_to_direction(
    east,
    north
):

    direction = math.degrees(
        math.atan2(
            east,
            north
        )
    )

    return direction % 360


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 68)
    print(
        "OceanTraceAI - Gulf of Mexico Backward Drift"
    )
    print("=" * 68)

    print("\nSCENARIO")
    print("-" * 68)

    print(
        "Scenario             : GULF_VALIDATION_01338"
    )

    print(
        "Observation time     : "
        f"{OBSERVATION_TIME} UTC"
    )

    print(
        f"Spill latitude       : "
        f"{SPILL_LATITUDE:.6f}"
    )

    print(
        f"Spill longitude      : "
        f"{SPILL_LONGITUDE:.6f}"
    )

    print(
        f"Backtrack duration   : "
        f"{BACKTRACK_HOURS:.1f} hours"
    )


    # ========================================================
    # WIND
    # ========================================================

    print("\n" + "=" * 68)
    print(
        "FETCHING HISTORICAL WIND"
    )
    print("=" * 68)

    wind_speed, wind_direction_from = (
        get_historical_wind(
            SPILL_LATITUDE,
            SPILL_LONGITUDE,
            OBSERVATION_TIME
        )
    )

    print(
        f"Wind speed           : "
        f"{wind_speed:.3f} m/s"
    )

    print(
        f"Wind direction FROM  : "
        f"{wind_direction_from:.2f}°"
    )

    (
        wind_east,
        wind_north,
        wind_direction_toward
    ) = direction_to_vector(
        wind_speed,
        wind_direction_from
    )


    # ========================================================
    # COPERNICUS CURRENT
    # ========================================================

    print("\n" + "=" * 68)
    print(
        "FETCHING OCEAN CURRENT"
    )
    print("=" * 68)

    (
        current_speed,
        current_direction
    ) = get_ocean_current(
        SPILL_LATITUDE,
        SPILL_LONGITUDE,
        OBSERVATION_TIME
    )

    print(
        "\n========== COPERNICUS CURRENT =========="
    )

    print(
        f"Current speed        : "
        f"{current_speed:.6f} m/s"
    )

    print(
        f"Current direction    : "
        f"{current_direction:.2f}° TOWARD"
    )

    current_radians = math.radians(
        current_direction
    )

    current_east = (
        current_speed *
        math.sin(current_radians)
    )

    current_north = (
        current_speed *
        math.cos(current_radians)
    )


    # ========================================================
    # RESULTANT TRANSPORT
    # ========================================================

    # Wind contribution is weighted down because
    # surface current is the dominant transport term.

    resultant_east = (
        current_east +
        wind_east * 0.03
    )

    resultant_north = (
        current_north +
        wind_north * 0.03
    )

    resultant_speed = math.sqrt(
        resultant_east ** 2 +
        resultant_north ** 2
    )

    transport_direction = (
        vector_to_direction(
            resultant_east,
            resultant_north
        )
    )


    # ========================================================
    # BACKWARD DRIFT
    # ========================================================

    print("\n" + "=" * 68)
    print(
        "RUNNING BACKWARD DRIFT"
    )
    print("=" * 68)

    seconds = (
        BACKTRACK_HOURS *
        3600
    )

    # Reverse the transport vector.

    east_displacement_m = (
        -resultant_east *
        seconds
    )

    north_displacement_m = (
        -resultant_north *
        seconds
    )

    displacement_east_km = (
        east_displacement_m /
        1000
    )

    displacement_north_km = (
        north_displacement_m /
        1000
    )

    # Approximate geographical conversion.

    latitude_km_per_degree = 111.32

    longitude_km_per_degree = (
        111.32 *
        math.cos(
            math.radians(
                SPILL_LATITUDE
            )
        )
    )

    source_latitude = (
        SPILL_LATITUDE +
        displacement_north_km /
        latitude_km_per_degree
    )

    source_longitude = (
        SPILL_LONGITUDE +
        displacement_east_km /
        longitude_km_per_degree
    )

    source_time = (
        OBSERVATION_TIME -
        timedelta(
            hours=BACKTRACK_HOURS
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    print("\n" + "=" * 68)
    print(
        "BACKWARD DRIFT RESULT"
    )
    print("=" * 68)

    print(
        f"Observed spill      : "
        f"{SPILL_LATITUDE:.6f}, "
        f"{SPILL_LONGITUDE:.6f}"
    )

    print(
        f"Wind toward         : "
        f"{wind_direction_toward:.2f}°"
    )

    print(
        f"Resultant speed     : "
        f"{resultant_speed:.6f} m/s"
    )

    print(
        f"Transport direction : "
        f"{transport_direction:.2f}° TOWARD"
    )

    print(
        f"X displacement      : "
        f"{displacement_east_km:.3f} km"
    )

    print(
        f"Y displacement      : "
        f"{displacement_north_km:.3f} km"
    )

    print(
        f"Estimated source    : "
        f"{source_latitude:.6f}, "
        f"{source_longitude:.6f}"
    )

    print(
        f"Estimated source time: "
        f"{source_time} UTC"
    )


    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "scenario",
            "observation_time_utc",
            "spill_latitude",
            "spill_longitude",
            "wind_speed_ms",
            "wind_direction_from_deg",
            "current_speed_ms",
            "current_direction_toward_deg",
            "resultant_speed_ms",
            "transport_direction_toward_deg",
            "backtrack_hours",
            "source_latitude",
            "source_longitude",
            "source_time_utc"
        ])

        writer.writerow([
            "GULF_VALIDATION_01338",
            OBSERVATION_TIME.strftime(
                "%Y-%m-%dT%H:%M:%S"
            ),
            SPILL_LATITUDE,
            SPILL_LONGITUDE,
            wind_speed,
            wind_direction_from,
            current_speed,
            current_direction,
            resultant_speed,
            transport_direction,
            BACKTRACK_HOURS,
            source_latitude,
            source_longitude,
            source_time.strftime(
                "%Y-%m-%dT%H:%M:%S"
            )
        ])


    print("\n" + "=" * 68)
    print(
        "✓ GULF BACKTRACKING COMPLETE"
    )
    print("=" * 68)

    print(
        "\n✓ Estimated Gulf origin saved to:"
    )

    print(
        OUTPUT_FILE
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()