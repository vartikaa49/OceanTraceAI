import requests
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# OCEANTRACEAI - INDIAN OCEAN ENVIRONMENT
# ============================================================

SCENARIO_FILE = Path(
    "data/environment/indian_test_scenario.csv"
)


# ============================================================
# HISTORICAL WIND
# ============================================================

def get_historical_wind(
    latitude,
    longitude,
    datetime_utc
):
    """
    Get historical 10 m wind conditions
    from Open-Meteo.
    """

    date = datetime_utc.strftime(
        "%Y-%m-%d"
    )

    hour = datetime_utc.hour

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
    )

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": date,
        "end_date": date,
        "hourly": (
            "wind_speed_10m,"
            "wind_direction_10m"
        ),
        "wind_speed_unit": "ms",
        "timezone": "UTC"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    wind_speed = data["hourly"][
        "wind_speed_10m"
    ][hour]

    wind_direction = data["hourly"][
        "wind_direction_10m"
    ][hour]

    return (
        wind_speed,
        wind_direction
    )


# ============================================================
# LOAD INDIAN OCEAN SCENARIO
# ============================================================

def load_indian_scenario():

    if not SCENARIO_FILE.exists():

        raise FileNotFoundError(
            f"Indian Ocean scenario not found:\n"
            f"{SCENARIO_FILE}"
        )

    df = pd.read_csv(
        SCENARIO_FILE
    )

    if df.empty:

        raise ValueError(
            "Indian Ocean scenario is empty."
        )

    row = df.iloc[0]

    latitude = float(
        row["spill_latitude"]
    )

    longitude = float(
        row["spill_longitude"]
    )

    observation_time = datetime.fromisoformat(
        row["observation_time_utc"]
    )

    return (
        latitude,
        longitude,
        observation_time
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 68)
    print("       OCEANTRACEAI INDIAN OCEAN WIND TEST")
    print("=" * 68)

    try:

        (
            latitude,
            longitude,
            acquisition_time
        ) = load_indian_scenario()

        print()
        print("Scenario loaded:")
        print(
            f"Latitude  : {latitude:.6f}"
        )
        print(
            f"Longitude : {longitude:.6f}"
        )
        print(
            f"Time UTC  : {acquisition_time}"
        )

        print()
        print("Requesting historical wind...")

        (
            wind_speed,
            wind_direction
        ) = get_historical_wind(
            latitude,
            longitude,
            acquisition_time
        )

        print()
        print("Environmental data:")
        print(
            f"Wind speed     : "
            f"{wind_speed} m/s"
        )

        print(
            f"Wind direction : "
            f"{wind_direction}°"
        )

        print()
        print("✓ REAL HISTORICAL WIND DATA RECEIVED")

        print()
        print("=" * 68)
        print("                 TEST SUCCESSFUL")
        print("=" * 68)

    except Exception as e:

        print()
        print(
            f"ERROR: {e}"
        )

        print()
        print("=" * 68)
        print("                  TEST FAILED")
        print("=" * 68)