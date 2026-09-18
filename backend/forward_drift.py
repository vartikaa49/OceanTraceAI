import math
import csv
from pathlib import Path


# ============================================================
# OceanTraceAI - Indian Ocean Forward Spill Drift Forecast
# ============================================================

INPUT_FILE = Path(
    "data/environment/indian_backtracked_origin.csv"
)

OUTPUT_FILE = Path(
    "data/environment/indian_forward_drift_forecast.csv"
)

# Forecast horizons
FORECAST_HOURS = [0, 6, 12, 24]

# Wind-driven surface drift coefficient
WIND_DRIFT_FACTOR = 0.03


# ============================================================
# VECTOR CALCULATION
# ============================================================

def calculate_vector(speed, direction_deg):
    """
    Convert speed + compass bearing into
    east/north velocity components.

    Direction convention:
        0   = North
        90  = East
        180 = South
        270 = West
    """

    direction_rad = math.radians(
        direction_deg
    )

    east = (
        speed *
        math.sin(direction_rad)
    )

    north = (
        speed *
        math.cos(direction_rad)
    )

    return east, north


# ============================================================
# FORWARD POSITION
# ============================================================

def calculate_forward_position(
    latitude,
    longitude,
    hours,
    wind_speed,
    wind_direction_from,
    current_speed,
    current_direction
):
    """
    Calculate predicted future spill position.

    Wind direction from Open-Meteo is a FROM direction,
    so it is converted to a TOWARD direction.
    """

    # --------------------------------------------------------
    # Convert wind FROM → TOWARD
    # --------------------------------------------------------

    wind_direction_toward = (
        wind_direction_from + 180
    ) % 360

    # --------------------------------------------------------
    # Wind-driven surface drift
    # --------------------------------------------------------

    wind_drift_speed = (
        WIND_DRIFT_FACTOR *
        wind_speed
    )

    wind_east, wind_north = calculate_vector(
        wind_drift_speed,
        wind_direction_toward
    )

    # --------------------------------------------------------
    # Ocean current
    # --------------------------------------------------------

    current_east, current_north = calculate_vector(
        current_speed,
        current_direction
    )

    # --------------------------------------------------------
    # Combined transport
    # --------------------------------------------------------

    total_east = (
        current_east +
        wind_east
    )

    total_north = (
        current_north +
        wind_north
    )

    resultant_speed = math.sqrt(
        total_east ** 2 +
        total_north ** 2
    )

    resultant_direction = (
        math.degrees(
            math.atan2(
                total_east,
                total_north
            )
        ) + 360
    ) % 360

    # --------------------------------------------------------
    # Distance travelled
    # --------------------------------------------------------

    time_seconds = (
        hours *
        3600
    )

    east_distance = (
        total_east *
        time_seconds
    )

    north_distance = (
        total_north *
        time_seconds
    )

    # --------------------------------------------------------
    # Convert metres to latitude/longitude
    # --------------------------------------------------------

    latitude_change = (
        north_distance /
        111000
    )

    longitude_change = (
        east_distance /
        (
            111000 *
            math.cos(
                math.radians(
                    latitude
                )
            )
        )
    )

    predicted_latitude = (
        latitude +
        latitude_change
    )

    predicted_longitude = (
        longitude +
        longitude_change
    )

    return (
        predicted_latitude,
        predicted_longitude,
        resultant_speed,
        resultant_direction
    )


# ============================================================
# LOAD INDIAN CASE
# ============================================================

def load_current_case():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nInput file not found:\n{INPUT_FILE}\n"
        )

    with open(
        INPUT_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        rows = list(reader)

    if not rows:

        raise ValueError(
            "indian_backtracked_origin.csv is empty."
        )

    return rows[0]


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "        OCEANTRACEAI INDIAN OCEAN"
    )

    print(
        "        FORWARD DRIFT FORECAST"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Load case
    # --------------------------------------------------------

    row = load_current_case()


    # --------------------------------------------------------
    # Detected spill location
    # --------------------------------------------------------

    spill_latitude = float(
        row["spill_latitude"]
    )

    spill_longitude = float(
        row["spill_longitude"]
    )

    wind_speed = float(
        row["wind_speed_ms"]
    )

    wind_direction = float(
        row["wind_direction_from_deg"]
    )

    current_speed = float(
        row["current_speed_ms"]
    )

    current_direction = float(
        row["current_direction_toward_deg"]
    )

    image_id = row["image_id"]


    print("\nSCENARIO")
    print("-" * 70)

    print(
        "Scenario  : INDIAN_OCEAN_TEST"
    )

    print(
        f"Image ID  : {image_id}"
    )

    print(
        f"Latitude  : "
        f"{spill_latitude:.6f}"
    )

    print(
        f"Longitude : "
        f"{spill_longitude:.6f}"
    )


    # --------------------------------------------------------
    # Environmental conditions
    # --------------------------------------------------------

    print("\nENVIRONMENTAL CONDITIONS")
    print("-" * 70)

    print(
        f"Wind     : "
        f"{wind_speed:.2f} m/s "
        f"@ {wind_direction:.1f}° FROM"
    )

    print(
        f"Current  : "
        f"{current_speed:.3f} m/s "
        f"@ {current_direction:.2f}° TOWARD"
    )


    # --------------------------------------------------------
    # Forecast
    # --------------------------------------------------------

    forecast_rows = []

    print("\nFORWARD FORECAST")
    print("-" * 70)


    for hours in FORECAST_HOURS:

        (
            predicted_latitude,
            predicted_longitude,
            resultant_speed,
            resultant_direction
        ) = calculate_forward_position(

            spill_latitude,
            spill_longitude,
            hours,
            wind_speed,
            wind_direction,
            current_speed,
            current_direction
        )


        # ----------------------------------------------------
        # Simple uncertainty estimate
        #
        # This represents an increasing uncertainty radius.
        # It is NOT a probability.
        # ----------------------------------------------------

        uncertainty_km = (
            1.0 +
            hours * 0.25
        )


        forecast_rows.append({

            "image_id":
                image_id,

            "forecast_hours":
                hours,

            "predicted_latitude":
                predicted_latitude,

            "predicted_longitude":
                predicted_longitude,

            "resultant_speed_ms":
                resultant_speed,

            "resultant_direction_deg":
                resultant_direction,

            "uncertainty_radius_km":
                uncertainty_km
        })


        print(
            f"\nT+{hours:02d} HOURS"
        )

        print(
            f"  Latitude      : "
            f"{predicted_latitude:.6f}"
        )

        print(
            f"  Longitude     : "
            f"{predicted_longitude:.6f}"
        )

        print(
            f"  Transport     : "
            f"{resultant_speed:.3f} m/s"
        )

        print(
            f"  Direction     : "
            f"{resultant_direction:.2f}° TOWARD"
        )

        print(
            f"  Uncertainty   : "
            f"±{uncertainty_km:.2f} km"
        )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    fieldnames = [

        "image_id",

        "forecast_hours",

        "predicted_latitude",

        "predicted_longitude",

        "resultant_speed_ms",

        "resultant_direction_deg",

        "uncertainty_radius_km"
    ]


    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            forecast_rows
        )


    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "✓ INDIAN FORWARD FORECAST COMPLETE"
    )

    print("=" * 70)

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()