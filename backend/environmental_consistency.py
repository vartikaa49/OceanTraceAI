import pandas as pd
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2, degrees


ORIGIN_FILE = Path("data/environment/backtracked_origin.csv")
AIS_FILE = Path("data/ais/synthetic_gulf_ais.csv")


def angle_difference(angle1, angle2):
    difference = abs(angle1 - angle2)

    if difference > 180:
        difference = 360 - difference

    return difference


def load_environmental_data():

    if not ORIGIN_FILE.exists():
        raise FileNotFoundError(
            f"Backtracked origin not found:\n{ORIGIN_FILE}"
        )

    df = pd.read_csv(ORIGIN_FILE)

    if df.empty:
        raise ValueError("Backtracked origin file is empty.")

    row = df.iloc[0]

    return (
        float(row["origin_latitude"]),
        float(row["origin_longitude"]),
        float(row["wind_speed_ms"]),
        float(row["wind_direction_deg"]),
        float(row["current_speed_ms"]),
        float(row["current_direction_deg"])
    )


def calculate_environmental_drift(
    wind_speed,
    wind_direction_from,
    current_speed,
    current_direction
):
    """
    Calculate resultant surface transport direction.

    Open-Meteo wind direction is meteorological:
    it tells us where the wind is COMING FROM.

    Therefore:
        wind toward = wind from + 180 degrees

    Ocean current direction is already a TOWARD direction.
    """

    # ---------------------------------------------------------
    # WIND
    # ---------------------------------------------------------

    wind_direction_toward = (
        wind_direction_from + 180
    ) % 360

    wind_rad = radians(
        wind_direction_toward
    )

    wind_drift_speed = (
        0.03 * wind_speed
    )

    wind_x = (
        wind_drift_speed
        * sin(wind_rad)
    )

    wind_y = (
        wind_drift_speed
        * cos(wind_rad)
    )

    # ---------------------------------------------------------
    # OCEAN CURRENT
    # ---------------------------------------------------------

    current_rad = radians(
        current_direction
    )

    current_x = (
        current_speed
        * sin(current_rad)
    )

    current_y = (
        current_speed
        * cos(current_rad)
    )

    # ---------------------------------------------------------
    # RESULTANT VECTOR
    # ---------------------------------------------------------

    total_x = wind_x + current_x
    total_y = wind_y + current_y

    drift_speed = sqrt(
        total_x ** 2
        + total_y ** 2
    )

    drift_direction = degrees(
        atan2(total_x, total_y)
    )

    if drift_direction < 0:
        drift_direction += 360

    return drift_speed, drift_direction


def calculate_vessel_direction(vessel_df):

    vessel_df = vessel_df.sort_values(
        "base_date_time"
    )

    if len(vessel_df) < 2:
        return None

    first = vessel_df.iloc[0]
    last = vessel_df.iloc[-1]

    lat1 = radians(float(first["latitude"]))
    lat2 = radians(float(last["latitude"]))

    delta_lon = radians(
        float(last["longitude"])
        - float(first["longitude"])
    )

    y = (
        sin(delta_lon)
        * cos(lat2)
    )

    x = (
        cos(lat1)
        * sin(lat2)
        -
        sin(lat1)
        * cos(lat2)
        * cos(delta_lon)
    )

    bearing = degrees(
        atan2(y, x)
    )

    if bearing < 0:
        bearing += 360

    return bearing


def calculate_consistency(
    vessel_direction,
    environmental_direction
):

    difference = angle_difference(
        vessel_direction,
        environmental_direction
    )

    score = 100 * (
        1 - difference / 180
    )

    return round(
        max(0, min(100, score)),
        2
    )


def main():

    print("=" * 65)
    print(
        "   OCEANTRACEAI ENVIRONMENTAL CONSISTENCY"
    )
    print("=" * 65)

    (
        origin_lat,
        origin_lon,
        wind_speed,
        wind_direction_from,
        current_speed,
        current_direction
    ) = load_environmental_data()

    print("\nBACKTRACKED ORIGIN")
    print("-" * 64)

    print(
        f"Latitude  : {origin_lat:.6f}"
    )

    print(
        f"Longitude : {origin_lon:.6f}"
    )

    print("\nENVIRONMENTAL CONDITIONS")
    print("-" * 64)

    print(
        f"Wind speed       : "
        f"{wind_speed:.2f} m/s"
    )

    print(
        f"Wind FROM        : "
        f"{wind_direction_from:.2f}°"
    )

    wind_direction_toward = (
        wind_direction_from + 180
    ) % 360

    print(
        f"Wind TOWARD      : "
        f"{wind_direction_toward:.2f}°"
    )

    print(
        f"Current speed    : "
        f"{current_speed:.6f} m/s"
    )

    print(
        f"Current direction: "
        f"{current_direction:.2f}°"
    )

    (
        drift_speed,
        drift_direction
    ) = calculate_environmental_drift(
        wind_speed,
        wind_direction_from,
        current_speed,
        current_direction
    )

    print("\nRESULTANT ENVIRONMENTAL DRIFT")
    print("-" * 64)

    print(
        f"Drift speed     : "
        f"{drift_speed:.6f} m/s"
    )

    print(
        f"Drift direction : "
        f"{drift_direction:.2f}°"
    )

    if not AIS_FILE.exists():
        raise FileNotFoundError(
            f"AIS file not found:\n{AIS_FILE}"
        )

    ais_df = pd.read_csv(
        AIS_FILE
    )

    ais_df["base_date_time"] = pd.to_datetime(
        ais_df["base_date_time"],
        utc=True
    )

    print("\nVESSEL ENVIRONMENTAL CONSISTENCY")
    print("-" * 64)

    results = []

    for mmsi, group in ais_df.groupby("mmsi"):

        vessel_name = group[
            "vessel_name"
        ].iloc[0]

        vessel_direction = (
            calculate_vessel_direction(
                group
            )
        )

        if vessel_direction is None:
            score = 0.0
        else:
            score = calculate_consistency(
                vessel_direction,
                drift_direction
            )

        results.append({
            "mmsi": mmsi,
            "vessel_name": vessel_name,
            "vessel_direction_deg": round(
                vessel_direction, 2
            ),
            "environmental_drift_direction_deg": round(
                drift_direction, 2
            ),
            "environmental_consistency_score": score
        })

        print(
            f"\n{vessel_name}"
        )

        print(
            f"Vessel direction : "
            f"{vessel_direction:.2f}°"
        )

        print(
            f"Environmental drift : "
            f"{drift_direction:.2f}°"
        )

        print(
            f"Consistency score : "
            f"{score:.2f}"
        )

    output_file = Path(
        "data/ais/ais_environmental_scores.csv"
    )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        "environmental_consistency_score",
        ascending=False
    )

    result_df.to_csv(
        output_file,
        index=False
    )

    print("\n✓ Environmental scores saved:")
    print(output_file)

    print("\n" + "=" * 65)
    print(
        "ENVIRONMENTAL CONSISTENCY COMPLETE"
    )
    print("=" * 65)


if __name__ == "__main__":
    main()