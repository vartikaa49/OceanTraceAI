import math
from pathlib import Path
from math import radians, sin, cos, sqrt, atan2

import pandas as pd


# ============================================================
# OCEANTRACEAI - AIS VESSEL ATTRIBUTION
# ============================================================

ORIGIN_FILE = Path(
    "data/environment/indian_backtracked_origin.csv"
)

AIS_FILE = Path(
    "data/ais/synthetic_indian_ocean_ais.csv"
)

OUTPUT_FILE = Path(
    "data/ais/ais_indian_attribution.csv"
)


# ============================================================
# ATTRIBUTION PARAMETERS
# ============================================================

TEMPORAL_WINDOW_HOURS = 6

SPATIAL_RADIUS_KM = 100


# Final evidence weights
PROXIMITY_WEIGHT = 0.25
TEMPORAL_WEIGHT = 0.15
TRAJECTORY_WEIGHT = 0.30
ANOMALY_WEIGHT = 0.10
ENVIRONMENTAL_WEIGHT = 0.20


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate great-circle distance between
    two latitude/longitude points.

    Returns distance in kilometres.
    """

    R = 6371.0

    lat1_rad = radians(lat1)
    lat2_rad = radians(lat2)

    delta_lat = radians(
        lat2 - lat1
    )

    delta_lon = radians(
        lon2 - lon1
    )

    a = (
        sin(delta_lat / 2) ** 2
        +
        cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a)
    )

    return R * c


# ============================================================
# BEARING
# ============================================================

def calculate_bearing(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate bearing from point 1 to point 2.

    Bearing is measured clockwise from North.
    """

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    delta_lon = radians(
        lon2 - lon1
    )

    x = (
        sin(delta_lon)
        * cos(lat2)
    )

    y = (
        cos(lat1)
        * sin(lat2)
        -
        sin(lat1)
        * cos(lat2)
        * cos(delta_lon)
    )

    bearing = (
        math.degrees(
            atan2(x, y)
        )
        + 360
    ) % 360

    return bearing


# ============================================================
# ANGULAR DIFFERENCE
# ============================================================

def angular_difference(
    angle1,
    angle2
):
    """
    Smallest absolute difference between
    two compass directions.
    """

    difference = abs(
        angle1 - angle2
    )

    return min(
        difference,
        360 - difference
    )


# ============================================================
# LOAD BACKTRACKED INDIAN ORIGIN
# ============================================================

def load_origin():

    if not ORIGIN_FILE.exists():

        raise FileNotFoundError(
            f"\nIndian backtracked origin not found:\n"
            f"{ORIGIN_FILE}\n\n"
            f"Run first:\n"
            f"python backend\\drift.py"
        )

    df = pd.read_csv(
        ORIGIN_FILE
    )

    if df.empty:

        raise ValueError(
            "Indian backtracked origin file is empty."
        )

    row = df.iloc[0]

    image_id = str(
        row["image_id"]
    )

    origin_latitude = float(
        row["origin_latitude"]
    )

    origin_longitude = float(
        row["origin_longitude"]
    )

    observation_time = pd.to_datetime(
        row["observation_time_utc"],
        utc=True
    )

    source_time = pd.to_datetime(
        row["estimated_source_time_utc"],
        utc=True
    )

    # Environmental transport direction generated
    # by backward drift.
    transport_direction = float(
        row["transport_direction_toward_deg"]
    )

    transport_speed = float(
        row["transport_speed_ms"]
    )

    return (
        image_id,
        origin_latitude,
        origin_longitude,
        observation_time,
        source_time,
        transport_direction,
        transport_speed
    )


# ============================================================
# LOAD AIS
# ============================================================

def load_ais():

    if not AIS_FILE.exists():

        raise FileNotFoundError(
            f"\nIndian AIS file not found:\n"
            f"{AIS_FILE}\n\n"
            f"Run first:\n"
            f"python backend\\generate_gulf_ais.py"
        )

    print(
        "\nLoading Indian AIS data..."
    )

    print(
        f"File: {AIS_FILE}"
    )

    df = pd.read_csv(
        AIS_FILE
    )

    if df.empty:

        raise ValueError(
            "AIS dataset is empty."
        )

    df["base_date_time"] = pd.to_datetime(
        df["base_date_time"],
        utc=True
    )

    print(
        f"✓ AIS records loaded: "
        f"{len(df)}"
    )

    print(
        f"✓ Unique vessels: "
        f"{df['mmsi'].nunique()}"
    )

    return df


# ============================================================
# TEMPORAL FILTER
# ============================================================

def temporal_filter(
    ais_df,
    source_time
):
    """
    Keep AIS records close to the estimated
    spill-source time.
    """

    time_difference = (
        ais_df["base_date_time"]
        - source_time
    ).abs()

    filtered = ais_df[
        time_difference
        <= pd.Timedelta(
            hours=TEMPORAL_WINDOW_HOURS
        )
    ].copy()

    filtered[
        "time_difference_hours"
    ] = (
        time_difference.loc[
            filtered.index
        ]
        .dt.total_seconds()
        .abs()
        / 3600
    )

    return filtered


# ============================================================
# SPATIAL FILTER
# ============================================================

def spatial_filter(
    ais_df,
    origin_lat,
    origin_lon
):

    distances = []

    for _, row in ais_df.iterrows():

        distance = haversine_distance(
            origin_lat,
            origin_lon,
            float(row["latitude"]),
            float(row["longitude"])
        )

        distances.append(
            distance
        )

    filtered_df = ais_df.copy()

    filtered_df[
        "distance_from_origin_km"
    ] = distances

    filtered_df = filtered_df[
        filtered_df[
            "distance_from_origin_km"
        ]
        <= SPATIAL_RADIUS_KM
    ].copy()

    return filtered_df


# ============================================================
# PROXIMITY SCORE
# ============================================================

def calculate_proximity_score(
    distance_km
):

    score = (
        100
        * (
            1
            -
            distance_km
            / SPATIAL_RADIUS_KM
        )
    )

    return max(
        0,
        min(
            100,
            score
        )
    )


# ============================================================
# TEMPORAL SCORE
# ============================================================

def calculate_temporal_score(
    time_gap_hours
):

    score = (
        100
        * (
            1
            -
            time_gap_hours
            / TEMPORAL_WINDOW_HOURS
        )
    )

    return max(
        0,
        min(
            100,
            score
        )
    )


# ============================================================
# TRAJECTORY SCORE
# ============================================================

def calculate_trajectory_score(
    vessel_df,
    origin_lat,
    origin_lon
):

    if len(vessel_df) < 2:

        return 0.0

    vessel_df = (
        vessel_df
        .sort_values(
            "base_date_time"
        )
        .copy()
    )

    # --------------------------------------------------------
    # First and last position
    # --------------------------------------------------------

    first = vessel_df.iloc[0]

    last = vessel_df.iloc[-1]

    first_distance = haversine_distance(
        origin_lat,
        origin_lon,
        float(first["latitude"]),
        float(first["longitude"])
    )

    last_distance = haversine_distance(
        origin_lat,
        origin_lon,
        float(last["latitude"]),
        float(last["longitude"])
    )

    # --------------------------------------------------------
    # Approach component
    # --------------------------------------------------------

    if first_distance > 0:

        approach_ratio = (
            first_distance
            -
            last_distance
        ) / first_distance

    else:

        approach_ratio = 1.0

    approach_score = (
        50
        +
        approach_ratio * 50
    )

    approach_score = max(
        0,
        min(
            100,
            approach_score
        )
    )

    # --------------------------------------------------------
    # Closest approach
    # --------------------------------------------------------

    minimum_distance = (
        vessel_df[
            "distance_from_origin_km"
        ].min()
        if
        "distance_from_origin_km"
        in vessel_df.columns
        else
        vessel_df.apply(
            lambda row:
            haversine_distance(
                origin_lat,
                origin_lon,
                float(row["latitude"]),
                float(row["longitude"])
            ),
            axis=1
        ).min()
    )

    proximity_component = max(
        0,
        100
        * (
            1
            -
            minimum_distance
            / 20
        )
    )

    # --------------------------------------------------------
    # Final trajectory score
    # --------------------------------------------------------

    trajectory_score = (
        0.5 * approach_score
        +
        0.5 * proximity_component
    )

    return round(
        max(
            0,
            min(
                100,
                trajectory_score
            )
        ),
        2
    )


# ============================================================
# AIS ANOMALY SCORE
# ============================================================

def calculate_anomaly_score(
    vessel_df
):

    vessel_df = (
        vessel_df
        .sort_values(
            "base_date_time"
        )
    )

    if len(vessel_df) < 2:

        return 0.0

    timestamps = (
        vessel_df[
            "base_date_time"
        ]
        .sort_values()
    )

    time_gaps = (
        timestamps.diff()
        .dt.total_seconds()
        / 60
    )

    maximum_gap = (
        time_gaps.max()
    )

    if maximum_gap >= 30:

        return 100.0

    elif maximum_gap >= 15:

        return 70.0

    elif maximum_gap >= 5:

        return 40.0

    else:

        return 0.0


# ============================================================
# ENVIRONMENTAL CONSISTENCY
# ============================================================

def calculate_environmental_consistency(
    vessel_df,
    origin_lat,
    origin_lon,
    transport_direction
):
    """
    Compare the vessel's movement direction with
    the estimated environmental transport direction.

    This is supporting evidence only.
    It does NOT prove causation.
    """

    vessel_df = (
        vessel_df
        .sort_values(
            "base_date_time"
        )
        .copy()
    )

    if len(vessel_df) < 2:

        return 0.0

    bearings = []

    # Use consecutive AIS positions.
    for i in range(
        len(vessel_df) - 1
    ):

        current = vessel_df.iloc[i]

        next_point = vessel_df.iloc[i + 1]

        lat1 = float(
            current["latitude"]
        )

        lon1 = float(
            current["longitude"]
        )

        lat2 = float(
            next_point["latitude"]
        )

        lon2 = float(
            next_point["longitude"]
        )

        if (
            lat1 == lat2
            and
            lon1 == lon2
        ):
            continue

        bearing = calculate_bearing(
            lat1,
            lon1,
            lat2,
            lon2
        )

        bearings.append(
            bearing
        )

    if not bearings:

        return 0.0

    # Average directional agreement.
    scores = []

    for bearing in bearings:

        difference = angular_difference(
            bearing,
            transport_direction
        )

        score = (
            100
            * (
                1
                -
                difference
                / 180
            )
        )

        scores.append(
            max(
                0,
                min(
                    100,
                    score
                )
            )
        )

    return round(
        sum(scores) / len(scores),
        2
    )


# ============================================================
# FINAL EVIDENCE SCORE
# ============================================================

def calculate_final_score(
    proximity,
    temporal,
    trajectory,
    anomaly,
    environmental
):

    score = (

        proximity
        * PROXIMITY_WEIGHT

        +

        temporal
        * TEMPORAL_WEIGHT

        +

        trajectory
        * TRAJECTORY_WEIGHT

        +

        anomaly
        * ANOMALY_WEIGHT

        +

        environmental
        * ENVIRONMENTAL_WEIGHT
    )

    return round(
        score,
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 68)

    print(
        "       OCEANTRACEAI INDIAN AIS ATTRIBUTION"
    )

    print("=" * 68)

    # --------------------------------------------------------
    # STEP 1 - LOAD ORIGIN
    # --------------------------------------------------------

    print(
        "\nSTEP 1: INDIAN BACKTRACKED ORIGIN"
    )

    print(
        "-" * 66
    )

    (
        image_id,
        origin_lat,
        origin_lon,
        observation_time,
        source_time,
        transport_direction,
        transport_speed
    ) = load_origin()

    print(
        f"Image ID          : {image_id}"
    )

    print(
        f"Estimated origin  : "
        f"{origin_lat:.6f}, "
        f"{origin_lon:.6f}"
    )

    print(
        f"Observation UTC   : "
        f"{observation_time}"
    )

    print(
        f"Estimated source  : "
        f"{source_time}"
    )

    print(
        f"Transport speed   : "
        f"{transport_speed:.6f} m/s"
    )

    print(
        f"Transport bearing : "
        f"{transport_direction:.2f}° TOWARD"
    )

    # --------------------------------------------------------
    # STEP 2 - LOAD AIS
    # --------------------------------------------------------

    print(
        "\nSTEP 2: INDIAN AIS DATA"
    )

    print(
        "-" * 66
    )

    ais_df = load_ais()

    # --------------------------------------------------------
    # STEP 3 - TEMPORAL FILTER
    # --------------------------------------------------------

    print(
        "\nSTEP 3: TEMPORAL FILTER"
    )

    print(
        "-" * 66
    )

    print(
        f"Reference time: "
        f"{source_time}"
    )

    print(
        f"Time window: "
        f"±{TEMPORAL_WINDOW_HOURS} hours"
    )

    temporal_df = temporal_filter(
        ais_df,
        source_time
    )

    print(
        f"Records after temporal filtering: "
        f"{len(temporal_df)}"
    )

    # --------------------------------------------------------
    # STEP 4 - SPATIAL FILTER
    # --------------------------------------------------------

    print(
        "\nSTEP 4: SPATIAL FILTER"
    )

    print(
        "-" * 66
    )

    print(
        f"Source radius: "
        f"{SPATIAL_RADIUS_KM} km"
    )

    candidate_df = spatial_filter(
        temporal_df,
        origin_lat,
        origin_lon
    )

    print(
        f"Records after spatial filtering: "
        f"{len(candidate_df)}"
    )

    if candidate_df.empty:

        print(
            "\n⚠ No AIS candidates found."
        )

        print(
            "Try increasing the spatial or "
            "temporal window."
        )

        return

    print(
        f"Candidate vessels: "
        f"{candidate_df['mmsi'].nunique()}"
    )

    # --------------------------------------------------------
    # STEP 5 - SCORE EACH VESSEL
    # --------------------------------------------------------

    print(
        "\nSTEP 5: VESSEL EVIDENCE SCORING"
    )

    print(
        "-" * 66
    )

    results = []

    for mmsi, vessel_df in (
        candidate_df.groupby("mmsi")
    ):

        vessel_df = vessel_df.copy()

        vessel_name = str(
            vessel_df[
                "vessel_name"
            ].iloc[0]
        )

        flag = str(
            vessel_df[
                "flag"
            ].iloc[0]
            if
            "flag"
            in vessel_df.columns
            else
            "Unknown"
        )

        # ----------------------------------------------------
        # Closest point
        # ----------------------------------------------------

        closest_row = (
            vessel_df
            .sort_values(
                "distance_from_origin_km"
            )
            .iloc[0]
        )

        minimum_distance = float(
            closest_row[
                "distance_from_origin_km"
            ]
        )

        minimum_time_gap = float(
            vessel_df[
                "time_difference_hours"
            ].min()
        )

        # ----------------------------------------------------
        # Individual scores
        # ----------------------------------------------------

        proximity_score = (
            calculate_proximity_score(
                minimum_distance
            )
        )

        temporal_score = (
            calculate_temporal_score(
                minimum_time_gap
            )
        )

        trajectory_score = (
            calculate_trajectory_score(
                vessel_df,
                origin_lat,
                origin_lon
            )
        )

        anomaly_score = (
            calculate_anomaly_score(
                vessel_df
            )
        )

        environmental_score = (
            calculate_environmental_consistency(
                vessel_df,
                origin_lat,
                origin_lon,
                transport_direction
            )
        )

        final_score = (
            calculate_final_score(
                proximity_score,
                temporal_score,
                trajectory_score,
                anomaly_score,
                environmental_score
            )
        )

        results.append(
            {
                "mmsi": mmsi,

                "vessel_name":
                    vessel_name,

                "flag":
                    flag,

                "distance_from_origin_km":
                    round(
                        minimum_distance,
                        2
                    ),

                "time_gap_hours":
                    round(
                        minimum_time_gap,
                        2
                    ),

                "proximity_score":
                    round(
                        proximity_score,
                        2
                    ),

                "temporal_score":
                    round(
                        temporal_score,
                        2
                    ),

                "trajectory_score":
                    round(
                        trajectory_score,
                        2
                    ),

                "ais_anomaly_score":
                    round(
                        anomaly_score,
                        2
                    ),

                "environmental_consistency_score":
                    round(
                        environmental_score,
                        2
                    ),

                "final_evidence_score":
                    final_score
            }
        )

    # --------------------------------------------------------
    # STEP 6 - RANK
    # --------------------------------------------------------

    print(
        "\nSTEP 6: FINAL VESSEL RANKING"
    )

    print(
        "-" * 66
    )

    result_df = pd.DataFrame(
        results
    )

    result_df = (
        result_df
        .sort_values(
            "final_evidence_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    result_df["rank"] = (
        result_df.index + 1
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # PRINT RANKING
    # --------------------------------------------------------

    for _, row in result_df.iterrows():

        print(
            f"\nRank #{int(row['rank'])}"
        )

        print(
            f"Vessel          : "
            f"{row['vessel_name']}"
        )

        print(
            f"MMSI            : "
            f"{row['mmsi']}"
        )

        print(
            f"Flag            : "
            f"{row['flag']}"
        )

        print(
            f"Distance        : "
            f"{row['distance_from_origin_km']:.2f} km"
        )

        print(
            f"Time gap        : "
            f"{row['time_gap_hours']:.2f} h"
        )

        print(
            f"Proximity       : "
            f"{row['proximity_score']:.1f}"
        )

        print(
            f"Temporal        : "
            f"{row['temporal_score']:.1f}"
        )

        print(
            f"Trajectory      : "
            f"{row['trajectory_score']:.1f}"
        )

        print(
            f"AIS anomaly     : "
            f"{row['ais_anomaly_score']:.1f}"
        )

        print(
            f"Environmental   : "
            f"{row['environmental_consistency_score']:.1f}"
        )

        print(
            f"FINAL EVIDENCE  : "
            f"{row['final_evidence_score']:.1f}"
        )

        print(
            "-" * 66
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 68
    )

    print(
        "       AIS ATTRIBUTION COMPLETE"
    )

    print(
        "=" * 68
    )

    print(
        f"\nCandidate vessels: "
        f"{len(result_df)}"
    )

    print(
        f"Reference origin: "
        f"{origin_lat:.6f}, "
        f"{origin_lon:.6f}"
    )

    print(
        f"Reference source time: "
        f"{source_time}"
    )

    print(
        "\nFINAL SCORE COMPONENTS:"
    )

    print(
        "25% Proximity"
    )

    print(
        "15% Temporal alignment"
    )

    print(
        "30% Trajectory consistency"
    )

    print(
        "10% AIS behavioral anomaly"
    )

    print(
        "20% Environmental consistency"
    )

    print(
        "\nOutput:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "The ranking identifies vessels for "
        "investigation."
    )

    print(
        "It does NOT prove that a vessel caused "
        "the oil spill."
    )

    print(
        "\nAIS data used in this demonstration "
        "is SYNTHETIC."
    )

    print(
        "=" * 68
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()