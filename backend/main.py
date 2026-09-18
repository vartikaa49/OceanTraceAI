from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse


# ============================================================
# OCEANTRACEAI - FASTAPI BACKEND
# SIH 26143
# INDIAN OCEAN DEMONSTRATION
# ============================================================

app = FastAPI(
    title="OceanTraceAI",
    description=(
        "Maritime Oil Spill Detection and "
        "Vessel Attribution System"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENVIRONMENT_DIR = (
    BASE_DIR / "data" / "environment"
)

AIS_DIR = (
    BASE_DIR / "data" / "ais"
)

OUTPUT_DIR = (
    BASE_DIR / "outputs"
)


# ------------------------------------------------------------
# Indian scenario files
# ------------------------------------------------------------

SPILL_FILE = (
    ENVIRONMENT_DIR /
    "indian_test_scenario.csv"
)

ORIGIN_FILE = (
    ENVIRONMENT_DIR /
    "indian_backtracked_origin.csv"
)

FORWARD_DRIFT_FILE = (
    ENVIRONMENT_DIR /
    "indian_forward_drift_forecast.csv"
)

AIS_ATTRIBUTION_FILE = (
    AIS_DIR /
    "ais_indian_attribution.csv"
)

AIS_TRACK_FILE = (
    AIS_DIR /
    "synthetic_indian_ocean_ais.csv"
)


# ============================================================
# ML VALIDATION OUTPUT
# ============================================================

VALIDATION_IMAGE_ID = "01338"

VALIDATION_IMAGE_FILE = (
    OUTPUT_DIR /
    "01338_result.png"
)

VALIDATION_PREDICTION_FILE = (
    OUTPUT_DIR /
    "01338_prediction.tif"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value, default=None):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return default


def safe_int(value, default=None):

    try:

        if value is None:
            return default

        if pd.isna(value):
            return default

        return int(value)

    except (
        ValueError,
        TypeError
    ):

        return default


def read_csv_safe(path):

    try:

        if not path.exists():

            print(
                f"WARNING: File not found: {path}"
            )

            return pd.DataFrame()

        return pd.read_csv(path)

    except Exception as e:

        print(
            f"ERROR reading {path}: {e}"
        )

        return pd.DataFrame()


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "system":
            "OceanTraceAI",

        "status":
            "OPERATIONAL",

        "problem_statement":
            "SIH 26143",

        "deployment":
            "Indian Ocean demonstration",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status":
            "healthy",

        "system":
            "OceanTraceAI",

        "problem_statement":
            "SIH 26143",
    }


# ============================================================
# LOAD SPILL SCENARIO
# ============================================================

def load_spill_scenario():

    df = read_csv_safe(
        SPILL_FILE
    )

    if df.empty:

        return None

    return df.iloc[0].to_dict()


# ============================================================
# LOAD BACKTRACKED ORIGIN
# ============================================================

def load_origin():

    df = read_csv_safe(
        ORIGIN_FILE
    )

    if df.empty:

        return None

    return df.iloc[0].to_dict()


# ============================================================
# LOAD FORWARD DRIFT
# ============================================================

def load_forward_drift():

    df = read_csv_safe(
        FORWARD_DRIFT_FILE
    )

    if df.empty:

        return []

    forecast = []

    for _, row in df.iterrows():

        forecast.append({

            "image_id":
                (
                    str(
                        row.get(
                            "image_id"
                        )
                    )
                    if row.get(
                        "image_id"
                    ) is not None
                    else None
                ),

            "forecast_hours":
                safe_float(
                    row.get(
                        "forecast_hours"
                    )
                ),

            "latitude":
                safe_float(
                    row.get(
                        "predicted_latitude"
                    )
                ),

            "longitude":
                safe_float(
                    row.get(
                        "predicted_longitude"
                    )
                ),

            "transport_speed_ms":
                safe_float(
                    row.get(
                        "resultant_speed_ms"
                    )
                ),

            "transport_direction_deg":
                safe_float(
                    row.get(
                        "resultant_direction_deg"
                    )
                ),

            "uncertainty_radius_km":
                safe_float(
                    row.get(
                        "uncertainty_radius_km"
                    )
                ),
        })

    return forecast


# ============================================================
# LOAD AIS ATTRIBUTION
# ============================================================

def load_ais_attribution():

    df = read_csv_safe(
        AIS_ATTRIBUTION_FILE
    )

    if df.empty:

        return []

    candidates = []

    for _, row in df.iterrows():

        candidate = {

            "rank":
                safe_int(
                    row.get(
                        "rank"
                    )
                ),

            "mmsi":
                (
                    str(
                        row.get(
                            "mmsi"
                        )
                    )
                    if row.get(
                        "mmsi"
                    ) is not None
                    else None
                ),

            "vessel_name":
                (
                    str(
                        row.get(
                            "vessel_name"
                        )
                    )
                    if row.get(
                        "vessel_name"
                    ) is not None
                    else None
                ),

            "flag":
                (
                    str(
                        row.get(
                            "flag"
                        )
                    )
                    if row.get(
                        "flag"
                    ) is not None
                    else None
                ),

            "distance_km":
                safe_float(
                    row.get(
                        "distance_from_origin_km"
                    )
                ),

            "time_gap_hours":
                safe_float(
                    row.get(
                        "time_gap_hours"
                    )
                ),

            "proximity_score":
                safe_float(
                    row.get(
                        "proximity_score"
                    )
                ),

            "temporal_score":
                safe_float(
                    row.get(
                        "temporal_score"
                    )
                ),

            "trajectory_score":
                safe_float(
                    row.get(
                        "trajectory_score"
                    )
                ),

            "ais_anomaly_score":
                safe_float(
                    row.get(
                        "ais_anomaly_score"
                    )
                ),

            "environmental_score":
                safe_float(
                    row.get(
                        "environmental_consistency_score"
                    )
                ),

            "final_evidence_score":
                safe_float(
                    row.get(
                        "final_evidence_score"
                    )
                ),
        }

        candidates.append(
            candidate
        )


    # Sort by evidence score

    candidates.sort(

        key=lambda x:
            (
                x[
                    "final_evidence_score"
                ]
                if x[
                    "final_evidence_score"
                ] is not None
                else -1
            ),

        reverse=True,
    )


    # Reassign ranks

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        candidate["rank"] = index


    return candidates


# ============================================================
# LOAD AIS TRACKS
# ============================================================

def load_ais_tracks():

    df = read_csv_safe(
        AIS_TRACK_FILE
    )

    if df.empty:

        return []

    mmsi_column = None

    for column in [
        "mmsi",
        "MMSI",
    ]:

        if column in df.columns:

            mmsi_column = column

            break

    if mmsi_column is None:

        return []

    tracks = []

    for mmsi, group in df.groupby(
        mmsi_column
    ):

        group = group.copy()

        coordinates = []

        for _, row in group.iterrows():

            latitude = safe_float(
                row.get(
                    "latitude"
                )
            )

            longitude = safe_float(
                row.get(
                    "longitude"
                )
            )

            if (
                latitude is None
                or longitude is None
            ):

                continue

            coordinates.append(
                [
                    latitude,
                    longitude,
                ]
            )


        if not coordinates:

            continue


        first_row = group.iloc[0]

        vessel_name = first_row.get(
            "vessel_name",
            "Unknown Vessel"
        )

        flag = first_row.get(
            "flag",
            "Unknown"
        )


        tracks.append({

            "mmsi":
                str(mmsi),

            "vessel_name":
                str(vessel_name),

            "flag":
                str(flag),

            "coordinates":
                coordinates,
        })


    return tracks


# ============================================================
# MAIN DASHBOARD API
# ============================================================

@app.get("/api/dashboard")
def dashboard():

    spill = load_spill_scenario()

    origin = load_origin()

    candidates = load_ais_attribution()

    forecast = load_forward_drift()


    if spill is None:

        spill = {}


    if origin is None:

        origin = {}


    # --------------------------------------------------------
    # SPILL
    # --------------------------------------------------------

    image_id = spill.get(
        "image_id",
        "INDIA_TEST_001"
    )

    spill_latitude = safe_float(
        spill.get(
            "spill_latitude"
        )
    )

    spill_longitude = safe_float(
        spill.get(
            "spill_longitude"
        )
    )

    spill_percentage = safe_float(
        spill.get(
            "spill_percentage"
        )
    )


    # --------------------------------------------------------
    # ORIGIN
    # --------------------------------------------------------

    origin_latitude = safe_float(
        origin.get(
            "origin_latitude"
        )
    )

    origin_longitude = safe_float(
        origin.get(
            "origin_longitude"
        )
    )

    observation_time = origin.get(
        "observation_time_utc"
    )

    source_time = origin.get(
        "estimated_source_time_utc"
    )

    backtrack_hours = safe_float(
        origin.get(
            "backtrack_hours"
        )
    )


    # --------------------------------------------------------
    # ENVIRONMENT
    # --------------------------------------------------------

    wind_speed = safe_float(
        origin.get(
            "wind_speed_ms"
        )
    )

    wind_direction = safe_float(
        origin.get(
            "wind_direction_from_deg"
        )
    )

    current_speed = safe_float(
        origin.get(
            "current_speed_ms"
        )
    )

    current_direction = safe_float(
        origin.get(
            "current_direction_toward_deg"
        )
    )

    transport_speed = safe_float(
        origin.get(
            "transport_speed_ms"
        )
    )

    transport_direction = safe_float(
        origin.get(
            "transport_direction_toward_deg"
        )
    )


    # --------------------------------------------------------
    # TOP AIS CANDIDATE
    # --------------------------------------------------------

    top_candidate = (

        candidates[0]

        if candidates

        else None
    )


    # --------------------------------------------------------
    # FORWARD FORECAST SUMMARY
    # --------------------------------------------------------

    final_forecast = (

        forecast[-1]

        if forecast

        else None
    )


    # --------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------

    return {

        "system": {

            "name":
                "OceanTraceAI",

            "status":
                "OPERATIONAL",

            "problem_statement":
                "SIH 26143",

            "deployment":
                "Indian Ocean demonstration",
        },


        "spill": {

            "detected":
                True,

            "image_id":
                image_id,

            "latitude":
                spill_latitude,

            "longitude":
                spill_longitude,

            "spill_percentage":
                spill_percentage,
        },


        "origin": {

            "latitude":
                origin_latitude,

            "longitude":
                origin_longitude,

            "observation_time_utc":
                observation_time,

            "source_time_utc":
                source_time,

            "backtrack_hours":
                backtrack_hours,
        },


        "environment": {

            "wind_speed_ms":
                wind_speed,

            "wind_direction_deg":
                wind_direction,

            "wind_direction_type":
                "FROM",

            "current_speed_ms":
                current_speed,

            "current_direction_deg":
                current_direction,

            "current_direction_type":
                "TOWARD",

            "resultant_speed_ms":
                transport_speed,

            "resultant_direction_deg":
                transport_direction,

            "resultant_direction_type":
                "TOWARD",
        },


        "forward_drift": {

            "available":
                len(forecast) > 0,

            "forecast_count":
                len(forecast),

            "forecast":
                forecast,

            "final_forecast":
                final_forecast,
        },


        "ais": {

            "candidate_count":
                len(candidates),

            "top_candidate":
                top_candidate,

            "candidates":
                candidates,
        },
    }


# ============================================================
# SPILL AREA
# ============================================================

@app.get("/api/spill-area")
def spill_area():

    prediction_file = (
        OUTPUT_DIR /
        "01338_prediction.tif"
    )


    if not prediction_file.exists():

        return {

            "available":
                False,

            "message":
                "Prediction file not found.",
        }


    try:

        import rasterio


        with rasterio.open(
            prediction_file
        ) as src:

            mask = src.read(1)

            spill_pixels = int(
                (mask > 0).sum()
            )

            total_pixels = mask.size

            percentage = (

                spill_pixels /
                total_pixels *
                100

                if total_pixels > 0

                else 0
            )


        return {

            "available":
                True,

            "image_id":
                VALIDATION_IMAGE_ID,

            "spill_pixels":
                spill_pixels,

            "total_pixels":
                total_pixels,

            "spill_percentage":
                round(
                    percentage,
                    4
                ),

            "note":
                (
                    "ML validation example "
                    "01338, not the live Indian "
                    "scenario."
                ),
        }


    except Exception as e:

        return {

            "available":
                False,

            "error":
                str(e),
        }


# ============================================================
# SATELLITE IMAGE
# ============================================================

@app.get("/api/satellite-image")
def satellite_image():

    if not VALIDATION_IMAGE_FILE.exists():

        return JSONResponse(

            status_code=404,

            content={

                "available":
                    False,

                "message":
                    "Satellite visualization not found.",
            },
        )


    return FileResponse(

        VALIDATION_IMAGE_FILE,

        media_type="image/png",
    )


# ============================================================
# SATELLITE METADATA
# ============================================================

@app.get("/api/satellite-metadata")
def satellite_metadata():

    return {

        "available":
            True,

        "image_id":
            VALIDATION_IMAGE_ID,

        "type":
            "Sentinel-1 SAR ML validation visualization",

        "model":
            "U-Net",

        "input_channels":
            2,

        "channels": [
            "VH",
            "VV",
        ],

        "training_channel_order":
            "VH, VV",

        "patch_size":
            256,

        "note":
            (
                "This visualization is from "
                "the trained Sentinel-1 oil-spill "
                "segmentation validation example. "
                "It is separate from the Indian "
                "Ocean environmental demonstration."
            ),
    }


# ============================================================
# VESSEL TRACKS
# ============================================================

@app.get("/api/tracks")
def vessel_tracks():

    tracks = load_ais_tracks()

    return {

        "count":
            len(tracks),

        "tracks":
            tracks,
    }


# ============================================================
# SPILL FOOTPRINT
# ============================================================

@app.get("/api/spill-footprint")
def spill_footprint():

    spill = load_spill_scenario()


    if spill is None:

        return {

            "available":
                False,

            "polygons":
                [],
        }


    latitude = safe_float(
        spill.get(
            "spill_latitude"
        )
    )

    longitude = safe_float(
        spill.get(
            "spill_longitude"
        )
    )


    if (
        latitude is None
        or longitude is None
    ):

        return {

            "available":
                False,

            "polygons":
                [],
        }


    # --------------------------------------------------------
    # Demonstration visualization footprint
    #
    # This is only for dashboard visualization.
    # It is NOT claiming that the ML mask on the
    # real SAFE product has this exact geometry.
    # --------------------------------------------------------

    lat_offset = 0.025

    lon_offset = 0.035


    polygon = [

        [
            latitude + lat_offset,
            longitude - lon_offset,
        ],

        [
            latitude + lat_offset * 0.5,
            longitude + lon_offset,
        ],

        [
            latitude - lat_offset,
            longitude + lon_offset * 0.7,
        ],

        [
            latitude - lat_offset * 0.7,
            longitude - lon_offset,
        ],

        [
            latitude + lat_offset,
            longitude - lon_offset,
        ],
    ]


    return {

        "available":
            True,

        "type":
            "demonstration_footprint",

        "polygons": [

            {

                "name":
                    "Predicted Spill Footprint",

                "coordinates":
                    polygon,
            }
        ],

        "note":
            (
                "Visualization footprint for "
                "the Indian demonstration scenario."
            ),
    }


# ============================================================
# FORWARD DRIFT FORECAST
# ============================================================

@app.get("/api/forward-drift")
def forward_drift():

    forecast = load_forward_drift()


    if not forecast:

        return {

            "available":
                False,

            "scenario":
                "Indian Ocean",

            "message":
                (
                    "Indian forward drift "
                    "forecast unavailable."
                ),

            "forecast":
                [],
        }


    start_point = forecast[0]

    final_point = forecast[-1]


    return {

        "available":
            True,

        "scenario":
            "Indian Ocean",

        "model":
            "Environmental drift projection",

        "description":
            (
                "Forward spill trajectory estimated "
                "using Copernicus ocean current and "
                "wind-driven surface transport."
            ),

        "start_position": {

            "latitude":
                start_point["latitude"],

            "longitude":
                start_point["longitude"],
        },

        "final_forecast_position": {

            "latitude":
                final_point["latitude"],

            "longitude":
                final_point["longitude"],
        },

        "forecast_points":
            forecast,

        "forecast_hours": [

            point["forecast_hours"]

            for point in forecast
        ],
    }


# ============================================================
# INVESTIGATION
# ============================================================

@app.get("/api/investigation")
def investigation():

    spill = load_spill_scenario()

    origin = load_origin()

    candidates = load_ais_attribution()

    forecast = load_forward_drift()


    top_candidate = (

        candidates[0]

        if candidates

        else None
    )


    return {

        "system":
            "OceanTraceAI",

        "problem_statement":
            "SIH 26143",

        "scenario":
            "Indian Ocean",


        "spill": {

            "image_id": (

                spill.get(
                    "image_id"
                )

                if spill

                else None
            ),

            "latitude": (

                safe_float(
                    spill.get(
                        "spill_latitude"
                    )
                )

                if spill

                else None
            ),

            "longitude": (

                safe_float(
                    spill.get(
                        "spill_longitude"
                    )
                )

                if spill

                else None
            ),
        },


        "backtracking": {

            "source_latitude": (

                safe_float(
                    origin.get(
                        "origin_latitude"
                    )
                )

                if origin

                else None
            ),

            "source_longitude": (

                safe_float(
                    origin.get(
                        "origin_longitude"
                    )
                )

                if origin

                else None
            ),

            "source_time_utc": (

                origin.get(
                    "estimated_source_time_utc"
                )

                if origin

                else None
            ),

            "backtrack_hours": (

                safe_float(
                    origin.get(
                        "backtrack_hours"
                    )
                )

                if origin

                else None
            ),
        },


        "environment": {

            "wind_speed_ms": (

                safe_float(
                    origin.get(
                        "wind_speed_ms"
                    )
                )

                if origin

                else None
            ),

            "wind_direction_from_deg": (

                safe_float(
                    origin.get(
                        "wind_direction_from_deg"
                    )
                )

                if origin

                else None
            ),

            "current_speed_ms": (

                safe_float(
                    origin.get(
                        "current_speed_ms"
                    )
                )

                if origin

                else None
            ),

            "current_direction_toward_deg": (

                safe_float(
                    origin.get(
                        "current_direction_toward_deg"
                    )
                )

                if origin

                else None
            ),

            "transport_speed_ms": (

                safe_float(
                    origin.get(
                        "transport_speed_ms"
                    )
                )

                if origin

                else None
            ),

            "transport_direction_toward_deg": (

                safe_float(
                    origin.get(
                        "transport_direction_toward_deg"
                    )
                )

                if origin

                else None
            ),
        },


        "forward_drift": {

            "available":
                len(forecast) > 0,

            "forecast":
                forecast,
        },


        "ais": {

            "candidate_count":
                len(candidates),

            "top_candidate":
                top_candidate,
        },
    }


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():

    print()

    print("=" * 60)

    print(
        "OceanTraceAI Backend"
    )

    print(
        "SIH Problem Statement 26143"
    )

    print(
        "Indian Ocean Demonstration"
    )

    print("=" * 60)

    print(
        f"Scenario file: {SPILL_FILE}"
    )

    print(
        f"Origin file:   {ORIGIN_FILE}"
    )

    print(
        f"Forward drift: {FORWARD_DRIFT_FILE}"
    )

    print(
        f"AIS file:      {AIS_ATTRIBUTION_FILE}"
    )

    print(
        f"Tracks file:   {AIS_TRACK_FILE}"
    )

    print("=" * 60)

    print()