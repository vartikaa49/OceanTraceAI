import subprocess
from pathlib import Path
from datetime import datetime
import math

import xarray as xr
import numpy as np
import pandas as pd


# ============================================================
# OCEANTRACEAI - INDIAN OCEAN CURRENT MODULE
# ============================================================

DATASET_ID = (
    "cmems_mod_glo_phy-cur_anfc_0.083deg_PT6H-i"
)

OUTPUT_DIR = Path(
    "data/environment"
)

SCENARIO_FILE = Path(
    "data/environment/indian_test_scenario.csv"
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
# GET OCEAN CURRENT
# ============================================================

def get_ocean_current(
    latitude,
    longitude,
    observation_time
):
    """
    Automatically downloads the nearest
    Copernicus Marine surface current.

    Returns:
        current_speed (m/s)
        current_direction (degrees)
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Copernicus data is available at 6-hour intervals.
    # Round observation time down to nearest 6-hour slot.
    # --------------------------------------------------------

    hour = (
        observation_time.hour // 6
    ) * 6

    current_time = observation_time.replace(
        hour=hour,
        minute=0,
        second=0,
        microsecond=0
    )


    timestamp = current_time.strftime(
        "%Y%m%d_%H%M"
    )

    output_name = (
        f"current_indian_{timestamp}"
    )

    output_file = (
        OUTPUT_DIR /
        f"{output_name}.nc"
    )


    print()
    print(
        "Fetching Copernicus Marine current..."
    )

    print(
        f"Location       : "
        f"{latitude:.6f}, "
        f"{longitude:.6f}"
    )

    print(
        f"Requested time : "
        f"{current_time}"
    )


    # --------------------------------------------------------
    # Small bounding box around spill.
    # --------------------------------------------------------

    min_lon = longitude - 0.10
    max_lon = longitude + 0.10

    min_lat = latitude - 0.10
    max_lat = latitude + 0.10


    # --------------------------------------------------------
    # Copernicus Marine subset command.
    # --------------------------------------------------------

    command = [

        "copernicusmarine",

        "subset",

        "--dataset-id",
        DATASET_ID,

        "--variable",
        "uo",

        "--variable",
        "vo",

        "--minimum-longitude",
        str(min_lon),

        "--maximum-longitude",
        str(max_lon),

        "--minimum-latitude",
        str(min_lat),

        "--maximum-latitude",
        str(max_lat),

        "--minimum-depth",
        "0",

        "--maximum-depth",
        "1",

        "--start-datetime",
        current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "--end-datetime",
        current_time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),

        "--output-directory",
        str(OUTPUT_DIR),

        "--output-filename",
        output_name,

        "--file-format",
        "netcdf",

        "--disable-progress-bar",
    ]


    # --------------------------------------------------------
    # Download data.
    # --------------------------------------------------------

    try:

        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError as e:

        raise RuntimeError(
            "Copernicus Marine download failed: "
            f"{e}"
        )


    if not output_file.exists():

        raise FileNotFoundError(
            "Expected current file was not created:\n"
            f"{output_file}"
        )


    print(
        "✓ Copernicus data downloaded."
    )

    print(
        f"File: {output_file}"
    )


    # --------------------------------------------------------
    # Open NetCDF.
    # --------------------------------------------------------

    ds = xr.open_dataset(
        output_file
    )


    uo = ds["uo"]
    vo = ds["vo"]


    # --------------------------------------------------------
    # Select nearest available grid location.
    # --------------------------------------------------------

    selected_uo = uo.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest"
    )

    selected_vo = vo.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest"
    )


    # --------------------------------------------------------
    # Convert to scalar.
    # --------------------------------------------------------

    u = float(
        selected_uo.values.squeeze()
    )

    v = float(
        selected_vo.values.squeeze()
    )


    if np.isnan(u) or np.isnan(v):

        ds.close()

        raise ValueError(
            "Copernicus returned NaN current "
            "values at the Indian Ocean location."
        )


    # --------------------------------------------------------
    # Current speed.
    # --------------------------------------------------------

    speed = math.sqrt(
        u ** 2 +
        v ** 2
    )


    # --------------------------------------------------------
    # Direction of current vector.
    #
    # u = eastward component
    # v = northward component
    #
    # Direction convention:
    # 0°   = North
    # 90°  = East
    # 180° = South
    # 270° = West
    # --------------------------------------------------------

    direction = math.degrees(
        math.atan2(
            u,
            v
        )
    )


    if direction < 0:

        direction += 360


    # --------------------------------------------------------
    # Actual Copernicus grid point selected.
    # --------------------------------------------------------

    selected_lat = float(
        selected_uo.latitude.values
    )

    selected_lon = float(
        selected_uo.longitude.values
    )


    print()
    print(
        "========== COPERNICUS CURRENT =========="
    )

    print(
        f"Nearest grid latitude  : "
        f"{selected_lat:.6f}"
    )

    print(
        f"Nearest grid longitude : "
        f"{selected_lon:.6f}"
    )

    print(
        f"Eastward current (u)   : "
        f"{u:.6f} m/s"
    )

    print(
        f"Northward current (v)  : "
        f"{v:.6f} m/s"
    )

    print(
        f"Current speed          : "
        f"{speed:.6f} m/s"
    )

    print(
        f"Current direction      : "
        f"{direction:.2f}°"
    )

    print(
        "========================================="
    )


    ds.close()


    return (
        speed,
        direction
    )


# ============================================================
# TEST INDIAN OCEAN CURRENT
# ============================================================

if __name__ == "__main__":

    print("=" * 68)
    print(
        "      OCEANTRACEAI INDIAN OCEAN CURRENT TEST"
    )
    print("=" * 68)


    try:

        (
            latitude,
            longitude,
            observation_time
        ) = load_indian_scenario()


        print()
        print("Scenario loaded:")

        print(
            f"Latitude  : "
            f"{latitude:.6f}"
        )

        print(
            f"Longitude : "
            f"{longitude:.6f}"
        )

        print(
            f"Time UTC  : "
            f"{observation_time}"
        )


        speed, direction = (
            get_ocean_current(
                latitude,
                longitude,
                observation_time
            )
        )


        print()
        print(
            "✓ REAL COPERNICUS CURRENT DATA RECEIVED"
        )


        print()
        print("=" * 68)
        print(
            "                  TEST SUCCESSFUL"
        )
        print("=" * 68)


    except Exception as e:

        print()
        print(
            f"ERROR: {e}"
        )

        print()
        print("=" * 68)
        print(
            "                   TEST FAILED"
        )
        print("=" * 68)