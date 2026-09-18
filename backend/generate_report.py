from pathlib import Path
from datetime import datetime
import pandas as pd


# ============================================================
# OCEANTRACEAI - INVESTIGATION REPORT GENERATOR
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPILL_FILE = (
    PROJECT_ROOT
    / "data"
    / "environment"
    / "detected_spill_location.csv"
)

ORIGIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "environment"
    / "backtracked_origin.csv"
)

AIS_FILE = (
    PROJECT_ROOT
    / "data"
    / "ais"
    / "ais_filtered_candidates.csv"
)

ENV_FILE = (
    PROJECT_ROOT
    / "data"
    / "ais"
    / "ais_environmental_scores.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "reports"

OUTPUT_FILE = (
    OUTPUT_DIR
    / "OceanTraceAI_Investigation_Report.html"
)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading investigation data...")

    spill = pd.read_csv(SPILL_FILE)
    origin = pd.read_csv(ORIGIN_FILE)
    ais = pd.read_csv(AIS_FILE)
    env = pd.read_csv(ENV_FILE)

    return spill, origin, ais, env


# ============================================================
# FORMAT HELPERS
# ============================================================

def fmt(value, digits=2):

    try:
        return f"{float(value):.{digits}f}"

    except (ValueError, TypeError):

        return "N/A"


def score_class(score):

    score = float(score)

    if score >= 80:
        return "high"

    elif score >= 60:
        return "medium"

    return "low"


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report():

    print("=" * 70)
    print("        OCEANTRACEAI INVESTIGATION REPORT GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    spill, origin, ais, env = load_data()

    if spill.empty:
        raise ValueError(
            "detected_spill_location.csv is empty."
        )

    if origin.empty:
        raise ValueError(
            "backtracked_origin.csv is empty."
        )

    if ais.empty:
        raise ValueError(
            "ais_filtered_candidates.csv is empty."
        )

    # --------------------------------------------------------
    # SPILL INFORMATION
    # --------------------------------------------------------

    spill_row = spill.iloc[0]

    origin_row = origin.iloc[0]

    # Keep leading zero in image ID
    image_id = str(
        spill_row["image_id"]
    ).split(".")[0].zfill(5)

    spill_lat = float(
        spill_row["latitude"]
    )

    spill_lon = float(
        spill_row["longitude"]
    )

    # --------------------------------------------------------
    # BACKTRACKED ORIGIN
    # --------------------------------------------------------

    origin_lat = float(
        origin_row["origin_latitude"]
    )

    origin_lon = float(
        origin_row["origin_longitude"]
    )

    # Correct column names from backtracked_origin.csv
    observation_time = str(
        origin_row["observation_time_utc"]
    )

    wind_speed = float(
        origin_row["wind_speed_ms"]
    )

    wind_direction = float(
        origin_row["wind_direction_deg"]
    )

    current_speed = float(
        origin_row["current_speed_ms"]
    )

    current_direction = float(
        origin_row["current_direction_deg"]
    )

    backtrack_hours = float(
        origin_row["backtrack_hours"]
    )

    # --------------------------------------------------------
    # AIS RANKING
    # --------------------------------------------------------

    ais = ais.sort_values(
        "rank"
    ).reset_index(drop=True)

    top = ais.iloc[0]

    top_vessel = str(
        top["vessel_name"]
    )

    top_mmsi = str(
        top["mmsi"]
    ).split(".")[0]

    top_flag = str(
        top["flag"]
    )

    top_score = float(
        top["final_evidence_score"]
    )

    # --------------------------------------------------------
    # CREATE AIS TABLE
    # --------------------------------------------------------

    ais_rows = ""

    for _, row in ais.iterrows():

        rank = int(
            row["rank"]
        )

        vessel = str(
            row["vessel_name"]
        )

        mmsi = str(
            row["mmsi"]
        ).split(".")[0]

        flag = str(
            row["flag"]
        )

        final_score = float(
            row["final_evidence_score"]
        )

        ais_rows += f"""

        <tr>

            <td>
                <span class="rank">
                    #{rank}
                </span>
            </td>

            <td>
                <strong>
                    {vessel}
                </strong>
            </td>

            <td>
                {mmsi}
            </td>

            <td>
                {flag}
            </td>

            <td>
                {fmt(row["distance_from_origin_km"])} km
            </td>

            <td>
                {fmt(row["time_gap_hours"])} h
            </td>

            <td>
                {fmt(row["proximity_score"])}%
            </td>

            <td>
                {fmt(row["temporal_score"])}%
            </td>

            <td>
                {fmt(row["trajectory_score"])}%
            </td>

            <td>
                {fmt(row["ais_anomaly_score"])}%
            </td>

            <td>
                {fmt(row["environmental_consistency_score"])}%
            </td>

            <td>

                <div class="score-cell {score_class(final_score)}">

                    {fmt(final_score)}%

                </div>

            </td>

        </tr>

        """

    # ========================================================
    # HTML
    # ========================================================

    html = f"""

<!DOCTYPE html>

<html lang="en">

<head>

<meta charset="UTF-8">

<meta name="viewport"
      content="width=device-width, initial-scale=1.0">

<title>
    OceanTraceAI Investigation Report
</title>


<style>

/* ==========================================================
   GLOBAL
   ========================================================== */

* {{
    box-sizing: border-box;
}}


body {{

    margin: 0;

    padding: 0;

    background:

        radial-gradient(
            circle at top right,
            rgba(0, 180, 255, 0.08),
            transparent 35%
        ),

        #07111f;

    color: #d9e7f5;

    font-family:
        Arial,
        Helvetica,
        sans-serif;

}}


/* ==========================================================
   CONTAINER
   ========================================================== */

.container {{

    width: 94%;

    max-width: 1500px;

    margin: auto;

    padding:
        35px
        0
        60px;

}}


/* ==========================================================
   HEADER
   ========================================================== */

.header {{

    border-bottom:
        1px solid #1d405c;

    padding-bottom:
        25px;

    margin-bottom:
        30px;

}}


.logo {{

    font-size:
        14px;

    letter-spacing:
        4px;

    color:
        #35c6ff;

    font-weight:
        bold;

}}


h1 {{

    margin:
        8px 0;

    font-size:
        38px;

    color:
        white;

}}


.subtitle {{

    color:
        #7f9bb5;

    font-size:
        15px;

}}


/* ==========================================================
   GRID
   ========================================================== */

.grid {{

    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                250px,
                1fr
            )
        );

    gap:
        18px;

    margin-bottom:
        25px;

}}


/* ==========================================================
   CARDS
   ========================================================== */

.card {{

    background:
        rgba(
            12,
            29,
            47,
            0.92
        );

    border:
        1px solid #1d405c;

    border-radius:
        12px;

    padding:
        22px;

    box-shadow:
        0 10px 30px
        rgba(
            0,
            0,
            0,
            0.25
        );

}}


.card-title {{

    color:
        #7292aa;

    font-size:
        12px;

    text-transform:
        uppercase;

    letter-spacing:
        2px;

    margin-bottom:
        10px;

}}


.value {{

    font-size:
        25px;

    font-weight:
        bold;

    color:
        white;

}}


.coordinates {{

    color:
        #35c6ff;

    font-family:
        monospace;

    font-size:
        16px;

}}


/* ==========================================================
   SECTIONS
   ========================================================== */

.section {{

    margin-top:
        30px;

    margin-bottom:
        30px;

}}


.section h2 {{

    font-size:
        21px;

    color:
        white;

    border-left:
        4px solid #35c6ff;

    padding-left:
        12px;

}}


.section p {{

    color:
        #9db3c7;

    line-height:
        1.7;

}}


/* ==========================================================
   INVESTIGATION NOTICE
   ========================================================== */

.alert {{

    padding:
        20px;

    border-radius:
        10px;

    background:
        rgba(
            255,
            174,
            0,
            0.08
        );

    border:
        1px solid
        rgba(
            255,
            174,
            0,
            0.35
        );

    color:
        #ffd98a;

    margin:
        20px 0;

    line-height:
        1.6;

}}


/* ==========================================================
   TOP SUSPECT
   ========================================================== */

.suspect {{

    background:

        linear-gradient(
            135deg,
            rgba(
                0,
                190,
                255,
                0.12
            ),
            rgba(
                0,
                80,
                130,
                0.08
            )
        );

    border:
        1px solid #227ca8;

    border-radius:
        14px;

    padding:
        28px;

    margin:
        25px 0;

}}


.suspect-name {{

    font-size:
        30px;

    color:
        white;

    font-weight:
        bold;

    margin-bottom:
        8px;

}}


.suspect-meta {{

    color:
        #83a9c2;

    font-family:
        monospace;

}}


.suspect-score {{

    margin-top:
        20px;

    font-size:
        42px;

    font-weight:
        bold;

    color:
        #35c6ff;

}}


/* ==========================================================
   TABLE
   ========================================================== */

table {{

    width:
        100%;

    border-collapse:
        collapse;

    font-size:
        13px;

}}


th {{

    background:
        #0d2238;

    color:
        #86a8c0;

    padding:
        13px 10px;

    text-align:
        left;

    white-space:
        nowrap;

}}


td {{

    padding:
        13px 10px;

    border-bottom:
        1px solid #19334b;

    white-space:
        nowrap;

}}


tr:hover {{

    background:
        rgba(
            35,
            160,
            220,
            0.05
        );

}}


.rank {{

    color:
        #35c6ff;

    font-weight:
        bold;

}}


/* ==========================================================
   SCORE BADGES
   ========================================================== */

.score-cell {{

    font-weight:
        bold;

    padding:
        5px 9px;

    border-radius:
        5px;

    display:
        inline-block;

}}


.score-cell.high {{

    color:
        #7dffbc;

    background:
        rgba(
            30,
            200,
            120,
            0.12
        );

}}


.score-cell.medium {{

    color:
        #ffd477;

    background:
        rgba(
            255,
            190,
            50,
            0.12
        );

}}


.score-cell.low {{

    color:
        #ff9b9b;

    background:
        rgba(
            255,
            70,
            70,
            0.12
        );

}}


/* ==========================================================
   PIPELINE
   ========================================================== */

.method {{

    display:
        grid;

    grid-template-columns:
        repeat(
            auto-fit,
            minmax(
                200px,
                1fr
            )
        );

    gap:
        15px;

}}


.method-card {{

    background:
        #0b1c2d;

    border:
        1px solid #17364f;

    border-radius:
        10px;

    padding:
        18px;

}}


.method-card strong {{

    display:
        block;

    color:
        #35c6ff;

    margin-bottom:
        7px;

}}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {{

    margin-top:
        50px;

    padding-top:
        20px;

    border-top:
        1px solid #1d405c;

    color:
        #617d94;

    font-size:
        12px;

    line-height:
        1.6;

}}


/* ==========================================================
   MOBILE
   ========================================================== */

@media(max-width: 900px) {{

    .container {{

        width:
            96%;

    }}

    h1 {{

        font-size:
            28px;

    }}

    table {{

        display:
            block;

        overflow-x:
            auto;

    }}

}}

</style>

</head>


<body>


<div class="container">


<!-- ======================================================
     HEADER
     ====================================================== -->

<div class="header">

    <div class="logo">
        OCEANTRACEAI
    </div>

    <h1>
        Maritime Oil Spill Investigation Report
    </h1>

    <div class="subtitle">

        Satellite Detection
        •
        Environmental Backtracking
        •
        AIS Attribution

    </div>

</div>


<!-- ======================================================
     OVERVIEW
     ====================================================== -->

<div class="grid">


    <div class="card">

        <div class="card-title">
            Satellite Observation
        </div>

        <div class="value">
            {image_id}
        </div>

    </div>


    <div class="card">

        <div class="card-title">
            Spill Location
        </div>

        <div class="coordinates">

            {spill_lat:.6f},
            {spill_lon:.6f}

        </div>

    </div>


    <div class="card">

        <div class="card-title">
            Estimated Source
        </div>

        <div class="coordinates">

            {origin_lat:.6f},
            {origin_lon:.6f}

        </div>

    </div>


    <div class="card">

        <div class="card-title">
            Investigation Time
        </div>

        <div
            class="value"
            style="font-size:17px;"
        >

            {observation_time}

        </div>

    </div>


</div>


<!-- ======================================================
     TOP CANDIDATE
     ====================================================== -->

<div class="section">

    <h2>
        Highest-Ranked Vessel Candidate
    </h2>


    <div class="suspect">

        <div class="suspect-name">

            {top_vessel}

        </div>


        <div class="suspect-meta">

            MMSI:
            {top_mmsi}

            &nbsp; | &nbsp;

            Flag:
            {top_flag}

        </div>


        <div class="suspect-score">

            {top_score:.2f}%

        </div>


        <div style="color:#7897ad;">

            Final evidence score

        </div>


    </div>

</div>


<!-- ======================================================
     DISCLAIMER
     ====================================================== -->

<div class="alert">

    <strong>
        Investigation Notice
    </strong>

    <br>
    <br>

    The AIS dataset used in this demonstration is
    <strong>
        synthetic AIS data
    </strong>
    generated for the OceanTraceAI prototype.

    The highest-ranked vessel is therefore an
    <strong>
        investigative candidate, not proof of responsibility.
    </strong>

    <br>
    <br>

    In a real deployment, the same pipeline would operate on
    authenticated historical or live AIS feeds.

</div>


<!-- ======================================================
     ENVIRONMENTAL BACKTRACKING
     ====================================================== -->

<div class="section">

    <h2>
        Environmental Backtracking
    </h2>


    <p>

        OceanTraceAI combines the detected spill location
        with historical wind and ocean-current information
        to estimate a probable source location using
        backward drift modelling.

    </p>


    <div class="grid">


        <div class="card">

            <div class="card-title">
                Wind Speed
            </div>

            <div class="value">

                {wind_speed:.2f} m/s

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Wind Direction
            </div>

            <div class="value">

                {wind_direction:.0f}°

            </div>

            <div class="subtitle">

                Meteorological FROM direction

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Ocean Current
            </div>

            <div class="value">

                {current_speed:.3f} m/s

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Current Direction
            </div>

            <div class="value">

                {current_direction:.2f}°

            </div>

            <div class="subtitle">

                Direction of movement

            </div>

        </div>


    </div>


    <div class="grid">


        <div class="card">

            <div class="card-title">
                Backtrack Duration
            </div>

            <div class="value">

                {backtrack_hours:.0f} hours

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Estimated Origin Latitude
            </div>

            <div class="coordinates">

                {origin_lat:.6f}°

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Estimated Origin Longitude
            </div>

            <div class="coordinates">

                {origin_lon:.6f}°

            </div>

        </div>


        <div class="card">

            <div class="card-title">
                Spill Observation
            </div>

            <div class="coordinates">

                {spill_lat:.6f}°,
                {spill_lon:.6f}°

            </div>

        </div>


    </div>

</div>


<!-- ======================================================
     AIS RANKING
     ====================================================== -->

<div class="section">

    <h2>
        AIS Candidate Ranking
    </h2>


    <p>

        Candidate vessels are filtered using temporal and
        spatial constraints around the estimated spill source
        and then scored using proximity, temporal alignment,
        trajectory behaviour, AIS anomalies and environmental
        consistency.

    </p>


    <div style="overflow-x:auto;">

        <table>


            <thead>

                <tr>

                    <th>
                        Rank
                    </th>

                    <th>
                        Vessel
                    </th>

                    <th>
                        MMSI
                    </th>

                    <th>
                        Flag
                    </th>

                    <th>
                        Distance
                    </th>

                    <th>
                        Time Gap
                    </th>

                    <th>
                        Proximity
                    </th>

                    <th>
                        Temporal
                    </th>

                    <th>
                        Trajectory
                    </th>

                    <th>
                        AIS Anomaly
                    </th>

                    <th>
                        Environmental
                    </th>

                    <th>
                        Final Evidence
                    </th>

                </tr>

            </thead>


            <tbody>

                {ais_rows}

            </tbody>


        </table>

    </div>

</div>


<!-- ======================================================
     PIPELINE
     ====================================================== -->

<div class="section">

    <h2>
        OceanTraceAI Investigation Pipeline
    </h2>


    <div class="method">


        <div class="method-card">

            <strong>
                01 — Satellite Detection
            </strong>

            Sentinel-1 SAR imagery is processed using
            a U-Net segmentation model to identify the
            probable oil-slick region.

        </div>


        <div class="method-card">

            <strong>
                02 — Geometry
            </strong>

            The detected mask is converted into spatial
            information including centroid coordinates.

        </div>


        <div class="method-card">

            <strong>
                03 — Environmental Backtracking
            </strong>

            Historical wind and ocean-current data are
            used to estimate the probable spill origin.

        </div>


        <div class="method-card">

            <strong>
                04 — AIS Filtering
            </strong>

            Vessel tracks are filtered according to the
            relevant spatial and temporal window.

        </div>


        <div class="method-card">

            <strong>
                05 — Evidence Scoring
            </strong>

            Multiple independent evidence signals are
            combined into a final candidate score.

        </div>


        <div class="method-card">

            <strong>
                06 — Investigator Output
            </strong>

            The system produces a ranked list of candidate
            vessels for human investigation.

        </div>


    </div>

</div>


<!-- ======================================================
     CONCLUSION
     ====================================================== -->

<div class="section">

    <h2>
        Investigation Conclusion
    </h2>


    <p>

        Based on the current prototype inputs and scoring
        model,

        <strong>
            {top_vessel}
        </strong>

        is the highest-ranked vessel candidate with a final
        evidence score of

        <strong>
            {top_score:.2f}%
        </strong>.

        This ranking reflects the available computational
        evidence and should be interpreted as an investigative
        lead rather than a definitive attribution.

    </p>

</div>


<!-- ======================================================
     FOOTER
     ====================================================== -->

<div class="footer">

    <strong>
        OceanTraceAI
    </strong>

    <br>

    Automated Maritime Oil Spill Detection and Vessel Attribution

    <br>
    <br>

    Report generated:

    {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    <br>
    <br>

    Prototype demonstration —
    synthetic AIS data used for attribution testing.

</div>


</div>


</body>

</html>

"""

    # ========================================================
    # SAVE
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.write_text(
        html,
        encoding="utf-8"
    )

    # ========================================================
    # SUCCESS MESSAGE
    # ========================================================

    print()

    print("✓ Investigation report generated")

    print()

    print(
        f"Satellite image : {image_id}"
    )

    print(
        f"Spill location  : "
        f"{spill_lat:.6f}, {spill_lon:.6f}"
    )

    print(
        f"Estimated source: "
        f"{origin_lat:.6f}, {origin_lon:.6f}"
    )

    print(
        f"Wind            : "
        f"{wind_speed:.2f} m/s @ "
        f"{wind_direction:.0f}° FROM"
    )

    print(
        f"Current         : "
        f"{current_speed:.3f} m/s @ "
        f"{current_direction:.2f}°"
    )

    print()

    print(
        f"Top candidate   : {top_vessel}"
    )

    print(
        f"Final score     : {top_score:.2f}%"
    )

    print()

    print("Saved to:")

    print(
        OUTPUT_FILE
    )

    print()

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    generate_report()