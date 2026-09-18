from pathlib import Path
import pandas as pd


# ============================================================
# OCEANTRACEAI - INDIAN OCEAN TEST SCENARIO
# ============================================================

OUTPUT_FILE = Path(
    "data/environment/indian_test_scenario.csv"
)

# Arabian Sea, west of India
SPILL_LATITUDE = 18.500000
SPILL_LONGITUDE = 70.500000

OBSERVATION_TIME = "2026-09-08T00:09:23"

BACKTRACK_HOURS = 12


# ============================================================
# CREATE SCENARIO
# ============================================================

def main():

    print("=" * 68)
    print("      OCEANTRACEAI INDIAN OCEAN TEST SCENARIO")
    print("=" * 68)

    print()
    print("Creating synthetic deployment scenario...")
    print()
    print(f"Spill latitude  : {SPILL_LATITUDE:.6f}")
    print(f"Spill longitude : {SPILL_LONGITUDE:.6f}")
    print(f"Observation UTC : {OBSERVATION_TIME}")
    print(f"Backtrack time  : {BACKTRACK_HOURS} hours")

    data = {
        "image_id": ["INDIA_TEST_001"],
        "observation_time_utc": [OBSERVATION_TIME],
        "spill_latitude": [SPILL_LATITUDE],
        "spill_longitude": [SPILL_LONGITUDE],
        "backtrack_hours": [BACKTRACK_HOURS],
    }

    df = pd.DataFrame(data)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("✓ Indian Ocean scenario created")

    print()
    print("Saved to:")
    print(OUTPUT_FILE)

    print()
    print("Scenario:")
    print(df.to_string(index=False))

    print()
    print("=" * 68)
    print("              SCENARIO CREATION COMPLETE")
    print("=" * 68)


if __name__ == "__main__":
    main()