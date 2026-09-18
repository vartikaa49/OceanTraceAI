import pandas as pd
from math import radians, sin, cos, sqrt, atan2


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two geographic points.
    Returns distance in kilometers.
    """

    R = 6371.0

    lat1 = radians(lat1)
    lat2 = radians(lat2)

    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1)
        * cos(lat2)
        * sin(dlon / 2) ** 2
    )

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


if __name__ == "__main__":

    print("========== AIS MODULE TEST ==========")

    spill_lat = 28.9081
    spill_lon = -89.0239

    vessel_lat = 28.9500
    vessel_lon = -89.0500

    distance = haversine_distance(
        spill_lat,
        spill_lon,
        vessel_lat,
        vessel_lon
    )

    print("Spill location:")
    print(f"Latitude  : {spill_lat}")
    print(f"Longitude : {spill_lon}")

    print("\nVessel location:")
    print(f"Latitude  : {vessel_lat}")
    print(f"Longitude : {vessel_lon}")

    print(f"\nDistance : {distance:.2f} km")

    print("====================================")