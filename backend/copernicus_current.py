import math
import xarray as xr


def get_current_from_netcdf(
    file_path,
    latitude,
    longitude
):
    """
    Read the nearest ocean-current grid point from
    a Copernicus Marine NetCDF file.

    Returns:
        current_speed (m/s)
        current_direction (degrees)
        uo (eastward velocity)
        vo (northward velocity)
    """

    ds = xr.open_dataset(file_path)

    point = ds.sel(
        latitude=latitude,
        longitude=longitude,
        method="nearest"
    )

    uo = float(point["uo"].values.squeeze())
    vo = float(point["vo"].values.squeeze())

    current_speed = math.sqrt(
        uo ** 2 + vo ** 2
    )

    # Direction the current is moving TO
    current_direction = (
        math.degrees(
            math.atan2(uo, vo)
        ) + 360
    ) % 360

    ds.close()

    return (
        current_speed,
        current_direction,
        uo,
        vo
    )


if __name__ == "__main__":

    file_path = "data/environment/test_current.nc"

    latitude = 28.9081
    longitude = -89.0239

    speed, direction, uo, vo = get_current_from_netcdf(
        file_path,
        latitude,
        longitude
    )

    print("\n========== COPERNICUS CURRENT ==========")
    print(f"Location          : {latitude}, {longitude}")
    print(f"Eastward velocity : {uo:.6f} m/s")
    print(f"Northward velocity: {vo:.6f} m/s")
    print(f"Current speed     : {speed:.6f} m/s")
    print(f"Current direction : {direction:.2f}°")
    print("=========================================")