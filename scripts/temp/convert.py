#!/usr/bin/env python3

import sys
from timeit import default_timer as d_timer
from glob import glob
import logging
import argparse
import xarray
import itertools


def get_lat_lon(loc):
    if loc == "pda":
        return (-26.25, -70.75)
    elif loc == "lc":
        return (-32.75, -71.25)
    elif loc == "na":
        return (-37.75, -73.25)
    elif loc == "sg":
        return (-29.75, -71.25)
    else:
        logger.info("Unknown location: {}".format(loc))
        sys.exit(1)

def get_file_list(time_p):
    input_prefix = "/esd/esd02/data/climate_models/echam/echam_output/ESD"
    file_list = []

    if time_p == "pd":
        input_folder = "e004"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    elif time_p == "pi":
        input_folder = "e007"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    elif time_p == "mh":
        input_folder = "e008"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    elif time_p == "lgm":
        input_folder = "e009"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    elif time_p == "pli":
        input_folder = "e010_hpc-bw_e5w2.3_PLIO_t159l31.1d/output_processed/monthly_mean"
        # Files: e010_echam_mm_100401.nc, ..., e010_echam_mm_101812.nc
        file_list = glob("{}/{}/e010_echam_mm_10*.nc".format(input_prefix, input_folder))
    elif time_p == "mi1":
        input_folder = "e011"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    elif time_p == "mi2":
        input_folder = "e012"
        file_list = glob("{}/{}/".format(input_prefix, input_folder))
    else:
        logger.info("Unknown time period: {}".format(time_p))
        sys.exit(1)

    file_list.sort()
    return file_list

def extract_data(ds, ds_lat, ds_lon):
    result = ds.sel(lat = ds_lat, lon = ds_lon, method="nearest")
    return result.values[0]

def process_files(args):
    (p_lat, p_lon) = get_lat_lon(args.location)

    logger.info("Location: {}, lat: {}, lon: {}".format(args.location, p_lat, p_lon))

    file_list = get_file_list(args.time)

    ds_lat = p_lat
    ds_lon = p_lon

    if ds_lon < 0.0:
        ds_lon = ds_lon + 360.0

    num_of_entries = args.num_of_years * 12

    surface_temperature = []
    precipitation = []
    surface_solar_radiation = []
    num_of_days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = itertools.accumulate(itertools.chain([0], itertools.cycle(num_of_days_per_month)))

    for f in file_list:
        ds = xarray.open_dataset(f)
        surface_temperature.append(extract_data(ds.tsurf, ds_lat, ds_lon))
        precipitation.append(extract_data(ds.aprl, ds_lat, ds_lon))
        surface_solar_radiation.append(extract_data(ds.srads, ds_lat, ds_lon))

    da_time = xarray.DataArray(
        data = itertools.islice(days, num_of_entries),
        dims = ["time"],
        attrs = {
            "_FillValue": "NaN",
            "units": "days since 1-1-15 00:00:00",
            "axis": "T",
            "long_name": "time",
            "standard_name": "time",
            "calendar": "0 yr B.P."
        }
    )

    da_lat = xarray.DataArray(
        data = [p_lat],
        dims = ["land_id"],
        attrs = {
            "_FillValue": "Nan",
            "standard_name": "latitude",
            "long_name": "latitude",
            "units": "degrees_north"
        }
    )

    da_lon = xarray.DataArray(
        data = [p_lon],
        dims = ["land_id"],
        attrs = {
            "_FillValue": "Nan",
            "standard_name": "longitude",
            "long_name": "longitude",
            "units": "degrees_east"
        }
    )

    da_land_id = xarray.DataArray(
        data = [1],
        dims = ["land_id"],
    )

    da_temp = xarray.DataArray(
        data = itertools.islice(itertools.cycle(surface_temperature), num_of_entries),
        dims = ["land_id", "time"],
        attrs = {
            "_FillValue": "-9999.0",
            "standard_name": "air_temperature",
            "long_name": "Near surface air temperature at 2m",
            "units": "K",
            "coordinates": "lon lat"
        }
    )

    da_prec = xarray.DataArray(
        data = itertools.islice(itertools.cycle(precipitation), num_of_entries),
        dims = ["land_id", "time"],
        attrs = {
            "_FillValue": "-9999.0",
            "standard_name": "precipitation_amount",
            "long_name": "Monthly precipitation amount",
            "units": "kg m-2",
            "coordinates": "lon lat"
        }
    )

    da_rad = xarray.DataArray(
        data = itertools.islice(itertools.cycle(surface_solar_radiation), num_of_entries),
        dims = ["land_id", "time"],
        attrs = {
            "_FillValue": "-9999.0",
            "standard_name": "surface_downwelling_shortwave_flux",
            "long_name": "Mean daily surface incident shortwave radiation",
            "units": "W m-2",
            "coordinates": "lon lat"
        }
    )

    ds_out_temp = xarray.Dataset({
        "time": da_time,
        "lat": da_lat,
        "lon": da_lon,
        "temp": da_temp,
        "land_id": da_land_id
    })

    ds_out_prec = xarray.Dataset({
        "time": da_time,
        "lat": da_lat,
        "lon": da_lon,
        "prec": da_prec,
        "land_id": da_land_id
    })

    ds_out_rad = xarray.Dataset({
        "time": da_time,
        "lat": da_lat,
        "lon": da_lon,
        "rad": da_rad,
        "land_id": da_land_id
    })

    ds_out_temp.to_netcdf("temperature_{}_{}.nc".format(args.location, args.time))
    ds_out_prec.to_netcdf("precipitation_{}_{}.nc".format(args.location, args.time))
    ds_out_rad.to_netcdf("radiation_{}_{}.nc".format(args.location, args.time))

    #ds_out_temp["lat"] = (("land_id", ), [p_lat])
    #ds_out_temp["lon"] = (("land_id", ), [p_lon])
    #ds_out_temp["temp"] = (("land_id", "time"), surface_temperature)

    #ds_out_temp.coords["time"] = (("time",), itertools.islice(days, num_of_years * 12))
    #ds_out_temp.coords["land_id"] = (("land_id",), [1])


if __name__ == "__main__":
    t_start = d_timer()

    logger = logging.getLogger('converter')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler('converter.log')
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    parser = argparse.ArgumentParser()
    parser.add_argument("location", help="The study area: pda, lc, na, sg")
    parser.add_argument("time", help="The time period: pd, pi, mh, lgm, pli, mi1, mi2")
    parser.add_argument("years", help="The number of years the data should be generated")
    args = parser.parse_args()

    process_files(args)

    t_end = d_timer()

    logger.info("Time taken: {}".format(t_end - t_start))


