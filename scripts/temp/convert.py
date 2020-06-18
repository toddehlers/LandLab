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

def process_files(file_list, p_lat, p_lon):
    ds_lat = p_lat
    ds_lon = p_lon

    if ds_lon < 0.0:
        ds_lon = ds_lon + 360.0

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

    ds_out_temp = xarray.Dataset()



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
    args = parser.parse_args()

    lat_lon = get_lat_lon(args.location)

    logger.info("Location: {}, lat_lon: {}".format(args.location, lat_lon))

    file_list = get_file_list(args.time)

    process_files(file_list, lat_lon[0], lat_lon[1])

    t_end = d_timer()

    logger.info("Time taken: {}".format(t_end - t_start))


