#!/usr/bin/env python3

import sys
from timeit import default_timer as d_timer
import glob
import logging
import argparse

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

lat = 0.0
lon = 0.0

if args.location == "pda":
    lat = -26.25
    lon = -70.75
elif args.location == "lc":
    lat = -32.75
    lon = -71.25
elif args.location == "na":
    lat = -37.75
    lon = -73.25
elif args.location = "ag":
    lat = -29.75
    lon = -71.25
else:
    logger.info("Unknown location: {}".format(args.location))
    sys.exit(1)

logger.info("Location: {}, lat: {}, lon: {}".format(args.location, lat, lon))

input_prefix = "/esd/esd02/data/climate_models/echam/echam_output/ESD/"

input_folder = ""

if args.time == "pd":
    input_folder = "e004"
elif args.time == "pi":
    input_folder = "e007"
elif args.time == "mh":
    input_folder = "e008"
elif args.time == "lgm":
    input_folder = "e009"
elif args.time == "pli":
    input_folder = "e010"
elif args.time == "mi1":
    input_folder = "e011"
elif args.time == "mi2":
    input_folder = "e012"
else:
    logger.info("Unknown time period: {}".format(args.location))
    sys.exit(1)

logger.info("Time: {}, folder: {}".format(args.time, input_folder))


