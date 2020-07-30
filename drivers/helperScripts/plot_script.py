#!/usr/bin/python3

import os
import sys
import glob
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
from tqdm import tqdm

if __name__ == "__main__":
    output_files = "ll_output/NC/"

    if not os.path.exists(output_files):
        print("Could not find path: '{}'".format(output_files))
        print("This script should be run from inside the simulation folder")
        sys.exit(1)

    topo_mean = []
    eros_mean = []
    sedi_mean = []
    prec_mean = []
    soil_mean = []
    vegi_mean = []
    tree_mean = []
    shrub_mean = []
    grass_mean = []

    for nc_file in tqdm(sorted(glob.glob(os.path.join(output_files, "*.nc")), key = lambda s: int(s[19:].split("__")[0]))):
        nc_data = netCDF4.Dataset(nc_file)

        parameters = [
            "topographic__elevation",
            "soil__depth",
            "sediment__flux",
            "precipitation",
            "erosion__rate",
        ]

        if "tree_fpc" and "shrub_fpc" and "grass_fpc" in nc_data.variables:
            parameters.append("tree_fpc")
            parameters.append("shrub_fpc")
            parameters.append("grass_fpc")
        else:
            parameters.append("vegetation__density")

        for p in parameters:
            parameter_data = nc_data.variables[p][:][0]
            # delete boundary nodes
            parameter_data = np.delete(parameter_data, 0 , axis = 0)
            parameter_data = np.delete(parameter_data, -1 , axis = 0)
            parameter_data = np.delete(parameter_data, 0 , axis = 1)
            parameter_data = np.delete(parameter_data, -1 , axis = 1)

            if p == "topographic__elevation":
                topo_mean.append(np.mean(parameter_data))
            elif p == "erosion__rate":
                eros_mean.append(np.mean(parameter_data))
            elif p == "sediment__flux":
                sedi_mean.append(np.mean(parameter_data))
            elif p == "precipitation":
                prec_mean.append(np.mean(parameter_data))
            elif p == "soil__depth":
                soil_mean.append(np.mean(parameter_data))
            elif p == "vegetation__density":
                vegi_mean.append(np.mean(parameter_data))
            elif p == "tree_fpc":
                tree_mean.append(np.mean(parameter_data))
            elif p == "shrub_fpc":
                shrub_mean.append(np.mean(parameter_data))
            elif p == "grass_fpc":
                grass_mean.append(np.mean(parameter_data))
            else:
                print("Unknown parameter: {}".format(p))
                sys.exit(1)

        # break

    fig, ax = plt.subplots(4,2, figsize = [15,15], sharex = True)

    ax[0,0].plot(topo_mean)
    ax[1,0].plot(eros_mean)
    ax[1,1].plot(sedi_mean)
    ax[0,1].plot(prec_mean)
    ax[2,0].plot(soil_mean)
    ax[2,1].plot(tree_mean)
    ax[3,0].plot(grass_mean)
    ax[3,1].plot(shrub_mean)

    fontsize = 20
    color = "red"

    ax[0,0].set_ylabel("topo mean [m]", fontsize = fontsize, color = color)
    ax[1,0].set_ylabel("eros mean [m/yr]", fontsize = fontsize, color = color)
    ax[1,1].set_ylabel("sedi mean []", fontsize = fontsize, color = color)
    ax[0,1].set_ylabel("prec mean [cm]", fontsize = fontsize, color = color)
    ax[2,0].set_ylabel("soil mean [m]", fontsize = fontsize, color = color)
    ax[2,1].set_ylabel("tree mean []", fontsize = fontsize, color = color)
    ax[3,0].set_ylabel("grass mean [m]", fontsize = fontsize, color = color)
    ax[3,1].set_ylabel("shrub mean [m]", fontsize = fontsize, color = color)

    ax[3,0].set_xlabel("time steps", fontsize = fontsize, color = color)
    ax[3,1].set_xlabel("time steps", fontsize = fontsize, color = color)

    cwd = os.getcwd()
    title = os.path.basename(cwd)
    fig.suptitle(title, fontsize = fontsize)

    plt.tight_layout(rect=[0, 0.001, 1, 0.95])

    plt.savefig("overview.png", dpi = 420)
