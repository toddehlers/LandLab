#!/usr/bin/python3

import os
import sys
import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import netCDF4
from progress.bar import Bar

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

    for nc_file in Bar("Processing netCDF output files").iter(sorted(glob.glob(os.path.join(output_files, "*.nc")), key = lambda s: int(s[:-3].split("__")[1]))):
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

    #topo_mean = [np.mean(v) for v in topo_mean]
    #eros_mean = [np.mean(v) for v in eros_mean]
    #sedi_mean = [np.mean(v) for v in sedi_mean]
    #prec_mean = [np.mean(v) for v in prec_mean]
    #soil_mean = [np.mean(v) for v in soil_mean]
    #vegi_mean = [np.mean(v) for v in vegi_mean]
    #tree_mean = [np.mean(v) for v in tree_mean]
    #shrub_mean = [np.mean(v) for v in shrub_mean]
    #grass_mean = [np.mean(v) for v in grass_mean]

    fig, ax = plt.subplots(4,2, figsize = [15,15], sharex = True)

    ax[0,0].plot(topo_mean)
    ax[1,0].plot(eros_mean)
    ax[1,1].plot(sedi_mean)
    ax[0,1].plot(prec_mean)
    ax[2,0].plot(soil_mean)
    ax[2,1].plot(tree_mean)
    ax[3,0].plot(grass_mean)
    ax[3,1].plot(shrub_mean)

    ax[0,0].set_ylabel("topo mean [m]", fontsize = 20, color = "red")
    ax[1,0].set_ylabel("eros mean [m/yr]", fontsize = 20, color = "red")
    ax[1,1].set_ylabel("sedi mean []", fontsize = 20, color = "red")
    ax[0,1].set_ylabel("prec mean [cm]", fontsize = 20, color = "red")
    ax[2,0].set_ylabel("soil mean [m]", fontsize = 20, color = "red")
    ax[2,1].set_ylabel("tree mean []", fontsize = 20, color = "red")
    ax[3,0].set_ylabel("grass mean [m]", fontsize = 20, color = "red")
    ax[3,1].set_ylabel("shrub mean [m]", fontsize = 20, color = "red")

    plt.tight_layout()

    plt.savefig("overview.png", dpi = 420)

    topo_min = min(topo_mean)
    topo_max = max(topo_mean)

    print(f"Topo min: {topo_min}m")
    print(f"Topo max: {topo_max}m")
    print(f"Total relief: {topo_max - topo_min}m")
