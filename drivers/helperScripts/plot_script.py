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

class SimData:
    def __init__(self):
        self.topo_mean = []
        self.eros_mean = []
        self.sedi_mean = []
        self.prec_mean = []
        self.soil_mean = []
        self.vegi_mean = []
        self.tree_mean = []
        self.shrub_mean = []
        self.grass_mean = []
        self.fontsize = 20
        self.color = "red"


    def append(self, p, data):
        if p == "topographic__elevation":
            self.topo_mean.append(data)
        elif p == "erosion__rate":
            self.eros_mean.append(data)
        elif p == "sediment__flux":
            self.sedi_mean.append(data)
        elif p == "precipitation":
            self.prec_mean.append(data)
        elif p == "soil__depth":
            self.soil_mean.append(data)
        elif p == "vegetation__density":
            self.vegi_mean.append(data)
        elif p == "tree_fpc":
            self.tree_mean.append(data)
        elif p == "shrub_fpc":
            self.shrub_mean.append(data)
        elif p == "grass_fpc":
            self.grass_mean.append(data)
        else:
            print("Unknown parameter: {}".format(p))
            sys.exit(1)

    def plot(self, ax, data, ylabel):
        ax.plot(data)
        ax.set_ylabel(ylabel, fontsize = self.fontsize, color = self.color)

    def plot1(self, filename):
        fig, ax = plt.subplots(4,2, figsize = [15,15], sharex = True)

        self.plot(ax[0,0], self.topo_mean, "topo mean [m]")
        self.plot(ax[1,0], self.eros_mean, "eros mean [m/yr]")
        self.plot(ax[1,1], self.sedi_mean, "sedi mean [m3/s]")
        self.plot(ax[0,1], self.prec_mean, "prec mean [cm/yr]")
        self.plot(ax[2,0], self.soil_mean, "soil mean [m]")
        self.plot(ax[2,1], self.tree_mean, "tree mean [%]")
        self.plot(ax[3,0], self.grass_mean, "grass mean [%]")
        self.plot(ax[3,1], self.shrub_mean, "shrub mean [%]")

        ax[3,0].set_xlabel("time steps", fontsize = self.fontsize, color = self.color)
        ax[3,1].set_xlabel("time steps", fontsize = self.fontsize, color = self.color)

        cwd = os.getcwd()
        title = os.path.basename(cwd)
        fig.suptitle(title, fontsize = self.fontsize)

        plt.tight_layout(rect=[0, 0.001, 1, 0.95])

        plt.savefig(filename, dpi = 420)

    def plot2(self, filename):
        pass


if __name__ == "__main__":
    output_files = "ll_output/NC/"

    if not os.path.exists(output_files):
        print("Could not find path: '{}'".format(output_files))
        print("This script should be run from inside the simulation folder")
        sys.exit(1)

    sim_data = SimData()

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

            sim_data.append(p, np.mean(parameter_data))

        # break

    sim_data.plot1("overview1.png")
    sim_data.plot2("overview1.png")

