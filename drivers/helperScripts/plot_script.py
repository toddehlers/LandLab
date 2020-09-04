#!/usr/bin/python3

import os
import sys
import glob
import configparser
import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import netCDF4
from tqdm import tqdm

class SimData:
    def __init__(self):
        # PLot data values
        self.elapsed_time = []
        self.topo_mean = []
        self.eros_mean = []
        self.sedi_mean = []
        self.prec_mean = []
        self.soil_mean = []
        self.temperature_mean = []
        self.vegi_mean_fpc = []
        self.vegi_mean_lai = []
        self.tree_mean_fpc = []
        self.tree_mean_lai = []
        self.shrub_mean_fpc = []
        self.shrub_mean_lai = []
        self.grass_mean_fpc = []
        self.grass_mean_lai = []

        # DEM files
        self.img_file1 = ""
        self.img_file2 = ""

        # Plot settings
        self.figsize = [15, 15]
        self.fontsize = 20
        self.color = "red"
        self.dpi = 420
        self.rect = [0, 0.001, 1, 0.95]
        cwd = os.getcwd()
        self.title = os.path.basename(cwd)
        self.plot_start = 0
        self.plot_end = 0

    def append(self, p, data):
        if p == "elapsed_time":
            self.elapsed_time = data
        elif p == "topographic__elevation":
            self.topo_mean.append(data)
        elif p == "erosion__rate":
            self.eros_mean.append(data)
        elif p == "sediment__flux":
            self.sedi_mean.append(data)
        elif p == "precipitation":
            self.prec_mean.append(data)
        elif p == "soil__depth":
            self.soil_mean.append(data)
        elif p == "temperature":
            self.temperature_mean.append(data)
        elif p == "vegetation__density":
            self.vegi_mean_fpc.append(data)
        elif p == "vegetation__density_lai":
            self.vegi_mean_lai.append(data)
        elif p == "tree_fpc":
            self.tree_mean_fpc.append(data)
        elif p == "tree_lai":
            self.tree_mean_lai.append(data)
        elif p == "shrub_fpc":
            self.shrub_mean_fpc.append(data)
        elif p == "shrub_lai":
            self.shrub_mean_lai.append(data)
        elif p == "grass_fpc":
            self.grass_mean_fpc.append(data)
        elif p == "grass_lai":
            self.grass_mean_lai.append(data)
        else:
            print("Unknown parameter: {}".format(p))
            sys.exit(1)

    def set_plot_range(self, plot_start, plot_end):
        self.plot_start = plot_start
        self.plot_end = plot_end

    def set_dem_files(self, img_file1, img_file2):
        self.img_file1 = img_file1
        self.img_file2 = img_file2

    def plot(self, ax, data, ylabel):
        ax.plot(self.elapsed_time, data)
        ax.set_ylabel(ylabel, fontsize = self.fontsize, color = self.color)

    def plot1(self, filename):
        fig, ax = plt.subplots(4,2, figsize = self.figsize, sharex = True)

        self.plot(ax[0,0], self.topo_mean, "topo mean [m]")
        self.plot(ax[0,1], self.eros_mean, "eros mean [m/yr]")
        self.plot(ax[1,0], self.sedi_mean, "sedi mean [m3/s]")
        self.plot(ax[1,1], self.prec_mean, "prec mean [cm/yr]")
        self.plot(ax[2,0], self.soil_mean, "soil mean [m]")
        self.plot(ax[2,1], self.tree_mean_fpc, "tree fpc mean [%]")
        self.plot(ax[3,0], self.grass_mean_fpc, "grass fpc mean [%]")
        self.plot(ax[3,1], self.shrub_mean_fpc, "shrub fpc mean [%]")

        ax[3,0].set_xlabel("elapsed time", fontsize = self.fontsize, color = self.color)
        ax[3,1].set_xlabel("elapsed time", fontsize = self.fontsize, color = self.color)

        fig.suptitle(self.title, fontsize = self.fontsize)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, self.dpi)

    def plot2(self, filename):
        fig, ax = plt.subplots(4,2, figsize = self.figsize, sharex = True)

        self.plot(ax[0,0], self.topo_mean, "TODO")
        self.plot(ax[0,1], self.eros_mean, "TODO")
        self.plot(ax[1,0], self.vegi_mean_fpc, "vegi_mean_fpc [?]")
        self.plot(ax[1,1], self.vegi_mean_lai, "vegi_mean_lai [?]")
        self.plot(ax[2,0], self.temperature_mean, "temperature mean [°C]")
        self.plot(ax[2,1], self.tree_mean_lai, "tree lai mean [?]")
        self.plot(ax[3,0], self.grass_mean_lai, "grass lai mean [?]")
        self.plot(ax[3,1], self.shrub_mean_lai, "shrub lai mean [?]")

        ax[3,0].set_xlabel("elapsed time [yrs]", fontsize = self.fontsize, color = self.color)
        ax[3,1].set_xlabel("elapsed time [yrs]", fontsize = self.fontsize, color = self.color)

        fig.suptitle(self.title, fontsize = self.fontsize)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, self.dpi)

    def plot3(self, filename):
        img1 = mpimg.imread(self.img_file1)
        img2 = mpimg.imread(self.img_file2)

        fig, ax = plt.subplots(1,2, figsize = self.figsize, sharex = True)

        ax[0,0].imshow(img1)
        ax[0,1].imshow(img2)

        ax[0,0].set_xlabel("DEM at {} yrs".format(self.plot_start), fontsize = self.fontsize, color = self.color)
        ax[0,1].set_xlabel("DEM at {} yrs".format(self.plot_end), fontsize = self.fontsize, color = self.color)

        fig.suptitle(self.title, fontsize = self.fontsize)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, self.dpi)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('inputFile.ini')

    plot_start = int(float(config['Runtime']['plot_start']))
    plot_end = int(float(config['Runtime']['plot_end']))

    if plot_start >= plot_end:
        print("Error in input file 'inputFile.ini': plot_start must be smaller than plot_end!")
        print("plot_start: {} >= plot_end: {} ".format(plot_start, plot_end))
        sys.exit(1)

    sim_data = SimData()
    sim_data.set_plot_range(plot_start, plot_end)

    all_files = glob.glob("ll_output/NC/*.nc")
    time_and_names = ((int(name.split("__")[1][:-3]), name) for name in all_files)

    for (elapsed_time, nc_file) in tqdm(sorted(time_and_names)):
        if elapsed_time < plot_start or elapsed_time > plot_end:
            continue

        nc_data = netCDF4.Dataset(nc_file) # pylint: disable=no-member

        elapsed_time = nc_data.getncattr("lgt.timestep")

        sim_data.append("elapsed_time", elapsed_time)

        parameters = [
            "topographic__elevation",
            "erosion__rate",
            "sediment__flux",
            "precipitation",
            "soil__depth",
            "temperature",
        ]

        if "tree_fpc" and "shrub_fpc" and "grass_fpc" in nc_data.variables:
            parameters.append("tree_fpc")
            parameters.append("shrub_fpc")
            parameters.append("grass_fpc")
        elif "tree_lai" and "shrub_lai" and "grass_lai" in nc_data.variables:
            parameters.append("tree_lai")
            parameters.append("shrub_lai")
            parameters.append("grass_lai")
        elif "vegetation__density_lai" in nc_data.variables:
            parameters.append("vegetation__density_lai")
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

    distance1 = sys.maxsize
    distance2 = sys.maxsize

    img_name1 = ""
    img_name2 = ""

    all_files = glob.glob("ll_output/DEM/*.png")
    time_and_names = ((int(name.split("__")[1][:-3]), name) for name in all_files)

    # Find both DEM images which are closest to plot_start and plot_end
    for (elapsed_time, img_name) in tqdm(sorted(time_and_names)):
        dist = abs(elapsed_time - plot_start)
        if dist < distance1:
            distance1 = dist
            img_name1 = img_name

        dist = abs(elapsed_time - plot_end)
        if dist < distance2:
            distance2 = dist
            img_name2 = img_name

    sim_data.set_dem_files(img_name1, img_name2)

    sim_data.plot1("overview1.png")
    sim_data.plot2("overview2.png")
    sim_data.plot3("overview3.png")

