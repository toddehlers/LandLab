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
        self.uplift_rate = 0.0

        # DEM files
        self.img_file1 = ""
        self.img_file2 = ""

        # Plot settings
        self.figsize = [15, 15]
        self.fontsize_label = 20
        self.fontsize_ticks = 14
        self.color = "red"
        self.dpi = 420
        self.rect = [0, 0.001, 1, 0.95]
        cwd = os.getcwd()
        self.title = os.path.basename(cwd)
        self.plot_start = 0
        self.plot_end = 0

    def append(self, p, data):
        if p == "elapsed_time":
            self.elapsed_time.append(data / 1000)
        elif p == "topographic__elevation":
            self.topo_mean.append(data)
        elif p == "erosion__rate":
            self.eros_mean.append(data * 1000)
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

    def set_uplift_rate(self, uplift_rate):
        self.uplift_rate = uplift_rate * 1000

    def set_dem_files(self, img_file1, img_file2):
        self.img_file1 = img_file1
        self.img_file2 = img_file2

    def plot(self, ax, data, ylabel):
        ax.plot(self.elapsed_time, data)
        ax.set_ylabel(ylabel, fontsize = self.fontsize_label, color = self.color)
        ax.xaxis.set_tick_params(labelsize = self.fontsize_ticks)
        ax.yaxis.set_tick_params(labelsize = self.fontsize_ticks)

    def plot1(self, filename):
        fig, ax = plt.subplots(4,2, figsize = self.figsize, sharex = True)

        ax[0,0].plot(self.elapsed_time, self.prec_mean)
        ax[0,0].plot(self.elapsed_time, self.temperature_mean, color = "red")
        ax[0,0].text(-0.15, 0.5, "prec. [cm/yr]", color = "tab:blue", fontsize = self.fontsize_label,
            rotation = "vertical", transform = ax[0,0].transAxes, verticalalignment = "center")
        ax[0,0].text(-0.1, 0.5, "temp. [°C]", color = "red", fontsize = self.fontsize_label,
            rotation = "vertical", transform = ax[0,0].transAxes, verticalalignment = "center")
        ax[0,0].xaxis.set_tick_params(labelsize = self.fontsize_ticks)
        ax[0,0].yaxis.set_tick_params(labelsize = self.fontsize_ticks)

        self.plot(ax[0,1], self.eros_mean, "erosion rate [mm/yr]")
        self.plot(ax[1,0], self.sedi_mean, "CO2, TODO")
        self.plot(ax[1,1], self.sedi_mean, "sedi mean [$m^3$/s]")
        self.plot(ax[2,0], self.soil_mean, "soil thickness [m]")
        self.plot(ax[2,1], self.tree_mean_fpc, "tree FPC mean [%]")
        self.plot(ax[3,0], self.grass_mean_fpc, "grass FPC mean [%]")
        self.plot(ax[3,1], self.shrub_mean_fpc, "shrub FPC mean [%]")

        uplift_rate = [self.uplift_rate for i in self.elapsed_time]
        ax[0,1].plot(self.elapsed_time, uplift_rate, color = "red", linestyle = "--")

        ax[3,0].set_xlabel("elapsed time [kyr]", fontsize = self.fontsize_label, color = self.color)
        ax[3,1].set_xlabel("elapsed time [kyr]", fontsize = self.fontsize_label, color = self.color)

        fig.suptitle(self.title, fontsize = self.fontsize_label)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, dpi = self.dpi)

    def plot2(self, filename):
        fig, ax = plt.subplots(4,2, figsize = self.figsize, sharex = True)

        self.plot(ax[0,0], self.topo_mean, "TODO")
        self.plot(ax[0,1], self.eros_mean, "TODO")
        self.plot(ax[1,0], self.vegi_mean_fpc, "vegi FPC mean [%]")
        self.plot(ax[1,1], self.vegi_mean_lai, "vegi LAI mean")
        self.plot(ax[2,0], self.topo_mean, "mean elevation [m]")
        self.plot(ax[2,1], self.tree_mean_lai, "tree LAI mean")
        self.plot(ax[3,0], self.grass_mean_lai, "grass LAI mean")
        self.plot(ax[3,1], self.shrub_mean_lai, "shrub LAI mean")

        ax[3,0].set_xlabel("elapsed time [kyr]", fontsize = self.fontsize_label, color = self.color)
        ax[3,1].set_xlabel("elapsed time [kyr]", fontsize = self.fontsize_label, color = self.color)

        fig.suptitle(self.title, fontsize = self.fontsize_label)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, dpi = self.dpi)

    def plot3(self, filename):
        img1 = mpimg.imread(self.img_file1)
        img2 = mpimg.imread(self.img_file2)

        fig, ax = plt.subplots(1,2, figsize = self.figsize, sharex = True)

        ax[0].imshow(img1)
        ax[1].imshow(img2)

        ax[0].get_xaxis().set_ticks([])
        ax[0].get_yaxis().set_ticks([])
        ax[1].get_xaxis().set_ticks([])
        ax[1].get_yaxis().set_ticks([])

        fig.suptitle(self.title, fontsize = self.fontsize_label)

        plt.tight_layout(rect = self.rect)
        plt.savefig(filename, dpi = self.dpi)

    def debug_output(self):
        print("elapsed_time: {}".format(len(self.elapsed_time)))
        #print("self.vegi_mean_fpc: {}".format(len(self.vegi_mean_fpc)))
        #print("self.vegi_mean_lai: {}".format(len(self.vegi_mean_lai)))
        #print("img_file1: {}".format(self.img_file1))
        #print("img_file2: {}".format(self.img_file2))


def extract_time(name):
    (name, ext) = os.path.splitext(name)
    return int(name.split("__")[1])

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("inputFile.ini")

    plot_start = float(config["Runtime"]["plot_start"])
    plot_end = float(config["Runtime"]["plot_end"])
    step_size_dt = float(config["Runtime"]["dt"])

    if plot_start >= plot_end:
        print("Error in input file 'inputFile.ini': plot_start must be smaller than plot_end!")
        print("plot_start: {} >= plot_end: {} ".format(plot_start, plot_end))
        sys.exit(1)

    sim_data = SimData()

    uplift_rate = float(config["Uplift"]["upliftRate"])
    sim_data.set_uplift_rate(uplift_rate)

    all_files = glob.glob("ll_output/NC/*.nc")
    time_and_names = ((extract_time(name), name) for name in all_files)

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
            "vegetation__density_lai",
            "vegetation__density",
        ]

        if "tree_fpc" and "shrub_fpc" and "grass_fpc" in nc_data.variables:
            parameters.append("tree_fpc")
            parameters.append("shrub_fpc")
            parameters.append("grass_fpc")
        if "tree_lai" and "shrub_lai" and "grass_lai" in nc_data.variables:
            parameters.append("tree_lai")
            parameters.append("shrub_lai")
            parameters.append("grass_lai")

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
    time_and_names = ((extract_time(name), name) for name in all_files)

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

    sim_data.debug_output()

    sim_data.plot1("overview1.png")
    sim_data.plot2("overview2.png")
    sim_data.plot3("overview3.png")

