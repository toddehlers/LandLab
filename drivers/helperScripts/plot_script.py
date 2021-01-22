#!/usr/bin/python3

import os
import sys
import glob
import configparser
import logging
import numpy as np
import netCDF4
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SimData:
    def __init__(self):
        # PLot data values
        self.elapsed_time = []
        self.topo_mean = []
        self.eros_mean = []
        self.co2_mean = []
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
        self.map_elevation1 = []
        self.map_elevation2 = []
        self.map_erosion_rate1 = []
        self.map_erosion_rate2 = []
        self.burned_area_frac = []
        self.runoff_mean = []
        self.evapo_trans_soil = []
        self.evapo_trans_area = []
        self.net_primary_productivity = []

        self.uplift_rate = 0.0
        self.node_spacing = 0.0
        self.num_of_nodes = 0.0

        # Plot settings
        self.figsize = [15, 15]
        self.fontsize_label = 20
        self.fontsize_ticks = 14
        self.color = "red"
        self.dpi = 420
        self.rect = [0.01, 0.001, 1, 0.95]
        self.plot_start = 0
        self.plot_end = 0
        self.plot_time_type = "Normal"

        # Misc
        self.max_elevation = None
        self.min_elevation = None
        self.max_erosion = None
        self.min_erosion = None
        self.time_label = None
        self.title = None
        self.sup_title = None
        self.time_label = None
        self.unknown_params = set() # keeps track of unknown parameter names encountered


    def append(self, p_, data):
        if p_ == "elapsed_time":
            self.elapsed_time.append(data / 1000)
        elif p_ == "topographic__elevation":
            self.topo_mean.append(data)
        elif p_ == "erosion__rate":
            self.eros_mean.append(data * 1000)
        elif p_ == "co2":
            self.co2_mean.append(data)
        elif p_ == "sediment__flux":
            self.sedi_mean.append(data)
        elif p_ == "precipitation":
            self.prec_mean.append(data)
        elif p_ == "soil__depth":
            self.soil_mean.append(data)
        elif p_ == "temperature":
            self.temperature_mean.append(data)
        elif p_ == "vegetation__density":
            self.vegi_mean_fpc.append(data * 100)
        elif p_ == "vegetation__density_lai":
            self.vegi_mean_lai.append(data)
        elif p_ == "tree_fpc":
            self.tree_mean_fpc.append(data * 100)
        elif p_ == "tree_lai":
            self.tree_mean_lai.append(data)
        elif p_ == "shrub_fpc":
            self.shrub_mean_fpc.append(data * 100)
        elif p_ == "shrub_lai":
            self.shrub_mean_lai.append(data)
        elif p_ == "grass_fpc":
            self.grass_mean_fpc.append(data * 100)
        elif p_ == "grass_lai":
            self.grass_mean_lai.append(data)
        elif p_ == "burned_area_frac":
            self.burned_area_frac.append(data * 100) # convert to %
        elif p_ == "runoff":
            self.runoff_mean.append(data)
        elif p_ == "evapo_trans_soil":
            self.evapo_trans_soil.append(data)
        elif p_ == "evapo_trans_area":
            self.evapo_trans_area.append(data)
        elif p_ == "net_primary_productivity":
            self.net_primary_productivity.append(data)
        else:
            if p_ not in self.unknown_params:
                logging.warning("Ignoring unknown parameter: %s", p_)
                self.unknown_params.add(p_)
            else:
                pass

    def set_uplift_rate(self, uplift_rate_):
        self.uplift_rate = uplift_rate_ * 1000

    def set_node_spacing(self, node_spacing_):
        # Convert from [m] to [km]
        self.node_spacing = node_spacing_ / 1000.0

    def set_num_of_nodes(self, num_of_nodes_):
        self.num_of_nodes = num_of_nodes_ - 1.0

    def set_map_elevation1(self, map_elevation):
        self.map_elevation1 = map_elevation

    def set_map_elevation2(self, map_elevation):
        self.map_elevation2 = map_elevation
        self.max_elevation = max(np.max(self.map_elevation1), np.max(self.map_elevation2))
        self.min_elevation = min(np.min(self.map_elevation1), np.min(self.map_elevation2))

    def set_map_erosion1(self, map_erosion):
        self.map_erosion_rate1 = map_erosion * 1000

    def set_map_erosion2(self, map_erosion):
        self.map_erosion_rate2 = map_erosion * 1000
        self.max_erosion = max(np.max(self.map_erosion_rate1), np.max(self.map_erosion_rate2))
        self.min_erosion = min(np.min(self.map_erosion_rate1), np.min(self.map_erosion_rate2))

    def set_plot_time_type(self, plot_time_type):
        self.plot_time_type = plot_time_type

        if plot_time_type == "Normal":
            self.time_label = "elapsed time"
        elif plot_time_type == "LGM":
            self.time_label = "time before present"
            right_end = self.elapsed_time[-1]
            self.elapsed_time = [abs(t - right_end) for t in self.elapsed_time]
        else:
            self.time_label = "Unknown"

    def set_title(self, title, sup_title):
        self.title = title

        if len(sup_title) > 0:
            self.sup_title = sup_title
        else:
            cwd = os.getcwd()
            self.sup_title = os.path.basename(cwd)

    def plot(self, ax, data, ylabel):
        ax.plot(self.elapsed_time, data)
        ax.set_ylabel(ylabel, fontsize=self.fontsize_label, color=self.color)
        ax.xaxis.set_tick_params(labelsize=self.fontsize_ticks)
        ax.yaxis.set_tick_params(labelsize=self.fontsize_ticks)

    def plot_image(self, ax, image_data, color_map, cbar_label, vmin, vmax):
        end = self.num_of_nodes * self.node_spacing
        extent = (0, end, 0, end)

        img = ax.imshow(image_data, cmap=color_map, vmin=vmin, vmax=vmax, extent=extent, origin="lower")

        cbar = ax.figure.colorbar(img, ax=ax, fraction=0.045)
        cbar.ax.set_ylabel(cbar_label, fontsize=self.fontsize_label, color=self.color)
        cbar.ax.tick_params(labelsize=self.fontsize_ticks)

        ax.xaxis.set_tick_params(labelsize=self.fontsize_ticks)
        ax.yaxis.set_tick_params(labelsize=self.fontsize_ticks)

        #ax.invert_yaxis()
        #ax.axis("tight")

    def plot_elevation(self, ax, data):
        self.plot_image(ax, data, "terrain", "elevation [$m$]", self.min_elevation, self.max_elevation)

    def plot_erosion_rate(self, ax, data):
        self.plot_image(ax, data, "hot", "erosion rate [$mm/yr$]", self.min_erosion, self.max_erosion)

    def plot1(self, filename):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize, sharex=True)

        ax[0, 0].plot(self.elapsed_time, self.prec_mean)
        ax[0, 0].plot(self.elapsed_time, self.temperature_mean, color="red")
        ax[0, 0].text(-0.15, 0.5, "prec. [$cm/yr$]", color="tab:blue", fontsize=self.fontsize_label,
            rotation="vertical", transform=ax[0, 0].transAxes, verticalalignment="center")
        ax[0, 0].text(-0.1, 0.5, "temp. [$°C$]", color="red", fontsize=self.fontsize_label,
            rotation="vertical", transform=ax[0, 0].transAxes, verticalalignment="center")
        ax[0, 0].xaxis.set_tick_params(labelsize=self.fontsize_ticks)
        ax[0, 0].yaxis.set_tick_params(labelsize=self.fontsize_ticks)

        self.plot(ax[0, 1], self.eros_mean, "erosion rate [$mm/yr$]")
        self.plot(ax[1, 0], self.co2_mean, "CO2 [$ppm$]")
        self.plot(ax[1, 1], self.sedi_mean, "sedi mean [$m^3/s$]")
        self.plot(ax[2, 0], self.soil_mean, "soil thickness [$m$]")
        self.plot(ax[2, 1], self.tree_mean_fpc, "tree FPC mean [%]")
        self.plot(ax[3, 0], self.grass_mean_fpc, "grass FPC mean [%]")
        self.plot(ax[3, 1], self.shrub_mean_fpc, "shrub FPC mean [%]")

        uplift_rate_ = [self.uplift_rate for i in self.elapsed_time]
        ax[0, 1].plot(self.elapsed_time, uplift_rate_, color="red", linestyle="--")

        ax[3, 0].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)
        ax[3, 1].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)

        if self.plot_time_type == "LGM":
            ax[0, 0].invert_xaxis()

        fig.suptitle(self.sup_title, fontsize=self.fontsize_label)

        plt.tight_layout(rect=self.rect)
        plt.savefig(filename, dpi=self.dpi)

    def plot2(self, filename):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize, sharex=True)

        self.plot(ax[0, 0], self.burned_area_frac, "burned area [%]")
        self.plot(ax[0, 1], self.vegi_mean_fpc, "vegi FPC mean [%]")
        self.plot(ax[1, 0], self.eros_mean, "erosion rate [$mm/yr$]")
        self.plot(ax[1, 1], self.vegi_mean_lai, "vegi LAI mean")
        self.plot(ax[2, 0], self.topo_mean, "mean elevation [$m$]")
        self.plot(ax[2, 1], self.tree_mean_lai, "tree LAI mean")
        self.plot(ax[3, 0], self.grass_mean_lai, "grass LAI mean")
        self.plot(ax[3, 1], self.shrub_mean_lai, "shrub LAI mean")

        ax[3, 0].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)
        ax[3, 1].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)

        if self.plot_time_type == "LGM":
            ax[0, 0].invert_xaxis()

        fig.suptitle(self.sup_title, fontsize=self.fontsize_label)

        plt.tight_layout(rect=self.rect)
        plt.savefig(filename, dpi=self.dpi)

    def plot3(self, filename):
        fig, ax = plt.subplots(4, 2, figsize=self.figsize, sharex=True)

        self.plot(ax[0, 0], self.runoff_mean, "runoff [?]")
        self.plot(ax[0, 1], self.net_primary_productivity, "net primary productivity [?]")
        self.plot(ax[1, 0], self.evapo_trans_soil, "evapo_trans_soil [?]")
        self.plot(ax[1, 1], self.evapo_trans_area, "evapo_trans_mean [?]")

        ax[3, 0].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)
        ax[3, 1].set_xlabel("{} [$kyr$]".format(self.time_label), fontsize=self.fontsize_label, color=self.color)

        if self.plot_time_type == "LGM":
            ax[0, 0].invert_xaxis()

        fig.suptitle(self.sup_title, fontsize=self.fontsize_label)

        plt.tight_layout(rect=self.rect)
        plt.savefig(filename, dpi=self.dpi)

    def plot4(self, filename):
        fig, ax = plt.subplots(2, 2, figsize=self.figsize, sharex=True, sharey=True)

        self.plot_elevation(ax[0, 0], self.map_elevation1)
        self.plot_elevation(ax[1, 0], self.map_elevation2)
        self.plot_erosion_rate(ax[0, 1], self.map_erosion_rate1)
        self.plot_erosion_rate(ax[1, 1], self.map_erosion_rate2)

        ax[0, 0].set_title("{}: {:.2f} [$kyr$]".format(self.time_label, self.elapsed_time[0]), fontsize=self.fontsize_label)
        ax[1, 0].set_title("{}: {:.2f} [$kyr$]".format(self.time_label, self.elapsed_time[-1]), fontsize=self.fontsize_label)

        ax[1, 0].set_xlabel("X($km$)", fontsize=self.fontsize_label, color=self.color)
        ax[1, 1].set_xlabel("X($km$)", fontsize=self.fontsize_label, color=self.color)

        ax[0, 0].set_ylabel("Y($km$)", fontsize=self.fontsize_label, color=self.color)
        ax[1, 0].set_ylabel("Y($km$)", fontsize=self.fontsize_label, color=self.color)

        fig.suptitle(self.sup_title, fontsize=self.fontsize_label)

        plt.tight_layout(rect=self.rect)
        plt.savefig(filename, dpi=self.dpi)

    def debug_output(self):
        print("elapsed_time: {}".format(len(self.elapsed_time)))

def extract_time(name):
    (name, _) = os.path.splitext(name)
    return int(name.split("__")[1])

if __name__ == "__main__":
    logging.basicConfig(filename="plot.log", level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d:%(funcName)s - %(message)s")

    config = configparser.ConfigParser()
    config.read("inputFile.ini")

    plot_start = float(config["Plot"]["plot_start"])
    plot_end = float(config["Plot"]["plot_end"])
    spin_up = float(config["Runtime"]["spin_up"])

    if plot_start < spin_up:
        step_size = float(config["Output"]["outIntSpinUp"])
        plot_start = int(round(plot_start / step_size) * step_size)
    else:
        step_size = float(config["Output"]["outIntTransient"])
        plot_start = int(round(plot_start / step_size) * step_size)

    if plot_end < spin_up:
        step_size = float(config["Output"]["outIntSpinUp"])
        plot_end = int(round(plot_end / step_size) * step_size)
    else:
        step_size = float(config["Output"]["outIntTransient"])
        plot_end = int(round(plot_end / step_size) * step_size)

    if plot_start >= plot_end:
        print("Error in input file 'inputFile.ini': plot_start must be smaller than plot_end!")
        print("plot_start: {} >= plot_end: {} ".format(plot_start, plot_end))
        sys.exit(1)

    logging.debug("plot_start: %f, plot_end: %f, spin_up: %f", plot_start, plot_end, spin_up)

    sim_data = SimData()

    uplift_rate = float(config["Uplift"]["upliftRate"])
    sim_data.set_uplift_rate(uplift_rate)

    node_spacing = float(config["Grid"]["dx"])
    sim_data.set_node_spacing(node_spacing)

    num_of_nodes = float(config["Grid"]["ncols"])
    sim_data.set_num_of_nodes(num_of_nodes)

    logging.debug("uplift_rate: %f, node_spacing: %f, num_of_nodes: %f", uplift_rate, node_spacing, num_of_nodes)

    all_files = glob.glob("ll_output/NC/*.nc")
    time_and_names = ((extract_time(name), name) for name in all_files)

    for (elapsed_time, nc_file) in tqdm(sorted(time_and_names)):
        if elapsed_time < plot_start or elapsed_time > plot_end:
            continue

        nc_data = netCDF4.Dataset(nc_file) # pylint: disable=no-member

        if plot_start == elapsed_time:
            sim_data.set_map_elevation1(nc_data.variables["topographic__elevation"][:][0])
            sim_data.set_map_erosion1(nc_data.variables["erosion__rate"][:][0])
        elif plot_end == elapsed_time:
            sim_data.set_map_elevation2(nc_data.variables["topographic__elevation"][:][0])
            sim_data.set_map_erosion2(nc_data.variables["erosion__rate"][:][0])

        sim_data.append("elapsed_time", elapsed_time)

        logging.debug("elapsed_time: %d, nc_file: %s", elapsed_time, nc_file)

        for p_name, p in nc_data.variables.items():
            if not hasattr(p, "shape"):
                continue

            logging.debug("parameter: %s, shape: %s", p_name, p.shape)

            if p.shape == (1, num_of_nodes, num_of_nodes):
                parameter_data = p[:][0]
                # delete boundary nodes
                parameter_data = np.delete(parameter_data, 0, axis=0)
                parameter_data = np.delete(parameter_data, -1, axis=0)
                parameter_data = np.delete(parameter_data, 0, axis=1)
                parameter_data = np.delete(parameter_data, -1, axis=1)

                sim_data.append(p_name, np.mean(parameter_data))


    plot_title = config["Plot"].get("plot_title", "")
    plot_sup_title = config["Plot"].get("plot_sup_title", "")
    sim_data.set_title(plot_title, plot_sup_title)

    file_type = config["Plot"].get("plot_file_type", "png")

    logging.debug("plot_title: %s, plot_sup_title: %s, file_type: %s", plot_title, plot_sup_title, file_type)

    sim_data.set_plot_time_type("Normal")
    sim_data.plot1("overview1_elapsed_time.{}".format(file_type))
    sim_data.plot2("overview2_elapsed_time.{}".format(file_type))
    sim_data.plot3("overview3_elapsed_time.{}".format(file_type))
    sim_data.plot4("overview4_elapsed_time.{}".format(file_type))

    sim_data.set_plot_time_type("LGM")
    sim_data.plot1("overview1_time_before_pd.{}".format(file_type))
    sim_data.plot2("overview2_time_before_pd.{}".format(file_type))
    sim_data.plot3("overview3_time_before_pd.{}".format(file_type))
    sim_data.plot4("overview4_time_before_pd.{}".format(file_type))
