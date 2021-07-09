import logging

import numpy as np
import pandas as pd

# set of scripts which makes post-processed lpj-output landlab compatible

NODE_STR = "node"
YEAR_STR = "Year"
STAND_STR = "Stand"
TOTAL_STR = "Total"
LANDFORM_ID_STR = "landform__ID"

def _calc_fpc(lai):
    """Calculate FPC using the LPJ-GUESS method

    """
    return 1.0 - np.exp(-0.5 * lai)

def map_data_per_landform_on_grid(grid, data_array, data_name):
    data_grid = np.zeros(np.shape(grid.at_node[data_name]))

    if data_array is not None:
        for landform in data_array.dtype.names[1:]:
            data_grid[grid.at_node[LANDFORM_ID_STR] == int(landform)] = data_array[landform]
            logging.debug("create_input_for_landlab, landform: {}".format(landform))

    return data_grid

def map_vegi_per_landform_on_grid(grid, vegi_array):
    """
    extract the tree fractional cover per landform

    assumes that the landlab grid object which is passed already
    has a data field "landform__id" which is used to create
    numpy arrays with correct dimensions for mapping vegetation
    data
    """

    return map_data_per_landform_on_grid(grid, vegi_array, LANDFORM_ID_STR)

# For precipitation:
# Extract the precipitation values per landform and maps it in the "precipitation"
# datafield of the landlab grid object
#
# Right now (15.11.2018) this method is a little bit overkill because we don"t have
# spatial variable rainfall. But for future uses with a more precise downscaling or
# bigger grids, this would be important.

def process_vegetation_data(data, index_cols, other_cols):
    data_filtered = data[index_cols + other_cols].groupby(index_cols, sort=False).mean()
    fpc_data = data_filtered.apply(_calc_fpc, 1).sum(axis=1)
    fpc_data = fpc_data.reset_index().set_index(index_cols)
    fpc_data = fpc_data.mean(level=1).T

    lai_data = data_filtered.sum(axis=1)
    lai_data = lai_data.reset_index().set_index(index_cols)
    lai_data = lai_data.mean(level=1).T

    return (fpc_data.to_records(), lai_data.to_records())

def import_vegetation(grid, vegi_mapping_method, filename):
    csv_data = pd.read_table(filename, delim_whitespace=True)
    csv_data = csv_data[csv_data.Stand > 0]
    index_cols = [YEAR_STR, STAND_STR]

    total_col = [TOTAL_STR]

    (total_fpc, total_lai) = process_vegetation_data(csv_data, index_cols, total_col)

    if "vegetation__density" not in grid.keys(NODE_STR):
        grid.add_zeros(NODE_STR, "vegetation__density")
    if "vegetation__density_lai" not in grid.keys(NODE_STR):
        grid.add_zeros(NODE_STR, "vegetation__density_lai")

    grid.at_node["vegetation__density"] = map_vegi_per_landform_on_grid(grid, total_fpc)
    grid.at_node["vegetation__density_lai"] = map_vegi_per_landform_on_grid(grid, total_lai)


    if vegi_mapping_method == "individual":
        tree_cols = ["TeBE_tm", "TeBE_itm", "TeBE_itscl", "TeBS_tm", "TeBS_itm", "TeNE", "BBS_itm", "BBE_itm"]
        shrub_cols = ["BE_s", "TeR_s", "TeE_s"]
        grass_cols = ["C3G"]

        (tree_fpc, tree_lai) = process_vegetation_data(csv_data, index_cols, tree_cols)
        (shrub_fpc, shrub_lai) = process_vegetation_data(csv_data, index_cols, shrub_cols)
        (grass_fpc, grass_lai) = process_vegetation_data(csv_data, index_cols, grass_cols)

        if "tree_fpc" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "tree_fpc")
        if "tree_lai" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "tree_lai")

        grid.at_node["tree_fpc"] = map_vegi_per_landform_on_grid(grid, tree_fpc)
        grid.at_node["tree_lai"] = map_vegi_per_landform_on_grid(grid, tree_lai)

        if "shrub_fpc" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "shrub_fpc")
        if "shrub_lai" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "shrub_lai")

        grid.at_node["shrub_fpc"] = map_vegi_per_landform_on_grid(grid, shrub_fpc)
        grid.at_node["shrub_lai"] = map_vegi_per_landform_on_grid(grid, shrub_lai)

        if "grass_fpc" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "grass_fpc")
        if "grass_lai" not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, "grass_lai")

        grid.at_node["grass_fpc"] = map_vegi_per_landform_on_grid(grid, grass_fpc)
        grid.at_node["grass_lai"] = map_vegi_per_landform_on_grid(grid, grass_lai)


def import_csv_data(grid, filename, data_name, factor=None):
    csv_data = pd.read_table(filename, delim_whitespace=True)

    month_cols = "Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec".split(",")
    index_cols = [YEAR_STR, STAND_STR]

    filtered_data = csv_data[index_cols + month_cols][csv_data.Stand > 0]
    filtered_data["Annual"] = filtered_data[month_cols].sum(axis=1)

    cleared_data = filtered_data.drop(columns=month_cols).set_index(index_cols)

    if factor:
        final_data = cleared_data.mean(level=1).T / factor
    else:
        final_data = cleared_data.mean(level=1).T

    if data_name not in grid.keys(NODE_STR):
        grid.add_zeros(NODE_STR, data_name)

    grid.at_node[data_name] = map_data_per_landform_on_grid(grid, final_data.to_records(), data_name)

def import_precipitation(grid, filename):
    import_csv_data(grid, filename, "precipitation", 10.0)

def import_temperature(grid, filename):
    import_csv_data(grid, filename, "temperature", 10.0)

def import_radiation(grid, filename):
    import_csv_data(grid, filename, "radiation")

def import_co2(grid, filename):
    co2_name = "co2"
    csv_data = pd.read_table(filename, delim_whitespace=True)
    co2_values = csv_data[co2_name]
    co2_value = co2_values.mean()


    if co2_name not in grid.keys(NODE_STR):
        grid.add_zeros(NODE_STR, co2_name)

    shape = np.shape(grid.at_node[co2_name])
    grid.at_node[co2_name] = np.full(shape, co2_value)

def import_fire(grid, filename):
    """
    Read LPJ-GUESS fire related output.

    LPJ-GUESS outputs a fire return interval in years
    (which is the reciprocal of the fractional burned area)
    for every year, Stand (containing the landform__ID), and Patch.
    These intervals are converted back into fractional area burned
    and averaged per landform__ID.
    """
    try:
        data = pd.read_table(filename, delim_whitespace=True)
        data = data[data.Stand > 0]

        assert len(data[["Lon", "Lat"]].drop_duplicates()) == 1, "Data must not contain more than one (Lat, Lon) combination: {}".format(filename)

        baf_name = "burned_area_frac"

        data[baf_name] = 1.0 / data.FireRT # convert return time back into burned area fraction
        data_mean = data[[STAND_STR, baf_name]].groupby(STAND_STR, sort=False).mean() # average burned area over years and patches

        if baf_name not in grid.keys(NODE_STR):
            grid.add_zeros(NODE_STR, baf_name)

        grid.at_node[baf_name] = map_data_per_landform_on_grid(grid, data_mean.T.to_records(), baf_name)
    except FileNotFoundError:
        logging.error("Could not open file '%s' for burned_area values", filename)

def import_runoff(grid, filename):
    import_csv_data(grid, filename, "runoff")

def import_evapo_trans_soil(grid, filename):
    import_csv_data(grid, filename, "evapo_trans_soil")

def import_evapo_trans_area(grid, filename):
    csv_data = pd.read_table(filename, delim_whitespace=True)
    csv_data = csv_data[csv_data.Stand > 0]
    index_cols = [YEAR_STR, STAND_STR]

    total_col = [TOTAL_STR]

    data_filtered = csv_data[index_cols + total_col].groupby(index_cols, sort=False).mean()

    et_area = data_filtered.sum(axis=1)
    et_area = et_area.reset_index().set_index(index_cols)
    et_area = et_area.mean(level=1).T

    et_name = "evapo_trans_area"

    if et_name not in grid.keys(NODE_STR):
        grid.add_zeros(NODE_STR, et_name)

    grid.at_node[et_name] = map_data_per_landform_on_grid(grid, et_area.to_records(), et_name)

def import_npp(grid, filename):
    import_csv_data(grid, filename, "net_primary_productivity")

def lpj_import_one_step(grid, vegi_mapping_method):
    """
    main function for input_conversion to be called from landlab driver file
    """

    import_vegetation(grid, vegi_mapping_method, "temp_lpj/output/sp_lai.out")
    import_precipitation(grid, "temp_lpj/output/sp_mprec.out")
    import_temperature(grid, "temp_lpj/output/sp_mtemp.out")
    import_radiation(grid, "temp_lpj/output/sp_mrad.out")
    import_co2(grid, "temp_lpj/output/climate.out")
    import_fire(grid, "temp_lpj/output/sp_firert.out")

    import_runoff(grid, "temp_lpj/output/sp_mrunoff.out")
    import_evapo_trans_soil(grid, "temp_lpj/output/sp_mevap.out")
    import_evapo_trans_area(grid, "temp_lpj/output/sp_aaet.out")
    import_npp(grid, "temp_lpj/output/sp_mnpp.out")
