import numpy as np
import pandas as pd

"""
set of scripts which makes post-processed lpj-output landlab compatible
"""

def _calc_fpc(lai):
    """Calculate FPC using the LPJ-GUESS method

    """
    return (1.0 - np.exp(-0.5 * lai)) * 100

def map_data_per_landform_on_grid(grid, data_array, data_name):
    data_grid = np.zeros(np.shape(grid.at_node[data_name]))

    if data_array is not None:
        for landform in data_array.dtype.names[1:]:
            data_grid[grid.at_node['landform__ID'] == int(landform)] = data_array[landform]

    return data_grid

def map_vegi_per_landform_on_grid(grid, vegi_array):
    """
    extract the tree fractional cover per landform
    
    assumes that the landlab grid object which is passed already
    has a data field 'landform__id' which is used to create
    numpy arrays with correct dimensions for mapping vegetation
    data
    """

    return map_data_per_landform_on_grid(grid, vegi_array, "landform__ID")

# For precipitation:
# Extract the precipitation values per landform and maps it in the 'precipitation'
# datafield of the landlab grid object
#
# Right now (15.11.2018) this method is a little bit overkill because we don't have
# spatial variable rainfall. But for future uses with a more precise downscaling or 
# bigger grids, this would be important.

def process_vegetation_data(data, index_cols, other_cols):
    data_filtered = data[index_cols + other_cols].groupby(index_cols, sort = False).mean()
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
    index_cols = ['Year', 'Stand']

    total_col = ['Total']

    (total_fpc, total_lai) = process_vegetation_data(csv_data, index_cols, total_col)

    if 'vegetation__density' not in grid.keys('node'):
        grid.add_zeros('node', 'vegetation__density')
    if 'vegetation__density_lai' not in grid.keys('node'):
        grid.add_zeros('node', 'vegetation__density_lai')

    grid.at_node['vegetation__density'] = map_vegi_per_landform_on_grid(grid, total_fpc)
    grid.at_node['vegetation__density_lai'] = map_vegi_per_landform_on_grid(grid, total_lai)


    if vegi_mapping_method == 'individual':
        tree_cols = ['TeBE_tm','TeBE_itm','TeBE_itscl','TeBS_itm','TeNE','BBS_itm','BBE_itm']
        shrub_cols = ['BE_s','TeR_s','TeE_s']
        grass_cols = ['C3G']

        (tree_fpc, tree_lai) = process_vegetation_data(csv_data, index_cols, tree_cols)
        (shrub_fpc, shrub_lai) = process_vegetation_data(csv_data, index_cols, shrub_cols)
        (grass_fpc, grass_lai) = process_vegetation_data(csv_data, index_cols, grass_cols)

        if 'tree_fpc' not in grid.keys('node'):
            grid.add_zeros('node', 'tree_fpc')
        if 'tree_lai' not in grid.keys('node'):
            grid.add_zeros('node', 'tree_lai')

        grid.at_node['tree_fpc'] = map_vegi_per_landform_on_grid(grid, tree_fpc)
        grid.at_node['tree_lai'] = map_vegi_per_landform_on_grid(grid, tree_lai)

        if 'shrub_fpc' not in grid.keys('node'):
            grid.add_zeros('node', 'shrub_fpc')
        if 'shrub_lai' not in grid.keys('node'):
            grid.add_zeros('node', 'shrub_lai')

        grid.at_node['shrub_fpc']  = map_vegi_per_landform_on_grid(grid, shrub_fpc)
        grid.at_node['shrub_lai']  = map_vegi_per_landform_on_grid(grid, shrub_lai)

        if 'grass_fpc' not in grid.keys('node'):
            grid.add_zeros('node', 'grass_fpc')
        if 'grass_lai' not in grid.keys('node'):
            grid.add_zeros('node', 'grass_lai')

        grid.at_node['grass_fpc'] = map_vegi_per_landform_on_grid(grid, grass_fpc)
        grid.at_node['grass_lai'] = map_vegi_per_landform_on_grid(grid, grass_lai)


def import_csv_data(grid, filename, data_name):
    csv_data = pd.read_table(filename, delim_whitespace=True)

    month_cols = "Jan,Feb,Mar,Apr,May,Jun,Jul,Aug,Sep,Oct,Nov,Dec".split(',')
    index_cols = ["Year", "Stand"]

    filtered_data = csv_data[index_cols + month_cols][csv_data.Stand > 0]
    filtered_data["Annual"] = filtered_data[month_cols].sum(axis = 1)

    cleared_data = filtered_data.drop(columns=month_cols).set_index(index_cols)

    final_data = cleared_data.mean(level=1).T / 10.0

    if data_name not in grid.keys('node'):
        grid.add_zeros('node', data_name)
    
    grid.at_node[data_name] = map_data_per_landform_on_grid(grid, final_data.to_records(), data_name)

def import_precipitation(grid, filename):
    import_csv_data(grid, filename, "precipitation")

def import_temperature(grid, filename):
    import_csv_data(grid, filename, "temperature")

def lpj_import_run_one_step(grid, vegi_mapping_method):
    """
    main function for input_conversion to be called from landlab driver file
    """

    import_vegetation(grid, vegi_mapping_method, "temp_lpj/output/sp_lai.out")
    import_precipitation(grid, "temp_lpj/output/sp_mprec.out")
    import_temperature(grid, "temp_lpj/output/sp_mtemp.out")
