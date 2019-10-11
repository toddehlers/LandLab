import numpy as np
import netCDF4

def gen_data(file_name_base, lat, lon, var_name, var_value, var_description, var_long_name, var_units):
    file_name = file_name_base + "_" + var_name + ".nc"
    f = netCDF4.Dataset(file_name, "w", format = "NETCDF4")

    num_of_elements = 365 * 1000

    f.createDimension("dim_lat", 1)
    f.createDimension("dim_lon", 1)
    f.createDimension("dim_time", num_of_elements)
    f.createDimension("dim_prec", num_of_elements)

    num_type = "f4"

    lat = f.createVariable("lat", num_type, "dim_lat")
    lon = f.createVariable("lon", num_type, "dim_lon")
    time = f.createVariable("time", num_type, "dim_time")
    var_instance = f.createVariable(var_name, num_type, "dim_prec")

    lat[:] = np.full(1, lat)
    lon[:] = np.full(1, lon)
    time[:] = np.arange(0, num_of_elements, 1)
    var_instance[:] = np.full(num_of_elements, var_value)

    time.axis = "T"
    var_instance.coordinates = "lat lon"

    lat.standard_name = "latitude"
    lon.standard_name = "longitude"
    time.standard_name = "time"
    var_instance.standard_name = var_description

    lat.long_name = "latitude"
    lon.long_name = "longitude"
    time.long_name = "time"
    var_instance.long_name = var_long_name

    lat.units = "degrees_north"
    lon.units = "degrees_east"
    time.units = "day"
    var_instance.units = var_units

    f.close()

def gen_data_for_location(file_name_base, lat, lon, prec, temp, rad):
    gen_data(file_name_base, lat, lon, "prec", prec, "precipitation amount", "daily precipitation amount", "mm per day")
    gen_data(file_name_base, lat, lon, "temp", temp, "temperature", "temperature", "K")
    gen_data(file_name_base, lat, lon, "rad", rad, "radiation", "radiation", "W m-2")

gen_data_for_location("LaCampana_LGM", lat = -32.75, lon = -71.25, prec = 0.92, temp = 284.6, rad = 249.1)
