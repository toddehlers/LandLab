import numpy as np
import netCDF4

def gen_data(file_name_base, lat_val, lon_val, var_name, var_value, var_description, var_long_name, var_units):
    file_name = file_name_base + "_" + var_name + ".nc"
    f = netCDF4.Dataset(file_name, "w", format = "NETCDF4")

    num_of_elements = 365 * 1000

    f.createDimension("time", num_of_elements)
    f.createDimension("land_id", 1)

    num_type = "f8"

    time = f.createVariable("time", num_type, "time")
    lat = f.createVariable("lat", num_type, "land_id")
    lon = f.createVariable("lon", num_type, "land_id")
    var_instance = f.createVariable(var_name, num_type, ("land_id", "time"))
    land_id = f.createVariable("land_id", "i8", "land_id")

    time[:] = np.arange(0, num_of_elements, 1)
    lat[:] = np.full(1, lat_val)
    lon[:] = np.full(1, lon_val)
    var_instance[:] = np.full(num_of_elements, var_value)
    land_id[:] = np.full(1, 0)

    time.axis = "T"
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "day"

    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"

    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"

    var_instance.coordinates = "lon lat"
    var_instance.standard_name = var_description
    var_instance.long_name = var_long_name
    var_instance.units = var_units

    f.close()

def gen_data_for_location(file_name_base, lat, lon, prec, temp, rad):
    gen_data(file_name_base, lat, lon, "prec", prec, "precipitation amount", "daily precipitation amount", "mm per day")
    gen_data(file_name_base, lat, lon, "temp", temp, "temperature", "temperature", "K")
    gen_data(file_name_base, lat, lon, "rad", rad, "radiation", "radiation", "W m-2")

gen_data_for_location("LaCampana_LGM", lat = -32.75, lon = -71.25, prec = 0.92, temp = 284.6, rad = 249.1)
