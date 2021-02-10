import itertools
import operator

import numpy as np
import netCDF4

NUM_OF_YEARS = 22000
NUM_OF_MONTHS = 12 * NUM_OF_YEARS
NUM_OF_DAYS = 365 * NUM_OF_YEARS

# Please set mode accordingly
mode = "monthly"
# mode = "daily"

if mode == "monthly":
    NUM_OF_ELEMENTS = NUM_OF_MONTHS
elif mode == "daily":
    NUM_OF_ELEMENTS = NUM_OF_DAYS
else:
    raise Exception("Unknown mode: {}".format(mode))

def gen_common(file_name_base, lat_val, lon_val, var_name):
    file_name = "{}_{}.nc".format(file_name_base, var_name)
    f = netCDF4.Dataset(file_name, "w", format = "NETCDF4") # pylint: disable=no-member

    f.createDimension("time", NUM_OF_ELEMENTS)
    f.createDimension("land_id", 1)

    num_type = "f8"

    time = f.createVariable("time", num_type, "time")
    lat = f.createVariable("lat", num_type, "land_id")
    lon = f.createVariable("lon", num_type, "land_id")
    land_id = f.createVariable("land_id", "i8", "land_id")

    if mode == "monthly":
        time[:] = np.arange(0, NUM_OF_ELEMENTS, 1)
    elif mode == "daily":
        days = itertools.cycle([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
        days = itertools.chain([0], days)
        days = itertools.accumulate(days, operator.add)
        time[:] = np.array(list(itertools.islice(days, NUM_OF_ELEMENTS)))
    else:
        raise Exception("Unknown mode: {}".format(mode))

    lat[:] = np.full(1, lat_val)
    lon[:] = np.full(1, lon_val)
    land_id[:] = np.full(1, 0)

    time.axis = "T"
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "day"
    time.calendar = "{} yr B.P.".format(NUM_OF_YEARS)

    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"

    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"

    return f

def gen_var_data(f, var_name, var_value, var_description,
             var_long_name, var_units, var_code = None):
    num_type = "f8"

    var_instance = f.createVariable(var_name, num_type, ("land_id", "time"))
    var_instance[:] = np.full(NUM_OF_ELEMENTS, var_value)
    var_instance.coordinates = "lat lon"
    var_instance.standard_name = var_description
    var_instance.long_name = var_long_name
    var_instance.units = var_units
    if var_code:
        var_instance.code = var_code
    var_instance.table = "128"


def gen_data_for_location(file_name_base, lat, lon, prec, wet, temp, rad):
    f = gen_common(file_name_base, lat, lon, "prec")
    gen_var_data(f, "prec", prec, "precipitation_amount", "daily precipitation amount", "kg m-2", "260")
    gen_var_data(f, "wet", wet, "number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold", "wet_days", "count")
    f.close()

    f = gen_common(file_name_base, lat, lon, "temp")
    gen_var_data(f, "temp", temp, "air_temperature", "Near surface air temperature at 2m", "K", "167")
    f.close()

    f = gen_common(file_name_base, lat, lon, "rad")
    gen_var_data(f, "rad", rad, "surface_downwelling_shortwave_flux", "Mean daily surface incident shortwave radiation", "W m-2", "176")
    f.close()

gen_data_for_location("LaCampana_LGM", lat = -32.75, lon = -71.25, prec = 0.92, wet = 8.24, temp = 284.6, rad = 249.1)

with open("co2_data.txt", "w+") as f:
    for i in range(1, NUM_OF_ELEMENTS):
        f.write("{} 200\n".format(i))
