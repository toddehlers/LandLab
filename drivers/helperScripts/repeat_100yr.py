import itertools
import operator

import numpy as np
import netCDF4

# variables:
#         double time(time) ;
#                 time:_FillValue = NaN ;
#                 time:units = "days since 1-1-15 00:00:00" ;
#                 time:axis = "T" ;
#                 time:long_name = "time" ;
#                 time:standard_name = "time" ;
#                 time:calendar = "22000 yr B.P." ;
#         double lat(land_id) ;
#                 lat:_FillValue = NaN ;
#                 lat:standard_name = "latitude" ;
#                 lat:long_name = "latitude" ;
#                 lat:units = "degrees_north" ;
#         double lon(land_id) ;
#                 lon:_FillValue = NaN ;
#                 lon:standard_name = "longitude" ;
#                 lon:long_name = "longitude" ;
#                 lon:units = "degrees_east" ;
#         double temp(land_id, time) ;
#                 temp:_FillValue = -9999. ;
#                 temp:standard_name = "air_temperature" ;
#                 temp:long_name = "Near surface air temperature at 2m" ;
#                 temp:units = "K" ;
#                 temp:coordinates = "lon lat" ;
#         int64 land_id(land_id) ;
# data:

#        double rad(land_id, time) ;
#                rad:_FillValue = -9999. ;
#                rad:standard_name = "surface_downwelling_shortwave_flux" ;
#                rad:long_name = "Mean daily surface incident shortwave radiation" ;
#                rad:units = "W m-2" ;
#                rad:coordinates = "lon lat" ;

#        double prec(land_id, time) ;
#                prec:_FillValue = -9999. ;
#                prec:standard_name = "precipitation_amount" ;
#                prec:long_name = "Monthly precipitation amount" ;
#                prec:units = "kg m-2" ;
#                prec:coordinates = "lon lat" ;
#        double wet(land_id, time) ;
#                wet:_FillValue = -9999. ;
#                wet:standard_name = "number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold" ;
#                wet:long_name = "wet_days" ;
#                wet:units = "count" ;
#                wet:coordinates = "lon lat" ;



def extract_and_repeat(file_prefix, first_day, last_day, count):
    prec_ds = netCDF4.Dataset("{}_prec.nc".format(file_prefix), "r") # pylint: disable=no-member
    temp_ds = netCDF4.Dataset("{}_temp.nc".format(file_prefix), "r") # pylint: disable=no-member
    rad_ds = netCDF4.Dataset("{}_rad.nc".format(file_prefix), "r") # pylint: disable=no-member

    num_of_elements = prec_ds.variables["time"].shape[0]

    print("num_of_elements: {}".format(num_of_elements))

    longitude = prec_ds["lon"][0].item()
    latitude = prec_ds["lat"][0].item()
    land_id = int(prec_ds["land_id"][0].item())

    print("longitude: {}".format(longitude))
    print("latitude: {}".format(latitude))
    print("land_id: {}".format(land_id))

    prec_data = []
    wet_data = []
    temp_data = []
    rad_data = []

    for i in range(0, num_of_elements):
        current_time = prec_ds["time"][i].item()
        if current_time >= first_day:
            if current_time <= last_day:
                prec_data.append(prec_ds["prec"][0][i].item())
                wet_data.append(prec_ds["wet"][0][i].item())
                temp_data.append(temp_ds["temp"][0][i].item())
                rad_data.append(rad_ds["rad"][0][i].item())
            else:
                break

    prec_ds.close()
    temp_ds.close()
    rad_ds.close()

    duration = last_day - first_day
    elements_in_list = len(prec_data)
    number_of_elements_new = int(elements_in_list * count)

    print("duration: {}".format(duration))
    print("elements_in_list: {}".format(elements_in_list))
    print("number_of_elements_new: {}".format(number_of_elements_new))

    days = itertools.cycle([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    days = itertools.chain([0], days)
    days = itertools.accumulate(days, operator.add)
    days = list(itertools.islice(days, number_of_elements_new))

    prec_data = itertools.cycle(prec_data)
    prec_data = list(itertools.islice(prec_data, number_of_elements_new))

    wet_data = itertools.cycle(wet_data)
    wet_data = list(itertools.islice(wet_data, number_of_elements_new))

    export_data("{}_repeat_prec.nc".format(file_prefix), days, longitude, latitude, land_id,
    [(prec_data, "prec", "precipitation_amount", "Monthly precipitation amount", "kg m-2"),
     (wet_data, "wet", "number_of_days_with_lwe_thickness_of_precipitation_amount_above_threshold", "wet_days", "count")])

    del prec_data
    del wet_data

    temp_data = itertools.cycle(temp_data)
    temp_data = list(itertools.islice(temp_data, number_of_elements_new))

    export_data("{}_repeat_temp.nc".format(file_prefix), days, longitude, latitude, land_id,
    [(temp_data, "temp", "air_temperature", "Near surface air temperature at 2m", "K")])

    del temp_data

    rad_data = itertools.cycle(rad_data)
    rad_data = list(itertools.islice(rad_data, number_of_elements_new))

    export_data("{}_repeat_rad.nc".format(file_prefix), days, longitude, latitude, land_id,
    [(rad_data, "rad", "surface_downwelling_shortwave_flux", "Mean daily surface incident shortwave radiation", "W m-2")])

    del rad_data

def export_data(filename, days, longitude, latitude, land_id_orig, data):
    num_of_elements = len(days)
    netcdf_out_ds = netCDF4.Dataset(filename, "w", format = "NETCDF4") # pylint: disable=no-member

    print("num_of_elements: {}".format(num_of_elements))
    print("filename: {}".format(filename))

    netcdf_out_ds.createDimension("time", num_of_elements)
    netcdf_out_ds.createDimension("land_id", 1)

    num_type = "f8"

    time = netcdf_out_ds.createVariable("time", num_type, "time", fill_value="NaN")
    lat = netcdf_out_ds.createVariable("lat", num_type, "land_id", fill_value="NaN")
    lon = netcdf_out_ds.createVariable("lon", num_type, "land_id", fill_value="NaN")
    land_id = netcdf_out_ds.createVariable("land_id", "i8", "land_id")

    time[:] = np.array(days)

    lat[:] = np.full(1, latitude)
    lon[:] = np.full(1, longitude)
    land_id[:] = np.full(1, land_id_orig)

    time.axis = "T"
    time.standard_name = "time"
    time.long_name = "time"
    time.units = "day"
    time.calendar = "{} yr B.P.".format(num_of_elements / 12)

    lat.standard_name = "latitude"
    lat.long_name = "latitude"
    lat.units = "degrees_north"

    lon.standard_name = "longitude"
    lon.long_name = "longitude"
    lon.units = "degrees_east"

    for d in data:
        var_value = d[0]
        var_name = d[1]
        var_description = d[2]
        var_long_name = d[3]
        var_units = d[4]

        var_instance = netcdf_out_ds.createVariable(var_name, num_type, ("land_id", "time"), fill_value="-9999.0")
        var_instance[:] = np.full(num_of_elements, var_value)
        var_instance.coordinates = "lon lat"
        var_instance.standard_name = var_description
        var_instance.long_name = var_long_name
        var_instance.units = var_units

    netcdf_out_ds.close()

if __name__ == "__main__":
    extract_and_repeat("Nahuelbuta_TraCE21ka", 0, int(365.2425 * 100), 220)
