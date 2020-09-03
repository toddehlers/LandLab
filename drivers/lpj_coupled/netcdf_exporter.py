import shutil

from landlab.io.netcdf import write_netcdf

class NetCDFExporter:
    def __init__(self, latitude, longitude, dx, spin_up, classificationType, elevationStepBin):
        self.latitude = latitude
        self.longitude = longitude
        self.dx = dx
        self.dy = dx # In all the simulation dx and dy have the same values
        self.spin_up = spin_up
        self.classificationType = classificationType
        self.elevationStepBin = elevationStepBin
        # TODO: Make path configurable
        self.tmp_output_file = "temp_output/current_output.nc"
        self.permanent_output_prefix = "ll_output/NC/output"

    def write(self, mg, elapsed_time):
        write_netcdf(self.tmp_output_file,
            mg, format="NETCDF4", attrs =
                {"lgt.lat" : self.latitude,
                 "lgt.lon" : self.longitude,
                 "lgt.dx"  : self.dx,
                 "lgt.dy"  : self.dx,
                 "lgt.timestep" : elapsed_time,
                 "lgt.spinup" : int(elapsed_time < self.spin_up),
                 "lgt.classification" : self.classificationType,
                 "lgt.elevation_step" : self.elevationStepBin})

    def write_permanent(self, mg, elapsed_time):
        self.write(mg, elapsed_time)
        shutil.copy(self.tmp_output_file, "{}__{}.nc".format(self.permanent_output_prefix, elapsed_time))
