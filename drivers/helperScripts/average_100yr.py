import xarray as xr
import logging

import numpy as np

YEAR_IN_DAYS = 365.2425
# YEAR_IN_DAYS = 362.40
YEARS_100 = YEAR_IN_DAYS * 100.0

class SimulationData:
    def __init__(self, filename, parameter):
        self.filename = filename
        self.parameter = parameter
        self.ds = xr.open_dataset(self.filename, decode_times=False)
        self.data = self.ds.variables[parameter]
        self.days = self.ds.variables["time"]

        logging.info("Mean of '%s': %f", self.parameter, np.mean(self.data))

    def avg_intervall(self, first_day, last_day):
        duration = last_day - first_day
        dayid = (self.days >= first_day) & (self.days <= last_day)
        logging.info("Average over %.0f days (%.0f years) for '%s': %f (first: %.0f)", duration, duration / YEAR_IN_DAYS,
            self.parameter, np.mean(self.data[:, dayid]), first_day)

    def avg_first_100_years(self):
        self.avg_intervall(0, YEARS_100)

    def avg_last_100_years(self):
        last = self.days[-1].item()
        self.avg_intervall(last - YEARS_100, last)

    def avg_first_and_last(self):
        self.avg_first_100_years()
        self.avg_last_100_years()

if __name__ == "__main__":
    logging.basicConfig(filename="average.log", level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sd = SimulationData("Nahuelbuta_TraCE21ka_prec.nc", "prec")
    sd.avg_first_and_last()

    sd = SimulationData("Nahuelbuta_TraCE21ka_temp.nc", "temp")
    sd.avg_first_and_last()

    sd = SimulationData("Nahuelbuta_TraCE21ka_rad.nc", "rad")
    sd.avg_first_and_last()
