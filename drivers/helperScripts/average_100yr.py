import xarray as xr
import numpy as np
from statistics import mean

df = xr.open_dataset('Nahuelbuta_TraCE21ka_prec.nc', decode_times=False)['prec'][:].to_dataframe()
print(df)

ds = xr.open_dataset(f"Nahuelbuta_TraCE21ka_prec.nc", decode_times=False)
print("Mean of precipitation : " , np.mean(ds.prec))

prec = ds.variables['prec']
day = ds.variables['time']
dayid = (day >=0 ) & (day <=36240 )
precN = prec[:]
precN = precN[:, dayid]
precN_1 = precN
print ("100 years LGM precipitation: ", np.mean(precN_1))

dayid = (day >=8008329 ) & (day <=8044569 )
precN = prec[:]
precN = precN[:, dayid]
precN_1 = precN
print ("Last 100 years precipitation: ", np.mean(precN_1))



ds = xr.open_dataset(f"Nahuelbuta_TraCE21ka_temp.nc", decode_times=False)
print("Mean of tenperature : " , np.mean(ds.temp))

temp = ds.variables['temp']
day = ds.variables['time']
dayid = (day >=0 ) & (day <=36240 )
tempN = temp[:]
tempN = tempN[:, dayid]
tempN_1 = tempN
print ("100 years LGM temperature: ", np.mean(tempN_1))

dayid = (day >=8008329 ) & (day <=8044569 )
tempN = temp[:]
tempN = tempN[:, dayid]
tempN_1 = tempN
print ("Last 100 years temperature: ", np.mean(tempN_1))



ds = xr.open_dataset(f"Nahuelbuta_TraCE21ka_rad.nc", decode_times=False)
print("Mean of radiation : " , np.mean(ds.rad))

rad = ds.variables['rad']
day = ds.variables['time']
dayid = (day >=0 ) & (day <=36240 )
radN = rad[:]
radN = radN[:, dayid]
radN_1 = radN
print ("100 years LGM radiation: ", np.mean(radN_1))

dayid = (day >=8008329 ) & (day <=8044569 )
radN = rad[:]
radN = radN[:, dayid]
radN_1 = radN
print ("Last 100 years radiation: ", np.mean(radN_1))
