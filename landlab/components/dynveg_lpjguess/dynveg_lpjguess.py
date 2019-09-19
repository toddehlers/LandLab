import coloredlogs
from enum import Enum
import glob
from landlab import Component
import logging
import numpy as np
import os
import xarray as xr
import shutil
from string import Template
import subprocess
import sys
import time
from tqdm import tqdm
from typing import Dict, List, Optional

# this is a bit hacky - import scripts main as a function and use here
# should be refactored into this script
from .create_input_for_lpjguess import main as create_input_main


# logging setup
logPath = '.'
fileName = 'dynveg_lpjguess'

FORMAT="%(levelname).1s %(asctime)s %(filename)s:%(lineno)s - %(funcName).15s :: %(message)s"
logging.basicConfig(
    level=logging.DEBUG,
    format = FORMAT,
    handlers=[
        logging.FileHandler("{0}/{1}.log".format(logPath, fileName)),
        logging.StreamHandler()
    ])

log = logging.getLogger(__name__)
coloredlogs.install(level='CRITICAL', fmt=FORMAT, datefmt="%H:%M:%S")

def add_time_attrs(ds, calendar_year=0):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" 
    ds['time'].attrs['axis'] = "T" 
    ds['time'].attrs['long_name'] = "time" 
    ds['time'].attrs['standard_name'] = "time" 
    ds['time'].attrs['calendar'] = "%d yr B.P." % calendar_year


def fill_template(template: str, data: Dict[str, str]) -> str:
    """Fill template file with specific data from dict"""
    log.debug('Fill LPJ-GUESS ins template')
    with open( template, 'rU' ) as f:
        src = Template( f.read() )
    return src.substitute(data)

def split_climate(time_step,
                  ds_files:List[str],
                  CO2FILE:str, 
                  dt:int, 
                  ds_path:Optional[str]=None, 
                  dest_path:Optional[str]=None) -> None: 
    """Split climte files into dt-length chunks"""

    log.debug('ds_path: %s' % ds_path)
    log.debug('dest_path: %s' % dest_path)
    log.debug(ds_files)

    for ds_file in ds_files:
        fpath = os.path.join(ds_path, ds_file) if ds_path else ds_file
        log.debug(fpath)

        with xr.open_dataset(fpath, decode_times=False) as ds:
            n_episodes_monthly = len(ds.time) // (dt*12)
            n_rest_monthly     = len(ds.time) % (dt*12)
            
            n_episodes_daily = len(ds.time) // (dt*365)
            n_rest_daily = len(ds.time) % (dt*365)
            log.debug('Number of climate episodes: %d' % n_episodes_daily)
            
            if time_step == "monthly":
                episode_int  = np.repeat(list(range(n_episodes_monthly)), dt*12)
                episode_rest = np.repeat(episode_int[-1] + 1 , n_rest_monthly)
                episode = np.hstack([episode_int, episode_rest])
            else:
                episode_int  = np.repeat(list(range(n_episodes_daily)), dt*365)
                episode_rest = np.repeat(episode_int[-1] + 1 , n_rest_daily)
                episode = np.hstack([episode_int, episode_rest])

            ds['grouper'] = xr.DataArray(episode, coords=[('time', ds.time.values)])
            log.info('Splitting file %s' % ds_file)

            for g_cnt, ds_grp in tqdm(ds.groupby(ds.grouper)):
                del ds_grp['grouper']

                # modify time coord
                # use first dt years data
                if g_cnt == 0:
                    if time_step == "monthly":
                        time_ = ds_grp['time'][:dt*12]
                    else:
                        time_ = ds_grp['time'][:dt*365]

                #add_time_attrs(ds_grp, calendar_year=22_000)
                add_time_attrs(ds_grp, calendar_year=1_000)
                foutname = os.path.basename(fpath.replace('.nc',''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')
        
    # copy co2 file
    src = os.path.join(ds_path, CO2FILE) if ds_path else CO2FILE
    log.debug('co2_path: %s' % ds_path) 
    shutil.copyfile(src, os.path.join(dest_path, CO2FILE))
            

def generate_landform_files(self) -> None:
    log.info('Convert landlab netcdf data to lfdata fromat')
    create_input_main()

def execute_lpjguess(self) -> None:
    '''Run LPJ-Guess for one time-step'''
    log.info('Execute LPJ-Guess run')
    p = subprocess.call([self._binpath, '-input', 'sp', 'lpjguess.ins'], cwd=self._dest)
    #p.wait()

def move_state(self) -> None:
    '''Move state dumpm files into loaddir for next timestep'''
    log.info('Move state to loaddir')
    state_files = glob.glob(os.path.join(self._dest, 'dumpdir_eor/*'))
    for state_file in state_files:
        shutil.copy(state_file, os.path.join(self._dest, 'loaddir'))

        # for debugging:
        shutil.copy(state_file, os.path.join('tmp.state'))
            
def prepare_filestructure(dest:str,template_path:str,  source:Optional[str]=None) -> None:
    log.debug('Prepare file structure')
    log.debug('Dest: %s' % dest)
    if os.path.isdir(dest):
        log.fatal('Destination folder exists...')
        exit(-1)
        #time.sleep(3)
        #shutil.rmtree(dest)
    if source:
        shutil.copytree(source, dest)        
    else:
        shutil.copytree(template_path, dest)
    os.makedirs(os.path.join(dest, 'input', 'lfdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'input', 'climdata'), exist_ok=True)
    os.makedirs(os.path.join(dest, 'output'), exist_ok=True)


def prepare_input(dest:str,CO2_path:str,  template_path:str, forcings_path, input_path:str, input_name:str, time_step) -> None:
    log.debug('Prepare input')
    log.debug('dest: %s' % dest)
    
    prepare_filestructure(dest, template_path)

    # move this to a config or make it smarter
    vars = ['prec', 'temp', 'rad']
    #TODO: CHANGE HARDCODING OF file_name
    ds_files = [str(input_name) + '_%s.nc' % v for v in vars ]
    #ds_files = ['coupl_%s_35ka_lcy_landid.nc' % v for v in vars]
    split_climate(time_step ,ds_files, CO2FILE=CO2_path,  dt=100, ds_path=os.path.join(forcings_path, 'climdata'),
                                    dest_path=os.path.join(input_path, 'input', 'climdata'))

def prepare_runfiles(self, step_counter:int, ins_file:str, input_name:str) -> None:
    """Prepare files specific to this dt run"""
    # fill template files with per-run data:
    log.warn('REPEATING SPINUP FOR EACH DT !!!')
    #restart = '0' if step_counter == 0 else '1'
    restart = '0'

    run_data = {# climate data
                'CLIMPREC': str(input_name) + '_prec_%s.nc' % str(int(step_counter)).zfill(6),
                'CLIMWET':  str(input_name) + '_prec_%s.nc' % str(int(step_counter)).zfill(6),
                'CLIMRAD':  str(input_name) + '_rad_%s.nc'  % str(int(step_counter)).zfill(6),
                'CLIMTEMP': str(input_name) + '_temp_%s.nc' % str(int(step_counter)).zfill(6),
                # landform files
                'LFDATA': 'lpj2ll_landform_data.nc',
                'SITEDATA': 'lpj2ll_site_data.nc',
                # setup data
                'GRIDLIST': 'landid.txt',
                'NYEARSPINUP': '500',
                'RESTART': restart
                }

    insfile = fill_template( os.path.join(self._dest, ins_file), run_data )
    open(os.path.join(self._dest, 'lpjguess.ins'), 'w').write(insfile)

class DynVeg_LpjGuess(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    def __init__(self,
     LPJGUESS_TIME_INTERVAL:str,
     LPJGUESS_INPUT_PATH:str,
     LPJGUESS_TEMPLATE_PATH:str,
     LPJGUESS_FORCINGS_PATH:str,
     LPJGUESS_INS_FILE_TPL:str,
     LPJGUESS_BIN:str,
     LPJGUESS_CO2FILE:str,
     LPJGUESS_FORCINGS_STRING:str):
        self._spinup = True
        self._timesteps = [0]
        self._dest = LPJGUESS_INPUT_PATH
        self._templatepath = LPJGUESS_INS_FILE_TPL
        self._binpath = LPJGUESS_BIN
        self._forcingsstring = LPJGUESS_FORCINGS_STRING
        prepare_input(
            self._dest,
            LPJGUESS_CO2FILE, 
            LPJGUESS_TEMPLATE_PATH, 
            LPJGUESS_FORCINGS_PATH, 
            LPJGUESS_INPUT_PATH,
            LPJGUESS_FORCINGS_STRING,
            LPJGUESS_TIME_INTERVAL)

    #@property
    #def spinup(self):
    #    return self._spinup
    
    @property
    def timestep(self):
        '''Current timestep of sim'''
        #if len(self._timesteps) > 0:
        #    return self._timesteps[-1]
        #return None
        len(self._timesteps)
        return None


    @property
    def elapsed(self):
        '''Total sim time elapsed'''
        return sum(self._timesteps)

    def run_one_step(self, step_counter, dt:int=100) -> None:
        '''Run one lpj simulation step (duration: dt)'''
        self.prepare_runfiles(step_counter, self._templatepath,self._forcingsstring)
        self.generate_landform_files()
        self.execute_lpjguess()
        self.move_state()
        #if self.timestep == 0:
        #    self._spinup = False
        self._timesteps.append( dt )

DynVeg_LpjGuess.prepare_runfiles = prepare_runfiles
DynVeg_LpjGuess.generate_landform_files = generate_landform_files
DynVeg_LpjGuess.execute_lpjguess = execute_lpjguess
DynVeg_LpjGuess.move_state = move_state

#TODO: Check if you need this shit!
#if __name__ == '__main__':
#    # silence debug logging by setup loglevel to INFO here
#    logging.getLogger().setLevel(logging.INFO)
#    log.info('DynVeg LPJ-Guess Component')
#    DT = 100
#    component = DynVeg_LpjGuess(LPJGUESS_INPUT_PATH)
#
#    for i in range(2):
#        component.run_one_step(dt=DT)
