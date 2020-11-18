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

def add_time_attrs(ds, calendar_year=0):
    ds['time'].attrs['units'] = "days since 1-1-15 00:00:00" 
    ds['time'].attrs['axis'] = "T" 
    ds['time'].attrs['long_name'] = "time" 
    ds['time'].attrs['standard_name'] = "time" 
    ds['time'].attrs['calendar'] = "%d yr B.P." % calendar_year


def fill_template(template: str, data: Dict[str, str]) -> str:
    """Fill template file with specific data from dict"""
    loggging.debug('Fill LPJ-GUESS ins template')
    with open( template, 'rU' ) as f:
        src = Template( f.read() )
    return src.substitute(data)

def split_climate(time_step,
                  ds_files:List[str],
                  co2_file:str, 
                  dt:int,
                  calendar_year:int,
                  ds_path:Optional[str]=None, 
                  dest_path:Optional[str]=None) -> None: 
    """Split climte files into dt-length chunks"""

    loggging.debug('ds_path: %s' % ds_path)
    loggging.debug('dest_path: %s' % dest_path)
    loggging.debug(ds_files)

    for ds_file in ds_files:
        fpath = os.path.join(ds_path, ds_file) if ds_path else ds_file
        loggging.debug(fpath)

        with xr.open_dataset(fpath, decode_times=False) as ds:
            n_episodes_monthly = len(ds.time) // (dt*12)
            n_rest_monthly     = len(ds.time) % (dt*12)
            
            n_episodes_daily = len(ds.time) // (dt*365)
            n_rest_daily = len(ds.time) % (dt*365)
            
            if time_step == "monthly":
                episode_int  = np.repeat(list(range(n_episodes_monthly)), dt*12)
                episode_rest = np.repeat(episode_int[-1] + 1 , n_rest_monthly)
                episode = np.hstack([episode_int, episode_rest])
                loggging.debug('Number of climate episodes (monthly): %d' % n_episodes_monthly)
            else:
                episode_int  = np.repeat(list(range(n_episodes_daily)), dt*365)
                episode_rest = np.repeat(episode_int[-1] + 1 , n_rest_daily)
                episode = np.hstack([episode_int, episode_rest])
                loggging.debug('Number of climate episodes (daily): %d' % n_episodes_daily)

            ds['grouper'] = xr.DataArray(episode, coords=[('time', ds.time.values)])
            loggging.info('Splitting file %s' % ds_file)

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
                #add_time_attrs(ds_grp, calendar_year=1_000)
                add_time_attrs(ds_grp, calendar_year)
                foutname = os.path.basename(fpath.replace('.nc',''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')
        
    # copy co2 file
    src = os.path.join(ds_path, co2_file) if ds_path else co2_file
    loggging.debug('co2_path: %s' % ds_path) 
    shutil.copyfile(src, os.path.join(dest_path, co2_file))
            

def generate_landform_files(self) -> None:
    loggging.info('Convert landlab netcdf data to lfdata fromat')
    create_input_main()

def execute_lpjguess(self) -> None:
    '''Run LPJ-Guess for one time-step'''
    loggging.info('Execute LPJ-Guess run')
    p = subprocess.call([self._binpath, '-input', 'sp', 'lpjguess.ins'], cwd=self._dest)
    #p.wait()

def move_state(self) -> None:
    '''Move state dumpm files into loaddir for next timestep'''
    loggging.info('Move state to loaddir')
    state_files = glob.glob(os.path.join(self._dest, 'dumpdir_eor/*'))
    for state_file in state_files:
        shutil.copy(state_file, os.path.join(self._dest, 'loaddir'))

        # for debugging:
        shutil.copy(state_file, os.path.join('tmp.state'))

def prepare_filestructure(dest:str,template_path:str,  source:Optional[str]=None) -> None:
    loggging.debug('Prepare file structure')
    loggging.debug('Dest: %s' % dest)
    if os.path.isdir(dest):
        loggging.fatal('Destination folder exists...')
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


def prepare_input(dest:str, co2_file:str,  template_path:str, forcings_path,
        input_name:str, time_step:str, calendar_year:int, dt:int) -> None:
    loggging.debug('Prepare input')
    loggging.debug('dest: %s' % dest)
    
    prepare_filestructure(dest, template_path)

    # move this to a config or make it smarter
    vars = ['prec', 'temp', 'rad']
    #TODO: CHANGE HARDCODING OF file_name
    ds_files = [str(input_name) + '_%s.nc' % v for v in vars ]
    #ds_files = ['coupl_%s_35ka_lcy_landid.nc' % v for v in vars]
    split_climate(time_step, ds_files, co2_file, dt, calendar_year,
        os.path.join(forcings_path, 'climdata'), os.path.join(dest, 'input', 'climdata'))

def prepare_runfiles(self, step_counter:int, ins_file:str, input_name:str, co2_file:str) -> None:
    """Prepare files specific to this dt run"""
    # fill template files with per-run data:
    loggging.warn('REPEATING SPINUP FOR EACH DT !!!')
    restart = '0' if step_counter == 0 else '1'
    #restart = '0'

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
                'RESTART': restart,
                'CO2FILE': co2_file
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
     LPJGUESS_FORCINGS_STRING:str,
     LPJGUESS_CALENDAR_YEAR:int,
     dt: int):
        self._spinup = True
        self._timesteps = [0]
        self._dest = LPJGUESS_INPUT_PATH
        self._templatepath = LPJGUESS_INS_FILE_TPL
        self._binpath = LPJGUESS_BIN
        self._forcingsstring = LPJGUESS_FORCINGS_STRING
        self._co2_file = LPJGUESS_CO2FILE

        prepare_input(
            self._dest,
            self._co2_file, 
            LPJGUESS_TEMPLATE_PATH, 
            LPJGUESS_FORCINGS_PATH, 
            self._forcingsstring,
            LPJGUESS_TIME_INTERVAL,
            LPJGUESS_CALENDAR_YEAR,
            dt)

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
        self.prepare_runfiles(step_counter, self._templatepath, self._forcingsstring, self._co2_file)
        self.generate_landform_files()
        self.execute_lpjguess()
        self.move_state()
        #if self.timestep == 0:
        #    self._spinup = False
        self._timesteps.append( dt )

        #backup lpj results
        shutil.copy('temp_lpj/output/sp_firert.out', f"debugging/sp_firert.{str(step_counter).zfill(6)}.out")
        shutil.copy('temp_lpj/output/sp_lai.out', f"debugging/sp_lai.{str(step_counter).zfill(6)}.out")
        shutil.copy('temp_lpj/output/sp_mprec.out', f"debugging/sp_mprec.{str(step_counter).zfill(6)}.out")
        shutil.copy('temp_lpj/output/sp_tot_runoff.out', f"debugging/sp_tot_runoff.{str(step_counter).zfill(6)}.out")
        shutil.copy('temp_lpj/output/climate.out', f"debugging/climate.{str(step_counter).zfill(6)}.out")


DynVeg_LpjGuess.prepare_runfiles = prepare_runfiles
DynVeg_LpjGuess.generate_landform_files = generate_landform_files
DynVeg_LpjGuess.execute_lpjguess = execute_lpjguess
DynVeg_LpjGuess.move_state = move_state

