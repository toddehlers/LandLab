
import glob
import logging
import os
import shutil
import subprocess
from string import Template
from typing import Dict, List, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm

from landlab import Component


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
    logging.debug('Fill LPJ-GUESS ins template')
    with open(template, 'rU') as f:
        src = Template(f.read())
    return src.substitute(data)

def split_climate(time_step,
                  ds_files: List[str],
                  co2_file: str,
                  dt: int,
                  calendar_year: int,
                  ds_path: Optional[str] = None,
                  dest_path: Optional[str] = None) -> None:
    """Split climte files into dt-length chunks"""

    logging.debug('ds_path: %s', ds_path)
    logging.debug('dest_path: %s', dest_path)
    logging.debug(ds_files)

    for ds_file in ds_files:
        fpath = os.path.join(ds_path, ds_file) if ds_path else ds_file
        logging.debug(fpath)

        with xr.open_dataset(fpath, decode_times=False) as ds:
            n_episodes_monthly = len(ds.time) // (dt*12)
            n_rest_monthly     = len(ds.time) % (dt*12)

            n_episodes_daily = len(ds.time) // (dt*365)
            n_rest_daily = len(ds.time) % (dt*365)

            if time_step == "monthly":
                episode_int  = np.repeat(list(range(n_episodes_monthly)), dt*12)
                episode_rest = np.repeat(episode_int[-1] + 1, n_rest_monthly)
                episode = np.hstack([episode_int, episode_rest])
                logging.debug('Number of climate episodes (monthly): %d', n_episodes_monthly)
            else:
                episode_int  = np.repeat(list(range(n_episodes_daily)), dt*365)
                episode_rest = np.repeat(episode_int[-1] + 1, n_rest_daily)
                episode = np.hstack([episode_int, episode_rest])
                logging.debug('Number of climate episodes (daily): %d', n_episodes_daily)

            ds['grouper'] = xr.DataArray(episode, coords=[('time', ds.time.values)])
            logging.info('Splitting file %s', ds_file)

            for g_cnt, ds_grp in tqdm(ds.groupby(ds.grouper)):
                del ds_grp['grouper']

#                if g_cnt == 0:
#                    if time_step == "monthly":
#                        ds_grp['time'][:dt*12]
#                    else:
#                        ds_grp['time'][:dt*365]

                add_time_attrs(ds_grp, calendar_year)
                foutname = os.path.basename(fpath.replace('.nc', ''))
                foutname = os.path.join(dest_path, '%s_%s.nc' % (foutname, str(g_cnt).zfill(6)))
                ds_grp.to_netcdf(foutname, format='NETCDF4_CLASSIC')

    # copy co2 file
    src = os.path.join(ds_path, co2_file) if ds_path else co2_file
    logging.debug('co2_path: %s', ds_path)
    shutil.copyfile(src, os.path.join(dest_path, co2_file))


def prepare_filestructure(dest: str, template_path: str, source: Optional[str] = None) -> None:
    logging.debug('Prepare file structure')
    logging.debug('Dest: %s', dest)
    if os.path.isdir(dest):
        logging.fatal('Destination folder exists...')
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


def prepare_input(dest: str, co2_file: str, template_path: str, forcings_path,
        input_name: str, time_step: str, calendar_year: int, dt: int) -> None:
    logging.debug('Prepare input')
    logging.debug('dest: %s', dest)

    prepare_filestructure(dest, template_path)

    # move this to a config or make it smarter
    variables = ['prec', 'temp', 'rad']
    #TODO: CHANGE HARDCODING OF file_name
    ds_files = [str(input_name) + '_%s.nc' % v for v in variables]
    #ds_files = ['coupl_%s_35ka_lcy_landid.nc' % v for v in variables]
    split_climate(time_step, ds_files, co2_file, dt, calendar_year,
        os.path.join(forcings_path, 'climdata'), os.path.join(dest, 'input', 'climdata'))

def generate_landform_files() -> None:
    logging.info('Convert landlab netcdf data to lfdata fromat')
    create_input_main()

class DynVeg_LpjGuess(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    def __init__(self,
     LPJGUESS_TIME_INTERVAL: str,
     LPJGUESS_INPUT_PATH: str,
     LPJGUESS_TEMPLATE_PATH: str,
     LPJGUESS_FORCINGS_PATH: str,
     LPJGUESS_INS_FILE_TPL: str,
     LPJGUESS_BIN: str,
     LPJGUESS_CO2FILE: str,
     LPJGUESS_FORCINGS_STRING: str,
     LPJGUESS_CALENDAR_YEAR: int,
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


    @property
    def elapsed(self):
        '''Total sim time elapsed'''
        return sum(self._timesteps)

    def run_one_step(self, step_counter, dt: int = 100) -> None:
        '''Run one lpj simulation step (duration: dt)'''
        self.prepare_runfiles(step_counter, self._templatepath, self._forcingsstring, self._co2_file)
        generate_landform_files()
        self.execute_lpjguess()
        self.move_state()
        #if self.timestep == 0:
        #    self._spinup = False
        self._timesteps.append(dt)

    def prepare_runfiles(self, step_counter: int, ins_file: str, input_name: str, co2_file: str) -> None:
        """Prepare files specific to this dt run"""
        # fill template files with per-run data:
        logging.warning('REPEATING SPINUP FOR EACH DT !!!')
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

        insfile = fill_template(os.path.join(self._dest, ins_file), run_data)
        open(os.path.join(self._dest, 'lpjguess.ins'), 'w').write(insfile)

    def execute_lpjguess(self) -> None:
        '''Run LPJ-Guess for one time-step'''
        logging.info('Execute LPJ-Guess run')
        subprocess.call([self._binpath, '-input', 'sp', 'lpjguess.ins'], cwd=self._dest)

    def move_state(self) -> None:
        '''Move state dumpm files into loaddir for next timestep'''
        logging.info('Move state to loaddir')
        state_files = glob.glob(os.path.join(self._dest, 'dumpdir_eor/*'))
        for state_file in state_files:
            shutil.copy(state_file, os.path.join(self._dest, 'loaddir'))

            # for debugging:
            shutil.copy(state_file, os.path.join('tmp.state'))
