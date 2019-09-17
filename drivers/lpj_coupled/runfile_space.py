"""
Main driver file for coupled model-run between LANDLAB and LPJGUESS.
Derived from the messy thing I produced to glue/patch the model together.
"""

## Import necessary Python and Landlab Modules
#basic grid setup
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
#landlab components
from landlab.components.flow_routing import FlowRouter
from landlab.components import ExponentialWeatherer
from landlab.components import DepthDependentDiffuser
from landlab.components import FastscapeEroder
from landlab.components import Space
from landlab.components import DepressionFinderAndRouter
from landlab.components import SteepnessFinder
from landlab.components import rainfallOscillation as ro
from landlab.components import DynVeg_LpjGuess
#input/output
from landlab import imshow_grid
from landlab.components import landformClassifier
from landlab.io.netcdf import write_netcdf
from landlab.io.netcdf import read_netcdf
#coupling-specific
from create_input_for_landlab import lpj_import_run_one_step 
from create_all_landforms import create_all_landforms
#external modules
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['agg.path.chunksize'] = 200000000
import time
import logging
import numpy as np
import os.path
import shutil
#import the .py-inputfile
#from inputFile import *

import configparser

logger = logging.getLogger('runfile')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('landlab.log')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

config = configparser.ConfigParser()
config.read('inputFile.ini')

nrows = int(config['Grid']['nrows'])
ncols = int(config['Grid']['ncols'])
dx = int(config['Grid']['dx'])

totalT = float(config['Runtime']['totalT'])
ssT = float(config['Runtime']['ssT'])
sfT = float(config['Runtime']['sfT'])
spin_up = float(config['Runtime']['spin_up'])
dt = float(config['Runtime']['dt'])

upliftRate = float(config['Uplift']['upliftRate'])

linDiffBase = float(config['Surface']['linDiffBase'])
alphaDiff = float(config['Surface']['alphaDiff'])

critArea = float(config['Erosion']['critArea'])
aqDens = float(config['Erosion']['aqDens'])
grav = float(config['Erosion']['grav'])
nSoil = float(config['Erosion']['nSoil'])
nVRef = float(config['Erosion']['nVRef'])
vRef = float(config['Erosion']['vRef'])
w = float(config['Erosion']['w'])

k_sediment = float(config['Erosion_SPACE']['k_sediment'])
k_bedrock = float(config['Erosion_SPACE']['k_bedrock'])
Ff = float(config['Erosion_SPACE']['Ff'])
phi = float(config['Erosion_SPACE']['phi'])
Hstar = float(config['Erosion_SPACE']['Hstar'])
vs = float(config['Erosion_SPACE']['vs'])
m = float(config['Erosion_SPACE']['m'])
n = float(config['Erosion_SPACE']['n'])
sp_crit_sedi = float(config['Erosion_SPACE']['sp_crit_sedi'])
sp_crit_bedrock = float(config['Erosion_SPACE']['sp_crit_bedrock'])
solverMethod = config['Erosion_SPACE']['solverMethod']
solver = config['Erosion_SPACE']['solver']

initialSoilDepth = float(config['Lithology']['initialSoilDepth'])
soilProductionRate = float(config['Lithology']['soilProductionRate'])
soilProductionDecayDepth = float(config['Lithology']['soilProductionDecayDepth'])

baseRainfall = float(config['Climate']['baseRainfall'])
rfA = float(config['Climate']['rfA'])

vp = float(config['Vegetation']['vp'])
sinAmp = float(config['Vegetation']['sinAmp'])
sinPeriod = float(config['Vegetation']['sinPeriod'])

latitude = float(config['LPJ']['latitude'])
longitude = float(config['LPJ']['longitude'])
classificationType = config['LPJ']['classificationType']
elevationStepBin = float(config['LPJ']['elevationStepBin'])
LPJGUESS_INPUT_PATH = config['LPJ']['LPJGUESS_INPUT_PATH']
LPJGUESS_TEMPLATE_PATH = config['LPJ']['LPJGUESS_TEMPLATE_PATH']
LPJGUESS_FORCINGS_PATH = config['LPJ']['LPJGUESS_FORCINGS_PATH']
LPJGUESS_INS_FILE_TPL = config['LPJ']['LPJGUESS_INS_FILE_TPL']
# LPJGUESS_BIN = os.environ["LANDLABDRIVER"] + ""
LPJGUESS_BIN = "guess" # Should be in PATH
LPJGUESS_CO2FILE = config['LPJ']['LPJGUESS_CO2FILE']
LPJGUESS_FORCINGS_STRING = config['LPJ']['LPJGUESS_FORCINGS_STRING']
LPJGUESS_TIME_INTERVAL = config['LPJ']['LPJGUESS_TIME_INTERVAL']

outInt = int(config['Output']['outIntSpinUp'])

##----------------------Basic setup of global variables------------------------
#Set basic output interval for transient conditions
#Number of total-timestep (nt)
nt = int(totalT / dt)
#time-vector (total and transient), used for plotting later
timeVec = np.arange(0, totalT, dt)
transTimeVec = np.arange(0, (totalT - ssT), dt)
transientRainfallTimespan = int(totalT - ssT)
#calculate the uplift per timestep
uplift_per_step = upliftRate * dt
#Create Limits for DHDT plot.
DHDTLowLim = upliftRate - (upliftRate * 1)
DHDTHighLim = upliftRate + (upliftRate * 1)
#Number of total produced outputs
no = totalT / outInt
#number of zeros for file_naming. Don't meddle with this.
zp = len(str(int(no)))

logger.info("finished with parameter-initiation")

##----------------------Grid Setup---------------------------------------------
#This initiates a Modelgrid with dimensions nrows x ncols and spatial scaling of dx
mg = RasterModelGrid((nrows,ncols), dx)
#Initate all the fields that are needed for calculations
mg.add_zeros('node', 'topographic__elevation')
mg.add_zeros('node', 'bedrock__elevation')
mg.add_zeros('node', 'soil_production__rate')
mg.add_zeros('node', 'soil__depth')
mg.add_zeros('node', 'tpi__mask')
mg.add_zeros('node', 'erosion__rate')
mg.add_zeros('node', 'median_soil__depth')
mg.add_zeros('node', 'vegetation__density')
mg.add_zeros('node', 'fluvial_erodibility__soil')
mg.add_zeros('node', 'fluvial_erodibility__bedrock')

#this checks if there is a initial topography we like to start with. 
#initial topography must be of filename/type 'topoSeed.npy'
if os.path.isfile('initial_topography.npy'):
    topoSeed = np.load('initial_topography.npy')
    mg.at_node['topographic__elevation'] += topoSeed + baseElevation
    mg.at_node['bedrock__elevation'] += topoSeed + baseElevation
    if os.path.isfile('initial_soildepth.npy'):
        soilSeed = np.load('initial_soildepth.npy')
        mg.at_node['soil__depth'] = soilSeed
        mg.at_node['bedrock__elevation'] = mg.at_node['topographic__elevation'] - soilSeed
        logger.info('Using provided soil-thickness data')
    else:
        mg.at_node['soil__depth'] += initialSoilDepth
        logger.info('Adding 1m of soil everywhere.')
    logger.info('Using pre-existing topography from file topoSeed.npy')
else:
    topo_tilt = mg.node_y/100000000 + mg.node_x/100000000
    mg.at_node['topographic__elevation'] += (np.random.rand(mg.at_node.size)/10000 + initialSoilDepth)
    mg.at_node['topographic__elevation'] += topo_tilt
    mg.at_node['bedrock__elevation'] += (np.random.rand(mg.at_node.size)/10000 + initialSoilDepth)
    mg.at_node['bedrock__elevation'] += topo_tilt
    mg.at_node['soil__depth'] += initialSoilDepth
    logger.info('No pre-existing topography. Creating own random noise topo.')

# Create boundary conditions of the model grid (either closed or fixed-head)
for edge in (mg.nodes_at_left_edge, mg.nodes_at_right_edge,
        mg.nodes_at_top_edge, mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = FIXED_VALUE_BOUNDARY

boundary = config['Grid']['boundary'].strip()

for c in boundary:
    if c == 'E':
        mg.status_at_node[mg.nodes_at_right_edge] = CLOSED_BOUNDARY
    elif c == 'S':
        mg.status_at_node[mg.nodes_at_bottom_edge] = CLOSED_BOUNDARY
    elif c == 'W':
        mg.status_at_node[mg.nodes_at_left_edge] = CLOSED_BOUNDARY
    elif c == 'N':
        mg.status_at_node[mg.nodes_at_top_edge] = CLOSED_BOUNDARY
    elif c == 'P':
        # Create one single outlet node
        mg.set_watershed_boundary_condition_outlet_id(0,mg['node']['topographic__elevation'],-9999)
    else:
        logger.error("Unknown boundary parameter: {}".format(c))

#create mask datafield which defaults to 1 to all core nodes and to 0 for
#boundary nodes. LPJGUESS needs this
mg.at_node['tpi__mask'][mg.core_nodes] = 1
mg.at_node['tpi__mask'][mg.boundary_nodes] = 0

logger.info("finished with setup of modelgrid")

##---------------------------------Vegi implementation--------------------------#
##Set up a timeseries for vegetation-densities
vegiTimeseries  = np.zeros(int(totalT / dt)) + vp
#this incorporates a vegi step-function at timestep sfT with amplitude sfA
mg.at_node['vegetation__density'][:] = vp
#This maps the vegetation density on the nodes to the links between the nodes
vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')

##These are the necesseray calculations for implementing the vegetation__density
##in the fluvial routines
nSoil_to_15 = np.power(nSoil, 1.5)
Ford = aqDens * grav * nSoil_to_15
n_v_frac = nSoil + (nVRef * ((mg.at_node['vegetation__density'] / vRef)**w)) #self.vd = VARIABLE!
Prefect = np.power(n_v_frac, 0.9)
Kvs = k_sediment * Ford/Prefect
Kvb = k_bedrock  * Ford/Prefect

##These are the calcultions to calculate the linear diffusivity based on vegis
linDiff = mg.zeros('node', dtype = float)
linDiff = linDiffBase * np.exp(-alphaDiff * vegiLinks)

logger.info("finished setting up the vegetation fields and Kdiff and Kriv")

##---------------------------------Rain implementation--------------------------#
##Set up a Timeseries of rainfall values
rainTimeseries = np.zeros(int(totalT / dt)) + baseRainfall
mg.add_zeros('node', 'rainvalue')
mg.at_node['rainvalue'][:] = int(baseRainfall)

##---------------------------------Component initialization---------------------#


fr = FlowRouter(mg,method = 'd8', runoff_rate = baseRainfall)

lm = DepressionFinderAndRouter(mg)

expWeath = ExponentialWeatherer(mg, soil_production__maximum_rate =
        soilProductionRate, soil_production__decay_depth = soilProductionDecayDepth)

sf = SteepnessFinder(mg,
                    min_drainage_area = 1e6)

sp = Space(mg, K_sed=Kvs, K_br=Kvb, 
           F_f=Ff, phi=phi, H_star=Hstar, v_s=vs, m_sp=m, n_sp=n,
           sp_crit_sed=sp_crit_sedi, sp_crit_br=sp_crit_bedrock,
           solver = solver)

lc = landformClassifier(mg)

DDdiff = DepthDependentDiffuser(mg, 
            linear_diffusivity = linDiff,
            soil_transport_decay_depth = 2)

lpj = DynVeg_LpjGuess(LPJGUESS_TIME_INTERVAL,
                    LPJGUESS_INPUT_PATH,
                    LPJGUESS_TEMPLATE_PATH,
                    LPJGUESS_FORCINGS_PATH,
                    LPJGUESS_INS_FILE_TPL,
                    LPJGUESS_BIN,
                    LPJGUESS_CO2FILE,
                    LPJGUESS_FORCINGS_STRING)

logger.info("finished with the initialization of the erosion components")   
elapsed_time = 0
counter = 0
while elapsed_time < totalT:

    #create copy of "old" topography
    z0 = mg.at_node['topographic__elevation'].copy()

    #Call the erosion routines.
    fr.run_one_step()
    lm.map_depressions()
    floodedNodes = np.where(lm.flood_status==3)[0]
    sp.run_one_step(dt = dt, flooded_nodes = floodedNodes)
    
    #fetch the nodes where space eroded the bedrock__elevation over topographic__elevation
    #after conversation with charlie shobe:
    b = mg.at_node['bedrock__elevation']
    b[:] = np.minimum(b, mg.at_node['topographic__elevation'])

    #calculate regolith-production rate
    expWeath.calc_soil_prod_rate()
    
    #Generate and move the soil around.
    DDdiff.run_one_step(dt=dt)

    #run the landform classifier
    lc.run_one_step(elevationStepBin, 300, classtype = classificationType)

    #run lpjguess once at the beginning and then each timestep after the spinup.
    if elapsed_time < spin_up:
        outInt = int(config['Output']['outIntSpinUp'])
        if elapsed_time == 0:
            
            #create all possible landform__ID's in here ONCE before lpjguess is called
            create_all_landforms(upliftRate, totalT, elevationStepBin, mg)
            write_netcdf('./temp_output/current_output.nc',
                   mg,format='NETCDF4', attrs = {'lgt.lat' : latitude,
                                                 'lgt.lon' : longitude,
                                                 'lgt.dx'  : dx,
                                                 'lgt.dy'  : dx,
                                                 'lgt.timestep' : elapsed_time,
                                                 'lgt.classification' : classificationType,
                                                 'lgt.elevation_step' : elevationStepBin})

            lpj.run_one_step(counter, dt = dt)
            #backup lpj results
            shutil.copy('./temp_lpj/output/sp_lai.out', f"./debugging/sp_lai.{str(counter).zfill(6)}.out" )
            shutil.copy('./temp_lpj/output/sp_mprec.out', f"./debugging/sp_mprec.{str(counter).zfill(6)}.out" )
            shutil.copy('./temp_lpj/output/sp_tot_runoff.out', f"./debugging/sp_tot_runoff.{str(counter).zfill(6)}.out" )
            shutil.copy('./temp_lpj/output/climate.out', f"./debugging/climate.{str(counter).zfill(6)}.out" )
            #import lpj lai and precipitation data
            lpj_import_run_one_step(mg,'./temp_lpj/output/sp_lai.out',
                    var='lai', method = LPJGUESS_VEGI_MAPPING)
            lpj_import_run_one_step(mg,'./temp_lpj/output/sp_mprec.out', var='mprec')
            #reinitialize the flow router
            fr = FlowRouter(mg,method = 'd8', runoff_rate = mg.at_node['precipitation'])

        elif elapsed_time > 0:
            pass
    elif elapsed_time >= spin_up:
        outInt = int(config['Output']['outIntTransient'])
        #reset counter to 1, to get right position in climate file
        if elapsed_time == spin_up:
            counter = 1
        else:
            pass
        lpj.run_one_step(counter, dt = dt)
        shutil.copy('./temp_lpj/output/sp_lai.out', f"./debugging/sp_lai.{str(counter).zfill(6)}.out" )
        shutil.copy('./temp_lpj/output/sp_mprec.out', f"./debugging/sp_mprec.{str(counter).zfill(6)}.out" )
        shutil.copy('./temp_lpj/output/sp_tot_runoff.out', f"./debugging/sp_tot_runoff.{str(counter).zfill(6)}.out" )
        shutil.copy('./temp_lpj/output/climate.out', f"./debugging/climate.{str(counter).zfill(6)}.out" )
        #import lpj lai and precipitation data
        lpj_import_run_one_step(mg,'./temp_lpj/output/sp_lai.out', var='lai',
                method =  LPJGUESS_VEGI_MAPPING)
        lpj_import_run_one_step(mg,'./temp_lpj/output/sp_mprec.out', var='mprec')
        #reinitialize the flow router
        fr = FlowRouter(mg,method = 'd8', runoff_rate = mg.at_node['precipitation'])
 
    
    #apply uplift
    mg.at_node['bedrock__elevation'][mg.core_nodes] += uplift_per_step
    
    #set soil-depth to zero at outlet node
    #TODO: try disabling this to get non-zero soil depth?
    mg.at_node['soil__depth'][0] = 0
    
    #recalculate topographic elevation
    mg.at_node['topographic__elevation'][:] = \
            mg.at_node['bedrock__elevation'][:] + mg.at_node['soil__depth'][:]

    #Calculate median soil-depth
    for ids in np.unique(mg.at_node['landform__ID'][:]):
        _soilIDS = np.where(mg.at_node['landform__ID']==ids)
        mg.at_node['median_soil__depth'][_soilIDS] = np.median(mg.at_node['soil__depth'][_soilIDS]) 

    #Calculate dhdt and E
    dh = (mg.at_node['topographic__elevation'] - z0)
    dhdt = dh/dt
    erosionMatrix = upliftRate - dhdt
    mg.at_node['erosion__rate'] = erosionMatrix

    #update vegetation_density on links
    vegiLinks = mg.map_mean_of_link_nodes_to_link('vegetation__density')
    #update LinearDiffuser
    linDiff = linDiffBase*np.exp(-alphaDiff * vegiLinks)
    #reinitalize Diffuser
    DDdiff = DepthDependentDiffuser(mg, 
            linear_diffusivity = linDiff,
            soil_transport_decay_depth = soilProductionDecayDepth)

    #update K_sp
    #after the first-timestep there is LPJ information about phenologic groups so now use them instead of total vegetation-cover
    if LPJGUESS_VEGI_MAPPING == "individual":
        n_grass_fpc = nGrass * (mg.at_node['grass_fpc'] / vRef)**w
        n_tree_fpc  = nTree  * (mg.at_node['tree_fpc']  / vRef)**w
        n_shrub_fpc = nShrub * (mg.at_node['shrub_fpc'] / vRef)**w
        n_total  = (nSoil + n_tree_fpc + n_shrub_fpc + n_grass_fpc)
        n_v_frac = n_total
    elif LPJGUESS_VEGI_MAPPING == "cumulative":
        n_v_frac = nSoil + (nVRef * (mg.at_node['vegetation__density'] / vRef)) #self.vd = VARIABLE!
    else:
        logger.info('Unsupported Argument for Vegetation Mapping')
    
    n_v_frac_to_w = np.power(n_v_frac, w)
    Prefect = np.power(n_v_frac_to_w, 0.9)
    Kvs = k_sediment * Ford/Prefect
    Kvb = k_bedrock  * Ford/Prefect
    sp.K_sed = Kvs
    sp.K_bed = Kvb
    #write the erodibility values in an grid-field. This is not used for calculations, just for visualiziation afterwards.
    mg.at_node['fluvial_erodibility__soil']    = Kvs
    mg.at_node['fluvial_erodibility__bedrock'] = Kvb

    #increment counter
    counter += 1


    #Run the output loop every outInt-times
    if elapsed_time % outInt  == 0:

        logger.info('Elapsed Time: {}, writing output!'.format(elapsed_time))
        ##Create DEM
        plt.figure()
        #imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation',cmap='terrain')
        imshow_grid(mg,'topographic__elevation',grid_units=['m','m'],var_name = 'Elevation [m]',cmap='terrain', plot_name='Time: {} [yrs]'.format(elapsed_time))
        plt.savefig('./ll_output/DEM/DEM_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Bedrock Elevation Map
        plt.figure()
        imshow_grid(mg,'bedrock__elevation', grid_units=['m','m'], var_name = 'bedrock', cmap='jet')
        plt.savefig('./ll_output/BED/BED_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create Slope - Area Map
        plt.figure()
        plt.loglog(mg.at_node['drainage_area'][np.where(mg.at_node['drainage_area'] > 0)],
           mg.at_node['topographic__steepest_slope'][np.where(mg.at_node['drainage_area'] > 0)],
           marker='.',linestyle='None')
        plt.xlabel('Area')
        plt.ylabel('Slope')
        plt.savefig('./ll_output/SA/SA_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        ##Create NetCDF Output
        write_netcdf('./ll_output/NC/output{}'.format(elapsed_time)+'__'+str(int(elapsed_time/outInt)).zfill(zp)+'.nc',
                mg,format='NETCDF4', attrs = {'lgt.lat' : latitude,
                                              'lgt.lon' : longitude,
                                              'lgt.dx'  : dx,
                                              'lgt.dy'  : dx,
                                              'lgt.timestep' : elapsed_time,
                                              'lgt.classification' : classificationType,
                                              'lgt.elevation_step' : elevationStepBin})
        ##Create NetCDF Output for LPJ
        os.rename('./temp_output/current_output.nc',
                './temp_output/current_backup'+str(counter)+ '.nc')
        os.remove('./temp_output/current_backup'+str(counter)+'.nc')
        write_netcdf('./temp_output/current_output.nc',
                mg,format='NETCDF4', attrs = {'lgt.lat' : latitude,
                                              'lgt.lon' : longitude,
                                              'lgt.dx'  : dx,
                                              'lgt.dy'  : dx,
                                              'lgt.timestep' : elapsed_time,
                                              'lgt.classification' : classificationType,
                                              'lgt.elevation_step' : elevationStepBin})
                
        ##Create erosion_diffmaps
        plt.figure()
        imshow_grid(mg,erosionMatrix,grid_units=['m','m'],var_name='Erosion m/yr',cmap='jet',limits=[DHDTLowLim,DHDTHighLim])
        plt.savefig('./ll_output/DHDT/eMap_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        
        ##Create Soil Depth Maps
        plt.figure()
        imshow_grid(mg,'soil__depth',grid_units=['m','m'],var_name=
                'Elevation',cmap='terrain', limits = [0, 1.5])
        plt.savefig('./ll_output/SoilDepth/SD_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        #Create SoilProd Maps
        plt.figure()
        imshow_grid(mg,'soil_production__rate')
        plt.savefig('./ll_output/SoilP/SoilP_'+str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()
        #create Vegi_Density maps
        plt.figure()
        imshow_grid(mg, 'vegetation__density', limits = [0,1])
        plt.savefig('./ll_output/Veg/vegidensity_'+ str(int(elapsed_time/outInt)).zfill(zp)+'.png')
        plt.close()


    elapsed_time += dt #update elapsed time
logger.info('End of  Main Loop. So far it took {}s to get here. No worries homeboy...'.format(tE-t0))
