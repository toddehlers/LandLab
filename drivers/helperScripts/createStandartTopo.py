"""
This script creates a standart Area with an 'burned-in' river network by
running the fastscape algorithm for a few time-steps with a high K. 
Output is in numpy-inherent .npy format.
"""
## Import necessary Python and Landlab Modules
import numpy as np
from landlab import RasterModelGrid
from landlab import CLOSED_BOUNDARY, FIXED_VALUE_BOUNDARY
from landlab.components import FlowRouter
from landlab.components import FastscapeEroder
from landlab.components import DepressionFinderAndRouter
from landlab import imshow_grid
from matplotlib import pyplot as plt
import time
import logging

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename='landlab.log')

#---------------------------Parameter Definitions------------------------------#
##Model Grid:##
ncols = 21 
nrows = 21 
dx    = 100

#This is the total amount of steps the fastscape eroder runs
nSteps = int(50)

#Parameters used for Fastscape
ksp = 0.1 # Adapt to domain size
msp = 0.5
nsp = 1 
thresholdSP = 2.e-4 # May not bee needed

#Grid setup
mg = RasterModelGrid((nrows,ncols), dx)

#only uncomment this if there is a pre-existing topography you want to load. 
#right now this only works if the topo was saved in numpys .npy format.
try:
    topoSeed = np.load('topoSeed.npy')
    logging.info('loaded topoSeed.npy')
except:
    logging.info('There is no file containing a initial topography')

#Initate all the fields that are needed for calculations
mg.add_zeros('node','topographic__elevation')
#checks if standart topo is used. if not creates own
if 'topoSeed' in locals():
    mg.at_node['topographic__elevation'] += topoSeed
    logging.info('Using pre-existing topography from file topoSeed.npy')
else:
    mg.at_node['topographic__elevation'] += np.random.rand(mg.at_node.size)/10000 
    logging.info('No pre-existing topography. Creating own random noise topo.')

#Create boundary conditions of the model grid (either closed or fixed-head)
for edge in (mg.nodes_at_left_edge,mg.nodes_at_right_edge,
        mg.nodes_at_top_edge, mg.nodes_at_bottom_edge):
    mg.status_at_node[edge] = CLOSED_BOUNDARY

# Create one single outlet node, remove if FIXED_VALUE_BOUNDARY is used above
mg.set_watershed_boundary_condition_outlet_id(0,mg['node']['topographic__elevation'],-9999)

# You may want to add a Gauss elevation in order to speed up depression finding
#gauss_x, gauss_y = np.meshgrid(np.linspace(-1,1,ncols), np.linspace(-1,1,nrows))
#gauss_d = np.sqrt(gauss_x * gauss_x + gauss_y * gauss_y)
#sigma, mu = 1.0, 0.0
#gauss_g = np.exp(-( (gauss_d - mu)**2 / ( 2.0 * sigma**2 ) ) )
#gauss_g2 = np.reshape(gauss_g, (ncols*nrows, ))
#mg.at_node['topographic__elevation'] += gauss_g2


#Initialize Fastscape
fc = FastscapeEroder(mg,
                    K_sp = ksp ,
                    m_sp = msp,
                    n_sp = nsp,
                    rainfall_intensity = 1)
fr = FlowRouter(mg)
lm = DepressionFinderAndRouter(mg)

for i in range(nSteps):
    logging.info('Current step: {}'.format(i))
    fr.run_one_step(dt=1)
    lm.map_depressions()
    fc.run_one_step(dt=1)
    mg.at_node['topographic__elevation'][mg.core_nodes] += 0.0002

z = mg.at_node['topographic__elevation']

plt.figure()
imshow_grid(mg,z)
plt.savefig('intialTopography.png')
plt.close()

np.save('topoSeed',z)

logging.info('Done.')
logging.info('I have created initialTopography.png for you and topoSeed.npy for landlab')
