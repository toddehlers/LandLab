#Welcome Padawan
#First steps in creating your landlab simulation
#Created by:
#	Manuel Schmid
#	Wilhelmstrasse 56
#	AG Ehlers

1. run "python createStandartTopo.py"
	- This will create a initial drainage network using a
	  supercharged-version of the Fastscape eroder.
	  It will save an .png file of the topography for control-purposes
	  within this directory as "initialTopography.png"
	  and a file "topoSeed.npy" which will be used by the model-driver
	- Right now the boundary-condition standart is a single
	  outlet-note at the S-W-Edge of the modelgrid.
	  IF you need something different, you need to check the
          .set_boundary_condition() function within THIS script.

	- Important Parameters:
		- ncols = number of grid-columns
		- nrows = number of grid-rows
		- dx    = spatial resolution between nodes

2. modify "inputFile.py"
	- This is the main input file for your simulation.
	  All parameters in there should be well commented.
	
	- Important:
		- ncols/nrows/dx MUST be equal to the ones you used in 
		  "createStandartTopo.py"

(IF you want to start your simulation on the Minicluster, jump to Step 3.2)
3.1 run "python runfile_*":
	- This will start your model. 
	- Note:
		the *-wildcard in the filename depends on the simulation
		you are running, but they always start with "runfile_..." 
		so it should be straighforward.

3.2 modify "Slurm_runfile.sbatch":
	- This will sent your model as a batch-job to our minicluster.
	  (ESD-Main-Cluster is not yet supported, but will be soon)
	  
	- Important Parameters:
		- SBATCH -J = Name of your simulation for the scheduler
		- SBATCH --mail-user = Your email adress for status updates 
		- SBATCH -w = Node you want to run on
		- PYTHONBIN = /Path/to/your/python/install/binaries
		- RUNFILE   = name of the runfile in your directory  


For any questions/bugs/recommendations/complaints/compliments contact:

Manuel.Schmid@uni-tuebingen.de




