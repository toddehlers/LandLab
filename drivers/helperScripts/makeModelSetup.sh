#!/bin/bash -l

## Sets up landlab model setup structure
## NOTE: If you set this up for your personal folder-setup
## then you need to change the modelpaths
## Created by: Manuel Schmid, 28th May, 2018
## Additions by: Willi Kappler, 2019.05.2

if [[ -z "${LANDLABDRIVER}" ]]; then
    echo "Environment variable LANDLABDRIVER is not set"
    echo "You should load the module first:"
    echo "module load landlab"
    exit 1
fi

function setup_folders {
	# Catches the user input for folder-name
	read -p "Enter Simulationfolder name: " foldername
	mkdir ${foldername}
	cd ./${foldername}

	# Set up correct folder structure
	if [ ! -d DEM ] ; then mkdir DEM ; fi
	if [ ! -d ACC ] ; then mkdir ACC ; fi
	if [ ! -d DHDT ] ; then mkdir DHDT ; fi
	if [ ! -d NC ] ; then mkdir NC ; fi
	if [ ! -d SA ] ; then mkdir SA ; fi
	if [ ! -d dd ] ; then mkdir dd ; fi
	if [ ! -d CSVOutput ] ; then mkdir CSVOutput ; fi
	if [ ! -d SoilDepth ] ; then mkdir SoilDepth ; fi
	if [ ! -d Ksn ] ; then mkdir Ksn ; fi
	echo "Folder structure set up."

	echo "Greetings User. Setting up $2"
	cp ${LANDLABDRIVER}/$1/inputFile.ini .
	cp ${LANDLABDRIVER}/$1/Slurm_runfile.sbatch .
	cp ${LANDLABDRIVER}/README.txt .

	if [ $1 == "lpj_coupled" ]; then
		# For LPJ coupling more files are needed:
 		# cp -r ${LANDLABDRIVER}/lpj_coupled/temp_lpj .
		mkdir temp_output
		mkdir debugging
		mkdir ll_output
		cp -r ${LANDLABDRIVER}/lpj_coupled/lpjguess.template .
		cp -r ${LANDLABDRIVER}/lpj_coupled/forcings .
	fi
}

# Check the passed arguments
case "$1" in
	bedrock)
		setup_folders pureBedrock "Bedrock Simulation"
	;;
	soil)
		setup_folders soilLayer "Soil/Fastscape Simulation"
	;;
	space)
		setup_folders soilLayerSpace "Space Simulation"
	;;
	lpj)
		setup_folders lpj_coupled "LPJ Coupled"
  ;;
  *)
		echo "This script sets up the landlab model setup structure"
		echo -e "Use on of the following parameters:\n"
		echo "bedrock"
		echo -e "\tUse the detachment-limited only model without soil cover"
		echo "soil"
		echo -e "\tUse the detachment-limited model with soil cover and weathering"
		echo "space"
		echo -e "\tUse the space-fluvial model with soil cover and weathering"
		echo "lpj"
		echo -e "\tUse the coupled lpj-landlab model \n"
		exit 0
	;;
esac

echo "Folder structure is setup. I recommend reading README.txt within your new directory"
echo "Happy researching."
