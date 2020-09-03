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

	echo "Greetings User. Setting up $2"
	cp ${LANDLABDRIVER}/$1/inputFile.ini .
	cp ${LANDLABDRIVER}/$1/runfile.qsub .
	cp ${LANDLABDRIVER}/README.txt .

	# Set up correct folder structure
	if [ $1 == "lpj_coupled" ]; then
		cp -r ${LANDLABDRIVER}/lpj_coupled/lpjguess.template .
		cp -r ${LANDLABDRIVER}/lpj_coupled/forcings .
		mkdir -p temp_output
		mkdir -p debugging
		mkdir -p ll_output/BED
		mkdir -p ll_output/DEM
		mkdir -p ll_output/DHDT
		mkdir -p ll_output/NC
		mkdir -p ll_output/SA
		mkdir -p ll_output/SoilDepth
		mkdir -p ll_output/SoilP
		mkdir -p ll_output/Veg
	else
		mkdir -p BED
		mkdir -p DEM
		mkdir -p ACC
		mkdir -p DHDT
		mkdir -p NC
		mkdir -p SA
		mkdir -p dd
		mkdir -p CSVOutput
		mkdir -p SoilDepth
		mkdir -p SoilP
		mkdir -p Ksn
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
