#!/bin/bash -l

## Sets up landlab model setup structure
## NOTE: If you set this up for your personal folder-setup
## then you need to change the modelpaths
## Created by: Manuel Schmid, 28th May, 2018
## Additions by: Willi Kappler, 2019.05.2

LANDLABDRIVER=/usr/share/modules/Modules/3.2.10/landlab/drivers

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
  cp ${LANDLABDRIVER}/$1/inputFile.py .
  cp ${LANDLABDRIVER}/helperScripts/createStandartTopo.py .
  cp ${LANDLABDRIVER}/Slurm_runfile.sbatch .
  cp ${LANDLABDRIVER}/README.txt .
}

# Check the passed arguments
case "$1" in
	-b|--bedrock)
		setup_folders pureBedrock "Bedrock Simulation"
	;;
	-s|--soil)
		setup_folders soilLayer "Soil/Fastscape Simulation"
	;;
	-S|--soilSpace)
		setup_folders soilLayerSpace "Space Simulation"
	;;
	-l|--lpjCoupled)
		setup_folders lpj_coupled "LPJ Coupled"
  ;;
  *)
		echo "This script sets up the landlab model setup structure"
		echo -e "Use on of the following parameters:\n"
		echo "-b|--bedrock"
		echo -e "\tUse the detachment-limited only model without soil cover"
		echo "-s|--soil"
		echo -e "\tUse the detachment-limited model with soil cover and weathering"
		echo "-S|--soilSpace"
		echo -e "\tUse the space-fluvial model with soil cover and weathering"
		echo "-l|--lpjCoupled"
		echo -e "\tUse the coupled lpj-landlab model \n"
		exit 0
	;;
esac

echo "Folder structure is setup. I recommend reading README.txt within your new directory"
echo "Happy researching."
