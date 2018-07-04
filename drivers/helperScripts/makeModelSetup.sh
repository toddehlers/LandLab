#!/bin/bash -l

##Sets up landlab model setup structure
##NOTE: If you set this up for your personal folder-setup
##then you need to change the modelpaths
##Created by: Manuel Schmid, 28th May, 2018


##Parameters:
#	-b|--bedrock 
#		Use the detachment-limited only model without soil cover
#	-s|--soil
#		Use the detachment-limited model with soil cover and weathering
#	-S|--soilSpace
#		Use the space-fluvial model with soil cover and weatherin

#Set up Model Paths
CURRENTDIR=$(pwd)
PYTHONDIR=/esd/esd01/data/mschmid/anaconda3/bin
LANDLABDIR=/esd/esd01/data/mschmid/landlab/drivers

#Set up correct folder structure
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

#check the passed arguments
Params=""
while (("$#")); do
	case "$1" in 
		-b|--bedrock)
			echo "Setting up bedrock model."
			cp ${LANDLABDIR}/pureBedrock/inputFile.py .
			cp ${LANDLABDIR}/pureBedrock/runfile_textinput.py .
			break
		;;
		-s|--soil)
			echo "Setting up soil model."
			cp ${LANDLABDIR}/soilLayer/inputFile.py .
			cp ${LANDLABDIR}/soilLayer/runfile_textinput_soil.py .
			break
		;;
		-S|--soilSpace)
			echo "Setting up soil/space model."
			cp ${LANDLABDIR}/soilLayerSpace/inputFile.py .
			cp ${LANDLABDIR}/soilLayerSpace/runfile_textinput_soilSpace.py .
			break
		;;
		-*|--*)
			echo "Error: unsupported argument."
			exit 1
			;;
	esac
done

	
