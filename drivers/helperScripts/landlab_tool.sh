#!/bin/bash

LANDLABDRIVER=/usr/share/modules/Modules/3.2.10/landlab/drivers

case "$1" in
    init)
        bash $LANDLABDRIVER/helperScripts/makeModelSetup.sh $2
    ;;
    run)
        case "$2" in
            bedrock)
                python3 $LANDLABDRIVER/pureBedrock/runfile_textinput.py
            ;;
            soil)
                python3 $LANDLABDRIVER/soilLayer/runfile_textinput_soil.py
            ;;
            space)
                python3 $LANDLABDRIVER/soilLayerSpace/runfile_textinput_soilSpace.py
            ;;
            lpj)
                python3 $LANDLABDRIVER/lpj_coupled/runfile_space.py
            ;;
        esac
    ;;
    reset_output)
        rm ACC/* CSVOutput/* DEM/* DHDT/* Ksn/* NC/* SA/* SoilDepth/* dd/* dynveg_lpjguess.log Multiplot_absolut.png vegi_P_bugfix.png
        mkdir ACC CSVOutput DEM DHDT Ksn NC SA SoilDepth dd
    ;;
    create_topo)
        $LANDLABDRIVER/helperScripts/createStandartTopo.py
    ;;
    *)
        echo "This script initializes and runs the apropriate landlab model."
        echo -e "The following sub-commands are available:\n"

        echo -e "\t- init <sim_type>: initialize the specified simulation with all configuration files and folder structure"
        echo -e "\t- run <sim_type>: runs the specified simulation in the current folder"
        echo -e "\t- reset_output: removes all of the output files"
        echo -e "\t- create_topo: create initial topography"

        echo -e "\n"

        echo -e "The following simulation types (<sim_type>) are available:\n"

        echo -e "\tbedrock: the detachment-limited only model without soil cover"
        echo -e "\tsoil: the detachment-limited model with soil cover and weathering"
        echo -e "\tspace: he space-fluvial model with soil cover and weathering"
        echo -e "\tlpj: the coupled lpj-landlab model\n"

        echo -e "\n"

        echo -e "Examples:\n"

        echo "Initialize the soil simulation:"
        echo "$0 init soil"

        echo -e "\n"

        echo "Run the lpj coupled simulation:"
        echo "$0 run lpj"
    ;;
esac
