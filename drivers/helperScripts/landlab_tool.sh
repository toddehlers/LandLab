#!/bin/bash

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
        python3 $LANDLABDRIVER/helperScripts/createStandartTopo.py
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

        echo -e "\t- bedrock: the detachment-limited only model without soil cover"
        echo -e "\t- soil: the detachment-limited model with soil cover and weathering"
        echo -e "\t- space: he space-fluvial model with soil cover and weathering"
        echo -e "\t- lpj: the coupled lpj-landlab model\n"

        echo -e "\n"

        echo -e "Examples:\n"

        SCRIPT_NAME=$(basename $0)

        echo "Initialize the soil simulation:"
        echo "$SCRIPT_NAME init soil"

        echo -e "\n"

        echo "Run the lpj coupled simulation:"
        echo "$SCRIPT_NAME run lpj"
    ;;
esac
