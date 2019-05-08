#!/bin/bash

LANDLABDRIVER=/usr/share/modules/Modules/3.2.10/landlab/drivers

case "$1" in
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
    reset_output)
        rm ACC/* CSVOutput/* DEM/* DHDT/* Ksn/* NC/* SA/* SoilDepth/* dd/* dynveg_lpjguess.log Multiplot_absolut.png vegi_P_bugfix.png
        mkdir ACC CSVOutput DEM DHDT Ksn NC SA SoilDepth dd
    ;;
    *)
        echo "This script runs the apropriate landlab model:\n"
        echo "bedrock: the detachment-limited only model without soil cover"
        echo "soil: the detachment-limited model with soil cover and weathering"
        echo "space: he space-fluvial model with soil cover and weathering"
        echo "lpj: the coupled lpj-landlab model"
        echo "reset_output: removes all of the output files"
    ;;
esac
