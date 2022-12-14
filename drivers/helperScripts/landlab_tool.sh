#!/bin/bash

PYTHON_BIN=/usr/bin/python3

echo "landlab_tool.sh, current working directory: $(pwd)"

if [[ -z "${LANDLABDRIVER}" ]]; then
    echo "Environment variable LANDLABDRIVER is not set"
    echo "You should load the module first:"
    echo "module load landlab"
    echo "(This is only relevant when not using the singularity container)"
    exit 1
fi

if [[ -z "${LANDLABTEMPLATE}" ]]; then
    echo "Environment variable LANDLABTEMPLATE is not set"
    echo "You should load the module first:"
    echo "module load landlab"
    echo "(This is only relevant when not using the singularity container)"
    exit 1
fi

echo "LANDLABDRIVER: $LANDLABDRIVER"
echo "LANDLABTEMPLATE: $LANDLABTEMPLATE"

case "$1" in
    init)
        bash $LANDLABDRIVER/helperScripts/makeModelSetup.sh $2
    ;;
    run)
        case "$2" in
            bedrock)
                $PYTHON_BIN $LANDLABDRIVER/pureBedrock/runfile_textinput.py
            ;;
            soil)
                $PYTHON_BIN $LANDLABDRIVER/soilLayer/runfile_textinput_soil.py
            ;;
            space)
                $PYTHON_BIN $LANDLABDRIVER/soilLayerSpace/runfile_textinput_soilSpace.py
            ;;
            lpj)
                $PYTHON_BIN $LANDLABDRIVER/lpj_coupled/runfile_space.py
            ;;
            *)
                echo "Unknown simulation type: $2"
            ;;
        esac
    ;;
    reset_output)
        rm -f BED/*
        rm -f ACC/*
        rm -f CSVOutput/*
        rm -f DEM/*
        rm -f DHDT/*
        rm -f Ksn/*
        rm -f NC/*
        rm -f SA/*
        rm -f SoilDepth/*
        rm -f SoilP/*
        rm -f dd/*
        rm -f dynveg_lpjguess.log
        rm -f Multiplot_absolut.png
        rm -f vegi_P_bugfix.png

        # For LPJ:
        rm -f -r temp_lpj
        rm -f temp_output/*
        rm -rf debugging/*
        rm -f myjob*
        for f in ll_output/*; do rm -f "$f"/*; done

    ;;
    create_topo)
        $PYTHON_BIN $LANDLABDRIVER/helperScripts/createStandartTopo.py
    ;;
    template)
        case "$2" in
            az)
                cp -v -r $LANDLABTEMPLATE/TemplateAZ .
            ;;
            lc)
                cp -v -r $LANDLABTEMPLATE/TemplateLC .
            ;;
            na)
                cp -v -r $LANDLABTEMPLATE/TemplateNA .
            ;;
            sg)
                cp -v -r $LANDLABTEMPLATE/TemplateSG .
            ;;
            *)
                echo "Unknown template"
                echo "Use one of these: az, lc, na, sg"
            ;;
        esac
    ;;
    plot)
        $PYTHON_BIN $LANDLABDRIVER/helperScripts/plot_script.py
    ;;
    *)
        echo "This script initializes and runs the apropriate landlab model."
        echo -e "The following sub-commands are available:\n"

        echo -e "\t- init <sim_type>: initialize the specified simulation with all configuration files and folder structure"
        echo -e "\t- run <sim_type>: runs the specified simulation in the current folder"
        echo -e "\t- reset_output: removes all of the output files"
        echo -e "\t- create_topo: create initial topography"
        echo -e "\t- template <location>: copy the data for a different location into the current folder"
        echo -e "\t- plot: creates an overview plot in the simulation folder"

        echo -e "\n"

        echo -e "The following simulation types (<sim_type>) are available:\n"

        echo -e "\t- bedrock: the detachment-limited only model without soil cover"
        echo -e "\t- soil: the detachment-limited model with soil cover and weathering"
        echo -e "\t- space: he space-fluvial model with soil cover and weathering"
        echo -e "\t- lpj: the coupled lpj-landlab model\n"

        echo -e "\n"

        echo -e "The following locations (<location>) are available:\n"

        echo -e "\t- az: Pan de Azucar"
        echo -e "\t- lc: La Campana"
        echo -e "\t- na: Nahuelbuta"
        echo -e "\t- sg: Santa Gracia"

        echo -e "\n"

        echo -e "Examples:\n"

        SCRIPT_NAME=$(basename $0)

        echo "Initialize the soil simulation:"
        echo "$SCRIPT_NAME init soil"

        echo -e "\n"

        echo "Run the lpj coupled simulation:"
        echo "$SCRIPT_NAME run lpj"

        echo -e "\n"

        echo "Copy data from La Campana into current folder:"
        echo "$SCRIPT_NAME template lc"
    ;;
esac
