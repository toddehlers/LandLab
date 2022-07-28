#!/bin/bash -l
## Example run script for SLURM

## General configuration options
#SBATCH -J Landlab_test
#SBATCH -o Landlab_test.%j.%N.out
#SBATCH -e Landlab_test.%j.%N_Err.out
#SBATCH --mail-user=willi.kappler@uni-tuebingen.de
#SBATCH --mail-type=ALL

## Machine and CPU configuration
## Number of tasks per job:
#SBATCH -n 1
## Number of nodes:
#SBATCH -N 1

echo "SLURM, current folder: $(pwd)"

echo "SLURM, path: $PATH"
echo "SLURM, apptainer binary: $(ls -l /usr/bin/apptainer)"
echo "SLURM, submit host: $SLURM_SUBMIT_HOST"
echo "SLURM, node name: $SLURMD_NODENAME"
cd $SLURM_SUBMIT_DIR

echo "SLURM, current folder: $(pwd)"

apptainer exec --bind /esd /esd/esd01/share/archive/software/singularity_container/landlab_pecube_icecascade/ubuntu_18_04.sif landlab_tool.sh run lpj

apptainer exec --bind /esd /esd/esd01/share/archive/software/singularity_container/landlab_pecube_icecascade/ubuntu_18_04.sif landlab_tool.sh plot

