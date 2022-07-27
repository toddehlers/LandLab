#!/bin/bash -l
## Example run script for SLURM

## General configuration options
#SBATCH -J Landlab_Jobname
#SBATCH -o myjob.%j.%N.out
#SBATCH -e myjob.%j.%N_Err.out
#SBATCH --mail-user=some.user@uni-tuebingen.de
#SBATCH --mail-type=ALL

## Machine and CPU configuration
## Number of tasks per job:
#SBATCH -n 1
## Number of nodes:
#SBATCH -N 1

apptainer exec /esd/esd01/share/archive/software/singularity_container/landlab_pecube_icecascade/ubuntu_18_04.sif landlab_tool.sh lpj run

apptainer exec /esd/esd01/share/archive/software/singularity_container/landlab_pecube_icecascade/ubuntu_18_04.sif landlab_tool.sh plot


