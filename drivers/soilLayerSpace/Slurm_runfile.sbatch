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

module load landlab
landlab_tool.sh run space
