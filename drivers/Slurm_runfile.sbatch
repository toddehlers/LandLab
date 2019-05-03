#!/bin/bash -l
## Example run script for SLURM

## General configuration options
#SBATCH -J MyProject
#SBATCH -o myjob.%j.%N.out
#SBATCH -e myjob.%j.%N_Err.out
#SBATCH --mail-user=nobody@nowhere.com
#SBATCH --mail-type=ALL

## Machine and CPU configuration
## Number of tasks per job:
#SBATCH -n 1
## Number of nodes:
#SBATCH -N 1

module load landlab

python3 ./runfile_textinput.py
