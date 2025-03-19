#!/bin/bash -l

#SBATCH --partition=day
#SBATCH --time=1-00:00:00
#SBATCH --account=bela
#SBATCH --nodes=2

#SBATCH --job-name=mjax
#SBATCH --output=mjax.out    
#SBATCH --error=mjax.err

conda activate baflnew
srun --nodes=2 --ntasks-per-node=4 python multijax.py
