#!/bin/bash
#SBATCH --partition=multiple
#SBATCH --nodes=3
#SBATCH --time=30:00:00
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=20

python weighting_parallel.py