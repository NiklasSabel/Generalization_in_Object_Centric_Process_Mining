#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --export=NONE

python run_variant_model.py