#!/bin/bash
#SBATCH --partition=multiple
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=40



python run_variant_model_negative.py