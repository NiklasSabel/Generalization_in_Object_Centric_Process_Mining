#!/bin/bash
#SBATCH --partition=multiple
#SBATCH --nodes=2
#SBATCH --time=30:00:00
#SBATCH --export=NONE
#SBATCH --ntasks-per-node=6



python run_variant_model_negative.py