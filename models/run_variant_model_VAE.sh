#!/bin/bash
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --time=72:00:00
#SBATCH --export=NONE
#SBATCH --mem=180000mb



python run_variant_model_VAE.py