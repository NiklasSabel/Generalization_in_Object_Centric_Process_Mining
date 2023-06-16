#!/bin/bash
#SBATCH --partition=multiple
#SBATCH --nodes=6
#SBATCH --time=30:00:00
#SBATCH --export=NONE


python run_variant_model_negative.py