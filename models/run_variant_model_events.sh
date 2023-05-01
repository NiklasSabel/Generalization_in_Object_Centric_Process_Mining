#!/bin/bash
#SBATCH --partition=dev_single
#SBATCH --nodes=1
#SBATCH --time=15
#SBATCH --export=NONE

python run_variant_model_events.py