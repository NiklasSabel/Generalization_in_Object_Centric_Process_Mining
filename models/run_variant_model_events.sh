#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=25:00:00
#SBATCH --export=NONE

python run_variant_model_events.py