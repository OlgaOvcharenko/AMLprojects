#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00

mkdir -p logs
mkdir -p results
mkdir -p models

module load python

nvidia-smi

source "python_venv/bin/activate"

python lstm/main.py
