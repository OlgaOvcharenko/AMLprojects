#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:32G
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs

module load python

nvidia-smi

source "python_venv/bin/activate"

#python3 try_euler.py
python3 main.py
