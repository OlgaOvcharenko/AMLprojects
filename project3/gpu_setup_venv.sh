#!/bin/bash

#SBATCH -o logs/log-%j.out
#SBATCH --nodes=1
#SBATCH --gpus=rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

module load python

if [[ ! -d "python_env" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  python -m venv python_venv

  source "python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  pip install -r requirements.txt
fi
