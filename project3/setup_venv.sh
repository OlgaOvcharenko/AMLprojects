#!/bin/bash

if [[ ! -d "python_env" ]]; then
  echo "Create Python Virtual Environment on $HOSTNAME"

  module load eth_proxy gcc/11.4.0 python/3.11.6 cuda/12.1.1 

  source "python_venv/bin/activate"

  pip install --upgrade pip
  pip install --upgrade pip

  pip3 install -r requirements.txt
fi
