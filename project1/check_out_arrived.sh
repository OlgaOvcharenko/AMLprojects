#!/bin/bash

if [ -f data/out.csv ]; then
    counter=$(ls data | wc -l | sed 's/ //g')
    mv data/out.csv data/out_${counter}.csv
fi
