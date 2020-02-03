#!/bin/bash
#PBS -l select=5:ncpus=15:mem=5GB
#PBS -l walltime=48:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -V
#PBS -m be

cd AlphaNPI2

mkdir -p results

cd ./validation/

export PYTHONPATH=../

bash validate_quicksort_model.sh $filename $operations
