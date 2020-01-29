#!/bin/bash
#PBS -l select=3:ncpus=10:mem=5GB
#PBS -l walltime=48:0:0
#PBS -q common_cpuQ
#PBS -M giovanni.detoni@studenti.unitn.it
#PBS -V
#PBS -m be

# Strict bash mode
# Disabled since it does not work well
# on the cluster environment
#set -euo pipefail
#IFS=$'\n\t'

cd AlphaNPI2

mkdir -p ${output_dir_tb}

cd ./trainings/

export PYTHONPATH=../

if [ ${command_set} != "none" ]; then
  if [ ${stack} != "none" ]; then
    python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model --save-results \
      --num-cpus 10 --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      ${command_set} ${stack}
  else
    python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model --save-results \
      --num-cpus 10 --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors} \
      ${command_set}
  fi
else
  if [ ${stack} != "none" ]; then
    python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model --save-results \
      --num-cpus 10 --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors}\
      ${stack}
  else
    python3 train_quicksorting.py --seed=${seed} --tensorboard --verbose --save-model --save-results \
      --num-cpus 10 --tb-base-dir ${output_dir_tb} --penalize-level-0 --keep-training --sample-error-prob ${train_errors}
  fi
fi