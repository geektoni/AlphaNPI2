#!/bin/bash

export command_set="none"
if [ ${1} == "complete" ]; then
  export command_set=""
elif [ ${1} == "without-partition-update" ]; then
  export command_set="--without-partition-update"
else
  export command_set="--reduced-operation-set"
fi

export stack=""
if [ ${2} == "True" ]; then
  export stack="--expose-stack"
else
  export stack="none"
fi

export train_errors=""
if [ ${3} == "True" ]; then
  export train_errors="0.0"
else
  export train_errors="0.3"
fi

export output_dir_tb=${4}
export seed=${5}

export result_name=${1}-${2}-${3}-${5}

qsub -V -N "$result_name" -q common_cpuQ train_model.sh
