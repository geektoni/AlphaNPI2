#!/bin/bash

export filename=$1
export operations=$2

qsub -V -N "$3" -q common_cpuQ validate_model.sh
