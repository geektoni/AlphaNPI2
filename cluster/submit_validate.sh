#!/bin/bash

export filename=$1
export operations=$2

qsub -V -N "$filename" -q common_cpuQ validate_model.sh
