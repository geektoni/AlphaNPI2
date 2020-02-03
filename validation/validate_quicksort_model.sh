#!/bin/bash

# Decide which action set we are using
if [ $2 == "reduced" ]; then
  actions=( "PARTITION" "QUICKSORT_UPDATE" "QUICKSORT" )
elif [ $2 == "no-partition" ]; then
  actions=( "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE" "QUICKSORT" )
else
  actions=( "PARTITION_UPDATE" "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE" "QUICKSORT" )
fi

# Run the validation for all the actions
for action in ${actions[@]}
do
  echo "[*] Trying action ${action} for model $1"
  python validate_quicksorting.py --load-path $1 --operation ${action} --save-results --verbose
done