#!/bin/bash

operations=("QUICKSORT" "PARTITION_UPDATE" "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE")

for op in ${operations[@]};
do
  if [ ${op} == "PARTITION_UPDATE" ]; then
    python parse_validation_results.py --op ${op} \
      --dir ../evaluation_results/validations --save --legend --line --title --net
    python parse_validation_results.py --op ${op} \
      --dir ../evaluation_results/validations --save --legend --line --title
  else
    python parse_validation_results.py --op ${op} \
      --dir ../evaluation_results/validations --save --legend --line --title --net
    python parse_validation_results.py --op ${op} \
      --dir ../evaluation_results/validations --save --legend --line --title
  fi
done
