#!/bin/bash

operations=("PARTITION_UPDATE" "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE" "QUICKSORT")

for op in ${operations[@]};
do
  if [ ${op} == "PARTITION_UPDATE" ]; then
    python parse_validation_results.py --op ${op} \
      --dir "../final_validations" --save --legend --line --title --net
  else
    python parse_validation_results.py --op ${op} \
      --dir "../final_validations" --save --legend --line --net
  fi
done