#!/bin/bash

execute_model() {
  python validate_quicksorting.py --output-dir ../final_validations --op $2 --load-path $1
}

#operations=("PARTITION_UPDATE" "PARTITION" "SAVE_LOAD_PARTITION" "QUICKSORT_UPDATE" "QUICKSORT")

for op in ${operations[@]};
do
    ((j=j%1)); ((j++==0)) && wait
    for f in `ls ../final_models/*.pth`;
    do
      #((i=i%5)); ((i++==0)) && wait
      echo "[*] Executing " ${op} ${f}
      execute_model ${f} ${op}
    done
done

