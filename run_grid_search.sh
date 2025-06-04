#!/bin/bash -ex

LOG_FILE="grid_search_results.log"
MAX_CONCURRENT_JOBS=8

function check_gpu_status() {
    local gpu_id=$1
    local threshold=2000
    local gpu_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $gpu_id)
    if [ "$gpu_memory" -lt "$threshold" ]; then
        echo "1"
    else
        echo "0"
    fi
}

function get_available_gpu() {
    for gpu_id in {0..7}; do
        if [ "$(check_gpu_status $gpu_id)" -eq 1 ]; then
            echo $gpu_id
            return
        fi
    done
    echo "-1"
}

for lambda_ in 0.1 0.5 1.0 5.0 10.0
do
  for alpha_1 in 100 1000 10000 100000
  do
    for alpha_2 in 100 1000 10000 100000
    do
      for gamma in 0.1 0.5 1.0 5.0 10.0
      do
        while true
        do
            available_gpu=$(get_available_gpu)
            if [ "$available_gpu" -ne -1 ]; then
                echo "Running with lambda_=$lambda_, alpha_1=$alpha_1, alpha_2=$alpha_2, gamma=$gamma on GPU $available_gpu" | tee -a $LOG_FILE
                python DGP-GCL.py --DS ogbg-moltox21 --model_type dgp-gcl --lr 0.01 --device $available_gpu --DS_pair ogbg-moltox21+ogbg-molsider --lambda_ $lambda_ --alpha_1 $alpha_1 --alpha_2 $alpha_2 --gamma $gamma >> $LOG_FILE 2>&1 &
                sleep 10
                break
            else
                echo "No available GPUs. Waiting for a GPU to become available..." | tee -a $LOG_FILE
                sleep 30
            fi
        done
        running_jobs=$(jobs -p | wc -l)
        if [ "$running_jobs" -ge "$MAX_CONCURRENT_JOBS" ]; then
            wait -n
        fi
      done
    done
  done
done

wait
echo "All jobs completed." | tee -a $LOG_FILE