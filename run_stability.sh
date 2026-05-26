#!/bin/bash
# Stability test: seeds 42–51, 8 experiments per seed, 4 GPUs
# Each seed runs 8 experiments in parallel (2 per GPU), then waits before next seed.

MHCBAN="$(cd "$(dirname "$0")" && pwd)"
PY=/home/nibiohnproj9/cycheng/miniforge3/envs/jp_214/bin/python3
LOG=$MHCBAN/logs/stability
mkdir -p "$LOG"

CFG=$MHCBAN/configs/mhcban_config.yaml

for seed in {42..51}; do
    echo "========================================"
    echo "Launching seed $seed ..."
    echo "========================================"

    # GPU 0: hla_A_random + da_A_to_B
    CUDA_VISIBLE_DEVICES=0 nohup $PY -u train.py \
        --cfg $CFG --hla A --split random --seed $seed \
        > $LOG/hla_A_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=0 nohup $PY -u train_da.py \
        --cfg $CFG --source A --target B --seed $seed \
        > $LOG/da_A_to_B_seed${seed}.log 2>&1 &

    # GPU 1: hla_B_random + da_A_to_C
    CUDA_VISIBLE_DEVICES=1 nohup $PY -u train.py \
        --cfg $CFG --hla B --split random --seed $seed \
        > $LOG/hla_B_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 nohup $PY -u train_da.py \
        --cfg $CFG --source A --target C --seed $seed \
        > $LOG/da_A_to_C_seed${seed}.log 2>&1 &

    # GPU 2: hla_A_cluster + hla_B_cluster
    CUDA_VISIBLE_DEVICES=2 nohup $PY -u train.py \
        --cfg $CFG --hla A --split cluster --seed $seed \
        > $LOG/hla_A_cluster_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=2 nohup $PY -u train.py \
        --cfg $CFG --hla B --split cluster --seed $seed \
        > $LOG/hla_B_cluster_seed${seed}.log 2>&1 &

    # GPU 3: hla_C_random + hla_C_cluster
    CUDA_VISIBLE_DEVICES=3 nohup $PY -u train.py \
        --cfg $CFG --hla C --split random --seed $seed \
        > $LOG/hla_C_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=3 nohup $PY -u train.py \
        --cfg $CFG --hla C --split cluster --seed $seed \
        > $LOG/hla_C_cluster_seed${seed}.log 2>&1 &

    # Wait for all 8 to finish before launching next seed
    wait
    echo "Seed $seed done."
done

echo "All stability tests completed."
