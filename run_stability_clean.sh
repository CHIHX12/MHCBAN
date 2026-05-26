#!/bin/bash
# 方案1 + 方案2 穩定性測試（乾淨資料）：seeds 42–51
# 資料來源：datasets_clean/（已移除194筆PTM修飾胜肽）
# 結果輸出：results_clean/
# GPU 0: HLA-A random + DA: A→B
# GPU 1: HLA-B random + DA: A→C
# GPU 2: HLA-A cluster + HLA-B cluster
# GPU 3: HLA-C random + HLA-C cluster

PY=/home/nibiohnproj9/cycheng/miniforge3/envs/jp_214/bin/python3
MHCBAN=/home/nibiohnproj9/cycheng/TCR_A_B_C/MHCBAN
CFG=$MHCBAN/configs/mhcban_config_clean.yaml
DATA=$MHCBAN/datasets_clean
LOG=$MHCBAN/logs/stability_clean
mkdir -p $LOG

for seed in {42..51}; do
    echo "========================================"
    echo "Launching seed $seed ..."
    echo "========================================"

    # GPU 0: HLA-A random + DA A→B
    CUDA_VISIBLE_DEVICES=0 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla A --split random --seed $seed --dataset_dir $DATA \
        > $LOG/hla_A_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=0 nohup $PY -u $MHCBAN/train_da.py \
        --cfg $CFG --source A --target B --seed $seed --dataset_dir $DATA \
        > $LOG/da_A_to_B_seed${seed}.log 2>&1 &

    # GPU 1: HLA-B random + DA A→C
    CUDA_VISIBLE_DEVICES=1 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla B --split random --seed $seed --dataset_dir $DATA \
        > $LOG/hla_B_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 nohup $PY -u $MHCBAN/train_da.py \
        --cfg $CFG --source A --target C --seed $seed --dataset_dir $DATA \
        > $LOG/da_A_to_C_seed${seed}.log 2>&1 &

    # GPU 2: HLA-A cluster + HLA-B cluster
    CUDA_VISIBLE_DEVICES=2 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla A --split cluster --seed $seed --dataset_dir $DATA \
        > $LOG/hla_A_cluster_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=2 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla B --split cluster --seed $seed --dataset_dir $DATA \
        > $LOG/hla_B_cluster_seed${seed}.log 2>&1 &

    # GPU 3: HLA-C random + HLA-C cluster
    CUDA_VISIBLE_DEVICES=3 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla C --split random --seed $seed --dataset_dir $DATA \
        > $LOG/hla_C_random_seed${seed}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=3 nohup $PY -u $MHCBAN/train.py \
        --cfg $CFG --hla C --split cluster --seed $seed --dataset_dir $DATA \
        > $LOG/hla_C_cluster_seed${seed}.log 2>&1 &

    wait
    echo "Seed $seed done."
done

echo "All clean stability tests (plan1 + plan2) completed."
