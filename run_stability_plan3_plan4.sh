#!/bin/bash
# 方案3 + 方案4 穩定性測試：seeds 42–51
# GPU 0: 方案3 Chain A — Pre-train HLA-A → Fine-tune HLA-B cluster
# GPU 1: 方案3 Chain B — Pre-train HLA-A+B → Fine-tune HLA-C cluster + random
# GPU 2: 方案4 — Pan-HLA A+B+C random
# GPU 3: 方案4 — Pan-HLA A+B+C cluster
# Each seed runs 4 chains in parallel, then waits before next seed.

PY=/home/nibiohnproj9/cycheng/miniforge3/envs/jp_214/bin/python3
MHCBAN=/home/nibiohnproj9/cycheng/TCR_A_B_C/MHCBAN
CFG=$MHCBAN/configs/mhcban_config.yaml
LOG=$MHCBAN/logs/stability_plan3_plan4
mkdir -p $LOG

for seed in {42..51}; do
    echo "========================================"
    echo "Launching seed $seed ..."
    echo "========================================"

    # GPU 0: 方案3 Chain A
    CUDA_VISIBLE_DEVICES=0 nohup bash -c "
      $PY -u $MHCBAN/train_pretrain.py --cfg $CFG --hla A --split random --seed $seed &&
      $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
        --checkpoint $MHCBAN/results/pretrain_HLAA_random_seed${seed} \
        --hla B --split cluster --freeze none --lr 1e-4 --epochs 50 --seed $seed
    " > $LOG/plan3_A_to_B_seed${seed}.log 2>&1 &

    # GPU 1: 方案3 Chain B
    CUDA_VISIBLE_DEVICES=1 nohup bash -c "
      $PY -u $MHCBAN/train_pretrain.py --cfg $CFG --hla A B --split random --seed $seed &&
      $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
        --checkpoint $MHCBAN/results/pretrain_HLAAB_random_seed${seed} \
        --hla C --split cluster --freeze encoder --lr 1e-4 --epochs 50 --seed $seed &&
      $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
        --checkpoint $MHCBAN/results/pretrain_HLAAB_random_seed${seed} \
        --hla C --split random --freeze encoder --lr 1e-4 --epochs 50 --seed $seed
    " > $LOG/plan3_AB_to_C_seed${seed}.log 2>&1 &

    # GPU 2: 方案4 Pan random
    CUDA_VISIBLE_DEVICES=2 nohup $PY -u $MHCBAN/train_pretrain.py \
      --cfg $CFG --hla A B C --split random --seed $seed \
      > $LOG/plan4_pan_random_seed${seed}.log 2>&1 &

    # GPU 3: 方案4 Pan cluster
    CUDA_VISIBLE_DEVICES=3 nohup $PY -u $MHCBAN/train_pretrain.py \
      --cfg $CFG --hla A B C --split cluster --seed $seed \
      > $LOG/plan4_pan_cluster_seed${seed}.log 2>&1 &

    wait
    echo "Seed $seed done."
done

echo "All stability tests (plan3 + plan4) completed."
