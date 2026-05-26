#!/bin/bash
# 方案3 (Transfer Learning) + 方案4 (Pan-HLA) 同時執行
# GPU 0: 方案3 — Pre-train HLA-A → Fine-tune HLA-B cluster
# GPU 1: 方案3 — Pre-train HLA-A+B → Fine-tune HLA-C cluster + random
# GPU 2: 方案4 — Pan-HLA A+B+C random
# GPU 3: 方案4 — Pan-HLA A+B+C cluster

PY=/home/nibiohnproj9/cycheng/miniforge3/envs/jp_214/bin/python3
MHCBAN=/home/nibiohnproj9/cycheng/TCR_A_B_C/MHCBAN
CFG=$MHCBAN/configs/mhcban_config.yaml
LOG=$MHCBAN/logs
mkdir -p $LOG

echo "======================================"
echo "Launching 方案3 + 方案4"
echo "======================================"

# GPU 0: 方案3 Chain A — Pre-train HLA-A → Fine-tune HLA-B cluster (freeze=none)
CUDA_VISIBLE_DEVICES=0 nohup bash -c "
  echo '[GPU0] Phase1: Pre-train HLA-A random' &&
  $PY -u $MHCBAN/train_pretrain.py --cfg $CFG --hla A --split random &&
  echo '[GPU0] Phase2: Fine-tune HLA-B cluster (freeze=none)' &&
  $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
    --checkpoint $MHCBAN/results/pretrain_HLAA_random_seed42 \
    --hla B --split cluster --freeze none --lr 1e-4 --epochs 50
" > $LOG/plan3_A_to_B.log 2>&1 &
PID0=$!
echo "Plan3 A→B    PID: $PID0  (GPU 0)  log: $LOG/plan3_A_to_B.log"

# GPU 1: 方案3 Chain B — Pre-train HLA-A+B → Fine-tune HLA-C cluster + random (freeze=encoder)
CUDA_VISIBLE_DEVICES=1 nohup bash -c "
  echo '[GPU1] Phase1: Pre-train HLA-A+B random' &&
  $PY -u $MHCBAN/train_pretrain.py --cfg $CFG --hla A B --split random &&
  echo '[GPU1] Phase2a: Fine-tune HLA-C cluster (freeze=encoder)' &&
  $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
    --checkpoint $MHCBAN/results/pretrain_HLAAB_random_seed42 \
    --hla C --split cluster --freeze encoder --lr 1e-4 --epochs 50 &&
  echo '[GPU1] Phase2b: Fine-tune HLA-C random (freeze=encoder)' &&
  $PY -u $MHCBAN/train_finetune.py --cfg $CFG \
    --checkpoint $MHCBAN/results/pretrain_HLAAB_random_seed42 \
    --hla C --split random --freeze encoder --lr 1e-4 --epochs 50
" > $LOG/plan3_AB_to_C.log 2>&1 &
PID1=$!
echo "Plan3 AB→C   PID: $PID1  (GPU 1)  log: $LOG/plan3_AB_to_C.log"

# GPU 2: 方案4 — Pan-HLA A+B+C random split
CUDA_VISIBLE_DEVICES=2 nohup $PY -u $MHCBAN/train_pretrain.py \
  --cfg $CFG --hla A B C --split random \
  > $LOG/plan4_pan_random.log 2>&1 &
PID2=$!
echo "Plan4 pan random  PID: $PID2  (GPU 2)  log: $LOG/plan4_pan_random.log"

# GPU 3: 方案4 — Pan-HLA A+B+C cluster split (allele cold-start)
CUDA_VISIBLE_DEVICES=3 nohup $PY -u $MHCBAN/train_pretrain.py \
  --cfg $CFG --hla A B C --split cluster \
  > $LOG/plan4_pan_cluster.log 2>&1 &
PID3=$!
echo "Plan4 pan cluster PID: $PID3  (GPU 3)  log: $LOG/plan4_pan_cluster.log"

echo ""
echo "All launched. Monitor with:"
echo "  tail -f $LOG/plan3_A_to_B.log"
echo "  tail -f $LOG/plan3_AB_to_C.log"
echo "  tail -f $LOG/plan4_pan_random.log"
echo "  tail -f $LOG/plan4_pan_cluster.log"
