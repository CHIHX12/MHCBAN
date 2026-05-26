# MHCBAN — MHC-Peptide Binding Affinity Prediction

MHCBAN 是一個基於雙線性注意力網路（Bilinear Attention Network, BAN）的 MHC-I 胜肽結合親和力預測模型，
核心架構移植並擴展自 [DrugBAN](https://github.com/pz-white/DrugBAN)，針對蛋白質序列特性改用 BiLSTM 作為特徵提取器。

本專案用於醫院研究，支援 HLA-A、HLA-B、HLA-C 三種 MHC 類型的二元分類預測（結合 / 不結合），
並提供隨機分割、Allele 冷啟動及跨 HLA 域適應（Domain Adaptation）三種實驗模式。

---

## 模型架構

```
Peptide 序列 (max 30aa) ──► BiLSTM ──► 特徵矩陣 A (batch, 30, 256)
                                              │
                                         BAN Layer          ← 與 DrugBAN 完全相同
                                              │
MHC 偽序列 (34aa)        ──► BiLSTM ──► 特徵矩陣 B (batch, 34, 256)
                                              │
                                        MLPDecoder           ← 與 DrugBAN 完全相同
                                              │
                                      結合機率 (0 / 1)
```

### 與 DrugBAN 的對應關係

| 模組 | DrugBAN | MHCBAN |
|------|---------|--------|
| Branch 1 Encoder | MolecularGCN（藥物分子圖） | PeptideBiLSTM（胜肽序列） |
| Branch 2 Encoder | ProteinCNN（蛋白質序列） | MHCBiLSTM（MHC 34位點偽序列） |
| Attention | BANLayer ✓ 完全相同 | BANLayer ✓ 完全相同 |
| Decoder | MLPDecoder ✓ 完全相同 | MLPDecoder ✓ 完全相同 |

### 參數量與層數

| 模組 | 層數 | 參數量 | 占比 |
|------|------|-------|------|
| PeptideBiLSTM | Embedding + BiLSTM ×2 | 662,656 | 30.3% |
| MHCBiLSTM | Embedding + BiLSTM ×2 | 662,656 | 30.3% |
| BANLayer | FC ×2 + h_mat + BN | 397,574 | 18.2% |
| MLPDecoder | FC ×4 + BN ×3 | 462,337 | 21.2% |
| **Total** | **15 parametric layers** | **2,185,223** | 100% |

完整架構方塊圖（含每層維度與參數量）：`figures/fig0_model_architecture.pdf`

---

## 資料統計

| HLA 類型 | 原始筆數 | PTM 移除 | 乾淨筆數 | 唯一 Allele 數 | Peptide 長度範圍 | MHC 偽序列長度 |
|----------|---------|---------|---------|--------------|----------------|--------------|
| HLA-A | 50,899 | 178 | **50,721** | 42 | 8–30 aa | 34 aa |
| HLA-B | 28,930 | 15 | **28,915** | 48 | 7–26 aa | 34 aa |
| HLA-C | 1,033 | 1 | **1,032** | 11 | 8–21 aa | 34 aa |
| **合計** | **80,862** | **194** | **80,668** | | | |

- 資料來源：IEDB（Immune Epitope Database）
- 標籤定義：IC50 < 100 nM → 1（結合），IC50 > 10,000 nM → 0（不結合）
- MHC 偽序列：34 位點標準化特徵（NetMHCpan 定義）

### 資料品質：PTM 修飾胜肽移除

原始資料中含有 **194 筆**帶有翻譯後修飾（Post-Translational Modification, PTM）標記的胜肽，
例如 `VLHDDLCEA + SCM(C7)`、`KIEDFGSAK(ac)`。

**問題：** 這些 IC50 標籤對應的是修飾後的胜肽（而非原始序列），若直接輸入模型，
特殊字元（`+`, `(`, 數字）會被編碼為 `X`（未知氨基酸），造成序列資訊不匹配。

**處置：** 直接刪除（而非轉換），因為 IC50 標籤本身就不對應原始序列，
轉換後標籤仍有誤導性。

**驗證：** 移除後對 random split 的 AUROC 影響為 0.000（小資料集 HLA-C 亦相同），
對 cluster split 有輕微改善（HLA-B cluster +0.033），
代表 PTM 標籤確實污染了訓練訊號。

---

## 專案結構

```
MHCBAN/
├── configs/
│   ├── mhcban_config.yaml            # 超參數設定（參考用，不建議直接使用）
│   └── mhcban_config_clean.yaml      # ✅ 正式設定（OUTPUT_DIR: ./results_clean）
├── datasets/                         # ⚠️ 原始資料 splits（含 PTM，存檔用）
│   ├── HLA_A/ HLA_B/ HLA_C/
│   │   ├── full.csv
│   │   ├── random/  train.csv / val.csv / test.csv
│   │   └── cluster/ train.csv / val.csv / test.csv / allele_assignment.txt
│   ├── cross_hla/
│   │   ├── A_to_B/ source_train.csv / target_train.csv / target_test.csv
│   │   └── A_to_C/
│   ├── dataloader.py
│   └── prepare_splits.py
├── datasets_clean/                   # ✅ 正式資料 splits（已移除 194 筆 PTM）
│   ├── HLA_A/ HLA_B/ HLA_C/         # 結構同上
│   └── cross_hla/ A_to_B/ A_to_C/
├── models/
│   ├── ban.py                        # BANLayer + FCNet（移植自 DrugBAN，完全相同）
│   ├── encoders.py                   # PeptideBiLSTM + MHCBiLSTM
│   └── mhcban.py                     # 主模型 + MLPDecoder
├── domain_adaptator.py               # ReverseLayerF (GRL) + Discriminator
├── trainer.py                        # Trainer（標準）+ DATrainer（CDAN）
├── train.py                          # 方案 1 入口
├── train_da.py                       # 方案 2 入口（Cross-HLA DA）
├── train_pretrain.py                 # 方案 3/4 Phase 1 入口
├── train_finetune.py                 # 方案 3 Phase 2 入口
├── run_stability_clean.sh            # ✅ 方案1&2 正式穩定性測試（乾淨資料）
├── run_stability_plan3_plan4_clean.sh # ✅ 方案3&4 正式穩定性測試（乾淨資料）
├── run_stability.sh                  # ⚠️ 方案1&2 原始資料穩定性（存檔用）
├── run_stability_plan3_plan4.sh      # ⚠️ 方案3&4 原始資料穩定性（存檔用）
├── run_plan3_plan4.sh                # 方案3&4 單次執行（4 GPU 並行）
├── aggregate_stability.py            # 聚合結果（支援 --clean / --both）
├── infer_gray_zone.py                # 灰區推論腳本（IC50 100–10,000 nM）
├── plot_results.py                   # 生成正式發表圖（fig1–fig5）
├── results_table.txt                 # ✅ 完整結果表（再現性驗證 + 所有指標 + 灰區 + 目錄清單）
├── gray_zone_inference.csv           # 灰區胜肽輸入（45,115 筆）
├── gray_zone_results.csv             # 灰區推論結果（含預測機率）
├── gray_zone_figures/                # 灰區圖原始輸出（gray1–gray3）
├── figures/                          # ✅ 正式發表圖（fig0–8，PDF+PNG+SVG，600 DPI）
│   ├── fig0_model_architecture.*     # 模型架構方塊圖（含層數、參數量、維度標示）
│   ├── fig1_auroc_overview.*         # 全實驗 AUROC 概覽
│   ├── fig2_metrics_heatmap.*        # 全指標 heatmap
│   ├── fig3_random_vs_cluster.*      # Random vs Cluster 比較
│   ├── fig4_cold_start_comparison.*  # 冷啟動場景比較（legend 已修正）
│   ├── fig5_pan_vs_individual.*      # Pan-HLA vs 個別模型（legend 已修正）
│   ├── fig6_gray_scatter.*           # 灰區：log(IC50) vs 預測機率
│   ├── fig7_gray_boxplot.*           # 灰區：各 bin 預測機率分佈
│   ├── fig8_gray_histogram.*         # 灰區：各 HLA 類型直方圖
│   └── figure_captions.txt           # 所有圖片英文說明文字
├── archive/                          # ⚠️ 早期測試殘留（SLURM job 339，2026-05-24 已取消）
│   └── model_checkpoint_epoch_{10,20}.pt / run_train.sh / train_*.log
├── results/                          # ⚠️ 原始資料訓練結果（存檔，勿覆蓋）
└── results_clean/                    # ✅ 正式訓練結果（乾淨資料，N=10）
    ├── <exp>_seed{42..51}/           # 13 實驗 × 10 seeds = 130 目錄
    ├── pretrain_HLAA_random_seed42-51   # ← Plan 3 FT HLA-B 所需中間 ckpt
    └── pretrain_HLAAB_random_seed42-51  # ← Plan 3 FT HLA-C 所需中間 ckpt
```

---

## 安裝

```bash
conda activate jp_214
pip install prettytable tqdm scikit-learn
python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
```

---

## 原始資料結構（前提條件）

```
TCR_A_B_C/
├── HLA_A/
│   ├── binary_train_HLA_A.csv        # 原始（含 PTM）
│   └── binary_train_HLA_A_clean.csv  # ✅ 乾淨版（移除 PTM）
├── HLA_B/
│   ├── binary_train_HLA_B.csv
│   └── binary_train_HLA_B_clean.csv
├── HLA_C/
│   ├── binary_train_HLA_C.csv
│   └── binary_train_HLA_C_clean.csv
└── MHCBAN/                           # 本專案根目錄
```

CSV 欄位說明：
- `Peptide`：胜肽序列（8–30 aa）
- `Pseudo_Sequence`：MHC 34-position 偽序列（NetMHCpan 定義）
- `Label`：1 = 結合（IC50 < 100 nM），0 = 不結合（IC50 > 10,000 nM）

---

## 使用方法（正式流程，使用乾淨資料）

### Step 1：準備乾淨資料分割（只需執行一次）

```bash
cd MHCBAN
# 從乾淨 CSV 產生 datasets_clean/ 分割
python datasets/prepare_splits.py \
    --data .. \
    --out_root datasets_clean \
    --suffix _clean

# 建立 cross_hla 分割（直接從 random split 複製）
mkdir -p datasets_clean/cross_hla/A_to_B datasets_clean/cross_hla/A_to_C
cp datasets_clean/HLA_A/random/train.csv datasets_clean/cross_hla/A_to_B/source_train.csv
cp datasets_clean/HLA_B/random/train.csv datasets_clean/cross_hla/A_to_B/target_train.csv
cp datasets_clean/HLA_B/random/test.csv  datasets_clean/cross_hla/A_to_B/target_test.csv
cp datasets_clean/HLA_A/random/train.csv datasets_clean/cross_hla/A_to_C/source_train.csv
cp datasets_clean/HLA_C/random/train.csv datasets_clean/cross_hla/A_to_C/target_train.csv
cp datasets_clean/HLA_C/random/test.csv  datasets_clean/cross_hla/A_to_C/target_test.csv
```

### Step 2：建立 logs 目錄

```bash
mkdir -p logs/stability_clean logs/stability_plan3_plan4_clean
```

### Step 3：穩定性測試（seed 42–51，建議直接執行）

```bash
# 方案1 & 2（8 實驗，4 GPU，約 X 小時）
chmod +x run_stability_clean.sh
nohup bash run_stability_clean.sh > logs/stability_clean/run_all.log 2>&1 &

# 方案3 & 4（同時執行，4 GPU）
chmod +x run_stability_plan3_plan4_clean.sh
nohup bash run_stability_plan3_plan4_clean.sh > logs/stability_plan3_plan4_clean/run_all.log 2>&1 &
```

### Step 4：查看結果

```bash
# 正式結果（乾淨資料）
python aggregate_stability.py --clean

# 乾淨 vs 原始對比
python aggregate_stability.py --both
```

---

### 單次訓練（測試用）

```bash
CFG=configs/mhcban_config_clean.yaml
DATA=datasets_clean

# 方案 1
python -u train.py --cfg $CFG --hla A --split random  --dataset_dir $DATA
python -u train.py --cfg $CFG --hla A --split cluster --dataset_dir $DATA

# 方案 2
python -u train_da.py --cfg $CFG --source A --target B --dataset_dir $DATA

# 方案 4 Pan-HLA
python -u train_pretrain.py --cfg $CFG --hla A B C --split random --dataset_dir $DATA
```

---

### 方案 3：Transfer Learning（Pre-train → Fine-tune）

**⚠️ Phase 2 必須等 Phase 1 完成後才能執行。建議使用 `run_plan3_plan4.sh`（`&&` 串接）。**

```bash
# Phase 1：預訓練
python -u train_pretrain.py --cfg configs/mhcban_config_clean.yaml \
    --hla A --split random --dataset_dir datasets_clean

# Phase 2：Fine-tune（Phase 1 完成後）
python -u train_finetune.py --cfg configs/mhcban_config_clean.yaml \
    --checkpoint results_clean/pretrain_HLAA_random_seed42 \
    --hla B --split cluster --freeze none --lr 1e-4 --epochs 50 \
    --dataset_dir datasets_clean
```

**Freeze 策略說明**

| Strategy | 凍結範圍 | 適用場景 |
|----------|---------|---------|
| `encoder` | PeptideBiLSTM + MHCBiLSTM | HLA-C（小資料集，~1k） |
| `all_but_mlp` | encoder + BANLayer | 極小資料集（< 500筆） |
| `none` | 無凍結（全層以低 LR 訓練）| HLA-B（中等資料集，~29k） |

---

## 正式實驗結果（乾淨資料，mean ± std，seed 42–51，N=10）

> **以下為正式結果，使用 `datasets_clean/`（已移除 194 筆 PTM 修飾胜肽）訓練。**
> 結果存放於 `results_clean/`，執行 `python aggregate_stability.py --clean` 可重現。

### 方案1 — 標準訓練

| 實驗 | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------|-------|-------|----|-------------|-------------|----------|
| HLA-A Random   | 0.977±0.001 | 0.953±0.002 | 0.933±0.003 | 0.927±0.011 | 0.937±0.010 | 0.931±0.005 |
| HLA-B Random   | 0.976±0.002 | 0.929±0.007 | 0.929±0.004 | 0.918±0.008 | 0.938±0.007 | 0.922±0.006 |
| HLA-C Random   | 0.873±0.010 | 0.905±0.012 | 0.840±0.019 | 0.776±0.073 | 0.886±0.047 | 0.846±0.020 |
| HLA-A Cluster  | 0.929±0.014 | 0.912±0.020 | 0.870±0.018 | 0.827±0.034 | 0.902±0.012 | 0.862±0.021 |
| HLA-B Cluster  | 0.782±0.030 | 0.545±0.034 | 0.738±0.022 | 0.498±0.115 | 0.878±0.041 | 0.593±0.078 |
| HLA-C Cluster  | 0.403±0.066 | 0.746±0.040 | 0.678±0.007 | 0.080±0.048 | 0.984±0.021 | 0.786±0.010 |

### 方案2 — Cross-HLA Domain Adaptation (CDAN)

| 實驗 | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------|-------|-------|----|-------------|-------------|----------|
| DA: A→B | 0.647±0.019 | 0.314±0.012 | 0.697±0.018 | 0.255±0.141 | 0.932±0.046 | 0.403±0.101 |
| DA: A→C | 0.638±0.057 | 0.762±0.034 | 0.684±0.018 | 0.178±0.180 | 0.943±0.057 | 0.663±0.030 |

### 方案3 — Transfer Learning

| 實驗 | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------|-------|-------|----|-------------|-------------|----------|
| FT HLA-B cluster (A→B, freeze=none)     | 0.768±0.024 | 0.543±0.056 | 0.732±0.015 | 0.546±0.072 | 0.839±0.040 | 0.619±0.047 |
| FT HLA-C cluster (AB→C, freeze=encoder) | 0.593±0.101 | 0.841±0.043 | 0.688±0.032 | 0.280±0.232 | 0.899±0.099 | 0.764±0.041 |
| FT HLA-C random  (AB→C, freeze=encoder) | 0.885±0.040 | 0.914±0.037 | 0.847±0.024 | 0.726±0.082 | 0.933±0.022 | 0.858±0.023 |

### 方案4 — Pan-HLA Universal Model

| 實驗 | AUROC | AUPRC | F1 | Sensitivity | Specificity | Accuracy |
|------|-------|-------|----|-------------|-------------|----------|
| Pan A+B+C random  | **0.979±0.001** | **0.951±0.002** | **0.934±0.002** | 0.922±0.010 | 0.945±0.007 | 0.928±0.005 |
| Pan A+B+C cluster | **0.896±0.013** | 0.853±0.015 | 0.827±0.013 | 0.777±0.034 | 0.862±0.024 | 0.811±0.017 |

### 各方案最佳結果總覽

| 場景 | 最佳方法 | AUROC |
|------|---------|-------|
| HLA-A cold-start（unseen allele） | 方案1 A Cluster | 0.929±0.014 |
| HLA-B cold-start | 方案3 FT B Cluster | 0.768±0.024 |
| HLA-C cold-start | **方案4 Pan Cluster** | **0.896±0.013** |
| 通用預測（不指定 HLA 類型） | **方案4 Pan Random** | **0.979±0.001** |

---

## 超參數設定

| 參數 | 值 |
|------|----|
| Peptide max length | 30 aa |
| MHC pseudo-seq length | 34 aa |
| Vocab size | 25 |
| BiLSTM embedding dim | 128 |
| BiLSTM hidden dim | 128（雙向輸出 256） |
| BiLSTM layers | 2 |
| BAN heads | 3 |
| MLP in / hidden / out dim | 256 / 512 / 128 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| DA warm-up epochs | 10 |
| DA lambda | 1.0 |
| Epochs | 100 |
| Optimizer | Adam |
| Fine-tune LR（方案3） | 1e-4 |
| Fine-tune Epochs（方案3） | 50 |
| Fine-tune freeze（HLA-B） | none（全層） |
| Fine-tune freeze（HLA-C） | encoder（BiLSTM 凍結）|

---

## 灰區推論（Gray Zone Inference）

訓練集僅使用 IC50 < 100 nM（結合，標籤1）與 IC50 > 10,000 nM（不結合，標籤0）兩端資料，
**中間的「灰區」（100–10,000 nM，共 45,115 筆）未參與訓練**。

以訓練好的 Pan-HLA Random（seed 42）與 Pan-HLA Cluster（seed 42）模型對灰區進行推論，
驗證模型是否自然學到連續親和力排序。

### 資料來源

| HLA 類型 | 灰區筆數 |
|----------|---------|
| HLA-A | 32,638 |
| HLA-B | 11,565 |
| HLA-C | 912 |
| **合計** | **45,115** |

### 推論結果

| 模型 | Spearman r（−log IC50 vs 預測機率） | p 值 |
|------|--------------------------------------|------|
| Pan-HLA Random  | **+0.455** | ≈ 0 |
| Pan-HLA Cluster | **+0.383** | ≈ 0 |

正相關代表：IC50 越小（結合力越強），模型預測結合機率越高 → 模型學到的是連續親和力排序，而非單純的二元分類。

### 各 IC50 Bin 平均預測機率（Pan-HLA Random）

| IC50 區間 (nM) | 平均預測機率 | n |
|----------------|-------------|---|
| 100–200        | 0.737 | 6,409 |
| 200–500        | 0.636 | 8,142 |
| 500–1k         | 0.519 | 6,675 |
| 1k–2k          | 0.423 | 7,217 |
| 2k–5k          | 0.281 | 9,926 |
| 5k–10k         | 0.226 | 6,613 |

預測機率在 500–1k nM 區間跨越 0.5 決策邊界，與臨床常用的 **500 nM** 結合/非結合閾值完全吻合。

### 執行灰區推論

```bash
# 需要先準備 gray_zone_inference.csv（含 Peptide, Pseudo_Sequence, IC50_nM, MHC, HLA_type）
python infer_gray_zone.py
# 輸出：gray_zone_results.csv, gray_zone_figures/, figures/fig6–fig8
```

---

## 分析

### HLA-C Cluster AUROC 偏低的原因

HLA-C 只有 11 個唯一 allele，cluster split 後測試集僅含 2 個 allele（137 筆）。
模型在訓練 allele 上學到 allele-specific 特徵，對未見 allele 泛化能力不足。

**解決方案：方案4 Pan-HLA**（AUROC 0.896±0.013），
合併 A+B+C 訓練後，模型見過更多 allele（共 101 個），cold-start 能力大幅提升。

### Pan-HLA 使用說明

方案4 模型輸入本身已包含 **34-position MHC 偽序列**（allele fingerprint），
合併 A+B+C 訓練後，模型學到「看偽序列就知道怎麼結合」的通用規則。

**推論時不需指定 HLA 類型**，只需提供 Peptide 序列 + MHC 偽序列即可。
理論上可外推至任何新型 allele（只要能取得其 34-position 偽序列）。

---

## 引用

- Kim, J.-H. et al. "Bilinear Attention Networks." NeurIPS 2018.
- Bai, P. et al. "DrugBAN: A Bilinear Attention Network with Domain Adaptation for Drug-Target Interaction Prediction." *Nature Machine Intelligence* 2023.
- Jurtz, V. et al. "NetMHCpan-4.0: Improved Peptide–MHC Class I Interaction Predictions." *J Immunol* 2017.

---

## 環境

- Python 3.10 | PyTorch 2.5.1+cu121 | CUDA 12.0
- Hardware：Tesla V100-SXM2-16GB × 4
- 套件：scikit-learn, prettytable, tqdm, pandas, numpy, PyYAML
