"""
MHCBAN 方案3 — Phase 2: Fine-tuning on a target HLA type.

Loads a pre-trained checkpoint (from train_pretrain.py), applies a freeze strategy,
then fine-tunes on the specified target HLA split.

Freeze strategies:
  encoder      — freeze peptide_extractor + mhc_extractor (BiLSTM only)
                 Best for tiny target datasets (e.g. HLA-C, ~1k samples).
  all_but_mlp  — freeze encoders + BAN; only train MLPDecoder
                 Most aggressive; use only when target is extremely small.
  none         — all layers trainable with the specified LR (default 1e-4)
                 Best for medium target datasets (e.g. HLA-B, ~29k samples).

Usage:
    # Fine-tune HLA-C cluster (frozen encoder, from A+B pre-trained model)
    python -u train_finetune.py \\
        --cfg configs/mhcban_config.yaml \\
        --checkpoint results/pretrain_HLAAB_random_seed42/best_model_epoch_*.pth \\
        --hla C --split cluster --freeze encoder --lr 1e-4 --epochs 50

    # Fine-tune HLA-B cluster (all layers, lower LR, from A pre-trained model)
    python -u train_finetune.py \\
        --cfg configs/mhcban_config.yaml \\
        --checkpoint results/pretrain_HLAA_random_seed42/best_model_epoch_*.pth \\
        --hla B --split cluster --freeze none --lr 1e-4 --epochs 50
"""
import torch
import yaml
import os
import glob
import argparse
import warnings
from time import time

from models.mhcban import MHCBAN
from datasets.dataloader import load_hla_splits
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MHCBAN Fine-tuning from pre-trained checkpoint")
parser.add_argument('--cfg',        required=True,  type=str,
                    help="path to config yaml")
parser.add_argument('--checkpoint', required=True,  type=str,
                    help="path to pre-trained .pth file, or directory containing best_model_epoch_*.pth")
parser.add_argument('--hla',        required=True,  type=str, choices=['A', 'B', 'C'],
                    help="target HLA type for fine-tuning")
parser.add_argument('--split',      default='cluster', type=str,
                    choices=['random', 'cluster'],
                    help="data split strategy (default: cluster for cold-start evaluation)")
parser.add_argument('--freeze',     default='encoder', type=str,
                    choices=['encoder', 'all_but_mlp', 'none'],
                    help="which layers to freeze (default: encoder)")
parser.add_argument('--lr',         default=1e-4,   type=float,
                    help="fine-tuning learning rate (default: 1e-4)")
parser.add_argument('--epochs',     default=50,     type=int,
                    help="number of fine-tuning epochs (default: 50)")
parser.add_argument('--seed',       default=None,   type=int,
                    help="override config seed")
parser.add_argument('--dataset_dir', default=None, type=str,
                    help="override dataset directory (default: datasets/)")
args = parser.parse_args()

ENCODER_MODULES = ['peptide_extractor', 'mhc_extractor']
BAN_MODULES     = ['bcn']
MLP_MODULES     = ['mlp_classifier']


def resolve_checkpoint(path_or_dir: str) -> str:
    """Accept either a direct .pth path or a result directory."""
    if os.path.isfile(path_or_dir):
        return path_or_dir
    # Try glob inside directory
    pattern = os.path.join(path_or_dir, 'best_model_epoch_*.pth')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No best_model_epoch_*.pth found in: {path_or_dir}")
    if len(matches) > 1:
        print(f"[warn] Multiple checkpoints found, using latest: {matches[-1]}")
    return matches[-1]


def apply_freeze(model, strategy: str):
    """Freeze parameters according to strategy and report counts."""
    if strategy == 'none':
        for p in model.parameters():
            p.requires_grad = True
        return

    freeze_names = ENCODER_MODULES if strategy == 'encoder' else (
        ENCODER_MODULES + BAN_MODULES)

    frozen = trainable = 0
    for name, param in model.named_parameters():
        top = name.split('.')[0]
        if top in freeze_names:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    print(f"Freeze strategy '{strategy}': "
          f"frozen={frozen:,} params | trainable={trainable:,} params")


def main():
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore")

    with open(args.cfg, 'r') as f:
        config = yaml.safe_load(f)

    if args.seed is not None:
        config["SOLVER"]["SEED"] = args.seed

    seed = config["SOLVER"]["SEED"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Override epochs and LR from CLI args
    config["SOLVER"]["EPOCHS"] = args.epochs
    config["SOLVER"]["LR"]     = args.lr

    exp_name = f"finetune_HLA{args.hla}_{args.split}_freeze{args.freeze}_seed{seed}"
    config["RESULT"]["OUTPUT_DIR"] = os.path.join(
        config["RESULT"]["OUTPUT_DIR"], exp_name)
    os.makedirs(config["RESULT"]["OUTPUT_DIR"], exist_ok=True)

    ckpt_path = resolve_checkpoint(args.checkpoint)
    print(f"Config       : {args.cfg}")
    print(f"Checkpoint   : {ckpt_path}")
    print(f"Target HLA   : HLA-{args.hla}  |  split: {args.split}")
    print(f"Freeze       : {args.freeze}")
    print(f"LR / Epochs  : {args.lr} / {args.epochs}")
    print(f"Device       : {device}")
    print(f"Output dir   : {config['RESULT']['OUTPUT_DIR']}\n")

    # ── Model ─────────────────────────────────────────────────────────
    model = MHCBAN(**config).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    print(f"Loaded pre-trained weights from: {ckpt_path}")

    apply_freeze(model, args.freeze)

    with open(os.path.join(config["RESULT"]["OUTPUT_DIR"],
                           "model_architecture.txt"), "w") as f:
        f.write(str(model))

    # ── Data ──────────────────────────────────────────────────────────
    dataset_dir = args.dataset_dir if args.dataset_dir else \
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_loader, val_loader, test_loader = load_hla_splits(
        hla_type=args.hla,
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=config["SOLVER"]["BATCH_SIZE"]
    )

    # ── Optimiser: only update trainable params ────────────────────────
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable_params, lr=args.lr)
    torch.backends.cudnn.benchmark = True

    # ── Train ──────────────────────────────────────────────────────────
    trainer = Trainer(model, opt, device,
                      train_loader, val_loader, test_loader,
                      **config)
    result = trainer.train()

    print(f"\nFine-tuning done. Results saved to: {config['RESULT']['OUTPUT_DIR']}")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
