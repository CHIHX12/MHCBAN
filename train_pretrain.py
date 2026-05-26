"""
MHCBAN 方案3 — Phase 1: Pre-training on combined HLA data.

Pre-trains MHCBAN on one or more HLA types (e.g. HLA-A+B) and saves the best
checkpoint for downstream fine-tuning.

Usage:
    # Pre-train on HLA-A only (random split)
    python -u train_pretrain.py --cfg configs/mhcban_config.yaml --hla A

    # Pre-train on HLA-A + HLA-B (random split)
    python -u train_pretrain.py --cfg configs/mhcban_config.yaml --hla A B

    # Pre-train on HLA-A + HLA-B (cluster split)
    python -u train_pretrain.py --cfg configs/mhcban_config.yaml --hla A B --split cluster

Output checkpoint: results/pretrain_HLA{AB}_random_seed42/best_model.pt
"""
import torch
import yaml
import os
import argparse
import warnings
from time import time

from models.mhcban import MHCBAN
from datasets.dataloader import load_combined_hla
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MHCBAN Pre-training on combined HLA data")
parser.add_argument('--cfg',   required=True, type=str,
                    help="path to config yaml")
parser.add_argument('--hla',   required=True, type=str, nargs='+', choices=['A', 'B', 'C'],
                    help="HLA type(s) to combine for pre-training, e.g. --hla A B")
parser.add_argument('--split', default='random', type=str, choices=['random', 'cluster'],
                    help="data split strategy")
parser.add_argument('--seed',  default=None, type=int,
                    help="override config seed")
parser.add_argument('--dataset_dir', default=None, type=str,
                    help="override dataset directory (default: datasets/)")
args = parser.parse_args()


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

    hla_tag = ''.join(sorted(args.hla))
    exp_name = f"pretrain_HLA{hla_tag}_{args.split}_seed{seed}"
    config["RESULT"]["OUTPUT_DIR"] = os.path.join(
        config["RESULT"]["OUTPUT_DIR"], exp_name)
    os.makedirs(config["RESULT"]["OUTPUT_DIR"], exist_ok=True)

    print(f"Config     : {args.cfg}")
    print(f"Pre-train  : HLA-{'+'.join(args.hla)}  |  split: {args.split}")
    print(f"Device     : {device}")
    print(f"Output dir : {config['RESULT']['OUTPUT_DIR']}\n")

    dataset_dir = args.dataset_dir if args.dataset_dir else \
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_loader, val_loader, test_loader = load_combined_hla(
        hla_types=args.hla,
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=config["SOLVER"]["BATCH_SIZE"]
    )

    model = MHCBAN(**config).to(device)
    print(model)
    with open(os.path.join(config["RESULT"]["OUTPUT_DIR"],
                           "model_architecture.txt"), "w") as f:
        f.write(str(model))

    opt = torch.optim.Adam(model.parameters(), lr=config["SOLVER"]["LR"])
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device,
                      train_loader, val_loader, test_loader,
                      **config)
    result = trainer.train()

    print(f"\nPre-training done. Checkpoint saved to: {config['RESULT']['OUTPUT_DIR']}")
    print(f"Use --checkpoint {config['RESULT']['OUTPUT_DIR']}/best_model.pt in train_finetune.py")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
