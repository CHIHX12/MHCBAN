"""
MHCBAN Cross-HLA Domain Adaptation training (CDAN).
Source domain: HLA-A (large, labeled).
Target domain: HLA-B or HLA-C (small, labels only for evaluation).

Usage:
    python -u train_da.py --cfg configs/mhcban_config.yaml --source A --target B
    python -u train_da.py --cfg configs/mhcban_config.yaml --source A --target C
"""
import torch
import yaml
import os
import argparse
import warnings
from time import time

from models.mhcban import MHCBAN
from datasets.dataloader import load_cross_hla
from domain_adaptator import Discriminator
from trainer import DATrainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MHCBAN Cross-HLA Domain Adaptation")
parser.add_argument('--cfg',    required=True, type=str, help="path to config yaml")
parser.add_argument('--source', required=True, type=str, choices=['A', 'B', 'C'],
                    help="source HLA type (labeled, large)")
parser.add_argument('--target', required=True, type=str, choices=['A', 'B', 'C'],
                    help="target HLA type (unlabeled for DA, small)")
parser.add_argument('--alpha',  default=1.0, type=float,
                    help="GRL gradient reversal strength")
parser.add_argument('--seed',   default=None, type=int,
                    help="override config seed (for stability testing)")
parser.add_argument('--dataset_dir', default=None, type=str,
                    help="override dataset directory (default: datasets/)")
args = parser.parse_args()


def main():
    assert args.source != args.target, "Source and target must be different HLA types."
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

    config["RESULT"]["OUTPUT_DIR"] = os.path.join(
        config["RESULT"]["OUTPUT_DIR"],
        f"DA_HLA{args.source}_to_HLA{args.target}_seed{seed}")
    os.makedirs(config["RESULT"]["OUTPUT_DIR"], exist_ok=True)

    print(f"Config      : {args.cfg}")
    print(f"DA direction: HLA-{args.source} (source) → HLA-{args.target} (target)")
    print(f"Device      : {device}")
    print(f"Output dir  : {config['RESULT']['OUTPUT_DIR']}\n")

    # ── Data ──────────────────────────────────────────────────────────
    dataset_dir = args.dataset_dir if args.dataset_dir else \
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    multi_loader, val_loader, test_loader = load_cross_hla(
        source_hla=args.source,
        target_hla=args.target,
        dataset_dir=dataset_dir,
        batch_size=config["SOLVER"]["BATCH_SIZE"]
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = MHCBAN(**config).to(device)
    print(model)
    with open(os.path.join(config["RESULT"]["OUTPUT_DIR"],
                           "model_architecture.txt"), "w") as f:
        f.write(str(model))

    # ── Discriminator ─────────────────────────────────────────────────
    # Input = BAN output dim (IN_DIM), same as DrugBAN
    discriminator = Discriminator(
        input_size=config["DECODER"]["IN_DIM"],
        n_class=config["DECODER"]["BINARY"]
    ).to(device)

    # ── Optimisers ────────────────────────────────────────────────────
    opt    = torch.optim.Adam(model.parameters(),
                               lr=config["SOLVER"]["LR"])
    opt_da = torch.optim.Adam(discriminator.parameters(),
                               lr=config["DA"]["DA_LR"])

    torch.backends.cudnn.benchmark = True

    # ── Train ─────────────────────────────────────────────────────────
    trainer = DATrainer(
        model, opt, device,
        multi_loader, val_loader, test_loader,
        discriminator=discriminator,
        optim_da=opt_da,
        alpha=args.alpha,
        **config
    )
    result = trainer.train()

    print(f"\nDirectory for saving result: {config['RESULT']['OUTPUT_DIR']}")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
