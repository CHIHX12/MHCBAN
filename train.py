import torch
import yaml
import os
import argparse
import warnings
from time import time

from models.mhcban import MHCBAN
from datasets.dataloader import load_hla_splits
from trainer import Trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description="MHCBAN for MHC-peptide binding prediction")
parser.add_argument('--cfg',   required=True, type=str,
                    help="path to config yaml")
parser.add_argument('--hla',   required=True, type=str, choices=['A', 'B', 'C'],
                    help="HLA cluster (A / B / C)")
parser.add_argument('--split', default='random', type=str,
                    choices=['random', 'cluster'],
                    help="split strategy: random (sample-level) | cluster (allele cold-start)")
parser.add_argument('--seed', default=None, type=int,
                    help="override config seed (for stability testing)")
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

    config["RESULT"]["OUTPUT_DIR"] = os.path.join(
        config["RESULT"]["OUTPUT_DIR"], f"HLA_{args.hla}_{args.split}_seed{seed}")
    os.makedirs(config["RESULT"]["OUTPUT_DIR"], exist_ok=True)

    print(f"Config     : {args.cfg}")
    print(f"HLA cluster: HLA-{args.hla}  |  split: {args.split}")
    print(f"Device     : {device}")
    print(f"Output dir : {config['RESULT']['OUTPUT_DIR']}\n")

    # ── Data  (reads from datasets/HLA_<X>/random/) ───────────────────
    dataset_dir = args.dataset_dir if args.dataset_dir else \
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets')
    train_loader, val_loader, test_loader = load_hla_splits(
        hla_type=args.hla,
        dataset_dir=dataset_dir,
        split=args.split,
        batch_size=config["SOLVER"]["BATCH_SIZE"]
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = MHCBAN(**config).to(device)
    print(model)
    with open(os.path.join(config["RESULT"]["OUTPUT_DIR"],
                           "model_architecture.txt"), "w") as f:
        f.write(str(model))

    # ── Optimiser ─────────────────────────────────────────────────────
    opt = torch.optim.Adam(model.parameters(), lr=config["SOLVER"]["LR"])
    torch.backends.cudnn.benchmark = True

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(model, opt, device,
                      train_loader, val_loader, test_loader,
                      **config)
    result = trainer.train()

    print(f"\nDirectory for saving result: {config['RESULT']['OUTPUT_DIR']}")
    return result


if __name__ == '__main__':
    s = time()
    result = main()
    e = time()
    print(f"\nTotal running time: {round(e - s, 2)}s")
