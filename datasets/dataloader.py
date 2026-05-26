import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

AA_TO_IDX = {aa: i + 1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYX')}
AA_TO_IDX['0'] = 0  # padding


class MHCDataset(Dataset):
    def __init__(self, df, max_peptide_len=30, max_mhc_len=34):
        self.df = df.reset_index(drop=True)
        self.max_peptide_len = max_peptide_len
        self.max_mhc_len = max_mhc_len

    def encode_seq(self, seq, max_len):
        seq = seq.upper()
        encoded = [AA_TO_IDX.get(aa, AA_TO_IDX['X']) for aa in seq]
        if len(encoded) < max_len:
            encoded += [0] * (max_len - len(encoded))
        else:
            encoded = encoded[:max_len]
        return torch.tensor(encoded, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        v_peptide = self.encode_seq(row['Peptide'],        self.max_peptide_len)
        v_mhc     = self.encode_seq(row['Pseudo_Sequence'], self.max_mhc_len)
        label     = torch.tensor(int(row['Label']), dtype=torch.float)
        return v_peptide, v_mhc, label


class MultiDataLoader:
    """Zip source and target loaders; cycle the shorter one (mirrors DrugBAN)."""
    def __init__(self, dataloaders, n_batches):
        self.dataloaders = dataloaders
        self.n_batches = n_batches

    def __iter__(self):
        self._iters = [iter(dl) for dl in self.dataloaders]
        self._step = 0
        return self

    def __next__(self):
        if self._step >= self.n_batches:
            raise StopIteration
        self._step += 1
        batches = []
        for i, it in enumerate(self._iters):
            try:
                batches.append(next(it))
            except StopIteration:           # cycle shorter loader
                self._iters[i] = iter(self.dataloaders[i])
                batches.append(next(self._iters[i]))
        return batches

    def __len__(self):
        return self.n_batches


def load_cross_hla(source_hla, target_hla, dataset_dir='datasets',
                   batch_size=64, **kwargs):
    """
    Load pre-built cross_hla/X_to_Y/ CSVs for domain adaptation.
    Returns: multi_loader (train), target_val_loader, target_test_loader
    """
    import os
    pair_dir = os.path.join(dataset_dir, 'cross_hla',
                            f'{source_hla}_to_{target_hla}')

    df_src   = pd.read_csv(os.path.join(pair_dir, 'source_train.csv'))
    df_tgt   = pd.read_csv(os.path.join(pair_dir, 'target_train.csv'))
    df_test  = pd.read_csv(os.path.join(pair_dir, 'target_test.csv'))

    # val = 20 % of target_train (random)
    df_tgt   = df_tgt.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val    = int(len(df_tgt) * 0.2)
    df_tval  = df_tgt.iloc[:n_val]
    df_tgt   = df_tgt.iloc[n_val:]

    print(f"Cross-HLA HLA-{source_hla}→HLA-{target_hla}: "
          f"source_train={len(df_src)} | target_train={len(df_tgt)} | "
          f"target_val={len(df_tval)} | target_test={len(df_test)}")

    src_loader  = DataLoader(MHCDataset(df_src),  batch_size=batch_size,
                             shuffle=True, drop_last=True)
    tgt_loader  = DataLoader(MHCDataset(df_tgt),  batch_size=batch_size,
                             shuffle=True, drop_last=True)
    val_loader  = DataLoader(MHCDataset(df_tval), batch_size=batch_size,
                             shuffle=False)
    test_loader = DataLoader(MHCDataset(df_test), batch_size=batch_size,
                             shuffle=False)

    n_batches    = max(len(src_loader), len(tgt_loader))
    multi_loader = MultiDataLoader([src_loader, tgt_loader], n_batches)
    return multi_loader, val_loader, test_loader


def load_combined_hla(hla_types, dataset_dir='datasets', split='random',
                      batch_size=64, **kwargs):
    """
    Concatenate train/val/test CSVs from multiple HLA types for pre-training.

    Args:
        hla_types: list of HLA letters, e.g. ['A', 'B'] or ['A', 'B', 'C']
        split: 'random' or 'cluster'

    Returns: train_loader, val_loader, test_loader
    """
    import os
    dfs_train, dfs_val, dfs_test = [], [], []
    for hla in hla_types:
        base = os.path.join(dataset_dir, f'HLA_{hla}', split)
        dfs_train.append(pd.read_csv(os.path.join(base, 'train.csv')))
        dfs_val.append(pd.read_csv(os.path.join(base, 'val.csv')))
        dfs_test.append(pd.read_csv(os.path.join(base, 'test.csv')))

    df_train = pd.concat(dfs_train, ignore_index=True)
    df_val   = pd.concat(dfs_val,   ignore_index=True)
    df_test  = pd.concat(dfs_test,  ignore_index=True)

    tag = '+'.join(f'HLA-{h}' for h in hla_types)
    print(f"Combined {tag} [{split}]: "
          f"train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

    train_loader = DataLoader(MHCDataset(df_train), batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(MHCDataset(df_val),   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(MHCDataset(df_test),  batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader


def load_hla_splits(hla_type, dataset_dir='datasets', split='random',
                    batch_size=64, **kwargs):
    """
    Load pre-split CSVs from datasets/HLA_<type>/<split>/.
    Mirrors DrugBAN's main.py reading pattern.

    Returns: train_loader, val_loader, test_loader
    """
    import os
    base = os.path.join(dataset_dir, f'HLA_{hla_type}', split)

    df_train = pd.read_csv(os.path.join(base, 'train.csv'))
    df_val   = pd.read_csv(os.path.join(base, 'val.csv'))
    df_test  = pd.read_csv(os.path.join(base, 'test.csv'))

    print(f"HLA-{hla_type} [{split}]: "
          f"train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")

    train_loader = DataLoader(MHCDataset(df_train), batch_size=batch_size,
                              shuffle=True,  drop_last=True)
    val_loader   = DataLoader(MHCDataset(df_val),   batch_size=batch_size,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(MHCDataset(df_test),  batch_size=batch_size,
                              shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
