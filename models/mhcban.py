import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoders import PeptideBiLSTM, MHCBiLSTM
from .ban import BANLayer
from torch.nn.utils.weight_norm import weight_norm

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=1):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.bn3 = nn.BatchNorm1d(out_dim)
        self.fc4 = nn.Linear(out_dim, binary)

    def forward(self, x):
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.fc4(x)
        return x

class MHCBAN(nn.Module):
    def __init__(self, **config):
        super(MHCBAN, self).__init__()
        
        # Encoders
        self.peptide_extractor = PeptideBiLSTM(
            vocab_size=config["PEPTIDE"]["VOCAB_SIZE"],
            emb_dim=config["PEPTIDE"]["EMB_DIM"],
            hidden_dim=config["PEPTIDE"]["HIDDEN_DIM"],
            num_layers=config["PEPTIDE"]["NUM_LAYERS"],
            dropout=config["PEPTIDE"]["DROPOUT"]
        )
        
        self.mhc_extractor = MHCBiLSTM(
            vocab_size=config["MHC"]["VOCAB_SIZE"],
            emb_dim=config["MHC"]["EMB_DIM"],
            hidden_dim=config["MHC"]["HIDDEN_DIM"],
            num_layers=config["MHC"]["NUM_LAYERS"],
            dropout=config["MHC"]["DROPOUT"]
        )
        
        # BAN Layer
        # hidden_dim * 2 because of BiLSTM
        v_dim = config["PEPTIDE"]["HIDDEN_DIM"] * 2
        q_dim = config["MHC"]["HIDDEN_DIM"] * 2
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        ban_heads = config["BCN"]["HEADS"]
        
        self.bcn = weight_norm(
            BANLayer(v_dim=v_dim, q_dim=q_dim, h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None)
        
        # Decoder
        self.mlp_classifier = MLPDecoder(
            in_dim=mlp_in_dim,
            hidden_dim=config["DECODER"]["HIDDEN_DIM"],
            out_dim=config["DECODER"]["OUT_DIM"],
            binary=config["DECODER"]["BINARY"]
        )

    def forward(self, v_peptide, v_mhc, mode="train"):
        # Extractor outputs: (batch, seq_len, feat_dim)
        f_peptide = self.peptide_extractor(v_peptide)
        f_mhc = self.mhc_extractor(v_mhc)
        
        # BAN Layer: bilinear attention between peptide and MHC
        f, att = self.bcn(f_peptide, f_mhc)
        
        # Classifier
        score = self.mlp_classifier(f)
        
        if mode == "train":
            return f_peptide, f_mhc, f, score
        elif mode == "eval":
            return f_peptide, f_mhc, score, att
