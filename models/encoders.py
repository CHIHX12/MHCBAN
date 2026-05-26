import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTMEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, num_layers, dropout=0.2):
        super(BiLSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.dropout(self.embedding(x))
        # lstm_out shape: (batch, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(embedded)
        return lstm_out

class PeptideBiLSTM(BiLSTMEncoder):
    def __init__(self, vocab_size=25, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(PeptideBiLSTM, self).__init__(vocab_size, emb_dim, hidden_dim, num_layers, dropout)

class MHCBiLSTM(BiLSTMEncoder):
    def __init__(self, vocab_size=25, emb_dim=128, hidden_dim=128, num_layers=2, dropout=0.2):
        super(MHCBiLSTM, self).__init__(vocab_size, emb_dim, hidden_dim, num_layers, dropout)
