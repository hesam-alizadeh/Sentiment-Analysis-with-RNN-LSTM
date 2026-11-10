import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=128, num_layers=1, dropout=0.3, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.lstm(packed)
        last_hidden = hidden[-1]
        logits = self.fc(self.dropout(last_hidden))
        return logits
