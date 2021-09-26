import torch
import torch.nn as nn
import torch.nn.functional as F
import json

with open('data/dictionary_char.json') as json_file:
    data = json.load(json_file)
n_class = 77
tokens_size = len(data)

class Classifier(nn.Module):
    def __init__(self, n_tokens=tokens_size, emb_size=64, hid_size=128, n_class=n_class):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=n_tokens, embedding_dim=emb_size)
        self.cnn1 = nn.Sequential(
            nn.Conv1d(emb_size, hid_size, kernel_size=3),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hid_size)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=3),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hid_size)
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=3),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.BatchNorm1d(hid_size)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(hid_size, hid_size, kernel_size=3),
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
            nn.BatchNorm1d(hid_size)
        )
        self.linear = nn.Sequential(
            nn.Linear(hid_size, 4 * hid_size),
            nn.ReLU(),
            nn.Linear(hid_size * 4, hid_size * 4),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(hid_size * 4, hid_size * 4),
            nn.ReLU(),
            nn.Linear(4 * hid_size, n_class)
        )

    def __call__(self, input_ix):
        x = self.emb(input_ix).transpose(1, 2)
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x).transpose(1, 2)
        return self.linear(x).squeeze()
