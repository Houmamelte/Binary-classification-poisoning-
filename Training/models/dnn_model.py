import torch
import torch.nn as nn



class LightDNN(nn.Module):
    def __init__(self, embedding_layer, embedding_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = embedding_layer
        self.embedding.weight.requires_grad = False

        self.block1 = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        emb = self.embedding(x)
        pooled = torch.cat([emb.mean(1), emb.max(1).values], dim=1)
        x = self.block1(pooled)
        x = self.block2(x)
        return self.output(x).squeeze(1)