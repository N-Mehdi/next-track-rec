"""SASRec encoder (causal self-attention over the listening sequence).

Pre-norm transformer blocks, learned positions, item embeddings shared between input
and output. Trained with a sampled softmax over uniformly drawn negatives shared by
the batch.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class Block(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.dropout_p = dropout
        self.norm_attention = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        batch, length, dim = x.shape
        qkv = self.qkv(self.norm_attention(x)).view(batch, length, 3, self.heads, dim // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        p = self.dropout_p if self.training else 0.0
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        x = x + self.dropout(self.out(attended))
        return x + self.dropout(self.ff(self.norm_ff(x)))


class SASRec(nn.Module):
    def __init__(self, num_items, max_len, dim, layers, heads, dropout):
        super().__init__()
        self.num_items = num_items
        self.dim = dim
        self.items = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.positions = nn.Embedding(max_len, dim)
        blocks = []
        for _ in range(layers):
            blocks.append(Block(dim, heads, dropout))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.items.weight[0].zero_()

    def encode(self, inputs):
        """Hidden state at every position, shape (batch, length, dim)."""
        batch, length = inputs.shape
        device = inputs.device
        padding = inputs == 0
        causal = torch.ones(length, length, dtype=torch.bool, device=device).tril()
        diagonal = torch.eye(length, dtype=torch.bool, device=device)
        # A query never sees padded keys, but always sees itself to avoid empty rows.
        allowed = (causal[None] & ~padding[:, None, :]) | diagonal[None]
        positions = torch.arange(length, device=device)
        x = self.items(inputs) * math.sqrt(self.dim) + self.positions(positions)[None]
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, allowed[:, None])
        return self.norm(x)

    def user_vectors(self, inputs):
        """Representation after the latest event (last position, inputs are left padded)."""
        return self.encode(inputs)[:, -1]

    def loss(self, inputs, targets, num_negatives):
        hidden = self.encode(inputs)
        valid = (inputs > 0) & (targets > 0)
        queries = hidden[valid]
        positives = targets[valid]
        negatives = torch.randint(1, self.num_items + 1, (num_negatives,), device=inputs.device)
        positive_logits = (queries * self.items(positives)).sum(dim=-1, keepdim=True)
        negative_logits = queries @ self.items(negatives).T
        logits = torch.cat([positive_logits, negative_logits], dim=1).float()
        collision = negatives[None, :] == positives[:, None]
        logits[:, 1:] = logits[:, 1:].masked_fill(collision, float("-inf"))
        labels = torch.zeros(len(positives), dtype=torch.long, device=inputs.device)
        return F.cross_entropy(logits, labels)
