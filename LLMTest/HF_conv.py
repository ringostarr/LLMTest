# hf_transformer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig


class CustomTransformerConfig(PretrainedConfig):
    model_type = "custom-transformer"

    def __init__(
        self,
        vocab_size=50304,
        n_embed=512,
        n_heads=8,
        n_layers=4,
        block_size=512,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.dropout = dropout



class Head(nn.Module):
    def __init__(self, head_size, block_size, n_embed, dropout):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size, block_size, n_embed, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_size, block_size, n_embed, dropout)
            for _ in range(n_heads)
        ])
        self.proj = nn.Linear(n_heads * head_size, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_heads
        self.sa = MultiHeadAttention(config.n_heads, head_size, config.block_size, config.n_embed, config.dropout)
        self.ff = FeedForward(config.n_embed, config.dropout)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class HFCustomTransformer(PreTrainedModel):
    config_class = CustomTransformerConfig

    def __init__(self, config):
        super().__init__(config)
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)
        self.blocks = nn.Sequential(*[
            Block(config) for _ in range(config.n_layers)
        ])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)

        self.post_init()  # HF magic for initializing weights

    def forward(self, input_ids, labels=None):
        B, T = input_ids.shape
        tok = self.token_embedding_table(input_ids)
        pos = self.position_embedding_table(torch.arange(T, device=input_ids.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if labels is None:
            return {"logits": logits}
        else:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return {"loss": loss, "logits": logits}
