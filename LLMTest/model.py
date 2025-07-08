import torch
import torch.nn as nn
import torch.nn.functional as F


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
        weights = q @ k.transpose(-2, -1) * (C ** -0.5)
        weights = weights.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(weights, dim=-1)
        weights = self.dropout(weights)
        v = self.value(x)
        return weights @ v


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
    def __init__(self, n_embed, n_heads, block_size, dropout):
        super().__init__()
        head_size = n_embed // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, block_size, n_embed, dropout)
        self.ff = FeedForward(n_embed, dropout)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class TransformerLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config['vocab_size'], config['n_embed'])
        self.position_embedding_table = nn.Embedding(config['block_size'], config['n_embed'])
        self.blocks = nn.Sequential(*[
            Block(config['n_embed'], config['n_heads'], config['block_size'], config['dropout'])
            for _ in range(config['n_layers'])
        ])
        self.ln_f = nn.LayerNorm(config['n_embed'])
        self.lm_head = nn.Linear(config['n_embed'], config['vocab_size'])

    def forward(self, idx, targets=None):
        B, T = idx.shape
        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx