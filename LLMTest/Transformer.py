#classic transformer , big thanks to andrej karthay
# - Akshay
# Jan 2024
import time

import torch
import torch.nn as nn
from torch.nn import functional as F


import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))

batch_size = 64
block_size = 256
max_epochs = 500
eval_interval = 500
learning_rate = 3e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eval_epochs = 100  # how many batches to evaluate for
n_embed = 384  # size of the token embedding
n_heads = 6  # number of attention heads
n_layers = 4  # number of layers
dropout = 0.2  # dropout rate
print(device)

torch.manual_seed(23)

with open('juliusdset.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all unique characters in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from character to index and vice versa
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
# Define encode and decode functions
encode = lambda x: torch.tensor([ctoi[c] for c in x], dtype=torch.long)  # encoder: char to index
decode = lambda x: ''.join([itoc[i] for i in x])  # decoder: index to char

# train and test split
data = torch.tensor(encode(text), dtype=torch.long).detach()
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))  # randomly select starting indices
    x = torch.stack([data[i: i + block_size] for i in ix])  # input sequence
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])  # target sequence
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()  # set the model to evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            x, y = get_batch(split)
            probs, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """one head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute the attention weights/scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C ** -0.5  # (B,T,C) @ (B,C,T) = (B,T,T)
        # mask out the future tokens
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        # softmax to get the attention weights
        wei = wei.softmax(dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # compute the weighted sum of the values
        v = self.value(x)  # (B,T,C)
        out = wei @ v  # (B,T,T) @ (B,T,C) = (B,T,C)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """A simlpe linear layer followed by nonolinearity"""

    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.ReLU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class BLM(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)  # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B, T) tensors of integers
        # B is the batch size, T is the sequence length
        token_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = token_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        probs = self.lm_head(x)  # (B, T, vocab_size)

        if (targets is None):
            # if targets is None, we are in inference mode
            # return the probs
            return probs, None
        else:
            # if targets is not None, we are in training mode
            # compute the loss
            B, T, C = probs.shape
            probs = probs.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(probs, targets)
            return probs, loss

    def generate(self, input, max_new_tokens):
        res_string = input
        for _ in range(max_new_tokens):
            idx_cond = res_string[:, -block_size:]
            probs, loss = self(idx_cond)
            # focus only on last time step
            probs = probs[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(probs, dim=-1)  # (B, C)

            next_char = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append the new tokens to the end of the sequence
            res_string = torch.cat([res_string, next_char], dim=1)  # (B, T+1)
        return res_string

checkpoint_path = 'JC_model_checkpoint.pth'
model = BLM()
m = model.to(device)

# print number of parameters in model
print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
# create pytorch optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_val_loss = float('inf')

# training loop
for epoch in range(max_epochs):
    x, y = get_batch('train')
    probs, loss = model(x, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if epoch % 10 == 0:
        print(f'epoch {epoch}, loss {loss.item():.3f}')
    # evaluate the model
    if epoch % eval_interval == 0:
        losses = estimate_loss()
        print(f'epoch {epoch}, train loss {losses["train"]:.3f}, val loss {losses["val"]:.3f}')
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            print(f"Validation loss improved to {best_val_loss:.3f}. Saving new best checkpoint...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': m.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                # 'scheduler_state_dict': scheduler.state_dict(), # Include if you use a scheduler
            }, checkpoint_path)
        print('')

# generate some text from model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=1000)[0].tolist()))
open('output.txt', 'w').write(decode(m.generate(context, max_new_tokens=4000)[0].tolist()))


print('')