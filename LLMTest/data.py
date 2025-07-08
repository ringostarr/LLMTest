import torch
from pathlib import Path

class CharDataset:
    def __init__(self, file_path, block_size, train_split=0.9):
        self.block_size = block_size

        # Load text
        self.text = Path(file_path).read_text(encoding='utf-8')
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

        # Create character <-> index mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda idxs: ''.join([self.itos[i] for i in idxs])

        # Encode entire text
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        n = int(len(data) * train_split)
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, batch_size, device):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)

class WordDataset:
    def __init__(self, file_path, block_size, train_split=0.7):
        self.block_size = block_size

        # Load text
        text = Path(file_path).read_text(encoding='utf-8')

        # Split into words
        words = text.split()
        vocab = sorted(set(words))
        self.vocab_size = len(vocab)

        # Create word <-> index mappings
        self.wtoi = {w: i for i, w in enumerate(vocab)}
        self.itow = {i: w for w, i in self.wtoi.items()}
        self.encode = lambda s: [self.wtoi[w] for w in s.split()]
        self.decode = lambda idxs: " ".join([self.itow[i] for i in idxs])

        # Encode entire text as a tensor of word indices
        data = torch.tensor(self.encode(text), dtype=torch.long)
        n = int(len(data) * train_split)
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, batch_size, device):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(device), y.to(device)