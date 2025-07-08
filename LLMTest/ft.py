#Idea is to fine-tune and change the "thou" to "you" and maybe "arth" to are
#classic transformer , big thanks to andrej karthay
# - Akshay
# Jan 2024
import torch
print(torch.cuda.is_available())  # Should return True

with open('juliusdset.txt', 'r', encoding='utf-8') as f:
    text = f.read()
words = sorted(list(set(text.split())))
vocab_size = len(words)
wtoi = {w: i for i, w in enumerate(words)}
itow = {i: w for i, w in enumerate(words)}

encode = lambda s: torch.tensor([wtoi[w] for w in s.split()], dtype=torch.long)
decode = lambda idxs: " ".join([itow[i] for i in idxs])

print(len(wtoi))
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from character to index and vice versa
ctoi = {c: i for i, c in enumerate(chars)}
itoc = {i: c for i, c in enumerate(chars)}
# Define encode and decode functions
encode = lambda x: torch.tensor([ctoi[c] for c in x], dtype=torch.long)  # encoder: char to index
decode = lambda x: ''.join([itoc[i] for i in x])  # decoder: index to char

print(len(ctoi))