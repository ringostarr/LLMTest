import torch
from data import CharDataset
from data import WordDataset
from model import TransformerLM
from config import config

torch.manual_seed(23)

# Load data
dataset = WordDataset(config["data_path"], config["block_size"])
config["vocab_size"] = dataset.vocab_size

# Initialize model
model = TransformerLM(config).to(config["device"])
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

best_val_loss = float('inf')
losses_train, losses_val = [], []
print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(config["eval_batches"])
        for k in range(config["eval_batches"]):
            X, Y = dataset.get_batch(split, config["batch_size"], config["device"])
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
load = False
if load == True:
    model = TransformerLM(config).to(config['device'])
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
    checkpoint = torch.load(config['checkpoint_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config['device'])
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=1000)
    generated_text = dataset.decode(generated[0].tolist())
    print(generated_text)

    with open("output.txt", "w", encoding='utf-8') as f:
        f.write(generated_text)
    exit()

# Training loop
for epoch in range(config["max_epochs"]):
    x, y = dataset.get_batch("train", config["batch_size"], config["device"])
    _, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"epoch {epoch}, loss: {loss.item():.3f}")
    if epoch % config["eval_interval"] == 0:
        losses = estimate_loss()
        print(f"epoch {epoch}, train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")
        losses_val.append(losses["val"])
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            print("Saving checkpoint...")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss
            }, config["checkpoint_path"])

# Text generation
context = torch.zeros((1, 1), dtype=torch.long, device=config["device"])
generated = model.generate(context, max_new_tokens=1000)
decoded = dataset.decode(generated[0].tolist())
open('output.txt', 'w').write(dataset.decode(model.generate(context, max_new_tokens=4000)[0].tolist()))


print('')
print(decoded)

# Save output
