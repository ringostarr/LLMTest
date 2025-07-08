import torch
config = {
    # Data
    "data_path": "juliusdset.txt",
    "block_size": 128,
    "batch_size": 64,

    # Model
    "vocab_size": None,  # Will be set dynamically
    "n_embed": 512,
    "n_heads": 8,
    "n_layers": 6,
    "dropout": 0.2,

    # Training
    "max_epochs": 3000,
    "eval_interval": 500,
    "eval_batches": 100,
    "learning_rate": 1e-5,
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # Checkpoint
    "checkpoint_path": "JC_model_words_checkpoint.pth"
}