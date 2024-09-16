import torch
from model import GPT, GPTConfig
from dataclasses import dataclass
import pickle


# Train Config
@dataclass
class TrainConfig:
    max_iters: int = 10000
    eval_interval: int = 300
    eval_iters: int = 200


# Initializing Model Configuration
config = GPTConfig(
    block_size=8,  # Length of input sequence
    batch_size=32,  # Number of batches to train per iteration
    n_layer=1,  # Number of transformer layers
    n_head=2,  # Number of attention heads
    n_embd=32,  # Embedding dimensionality
    dropout=0.1,  # Dropout rate
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# loading the train config
train_config = TrainConfig()


# Getting the input training data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# building the simple char int tokenizer
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda arr: "".join([itos[i] for i in arr])

# loading and distributing the data in test and validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


# Getting random batches of block size
def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y


def estimate_loss():
    print("estimating the loss")

    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(train_config.eval_iters)
        for k in range(train_config.eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# Initialize the model
model = GPT(config, vocab_size)
model = model.to(config.device)

# using the Adam optimiser for optimising the parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# training the model - estimate loss --> calculate gradients --> adjust parameters
print("starting to train the model")
for iter in range(train_config.max_iters):
    # sample a batch
    xb, yb = get_batch("train")

    # evaluating the loss periodically
    if iter % train_config.eval_interval == 0:
        losses = estimate_loss()
        print(losses)

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Save the model weights (state) and tokenizer
torch.save(model.state_dict(), "model.pt")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump((stoi, itos), f)

print("finished training the model")
