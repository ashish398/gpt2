import torch
import pickle
from model import GPT, GPTConfig

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

with open("tokenizer.pkl", "rb") as f:
    stoi, itos = pickle.load(f)

vocab_size = len(stoi)
encode = lambda s: [stoi.get(c, 0) for c in s]
decode = lambda arr: "".join([itos[i] for i in arr])

# Initialize the model
model = GPT(config, vocab_size)
model = model.to(config.device)

# Load the trained weights
model.load_state_dict(torch.load("model.pt", map_location=config.device))
model.eval()  # this puts the model in eval state which changes the way dropout layer behaves (also batchNorm but we dont have it in out model)


# generate from the model
max_new_tokens = 2000  # Maximum number of new tokens to generate (each token is 1 char)

context = torch.zeros(
    (1, 1), dtype=torch.long, device=config.device
)  # initialisingthe context with a new line char

generated = model.generate(
    context, max_new_tokens=max_new_tokens
)  # tensor of generated tokens
print(decode(generated[0].tolist()))  # [0] because there is only 1 batch
