import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 8
    batch_size: int = 32
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 32
    dropout: float = 0.0
    bias: bool = True
    device: str = "cpu"


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


# THE MAIN ATTENTION HEAD
class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.query = nn.Linear(
            config.n_embd, head_size
        )  # queries --> what am i looking for

        self.key = nn.Linear(config.n_embd, head_size)  # keys --> what do i contain

        self.value = nn.Linear(
            config.n_embd, head_size
        )  # value --> for this head what does the input exposes

        self.register_buffer(
            "tril", torch.tril(torch.ones(config.block_size, config.block_size))
        )  # way of pytorch to store a non parameter variables

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)
        k = self.key(x)
        wei = q @ k.transpose(-2, -1)
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # masking with a lower triangular matrix to make the other values as negative infinity

        wei = F.softmax(wei, dim=-1)  # B, T, T
        v = self.value(x)  # B, T, C
        out = wei @ v  # B, T, C
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        out = torch.cat(
            [h(x) for h in self.heads], dim=-1
        )  # concatenating the outputs from all the heads

        out = self.proj(out)
        return out


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)  # Self attention block
        self.ffwd = MLP(config)  # Feed Forward Layer
        self.ln1 = nn.LayerNorm(
            config.n_embd, bias=config.bias
        )  # Norm Layer 1 after Self attention

        self.ln2 = nn.LayerNorm(
            config.n_embd, bias=config.bias
        )  # Norm Layer 2 after Feed forward

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # adding residual connection
        x = x + self.ffwd(self.ln2(x))  # adding residual connection
        return x


# MAIN MODEL
class GPT(nn.Module):
    def __init__(self, config: GPTConfig, vocab_size):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, config.n_embd),  # input token embedding
                wpe=nn.Embedding(
                    config.block_size, config.n_embd
                ),  # positional encoding
                drop=nn.Dropout(config.dropout),  # dropout layer for regularisation
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),  # list of attention blocks
                ln_f=nn.LayerNorm(
                    config.n_embd, bias=config.bias
                ),  # final normalisation
            )
        )
        self.lm_head = nn.Linear(
            config.n_embd, vocab_size
        )  # changing back the dimesnions from embedding to vocab size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.transformer.wte(idx)  # B, T, C
        pos_emb = self.transformer.wpe(
            torch.arange(T, device=self.config.device)
        )  # T, C
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # B, T, vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, loss = self(idx_cond)
            # here logits are again B, C, T and we want to focus only on last time that is the last char in block (since it is bigram)
            logits = logits[:, -1, :]  # becomes (B, C)
            probs = F.softmax(
                logits, dim=1
            )  # applying softmax on -ve-log-prob to get prob
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T + 1)
        return idx
