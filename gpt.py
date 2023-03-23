#!/usr/bin/env python
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader

device = torch.device("cpu")


@dataclass
class Config:
    vocab_size: int
    block_size: int
    emb_size: int
    head_num: int
    head_size: int
    layer_num: int


class MultiHeadAttension(nn.Module):

    def __init__(self, c: Config):
        super().__init__()
        assert c.emb_size / c.head_size == c.head_num

        self.head_size = c.head_size
        self.head_num = c.head_num
        self.attn = nn.Linear(
            c.emb_size, 3 * c.head_num * c.head_size, bias=False)
        self.ffn = nn.Linear(c.head_num * c.head_size, c.emb_size, bias=False)

    # x: (B, L, C)
    # return: (B, L, C)
    def forward(self, x):
        B, L, C = x.shape

        z = self.attn(x)  # (B, L, 3 * hn * hs)
        k, q, v = torch.split(
            z, self.head_num * self.head_size, dim=2)  # (B, L, hn * hs)

        k = k.view(B, L, self.head_num, self.head_size).permute(
            0, 2, 1, 3)  # (B, hn, L, hs)
        q = q.view(B, L, self.head_num, self.head_size).permute(0, 2, 1, 3)
        v = v.view(B, L, self.head_num, self.head_size).permute(0, 2, 1, 3)

        q = q.permute(0, 1, 3, 2)  # (B, hn, hs, L)
        attn = (k @ q) / self.head_size**0.5  # (B, hn, L, L)
        mask = torch.tril(torch.ones(L, L)) == 0
        mask = mask.to(device)
        attn = attn.masked_fill(mask, -float('inf'))  # (B, hn, L, L)
        attn = F.softmax(attn, dim=3)

        y = attn @ v  # (B, hn, L, hs)
        y = y.permute(0, 2, 1, 3)  # (B, L, hn, hs)
        y = y.contiguous().view(B, L, -1)  # (B, L, hn * hs)
        y = self.ffn(y)  # (B, L, C)

        return y


class FeedForward(nn.Module):

    def __init__(self, emb_size):
        super().__init__()
        self.linear1 = nn.Linear(emb_size, 2 * emb_size)
        self.linear2 = nn.Linear(2 * emb_size, emb_size)

    # (B, L, C)
    def forward(self, x):
        y = self.linear1(x)
        y = torch.relu(y)
        y = self.linear2(y)

        return y


class Block(nn.Module):

    def __init__(self, c: Config):
        super().__init__()

        assert c.emb_size % c.head_size == 0
        assert c.emb_size / c.head_size == c.head_num

        self.mha = MultiHeadAttension(c)
        self.lnorm1 = nn.LayerNorm(c.emb_size)
        self.lnorm2 = nn.LayerNorm(c.emb_size)
        self.ffn = FeedForward(c.emb_size)

    # x: (B, L, emb)
    def forward(self, x):
        y = self.mha(x) + x
        y = self.lnorm1(y)
        y = self.ffn(y) + y
        y = self.lnorm2(y)
        return y


class Transformer(nn.Module):

    def __init__(self, c: Config):
        super().__init__()
        self.embed = nn.Embedding(c.vocab_size, c.emb_size)
        self.blocks = nn.Sequential(
            *[Block(c) for _ in range(c.layer_num)]
        )
        self.proj = nn.Linear(c.emb_size, c.vocab_size)

    # return (L, C)
    def pos_encoding(self, x):
        B, L, C = x.shape
        pos = torch.arange(0, L).view(-1, 1)  # (L, 1)
        div = 2 * torch.arange(0, C) / C  # (C)
        div = torch.pow(10000, div)  # (C)
        e = pos / div
        pe = torch.zeros(L, C)
        pe[:, 0::2] = torch.sin(e[:, 0::2])
        pe[:, 1::2] = torch.cos(e[:, 1::2])

        pe = pe.to(device)
        return pe

    # (B, L) -> (B, L, C)
    def forward(self, x):
        y = self.embed(x)  # (B, L, emb)
        y = y + self.pos_encoding(y)  # (B, L, emb)
        y = self.blocks(y)  # (B, L, emb)
        y = self.proj(y)  # (B, L, vocab)

        return y


class CharDataset(Dataset):
    # data: a list of integer token (L)
    def __init__(self, data, block_size):
        B = len(data) // block_size
        self.X = torch.tensor(data[:B*block_size]).view(B, block_size)
        self.Y = torch.tensor(data[1:B*block_size+1]).view(B, block_size)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_txt(path):
    txt = open(path, 'r').read()
    chars = list(set(txt))
    chars.sort()

    vocab_size = len(chars)
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}
    data = [ctoi[c] for c in txt]

    return data, ctoi, itoc, vocab_size


@torch.no_grad()
def estimate_loss(model, tr_ds, va_ds, size, batch_size):
    model.eval()
    losses = []

    n = size // batch_size

    for ds in [tr_ds, va_ds]:
        x, y = ds[:batch_size]
        logits = model(x)  # (B, L, C)
        B, L, C = logits.shape
        loss = F.cross_entropy(logits.view(B*L, C), y.view(B*L))
        losses.append(loss.item())

    model.train()
    return losses


@torch.no_grad()
def sample(model, config, itoc):
    model.eval()

    max_len = 500
    tks = [0] * config.block_size

    for i in range(max_len):
        ctx = torch.tensor(tks[i:i+config.block_size])  # (L)
        ctx = ctx.view(1, -1)  # (B, L)

        logits = model(ctx)  # (B, L, C)
        probs = F.softmax(logits, dim=2)  # (B, L, C)
        # (C), # the last in the sequence is the newly generated
        probs = probs[0, -1, :]
        yi = torch.multinomial(probs, 1)
        tks.append(yi.item())

    tks = tks[config.block_size:]
    chars = [itoc[t] for t in tks]
    model.train()
    return "".join(chars)


def fit(model, tr_ds, va_ds, epoch, batch_size, eval_size):
    model.train()
    optim = torch.optim.Adam(model.parameters())
    tr_dl = DataLoader(tr_ds, batch_size)
    va_dl = DataLoader(va_ds, batch_size)
    lossi = []

    for i in range(epoch):
        for xb, yb in tr_dl:
            optim.zero_grad()
            logits = model(xb)  # (B, L, C)

            B, L, C = logits.shape
            loss = F.cross_entropy(logits.view(B*L, C), yb.view(B*L))
            loss.backward()
            optim.step()

        # tr_los, va_los = estimate_loss(model, tr_ds, va_ds, eval_size)
        # lossi.append((tr_los, va_los))
        # print(f"{i:5d}/{epoch}: {tr_los:.4f}  {va_los:.4f}")


if __name__ == "__main__":
    print("hi")

    epoch = 5000
    eval_size = 500
    block_size = 256

    raw, ctoi, itoc, vocab_size = load_txt("shakespeare.txt")
    dataset = CharDataset(raw, block_size)
    tr_ds, va_ds = random_split(dataset, [0.9, 0.1])

    config = Config(vocab_size, block_size, emb_size=256, head_num=4, head_size=64, layer_num=2)
    model = Transformer(config)
    model = model.to(device)
    count = sum([p.numel() for p in model.parameters()])
    print(f"total parameter: {count}")

    # fit(model, tr_ds, va_ds, epoch, eval_size)
    # tr_loss, va_loss = estimate_loss(model, tr_data, va_data, 10000)
    # print(f"train: {tr_loss:.4f}  valid: {va_loss:.4f}")
