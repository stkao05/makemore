import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
from model import Transformer, Config

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

block_size = 32
emb_size = 128
head_size = 64
head_num = emb_size // head_size
layer_num = 2
dropout = 0.2

# training config
epoch = 1
eval_interval = 200
eval_size = 500
batch_size = 32


def load_txt(path):
    txt = open(path, 'r').read()
    chars = list(set(txt))
    chars.sort()

    vocab_size = len(chars)
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for i, c in enumerate(chars)}
    data = [ctoi[c] for c in txt]

    return data, ctoi, itoc, vocab_size


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


@torch.no_grad()
def estimate_loss(model, tr_ds, va_ds, batch_size):
    model.eval()
    losses = []

    for ds in [tr_ds, va_ds]:
        x, y = ds[:batch_size]
        logits = model(x)  # (B, L, C)
        B, L, C = logits.shape
        loss = F.cross_entropy(logits.view(B*L, C), y.view(B*L))
        losses.append(loss.item())

    model.train()
    return losses


def fit(model, tr_ds, va_ds, epoch, batch_size, eval_interval, eval_size):
    model.train()
    optim = torch.optim.Adam(model.parameters())
    tr_dl = DataLoader(tr_ds, batch_size)
    lossi = []
    steps = epoch * len(tr_dl)
    i = 0

    for _ in range(epoch):
        for xb, yb in tr_dl:
            optim.zero_grad()
            logits = model(xb)  # (B, L, C)

            B, L, C = logits.shape
            loss = F.cross_entropy(logits.view(B*L, C), yb.view(B*L))
            loss.backward()
            optim.step()

            i += 1
            if i % eval_interval == 0 or i == steps-1:
                tr_los, va_los = estimate_loss(model, tr_ds, va_ds, eval_size)
                lossi.append((tr_los, va_los))
                print(f"{i:5d}/{steps}: {tr_los:.4f}  {va_los:.4f}")


raw, ctoi, itoc, vocab_size = load_txt("./shakespeare.txt")
dataset = CharDataset(raw, block_size)
tr_ds, va_ds = random_split(dataset, [0.9, 0.1])

config = Config(vocab_size, block_size, emb_size,
                head_num, head_size, layer_num, ctoi, dropout)
model = Transformer(config)
model = model.to(device)
count = sum([p.numel() for p in model.parameters()])

print(f"total parameter: {count}")
print(f"device: {device}")
print(f"train size: {len(tr_ds)}")

print("\n----------training----------")
fit(model, tr_ds, va_ds, epoch, batch_size, eval_interval, eval_size)

print("\n----------sampling----------")
print(model.sample(500))

# -------- loading ------------
# torch.save({
#     'config': config,
#     'model_state_dict': model.state_dict()},
#     "transformer.pt")
# state = torch.load("transformer.pt")
# config = state["config"]
# model = Transformer(config)
# model.load_state_dict(state["model_state_dict"])

# print(model.sample(500))
