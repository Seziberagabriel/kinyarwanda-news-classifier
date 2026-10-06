"""Step 3: BiLSTM classifier trained from scratch (optionally with Word2Vec-initialised embeddings)."""
import argparse
import re
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
from utils import load_splits, compute_metrics_dict, log_experiment, save_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="bilstm")
ap.add_argument("--emb", type=int, default=200)
ap.add_argument("--hid", type=int, default=128)
ap.add_argument("--maxlen", type=int, default=256)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--vocab_size", type=int, default=30000)
ap.add_argument("--w2v", action="store_true", help="initialise embeddings with Word2Vec (needs gensim)")
args = ap.parse_args()

torch.manual_seed(42)
dev = "cuda" if torch.cuda.is_available() else "cpu"
train, val, test, labels = load_splits()

tok = lambda t: re.findall(r"\w+", str(t).lower())
counts = Counter(w for t in train.text for w in tok(t))
vocab = {w: i + 2 for i, (w, c) in enumerate(counts.most_common(args.vocab_size)) if c >= 2}  # 0=pad 1=unk


def encode(t):
    ids = [vocab.get(w, 1) for w in tok(t)][: args.maxlen]
    return ids + [0] * (args.maxlen - len(ids))


def loader(df, shuffle=False, bs=64):
    X = torch.tensor([encode(t) for t in df.text])
    y = torch.tensor(df.label.values)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=shuffle)


tr_dl, va_dl, te_dl = loader(train, True), loader(val), loader(test)


class BiLSTM(nn.Module):
    def __init__(self, V, n_cls, emb, hid):
        super().__init__()
        self.emb = nn.Embedding(V, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(2 * hid, n_cls)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        return self.fc(self.drop(out.max(dim=1).values))  # max-pool over time


model = BiLSTM(len(vocab) + 2, len(labels), args.emb, args.hid).to(dev)

if args.w2v:
    from gensim.models import Word2Vec
    print("Training Word2Vec on the training text...")
    w2v = Word2Vec([tok(t) for t in train.text], vector_size=args.emb, window=5,
                   min_count=2, workers=4, epochs=10, seed=42)
    W = model.emb.weight.data.cpu().numpy()
    hit = 0
    for w, i in vocab.items():
        if w in w2v.wv:
            W[i] = w2v.wv[w]; hit += 1
    model.emb.weight.data.copy_(torch.tensor(W))
    print(f"Initialised {hit}/{len(vocab)} embeddings from Word2Vec")

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
lossf = nn.CrossEntropyLoss()


def predict(dl):
    model.eval(); P = []
    with torch.no_grad():
        for x, _ in dl:
            P += model(x.to(dev)).argmax(1).cpu().tolist()
    return np.array(P)


best_f1, best_state = -1, None
for ep in range(1, args.epochs + 1):
    model.train(); total = 0
    for x, y in tr_dl:
        opt.zero_grad()
        loss = lossf(model(x.to(dev)), y.to(dev))
        loss.backward(); opt.step(); total += loss.item()
    m = compute_metrics_dict(val.label, predict(va_dl))
    print(f"epoch {ep} loss {total/len(tr_dl):.4f} | val acc {m['accuracy']} f1_macro {m['f1_macro']}")
    if m["f1_macro"] > best_f1:
        best_f1 = m["f1_macro"]
        best_state = {k: v.clone() for k, v in model.state_dict().items()}

model.load_state_dict(best_state)
pred = predict(te_dl)
print(classification_report(test.label, pred, target_names=labels, zero_division=0))
cfg = f"emb={args.emb}, hid={args.hid}, maxlen={args.maxlen}, epochs={args.epochs}, lr={args.lr}, w2v={args.w2v}"
log_experiment(args.name, cfg, compute_metrics_dict(test.label, pred))
save_predictions(args.name, test.text, test.label, pred, labels)
