"""Diagnose WHY so many KINNEWS rows are removed during cleaning (use the numbers in your report).
Usage: python src/data_quality.py
"""
from datasets import load_dataset

ds = load_dataset("andreniyongabo/kinnews_kirnews", "kinnews_cleaned", trust_remote_code=True)
names = ds["train"].features["label"].names
lines = []


def say(s=""):
    print(s); lines.append(str(s))


for split in ["train", "test"]:
    d = ds[split].to_pandas()
    d["text"] = (d.title.fillna("") + ". " + d.content.fillna("")).str.strip()
    say(f"=== {split}: {len(d)} rows ===")
    say(f"empty title:   {(d.title.fillna('').str.strip() == '').sum()}")
    say(f"empty content: {(d.content.fillna('').str.strip() == '').sum()}")
    say(f"text shorter than 20 chars: {(d.text.str.len() <= 20).sum()}")
    say(f"duplicate full texts (extra copies): {d.text.duplicated().sum()}")
    say(f"duplicate titles (extra copies):     {d.title.duplicated().sum()}")
    say(f"duplicate contents (extra copies):   {d.content.duplicated().sum()}")
    g = d.groupby("text").label.nunique()
    say(f"texts that appear with CONFLICTING labels: {(g > 1).sum()}")
    say()

tr = {((a or "") + ". " + (b or "")).strip() for a, b in zip(ds["train"]["title"], ds["train"]["content"])}
te = [((a or "") + ". " + (b or "")).strip() for a, b in zip(ds["test"]["title"], ds["test"]["content"])]
say(f"test rows whose exact text also appears in train: {sum(t in tr for t in te)}")

import os
os.makedirs("results", exist_ok=True)
open("results/data_quality.txt", "w").write("\n".join(lines))
print("saved results/data_quality.txt")
