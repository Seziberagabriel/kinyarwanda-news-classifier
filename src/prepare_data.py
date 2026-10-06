"""Step 1: download KINNEWS, clean it, create train/val/test CSVs and dataset statistics."""
import json
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.model_selection import train_test_split

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

print("Loading KINNEWS (cleaned config)...")
ds = load_dataset("andreniyongabo/kinnews_kirnews", "kinnews_cleaned", trust_remote_code=True)
print(ds)

feat = ds["train"].features["label"]
labels = list(feat.names) if hasattr(feat, "names") else sorted(set(ds["train"]["label"]))
print("Labels:", labels)


def to_df(split):
    d = split.to_pandas()
    d["text"] = (d["title"].fillna("") + ". " + d["content"].fillna("")).str.strip()
    d["text"] = d["text"].str.replace(r"\s+", " ", regex=True)
    d["label_name"] = d["label"].map(lambda i: labels[i])
    return d[["text", "label", "label_name"]]


train_full, test = to_df(ds["train"]), to_df(ds["test"])
print(f"Raw sizes -> train: {len(train_full)}, test: {len(test)}")

# --- cleaning ---------------------------------------------------------------
n0 = len(train_full)
train_full = train_full[train_full.text.str.len() > 20]
train_full = train_full.drop_duplicates(subset="text")
print(f"Removed {n0 - len(train_full)} empty/duplicate training rows")

n0 = len(test)
test = test[test.text.str.len() > 20].drop_duplicates(subset="text")
leak = test.text.isin(set(train_full.text))
test = test[~leak]
print(f"Removed {n0 - len(test)} empty/duplicate/leaked test rows")

# --- validation split ---------------------------------------------------------
if "validation" in ds:
    val = to_df(ds["validation"])
    train = train_full
else:
    train, val = train_test_split(
        train_full, test_size=0.1, random_state=42, stratify=train_full.label
    )

for name, d in [("train", train), ("val", val), ("test", test)]:
    d.to_csv(f"data/{name}.csv", index=False)
with open("data/labels.json", "w", encoding="utf-8") as f:
    json.dump(labels, f, ensure_ascii=False)

# --- statistics for the report -----------------------------------------------
dist = pd.DataFrame(
    {n: d.label_name.value_counts() for n, d in [("train", train), ("val", val), ("test", test)]}
).fillna(0).astype(int)
dist.to_csv("results/class_distribution.csv")
print("\nClass distribution:\n", dist)

lengths = train.text.str.split().str.len()
print("\nWords per document (train):\n", lengths.describe())
print("Sizes -> train:", len(train), "val:", len(val), "test:", len(test))

dist["train"].sort_values().plot(kind="barh", figsize=(7, 5), title="Training class distribution")
plt.tight_layout(); plt.savefig("results/class_distribution.png", dpi=150); plt.close()

lengths.clip(upper=1500).plot(kind="hist", bins=50, figsize=(7, 4), title="Words per document (train)")
plt.tight_layout(); plt.savefig("results/length_distribution.png", dpi=150); plt.close()
print("Done. Files written to data/ and results/.")
