"""Step 5: error analysis for one model's predictions.
Usage: python src/error_analysis.py --name afriberta_base
"""
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="afriberta_base")
ap.add_argument("--n_samples", type=int, default=30)
args = ap.parse_args()

df = pd.read_csv(f"results/preds_{args.name}.csv")
df["words"] = df.text.str.split().str.len()
labels = sorted(df.true.unique())

report = classification_report(df.true, df.pred, labels=labels, zero_division=0)
print(report)
open(f"results/report_{args.name}.txt", "w").write(report)

# Confusion matrix
cm = confusion_matrix(df.true, df.pred, labels=labels)
fig, ax = plt.subplots(figsize=(10, 9))
ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=90)
ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)
ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix: {args.name}")
plt.tight_layout(); plt.savefig(f"results/confusion_{args.name}.png", dpi=150); plt.close()

errors = df[df.true != df.pred]
print(f"\nErrors: {len(errors)} / {len(df)} ({len(errors)/len(df):.1%})")

print("\nMost frequent confusions (true -> predicted):")
print(errors.groupby(["true", "pred"]).size().sort_values(ascending=False).head(10))

df["bucket"] = pd.cut(df.words, [0, 100, 250, 500, 1000, 1e9],
                      labels=["<=100", "101-250", "251-500", "501-1000", ">1000"])
by_len = df.groupby("bucket", observed=True).apply(lambda g: (g.true != g.pred).mean())
print("\nError rate by article length (words):\n", by_len.round(3))

sample = errors.sample(min(args.n_samples, len(errors)), random_state=1)
sample["text"] = sample.text.str.slice(0, 500)
sample["cause (fill in)"] = ""
sample.to_csv(f"results/errors_sample_{args.name}.csv", index=False)
print(f"\nWrote results/errors_sample_{args.name}.csv - read them and fill in the 'cause' column.")

good = df[df.true == df.pred].sample(10, random_state=1)
good["text"] = good.text.str.slice(0, 300)
good.to_csv(f"results/correct_examples_{args.name}.csv", index=False)
