"""Shared helpers: loading the prepared splits, computing metrics, logging experiments."""
import json
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score

DATA_DIR = "data"
RESULTS_DIR = "results"


def load_splits():
    """Return train, val, test DataFrames (columns: text, label, label_name) and label names."""
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    val = pd.read_csv(f"{DATA_DIR}/val.csv")
    test = pd.read_csv(f"{DATA_DIR}/test.csv")
    with open(f"{DATA_DIR}/labels.json", encoding="utf-8") as f:
        labels = json.load(f)
    return train, val, test, labels


def compute_metrics_dict(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(p, 4),
        "recall_macro": round(r, 4),
        "f1_macro": round(f, 4),
        "f1_weighted": round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4),
    }


def log_experiment(name, config, metrics):
    """Append one row to results/experiments.csv (this becomes your report's experiments table)."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = f"{RESULTS_DIR}/experiments.csv"
    row = {"model": name, "config": config, **metrics}
    df = pd.DataFrame([row])
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
    print(f"\n[logged] {name} | {config}\n{metrics}")


def save_predictions(name, texts, y_true, y_pred, labels):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df = pd.DataFrame(
        {
            "text": list(texts),
            "true": [labels[i] for i in y_true],
            "pred": [labels[i] for i in y_pred],
        }
    )
    path = f"{RESULTS_DIR}/preds_{name}.csv"
    df.to_csv(path, index=False)
    print(f"[saved] predictions -> {path}")
