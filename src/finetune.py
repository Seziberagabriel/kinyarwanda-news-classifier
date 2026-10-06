"""Step 4: fine-tune a pretrained multilingual/African-language transformer."""
import argparse
import math
import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import classification_report
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)
from utils import load_splits, compute_metrics_dict, log_experiment, save_predictions

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="castorini/afriberta_base")
ap.add_argument("--name", default="afriberta_base")
ap.add_argument("--lr", type=float, default=3e-5)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--max_train", type=int, default=0, help="use a random subset of N training rows (0 = all)")
ap.add_argument("--class_weights", action="store_true", help="weight the loss to help rare classes")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

set_seed(args.seed)
train, val, test, labels = load_splits()
if args.max_train:
    train = train.sample(args.max_train, random_state=args.seed)
    print(f"Using a training subset of {len(train)} rows")

tok = AutoTokenizer.from_pretrained(args.model)


def to_ds(df):
    ds = Dataset.from_pandas(df[["text", "label"]].reset_index(drop=True))
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=args.max_len), batched=True)
    return ds.remove_columns(["text"])


tr_ds, va_ds, te_ds = to_ds(train), to_ds(val), to_ds(test)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(labels),
    id2label={i: l for i, l in enumerate(labels)},
    label2id={l: i for i, l in enumerate(labels)},
)


def metrics_fn(p):
    return compute_metrics_dict(p.label_ids, p.predictions.argmax(-1))


# 10% of all training steps as warm-up (integer steps work in every transformers version)
total_steps = math.ceil(len(train) / args.batch) * args.epochs
warmup = int(0.1 * total_steps)

targs = TrainingArguments(
    output_dir=f"models/{args.name}_ckpt",
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    warmup_steps=warmup,
    fp16=torch.cuda.is_available(),
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    report_to="none",
    seed=args.seed,
)


class WeightedTrainer(Trainer):
    """Cross-entropy with per-class weights (inverse square-root frequency)."""
    def __init__(self, *a, class_weights=None, **kw):
        super().__init__(*a, **kw)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels_ = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_weights.to(outputs.logits.device, dtype=outputs.logits.dtype)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels_, weight=w)
        return (loss, outputs) if return_outputs else loss


common = dict(model=model, args=targs, train_dataset=tr_ds, eval_dataset=va_ds,
              data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics_fn)
if args.class_weights:
    counts = np.bincount(train.label.values, minlength=len(labels)).clip(min=1)
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    print("Class weights:", dict(zip(labels, w.round(2))))
    trainer = WeightedTrainer(class_weights=torch.tensor(w, dtype=torch.float), **common)
else:
    trainer = Trainer(**common)

trainer.train()

out = trainer.predict(te_ds)
pred = out.predictions.argmax(-1)
print(classification_report(out.label_ids, pred, target_names=labels, zero_division=0))
cfg = (f"{args.model}, lr={args.lr}, epochs={args.epochs}, max_len={args.max_len}, "
       f"batch={args.batch}, n_train={len(train)}, class_weights={args.class_weights}")
log_experiment(args.name, cfg, compute_metrics_dict(out.label_ids, pred))
save_predictions(args.name, test.text, test.label, pred, labels)

save_dir = f"models/{args.name}"
trainer.save_model(save_dir)
tok.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")
