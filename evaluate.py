"""
evaluate.py - The ONE place where models are scored and results are saved.

Every training script (logistic regression, MLP, GCN) calls these functions,
so all models are measured in exactly the same way and saved in the same format.

Functions:
    macro_f1(y_true, y_pred)                 -> one number, used during training (validation)
    compute_metrics(y_true, y_pred)          -> full report: accuracy, macro-F1, per-class, confusion matrix
    save_model_results(name, preds, y, masks)-> writes this model's column into predictions.csv
                                                and its test metrics into metrics.json

Self-test (no real data needed):
    python evaluate.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ART_DIR = Path("artifacts")
PRED_PATH = ART_DIR / "predictions.csv"
METRICS_PATH = ART_DIR / "metrics.json"

# Fixed class order for every metric and confusion matrix: 0 = web, 1 = ml.
# Kept identical to CLASS_NAMES in data.py.
CLASS_NAMES = {0: "web", 1: "ml"}
LABELS = sorted(CLASS_NAMES)

# Short column names used in predictions.csv
MODEL_COLUMNS = {"lr": "lr_pred", "mlp": "mlp_pred", "gcn": "gcn_pred"}


def _to_numpy(a):
    """Accept numpy arrays, lists or torch tensors."""
    if hasattr(a, "detach"):          # torch tensor
        a = a.detach().cpu().numpy()
    return np.asarray(a)


# ---------------------------------------------------------------- scoring
def macro_f1(y_true, y_pred):
    """Average of the per-class F1 scores. Used to pick the best checkpoint on validation."""
    return float(f1_score(_to_numpy(y_true), _to_numpy(y_pred),
                          labels=LABELS, average="macro", zero_division=0))


def compute_metrics(y_true, y_pred):
    """Full set of numbers shown in the app's Model comparison view."""
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    per_class = {
        CLASS_NAMES[c]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, c in enumerate(LABELS)
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": macro_f1(y_true, y_pred),
        "per_class": per_class,
        "class_order": [CLASS_NAMES[c] for c in LABELS],
        # rows = true class, columns = predicted class, in class_order
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "num_evaluated": int(len(y_true)),
    }


# ---------------------------------------------------------------- saving
def _split_names(masks, n):
    """Turn the three boolean masks into one column: 'train' / 'val' / 'test'."""
    split = np.empty(n, dtype=object)
    for name in ("train", "val", "test"):
        split[_to_numpy(masks[f"{name}_mask"]).astype(bool)] = name
    return split


def save_model_results(model, preds, y, masks, extra=None):
    """
    model : "lr", "mlp" or "gcn"
    preds : predicted class for EVERY node (length N), in node_id order
    y     : true labels for every node
    masks : dict with train_mask / val_mask / test_mask
    extra : optional dict saved next to the metrics (e.g. best epoch)

    Test metrics use ONLY test nodes. Validation metrics are saved too, for reference.
    """
    if model not in MODEL_COLUMNS:
        raise ValueError(f"model must be one of {list(MODEL_COLUMNS)}")

    preds, y = _to_numpy(preds).astype(int), _to_numpy(y).astype(int)
    n = len(y)
    if len(preds) != n:
        raise ValueError(f"need a prediction for every node: got {len(preds)}, expected {n}")

    ART_DIR.mkdir(exist_ok=True)

    # 1) predictions.csv - one row per node, one column per model
    if PRED_PATH.exists():
        df = pd.read_csv(PRED_PATH)
        if len(df) != n or not (df["true_label"].to_numpy() == y).all():
            raise ValueError("predictions.csv does not match the current data; delete it and rerun")
    else:
        df = pd.DataFrame({
            "node_id": np.arange(n),
            "split": _split_names(masks, n),
            "true_label": y,
        })
    df[MODEL_COLUMNS[model]] = preds

    # keep a fixed column order so the app always finds what it expects
    order = ["node_id", "split", "true_label"] + [c for c in MODEL_COLUMNS.values() if c in df]
    df[order].to_csv(PRED_PATH, index=False)

    # 2) metrics.json - one entry per model
    test = _to_numpy(masks["test_mask"]).astype(bool)
    val = _to_numpy(masks["val_mask"]).astype(bool)
    all_metrics = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
    all_metrics[model] = {
        "test": compute_metrics(y[test], preds[test]),
        "val": compute_metrics(y[val], preds[val]),
        **(extra or {}),
    }
    METRICS_PATH.write_text(json.dumps(all_metrics, indent=2))

    t = all_metrics[model]["test"]
    print(f"[{model}] test accuracy {t['accuracy']:.4f} | test macro-F1 {t['macro_f1']:.4f}")
    return all_metrics[model]


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    # Tiny hand-made example so you can check the maths without real data.
    y_true = [0, 0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 0, 1, 0]
    m = compute_metrics(y_true, y_pred)

    print("Self-test on 6 fake nodes")
    print("  true:", y_true)
    print("  pred:", y_pred)
    print(f"  accuracy  = {m['accuracy']:.3f}   (4 of 6 correct = 0.667)")
    print(f"  macro-F1  = {m['macro_f1']:.3f}   (average of web F1 0.750 and ml F1 0.500 = 0.625)")
    print("  confusion matrix (rows = true, columns = predicted):")
    print("            pred web  pred ml")
    for name, row in zip(m["class_order"], m["confusion_matrix"]):
        print(f"    {name:>4}     {row[0]:>5}    {row[1]:>5}")

    assert abs(m["accuracy"] - 4 / 6) < 1e-9
    assert abs(m["macro_f1"] - 0.625) < 1e-9
    assert m["confusion_matrix"] == [[3, 1], [1, 1]]
    print("Self-test passed.")