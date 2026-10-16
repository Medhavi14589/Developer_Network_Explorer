"""
train_baseline.py - Logistic regression on developer features only (no edges).

This is the simple baseline. If the GCN cannot beat it, the network connections
are not adding much on this split.

Run from the project root (after data.py):
    python train_baseline.py

Reads:   artifacts/graph_data.pt, artifacts/splits.npz
Writes:  artifacts/models/lr.joblib          (scaler + model, ready to reload)
         artifacts/predictions.csv           (lr_pred column)
         artifacts/metrics.json              ("lr" entry)
         artifacts/config.json               ("lr" entry: chosen settings)
"""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import SEED, load_graph_and_splits
from evaluate import macro_f1, save_model_results

ART_DIR = Path("artifacts")
MODEL_DIR = ART_DIR / "models"
CONFIG_PATH = ART_DIR / "config.json"

# At most two settings, as the README allows. Chosen by VALIDATION macro-F1 only.
CANDIDATES = {
    "plain":    {"C": 1.0, "class_weight": None},
    "balanced": {"C": 1.0, "class_weight": "balanced"},   # gives the smaller ML class more weight
}


def build_model(C, class_weight):
    """Scale each feature to mean 0 / std 1, then logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=2000, random_state=SEED),
    )


def main():
    x, _edge_index, y, masks = load_graph_and_splits()   # edges are not used by this model
    x, y = x.numpy(), y.numpy()
    train, val = masks["train_mask"].numpy(), masks["val_mask"].numpy()

    # 1) Fit each candidate on TRAIN nodes, score it on VALIDATION nodes.
    results = {}
    for name, params in CANDIDATES.items():
        model = build_model(**params)
        model.fit(x[train], y[train])            # scaler is also fitted on train only
        score = macro_f1(y[val], model.predict(x[val]))
        results[name] = (score, model)
        print(f"  {name:<9} validation macro-F1 = {score:.4f}")

    # 2) Keep the best one. The test set has not been looked at yet.
    best_name = max(results, key=lambda k: results[k][0])
    best_score, best_model = results[best_name]
    print(f"Chosen: {best_name} (validation macro-F1 {best_score:.4f})\n")

    # 3) Save the model so the app or a teammate can reload it without retraining.
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "lr.joblib")

    # 4) Predict every node and let evaluate.py score the TEST nodes and save everything.
    preds = best_model.predict(x)
    save_model_results(
        "lr", preds, y, masks,
        extra={"chosen_config": best_name,
               "val_macro_f1_per_config": {k: round(v[0], 4) for k, v in results.items()}},
    )

    # 5) Record the settings used.
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    config["lr"] = {
        "model": "StandardScaler + LogisticRegression",
        "candidates": CANDIDATES,
        "chosen": best_name,
        "max_iter": 2000,
        "seed": SEED,
        "uses_edges": False,
        "scikit-learn": sklearn.__version__,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))
    print(f"Saved {MODEL_DIR / 'lr.joblib'}, predictions.csv, metrics.json, config.json")


if __name__ == "__main__":
    main()