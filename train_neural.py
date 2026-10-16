"""
train_neural.py - Train the two neural models and compare them fairly.

    MLP : looks only at each developer's own features (like logistic regression)
    GCN : looks at the developer's features AND their neighbours' features

Both models have the same size, the same settings and the same training loop.
The ONLY difference is that the GCN uses the edges, so any gap between them
shows what the network connections add.

Run from the project root (after data.py):
    python train_neural.py              # both models
    python train_neural.py --model gcn  # just one

Reads:   artifacts/graph_data.pt, artifacts/splits.npz
Writes:  artifacts/models/mlp.pt, artifacts/models/gcn.pt   (best checkpoints)
         artifacts/history.csv      (model, epoch, train_loss, val_macro_f1)
         artifacts/predictions.csv  (mlp_pred, gcn_pred columns)
         artifacts/metrics.json     ("mlp", "gcn" entries)
         artifacts/config.json      ("mlp", "gcn" entries)
"""

import argparse
import copy
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torch_geometric
    from torch_geometric.nn import GCNConv
except ImportError:
    raise SystemExit("PyTorch Geometric is missing. Install it with:  pip install torch_geometric")

from data import SEED, load_graph_and_splits
from evaluate import macro_f1, save_model_results

ART_DIR = Path("artifacts")
MODEL_DIR = ART_DIR / "models"
HISTORY_PATH = ART_DIR / "history.csv"
CONFIG_PATH = ART_DIR / "config.json"

# ---------------------------------------------------------------- settings (from the README)
SETTINGS = {
    "hidden": 32,
    "dropout": 0.5,
    "lr": 0.01,
    "weight_decay": 5e-4,
    "max_epochs": 200,
    "patience": 20,          # stop after 20 epochs with no validation improvement
}

# At most two settings per model, chosen on VALIDATION macro-F1 only.
CANDIDATES = {
    "plain": {"class_weighted_loss": False},
    "weighted": {"class_weighted_loss": True},   # mistakes on the smaller ML class cost more
}


# ---------------------------------------------------------------- the two models
class MLP(nn.Module):
    """features -> hidden layer (32) -> 2 class scores. Never sees the edges."""

    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index=None):          # edge_index accepted but ignored
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, self.dropout, self.training)
        return self.lin2(x)


class GCN(nn.Module):
    """
    Same shape as the MLP, but each layer is a graph convolution:
    a developer's new vector = weighted average of its own vector and its neighbours'.
    Two layers -> information from neighbours AND neighbours-of-neighbours.
    """

    def __init__(self, in_dim, hidden, out_dim, dropout):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden, cached=True)   # cached: the graph never changes
        self.conv2 = GCNConv(hidden, out_dim, cached=True)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, self.dropout, self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)


MODELS = {"mlp": MLP, "gcn": GCN}


# ---------------------------------------------------------------- training
def standardize(x, train_mask):
    """Same idea as StandardScaler in the baseline: mean/std from TRAIN nodes only."""
    mean = x[train_mask].mean(dim=0)
    std = x[train_mask].std(dim=0).clamp_min(1e-8)
    return (x - mean) / std, mean, std


def train_one(kind, candidate, x, edge_index, y, masks):
    """Train one model with early stopping. Returns the best weights and the history."""
    torch.manual_seed(SEED)                         # same starting weights every run
    s = SETTINGS
    model = MODELS[kind](x.shape[1], s["hidden"], 2, s["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=s["lr"], weight_decay=s["weight_decay"])

    train, val = masks["train_mask"], masks["val_mask"]
    weight = None
    if CANDIDATES[candidate]["class_weighted_loss"]:
        counts = torch.bincount(y[train], minlength=2).float()
        weight = counts.sum() / (2 * counts)        # rarer class -> bigger weight

    best_f1, best_epoch, best_state, waited = -1.0, 0, None, 0
    history = []

    for epoch in range(1, s["max_epochs"] + 1):
        # --- one training step: loss uses TRAIN labels only ---
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)                  # GCN sees the whole graph (transductive)
        loss = F.cross_entropy(out[train], y[train], weight=weight)
        loss.backward()
        optimizer.step()

        # --- check on VALIDATION nodes (dropout off) ---
        model.eval()
        with torch.no_grad():
            pred = model(x, edge_index).argmax(dim=1)
        val_f1 = macro_f1(y[val], pred[val])
        history.append({"model": kind, "epoch": epoch,
                        "train_loss": round(loss.item(), 6), "val_macro_f1": round(val_f1, 6)})

        # --- early stopping: remember the best epoch, stop if no progress ---
        if val_f1 > best_f1:
            best_f1, best_epoch, waited = val_f1, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            waited += 1
            if waited >= s["patience"]:
                break

    return best_state, best_f1, best_epoch, history


def run(kind, x_raw, edge_index, y, masks):
    print(f"\n=== {kind.upper()} ===")
    x, mean, std = standardize(x_raw, masks["train_mask"])

    # 1) try each candidate, keep the one with the best VALIDATION macro-F1
    results = {}
    for name in CANDIDATES:
        start = time.time()
        state, f1, epoch, hist = train_one(kind, name, x, edge_index, y, masks)
        results[name] = (f1, epoch, state, hist)
        print(f"  {name:<9} best validation macro-F1 = {f1:.4f} at epoch {epoch:>3} "
              f"({len(hist)} epochs, {time.time() - start:.1f}s)")

    chosen = max(results, key=lambda k: results[k][0])
    best_f1, best_epoch, best_state, history = results[chosen]
    print(f"Chosen: {chosen}")

    # 2) save the best checkpoint (weights + the scaling numbers needed to reuse it)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_DIR / f"{kind}.pt"
    torch.save({"state_dict": best_state, "mean": mean, "std": std,
                "settings": SETTINGS, "candidate": chosen, "best_epoch": best_epoch}, ckpt_path)

    # 3) reload it from disk (proves the saved file works), then predict every node
    ckpt = torch.load(ckpt_path)
    model = MODELS[kind](x.shape[1], SETTINGS["hidden"], 2, SETTINGS["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(x, edge_index).argmax(dim=1)

    # 4) TEST evaluation happens only here, once, through the shared evaluate.py
    save_model_results(kind, preds, y, masks, extra={
        "chosen_config": chosen,
        "best_epoch": best_epoch,
        "val_macro_f1_per_config": {k: round(v[0], 4) for k, v in results.items()},
    })

    # 5) training curve of the chosen run -> history.csv (replace old rows for this model)
    new = pd.DataFrame(history)
    if HISTORY_PATH.exists():
        old = pd.read_csv(HISTORY_PATH)
        new = pd.concat([old[old["model"] != kind], new], ignore_index=True)
    new.to_csv(HISTORY_PATH, index=False)

    # 6) record settings
    config = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    config[kind] = {
        "model": "2-layer GCN (GCNConv)" if kind == "gcn" else "2-layer MLP",
        "uses_edges": kind == "gcn",
        **SETTINGS,
        "optimizer": "Adam",
        "selection": "best validation macro-F1 epoch",
        "candidates": CANDIDATES,
        "chosen": chosen,
        "best_epoch": best_epoch,
        "seed": SEED,
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
    }
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "gcn", "both"], default="both")
    args = parser.parse_args()

    x, edge_index, y, masks = load_graph_and_splits()
    x, y = x.float(), y.long()
    kinds = ["mlp", "gcn"] if args.model == "both" else [args.model]
    for kind in kinds:
        run(kind, x, edge_index, y, masks)
    print("\nSaved checkpoints, history.csv, predictions.csv, metrics.json, config.json")


if __name__ == "__main__":
    main()