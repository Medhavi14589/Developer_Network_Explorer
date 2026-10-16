"""
data.py - Build the GitHub developer graph, verify it, create the fixed
stratified 60/20/20 split, and save the shared artifacts.

Data source: the original MUSAE GitHub files published by the dataset authors
(Rozemberczki, Allen and Sarkar) on GitHub. Get them once with:

    git clone --depth 1 https://github.com/benedekrozemberczki/musae data/musae

Then run from the project root:

    python data.py

Creates:
    artifacts/graph_data.pt   -> {"x": [N,F], "edge_index": [2,E], "y": [N]}
    artifacts/splits.npz      -> train_mask, val_mask, test_mask (bool, length N)
    artifacts/metadata.json   -> dimensions, class mapping, counts, seed, preprocessing

Other scripts should NOT re-split. They load the saved files with
load_graph_and_splits() at the bottom of this file.
"""

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import sklearn
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- settings
SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.6, 0.2, 0.2
NUM_FEATURES = 128          # size of each developer's feature vector after compression

RAW_DIR = Path("data/musae/input")    # cloned raw files (git-ignored)
EDGES_FILE = RAW_DIR / "edges" / "git_edges.csv"
TARGET_FILE = RAW_DIR / "target" / "git_target.csv"
FEATURES_FILE = RAW_DIR / "features" / "git.json"

ART_DIR = Path("artifacts")           # shared outputs (committed)

# In git_target.csv the column "ml_target" is 1 for ML developers, 0 for web developers.
CLASS_NAMES = {0: "web", 1: "ml"}


# ---------------------------------------------------------------- 1. read raw files
def load_raw():
    """Read the three raw files. Returns edges (array of pairs), labels, sparse features."""
    missing = [p for p in (EDGES_FILE, TARGET_FILE, FEATURES_FILE) if not p.exists()]
    if missing:
        raise SystemExit(
            "\nRaw data not found: " + ", ".join(str(p) for p in missing) + "\n"
            "Get it once with:\n"
            "    git clone --depth 1 https://github.com/benedekrozemberczki/musae data/musae\n"
            "then run  python data.py  again."
        )

    # Labels. We keep only id and label; usernames are not used anywhere.
    target = pd.read_csv(TARGET_FILE, usecols=["id", "ml_target"]).sort_values("id")
    n = len(target)
    assert (target["id"].to_numpy() == np.arange(n)).all(), "node ids are not 0..N-1"
    y = target["ml_target"].to_numpy().astype(np.int64)

    # Edges: one row per mutual-follower pair.
    edges = pd.read_csv(EDGES_FILE).to_numpy().astype(np.int64)

    # Features: for each developer, a list of feature ids that are "on".
    # Turn it into a 0/1 matrix with one row per developer (stored sparse, mostly zeros).
    with open(FEATURES_FILE) as f:
        feats = json.load(f)
    rows, cols = [], []
    for node_id, ids in feats.items():
        rows.extend([int(node_id)] * len(ids))
        cols.extend(ids)
    num_raw = max(cols) + 1
    x_raw = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, num_raw)
    )
    x_raw.data[:] = 1.0            # in case an id appears twice for the same node

    return edges, y, x_raw


# ---------------------------------------------------------------- 2. preprocess
def compress_features(x_raw, k=NUM_FEATURES, seed=SEED):
    """
    The raw matrix has ~4,000 mostly-zero columns. TruncatedSVD squeezes it into k
    dense columns that keep most of the information, so the models and the saved
    file stay small. It looks only at the features, never at labels.
    """
    svd = TruncatedSVD(n_components=k, random_state=seed)
    x = svd.fit_transform(x_raw).astype(np.float32)
    explained = float(svd.explained_variance_ratio_.sum())
    return x, explained


def build_edge_index(edges, n):
    """Store every connection in both directions (A->B and B->A), drop duplicates and self-loops."""
    both = np.concatenate([edges, edges[:, ::-1]], axis=0)
    both = both[both[:, 0] != both[:, 1]]            # no self-loops
    both = np.unique(both, axis=0)                   # no duplicates
    assert both.min() >= 0 and both.max() < n, "edge refers to a node that does not exist"
    return both.T                                    # shape [2, E]


# ---------------------------------------------------------------- 3. verify
def verify(x, edge_index, y):
    """Sanity checks. Stops with an error if anything is inconsistent."""
    n = len(y)
    assert x.shape[0] == n, f"x has {x.shape[0]} rows, expected {n}"
    assert edge_index.shape[0] == 2
    assert not np.isnan(x).any(), "features contain NaN"
    labels = set(np.unique(y).tolist())
    assert labels <= set(CLASS_NAMES), f"unexpected labels {labels}"

    degree = np.bincount(edge_index[0], minlength=n)
    info = {
        "num_nodes": int(n),
        "num_features": int(x.shape[1]),
        "num_edge_entries": int(edge_index.shape[1]),          # each connection stored twice
        "num_undirected_edges": int(edge_index.shape[1] // 2),
        "num_isolated_nodes": int((degree == 0).sum()),
        "mean_degree": round(float(degree.mean()), 2),
        "class_counts": {CLASS_NAMES[c]: int((y == c).sum()) for c in sorted(labels)},
    }
    print("Dataset check passed:")
    for k, v in info.items():
        print(f"  {k}: {v}")
    return info


# ---------------------------------------------------------------- 4. split
def make_split(y, seed=SEED):
    """Stratified 60/20/20 split -> three boolean masks over all nodes."""
    y = np.asarray(y)
    n = len(y)
    idx = np.arange(n)

    train_idx, rest_idx = train_test_split(
        idx, test_size=VAL_FRAC + TEST_FRAC, stratify=y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=y[rest_idx], random_state=seed,
    )

    masks = {}
    for name, part in [("train_mask", train_idx), ("val_mask", val_idx), ("test_mask", test_idx)]:
        m = np.zeros(n, dtype=bool)
        m[part] = True
        masks[name] = m

    total = masks["train_mask"].astype(int) + masks["val_mask"] + masks["test_mask"]
    assert (total == 1).all(), "splits overlap or miss nodes"
    return masks


# ---------------------------------------------------------------- 5. save
def save_artifacts(x, edge_index, y, masks, info, explained):
    ART_DIR.mkdir(exist_ok=True)

    torch.save(
        {
            "x": torch.from_numpy(x),
            "edge_index": torch.from_numpy(np.ascontiguousarray(edge_index)),
            "y": torch.from_numpy(y),
        },
        ART_DIR / "graph_data.pt",
    )
    np.savez(ART_DIR / "splits.npz", **masks)

    split_counts = {
        split.replace("_mask", ""): {
            "total": int(m.sum()),
            **{CLASS_NAMES[c]: int(((y == c) & m).sum()) for c in CLASS_NAMES},
        }
        for split, m in masks.items()
    }

    metadata = {
        "data_source": "MUSAE GitHub dataset (Rozemberczki et al.), raw files from "
                       "https://github.com/benedekrozemberczki/musae",
        **info,
        "class_mapping": {str(k): v for k, v in CLASS_NAMES.items()},
        "node_id_convention": "node_id = row index in x and y (0 .. N-1) = id column in git_target.csv",
        "seed": SEED,
        "split_fractions": {"train": TRAIN_FRAC, "val": VAL_FRAC, "test": TEST_FRAC},
        "split_method": "sklearn train_test_split, stratified on y, two stages",
        "split_counts": split_counts,
        "preprocessing": (
            f"binary feature lists -> 0/1 matrix -> TruncatedSVD to {NUM_FEATURES} dims "
            f"(seed {SEED}, unsupervised, labels not used); edges made undirected, "
            "duplicates and self-loops removed"
        ),
        "svd_explained_variance": round(explained, 4),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit-learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    with open(ART_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved graph_data.pt, splits.npz, metadata.json to {ART_DIR}/")
    print("Split sizes:", {k: v["total"] for k, v in split_counts.items()})


# ---------------------------------------------------------------- for other scripts
def load_graph_and_splits(art_dir=ART_DIR):
    """
    Use this in train_baseline.py, train_neural.py and app.py:

        from data import load_graph_and_splits
        x, edge_index, y, masks = load_graph_and_splits()
        x_train, y_train = x[masks["train_mask"]], y[masks["train_mask"]]
    """
    art_dir = Path(art_dir)
    g = torch.load(art_dir / "graph_data.pt")
    s = np.load(art_dir / "splits.npz")
    masks = {k: torch.from_numpy(s[k]) for k in ("train_mask", "val_mask", "test_mask")}
    return g["x"], g["edge_index"], g["y"], masks


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    edges, y, x_raw = load_raw()
    print(f"Read {len(y)} developers, {len(edges)} connections, {x_raw.shape[1]} raw features")
    x, explained = compress_features(x_raw)
    print(f"Compressed features to {x.shape[1]} columns (keeps {explained:.1%} of the variance)\n")
    edge_index = build_edge_index(edges, len(y))
    info = verify(x, edge_index, y)
    masks = make_split(y)
    save_artifacts(x, edge_index, y, masks, info, explained)