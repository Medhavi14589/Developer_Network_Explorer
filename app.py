"""
app.py - Developer Network Explorer (Streamlit).

The app ONLY reads the saved files in artifacts/. It never trains anything,
so it starts quickly and needs no GPU.

Run from the project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
import torch

ART_DIR = Path("artifacts")
MAX_NEIGHBOURS = 50          # README: neighbourhood capped at 50 nodes
SAMPLE_SEED = 42             # fixed, so the same sample is drawn every time

MODEL_NAMES = {"lr": "Logistic regression", "mlp": "MLP", "gcn": "GCN"}
MODEL_USES = {"lr": "attributes only", "mlp": "attributes only", "gcn": "attributes + edges"}
PRED_COLS = {"lr": "lr_pred", "mlp": "mlp_pred", "gcn": "gcn_pred"}
CLASS_COLOURS = {"web": "#3b82c4", "ml": "#e08a2c"}   # blue / orange, colour-blind friendly

st.set_page_config(page_title="Developer Network Explorer", layout="wide")


# ---------------------------------------------------------------- loading (cached)
@st.cache_data
def load_tables():
    """Small files: read once, reuse on every click."""
    metadata = json.loads((ART_DIR / "metadata.json").read_text())
    metrics = json.loads((ART_DIR / "metrics.json").read_text())
    preds = pd.read_csv(ART_DIR / "predictions.csv")
    history_path = ART_DIR / "history.csv"
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    return metadata, metrics, preds, history


@st.cache_resource
def load_neighbours():
    """
    Turn edge_index into a fast lookup: neighbours of node i are
    dst[start[i]:start[i+1]]. Built once and kept in memory.
    """
    g = torch.load(ART_DIR / "graph_data.pt")
    src, dst = g["edge_index"].numpy()
    order = np.argsort(src, kind="stable")
    src, dst = src[order], dst[order]
    n = len(g["y"])
    start = np.searchsorted(src, np.arange(n + 1))
    return dst, start


def neighbours_of(node, dst, start):
    return dst[start[node]:start[node + 1]]


def class_name(label, metadata):
    return metadata["class_mapping"][str(int(label))]


def draw_graph(graph, colours, sizes, edge_alpha=0.3, labels=None):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    pos = nx.spring_layout(graph, seed=SAMPLE_SEED)
    nx.draw_networkx_edges(graph, pos, ax=ax, alpha=edge_alpha, width=0.8)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=colours, node_size=sizes,
                           edgecolors="white", linewidths=0.8)
    if labels:
        nx.draw_networkx_labels(graph, pos, labels=labels, ax=ax, font_size=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=k, markersize=9)
               for k, c in CLASS_COLOURS.items()]
    ax.legend(handles=handles, title="Recorded label", loc="upper right", frameon=False)
    ax.axis("off")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------- view 1
def network_overview(metadata, preds, dst, start):
    st.header("Network overview")
    st.write("Numbers below come from the loaded data files, not from documentation.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Developers (nodes)", f"{metadata['num_nodes']:,}")
    c2.metric("Connections (edges)", f"{metadata['num_undirected_edges']:,}")
    c3.metric("Features per developer", metadata["num_features"])
    c4.metric("Average connections", metadata["mean_degree"])

    left, right = st.columns(2)
    with left:
        st.subheader("Class distribution")
        counts = pd.Series(metadata["class_counts"], name="developers")
        st.bar_chart(counts, color="#3b82c4")
        share = counts / counts.sum()
        st.caption(" · ".join(f"{k}: {v:,} ({share[k]:.0%})" for k, v in counts.items()))
    with right:
        st.subheader("Train / validation / test split")
        split = pd.DataFrame(metadata["split_counts"]).T
        split.index.name = "split"
        st.dataframe(split, width="stretch")
        st.caption(f"Stratified 60/20/20, seed {metadata['seed']}. "
                   "Every model uses these same masks.")

    st.subheader("A small sample of the graph")
    size = st.slider("Nodes in sample", 20, 150, 60, step=10)
    # breadth-first walk from a fixed random developer, so the sample is connected
    rng = np.random.default_rng(SAMPLE_SEED)
    seed_node = int(rng.integers(metadata["num_nodes"]))
    picked, frontier = [seed_node], [seed_node]
    seen = {seed_node}
    while frontier and len(picked) < size:
        nxt = []
        for v in frontier:
            for u in neighbours_of(v, dst, start):
                u = int(u)
                if u not in seen and len(picked) < size:
                    seen.add(u)
                    picked.append(u)
                    nxt.append(u)
        frontier = nxt
    g = nx.Graph()
    g.add_nodes_from(picked)
    for v in picked:
        g.add_edges_from((v, int(u)) for u in neighbours_of(v, dst, start) if int(u) in seen)
    labels = preds.set_index("node_id").loc[picked, "true_label"]
    colours = [CLASS_COLOURS[class_name(labels[v], metadata)] for v in g.nodes]
    st.pyplot(draw_graph(g, colours, 60), width="content")
    st.caption(f"{len(picked)} of {metadata['num_nodes']:,} developers, reached by walking "
               f"outward from developer {seed_node}. A sample for illustration only.")


# ---------------------------------------------------------------- view 2
def developer_explorer(metadata, preds, dst, start):
    st.header("Developer explorer")
    st.write("Pick a developer who was **held out for testing**: no model saw their label "
             "during training.")

    test = preds[preds["split"] == "test"].copy()
    available = [m for m in PRED_COLS if PRED_COLS[m] in test]
    if not available:
        st.warning("No model predictions found yet. Run the training scripts first.")
        return

    only_disagree = st.checkbox("Only show developers where the models disagree")
    if only_disagree and len(available) > 1:
        cols = [PRED_COLS[m] for m in available]
        test = test[test[cols].nunique(axis=1) > 1]
    st.caption(f"{len(test):,} test developers to choose from.")
    if test.empty:
        return

    node = st.selectbox("Test developer (node ID)", test["node_id"].tolist())
    row = test.set_index("node_id").loc[node]
    truth = class_name(row["true_label"], metadata)

    st.subheader(f"Developer {node}")
    cols = st.columns(len(available) + 1)
    cols[0].metric("Recorded label", truth)
    for col, m in zip(cols[1:], available):
        guess = class_name(row[PRED_COLS[m]], metadata)
        col.metric(MODEL_NAMES[m], guess,
                   delta="correct" if guess == truth else "wrong",
                   delta_color="normal" if guess == truth else "inverse")

    # one-hop neighbourhood, capped
    nbrs = neighbours_of(node, dst, start)
    total = len(nbrs)
    shown = nbrs
    if total > MAX_NEIGHBOURS:
        rng = np.random.default_rng(SAMPLE_SEED + int(node))
        shown = np.sort(rng.choice(nbrs, MAX_NEIGHBOURS, replace=False))

    st.subheader("One-hop neighbourhood")
    if total == 0:
        st.info("This developer has no connections.")
        return
    all_labels = preds.set_index("node_id")["true_label"]
    nbr_classes = [class_name(all_labels[int(u)], metadata) for u in nbrs]
    ml_share = nbr_classes.count("ml") / total

    g = nx.Graph()
    g.add_node(int(node))
    g.add_edges_from((int(node), int(u)) for u in shown)
    colours = [CLASS_COLOURS[class_name(all_labels[v], metadata)] for v in g.nodes]
    sizes = [500 if v == node else 90 for v in g.nodes]

    left, right = st.columns([2, 1])
    with left:
        st.pyplot(draw_graph(g, colours, sizes, edge_alpha=0.4, labels={int(node): str(node)}),
                  width="content")
    with right:
        st.metric("Connections", total)
        st.metric("Neighbours recorded as ML", f"{ml_share:.0%}")
        if total > MAX_NEIGHBOURS:
            st.warning(f"Showing a random sample of {MAX_NEIGHBOURS} of {total} neighbours.")
        st.caption("This picture is context, not an explanation of any prediction. "
                   "The GCN uses the full graph, including neighbours of neighbours.")


# ---------------------------------------------------------------- view 3
def model_comparison(metadata, metrics, history):
    st.header("Model comparison")
    st.write("All scores are on the **test** developers, from one fixed split and seed. "
             "They do not show that one model is better in general.")

    models = [m for m in MODEL_NAMES if m in metrics]
    if not models:
        st.warning("No metrics found yet. Run the training scripts first.")
        return

    summary = pd.DataFrame([{
        "Model": MODEL_NAMES[m],
        "Uses": MODEL_USES[m],
        "Accuracy": metrics[m]["test"]["accuracy"],
        "Macro-F1": metrics[m]["test"]["macro_f1"],
        "Chosen setting": metrics[m].get("chosen_config", "-"),
    } for m in models]).set_index("Model")
    st.dataframe(summary.style.format({"Accuracy": "{:.4f}", "Macro-F1": "{:.4f}"}),
                 width="stretch")

    st.subheader("Per-class results")
    rows = []
    for m in models:
        for cls, v in metrics[m]["test"]["per_class"].items():
            rows.append({"Model": MODEL_NAMES[m], "Class": cls, "Precision": v["precision"],
                         "Recall": v["recall"], "F1": v["f1"], "Support": v["support"]})
    st.dataframe(pd.DataFrame(rows).set_index(["Model", "Class"])
                 .style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                 width="stretch")
    st.caption("Recall for ml = share of ML developers the model found.")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        m = st.selectbox("Model", models, format_func=MODEL_NAMES.get)
        cm = np.array(metrics[m]["test"]["confusion_matrix"])
        order = metrics[m]["test"]["class_order"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        ax.imshow(cm, cmap="Blues")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks(range(len(order)), [f"pred {c}" for c in order])
        ax.set_yticks(range(len(order)), [f"true {c}" for c in order])
        ax.set_title(MODEL_NAMES[m])
        fig.tight_layout()
        st.pyplot(fig, width="content")

    with right:
        st.subheader("Neural training curves")
        if history.empty:
            st.info("history.csv not found. Run train_neural.py.")
        else:
            h = history.copy()
            h["model"] = h["model"].map(MODEL_NAMES)
            st.caption("Validation macro-F1 per epoch")
            st.line_chart(h.pivot(index="epoch", columns="model", values="val_macro_f1"))
            st.caption("Training loss per epoch")
            st.line_chart(h.pivot(index="epoch", columns="model", values="train_loss"))
            st.caption("Each model keeps the epoch with the best validation macro-F1, "
                       "then stops after 20 epochs without improvement.")


# ---------------------------------------------------------------- page
def main():
    st.title("Developer Network Explorer")

    needed = ["metadata.json", "predictions.csv", "metrics.json", "graph_data.pt"]
    missing = [f for f in needed if not (ART_DIR / f).exists()]
    if missing:
        st.error("Missing files in artifacts/: " + ", ".join(missing)
                 + ". Run data.py, train_baseline.py and train_neural.py first.")
        st.stop()

    metadata, metrics, preds, history = load_tables()
    dst, start = load_neighbours()

    view = st.sidebar.radio("View", ["Network overview", "Developer explorer", "Model comparison"])
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "GitHub web vs ML developers (MUSAE dataset). Fixed data, fixed node IDs, "
        "no live GitHub lookup. Educational prototype, not a hiring tool."
    )

    if view == "Network overview":
        network_overview(metadata, preds, dst, start)
    elif view == "Developer explorer":
        developer_explorer(metadata, preds, dst, start)
    else:
        model_comparison(metadata, metrics, history)


main()