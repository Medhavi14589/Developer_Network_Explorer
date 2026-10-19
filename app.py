"""
app.py - Developer Network Explorer (Streamlit).

The app ONLY reads the saved files in artifacts/. It never trains anything,
so it starts quickly and needs no GPU.

Run from the project root:
    streamlit run app.py
"""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

ART_DIR = Path("artifacts")
MAX_NEIGHBOURS = 50          # README: neighbourhood capped at 50 nodes
SAMPLE_SEED = 42             # fixed, so the same sample is drawn every time

MODEL_NAMES = {"lr": "Logistic regression", "mlp": "MLP", "gcn": "GCN"}
MODEL_USES = {"lr": "Attributes only", "mlp": "Attributes only", "gcn": "Attributes + edges"}
PRED_COLS = {"lr": "lr_pred", "mlp": "mlp_pred", "gcn": "gcn_pred"}

# Validated categorical palette (colour-blind safe, fixed order).
CLASS_COLOURS = {"web": "#9085e9", "ml": "#d95926"}
CLASS_LABELS = {"web": "Web", "ml": "ML"}            # display names for the data keys
MODEL_COLOURS = {"lr": "#9085e9", "mlp": "#d95926", "gcn": "#199e70"}
INK, INK_2, MUTED = "#ede9f7", "#b6aec8", "#847b97"
GRID, AXIS, EDGE = "#211c2d", "#342c46", "#463d5c"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
SEQ = [[0, "#1d1830"], [1, "#6a55d0"]]   # violet ramp, dark enough for light text on every cell
FONT = "Manrope, system-ui, -apple-system, 'Segoe UI', sans-serif"

st.set_page_config(page_title="Developer Network Explorer", page_icon=":material/hub:",
                   layout="wide")

CSS = """
<style>
[data-testid="stApp"] {background: radial-gradient(900px 500px at 15% -5%, rgba(139,92,246,0.16), transparent 65%),
  radial-gradient(700px 400px at 100% 0%, rgba(99,60,200,0.10), transparent 60%), #0b0910;}
[data-testid="stHeader"] {background: rgba(11,9,16,0.75); backdrop-filter: blur(8px);}
.dne-title, .dne-value, .dne-section {font-family: 'Space Grotesk', Manrope, sans-serif;}
.block-container {max-width: 1240px; padding-top: 3.2rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -0.02em;}
.dne-hero {padding: 4px 0 18px 0;}
.dne-eyebrow {font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
  text-transform: uppercase; color: #b4a9f5; margin-bottom: 6px;}
.dne-title {font-size: 2.1rem; font-weight: 700; color: #ede9f7; line-height: 1.15;
  letter-spacing: -0.025em; margin: 0;}
.dne-sub {font-size: 1rem; color: #b6aec8; margin-top: 8px; max-width: 760px; line-height: 1.55;}
.dne-section {font-size: 1.05rem; font-weight: 600; color: #ede9f7; margin: 0 0 2px 0;}
.dne-note {font-size: 0.85rem; color: #847b97; line-height: 1.5;}
.dne-grid {display: grid; gap: 14px; margin-bottom: 18px;}
.dne-card {background: #16121f; border: 1px solid #2a2338; border-radius: 12px; padding: 16px 18px;}
.dne-card.accent {border-color: #8b5cf6; box-shadow: 0 0 0 1px #8b5cf6 inset, 0 0 28px rgba(139,92,246,0.22);}
.dne-label {font-size: 0.78rem; font-weight: 500; color: #847b97; text-transform: uppercase;
  letter-spacing: 0.05em;}
.dne-value {font-size: 1.75rem; font-weight: 650; color: #ede9f7; margin-top: 4px;
  letter-spacing: -0.02em;}
.dne-value.sm {font-size: 1.35rem;}
.dne-meta {font-size: 0.85rem; color: #b6aec8; margin-top: 4px;}
.dne-pill {display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px;
  border-radius: 999px; font-size: 0.8rem; font-weight: 500; border: 1px solid #342c46;
  background: #1f1a2b; color: #b6aec8;}
.dne-pill.good {background: #142a14; border-color: #1f4d1f; color: #5fd35f;}
.dne-pill.bad {background: #331717; border-color: #5c2424; color: #f08a8a;}
.dne-pill.info {background: #211a3a; border-color: #3b2f6b; color: #c4bbf7;}
.dne-dot {display: inline-block; width: 10px; height: 10px; border-radius: 50%;}
.dne-bar {display: flex; gap: 2px; height: 12px; border-radius: 6px; overflow: hidden;
  margin: 10px 0 8px 0;}
.dne-bar > div:first-child {border-radius: 6px 0 0 6px;}
.dne-bar > div:last-child {border-radius: 0 6px 6px 0;}
.dne-legend {display: flex; gap: 18px; font-size: 0.85rem; color: #b6aec8; flex-wrap: wrap;}
.dne-table {width: 100%; border-collapse: collapse; font-size: 0.9rem;
  font-variant-numeric: tabular-nums;}
.dne-table th {text-align: right; font-weight: 500; color: #847b97; font-size: 0.78rem;
  text-transform: uppercase; letter-spacing: 0.05em; padding: 6px 8px;
  border-bottom: 1px solid #2a2338;}
.dne-table td {text-align: right; padding: 9px 8px; border-bottom: 1px solid #1f1a2b; color: #ede9f7;}
.dne-table th:first-child, .dne-table td:first-child {text-align: left;}
.dne-table tr:last-child td {border-bottom: none;}
.dne-footer {margin-top: 36px; padding-top: 14px; border-top: 1px solid #2a2338;
  font-size: 0.8rem; color: #847b97;}
[data-testid="stMarkdownContainer"]:has(> .dne-note) {margin-bottom: 0;}
</style>
"""


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


# ---------------------------------------------------------------- small UI helpers
def html(markup):
    st.markdown(markup, unsafe_allow_html=True)


def hero(eyebrow, title, sub):
    html(f'<div class="dne-hero"><div class="dne-eyebrow">{eyebrow}</div>'
         f'<div class="dne-title">{title}</div><div class="dne-sub">{sub}</div></div>')


def section(title, note=None):
    html(f'<div class="dne-section">{title}</div>'
         + (f'<div class="dne-note" style="margin-bottom:8px">{note}</div>' if note else ""))


def card(label, value, meta="", small=False, accent=False):
    return (f'<div class="dne-card{" accent" if accent else ""}"><div class="dne-label">{label}</div>'
            f'<div class="dne-value{" sm" if small else ""}">{value}</div>'
            + (f'<div class="dne-meta">{meta}</div>' if meta else "") + "</div>")


def card_row(cards):
    html(f'<div class="dne-grid" style="grid-template-columns: repeat({len(cards)}, minmax(0, 1fr))">'
         + "".join(cards) + "</div>")


def dot(colour):
    return f'<span class="dne-dot" style="background:{colour}"></span>'


def proportion_bar(parts):
    """parts: list of (name, count, colour). A thin 100% bar with a legend underneath."""
    total = sum(c for _, c, _ in parts)
    bar = "".join(f'<div style="flex:{c};background:{col}" title="{n}: {c:,}"></div>'
                  for n, c, col in parts)
    legend = "".join(f'<span>{dot(col)} <b style="color:{INK}">{n}</b> {c:,} '
                     f'<span style="color:{MUTED}">({c / total:.0%})</span></span>'
                     for n, c, col in parts)
    return f'<div class="dne-bar">{bar}</div><div class="dne-legend">{legend}</div>'


def footer():
    html('<div class="dne-footer">MUSAE GitHub Web &amp; ML developers (Rozemberczki, Allen and '
         'Sarkar). Fixed dataset and node IDs, no live GitHub lookup. Educational prototype, '
         'not a hiring or assessment tool.</div>')


def style_fig(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        hoverlabel=dict(bgcolor="#1f1a2b", bordercolor="#342c46", font=dict(family=FONT, color=INK)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=12, color=INK_2), title_text=""),
    )
    # Static view: dragging must not zoom or pan (there is no visible way to reset it).
    fig.update_layout(dragmode=False)
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     fixedrange=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED),
                     fixedrange=True)
    return fig


def show(fig):
    st.plotly_chart(fig, width="stretch",
                    config={"displayModeBar": False, "scrollZoom": False, "doubleClick": False})


def network_figure(graph, label_of, degree_of, centre=None, height=460):
    """Interactive node-link drawing, coloured by recorded label, with hover details."""
    pos = nx.spring_layout(graph, seed=SAMPLE_SEED)
    ex, ey = [], []
    for u, v in graph.edges:
        ex += [pos[u][0], pos[v][0], None]
        ey += [pos[u][1], pos[v][1], None]
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", hoverinfo="skip", showlegend=False,
                               line=dict(color=EDGE, width=0.8)))
    for cls, colour in CLASS_COLOURS.items():
        nodes = [v for v in graph.nodes if label_of[v] == cls and v != centre]
        fig.add_trace(go.Scatter(
            x=[pos[v][0] for v in nodes], y=[pos[v][1] for v in nodes], mode="markers", name=CLASS_LABELS[cls],
            marker=dict(size=10, color=colour, line=dict(color="#16121f", width=1.5)),
            customdata=[[v, degree_of(v)] for v in nodes],
            hovertemplate=f"<b>Developer %{{customdata[0]}}</b><br>Recorded label: {CLASS_LABELS[cls]}"
                          "<br>Connections: %{customdata[1]:,}<extra></extra>"))
    if centre is not None:
        fig.add_trace(go.Scatter(
            x=[pos[centre][0]], y=[pos[centre][1]], mode="markers+text", name="Selected developer",
            text=[f"<b>{centre}</b>"], textposition="top center", textfont=dict(color=INK, size=12),
            marker=dict(size=24, color=CLASS_COLOURS[label_of[centre]], line=dict(color=INK, width=2)),
            hovertemplate=f"<b>Developer {centre}</b><br>Recorded label: {CLASS_LABELS[label_of[centre]]}"
                          f"<br>Connections: {degree_of(centre):,}<extra></extra>"))
    style_fig(fig, height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig


# ---------------------------------------------------------------- view 1
def network_overview():
    metadata, _, preds, _ = load_tables()
    dst, start = load_neighbours()
    hero("MUSAE GitHub dataset", "Network overview",
         "A graph of GitHub developers linked by mutual follows. Each developer is recorded as a "
         "<b>Web</b> or <b>machine learning (ML)</b> developer. All figures below are read from the "
         "loaded data files.")

    card_row([
        card("Developers", f"{metadata['num_nodes']:,}", "nodes in the graph"),
        card("Connections", f"{metadata['num_undirected_edges']:,}", "mutual-follower edges"),
        card("Features", metadata["num_features"], "per developer, after SVD"),
        card("Avg. connections", f"{metadata['mean_degree']:.1f}", "per developer"),
    ])

    counts = metadata["class_counts"]
    left, right = st.columns([1, 1.15], gap="medium")
    with left, st.container(border=True):
        section("Class balance", "Share of developers by recorded label.")
        html(proportion_bar([(CLASS_LABELS[k], v, CLASS_COLOURS[k]) for k, v in
                             sorted(counts.items(), key=lambda kv: -kv[1])]))
        html(f'<div class="dne-note" style="margin-top:14px">The classes are imbalanced, so the '
             f'model comparison reports <b>macro-F1</b>, which weights both classes equally, '
             f'alongside accuracy.</div>')
    with right, st.container(border=True):
        section("Train / validation / test split",
                f"Stratified 60/20/20, seed {metadata['seed']}. Every model uses the same masks.")
        rows = "".join(
            f"<tr><td>{name}</td><td>{c['total']:,}</td><td>{c['web']:,}</td><td>{c['ml']:,}</td>"
            f"<td>{c['total'] / metadata['num_nodes']:.0%}</td></tr>"
            for name, c in metadata["split_counts"].items())
        html(f'<table class="dne-table"><tr><th>Split</th><th>Total</th>'
             f'<th>{dot(CLASS_COLOURS["web"])} Web</th><th>{dot(CLASS_COLOURS["ml"])} ML</th>'
             f'<th>Share</th></tr>{rows}</table>')

    with st.container(border=True):
        top_l, top_r = st.columns([3, 1.2])
        with top_l:
            section("A connected sample of the graph",
                    "Hover over a developer to see their label and number of connections.")
        with top_r:
            size = st.slider("Developers in sample", 20, 150, 60, step=10)
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
        true = preds.set_index("node_id")["true_label"]
        label_of = {v: class_name(true[v], metadata) for v in picked}
        show(network_figure(g, label_of, lambda v: start[v + 1] - start[v], height=520))
        html(f'<div class="dne-note">{len(picked)} of {metadata["num_nodes"]:,} developers, '
             f'reached by walking outward from developer {seed_node}. For illustration only.</div>')
    footer()


# ---------------------------------------------------------------- view 2
def developer_explorer():
    metadata, _, preds, _ = load_tables()
    dst, start = load_neighbours()
    hero("Held-out test developers", "Developer explorer",
         "Pick a developer whose label no model saw during training, and compare what each "
         "model predicted with the recorded label.")

    test = preds[preds["split"] == "test"].copy()
    available = [m for m in PRED_COLS if PRED_COLS[m] in test]
    if not available:
        st.warning("No model predictions found yet. Run the training scripts first.")
        return

    with st.container(border=True):
        c1, c2 = st.columns([2.2, 1], vertical_alignment="bottom")
        with c2:
            only_disagree = st.toggle("Only where models disagree")
        if only_disagree and len(available) > 1:
            cols = [PRED_COLS[m] for m in available]
            test = test[test[cols].nunique(axis=1) > 1]
        with c1:
            if test.empty:
                st.info("No test developers match this filter.")
                return
            node = st.selectbox(f"Test developer · {len(test):,} available",
                                test["node_id"].tolist())

    row = test.set_index("node_id").loc[node]
    truth = class_name(row["true_label"], metadata)
    guesses = {m: class_name(row[PRED_COLS[m]], metadata) for m in available}
    n_right = sum(g == truth for g in guesses.values())
    agree = len(set(guesses.values())) == 1

    html(f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:18px 0 12px 0">'
         f'<span class="dne-title" style="font-size:1.5rem">Developer {node}</span>'
         f'<span class="dne-pill">{dot(CLASS_COLOURS[truth])} Recorded label: <b>{CLASS_LABELS[truth]}</b></span>'
         f'<span class="dne-pill info">{n_right} of {len(guesses)} models correct</span>'
         f'<span class="dne-pill">{"Models agree" if agree else "Models disagree"}</span></div>')

    cards = [card("Recorded label", f'{dot(CLASS_COLOURS[truth])} {CLASS_LABELS[truth]}',
                  "from the dataset", small=True)]
    for m, guess in guesses.items():
        ok = guess == truth
        status = ('<span class="dne-pill good">✓ Correct</span>' if ok
                  else '<span class="dne-pill bad">✕ Incorrect</span>')
        cards.append(card(MODEL_NAMES[m], f'{dot(CLASS_COLOURS[guess])} {CLASS_LABELS[guess]}',
                          f'<div style="display:flex;justify-content:space-between;align-items:center;'
                          f'gap:8px;margin-top:4px"><span>{MODEL_USES[m]}</span>{status}</div>',
                          small=True))
    card_row(cards)

    # one-hop neighbourhood, capped
    nbrs = neighbours_of(node, dst, start)
    total = len(nbrs)
    shown = nbrs
    if total > MAX_NEIGHBOURS:
        rng = np.random.default_rng(SAMPLE_SEED + int(node))
        shown = np.sort(rng.choice(nbrs, MAX_NEIGHBOURS, replace=False))

    if total == 0:
        st.info("This developer has no connections.")
        footer()
        return
    all_labels = preds.set_index("node_id")["true_label"]
    nbr_classes = [class_name(all_labels[int(u)], metadata) for u in nbrs]
    n_ml = nbr_classes.count("ml")

    g = nx.Graph()
    g.add_node(int(node))
    g.add_edges_from((int(node), int(u)) for u in shown)
    label_of = {v: class_name(all_labels[v], metadata) for v in g.nodes}

    left, right = st.columns([2.2, 1], gap="medium")
    with left, st.container(border=True):
        section("One-hop neighbourhood",
                f"Showing a random sample of {MAX_NEIGHBOURS} of {total:,} neighbours."
                if total > MAX_NEIGHBOURS else "Every direct connection of this developer.")
        show(network_figure(g, label_of, lambda v: start[v + 1] - start[v], centre=int(node)))
    with right:
        html(card("Connections", f"{total:,}", "mutual followers"))
        html(f'<div class="dne-card" style="margin-top:14px"><div class="dne-label">Neighbour labels'
             f'</div><div class="dne-value">{n_ml / total:.0%} <span style="font-size:1rem;'
             f'font-weight:500;color:{INK_2}">ML</span></div>'
             + proportion_bar([("Web", total - n_ml, CLASS_COLOURS["web"]),
                               ("ML", n_ml, CLASS_COLOURS["ml"])]) + "</div>")
        html('<div class="dne-note" style="margin-top:14px">This picture is context, not an '
             'explanation of any prediction. The GCN uses the full graph, including neighbours '
             'of neighbours.</div>')
    footer()


# ---------------------------------------------------------------- view 3
def model_comparison():
    _, metrics, _, history = load_tables()
    hero("Test-set results", "Model comparison",
         "Scores on the held-out test developers from one fixed split and seed. They do not show "
         "that one model is better in general.")

    models = [m for m in MODEL_NAMES if m in metrics]
    if not models:
        st.warning("No metrics found yet. Run the training scripts first.")
        return

    best = max(models, key=lambda m: metrics[m]["test"]["macro_f1"])
    cards = []
    for m in models:
        t = metrics[m]["test"]
        tag = ' · <span class="dne-pill info" style="padding:1px 8px">Highest macro-F1</span>' \
            if m == best else ""
        cards.append(
            f'<div class="dne-card{" accent" if m == best else ""}">'
            f'<div class="dne-label">{dot(MODEL_COLOURS[m])} {MODEL_NAMES[m]}</div>'
            f'<div class="dne-meta" style="margin-top:2px">{MODEL_USES[m]}{tag}</div>'
            f'<div style="display:flex;gap:28px;margin-top:12px">'
            f'<div><div class="dne-label" style="text-transform:none">Macro-F1</div>'
            f'<div class="dne-value">{t["macro_f1"]:.3f}</div></div>'
            f'<div><div class="dne-label" style="text-transform:none">Accuracy</div>'
            f'<div class="dne-value" style="color:{INK_2}">{t["accuracy"]:.3f}</div></div></div>'
            f'<div class="dne-meta" style="margin-top:8px;color:{MUTED}">Chosen setting: '
            f'{metrics[m].get("chosen_config", "-")}</div></div>')
    card_row(cards)

    with st.container(border=True):
        h_l, h_r = st.columns([2, 1.3], vertical_alignment="bottom")
        with h_l:
            section("Per-class results", "Recall for ML is the share of ML developers a model found.")
        with h_r:
            metric = st.segmented_control("Metric", ["F1", "Precision", "Recall"], default="Recall",
                                          label_visibility="collapsed") or "Recall"
        rows = []
        for m in models:
            for cls, v in metrics[m]["test"]["per_class"].items():
                rows.append({"Model": MODEL_NAMES[m], "key": m, "Class": CLASS_LABELS[cls],
                             "Precision": v["precision"], "Recall": v["recall"], "F1": v["f1"],
                             "Support": v["support"]})
        per_class = pd.DataFrame(rows)
        fig = go.Figure()
        for m in models:
            d = per_class[per_class["key"] == m]
            fig.add_trace(go.Bar(
                x=d["Class"], y=d[metric], name=MODEL_NAMES[m], marker_color=MODEL_COLOURS[m],
                marker_line=dict(color="#16121f", width=2), marker_cornerradius=4,
                text=[f"{v:.2f}" for v in d[metric]], textposition="outside",
                textfont=dict(color=INK_2, size=11), cliponaxis=False,
                hovertemplate=f"<b>{MODEL_NAMES[m]}</b><br>Class %{{x}}<br>{metric}: "
                              "%{y:.3f}<extra></extra>"))
        style_fig(fig, 340)
        fig.update_layout(barmode="group", bargap=0.55, bargroupgap=0.12)
        fig.update_yaxes(range=[0, 1.08], tickformat=".1f")
        fig.update_xaxes(showgrid=False, tickfont=dict(color=INK, size=13))
        show(fig)
        with st.expander("Show full table"):
            st.dataframe(per_class.drop(columns="key").set_index(["Model", "Class"])
                         .style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                         width="stretch")

    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        h_l, h_r = st.columns([1, 1.6], vertical_alignment="bottom")
        with h_l:
            section("Confusion matrix")
        with h_r:
            m = st.segmented_control("Model", models, format_func=MODEL_NAMES.get, default=best,
                                     label_visibility="collapsed") or best
        cm = np.array(metrics[m]["test"]["confusion_matrix"])
        order = metrics[m]["test"]["class_order"]
        row_share = cm / cm.sum(axis=1, keepdims=True)
        text = [[f"<b>{cm[i, j]:,}</b><br>{row_share[i, j]:.0%} of true {CLASS_LABELS[order[i]]}"
                 for j in range(len(order))] for i in range(len(order))]
        fig = go.Figure(go.Heatmap(
            z=row_share, x=[f"Predicted {CLASS_LABELS[c]}" for c in order], y=[f"True {CLASS_LABELS[c]}" for c in order],
            colorscale=SEQ, zmin=0, zmax=1, showscale=False, xgap=3, ygap=3,
            text=text, texttemplate="%{text}", textfont=dict(size=13, color=INK),
            customdata=cm, hovertemplate="%{y} → %{x}<br>%{customdata:,} developers<extra></extra>"))
        style_fig(fig, 320)
        fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(color=INK, size=12))
        fig.update_xaxes(showgrid=False, side="top", tickfont=dict(color=INK, size=12))
        show(fig)
        html('<div class="dne-note">Shading shows the share of each true class. '
             'The diagonal is correct predictions.</div>')

    with right, st.container(border=True):
        section("Neural training curves",
                "MLP and GCN share every setting. The only difference is that the GCN uses edges.")
        if history.empty:
            st.info("history.csv not found. Run train_neural.py.")
        else:
            curve = st.segmented_control("Curve", ["Validation macro-F1", "Training loss"],
                                         default="Validation macro-F1",
                                         label_visibility="collapsed") or "Validation macro-F1"
            col = "val_macro_f1" if curve == "Validation macro-F1" else "train_loss"
            fig = go.Figure()
            for key in [k for k in ("mlp", "gcn") if k in set(history["model"])]:
                h = history[history["model"] == key]
                fig.add_trace(go.Scatter(
                    x=h["epoch"], y=h[col], mode="lines", name=MODEL_NAMES[key],
                    line=dict(color=MODEL_COLOURS[key], width=2),
                    hovertemplate=f"<b>{MODEL_NAMES[key]}</b> · epoch %{{x}}<br>{curve}: "
                                  "%{y:.3f}<extra></extra>"))
                if col == "val_macro_f1":
                    b = h.loc[h[col].idxmax()]
                    fig.add_trace(go.Scatter(
                        x=[b["epoch"]], y=[b[col]], mode="markers", showlegend=False,
                        marker=dict(size=9, color=MODEL_COLOURS[key], line=dict(color="#16121f", width=2)),
                        hovertemplate=f"<b>{MODEL_NAMES[key]} best epoch</b> %{{x}}<br>"
                                      "Validation macro-F1: %{y:.3f}<extra></extra>"))
            style_fig(fig, 262)
            fig.update_layout(hovermode="x unified")
            fig.update_xaxes(title_text="Epoch", title_font=dict(color=MUTED, size=11))
            show(fig)
            html('<div class="dne-note">Each model keeps the epoch with the best validation '
                 'macro-F1 (marked), then stops after 20 epochs without improvement.</div>')
    footer()


# ---------------------------------------------------------------- page
def main():
    html(CSS)
    needed = ["metadata.json", "predictions.csv", "metrics.json", "graph_data.pt"]
    missing = [f for f in needed if not (ART_DIR / f).exists()]
    if missing:
        st.error("Missing files in artifacts/: " + ", ".join(missing)
                 + ". Run data.py, train_baseline.py and train_neural.py first.")
        st.stop()

    page = st.navigation([
        st.Page(network_overview, title="Network overview", icon=":material/hub:", default=True),
        st.Page(developer_explorer, title="Developer explorer", icon=":material/person_search:"),
        st.Page(model_comparison, title="Model comparison", icon=":material/insights:"),
    ], position="top")
    page.run()


main()
