# Developer Network Explorer

## Project overview

Developer Network Explorer is an educational machine learning project that explores a GitHub developer network and compares three approaches to node classification: Logistic Regression, a Multilayer Perceptron (MLP), and a Graph Convolutional Network (GCN).

The planned Streamlit application will let users explore network statistics, select a developer from the test set, inspect a sample of their connections, and compare model predictions with the recorded dataset label.

The project combines a reproducible training pipeline with an interactive application for presenting and understanding the results.

## Project status

**Stage: Initial setup and development.**

The project scope, model choices, application views, and team responsibilities have been planned. Implementation and evaluation are pending; no completed models or measured results are claimed yet.

Update this checklist as each milestone is verified:

- [x] Define project scope and team responsibilities.
- [ ] Set up the shared repository and Python environment.
- [ ] Load and verify the dataset.
- [ ] Save the shared train/validation/test split.
- [ ] Train Logistic Regression.
- [ ] Train MLP and GCN.
- [ ] Evaluate finalized models and save predictions.
- [ ] Complete the three Streamlit views.
- [ ] Verify setup and demo on the presentation machine.
- [ ] Add screenshots, report, and presentation.

**Last progress update:** TODO: Add date and a short summary.

## Objective

Compare classification based on a developer's own attributes with classification that also uses graph connections.

The project aims to:

- Implement Logistic Regression and MLP as feature-only baselines.
- Implement a two-layer GCN using node features and network connections.
- Evaluate all three models on the same held-out test nodes.
- Present predictions and model performance through an interactive interface.

A successful project means a working application and a fair comparison. The GCN is not assumed to outperform the other models.

## Application features

| Planned view | What users will see |
| --- | --- |
| Network overview | Dataset size, class distribution, and a small graph sample |
| Developer explorer | A selected test node, a sample of its neighborhood, predictions from all three models, and its recorded label |
| Model comparison | Accuracy, macro-F1, confusion matrices, and neural training curves |

Training will happen beforehand. The application will load saved predictions and results for a responsive demonstration.

**Screenshots:** TODO: Add screenshots after the views are implemented.

## Dataset

**Source:** [GitHub Social Network — Stanford SNAP](https://snap.stanford.edu/data/github-social.html).

The source describes a network collected in June 2019:

| Component | Meaning |
| --- | --- |
| Nodes | GitHub developers who starred at least 10 repositories |
| Edges | Mutual follower relationships |
| Features | Attributes derived from location, starred repositories, employer, and email |
| Labels | Web developer or machine learning developer, derived from job titles |
| Source-reported size | 37,700 nodes and 289,003 undirected edges |

These are source statistics. Record the actual loaded dimensions below; graph libraries may store an undirected edge in both directions.

### Download and preparation

1. Open the SNAP source page linked above.
2. Download the GitHub Social Network archive listed under **Files**.
3. Keep downloaded data under the local `data/` directory, excluded from Git.
4. Use the project's data loader once implemented to verify node alignment, construct features, and create the shared split.

If the team uses the PyTorch Geometric loader instead, replace these steps with the tested loader command and document its data source.

| Detail to record after loading | Value |
| --- | --- |
| Download method and access date | TODO |
| Expected local file paths | TODO |
| Loaded node count and edge representation | TODO |
| Feature dimensions and preprocessing | TODO |
| Label encoding: numeric value → category | TODO |
| Class counts | TODO |
| Split seed and saved split path | TODO |

## Models

| Model | Input | Role |
| --- | --- | --- |
| Logistic Regression | Node features | Traditional ML baseline |
| MLP | Node features | Neural baseline without connections |
| Two-layer GCN | Node features and graph edges | Graph-based classifier |

**Final configurations:** TODO: Record hyperparameters, neural hidden dimensions, dropout, optimizer, learning rate, training epochs, and checkpoint selection rule after validation.

## Workflow

1. **Load and inspect data:** Check features, labels, edges, and node-ID alignment.
2. **Create a shared split:** Save stratified 60% training, 20% validation, and 20% test masks.
3. **Train models:** Fit all three models using training labels.
4. **Validate:** Use validation results to select settings and neural checkpoints.
5. **Evaluate:** Freeze choices, evaluate on test nodes, and save metrics and predictions.
6. **Display results:** Load saved outputs into the Streamlit application.
7. **Prepare the demo:** Verify setup instructions and complete the report and slides.

## Technology stack

| Technology | Purpose |
| --- | --- |
| Python | Main programming language |
| NumPy and pandas | Numerical operations and result tables |
| scikit-learn | Logistic Regression, splitting, and evaluation metrics |
| PyTorch | MLP, neural training, and checkpoints |
| PyTorch Geometric (PyG) | Graph data representation and GCN layers |
| Streamlit | Interactive application |
| Matplotlib | Training curves and evaluation figures |
| NetworkX | Small graph samples and network visualization |
| Git and GitHub | Version control and team collaboration |

**Tested versions:** TODO: Record Python and dependency versions in `requirements.txt` after verifying the environment.

## Repository structure

The following is the **planned structure**. Update paths as implementation progresses.

| Path | Purpose |
| --- | --- |
| `README.md` | Project description, setup, usage, and results |
| `.gitignore` | Exclude environments, caches, secrets, data, and generated artifacts |
| `requirements.txt` | Tested Python dependencies |
| `app.py` | Streamlit entry point |
| `src/data.py` | Dataset loading, validation, and shared split |
| `src/train_baseline.py` | Logistic Regression training |
| `src/models.py` | MLP and GCN definitions |
| `src/train_neural.py` | Neural training and checkpoint selection |
| `src/evaluate.py` | Shared metrics and prediction export |
| `data/` | Local dataset files; excluded from Git |
| `artifacts/` | Saved splits, models, and predictions; excluded from Git initially |
| `results/` | Small metrics tables, figures, and training records |
| `docs/` | Synopsis, implementation guide, report, slides, and screenshots |

Because `data/` and `artifacts/` are excluded from Git, setup instructions must explain how to download or regenerate everything the application needs.

## Evaluation

All models will use the same stratified node split:

| Split | Percentage | Purpose |
| --- | --- | --- |
| Training | 60% | Learn model parameters |
| Validation | 20% | Select settings and checkpoints |
| Test | 20% | Measure final performance |

Evaluation rules:

- Only training-node labels contribute to model fitting or neural training loss.
- Fit learned preprocessing on training nodes.
- Use validation results for model selection; reserve test labels for final evaluation.
- Report accuracy, macro-F1, per-class precision/recall/F1, and confusion matrices.
- Use the same split and label mapping for every model.
- Document the random seed and final configurations.

The GCN uses a **transductive setup**: all node features and graph connections are available during training, while validation and test labels are excluded from the training loss.

### Results

No test results are available yet. Replace the placeholders with measured values and state the score scale used.

| Model | Test accuracy | Test macro-F1 |
| --- | --- | --- |
| Logistic Regression | Pending | Pending |
| MLP | Pending | Pending |
| GCN | Pending | Pending |

**Figures:** TODO: Link confusion matrices and neural training curves.

**Interpretation:** TODO: Explain whether graph connections helped in this experiment, where models made errors, and any limitations of the comparison.

## Team responsibilities

| Member | Primary responsibility |
| --- | --- |
| Shreyas | Dataset loading and validation, shared split, and Logistic Regression |
| Medhavi | MLP and GCN implementation, neural training, and saved checkpoints |
| Abhyuday | Shared evaluation, Streamlit application, integration, and presentation assembly |
| Everyone | Documentation for their component, code review, and demonstration rehearsal |

Agree on the data interface before integrating: node features, edge indices, labels, node IDs, and shared split masks. Export predictions with node IDs so the application displays the correct results.

## Limitations

- The application uses a fixed historical dataset and does not analyze arbitrary live GitHub usernames.
- The two dataset categories simplify developers' interests and are not a comprehensive description of their skills.
- This is an educational prototype, not a hiring or professional assessment tool.
- The transductive evaluation does not establish performance on entirely unseen networks.
- A displayed neighborhood is a small visual sample; GCN predictions use the graph supplied to the model.
- A single split and limited tuning cannot establish broad superiority of one model.
- Research extensions such as GraphSAGE, multiple datasets, and robustness experiments are outside the initial one-week scope.

## References

- [Stanford SNAP: GitHub Social Network](https://snap.stanford.edu/data/github-social.html)
- [Rozemberczki, Allen, and Sarkar: Multi-scale Attributed Node Embedding](https://arxiv.org/abs/1909.13021)
- [Kipf and Welling: Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- [PyTorch documentation](https://pytorch.org/docs/stable/)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
- [Streamlit documentation](https://docs.streamlit.io/)

TODO: Add any additional sources used during implementation.
