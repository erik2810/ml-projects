# Graph ML Lab

From-scratch implementations of graph machine learning models using **only PyTorch** — no NumPy, no PyG, no DGL. All message passing, aggregation, and graph operations are built on raw tensors.

This project spans two levels of complexity:

1. **Foundational graph ML** — GNNs, conditional graph generation, and graph VAE/diffusion on small social network skeletons
2. **Spatial graph generation** — joint discrete structure + continuous 3D position generation for tree morphologies (botanical trees, neuron-like structures), with research-grade evaluation

The spatial component addresses a core research challenge: generating graphs embedded in 3D space where **proximity does not imply connectivity** — unlike molecules, the topology of neuronal arbors and botanical trees is decoupled from their geometry.


## Project overview

| Module | Description |
|--------|-------------|
| **GNN** | Message passing layers (GCN, GAT) for node and graph classification |
| **Generator** | Conditional VAE that generates small graphs matching target structural properties |
| **Graph VAE + Diffusion** | Variational autoencoder and discrete denoising diffusion on social network skeletons |
| **Spatial Tree VAE** | Autoregressive generator for 3D trees — joint parent selection + position prediction |
| **Spatial Diffusion** | Joint discrete-continuous diffusion over graph structure and 3D node positions |
| **Benchmarks** | Evaluation pipeline with morphological metrics, Sholl analysis, spatial MMD |

Includes a FastAPI backend, browser-based frontend with interactive 3D tree visualization, and a [static demo](demo/) for GitHub Pages.


## Quickstart

```bash
git clone https://github.com/erik2810/ml-projects.git
cd ml-projects
./setup.sh          # macOS / Linux
# or setup.bat      # Windows

./run.sh            # starts on http://localhost:8000
```

The web UI has four panels: GNN Explorer, Graph Generator, VAE/Diffusion, and **Spatial 3D** (the research-level component).


## Project structure

```
core/                        # research framework (Lightning-ish, no Lightning dep)
  models/base.py             # BaseModel: training_step, validation_step, configure_optimizers
  data/base.py               # BaseDataset: train_loader, val_loader, test_loader
  training/trainer.py        # Trainer with callback hooks
  training/callbacks.py      # Callback, EarlyStopping, Checkpoint, ProgressLogger
  config/schema.py           # Pydantic ExperimentConfig / ModelConfig / DatasetConfig
  config/loader.py           # YAML (optional) / JSON experiment loader
  registry.py                # @register decorator for models and datasets
  cli.py                     # `python -m core.cli train|list|scaffold`

algorithms/                  # framework-native model + dataset pairs
  gnn/                       # GCN / GAT on Karate Club (node classification)
  graphvae/                  # Graph VAE + discrete diffusion
  generator/                 # Conditional graph generator (CVAE)
  hyperbolic/                # Hyperbolic GNN + link prediction (Riemannian + Euclidean opts)
  physics_gnn/               # Physics-informed GNN on a spring mesh

experiments/configs/         # YAML experiment definitions (karate_gcn.yaml, tree_hyperbolic.yaml, ...)

backend/                     # original, still-live implementations + FastAPI server
  core/                      # graph_utils, gnn, generator, graphvae, spatial, benchmarks
  api/                       # FastAPI route handlers (incl. spatial endpoints)
  app.py                     # FastAPI entry point

frontend/                    # browser UI (ES modules under static/js/components/)
demo/                        # static GitHub Pages demo
scripts/                     # standalone training scripts
tests/                       # pytest suite (160+ tests incl. core framework, algorithms, spatial)
```


## Framework at a glance

A small, Lightning-inspired layer that standardises training across the
algorithms above. Define a model by subclassing `BaseModel` and implementing
`training_step` (plus optional `validation_step` / `configure_optimizers`);
pair it with a `BaseDataset`; declare the experiment in YAML:

```yaml
# experiments/configs/karate_gcn.yaml
name: karate_gcn
model:
  name: gcn_node
  params: { in_features: 18, num_classes: 2, hidden: 16, n_layers: 2, lr: 0.01 }
dataset:
  name: karate_club
  params: { per_class: 4, seed: 0 }
training:
  max_epochs: 80
  early_stopping: { monitor: val_loss, patience: 15, mode: min }
```

Run via the CLI:

```bash
python -m core.cli list                                       # list registered models/datasets
python -m core.cli train experiments/configs/karate_gcn.yaml  # train + checkpoint
python -m core.cli scaffold my_model                          # emit a starter template
```

Multi-optimizer support (e.g. `[RiemannianAdam, AdamW]` for hyperbolic
models with mixed manifold/Euclidean parameters) is built in.


## The models

### 1. GNN from scratch

Full message passing pipeline without library abstractions:

- **GCN layer** — symmetric normalization + linear transform (Kipf & Welling, ICLR 2017)
- **GAT layer** — multi-head attention with additive decomposition (Velickovic et al., ICLR 2018)
- Evaluated on Zachary's Karate Club (semi-supervised node classification)

### 2. Conditional graph generator

A conditional VAE that learns to generate adjacency matrices given target structural properties:

- Conditions: normalized node count, edge density, clustering coefficient
- Training data: Erdos-Renyi and Barabasi-Albert random graph mixtures
- KL annealing to avoid posterior collapse (Bowman et al., CoNLL 2016)

### 3. Graph VAE + Diffusion

Two generative approaches on small graph distributions:

- **Graph VAE**: GCN encoder, inner-product decoder (Kipf & Welling, 2016; Simonovsky & Komodakis, 2018)
- **Discrete diffusion**: edge-flip noise + GCN denoiser (Austin et al., NeurIPS 2021)

### 4. Spatial tree VAE (research level)

Autoregressive generation of trees embedded in R^3. At each step the model decides:
1. **Which node to attach to** — discrete attention over existing nodes
2. **Where to place the child** — continuous Gaussian prediction of 3D offset
3. **Whether to stop** — Bernoulli gate

Architecture: position-aware GNN encoder to latent space, GRU-based autoregressive decoder with attention. Trained with KL warmup and teacher forcing decay.

This addresses the core challenge of combining discrete structural decisions with continuous geometric predictions — the topology and geometry are learned jointly but can express independence (proximity ≠ connectivity).

Draws from: GraphRNN (You et al., ICML 2018), Graph RNN with attention (Liao et al., NeurIPS 2019), extended to 3D positions with tree-specific structure.

### 5. Joint discrete-continuous diffusion (research level)

Simultaneous denoising of graph structure (edge bit-flips) and node positions (Gaussian noise):

- **Position process**: standard DDPM with cosine schedule (Nichol & Dhariwal, 2021)
- **Edge process**: independent bit-flip corruption (Austin et al., NeurIPS 2021)
- **Shared backbone**: position-aware GNN with sinusoidal time embedding

The denoiser uses relative 3D positions as edge features during message passing, allowing the model to learn the coupling between spatial layout and connectivity patterns. After generation, a spanning tree is extracted via BFS.

Related to DiGress (Vignac et al., ICLR 2023) but extended to jointly handle continuous 3D positions.


## Evaluation metrics

Standard graph generation metrics (novelty, uniqueness) are insufficient for spatial graphs. We implement:

| Metric | What it captures |
|--------|-----------------|
| **Spatial MMD** | Maximum Mean Discrepancy on morphological feature vectors (RBF kernel) |
| **Tree Edit Distance** | Joint structural + positional distance (approximate Zhang-Shasha) |
| **Sholl analysis** | Radial intersection profiles (Sholl, 1953) |
| **Strahler numbers** | Branching complexity ordering |
| **Segment length W1** | Wasserstein-1 distance on edge length distributions |
| **Branch angle W1** | Wasserstein-1 distance on branch angle distributions |
| **Morphological features** | 12-dim vector: branch points, tips, tortuosity, extent, etc. |

The `run_benchmark()` function provides a unified pipeline for training, generating, and evaluating models on synthetic datasets.


## Synthetic data

Two regimes for generating training data:

- **Branching trees** — stochastic growth process with directional persistence, random branching, and configurable gravitropism
- **Neuron morphologies** — biologically-inspired structure with soma, basal/apical dendrites, radial organization, and tapering (loosely inspired by Cuntz et al., PLoS Comp Bio 2010)

Both produce `SpatialGraph` objects with 3D positions, parent arrays, and optional radii/type annotations. SWC format I/O is supported for integration with neuroscience tools.


## Training scripts

```bash
python scripts/train_gnn.py
python scripts/train_generator.py
python scripts/train_vae.py
```

For spatial models, use the API or run benchmarks programmatically:

```python
from backend.core.benchmarks import run_baseline_comparison, DataConfig

results = run_baseline_comparison(
    data_config=DataConfig(regime='neuron', num_train=200, num_test=50),
)
```


## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers all modules including the core framework (Trainer, callbacks,
registry, CLI), framework-native algorithm wrappers, spatial graph
operations, synthetic generators, metrics, VAE forward/generate, and
diffusion forward/sample.

## Development

CI runs pytest across Python 3.10–3.12 and ruff on `core/`, `algorithms/`,
`experiments/`. To get the same checks locally:

```bash
pip install -e ".[dev]"
pip install pre-commit
pre-commit install
```


## Technical notes

- **No NumPy**: all tensor operations use `torch` exclusively
- **Dense adjacency**: `(N, N)` matrices rather than sparse COO — simple and readable for N < 100
- **Reproducibility**: `torch.manual_seed()` for deterministic results
- **SWC format**: standard neuroscience file format (Cannon et al., 1998) supported for import/export


## References

- Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017.
- Velickovic et al. *Graph Attention Networks.* ICLR 2018.
- Kipf & Welling. *Variational Graph Auto-Encoders.* NeurIPS Workshop 2016.
- Simonovsky & Komodakis. *GraphVAE: Towards Generation of Small Graphs Using Variational Autoencoders.* ICANN 2018.
- You et al. *GraphRNN: Generating Realistic Graphs with an Auto-Regressive Model.* ICML 2018.
- Liao et al. *Efficient Graph Generation with Graph Recurrent Attention Networks.* NeurIPS 2019.
- Austin et al. *Structured Denoising Diffusion Models in Discrete State-Spaces.* NeurIPS 2021.
- Vignac et al. *DiGress: Discrete Denoising Diffusion for Graph Generation.* ICLR 2023.
- Nichol & Dhariwal. *Improved Denoising Diffusion Probabilistic Models.* ICML 2021.
- Cuntz et al. *One Rule to Grow Them All.* PLoS Computational Biology, 2010.
- Kanari et al. *A Topological Representation of Branching Neuronal Morphologies.* Neuroinformatics, 2018.
- Sholl. *Dendritic Organization in the Neurons of the Visual and Motor Cortices of the Cat.* J. Anat., 1953.
- Zhang & Shasha. *Simple Fast Algorithms for the Editing Distance between Trees.* SIAM J. Comput., 1989.
- Cannon et al. *An On-line Archive of Reconstructed Hippocampal Neurons.* J. Neurosci. Methods, 1998.
- Bowman et al. *Generating Sentences from a Continuous Space.* CoNLL 2016.


## License

MIT
