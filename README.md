# scStruRe

**Structural Reliability-Guided Graph Contrastive Learning for Single-Cell RNA-seq Clustering**

scStruRe is a graph contrastive clustering framework for single-cell RNA-seq
data. It estimates edge-level structural reliability from complementary cell
graphs and uses this evidence in relational encoding, topology consistency, and
false-negative suppression.

This repository contains the research code accompanying the manuscript:
"Structural Reliability-Guided Graph Contrastive Learning for Single-Cell
RNA-seq Clustering."

## Method overview

scStruRe contains four main stages:

1. **Multi-source graph construction** in HVG, PCA, diffusion, and de novo
   gene-program activity spaces.
2. **Structural reliability scoring (SRS)** using cross-source support,
   within-graph rank confidence, and structural specificity.
3. **Reliability-aware learning** through SRS-RCB relational encoding,
   SRS-TC assignment consistency, and SRS-BFN false-negative suppression.
4. **Clustering** of the learned all-cell latent representation with K-means.

## Repository structure

```text
scStruRe/
├── main.py                 # training, inference, and evaluation
├── scTGCL.py               # backbone and contrastive model components
├── graph_prior.py          # graph loading and SRS construction
├── bfn.py                  # SRS-BFN candidate selection and weighting
├── build_graph.py          # multi-source graph construction
├── config.py               # default model and training settings
├── utils.py                # data, reproducibility, and clustering utilities
├── requirements.txt
├── CITATION.cff
├── data/README.md
└── scripts/run_three_seeds.sh
```

Datasets, generated graph files, model checkpoints, and experiment results are
not committed to the repository.

## Environment

Python 3.10 is recommended. The manuscript experiments used PyTorch 2.1.2,
CUDA 12.1, and an NVIDIA A100 GPU with 40 GB memory.

Create an isolated environment and install the dependencies:

```bash
conda create -n scstrure python=3.10 -y
conda activate scstrure
pip install -r requirements.txt
```

For a CUDA-specific PyTorch installation, follow the installation command that
matches the CUDA driver on your machine, then install the remaining packages.
CPU execution is supported by the code but is not recommended for full-scale
experiments.

## Data preparation

Place a standardized H5AD file in `data/`. The default evaluation label key is
`cell_type`. See [`data/README.md`](data/README.md) for the complete input
requirements.

The paper uses 2,500 HVGs for the model input and four graph sources with
`k=15`. Construct the graphs before training:

```bash
python build_graph.py \
  --h5ad data/zeisel.h5ad \
  --name Zeisel \
  --label_key cell_type \
  --n_top_genes 2500 \
  --n_pca 50 \
  --dm_dim 30 \
  --k 15 \
  --n_modules 120 \
  --auc_top_frac 0.05
```

This creates the runtime graph files under `data/Zeisel/graph/`:

```text
G_hvg.npz
G_pca.npz
G_diff.npz
G_pf.npz
```

`G_diff.npz` is the expression-guarded diffusion graph used by the released
implementation. Auxiliary graph and diagnostic files are also written to the
same directory.

## Train and evaluate

Run one Zeisel experiment with the paper settings:

```bash
python main.py \
  --dataset data/zeisel.h5ad \
  --n_clusters 9 \
  --graph_dir data/Zeisel/graph \
  --label_key cell_type \
  --epochs 160 \
  --batch_size 128 \
  --seed 42 \
  --select_metric final \
  --save_dir results/Zeisel/seed_42
```

The paper protocol uses the final-epoch model and independent seeds 42, 43, and
44. Run all three seeds with:

```bash
bash scripts/run_three_seeds.sh \
  data/zeisel.h5ad \
  9 \
  data/Zeisel/graph \
  cell_type \
  results/Zeisel
```

The prespecified cluster counts reported in the current manuscript are:

| Dataset | Clusters | Dataset | Clusters |
| --- | ---: | --- | ---: |
| PBMC | 8 | Muraro | 9 |
| Baron | 14 | Bach | 8 |
| Camp | 7 | Pollen | 5 |
| Zeisel | 9 | Klein | 4 |
| Shekhar | 19 | Diaphragm | 5 |

## Outputs

Each result directory contains:

- `model_final.pt`: final-epoch model checkpoint;
- `latent.npy`: all-cell latent representation;
- `latent_metadata.csv`: cell IDs, reference labels, and predicted clusters;
- `scStruRe_cluster_labels.csv`: K-means cluster assignments;
- `source_weights.csv`: learned prior-head graph-source weights.

ACC, NMI, and ARI are printed after clustering. The current script performs
single-seed evaluation; compute the arithmetic mean of the three independent
runs for paper-level reporting.

## Reproducibility notes

- The four graph files and the H5AD matrix must preserve exactly the same cell
  order.
- The paper protocol uses `--select_metric final`; no reference-label-based
  checkpoint selection is performed in this mode.
- `--n_clusters` must be provided explicitly and is used by both the auxiliary
  assignment head and final K-means.
- The released training code retains the stratified 80/20 split used in the
  experiments. Final clustering is evaluated on all cells by default
  (`--eval_scope full`).
- Stochastic GPU operations can still show small platform-dependent variation
  despite fixed random seeds.

## Citation

If you use this code, please cite the accompanying manuscript. Citation metadata
are provided in [`CITATION.cff`](CITATION.cff). The bibliographic entry will be
updated after publication.

## Contact

For questions about the code or experiments, please open a GitHub issue.
