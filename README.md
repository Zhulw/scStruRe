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
