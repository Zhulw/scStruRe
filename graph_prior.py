from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
from scipy import sparse


GRAPH_NAMES = ("hvg", "pca", "diff", "pf")
GRAPH_FILES = {
    "hvg": "G_hvg.npz",
    "pca": "G_pca.npz",
    "diff": "G_diff.npz",
    "pf": "G_pf.npz",
}


@dataclass
class BatchGraphPrior:
    B_local: torch.Tensor
    support: torch.Tensor
    srs: torch.Tensor
    availability: torch.Tensor
    beta_scale: torch.Tensor
    fn_risk: torch.Tensor

    @property
    def cov(self) -> torch.Tensor:
        """Backward-compatible alias for the former coverage field."""
        return self.availability


def _load_graph(path: str) -> sparse.csr_matrix:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing graph file: {path}")

    graph = sparse.load_npz(path).tocsr().astype(np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()
    return graph


def build_rank_bias(graph: sparse.csr_matrix, eps: float = 0.1) -> sparse.csr_matrix:
    graph = graph.tocsr().astype(np.float32)
    graph.setdiag(0)
    graph.eliminate_zeros()

    indptr = graph.indptr
    indices = graph.indices
    data = graph.data
    out = np.zeros_like(data, dtype=np.float32)

    for row in range(graph.shape[0]):
        start, end = indptr[row], indptr[row + 1]
        count = end - start
        if count == 0:
            continue
        if count == 1:
            out[start:end] = 1.0
            continue

        weights = data[start:end]
        if np.allclose(weights, weights[0]):
            out[start:end] = 1.0
            continue

        order = np.argsort(-weights, kind="mergesort")
        rank = np.empty(count, dtype=np.float32)
        rank[order] = np.arange(1, count + 1, dtype=np.float32)
        out[start:end] = eps + (1.0 - eps) * (count - rank) / max(count - 1, 1)

    rank_graph = sparse.csr_matrix(
        (out, indices.copy(), indptr.copy()),
        shape=graph.shape,
    )
    rank_graph.setdiag(0)
    rank_graph.eliminate_zeros()
    return rank_graph


class LocalGraphPrior:
    """Build batch-local priors from four global sparse graph sources."""

    def __init__(
        self,
        graph_dir: str,
        n_cells: int,
        rank_eps: float = 0.1,
        availability_support: int = 2,
        beta_base: float = 0.2,
    ):
        self.graph_dir = graph_dir
        self.n_cells = int(n_cells)
        self.rank_eps = float(rank_eps)
        self.availability_support = int(availability_support)
        self.beta_base = float(beta_base)

        if self.availability_support < 1 or self.availability_support > len(GRAPH_NAMES):
            raise ValueError("availability_support must be between 1 and 4")
        if not 0.0 <= self.beta_base <= 1.0:
            raise ValueError("beta_base must be in [0, 1]")

        self.graphs: Dict[str, sparse.csr_matrix] = {}
        self._load_graphs()
        self.degree, self.specificity_max = self._build_union_statistics()

    def _load_graphs(self) -> None:
        for name in GRAPH_NAMES:
            path = os.path.join(self.graph_dir, GRAPH_FILES[name])
            graph = build_rank_bias(_load_graph(path), eps=self.rank_eps)
            if graph.shape != (self.n_cells, self.n_cells):
                raise ValueError(
                    f"Graph {path} shape {graph.shape} does not match "
                    f"n_cells={self.n_cells}"
                )
            self.graphs[name] = graph

    def _presence_graph(self, graph: sparse.csr_matrix) -> sparse.csr_matrix:
        present = graph.maximum(graph.T).tocsr()
        present.data = np.ones_like(present.data, dtype=np.int16)
        present.setdiag(0)
        present.eliminate_zeros()
        return present.astype(np.int16)

    def _build_union_statistics(self):
        """Compute global union degree and the fixed specificity normalizer."""
        support = sparse.csr_matrix((self.n_cells, self.n_cells), dtype=np.int16)
        for name in GRAPH_NAMES:
            support = support + self._presence_graph(self.graphs[name])

        support.setdiag(0)
        support.eliminate_zeros()
        union = (support > 0).tocsr()
        degree = np.asarray(union.sum(axis=1)).ravel().astype(np.float32)

        rows, cols = union.nonzero()
        if rows.size == 0:
            raise ValueError("The multi-source union graph contains no edges")

        degree_pair = 0.5 * (degree[rows] + degree[cols])
        edge_specificity = 1.0 / np.log1p(degree_pair + 1.0)
        specificity_max = float(edge_specificity.max())
        if not np.isfinite(specificity_max) or specificity_max <= 0.0:
            raise ValueError("Invalid global specificity normalizer")

        return degree, specificity_max

    def get_batch(self, batch_idx, device: torch.device) -> BatchGraphPrior:
        idx = self._to_numpy_index(batch_idx)
        B_local = self._slice_graphs(idx, device)
        support, srs, availability = self._build_srs(B_local, idx, device)

        bsz = B_local.shape[1]
        diag = torch.arange(bsz, device=device)
        B_local[:, diag, diag] = 0.0
        support[diag, diag] = 0.0
        srs[diag, diag] = 0.0

        beta_scale = availability.view(-1, 1) * (
            self.beta_base + (1.0 - self.beta_base) * srs
        )
        beta_scale = torch.where(support > 0, beta_scale, torch.zeros_like(beta_scale))

        pair_availability = 0.5 * (
            availability.view(-1, 1) + availability.view(1, -1)
        )
        fn_risk = pair_availability * srs
        fn_risk = torch.where(support > 0, fn_risk, torch.zeros_like(fn_risk))
        beta_scale[diag, diag] = 0.0
        fn_risk[diag, diag] = 0.0

        return BatchGraphPrior(
            B_local=B_local,
            support=support.detach(),
            srs=srs.detach(),
            availability=availability.detach(),
            beta_scale=beta_scale.detach(),
            fn_risk=fn_risk.detach(),
        )

    def _to_numpy_index(self, batch_idx) -> np.ndarray:
        if isinstance(batch_idx, torch.Tensor):
            return batch_idx.detach().cpu().numpy().astype(np.int64)
        return np.asarray(batch_idx, dtype=np.int64)

    def _slice_graphs(self, idx: np.ndarray, device: torch.device) -> torch.Tensor:
        mats = []
        for name in GRAPH_NAMES:
            mat = self.graphs[name][idx][:, idx].toarray().astype(np.float32)
            np.fill_diagonal(mat, 0.0)
            mats.append(mat)

        return torch.from_numpy(np.stack(mats, axis=0)).to(
            device=device,
            dtype=torch.float32,
        )

    def _build_srs(
        self,
        B_local: torch.Tensor,
        idx: np.ndarray,
        device: torch.device,
    ):
        B_sym = 0.5 * (B_local + B_local.transpose(1, 2))
        present = ((B_local > 0) | (B_local.transpose(1, 2) > 0)).float()
        support = present.sum(dim=0)

        present_conf = (B_sym * present).sum(dim=0) / support.clamp_min(1.0)
        support_ratio = support / float(len(GRAPH_NAMES))

        degree = torch.from_numpy(self.degree[idx]).to(device=device, dtype=torch.float32)
        degree_pair = 0.5 * (degree.view(-1, 1) + degree.view(1, -1))
        specificity_raw = 1.0 / torch.log1p(degree_pair + 1.0)
        specificity = (specificity_raw / self.specificity_max).clamp(0.0, 1.0)

        # Fixed global edge reliability:
        # cross-source support × rank confidence × softened degree specificity.
        srs = support_ratio * present_conf * torch.sqrt(specificity)
        srs = torch.where(support > 0, srs, torch.zeros_like(srs))

        reliable = support >= float(self.availability_support)
        availability = (reliable.sum(dim=1) > 0).float()

        return support, srs, availability