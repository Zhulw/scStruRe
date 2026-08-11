

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse import csr_matrix, save_npz
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


# -------------------------
# 基础工具
# -------------------------

def to_csr(x) -> csr_matrix:
    return x.tocsr() if sp.issparse(x) else csr_matrix(x)


def log1p_sparse(x: csr_matrix) -> csr_matrix:
    x = x.copy().tocsr().astype(np.float32)
    x.data = np.log1p(x.data)
    return x


def clean_graph(graph: sp.spmatrix, mode: str = "max") -> csr_matrix:
    """候选图清洗：CSR、float32、去自环、对称化、去零；不做行归一化。"""
    graph = graph.tocsr().astype(np.float32)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    if mode == "max":
        graph = graph.maximum(graph.T).tocsr().astype(np.float32)
    elif mode == "mean":
        graph = ((graph + graph.T) * 0.5).tocsr().astype(np.float32)
    else:
        raise ValueError(f"未知对称化模式: {mode}")
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def bin_g(graph: sp.spmatrix) -> csr_matrix:
    return (graph > 0).astype(np.float32).tocsr()


def vals(graph: sp.spmatrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    graph = graph.tocsr()
    return np.asarray(graph[rows, cols]).ravel().astype(np.float32)


def mat(rows: np.ndarray, cols: np.ndarray, data: np.ndarray, shape) -> csr_matrix:
    return clean_graph(csr_matrix((data.astype(np.float32), (rows, cols)), shape=shape), mode="max")


def r01(values: np.ndarray) -> np.ndarray:
    """边权鲁棒映射到 [0, 1]，避免极端值主导。"""
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    p95 = float(np.percentile(values, 95))
    if p95 <= 0:
        p95 = float(values.max()) + EPS
    return np.clip(values / (p95 + EPS), 0.0, 1.0).astype(np.float32)


def edge_jaccard(a: sp.spmatrix, b: sp.spmatrix) -> float:
    aa = bin_g(a)
    bb = bin_g(b)
    inter = aa.multiply(bb).nnz
    union = ((aa + bb) > 0).nnz
    return float(inter / max(union, 1))


def neigh_overlap(a: sp.spmatrix, b: sp.spmatrix) -> Dict[str, float]:
    aa = bin_g(a)
    bb = bin_g(b)
    vals_out = []
    for i in range(aa.shape[0]):
        sa = set(aa.indices[aa.indptr[i]:aa.indptr[i + 1]])
        sb = set(bb.indices[bb.indptr[i]:bb.indptr[i + 1]])
        denom = max(min(len(sa), len(sb)), 1)
        vals_out.append(len(sa & sb) / denom)
    v = np.asarray(vals_out, dtype=np.float32)
    return {
        "neighbor_overlap_mean": float(v.mean()),
        "neighbor_overlap_q50": float(np.quantile(v, 0.50)),
        "neighbor_overlap_q90": float(np.quantile(v, 0.90)),
    }


def graph_summary(graph: sp.spmatrix, labels: Optional[np.ndarray] = None) -> Dict:
    graph = clean_graph(graph, mode="max")
    degree = np.asarray((graph > 0).sum(axis=1)).ravel().astype(float)
    weights = graph.data.astype(float)
    n_comp, comp_labels = connected_components(graph.maximum(graph.T), directed=False, return_labels=True)
    sizes = np.bincount(comp_labels) if comp_labels.size else np.array([0])
    info = {
        "cells": int(graph.shape[0]),
        "nnz": int(graph.nnz),
        "edges": int(graph.nnz // 2),
        "avg_deg": float(degree.mean()) if degree.size else 0.0,
        "deg_q50": float(np.percentile(degree, 50)) if degree.size else 0.0,
        "deg_q90": float(np.percentile(degree, 90)) if degree.size else 0.0,
        "deg_max": float(degree.max()) if degree.size else 0.0,
        "components": int(n_comp),
        "lcc": float(sizes.max() / max(graph.shape[0], 1)),
        "isolated": int((degree == 0).sum()),
        "w_mean": float(weights.mean()) if weights.size else 0.0,
        "w_q50": float(np.percentile(weights, 50)) if weights.size else 0.0,
        "w_q90": float(np.percentile(weights, 90)) if weights.size else 0.0,
    }
    if labels is not None:
        coo = graph.tocoo()
        mask = coo.row < coo.col
        if np.any(mask):
            rows, cols, data = coo.row[mask], coo.col[mask], coo.data[mask]
            same = labels[rows] == labels[cols]
            info["same_label_weighted"] = float(data[same].sum() / (data.sum() + EPS))
            info["same_label_edges"] = float(same.mean())
    return info


def save_json(path: Path, obj: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def print_graph_ok(name: str, graph: sp.spmatrix, labels: Optional[np.ndarray] = None):
    info = graph_summary(graph, labels)
    msg = (
        f"[OK] {name} | edges={info['edges']} avg_deg={info['avg_deg']:.2f} "
        f"comp={info['components']} lcc={info['lcc']:.3f}"
    )
    if "same_label_weighted" in info:
        msg += f" same_label={info['same_label_weighted']:.4f}"
    print(msg)


# -------------------------
# 表达矩阵准备
# -------------------------

def get_raw_matrix_and_genes(adata):
    if adata.raw is not None:
        x_raw = to_csr(adata.raw.X)
        var = adata.raw.var
        var_names = adata.raw.var_names
        source = "raw.X"
    else:
        x_raw = to_csr(adata.X)
        var = adata.var
        var_names = adata.var_names
        source = "X"

    if "feature_name" in var.columns:
        gene_names = var["feature_name"].astype(str).to_numpy()
    else:
        gene_names = var_names.astype(str).to_numpy()

    return x_raw.astype(np.float32), gene_names, source


def is_count_like(x: csr_matrix) -> bool:
    if x.data.size == 0:
        return False
    if float(x.data.min()) < -EPS:
        raise ValueError("表达矩阵含负值，不适合当前构图流程。")
    sample = x.data
    if sample.size > 200000:
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(sample.size, size=200000, replace=False)]
    max_v = float(sample.max())
    int_like = float(np.mean(np.abs(sample - np.round(sample)) < 1e-6))
    return (max_v > 50.0) or (max_v > 20.0 and int_like > 0.95)


def prepare_expression(adata, min_gene_frac: float = 0.05):
    """兼容原 G_hvg/G_pca/G_pf 的表达准备逻辑。细胞不重排、不过滤。"""
    x_raw, gene_names, matrix_source = get_raw_matrix_and_genes(adata)
    raw_genes = int(x_raw.shape[1])
    mode = "counts_like" if is_count_like(x_raw) else "log_or_normalized_like"

    min_cells = max(1, int(np.ceil(min_gene_frac * x_raw.shape[0])))
    gene_nnz = np.asarray((x_raw > 0).sum(axis=0)).ravel()
    keep = gene_nnz >= min_cells
    if int(keep.sum()) == 0:
        raise ValueError("低表达过滤后没有基因保留，请降低 min_gene_frac 或检查输入矩阵。")

    x = x_raw[:, keep].tocsr().astype(np.float32)
    gene_names = gene_names[keep]

    if mode == "counts_like":
        lib_size = np.asarray(x.sum(axis=1)).ravel().astype(np.float32)
        pos = lib_size > 0
        if not np.any(pos):
            raise ValueError("所有细胞 library size 为 0，无法归一化。")
        median_lib = float(np.median(lib_size[pos]))
        size_factor = lib_size / (median_lib + EPS)
        size_factor[size_factor <= 0] = 1.0
        x_expr = sp.diags(1.0 / size_factor).dot(x).tocsr().astype(np.float32)
        x_expr = log1p_sparse(x_expr)
        normalization = "median_library_normalize_then_log1p"
    else:
        x_expr = x.copy().tocsr().astype(np.float32)
        median_lib = None
        normalization = "use_as_is_after_gene_filter"

    info = {
        "matrix_source": matrix_source,
        "matrix_mode": mode,
        "normalization": normalization,
        "median_library_size": median_lib,
        "raw_genes": raw_genes,
        "filtered_genes": int(x_expr.shape[1]),
        "min_gene_frac": float(min_gene_frac),
        "min_cells": int(min_cells),
    }
    return x_expr, gene_names, info


def select_hvg_scanpy(x_expr: csr_matrix, gene_names: np.ndarray, n_top_genes: int):
    tmp = sc.AnnData(X=x_expr)
    tmp.var_names = gene_names
    tmp.var_names_make_unique()
    n_hvg = min(int(n_top_genes), x_expr.shape[1])
    sc.pp.highly_variable_genes(tmp, n_top_genes=n_hvg, flavor="seurat", subset=False)
    mask = tmp.var["highly_variable"].to_numpy()
    if int(mask.sum()) == 0:
        raise ValueError("HVG 选择结果为空。")
    x_hvg = x_expr[:, mask].toarray().astype(np.float32)
    hvg_genes = tmp.var_names[mask].astype(str).to_numpy()
    return x_hvg, hvg_genes


# -------------------------
# KNN / PCA / Diffmap
# -------------------------

def build_knn_graph(x: np.ndarray, k: int = 15, metric: str = "cosine") -> Tuple[csr_matrix, str]:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"输入矩阵必须二维，当前 shape={x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("输入矩阵存在 NaN/Inf")
    n = x.shape[0]
    if n <= k + 1:
        raise ValueError(f"n_cells={n} is too small for k={k}")

    algorithm = "brute" if metric in {"cosine", "correlation"} else "auto"
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric, algorithm=algorithm)
    nn.fit(x)
    dist, idx = nn.kneighbors(x)

    d = dist[:, 1:k + 1].astype(np.float32)
    nb = idx[:, 1:k + 1]

    if metric in {"cosine", "correlation"}:
        weights = np.clip(1.0 - d, 0.0, 1.0).astype(np.float32)
        weight_type = f"1_minus_{metric}_distance"
    else:
        sigma = np.median(d, axis=1).astype(np.float32)
        sigma = np.maximum(sigma, EPS)
        weights = np.exp(-(d ** 2) / (sigma[:, None] ** 2 + EPS)).astype(np.float32)
        weight_type = "local_gaussian_kernel"

    rows = np.repeat(np.arange(n), k)
    cols = nb.reshape(-1)
    data = weights.reshape(-1)
    graph = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float32)
    return clean_graph(graph, mode="max"), weight_type


def pca_embedding(x_hvg: np.ndarray, n_pca: int, seed: int = 42):
    n_comp = min(int(n_pca), x_hvg.shape[0] - 1, x_hvg.shape[1])
    n_comp = max(1, n_comp)
    x = x_hvg.astype(np.float32, copy=False)
    x = x - x.mean(axis=0, keepdims=True)
    pca = PCA(n_components=n_comp, random_state=seed)
    x_pca = pca.fit_transform(x).astype(np.float32)
    evr = float(np.sum(pca.explained_variance_ratio_))
    return x_pca, n_comp, evr


def build_diffmap_view(x_pca: np.ndarray, k: int = 15, dm_dim: int = 30):
    tmp = sc.AnnData(X=x_pca)
    tmp.obsm["X_pca_fixed"] = x_pca.astype(np.float32)
    sc.pp.neighbors(tmp, n_neighbors=k, use_rep="X_pca_fixed", metric="euclidean")

    n_comps = min(int(dm_dim) + 1, x_pca.shape[0] - 1)
    n_comps = max(2, n_comps)
    sc.tl.diffmap(tmp, n_comps=n_comps)

    x_dm_all = tmp.obsm["X_diffmap"]
    real_dim = min(int(dm_dim), x_dm_all.shape[1] - 1)
    real_dim = max(1, real_dim)
    x_dm = x_dm_all[:, 1:real_dim + 1].astype(np.float32)
    if not np.isfinite(x_dm).all():
        raise ValueError("X_dm 中存在 NaN/Inf")
    return x_dm, real_dim


# -------------------------
# G_expr / G_diff
# -------------------------

def build_g_expr(G_hvg: csr_matrix, G_pca: csr_matrix, q_pca=0.70, q_hvg=0.90, min_keep=3):
    gh = clean_graph(G_hvg)
    gp = clean_graph(G_pca)
    if gh.shape != gp.shape:
        raise ValueError(f"G_hvg 和 G_pca 维度不同: {gh.shape} vs {gp.shape}")

    bh, bp = bin_g(gh), bin_g(gp)
    union = ((bh + bp) > 0).astype(np.float32).tocsr()
    coo = union.tocoo()
    rows, cols = coo.row.astype(np.int64), coo.col.astype(np.int64)

    wh0, wp0 = vals(gh, rows, cols), vals(gp, rows, cols)
    has_h, has_p = wh0 > 0, wp0 > 0
    support = has_h.astype(np.float32) + has_p.astype(np.float32)

    wh, wp = np.zeros_like(wh0), np.zeros_like(wp0)
    if has_h.any():
        wh[has_h] = r01(wh0[has_h])
    if has_p.any():
        wp[has_p] = r01(wp0[has_p])

    mean_w = (wh + wp) / np.maximum(support, 1.0)
    diff = np.zeros_like(mean_w)
    both = support >= 2
    diff[both] = np.abs(wh[both] - wp[both])
    conf = np.clip(0.45 * (support / 2.0) + 0.45 * mean_w - 0.10 * diff, 0, 1)

    shared = has_h & has_p
    p_only = (~has_h) & has_p
    h_only = has_h & (~has_p)
    keep = shared.copy()
    if p_only.any():
        keep |= p_only & (conf >= np.quantile(conf[p_only], q_pca))
    if h_only.any():
        keep |= h_only & (conf >= np.quantile(conf[h_only], q_hvg))

    if min_keep > 0:
        for i in range(gh.shape[0]):
            idx = np.where(rows == i)[0]
            if idx.size and keep[idx].sum() < min_keep:
                top = idx[np.argsort(-conf[idx])[:min(min_keep, idx.size)]]
                keep[top] = True

    g = mat(rows[keep], cols[keep], conf[keep], gh.shape)
    info = {
        "from": ["G_hvg", "G_pca"],
        "q_pca": float(q_pca),
        "q_hvg": float(q_hvg),
        "min_keep": int(min_keep),
        "jaccard_hvg_pca": edge_jaccard(gh, gp),
        **neigh_overlap(gh, gp),
        "kept_shared": int((shared & keep).sum()),
        "kept_pca_only": int((p_only & keep).sum()),
        "kept_hvg_only": int((h_only & keep).sum()),
    }
    return g, info


def _q(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.quantile(arr, q))


def build_g_diff(G_dm: csr_matrix, G_expr: csr_matrix, x_pca: np.ndarray, q_long=0.90, q_keep=0.25, min_keep=3):
    gd = clean_graph(G_dm)
    gg = clean_graph(G_expr)
    if gd.shape != gg.shape:
        raise ValueError(f"G_dm 和 G_expr 维度不同: {gd.shape} vs {gg.shape}")
    if x_pca.shape[0] != gd.shape[0]:
        raise ValueError(f"X_pca 行数和图节点数不同: {x_pca.shape[0]} vs {gd.shape[0]}")

    coo = gd.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    if rows.size == 0:
        raise ValueError("G_dm 没有可用边，不能构建 G_diff")

    wd = r01(coo.data.astype(np.float32))
    wg = vals(gg, rows, cols)
    has_guard = wg > 0
    dm_only = ~has_guard

    gcoo = gg.tocoo()
    grows = gcoo.row.astype(np.int64)
    gcols = gcoo.col.astype(np.int64)
    if grows.size == 0:
        raise ValueError("G_expr 没有可用边，不能作为 diffusion guard")

    guard_dist = np.linalg.norm(x_pca[grows] - x_pca[gcols], axis=1)
    tau = float(np.quantile(guard_dist, q_long))
    dist = np.linalg.norm(x_pca[rows] - x_pca[cols], axis=1)

    pca_gate = np.exp(-dist / (tau + EPS)).astype(np.float32)
    conf = wd * pca_gate
    if has_guard.any():
        conf[has_guard] = np.maximum(conf[has_guard], 0.70 * wd[has_guard] + 0.30 * r01(wg[has_guard]))

    keep_guard = has_guard
    if dm_only.any():
        dm_only_thr = float(np.quantile(conf[dm_only], q_keep))
        keep_dm_only = dm_only & (conf >= dm_only_thr) & (dist <= tau)
    else:
        dm_only_thr = 1.0
        keep_dm_only = np.zeros_like(has_guard, dtype=bool)

    keep = keep_guard | keep_dm_only
    forced_idx = []

    if min_keep > 0:
        g_tmp = mat(rows[keep], cols[keep], conf[keep], gd.shape)
        deg = np.asarray((g_tmp > 0).sum(axis=1)).ravel().astype(np.int64)
        low_cells = np.where(deg < min_keep)[0]

        for i in low_cells:
            need = int(min_keep - deg[i])
            if need <= 0:
                continue
            cand = np.where((rows == i) & (~keep) & (dist <= tau))[0]
            if cand.size == 0:
                continue
            cand = cand[np.argsort(-conf[cand])]
            chosen = cand[:need]
            keep[chosen] = True
            forced_idx.extend(chosen.tolist())
            deg[i] += len(chosen)
            for j in cols[chosen]:
                deg[j] += 1

    forced_idx = np.asarray(forced_idx, dtype=np.int64)
    g = mat(rows[keep], cols[keep], conf[keep], gd.shape)
    info = {
        "from": ["G_dm"],
        "guard": "G_expr",
        "q_long": float(q_long),
        "q_keep": float(q_keep),
        "min_keep": int(min_keep),
        "jaccard_dm_guard": edge_jaccard(gd, gg),
        **neigh_overlap(gd, gg),
        "pca_long_tau": tau,
        "dm_only_conf_threshold": dm_only_thr,
        "raw_dm_edges": int(rows.size),
        "guard_supported_edges": int(has_guard.sum()),
        "dm_only_edges": int(dm_only.sum()),
        "kept_guard_edges": int(keep_guard.sum()),
        "kept_guard_long_edges": int((keep_guard & (dist > tau)).sum()),
        "kept_dm_only_before_min_keep": int(keep_dm_only.sum()),
        "removed_dm_only_long_edges": int((dm_only & (dist > tau)).sum()),
        "forced_keep_edges": int(forced_idx.size),
        "forced_keep_cells": int(len(np.unique(rows[forced_idx])) if forced_idx.size else 0),
        "forced_keep_conf_q50": _q(conf[forced_idx], 0.50),
        "forced_keep_conf_q90": _q(conf[forced_idx], 0.90),
        "forced_keep_dist_q50": _q(dist[forced_idx], 0.50),
        "forced_keep_dist_q90": _q(dist[forced_idx], 0.90),
        "rule": "long-edge hard filter is applied only to dm-only edges; min_keep only restores local G_dm edges with dist <= tau",
    }
    return g, info


# -------------------------
# G_pf
# -------------------------

def select_module_genes(x_expr: csr_matrix, gene_names: np.ndarray, n_module_genes: int):
    tmp = sc.AnnData(X=x_expr)
    tmp.var_names = gene_names
    tmp.var_names_make_unique()
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(n_module_genes, x_expr.shape[1]), flavor="seurat", subset=False)
    mask = tmp.var["highly_variable"].to_numpy()
    if int(mask.sum()) == 0:
        raise ValueError("module genes 选择结果为空。")
    x_gene = x_expr[:, mask].toarray().astype(np.float32)
    module_genes = tmp.var_names[mask].astype(str).to_numpy()
    return x_gene, module_genes


def build_gene_modules(x_gene: np.ndarray, n_modules: int, method: str = "ward", seed: int = 42):
    if x_gene.shape[1] < 2:
        raise ValueError("module genes 少于 2，无法构建 gene modules")
    n_modules = min(int(n_modules), x_gene.shape[1])
    n_modules = max(1, n_modules)

    gene_mat = x_gene.T.astype(np.float32)
    gene_mat = gene_mat - gene_mat.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(gene_mat, axis=1, keepdims=True)
    norm[norm <= 0] = 1.0
    gene_mat = gene_mat / norm

    if method == "ward":
        dist = pdist(gene_mat, metric="euclidean")
        z = linkage(dist, method="ward")
        module_id = fcluster(z, t=n_modules, criterion="maxclust")
    elif method == "kmeans":
        km = MiniBatchKMeans(n_clusters=n_modules, random_state=seed, batch_size=1024, n_init=10)
        module_id = km.fit_predict(gene_mat) + 1
    else:
        raise ValueError(f"未知 module_method: {method}")
    return module_id.astype(np.int32)


def aucell_like_score(x_gene: np.ndarray, module_id: np.ndarray, auc_top_frac: float = 0.05):
    n_cells, n_genes = x_gene.shape
    modules = np.unique(module_id)
    top_rank = max(1, int(np.ceil(auc_top_frac * n_genes)))
    x_pf = np.zeros((n_cells, len(modules)), dtype=np.float32)
    module_genes = [np.where(module_id == m)[0] for m in modules]

    for i in range(n_cells):
        order = np.argsort(-x_gene[i])
        rank_pos = np.empty(n_genes, dtype=np.int32)
        rank_pos[order] = np.arange(1, n_genes + 1)
        for j, genes in enumerate(module_genes):
            ranks = rank_pos[genes]
            weights = np.maximum(0.0, (top_rank - ranks + 1) / top_rank)
            x_pf[i, j] = weights.mean()
    return x_pf, modules


def save_modules(out_dir: Path, gene_names: np.ndarray, module_id: np.ndarray):
    path = out_dir / "pf_modules.tsv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("gene\tmodule\n")
        for gene, mid in zip(gene_names, module_id):
            f.write(f"{gene}\t{int(mid)}\n")
    return path


def save_module_summary(out_dir: Path, module_id: np.ndarray, x_pf: np.ndarray):
    modules = np.unique(module_id)
    module_sizes = np.array([np.sum(module_id == m) for m in modules], dtype=int)
    activity_mean = x_pf.mean(axis=0)
    activity_std = x_pf.std(axis=0)
    activity_q90 = np.percentile(x_pf, 90, axis=0)
    path = out_dir / "pf_module_summary.tsv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("module\tsize\tactivity_mean\tactivity_std\tactivity_q90\n")
        for j, m in enumerate(modules):
            f.write(f"{int(m)}\t{int(module_sizes[j])}\t{float(activity_mean[j]):.6g}\t{float(activity_std[j]):.6g}\t{float(activity_q90[j]):.6g}\n")
    stats = {
        "module_size_min": int(module_sizes.min()) if module_sizes.size else 0,
        "module_size_median": float(np.median(module_sizes)) if module_sizes.size else 0.0,
        "module_size_max": int(module_sizes.max()) if module_sizes.size else 0,
        "activity_std_min": float(activity_std.min()) if activity_std.size else 0.0,
        "activity_std_median": float(np.median(activity_std)) if activity_std.size else 0.0,
        "activity_std_max": float(activity_std.max()) if activity_std.size else 0.0,
        "low_activity_std_modules": int((activity_std <= 1e-6).sum()),
    }
    return path, stats


# -------------------------
# 主流程
# -------------------------

def infer_name(h5ad_path: Path) -> str:
    stem = h5ad_path.stem
    return stem


def build_graphs(args):
    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"找不到 h5ad 文件: {h5ad_path}")

    name = args.name or infer_name(h5ad_path)
    out_dir = Path("data") / name / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading h5ad: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Expression matrix: cells={adata.n_obs}, genes={adata.n_vars}")
    print("Cells are not filtered or reordered.")
    print(f"Output graph dir: {out_dir}")

    labels = None
    if args.label_key and args.label_key in adata.obs.columns:
        labels = adata.obs[args.label_key].astype(str).to_numpy()
        print(f"Label key for diagnostics: {args.label_key} | n_classes={len(np.unique(labels))}")
    elif args.label_key:
        print(f"Label key not found, diagnostics without label: {args.label_key}")

    x_expr, gene_names, prep_info = prepare_expression(adata, min_gene_frac=args.min_gene_frac)

    # G_hvg
    print("Building G_hvg")
    x_hvg, hvg_genes = select_hvg_scanpy(x_expr, gene_names, args.n_top_genes)
    G_hvg, hvg_weight = build_knn_graph(x_hvg, k=args.k, metric=args.hvg_metric)
    np.save(out_dir / "X_hvg.npy", x_hvg.astype(np.float32))
    save_npz(out_dir / "G_hvg.npz", G_hvg)
    info_hvg = {
        "graph": "G_hvg",
        "role": "gene-level expression detail graph",
        "input": str(h5ad_path),
        **prep_info,
        "hvg_genes": int(x_hvg.shape[1]),
        "k": int(args.k),
        "metric": args.hvg_metric,
        "weight_type": hvg_weight,
        "row_norm": False,
        **graph_summary(G_hvg, labels),
    }
    save_json(out_dir / "G_hvg.json", info_hvg)
    print_graph_ok("G_hvg", G_hvg, labels)

    # G_pca
    print(f"Building G_pca: n_pca={args.n_pca}")
    x_pca, pca_dim, evr = pca_embedding(x_hvg, args.n_pca, seed=args.seed)
    G_pca, pca_weight = build_knn_graph(x_pca, k=args.k, metric=args.pca_metric)
    np.save(out_dir / "X_pca.npy", x_pca.astype(np.float32))
    save_npz(out_dir / "G_pca.npz", G_pca)
    info_pca = {
        "graph": "G_pca",
        "role": "denoised expression backbone graph",
        "input": str(h5ad_path),
        "source_feature": "X_hvg.npy",
        "pca_dim": int(pca_dim),
        "explained_variance_ratio_sum": evr,
        "k": int(args.k),
        "metric": args.pca_metric,
        "weight_type": pca_weight,
        "row_norm": False,
        **prep_info,
        "hvg_genes": int(x_hvg.shape[1]),
        **graph_summary(G_pca, labels),
    }
    save_json(out_dir / "G_pca.json", info_pca)
    print_graph_ok("G_pca", G_pca, labels)

    # G_dm
    print(f"Building G_dm from X_pca: dm_dim={args.dm_dim}")
    x_dm, real_dm_dim = build_diffmap_view(x_pca, k=args.k, dm_dim=args.dm_dim)
    G_dm, dm_weight = build_knn_graph(x_dm, k=args.k, metric=args.dm_metric)
    np.save(out_dir / "X_dm.npy", x_dm.astype(np.float32))
    save_npz(out_dir / "G_dm.npz", G_dm)
    info_dm = {
        "graph": "G_dm",
        "role": "manifold-level diffusion graph",
        "source_feature": "X_pca.npy",
        "dm_dim_input": int(args.dm_dim),
        "dm_dim_final": int(real_dm_dim),
        "skip_first_diffmap_component": True,
        "k": int(args.k),
        "metric": args.dm_metric,
        "weight_type": dm_weight,
        "row_norm": False,
        **graph_summary(G_dm, labels),
    }
    save_json(out_dir / "G_dm.json", info_dm)
    print_graph_ok("G_dm", G_dm, labels)

    # G_expr
    print("Building G_expr from G_hvg + G_pca")
    G_expr, expr_info = build_g_expr(G_hvg, G_pca, q_pca=args.q_pca, q_hvg=args.q_hvg, min_keep=args.expr_min_keep)
    save_npz(out_dir / "G_expr.npz", G_expr)
    info_expr = {
        "graph": "G_expr",
        "role": "expression guard graph",
        **expr_info,
        "row_norm": False,
        **graph_summary(G_expr, labels),
    }
    save_json(out_dir / "G_expr.json", info_expr)
    print_graph_ok("G_expr", G_expr, labels)

    # G_diff
    print("Building G_diff from G_dm + G_expr guard")
    G_diff, diff_info = build_g_diff(G_dm, G_expr, x_pca, q_long=args.q_long, q_keep=args.q_keep, min_keep=args.diff_min_keep)
    save_npz(out_dir / "G_diff.npz", G_diff)
    info_diff = {
        "graph": "G_diff",
        "role": "expression-guarded diffusion graph",
        **diff_info,
        "row_norm": False,
        **graph_summary(G_diff, labels),
    }
    save_json(out_dir / "G_diff.json", info_diff)
    print_graph_ok("G_diff", G_diff, labels)

    # G_pf
    print(f"Building G_pf: genes={args.n_module_genes}, modules={args.n_modules}, method={args.module_method}")
    x_gene, module_genes = select_module_genes(x_expr, gene_names, args.n_module_genes)
    module_id = build_gene_modules(x_gene, args.n_modules, method=args.module_method, seed=args.seed)
    module_path = save_modules(out_dir, module_genes, module_id)
    x_pf, modules = aucell_like_score(x_gene, module_id, auc_top_frac=args.auc_top_frac)
    if not np.isfinite(x_pf).all():
        raise ValueError("X_pf 存在 NaN/Inf")
    module_summary_path, module_stats = save_module_summary(out_dir, module_id, x_pf)
    G_pf, pf_weight = build_knn_graph(x_pf, k=args.k, metric=args.pf_metric)
    np.save(out_dir / "X_pf.npy", x_pf.astype(np.float32))
    save_npz(out_dir / "G_pf.npz", G_pf)
    info_pf = {
        "graph": "G_pf",
        "role": "de_novo_gene_module_activity_graph",
        "input": str(h5ad_path),
        **prep_info,
        "module_genes": int(x_gene.shape[1]),
        "n_modules_input": int(args.n_modules),
        "n_modules_final": int(len(modules)),
        "module_method": args.module_method,
        "activity_method": "AUCell_like_rank_score",
        "auc_top_frac": float(args.auc_top_frac),
        "k": int(args.k),
        "metric": args.pf_metric,
        "weight_type": pf_weight,
        "row_norm": False,
        "output_modules": str(module_path),
        "output_module_summary": str(module_summary_path),
        **module_stats,
        **graph_summary(G_pf, labels),
    }
    save_json(out_dir / "G_pf.json", info_pf)
    print_graph_ok("G_pf", G_pf, labels)

    summary = {
        "name": name,
        "input_h5ad": str(h5ad_path),
        "output_dir": str(out_dir),
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "label_key": args.label_key if labels is not None else None,
        "graphs_for_scTGCL_prior": ["G_hvg", "G_pca", "G_diff", "G_pf"],
        "auxiliary_graphs": ["G_dm", "G_expr"],
        "parameters": vars(args),
        "graph_stats": {
            "G_hvg": graph_summary(G_hvg, labels),
            "G_pca": graph_summary(G_pca, labels),
            "G_dm": graph_summary(G_dm, labels),
            "G_expr": graph_summary(G_expr, labels),
            "G_diff": graph_summary(G_diff, labels),
            "G_pf": graph_summary(G_pf, labels),
        },
    }
    save_json(out_dir / "graph_build_summary.json", summary)

    print("\nFinished graph construction.")
    print(f"Output graph dir: {out_dir}")
    print("Use with main.py:")
    print(f"  --graph_dir {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad", required=True, help="input h5ad file")
    parser.add_argument("--name", default=None, help="dataset name under data/{name}/graph; default = h5ad stem")
    parser.add_argument("--label_key", default="cell_type")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--min_gene_frac", type=float, default=0.05)
    parser.add_argument("--n_top_genes", type=int, default=2500)
    parser.add_argument("--n_pca", type=int, default=50)
    parser.add_argument("--dm_dim", type=int, default=30)

    parser.add_argument("--hvg_metric", default="cosine", choices=["cosine", "correlation", "euclidean"])
    parser.add_argument("--pca_metric", default="euclidean", choices=["cosine", "correlation", "euclidean"])
    parser.add_argument("--dm_metric", default="euclidean", choices=["cosine", "correlation", "euclidean"])
    parser.add_argument("--pf_metric", default="cosine", choices=["cosine", "correlation", "euclidean"])

    parser.add_argument("--q_pca", type=float, default=0.70)
    parser.add_argument("--q_hvg", type=float, default=0.90)
    parser.add_argument("--expr_min_keep", type=int, default=3)

    parser.add_argument("--q_long", type=float, default=0.90)
    parser.add_argument("--q_keep", type=float, default=0.25)
    parser.add_argument("--diff_min_keep", type=int, default=3)

    parser.add_argument("--n_module_genes", type=int, default=5000)
    parser.add_argument("--n_modules", type=int, default=120)
    parser.add_argument("--auc_top_frac", type=float, default=0.05)
    parser.add_argument("--module_method", default="ward", choices=["ward", "kmeans"])

    args = parser.parse_args()
    build_graphs(args)


if __name__ == "__main__":
    main()
