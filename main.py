import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import argparse
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import scanpy as sc
from config import get_config
from graph_prior import LocalGraphPrior
from utils import (
    setup_seed, 
    create_augmented_data, 
    scRNASeqDataset, 
    evaluate,
    evaluate_clustering,
)
from scTGCL import ContrastiveLoss, scTGCL
from bfn import build_bfn_structure, summarize_build_info


# ============================================================================
# Argument Parser
# ============================================================================

def parse_args():
    """Parse command line arguments, using config defaults"""
    defaults = get_config()

    parser = argparse.ArgumentParser(
        description='scTGCL: Single-cell Transformer Graph Contrastive Learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Dataset
    parser.add_argument('--dataset', type=str, help='Dataset name to use for training')
    parser.add_argument("--n_clusters", type=int, help='Number of classes in the dataset')
    
    # Model architecture
    parser.add_argument('--embed_dim', type=int, default=defaults['embed_dim'],
                        help='Embedding dimension for transformer')
    parser.add_argument('--num_heads', type=int, default=defaults['num_heads'],
                        help='Number of attention heads')
    parser.add_argument('--free_heads', type=int, default=defaults['free_heads'],
                        help='Number of unconstrained attention heads')
    parser.add_argument('--latent_dim', type=int, default=defaults['latent_dim'],
                        help='Latent space dimension')
    parser.add_argument('--dropout', type=float, default=defaults['dropout'],
                        help='Dropout rate')

    # Data augmentation
    parser.add_argument('--mask_prob', type=float, default=defaults['mask_prob'],
                        help='Gene masking probability for augmentation')
    parser.add_argument('--attention_mask_prob', type=float, default=defaults['attention_mask_prob'],
                        help='Attention masking probability')

    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=defaults['lambda_recon'],
                        help='Weight for reconstruction loss')
    parser.add_argument('--lambda_impute', type=float, default=defaults['lambda_impute'],
                        help='Weight for imputation loss')
    parser.add_argument('--lambda_contrast', type=float, default=defaults['lambda_contrast'],
                        help='Weight for contrastive loss')
    parser.add_argument('--temperature', type=float, default=defaults['temperature'],
                        help='Temperature for contrastive loss')

    # SRS-guided topology consistency
    parser.add_argument('--lambda_srs_tc', type=float, default=defaults['lambda_srs_tc'],
                        help='Weight for SRS-guided topology consistency loss')
    parser.add_argument('--lambda_balance', type=float, default=defaults['lambda_balance'],
                        help='Weight for optional cluster-balance regularization')
    parser.add_argument('--srs_tc_min_support', type=int, default=defaults['srs_tc_min_support'],
                        help='Minimum multi-graph support used by SRS-TC')
    parser.add_argument('--srs_tc_min_srs', type=float, default=defaults['srs_tc_min_srs'],
                        help='Minimum SRS value used by SRS-TC')
    parser.add_argument('--srs_tc_metric', type=str, default=defaults['srs_tc_metric'],
                        choices=['mse', 'kl'],
                        help='Distribution distance used by SRS-TC')
    parser.add_argument('--srs_tc_temp', type=float, default=defaults['srs_tc_temp'],
                        help='Temperature used to sharpen cluster probabilities for SRS-TC')
    parser.add_argument('--lambda_confidence', type=float, default=defaults['lambda_confidence'],
                        help='Weight for entropy minimization on cluster probabilities')


    # Graph-prior attention
    parser.add_argument('--graph_dir', type=str, default=defaults['graph_dir'],
                        help='Directory containing G_hvg/G_pca/G_diff/G_pf .npz graph files')
    parser.add_argument('--tau', type=float, default=defaults['tau'],
                        help='Scale for normalized prior-head content logits')
    parser.add_argument('--graph_beta', type=float, default=defaults['graph_beta'],
                        help='Strength of graph-prior bias injected into prior heads')
    parser.add_argument('--rank_eps', type=float, default=defaults['rank_eps'],
                        help='Minimum rank bias assigned to the weakest neighbor')

    # SRS-RCB reliability control
    parser.add_argument(
        '--availability_support', '--cov_support',
        type=int,
        default=defaults['availability_support'],
        help='Minimum graph-source support defining reliable structural availability',
    )
    parser.add_argument('--beta_base', type=float, default=defaults['beta_base'])

    # BFN false-negative suppression
    parser.add_argument('--fn_lambda', type=float, default=defaults['fn_lambda'])
    parser.add_argument('--fn_min', type=float, default=defaults['fn_min'])
    parser.add_argument('--fn_topk', type=int, default=defaults['fn_topk'])
    parser.add_argument('--fn_threshold', type=float, default=defaults['fn_threshold'])
    parser.add_argument(
        '--fn_gate_mode',
        type=str,
        choices=['threshold', 'always', 'off'],
        default=defaults['fn_gate_mode'],
        help='Dataset-level BFN gate: threshold anchor, always on, or off',
    )
    parser.add_argument(
        '--fn_risk_mode',
        type=str,
        choices=['availability', 'srs'],
        default=defaults['fn_risk_mode'],
        help='Pair risk: SRS calibrated by pair availability, or SRS only',
    )
    parser.add_argument('--fn_min_support', type=int, default=defaults['fn_min_support'])
    parser.add_argument('--fn_min_risk', type=float, default=defaults['fn_min_risk'])


    # Training
    parser.add_argument('--lr', type=float, default=defaults['lr'],
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=defaults['weight_decay'],
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--epochs', type=int, default=defaults['epochs'],
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'],
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=defaults['seed'],
                        help='Random seed (default is randomly chosen at startup)')
    parser.add_argument('--graph_batch_neighbors', type=int, default=defaults['graph_batch_neighbors'],
                        help='Graph neighbors grouped with each anchor in training batches; 0 uses random batches')
    parser.add_argument('--graph_warmup_epochs', type=int, default=defaults['graph_warmup_epochs'],
                        help='Linearly warm up graph-prior strength over this many epochs')
    parser.add_argument('--select_metric', type=str, default=defaults['select_metric'],
                        choices=['recon', 'acc', 'nmi', 'ari', 'mean'],
                        help='Criterion used to keep the best checkpoint')
    parser.add_argument('--cluster_eval_interval', type=int, default=defaults['cluster_eval_interval'],
                        help='Epoch interval for clustering-based checkpoint selection')
    parser.add_argument('--batch_log_interval', type=int, default=defaults['batch_log_interval'],
                        help='Print training progress every N batches; 0 disables batch-level logging')
    parser.add_argument('--label_key', type=str, default=defaults['label_key'],
                        help='Column in adata.obs used as cell labels')
    parser.add_argument('--eval_scope', type=str, default=defaults['eval_scope'],
                        choices=['train', 'test', 'full'],
                        help='Evaluate on train, held-out test, or full dataset')
    parser.add_argument('--eval_batch_size', type=int, default=defaults['eval_batch_size'],
                        help='Batch size used for clustering evaluation; 0 reuses training batch size')

    # Paths
    parser.add_argument('--save_dir', type=str, default=defaults['save_dir'],
                        help='Directory to save results and visualizations')

    return parser.parse_args()


# ============================================================================
# Training Function
# ============================================================================

def unpack_batch(batch):
    if len(batch) == 3:
        return batch

    batch_x, batch_y = batch
    batch_idx = None
    return batch_x, batch_y, batch_idx


class GraphNeighborBatchSampler:
    """Batch sampler that keeps graph neighbors in the same mini-batch."""

    def __init__(
        self,
        dataset_indices,
        graph_prior,
        batch_size,
        neighbors_per_anchor=4,
        seed=42,
    ):
        self.dataset_indices = np.asarray(dataset_indices, dtype=np.int64)
        self.graph_prior = graph_prior
        self.batch_size = int(batch_size)
        self.neighbors_per_anchor = int(neighbors_per_anchor)
        self.seed = int(seed)
        self.epoch = 0
        self.global_to_local = {
            int(global_idx): local_idx
            for local_idx, global_idx in enumerate(self.dataset_indices)
        }
        self.local_neighbors = self._build_local_neighbors()

    def _build_local_neighbors(self):
        neighbors = []
        for global_idx in self.dataset_indices:
            local_set = set()
            for graph in self.graph_prior.graphs.values():
                row = graph.getrow(int(global_idx))
                for nb in row.indices:
                    local_nb = self.global_to_local.get(int(nb))
                    if local_nb is not None:
                        local_set.add(local_nb)
            neighbors.append(np.fromiter(local_set, dtype=np.int64))
        return neighbors

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        order = rng.permutation(len(self.dataset_indices))
        unused = np.ones(len(self.dataset_indices), dtype=bool)
        batch = []

        def add_item(local_idx):
            if unused[local_idx]:
                unused[local_idx] = False
                batch.append(int(local_idx))

        for anchor in order:
            if not unused[anchor]:
                continue

            add_item(anchor)

            neigh = self.local_neighbors[anchor]
            if neigh.size > 0 and self.neighbors_per_anchor > 0:
                neigh = neigh[unused[neigh]]
                if neigh.size > 0:
                    rng.shuffle(neigh)
                    for nb in neigh[:self.neighbors_per_anchor]:
                        add_item(int(nb))
                        if len(batch) >= self.batch_size:
                            break

            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

        if batch:
            yield batch

    def __len__(self):
        return int(np.ceil(len(self.dataset_indices) / float(self.batch_size)))


def build_fn_weights(batch_prior, config, device):
    """Build symmetric top-k BFN weights.

    Defaults reproduce the current anchor exactly.  Diagnostics may switch the
    dataset-level gate or use SRS-only risk without changing the positive-pair
    definition or the contrastive objective.
    """
    structure = build_bfn_structure(batch_prior, config, device)
    info = summarize_build_info(structure)
    if structure is None or not structure["gate_enabled"]:
        return None, info
    return structure["applied_weights"].detach(), info

def estimate_uncovered(dataset, batch_size, seed, graph_prior, device):
    if graph_prior is None:
        return 1.0

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    zero_count = 0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            _, _, batch_idx = unpack_batch(batch)
            batch_prior = graph_prior.get_batch(batch_idx, device)
            availability = batch_prior.availability
            zero_count += int((availability <= 0).sum().item())
            total_count += int(availability.numel())

    return float(zero_count / max(total_count, 1))


def get_batch_prior(graph_prior, batch_idx, device):
    if graph_prior is None or batch_idx is None:
        return None
    return graph_prior.get_batch(batch_idx, device)


def init_attention_srs_stats():
    return {
        'support_sum': 0.0,
        'support_count': 0,
        'high_sum': 0.0,
        'high_count': 0,
        'low_sum': 0.0,
        'low_count': 0,
        'corr_sum': 0.0,
        'corr_count': 0,
    }


def update_attention_srs_stats(stats, outputs, batch_prior):
    if batch_prior is None or outputs is None:
        return

    diag = outputs.get('attention_diag_orig', {})
    A_prior = diag.get('A_prior')
    if A_prior is None:
        return

    with torch.no_grad():
        A_prior = A_prior.detach()
        srs = batch_prior.srs.detach().to(device=A_prior.device, dtype=A_prior.dtype)
        support = batch_prior.support.detach().to(device=A_prior.device)

        bsz = srs.shape[0]
        off_diag = ~torch.eye(bsz, dtype=torch.bool, device=srs.device)
        edge_mask = (support > 0) & off_diag
        if int(edge_mask.sum().item()) == 0:
            return

        edge_attn = A_prior[edge_mask]
        edge_srs = srs[edge_mask]

        stats['support_sum'] += float(edge_attn.sum().item())
        stats['support_count'] += int(edge_attn.numel())

        if edge_attn.numel() >= 4:
            q25 = torch.quantile(edge_srs, 0.25)
            q75 = torch.quantile(edge_srs, 0.75)
            high_mask = edge_srs >= q75
            low_mask = edge_srs <= q25

            if int(high_mask.sum().item()) > 0:
                high = edge_attn[high_mask]
                stats['high_sum'] += float(high.sum().item())
                stats['high_count'] += int(high.numel())

            if int(low_mask.sum().item()) > 0:
                low = edge_attn[low_mask]
                stats['low_sum'] += float(low.sum().item())
                stats['low_count'] += int(low.numel())

            if float(edge_srs.std(unbiased=False).item()) > 1e-8 and float(edge_attn.std(unbiased=False).item()) > 1e-8:
                srs_centered = edge_srs - edge_srs.mean()
                attn_centered = edge_attn - edge_attn.mean()
                corr = (srs_centered * attn_centered).mean() / (
                    srs_centered.std(unbiased=False) * attn_centered.std(unbiased=False) + 1e-8
                )
                stats['corr_sum'] += float(corr.item())
                stats['corr_count'] += 1


def format_attention_srs_stats(stats):
    if stats['support_count'] == 0:
        return ''

    support_mean = stats['support_sum'] / max(stats['support_count'], 1)
    high_mean = stats['high_sum'] / max(stats['high_count'], 1) if stats['high_count'] > 0 else 0.0
    low_mean = stats['low_sum'] / max(stats['low_count'], 1) if stats['low_count'] > 0 else 0.0
    corr_mean = stats['corr_sum'] / max(stats['corr_count'], 1) if stats['corr_count'] > 0 else 0.0

    return (
        f", PriorAttn(edge={support_mean:.4f}, "
        f"highSRS={high_mean:.4f}, lowSRS={low_mean:.4f}, "
        f"corr={corr_mean:.4f})"
    )



def init_bfn_stats():
    return {
        'enabled_count': 0,
        'active_pairs': 0,
        'candidate_pairs': 0,
        'risk_sum': 0.0,
        'weight_sum': 0.0,
        'min_weight': 1.0,
        'batch_count': 0,
    }


def update_bfn_stats(stats, info):
    if info is None:
        return
    if int(info.get('enabled', 0)) <= 0:
        return
    stats['enabled_count'] += 1
    stats['batch_count'] += 1
    stats['candidate_pairs'] += int(info.get('candidate_pairs', 0))
    active_pairs = int(info.get('active_pairs', 0))
    stats['active_pairs'] += active_pairs
    if active_pairs > 0:
        stats['risk_sum'] += float(info.get('mean_risk', 0.0))
        stats['weight_sum'] += float(info.get('mean_weight', 1.0))
        stats['min_weight'] = min(stats['min_weight'], float(info.get('min_weight', 1.0)))


def format_bfn_stats(stats):
    if stats['enabled_count'] == 0:
        return ''
    active = int(stats['active_pairs'])
    candidates = int(stats['candidate_pairs'])
    if active <= 0:
        return f", BFN(active=0, cand={candidates})"
    mean_risk = stats['risk_sum'] / max(stats['batch_count'], 1)
    mean_weight = stats['weight_sum'] / max(stats['batch_count'], 1)
    return (
        f", BFN(active={active}, cand={candidates}, "
        f"risk={mean_risk:.4f}, weight={mean_weight:.4f}, "
        f"minW={stats['min_weight']:.4f})"
    )


def sharpen_cluster_prob(cluster_logits, config):
    if cluster_logits is None:
        return None
    temp = max(float(config.get('srs_tc_temp', 1.0)), 1e-3)
    return torch.softmax(cluster_logits / temp, dim=1)


def srs_topology_consistency_loss(cluster_logits, batch_prior, config):
    """SRS-guided topology consistency over batch-local reliable edges.

    The loss does not use labels. It encourages cells connected by reliable
    structural evidence to have similar cluster assignment distributions.
    """
    if cluster_logits is None or batch_prior is None:
        zero = torch.tensor(0.0, device=cluster_logits.device if cluster_logits is not None else 'cpu')
        return zero, {'active_edges': 0, 'mean_srs': 0.0, 'pair_dist': 0.0, 'entropy': 0.0, 'max_prob': 0.0}

    p = sharpen_cluster_prob(cluster_logits, config)
    device = p.device
    dtype = p.dtype
    support = batch_prior.support.detach().to(device=device)
    srs = batch_prior.srs.detach().to(device=device, dtype=dtype)

    bsz = p.shape[0]
    off_diag = ~torch.eye(bsz, dtype=torch.bool, device=device)
    edge_mask = (
        (support >= float(config['srs_tc_min_support']))
        & (srs >= float(config['srs_tc_min_srs']))
        & off_diag
    )

    active_edges = int(edge_mask.sum().item())
    if active_edges == 0:
        return p.sum() * 0.0, {'active_edges': 0, 'mean_srs': 0.0, 'pair_dist': 0.0, 'entropy': 0.0, 'max_prob': 0.0}

    p_i = p.unsqueeze(1).expand(-1, bsz, -1)
    p_j = p.unsqueeze(0).expand(bsz, -1, -1)

    metric = str(config.get('srs_tc_metric', 'mse')).lower()
    if metric == 'kl':
        eps = 1e-8
        log_i = torch.log(p_i.clamp_min(eps))
        log_j = torch.log(p_j.clamp_min(eps))
        dist = 0.5 * (
            (p_i * (log_i - log_j)).sum(dim=-1)
            + (p_j * (log_j - log_i)).sum(dim=-1)
        )
    else:
        dist = (p_i - p_j).pow(2).sum(dim=-1)

    weights = srs * edge_mask.float()
    loss = (weights * dist).sum() / weights.sum().clamp_min(1e-8)

    with torch.no_grad():
        pair_dist = float(dist[edge_mask].mean().item())
        mean_srs = float(srs[edge_mask].mean().item())
        entropy = float((-(p * torch.log(p.clamp_min(1e-8))).sum(dim=1)).mean().item())
        max_prob = float(p.max(dim=1).values.mean().item())

    return loss, {
        'active_edges': active_edges,
        'mean_srs': mean_srs,
        'pair_dist': pair_dist,
        'entropy': entropy,
        'max_prob': max_prob,
    }


def cluster_balance_loss(cluster_prob):
    """KL(mean cluster usage || uniform), used only when lambda_balance > 0."""
    if cluster_prob is None:
        return torch.tensor(0.0)
    eps = 1e-8
    mean_prob = cluster_prob.mean(dim=0)
    n_clusters = mean_prob.numel()
    return (mean_prob * torch.log(mean_prob.clamp_min(eps) * float(n_clusters))).sum()


def cluster_confidence_loss(cluster_prob):
    """Mean entropy of cluster probabilities. Minimizing it activates non-uniform assignments."""
    if cluster_prob is None:
        return torch.tensor(0.0)
    eps = 1e-8
    return (-(cluster_prob * torch.log(cluster_prob.clamp_min(eps))).sum(dim=1)).mean()


def init_tc_stats():
    return {
        'tc_sum': 0.0,
        'tc_count': 0,
        'active_edges': 0,
        'mean_srs_sum': 0.0,
        'pair_dist_sum': 0.0,
        'balance_sum': 0.0,
        'balance_count': 0,
        'conf_sum': 0.0,
        'conf_count': 0,
        'entropy_sum': 0.0,
        'max_prob_sum': 0.0,
        'prob_count': 0,
    }


def update_tc_stats(stats, loss_tc, tc_info, loss_balance, loss_conf=None):
    active_edges = int(tc_info.get('active_edges', 0))
    if active_edges > 0:
        stats['tc_sum'] += float(loss_tc.detach().item())
        stats['tc_count'] += 1
        stats['active_edges'] += active_edges
        stats['mean_srs_sum'] += float(tc_info.get('mean_srs', 0.0))
        stats['pair_dist_sum'] += float(tc_info.get('pair_dist', 0.0))

    if loss_balance is not None:
        stats['balance_sum'] += float(loss_balance.detach().item())
        stats['balance_count'] += 1
    if loss_conf is not None:
        stats['conf_sum'] += float(loss_conf.detach().item())
        stats['conf_count'] += 1
    if tc_info.get('entropy', 0.0) > 0.0 or tc_info.get('max_prob', 0.0) > 0.0:
        stats['entropy_sum'] += float(tc_info.get('entropy', 0.0))
        stats['max_prob_sum'] += float(tc_info.get('max_prob', 0.0))
        stats['prob_count'] += 1


def format_tc_stats(stats):
    if stats['tc_count'] == 0 and stats['balance_count'] == 0 and stats['conf_count'] == 0:
        return ''

    tc_mean = stats['tc_sum'] / max(stats['tc_count'], 1)
    mean_srs = stats['mean_srs_sum'] / max(stats['tc_count'], 1) if stats['tc_count'] > 0 else 0.0
    pair_dist = stats['pair_dist_sum'] / max(stats['tc_count'], 1) if stats['tc_count'] > 0 else 0.0
    balance = stats['balance_sum'] / max(stats['balance_count'], 1) if stats['balance_count'] > 0 else 0.0

    conf = stats['conf_sum'] / max(stats['conf_count'], 1) if stats['conf_count'] > 0 else 0.0
    entropy = stats['entropy_sum'] / max(stats['prob_count'], 1) if stats['prob_count'] > 0 else 0.0
    max_prob = stats['max_prob_sum'] / max(stats['prob_count'], 1) if stats['prob_count'] > 0 else 0.0

    return (
        f", TC(loss={tc_mean:.6f}, active={stats['active_edges']}, "
        f"srs={mean_srs:.4f}, pairDist={pair_dist:.6f}, "
        f"ent={entropy:.4f}, maxP={max_prob:.4f}, "
        f"bal={balance:.4f}, conf={conf:.4f})"
    )

def encode_dataset(model, data_loader, device, graph_prior=None):
    model.eval()
    all_latent = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            batch_x, batch_y, batch_idx = unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_prior = get_batch_prior(graph_prior, batch_idx, device)

            outputs = model(
                batch_x,
                x_aug=None,
                is_training=False,
                B_local=batch_prior.B_local if batch_prior is not None else None,
                beta_scale=batch_prior.beta_scale if batch_prior is not None else None,
            )
            all_latent.append(outputs['latent'].cpu().numpy())
            all_labels.append(batch_y.cpu().numpy())

    return np.vstack(all_latent), np.hstack(all_labels)


def save_source_weights(model, graph_prior, config):
    """Save learned graph-source weights from the selected best checkpoint."""
    if graph_prior is None:
        return None

    attention = getattr(model, 'multihead_attention', None)
    rho_prior = getattr(attention, 'rho_prior', None)
    if rho_prior is None:
        return None

    with torch.no_grad():
        source_weights = torch.softmax(rho_prior.detach(), dim=1).cpu().numpy()

    graph_names = list(getattr(graph_prior, 'graphs', {}).keys())
    if len(graph_names) != source_weights.shape[1]:
        graph_names = [
            f'source_{source_idx + 1}'
            for source_idx in range(source_weights.shape[1])
        ]

    dataset_path = str(config.get('dataset_path') or 'dataset')
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    rows = []
    for head_idx, head_weights in enumerate(source_weights, start=1):
        row = {
            'dataset': dataset_name,
            'seed': int(config['seed']),
            'head': f'H{head_idx}',
        }
        row.update({
            f'G_{graph_name}': float(weight)
            for graph_name, weight in zip(graph_names, head_weights)
        })
        rows.append(row)

    source_weights_df = pd.DataFrame(rows)
    os.makedirs(config['save_dir'], exist_ok=True)
    source_weights_path = os.path.join(config['save_dir'], 'source_weights.csv')
    source_weights_df.to_csv(
        source_weights_path,
        index=False,
        encoding='utf-8-sig',
        float_format='%.8f',
    )

    print('\nPrior-head graph-source weights (best checkpoint):')
    print(
        source_weights_df.drop(columns=['dataset', 'seed']).to_string(
            index=False,
            float_format=lambda value: f'{value:.6f}',
        )
    )
    print(f'Source weights saved to: {source_weights_path}')

    return source_weights_path


def train_model(
    model,
    train_loader,
    test_loader,
    selection_loader,
    config,
    device,
    graph_prior=None,
):
    """Train the contrastive transformer autoencoder"""
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['weight_decay']
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs']
    )
    
    # Loss functions
    mse_criterion = nn.MSELoss()
    contrastive_criterion = ContrastiveLoss(temperature=config['temperature'])
    
    # Loss weights
    lambda_recon = config['lambda_recon']
    lambda_impute = config['lambda_impute']
    lambda_contrast = config['lambda_contrast']
    lambda_srs_tc = float(config.get('lambda_srs_tc', 0.0))
    lambda_balance = float(config.get('lambda_balance', 0.0))
    lambda_confidence = float(config.get('lambda_confidence', 0.0))


    best_loss = float('inf')
    best_score = -float('inf')
    best_state = None
    best_epoch = 0
    best_metrics = None
    select_metric = config['select_metric']
    cluster_eval_interval = max(1, int(config['cluster_eval_interval']))
    batch_log_interval = max(0, int(config.get('batch_log_interval', 10)))
    target_graph_beta = float(getattr(model.multihead_attention, 'graph_beta', 0.0))
    
    print("\n" + "="*70)
    print("Starting Training...")
    print("="*70)
    print(
        f"Loss Weights: Recon={lambda_recon}, Impute={lambda_impute}, "
        f"Contrast={lambda_contrast}, SRS_TC={lambda_srs_tc}, "
        f"Balance={lambda_balance}, Confidence={lambda_confidence}"
    )
    if graph_prior is not None:
        print(
            f"Graph Prior: enabled, free_heads={config['free_heads']}, "
            f"tau={config['tau']}, graph_beta={config['graph_beta']}, "
            f"warmup={config['graph_warmup_epochs']}, "
            f"fn_scale={config['fn_scale']:.1f}, "
            f"fn_gate={config.get('fn_gate_mode', 'threshold')}, "
            f"fn_risk={config.get('fn_risk_mode', 'availability')}, "
            f"fn_topk={config['fn_topk']}, fn_min_support={config['fn_min_support']}, "
            f"fn_min_risk={config['fn_min_risk']}, "
            f"tc_min_support={config['srs_tc_min_support']}, "
            f"tc_metric={config['srs_tc_metric']}, "
            f"tc_temp={config['srs_tc_temp']}"
        )
    else:
        print("Graph Prior: disabled")
    print(f"Checkpoint Selection: {select_metric}")
    
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        if graph_prior is not None:
            warmup_epochs = int(config['graph_warmup_epochs'])
            if warmup_epochs > 0:
                beta_scale = min(1.0, float(epoch + 1) / float(warmup_epochs))
                model.multihead_attention.graph_beta = target_graph_beta * beta_scale
            else:
                model.multihead_attention.graph_beta = target_graph_beta

        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_impute_loss = 0
        train_contrast_loss = 0
        train_tc_loss = 0.0
        train_balance_loss = 0.0
        train_conf_loss = 0.0
        attn_srs_stats = init_attention_srs_stats()
        tc_stats = init_tc_stats()
        bfn_stats = init_bfn_stats()

        total_batches = len(train_loader)
        for batch_number, batch in enumerate(train_loader, start=1):
            batch_x, batch_y, batch_idx = unpack_batch(batch)
            batch_x = batch_x.to(device)
            batch_prior = get_batch_prior(graph_prior, batch_idx, device)
            
            # Create augmented data
            batch_x_aug, gene_mask = create_augmented_data(
                batch_x, 
                mask_prob=config['mask_prob']
            )
            
            # Forward pass
            outputs = model(
                batch_x,
                batch_x_aug,
                is_training=True,
                B_local=batch_prior.B_local if batch_prior is not None else None,
                beta_scale=batch_prior.beta_scale if batch_prior is not None else None,
            )
            update_attention_srs_stats(attn_srs_stats, outputs, batch_prior)
            fn_weights, bfn_info = build_fn_weights(batch_prior, config, device)
            update_bfn_stats(bfn_stats, bfn_info)
            
            # Compute losses
            loss_recon = mse_criterion(outputs['reconstructed'], batch_x)
            loss_impute = mse_criterion(outputs['reconstructed_aug'], batch_x)
            loss_contrast = contrastive_criterion(
                outputs['z_orig'],
                outputs['z_aug'],
                fn_weights=fn_weights,
            )

            need_raw_tc = lambda_srs_tc > 0.0
            if need_raw_tc:
                raw_loss_tc, raw_tc_info = srs_topology_consistency_loss(
                    outputs.get('cluster_logits'),
                    batch_prior,
                    config,
                )
            else:
                raw_loss_tc = outputs['latent'].sum() * 0.0
                raw_tc_info = {'active_edges': 0, 'mean_srs': 0.0, 'pair_dist': 0.0}

            if lambda_srs_tc > 0.0:
                loss_srs_tc = raw_loss_tc
                tc_info = raw_tc_info
            else:
                loss_srs_tc = outputs['latent'].sum() * 0.0
                tc_info = {'active_edges': 0, 'mean_srs': 0.0, 'pair_dist': 0.0}

            cluster_prob_tc = sharpen_cluster_prob(outputs.get('cluster_logits'), config)

            if lambda_balance > 0.0:
                raw_loss_balance = cluster_balance_loss(cluster_prob_tc)
            else:
                raw_loss_balance = outputs['latent'].sum() * 0.0
            loss_balance = (
                raw_loss_balance
                if lambda_balance > 0.0
                else outputs['latent'].sum() * 0.0
            )

            if lambda_confidence > 0.0:
                raw_loss_conf = cluster_confidence_loss(cluster_prob_tc)
            else:
                raw_loss_conf = outputs['latent'].sum() * 0.0
            loss_conf = (
                raw_loss_conf
                if lambda_confidence > 0.0
                else outputs['latent'].sum() * 0.0
            )

            update_tc_stats(tc_stats, loss_srs_tc, tc_info, loss_balance, loss_conf)


            # Total loss
            loss = (lambda_recon * loss_recon + 
                   lambda_impute * loss_impute + 
                   lambda_contrast * loss_contrast +
                   lambda_srs_tc * loss_srs_tc +
                   lambda_balance * loss_balance +
                   lambda_confidence * loss_conf)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += loss_recon.item()
            train_impute_loss += loss_impute.item()
            train_contrast_loss += loss_contrast.item()
            train_tc_loss += loss_srs_tc.item()
            train_balance_loss += loss_balance.item()
            train_conf_loss += loss_conf.item()

            if batch_log_interval > 0 and (
                batch_number % batch_log_interval == 0
                or batch_number == total_batches
            ):
                print(
                    f"Epoch {epoch + 1}/{config['epochs']} "
                    f"Batch {batch_number}/{total_batches}: "
                    f"Total={loss.item():.4f}, "
                    f"Recon={loss_recon.item():.4f}, "
                    f"Impute={loss_impute.item():.4f}, "
                    f"Contrast={loss_contrast.item():.4f}, "
                    f"TC={loss_srs_tc.item():.6f}, "
                    f"Balance={loss_balance.item():.4f}, "
                    f"Confidence={loss_conf.item():.4f}"
                )
        
        scheduler.step()
        
        # Evaluation
        model.eval()
        test_loss = 0
        
        with torch.no_grad():
            for batch in test_loader:
                batch_x, _, batch_idx = unpack_batch(batch)
                batch_x = batch_x.to(device)
                batch_prior = get_batch_prior(graph_prior, batch_idx, device)
                outputs = model(
                    batch_x,
                    x_aug=None,
                    is_training=False,
                    B_local=batch_prior.B_local if batch_prior is not None else None,
                    beta_scale=batch_prior.beta_scale if batch_prior is not None else None,
                )
                
                loss = mse_criterion(outputs['reconstructed'], batch_x)
                test_loss += loss.item()
        
        # Save best model
        if select_metric == 'recon' and test_loss < best_loss:
            best_loss = test_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_epoch = epoch + 1
            best_metrics = {'recon_loss': float(test_loss)}

        metric_text = ""
        need_cluster_eval = (
            select_metric != 'recon'
            and ((epoch + 1) % cluster_eval_interval == 0 or epoch == config['epochs'] - 1)
        )
        if need_cluster_eval:
            latent_eval, labels_eval = encode_dataset(
                model,
                selection_loader,
                device,
                graph_prior=graph_prior,
            )
            from sklearn.cluster import KMeans
            pred_eval = KMeans(
                n_clusters=config['n_clusters'],
                random_state=config['seed'],
                n_init=20,
            ).fit_predict(latent_eval)
            acc_eval, nmi_eval, ari_eval = evaluate_clustering(labels_eval, pred_eval)
            scores = {
                'acc': acc_eval,
                'nmi': nmi_eval,
                'ari': ari_eval,
                'mean': (acc_eval + nmi_eval + ari_eval) / 3.0,
            }
            score = scores[select_metric]
            metric_text = (
                f", EvalACC={acc_eval:.4f}, "
                f"EvalNMI={nmi_eval:.4f}, EvalARI={ari_eval:.4f}"
            )
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch + 1
                best_metrics = {
                    'acc': float(acc_eval),
                    'nmi': float(nmi_eval),
                    'ari': float(ari_eval),
                    'mean': float(scores['mean']),
                }

        if (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            attn_text = format_attention_srs_stats(attn_srs_stats)
            tc_text = format_tc_stats(tc_stats)
            bfn_text = format_bfn_stats(bfn_stats)
            print(f"Epoch {epoch+1}/{config['epochs']}: "
                  f"Total={train_loss:.4f} "
                  f"(Recon={train_recon_loss:.4f}, "
                  f"Impute={train_impute_loss:.4f}, "
                  f"Contrast={train_contrast_loss:.4f}), "
                  f"Test={test_loss:.4f}{metric_text}{attn_text}{tc_text}{bfn_text}, Time={elapsed:.2f}s")


    total_time = time.time() - start_time
    model.multihead_attention.graph_beta = target_graph_beta


    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }
        best_epoch = int(config['epochs'])
        best_metrics = {'fallback': 'final_epoch'}

    model.load_state_dict(best_state)

    os.makedirs(config['save_dir'], exist_ok=True)
    checkpoint_path = os.path.join(config['save_dir'], 'best_model.pt')
    torch.save(
        {
            'model_state_dict': best_state,
            'config': dict(config),
            'best_epoch': int(best_epoch),
            'select_metric': str(select_metric),
            'best_score': (
                float(best_score)
                if select_metric != 'recon' and np.isfinite(best_score)
                else None
            ),
            'best_recon_loss': (
                float(best_loss)
                if select_metric == 'recon' and np.isfinite(best_loss)
                else None
            ),
            'best_metrics': best_metrics,
        },
        checkpoint_path,
    )
    print(f"Best model saved to: {checkpoint_path} (epoch={best_epoch})")

    print(f"\nTraining completed in {total_time:.2f} seconds")
    print("="*70)
    
    return total_time


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Main execution pipeline driven by command line arguments"""

    args = parse_args()

    # Build config from parsed args (all defaults already come from get_config())
    config = vars(args).copy()
    data_path = config.pop('dataset')
    config['dataset_path'] = data_path

    print(f"\n{'#'*70}")
    print(f"Running scTGCL on '{data_path}' dataset")
    print(f"{'#'*70}")
    print("\nActive Configuration:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\nRandom Seed: {config['seed']}")
    setup_seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading dataset: {data_path}")
    data = sc.read_h5ad(data_path)
    X_all = data.X
    y_raw = data.obs[config['label_key']].astype(str).to_numpy()
    y_all = y_raw.copy()
    
    print(f"Data shape: {X_all.shape}, Labels shape: {y_all.shape}")
    print(f"Cell type distribution:\n{pd.DataFrame(y_all).value_counts()}")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(pd.DataFrame(y_all).values.ravel())
    
    all_idx = np.arange(X_all.shape[0], dtype=np.int64)

    # Split data
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_all,
        y_all,
        all_idx,
        test_size=0.2,
        random_state=config['seed'],
        stratify=y_all,
    )
    
    # Create datasets and loaders
    use_graph_prior = bool(config['graph_dir'])
    train_dataset = scRNASeqDataset(
        X_train,
        y_train,
        indices=idx_train,
        return_index=use_graph_prior,
    )
    test_dataset = scRNASeqDataset(
        X_test,
        y_test,
        indices=idx_test,
        return_index=use_graph_prior,
    )
    full_dataset = scRNASeqDataset(
        X_all,
        y_all,
        indices=all_idx,
        return_index=use_graph_prior,
    )
    
    eval_batch_size = int(config['eval_batch_size'])
    if eval_batch_size <= 0:
        eval_batch_size = int(config['batch_size'])

    train_eval_loader = DataLoader(train_dataset, batch_size=eval_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    test_eval_loader = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False)
    full_loader = DataLoader(full_dataset, batch_size=eval_batch_size, shuffle=False)

    graph_prior = None
    if use_graph_prior:
        graph_prior = LocalGraphPrior(
            graph_dir=config['graph_dir'],
            n_cells=X_all.shape[0],
            rank_eps=config['rank_eps'],
            availability_support=config['availability_support'],
            beta_base=config['beta_base'],
        )
        config['fn_uncovered'] = estimate_uncovered(
            train_dataset,
            config['batch_size'],
            config['seed'],
            graph_prior,
            device,
        )
        threshold_scale = (
            1.0
            if config['fn_uncovered'] >= float(config['fn_threshold'])
            else 0.0
        )
        gate_mode = str(config.get('fn_gate_mode', 'threshold')).lower()
        if gate_mode == 'threshold':
            config['fn_scale'] = threshold_scale
        elif gate_mode == 'always':
            config['fn_scale'] = 1.0
        elif gate_mode == 'off':
            config['fn_scale'] = 0.0
        else:
            raise ValueError(f"Unsupported fn_gate_mode: {gate_mode}")
        print(
            f"Graph prior availability: unavailable={config['fn_uncovered']:.4f}, "
            f"threshold_decision={int(threshold_scale > 0)}, "
            f"gate_mode={gate_mode}, BFN enabled={int(config['fn_scale'] > 0)}, "
            f"risk_mode={config.get('fn_risk_mode', 'availability')}"
        )
    else:
        config['fn_scale'] = 0.0

    graph_batch_neighbors = int(config['graph_batch_neighbors'])
    if graph_prior is not None and graph_batch_neighbors > 0:
        train_sampler = GraphNeighborBatchSampler(
            dataset_indices=train_dataset.indices,
            graph_prior=graph_prior,
            batch_size=config['batch_size'],
            neighbors_per_anchor=graph_batch_neighbors,
            seed=config['seed'],
        )
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
        print(f"Graph-aware training batches: neighbors_per_anchor={graph_batch_neighbors}")
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
        )


    # Initialize model
    model = scTGCL(
        input_dim=X_all.shape[1],
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        latent_dim=config['latent_dim'],
        dropout=config['dropout'],
        mask_prob=config['mask_prob'],
        attention_mask_prob=config['attention_mask_prob'],
        free_heads=config['free_heads'],
        tau=config['tau'],
        graph_beta=config['graph_beta'],
        n_clusters=config['n_clusters'],
    ).to(device)
    
    print(f"\nModel Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    eval_loaders = {
        'train': train_eval_loader,
        'test': test_eval_loader,
        'full': full_loader,
    }
    selection_loader = eval_loaders[config['eval_scope']]

    training_time = train_model(
        model,
        train_loader,
        test_loader,
        selection_loader,
        config,
        device,
        graph_prior=graph_prior,
    )

    save_source_weights(
        model,
        graph_prior,
        config,
    )

    eval_loader = eval_loaders[config['eval_scope']]
    eval_indices = {
        'train': idx_train,
        'test': idx_test,
        'full': all_idx,
    }[config['eval_scope']]
    latent, labels = encode_dataset(
        model,
        eval_loader,
        device,
        graph_prior=graph_prior,
    )
    
    # Evaluate
    acc, nmi, ari, cluster_labels = evaluate(
        latent=latent,
        labels=labels,
        n_clusters=config['n_clusters'],
        seed=config['seed'],
        save_dir=config['save_dir']
    )

    os.makedirs(config['save_dir'], exist_ok=True)
    latent_path = os.path.join(config['save_dir'], 'latent.npy')
    np.save(latent_path, latent.astype(np.float32, copy=False))

    obs_names = np.asarray(data.obs_names.astype(str))
    metadata = pd.DataFrame({
        'cell_index': np.asarray(eval_indices, dtype=np.int64),
        'cell_id': obs_names[eval_indices],
        'reference_label': y_raw[eval_indices],
        'reference_label_encoded': labels.astype(np.int64),
        'predicted_cluster': cluster_labels.astype(np.int64),
    })
    metadata_path = os.path.join(config['save_dir'], 'latent_metadata.csv')
    metadata.to_csv(metadata_path, index=False, encoding='utf-8-sig')

    print(f'Latent embedding saved to: {latent_path}')
    print(f'Latent metadata saved to: {metadata_path}')

    return model, acc, nmi, ari


if __name__ == "__main__":
    main()