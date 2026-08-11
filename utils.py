import torch
import numpy as np
import pandas as pd
import random
import os
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import Dataset


# ============================================================================
# Seed and Reproducibility
# ============================================================================

def setup_seed(seed):
    """
    Set random seeds for reproducibility
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================================
# Data Augmentation
# ============================================================================

def create_augmented_data(x, mask_prob=0.15):
    """
    Create augmented data by randomly masking genes to zero
    
    Args:
        x: Input data [batch_size, n_genes]
        mask_prob: Probability of masking each gene
        
    Returns:
        x_aug: Augmented data with some genes masked to zero
        mask: Boolean mask indicating which genes were masked
    """
    mask = torch.bernoulli(torch.ones_like(x) * mask_prob).bool()
    x_aug = x.clone()
    x_aug[mask] = 0
    return x_aug, mask


def mask_attention_matrix(attention_matrix, mask_prob=0.2):
    """
    Randomly mask some cell-cell similarities in attention matrix
    This simulates uncertainty in cell relationships
    
    Args:
        attention_matrix: [num_heads, batch_size, batch_size]
        mask_prob: Probability of masking cell-cell similarity
        
    Returns:
        masked_attention: Attention matrix with some entries masked to 0
    """
    batch_size = attention_matrix.shape[1]
    
    # Don't mask diagonal (self-attention)
    mask = torch.bernoulli(torch.ones_like(attention_matrix) * (1 - mask_prob))
    
    # Keep diagonal intact
    for h in range(attention_matrix.shape[0]):
        mask[h].fill_diagonal_(1)
    
    masked_attention = attention_matrix * mask
    
    # Re-normalize after masking
    row_sums = masked_attention.sum(dim=-1, keepdim=True)
    row_sums = torch.clamp(row_sums, min=1e-8)
    masked_attention = masked_attention / row_sums
    
    return masked_attention


# ============================================================================
# Dataset
# ============================================================================

class scRNASeqDataset(Dataset):
    """PyTorch Dataset for scRNA-seq data, with optional global cell indices."""
    
    def __init__(self, expression_matrix, labels=None, indices=None, return_index=False):
        if isinstance(expression_matrix, pd.DataFrame):
            expression_matrix = expression_matrix.values

        if sparse.issparse(expression_matrix):
            self.data = expression_matrix.astype(np.float32)
        else:
            self.data = np.asarray(expression_matrix, dtype=np.float32)
        
        if labels is not None:
            self.labels = labels if isinstance(labels, np.ndarray) else labels.values
        else:
            self.labels = np.zeros(len(self.data))

        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.indices = (
            np.asarray(indices, dtype=np.int64)
            if indices is not None
            else np.arange(len(self.labels), dtype=np.int64)
        )
        self.return_index = bool(return_index)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if sparse.issparse(x):
            x = x.toarray().ravel()

        item = (
            torch.FloatTensor(x),
            torch.LongTensor([self.labels[idx]])[0],
        )

        if self.return_index:
            return item + (torch.LongTensor([self.indices[idx]])[0],)

        return item


# ============================================================================
# Evaluation Metrics
# ============================================================================

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy using Hungarian algorithm
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
        
    Returns:
        float: Clustering accuracy
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate_clustering(y_true, y_pred):
    """
    Evaluate clustering performance using multiple metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted cluster labels
        
    Returns:
        tuple: (accuracy, NMI, ARI, cluster_labels)
    """
    unique_labels = np.unique(y_true)
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_true_encoded = np.array([label_map[label] for label in y_true])
    
    acc = cluster_acc(y_true_encoded, y_pred)
    nmi = normalized_mutual_info_score(y_true_encoded, y_pred)
    ari = adjusted_rand_score(y_true_encoded, y_pred)
    
    return acc, nmi, ari


# ============================================================================
# Data Loading
# ============================================================================

def load_data(data_path, label_path=None):
    """
    Load scRNA-seq data from file
    
    Args:
        data_path: Path to expression data CSV
        label_path: Path to label CSV (optional)
        
    Returns:
        tuple: (expression_matrix, labels)
    """
    # Load expression data
    X_all = pd.read_csv(data_path, index_col=0).to_numpy()
    
    # Load labels if provided
    if label_path:
        y_all = pd.read_csv(label_path, index_col=0)
    else:
        y_all = None
    
    return X_all, y_all


def evaluate(latent, labels, n_clusters, seed=42, save_dir='results'):
    """
    Perform clustering and evaluation on latent representations
    
    Args:
        latent: Latent representations [n_samples, latent_dim]
        labels: True labels
        n_clusters: Number of clusters for K-means
        seed: Random seed
        save_dir: Directory to save results
        
    Returns:
        tuple: (accuracy, NMI, ARI, cluster_labels)
    """
    from sklearn.cluster import KMeans
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("\nPerforming clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20)
    cluster_labels = kmeans.fit_predict(latent)
    
    # Save cluster labels
    pd.DataFrame(cluster_labels, columns=['cluster_label']).to_csv(
        os.path.join(save_dir, "scTGCL_cluster_labels.csv"), index=False
    )
    
    # Evaluate
    acc, nmi, ari = evaluate_clustering(labels, cluster_labels)
    print(f"CA={acc:.4f}, NMI={nmi:.4f}, ARI={ari:.4f}")
    
    return acc, nmi, ari, cluster_labels
