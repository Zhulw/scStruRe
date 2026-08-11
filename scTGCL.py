import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import mask_attention_matrix


# ============================================================================
# Contrastive Loss
# ============================================================================

class ContrastiveLoss(nn.Module):
    """Symmetric contrastive loss with optional false-negative down-weighting."""
    
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, z_orig, z_aug, fn_weights=None):
        """
        Args:
            z_orig: Original embeddings [batch_size, embed_dim]
            z_aug: Augmented embeddings [batch_size, embed_dim]
            
        Returns:
            fn_weights: Optional [batch_size, batch_size] weights. Values below
                1.0 reduce the contribution of likely false negatives.
        """
        batch_size = z_orig.shape[0]
        
        # Normalize embeddings
        z_orig = F.normalize(z_orig, dim=1)
        z_aug = F.normalize(z_aug, dim=1)
        
        # Compute similarity matrices
        sim_orig_aug = torch.mm(z_orig, z_aug.T) / self.temperature
        sim_orig_orig = torch.mm(z_orig, z_orig.T) / self.temperature
        sim_aug_aug = torch.mm(z_aug, z_aug.T) / self.temperature
        
        # Positive pairs: (i, i) - same cell in original and augmented
        positive_sim = torch.diag(sim_orig_aug)
        
        # Compute loss for original -> augmented
        exp_sim_orig_aug = torch.exp(sim_orig_aug)
        exp_sim_orig_orig = torch.exp(sim_orig_orig)
        
        # Remove diagonal from negative samples
        mask = torch.eye(batch_size, dtype=torch.bool, device=z_orig.device)
        exp_sim_orig_orig = exp_sim_orig_orig.masked_fill(mask, 0)

        if fn_weights is not None:
            weights = fn_weights.detach().to(device=z_orig.device, dtype=z_orig.dtype)
            weights = weights.masked_fill(mask, 1.0)
            exp_sim_orig_aug = exp_sim_orig_aug * weights
            exp_sim_orig_orig = exp_sim_orig_orig * weights
        
        denominator = exp_sim_orig_aug.sum(dim=1) + exp_sim_orig_orig.sum(dim=1)
        loss_orig_to_aug = -torch.log(torch.exp(positive_sim) / (denominator + 1e-8))
        
        # Compute loss for augmented -> original (symmetric)
        exp_sim_aug_orig = torch.exp(sim_orig_aug.T)
        exp_sim_aug_aug = torch.exp(sim_aug_aug)
        exp_sim_aug_aug = exp_sim_aug_aug.masked_fill(mask, 0)

        if fn_weights is not None:
            exp_sim_aug_orig = exp_sim_aug_orig * weights
            exp_sim_aug_aug = exp_sim_aug_aug * weights
        
        denominator_aug = exp_sim_aug_orig.sum(dim=1) + exp_sim_aug_aug.sum(dim=1)
        loss_aug_to_orig = -torch.log(torch.exp(positive_sim) / (denominator_aug + 1e-8))
        
        # Total contrastive loss
        loss = (loss_orig_to_aug.mean() + loss_aug_to_orig.mean()) / 2
        
        return loss


# ============================================================================
# Multi-Head Attention
# ============================================================================

class EnhancedMultiHeadAttention(nn.Module):
    """Multi-head self-attention with optional multi-source graph prior injection."""
    
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        free_heads=4,
        tau=0.5,
        graph_beta=0.5,
    ):
        super(EnhancedMultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert 0 <= free_heads <= num_heads, "free_heads must be in [0, num_heads]"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.free_heads = int(free_heads)
        self.prior_heads = int(num_heads - free_heads)
        self.head_dim = embed_dim // num_heads
        self.tau = float(tau)
        self.graph_beta = float(graph_beta)
        
        # Weight matrices for Q, K, V
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.scale = np.sqrt(self.head_dim)

        if self.prior_heads > 0:
            self.rho_prior = nn.Parameter(torch.zeros(self.prior_heads, 4))
        else:
            self.register_parameter("rho_prior", None)
        
    def forward(
        self,
        x,
        mask_attention=False,
        attention_mask_prob=0.2,
        B_local=None,
        beta_scale=None,
    ):
        """
        Args:
            x: [batch_size, embed_dim]
            mask_attention: Whether to apply attention masking
            attention_mask_prob: Probability of masking attention weights
            B_local: Optional [4, batch_size, batch_size] rank-bias tensors
            beta_scale: Optional [batch_size, batch_size] prior reliability scale
            
        Returns:
            output: [batch_size, embed_dim]
            attention_weights: [num_heads, batch_size, batch_size]
            qkv_matrices: Dictionary containing Q, K, V matrices
        """
        batch_size = x.shape[0]
        
        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        K = K.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        V = V.view(batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        
        # Free heads use standard scaled dot-product attention.
        # Prior heads use cosine-calibrated content logits when graph priors exist,
        # so the graph bias is not overwhelmed by uncontrolled QK magnitudes.
        dot_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        attention_scores = dot_scores

        pi_prior = None
        prior_bias = None
        prior_content_scores = None
        has_prior = (
            self.prior_heads > 0
            and B_local is not None
            and beta_scale is not None
        )

        if has_prior:
            pi_prior = F.softmax(self.rho_prior, dim=1)
            B_local = B_local.to(device=x.device, dtype=x.dtype)
            beta_scale = beta_scale.to(device=x.device, dtype=x.dtype)

            prior_bias = torch.einsum("hm,mbn->hbn", pi_prior, B_local)
            prior_bias = prior_bias * beta_scale.unsqueeze(0)

            Q_prior = F.normalize(Q[self.free_heads:], p=2, dim=-1)
            K_prior = F.normalize(K[self.free_heads:], p=2, dim=-1)
            prior_content_scores = self.tau * torch.bmm(
                Q_prior,
                K_prior.transpose(1, 2),
            )
            prior_scores = prior_content_scores + self.graph_beta * prior_bias

            if self.free_heads > 0:
                attention_scores = torch.cat(
                    [dot_scores[:self.free_heads], prior_scores],
                    dim=0,
                )
            else:
                attention_scores = prior_scores

        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply masking for augmented data
        if mask_attention:
            attention_weights = mask_attention_matrix(attention_weights, attention_mask_prob)
        
        # Apply attention to values
        attention_output = torch.bmm(attention_weights, V)
        attention_output = attention_output.transpose(0, 1).contiguous()
        attention_output = attention_output.view(batch_size, self.embed_dim)
        
        # Final linear projection
        output = self.W_o(attention_output)
        
        # Store Q, K, V for analysis
        qkv_matrices = {
            'Q': Q.transpose(0, 1),
            'K': K.transpose(0, 1),
            'V': V.transpose(0, 1),
        }

        attention_diag = {
            "A_free": attention_weights[:self.free_heads].mean(dim=0)
            if self.free_heads > 0
            else torch.zeros(batch_size, batch_size, device=x.device, dtype=x.dtype),
            "A_prior": attention_weights[self.free_heads:].mean(dim=0)
            if self.prior_heads > 0
            else torch.zeros(batch_size, batch_size, device=x.device, dtype=x.dtype),
            "pi_prior": pi_prior,
            "prior_bias": prior_bias,
            "prior_content_scores": prior_content_scores,
        }
        
        return output, attention_weights, qkv_matrices, attention_diag


# ============================================================================
# Main Model
# ============================================================================

class scTGCL(nn.Module):
    """Transformer-based Autoencoder with Contrastive Learning"""
    
    def __init__(
        self,
        input_dim,
        embed_dim=512,
        num_heads=8,
        latent_dim=128,
        dropout=0.1,
        mask_prob=0.15,
        attention_mask_prob=0.2,
        free_heads=4,
        tau=0.5,
        graph_beta=0.5,
        n_clusters=None,
    ):
        super(scTGCL, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.n_clusters = int(n_clusters) if n_clusters is not None else None
        self.mask_prob = mask_prob
        self.attention_mask_prob = attention_mask_prob
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 2048),
            nn.Linear(2048, embed_dim),
        )
        
        # Multi-head attention
        self.multihead_attention = EnhancedMultiHeadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads,
            free_heads=free_heads,
            tau=tau,
            graph_beta=graph_beta,
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embed_dim)
        
        # Encoder output (latent space)
        self.encoder_output = nn.Sequential(
            nn.Linear(embed_dim, latent_dim),
        )
        
        # Cluster assignment head for SRS-guided topology consistency.
        # It is unsupervised: ground-truth labels are not used in this head.
        self.cluster_head = (
            nn.Linear(latent_dim, self.n_clusters)
            if self.n_clusters is not None and self.n_clusters > 0
            else None
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(latent_dim, embed_dim // 2),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, embed_dim),
            nn.Linear(embed_dim, 2048),
            nn.Linear(2048, input_dim)
        )
        
    def forward(self, x, x_aug=None, is_training=True, B_local=None, beta_scale=None):
        """
        Args:
            x: Original input [batch_size, input_dim]
            x_aug: Augmented input [batch_size, input_dim] (optional)
            is_training: Whether in training mode
            
        Returns:
            Dictionary containing all outputs
        """
        # Process original data
        embedded = self.input_projection(x)
        attn_output, attn_weights, qkv_orig, diag_orig = self.multihead_attention(
            embedded,
            mask_attention=False,
            B_local=B_local,
            beta_scale=beta_scale,
        )
        
        # Residual + LayerNorm
        x1 = self.ln1(embedded + attn_output)
        
        # Latent representation
        latent = self.encoder_output(x1)
        
        # Optional clustering distribution for topology consistency.
        cluster_logits = self.cluster_head(latent) if self.cluster_head is not None else None
        cluster_prob = F.softmax(cluster_logits, dim=1) if cluster_logits is not None else None
        
        # Projection for contrastive learning
        z_orig = self.projection_head(latent)
        
        # Reconstruction
        reconstructed = self.decoder(latent)
        
        outputs = {
            'reconstructed': reconstructed,
            'latent': latent,
            'z_orig': z_orig,
            'cluster_logits': cluster_logits,
            'cluster_prob': cluster_prob,
            'x2_orig': x1,
            'attention_weights_orig': attn_weights,
            'qkv_orig': qkv_orig,
            'attention_diag_orig': diag_orig,
        }
        
        # Process augmented data if provided
        if x_aug is not None and is_training:
            embedded_aug = self.input_projection(x_aug)
            attn_output_aug, attn_weights_aug, qkv_aug, diag_aug = self.multihead_attention(
                embedded_aug,
                mask_attention=True,
                attention_mask_prob=self.attention_mask_prob,
                B_local=B_local,
                beta_scale=beta_scale,
            )
            
            x1_aug = self.ln1(embedded_aug + attn_output_aug)
            latent_aug = self.encoder_output(x1_aug)
            cluster_logits_aug = self.cluster_head(latent_aug) if self.cluster_head is not None else None
            cluster_prob_aug = F.softmax(cluster_logits_aug, dim=1) if cluster_logits_aug is not None else None
            z_aug = self.projection_head(latent_aug)
            reconstructed_aug = self.decoder(latent_aug)
            
            outputs.update({
                'reconstructed_aug': reconstructed_aug,
                'latent_aug': latent_aug,
                'z_aug': z_aug,
                'cluster_logits_aug': cluster_logits_aug,
                'cluster_prob_aug': cluster_prob_aug,
                'x2_aug': x1_aug,
                'attention_weights_aug': attn_weights_aug,
                'qkv_aug': qkv_aug,
                'attention_diag_aug': diag_aug,
            })
        
        return outputs
