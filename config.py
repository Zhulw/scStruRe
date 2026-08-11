def get_config():
    """
    Get default configuration for scTGCL model.

    Current fixed anchor:
    SRS-RCB + SRS-TC(support>=2) + SRS-BFN(support>=1)

    Notes:
    - SRS-TC is a positive consistency constraint, so it uses reliable
      multi-source consensus edges: support >= 2.
    - SRS-BFN is a false-negative suppression mechanism, not a positive
      pulling constraint, so it can protect broader potential neighbors:
      support >= 1.
    """
    config = {
        # =========================================================
        # Model architecture
        # =========================================================
        'embed_dim': 512,
        'num_heads': 8,
        'free_heads': 4,
        'latent_dim': 64,
        'dropout': 0.3,

        # =========================================================
        # Data augmentation
        # =========================================================
        'mask_prob': 0.25,
        'attention_mask_prob': 0.35,

        # =========================================================
        # Main loss weights
        # =========================================================
        'lambda_recon': 6,
        'lambda_impute': 1,
        'lambda_contrast': 0.8,
        'temperature': 0.5,

        # =========================================================
        # Stage 1: SRS-guided topology consistency
        # Fixed anchor: support >= 2
        # =========================================================
        'lambda_srs_tc': 0.05,
        'lambda_balance': 0.01,
        'lambda_confidence': 0.005,

        'srs_tc_min_support': 2,
        'srs_tc_min_srs': 0.0,
        'srs_tc_metric': 'mse',
        'srs_tc_temp': 0.25,


        # =========================================================
        # Stage 0: SRS-RCB graph-prior attention
        # =========================================================
        'graph_dir': None,
        'tau': 0.5,
        'graph_beta': 0.5,
        'rank_eps': 0.1,

        # =========================================================
        # SRS reliability control
        # =========================================================
        # Batch-local reliable structural availability.
        # A cell is available when at least one support>=2 edge is retained
        # in the current mini-batch.
        'availability_support': 2,
        'beta_base': 0.2,

        # =========================================================
        # Stage 3: SRS-aware false-negative suppression
        # Fixed anchor: support >= 1
        # =========================================================
        'fn_lambda': 0.3,
        'fn_min': 0.7,
        'fn_topk': 5,
        'fn_threshold': 0.3,
        'fn_gate_mode': 'threshold',
        'fn_risk_mode': 'availability',

        'fn_min_support': 1,
        'fn_min_risk': 0.0,


        # Runtime fields, overwritten during training
        'fn_scale': 1.0,
        'fn_uncovered': 0.0,

        # =========================================================
        # Training
        # =========================================================
        'lr': 0.001,
        'weight_decay': 1e-3,
        'epochs': 160,
        'batch_size': 128,
        'seed': 42,

        # Graph-aware batch construction
        'graph_batch_neighbors': 4,
        'graph_warmup_epochs': 20,

        # Checkpoint and evaluation
        'select_metric': 'mean',
        'cluster_eval_interval': 10,
        'batch_log_interval': 10,
        'label_key': 'cell_type',
        'eval_scope': 'full',
        'eval_batch_size': 0,

        # Output
        'save_dir': 'results',
    }

    return config