from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch


def build_bfn_structure(
    batch_prior,
    config: Mapping,
    device: torch.device,
) -> Optional[Dict]:
    """Build the BFN pair-risk mask and symmetric negative weights."""
    if batch_prior is None:
        return None

    support = batch_prior.support.detach().to(device=device)
    srs = batch_prior.srs.detach().to(device=device, dtype=torch.float32)
    availability = batch_prior.availability.detach().to(
        device=device,
        dtype=srs.dtype,
    )

    batch_size = int(support.shape[0])
    diagonal = torch.eye(batch_size, dtype=torch.bool, device=device)
    off_diagonal = ~diagonal

    risk_mode = str(config.get("fn_risk_mode", "availability")).lower()
    if risk_mode == "availability":
        pair_availability = 0.5 * (
            availability.view(-1, 1) + availability.view(1, -1)
        )
        raw_risk = srs * pair_availability
    elif risk_mode == "srs":
        pair_availability = torch.ones_like(srs)
        raw_risk = srs
    else:
        raise ValueError(f"Unsupported fn_risk_mode: {risk_mode}")

    min_support = int(config.get("fn_min_support", 1))
    support_candidate = (support >= float(min_support)) & off_diagonal
    risk = torch.where(
        support_candidate,
        raw_risk,
        torch.zeros_like(raw_risk),
    )
    risk = risk.masked_fill(diagonal, 0.0)

    min_risk = float(config.get("fn_min_risk", 0.0))
    if min_risk > 0.0:
        risk = torch.where(
            risk >= min_risk,
            risk,
            torch.zeros_like(risk),
        )

    risk_eligible = (risk > 0.0) & off_diagonal
    row_selected = torch.zeros_like(off_diagonal)

    topk = min(
        int(config.get("fn_topk", 0)),
        max(batch_size - 1, 0),
    )
    if topk > 0 and int(risk_eligible.sum().item()) > 0:
        values, indices = torch.topk(risk, k=topk, dim=1)
        selected = values > 0.0
        rows = torch.arange(batch_size, device=device).view(-1, 1)
        rows = rows.expand_as(indices)
        row_selected[rows[selected], indices[selected]] = True

    symmetric_selected = (row_selected | row_selected.T) & off_diagonal

    fn_lambda = float(config.get("fn_lambda", 0.0))
    fn_min = float(config.get("fn_min", 0.0))
    scale = float(config.get("fn_scale", 0.0))

    applied_weights = torch.ones_like(risk)
    if scale > 0.0 and int(row_selected.sum().item()) > 0:
        selected_risk = risk[row_selected]
        selected_weights = torch.clamp(
            1.0 - fn_lambda * scale * selected_risk,
            min=fn_min,
            max=1.0,
        )
        applied_weights[row_selected] = selected_weights
        applied_weights = torch.minimum(
            applied_weights,
            applied_weights.T,
        )

    applied_weights = applied_weights.masked_fill(diagonal, 1.0)

    return {
        "support": support,
        "srs": srs,
        "availability": availability,
        "pair_availability": pair_availability,
        "risk": risk,
        "support_candidate": support_candidate,
        "risk_eligible": risk_eligible,
        "row_selected": row_selected,
        "symmetric_selected": symmetric_selected,
        "applied_weights": applied_weights,
        "gate_enabled": bool(scale > 0.0),
        "risk_mode": risk_mode,
    }


def summarize_build_info(structure: Optional[Mapping]) -> Dict:
    """Return compact BFN fields used by the training log."""
    empty = {
        "enabled": 0,
        "active_pairs": 0,
        "candidate_pairs": 0,
        "mean_risk": 0.0,
        "mean_weight": 1.0,
        "min_weight": 1.0,
    }
    if structure is None:
        return empty

    applied_weights = structure["applied_weights"]
    off_diagonal = ~torch.eye(
        applied_weights.shape[0],
        dtype=torch.bool,
        device=applied_weights.device,
    )
    active = (applied_weights < 0.999999) & off_diagonal
    active_count = int(active.sum().item())
    candidate_count = int(structure["risk_eligible"].sum().item())

    if active_count > 0:
        mean_risk = float(structure["risk"][active].mean().item())
        mean_weight = float(applied_weights[active].mean().item())
        min_weight = float(applied_weights[active].min().item())
    else:
        mean_risk = 0.0
        mean_weight = 1.0
        min_weight = 1.0

    return {
        "enabled": int(bool(structure["gate_enabled"])),
        "active_pairs": active_count,
        "candidate_pairs": candidate_count,
        "mean_risk": mean_risk,
        "mean_weight": mean_weight,
        "min_weight": min_weight,
    }
