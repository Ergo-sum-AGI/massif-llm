# massif/observables.py

import torch
import numpy as np

def compute_persistence(h_sequence):
    """
    Compute persistence I_t for a sequence of hidden states.
    h_sequence: tensor of shape (T, d) - sequence of hidden states for one run
    Returns: tensor of shape (T-2,) - persistence values
    """
    eps = 1e-8
    # Normalize states
    h_norm = h_sequence / (h_sequence.norm(dim=-1, keepdim=True) + eps)
    # Velocity vectors
    v = h_norm[1:] - h_norm[:-1]
    v_norm = v / (v.norm(dim=-1, keepdim=True) + eps)
    # Persistence: cosine similarity of consecutive velocities
    persistence = (v_norm[1:] * v_norm[:-1]).sum(dim=-1)
    return persistence

def detect_flip(persistence, window=3):
    """
    Detect persistence flip: first step where mean of last `window` values > 0.
    Returns flip step index, or None if no flip detected.
    """
    for t in range(window, len(persistence)+1):
        if persistence[t-window:t].mean() > 0:
            return t
    return None

def compute_directional_alignment(h_ensemble):
    """
    Compute R_t (Kuramoto-inspired directional alignment) across ensemble.
    h_ensemble: tensor of shape (N, T, d) - N runs, T steps, d dimensions
    Returns: tensor of shape (T-2,) - R_t values
    """
    eps = 1e-8
    N, T, d = h_ensemble.shape
    h_norm = h_ensemble / (h_ensemble.norm(dim=-1, keepdim=True) + eps)
    v = h_norm[:, 1:] - h_norm[:, :-1]  # (N, T-1, d)
    u = v / (v.norm(dim=-1, keepdim=True) + eps)  # unit velocity (N, T-1, d)
    mean_u = u.mean(dim=0)  # (T-1, d)
    R_t = mean_u.norm(dim=-1)  # (T-1,)
    return R_t

def compute_curvature(h_sequence):
    """
    Compute mean curvature kappa for a hidden state sequence.
    Returns scalar mean curvature in radians.
    """
    persistence = compute_persistence(h_sequence)
    # Clamp to valid arccos range
    persistence_clamped = persistence.clamp(-1+1e-6, 1-1e-6)
    curvature = torch.arccos(persistence_clamped)
    return curvature.mean().item()

def compute_radial_variance(h_sequence):
    """
    Compute radial variance sigma_rho and CV for a hidden state sequence.
    """
    norms = h_sequence.norm(dim=-1)
    sigma = norms.std().item()
    mu = norms.mean().item()
    cv = sigma / (mu + 1e-8)
    return sigma, cv

def compute_lead_lag(R_t, flip_times):
    """
    Compute Delta_t = t_V_peak - mean_flip_time.
    R_t: directional alignment sequence
    flip_times: list of flip step indices across ensemble
    Returns: Delta_t scalar
    """
    V_t = torch.diff(R_t)
    t_V_peak = V_t.argmax().item()
    mean_flip = np.mean([f for f in flip_times if f is not None]) if any(
        f is not None for f in flip_times) else None
    if mean_flip is None:
        return None
    return t_V_peak - mean_flip

def compute_tau_eff(tau_collapse, flip_rate):
    """Effective warning: actionable precursor steps weighted by reliability."""
    return tau_collapse * (1 - flip_rate)