"""
Configuration and data utilities for the Self-Supervised State Transition Risk Model.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    """All hyperparameters and dimensional settings."""

    # ---- Sequence ----
    seq_len: int = 126              # Day 1 ~ Day 126
    feature_dim: int = 32           # daily behavior feature dimension
    hidden_dim: int = 128           # LSTM hidden dimension

    # ---- Sub-models ----
    num_sub_models: int = 8         # number of parallel sub-models
    state_dim: int = 64             # per sub-model output dimension

    # ---- Scoring ----
    score_window: int = 120         # evaluation window (Day 1 ~ Day 120)

    # ---- Training ----
    learning_rate: float = 1e-3
    l2_reg: float = 1e-4
    batch_size: int = 32
    epochs: int = 50


# ==========================================
# Data utilities
# ==========================================

def generate_synthetic_data(
    n_samples: int,
    config: Config,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic behavioral sequences + labels for demo / testing.

    Returns:
        behavior_sequences: (n_samples, seq_len, feature_dim)
        true_labels:        (n_samples,)  0/1 final risk event
        timepoint_labels:   (n_samples, score_window) per-timestep risk state
    """
    rng = np.random.RandomState(seed)
    behavior_sequences = rng.randn(n_samples, config.seq_len, config.feature_dim).astype(np.float32)
    true_labels = rng.randint(0, 2, n_samples).astype(np.float32)
    timepoint_labels = rng.rand(n_samples, config.score_window).astype(np.float32)
    return behavior_sequences, true_labels, timepoint_labels


def generate_self_supervised_labels(
    behavior_sequences: np.ndarray,
    config: Config,
) -> tuple:
    """
    Generate self-supervised pseudo-labels based on behavior change intensity.

    Strategy:
      - Compute L2 norm of adjacent timestep differences
      - High change → higher risk probability

    Returns:
        final_labels:      (n_samples,)  binary final risk label
        timepoint_labels:  (n_samples, score_window) per-timestep risk state
    """
    batch_size, seq_len, feature_dim = behavior_sequences.shape

    # Adjacent timestep differences
    diffs = np.diff(behavior_sequences, axis=1)  # (batch, seq_len-1, feature_dim)

    # L2 norm as change intensity
    change_intensity = np.linalg.norm(diffs, axis=-1)  # (batch, seq_len-1)

    # Normalize to [0, 1]
    min_v = change_intensity.min(axis=1, keepdims=True)
    max_v = change_intensity.max(axis=1, keepdims=True)
    change_intensity = (change_intensity - min_v) / (max_v + 1e-8)

    # Final label: whether max change exceeds threshold
    final_labels = (change_intensity.max(axis=1) > 0.5).astype(np.float32)

    # Timepoint labels: truncate/pad to score_window
    sw = config.score_window
    if seq_len - 1 < sw:
        padded = np.zeros((batch_size, sw), dtype=np.float32)
        padded[:, :seq_len - 1] = change_intensity
        timepoint_labels = padded
    else:
        timepoint_labels = change_intensity[:, :sw].astype(np.float32)

    return final_labels, timepoint_labels
