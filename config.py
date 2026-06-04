"""
Configuration and data utilities — shared across all schemes.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    """All hyperparameters and dimensional settings."""
    seq_len: int = 126
    feature_dim: int = 32
    hidden_dim: int = 128
    num_sub_models: int = 8
    state_dim: int = 64
    score_window: int = 120
    learning_rate: float = 1e-3
    l2_reg: float = 1e-4
    batch_size: int = 32
    epochs: int = 50


# ── Data generation ─────────────────────────────────────────

def generate_synthetic_data(
    n_samples: int, config: Config, seed: int = 42
) -> tuple:
    rng = np.random.RandomState(seed)
    behavior = rng.randn(n_samples, config.seq_len, config.feature_dim).astype(np.float32)
    labels = rng.randint(0, 2, n_samples).astype(np.float32)
    tp_labels = rng.rand(n_samples, config.score_window).astype(np.float32)
    return behavior, labels, tp_labels


def generate_self_supervised_labels(
    behavior: np.ndarray, config: Config
) -> tuple:
    batch_size, seq_len, _ = behavior.shape
    diffs = np.diff(behavior, axis=1)
    intensity = np.linalg.norm(diffs, axis=-1)
    mn, mx = intensity.min(axis=1, keepdims=True), intensity.max(axis=1, keepdims=True)
    intensity = (intensity - mn) / (mx + 1e-8)

    final = (intensity.max(axis=1) > 0.5).astype(np.float32)

    sw = config.score_window
    if seq_len - 1 < sw:
        pad = np.zeros((batch_size, sw), dtype=np.float32)
        pad[:, :seq_len - 1] = intensity
        tp = pad
    else:
        tp = intensity[:, :sw].astype(np.float32)

    return final, tp
