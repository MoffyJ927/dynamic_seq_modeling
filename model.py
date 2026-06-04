"""
Core model architecture for the Self-Supervised State Transition Risk Model.

Components:
  1. StateCompressionModel  – BiLSTM encoder
  2. SubModel / MultiSubModels  – 8 parallel Conv1D sub-models
  3. RiskScoreMatrix  – feedback-weighted aggregation (static, non-learnable)
  4. DecisionScoringModel  – LSTM + attention for final scoring
  5. SelfSupervisedStateTransitionRiskModel  – end-to-end model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple

from .config import Config


# ==========================================
# Step 1: State Compression
# ==========================================

class StateCompressionModel(nn.Module):
    """BiLSTM-based sequence encoder."""

    def __init__(self, config: Config):
        super().__init__()
        self.lstm1 = nn.LSTM(
            config.feature_dim, config.hidden_dim,
            batch_first=True, bidirectional=True,
        )
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(
            config.hidden_dim * 2, config.hidden_dim // 2,
            batch_first=True, bidirectional=True,
        )
        self.dropout2 = nn.Dropout(0.2)
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        return self.output_proj(x)


# ==========================================
# Step 2: Sub-Models
# ==========================================

class SubModel(nn.Module):
    """Single sub-model: Conv1D → sigmoid."""

    def __init__(self, config: Config, model_id: int):
        super().__init__()
        self.conv1d = nn.Conv1d(
            config.hidden_dim, config.state_dim, 3, padding=1,
        )
        self.layer_norm = nn.LayerNorm(config.state_dim)
        self.output_layer = nn.Linear(config.state_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)               # (B, C, T)
        x = F.relu(self.conv1d(x))
        x = x.transpose(1, 2)               # (B, T, C)
        x = self.layer_norm(x)
        return torch.sigmoid(self.output_layer(x).squeeze(-1))


class MultiSubModels(nn.Module):
    """Parallel sub-models."""

    def __init__(self, config: Config):
        super().__init__()
        self.sub_models = nn.ModuleList([
            SubModel(config, i) for i in range(config.num_sub_models)
        ])

    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [sm(compressed).unsqueeze(-1) for sm in self.sub_models], dim=-1,
        )   # (B, T, num_sub_models)


# ==========================================
# Step 3: Risk Score Matrix (Scheme 2: Context-aware gating)
# ==========================================

class RiskScoreMatrix(nn.Module):
    """
    Context-aware dynamic weighting of sub-model outputs.

    A gating network takes the compressed states as input and produces
    a softmax distribution over sub-models PER TIMESTEP.
    This replaces the static feedback weights with learnable, state-dependent weights.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_sub_models),
        )
        self.temperature = nn.Parameter(torch.tensor(2.0))

    def forward(
        self,
        sub_model_outputs: torch.Tensor,
        compressed_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sub_model_outputs: (batch, seq_len, num_sub_models)
            compressed_states: (batch, seq_len, hidden_dim)
        Returns:
            risk_scores: (batch, seq_len)
            weights:     (batch, seq_len, num_sub_models) — for interpretability
        """
        # Gate network: state → weight distribution per timestep
        logits = self.gate(compressed_states) / self.temperature  # (B, T, M)
        weights = F.softmax(logits, dim=-1)                        # (B, T, M)

        weighted = sub_model_outputs * weights                     # (B, T, M)
        risk_scores = weighted.mean(dim=-1)                        # (B, T)
        return risk_scores, weights


# ==========================================
# Step 4: Decision Scoring Model
# ==========================================

class DecisionScoringModel(nn.Module):
    """
    LSTM + attention for final decision scoring.
    Outputs both a global score and per-timestep evaluation.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Linear(64, 1)
        self.final_score_layer = nn.Linear(64, 1)
        self.per_timepoint_layer = nn.Linear(64, config.score_window)

    def forward(self, risk_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            risk_scores: (batch, score_window)
        Returns:
            final_score:        (batch, 1)
            per_timepoint_eval: (batch, score_window)
        """
        x = risk_scores.unsqueeze(-1)           # (B, T, 1)
        x, _ = self.lstm(x)
        x = self.dropout(x)                     # (B, T, 64)

        # Attention weights over timesteps
        attn = F.softmax(self.attention(x), dim=1)  # (B, T, 1)
        x = x * attn                              # (B, T, 64)

        # Global pooling
        pooled = x.mean(dim=1)                    # (B, 64)

        final_score = torch.sigmoid(self.final_score_layer(pooled))
        per_timepoint_eval = torch.sigmoid(self.per_timepoint_layer(pooled))
        return final_score, per_timepoint_eval


# ==========================================
# Step 5: End-to-End Model
# ==========================================

class SelfSupervisedStateTransitionRiskModel(nn.Module):
    """
    Full self-supervised model with context-aware gating:
      behavior_seq → compression → sub-models → gated risk matrix → decision scoring
    """

    def __init__(self, config: Config):
        super().__init__()
        self.state_compression = StateCompressionModel(config)
        self.multi_sub_models = MultiSubModels(config)
        self.risk_score_matrix = RiskScoreMatrix(config)
        self.decision_scoring = DecisionScoringModel(config)

    def forward(
        self,
        behavior_sequence: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            behavior_sequence: (batch, seq_len, feature_dim)
            # NOTE: `feedback` removed — replaced by learnable gating network
        Returns:
            dict with keys: final_score, per_timepoint_eval,
                            risk_scores, sub_model_outputs, compressed_states, weights
        """
        compressed = self.state_compression(behavior_sequence)
        sub_out = self.multi_sub_models(compressed)
        risk, weights = self.risk_score_matrix(sub_out, compressed)
        final, per_tp = self.decision_scoring(risk)

        return {
            'final_score': final,
            'per_timepoint_eval': per_tp,
            'risk_scores': risk,
            'sub_model_outputs': sub_out,
            'compressed_states': compressed,
            'weights': weights,  # new: per-timestep sub-model weights
        }
