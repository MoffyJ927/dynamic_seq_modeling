"""
Scheme 1: Context-Aware Gating.

RiskScoreMatrix uses a gate network that takes compressed states as input
and produces per-timestep softmax weights over sub-models.
Replaces static feedback with learnable, state-dependent weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .shared import StateCompressionModel, MultiSubModels, DecisionScoringModel
from ..config import Config


class RiskScoreMatrix(nn.Module):
    """Context-aware dynamic weighting of sub-model outputs.

    A gating network takes the compressed states as input and produces
    a softmax distribution over sub-models PER TIMESTEP.
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
            weights:     (batch, seq_len, num_sub_models)
        """
        logits = self.gate(compressed_states) / self.temperature
        weights = F.softmax(logits, dim=-1)          # (B, T, M)
        weighted = sub_model_outputs * weights       # (B, T, M)
        risk_scores = weighted.mean(dim=-1)          # (B, T)
        return risk_scores, weights


# ── Full Model ───────────────────────────────────────────────

class SelfSupervisedStateTransitionRiskModel(nn.Module):
    """End-to-end model with context-aware gating."""

    def __init__(self, config: Config):
        super().__init__()
        self.state_compression = StateCompressionModel(config)
        self.multi_sub_models = MultiSubModels(config)
        self.risk_score_matrix = RiskScoreMatrix(config)
        self.decision_scoring = DecisionScoringModel(config)

    def forward(self, behavior_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
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
            'weights': weights,                     # (B, T, M) per-timestep
        }
