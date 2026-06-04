"""
Scheme 3: Uncertainty-Weighted (Kendall et al. 2018).

RiskScoreMatrix uses learnable log-variance parameters — one per sub-model.
Weight ∝ exp(-log_var): sub-models with lower learned uncertainty
get higher weight. No external feedback needed.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .shared import StateCompressionModel, MultiSubModels, DecisionScoringModel
from ..config import Config


class RiskScoreMatrix(nn.Module):
    """Uncertainty-weighted aggregation.

    Each sub-model has a learnable log-variance. Weight ∝ exp(-log_var).
    """

    def __init__(self, config: Config):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(config.num_sub_models))

    def forward(
        self,
        sub_model_outputs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            sub_model_outputs: (batch, seq_len, num_sub_models)
        Returns:
            risk_scores: (batch, seq_len)
            weights:     (num_sub_models,) — global weights for interpretability
        """
        raw_weights = torch.exp(-self.log_vars)          # (M,)
        weights = raw_weights / raw_weights.sum()        # normalize
        weighted = sub_model_outputs * weights           # (B, T, M)
        risk_scores = weighted.mean(dim=-1)              # (B, T)
        return risk_scores, weights


# ── Full Model ───────────────────────────────────────────────

class SelfSupervisedStateTransitionRiskModel(nn.Module):
    """End-to-end model with uncertainty-based weighting."""

    def __init__(self, config: Config):
        super().__init__()
        self.state_compression = StateCompressionModel(config)
        self.multi_sub_models = MultiSubModels(config)
        self.risk_score_matrix = RiskScoreMatrix(config)
        self.decision_scoring = DecisionScoringModel(config)

    def forward(self, behavior_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        compressed = self.state_compression(behavior_sequence)
        sub_out = self.multi_sub_models(compressed)
        risk, weights = self.risk_score_matrix(sub_out)
        final, per_tp = self.decision_scoring(risk)
        return {
            'final_score': final,
            'per_timepoint_eval': per_tp,
            'risk_scores': risk,
            'sub_model_outputs': sub_out,
            'compressed_states': compressed,
            'weights': weights,                      # (M,) global per-sub-model
        }
