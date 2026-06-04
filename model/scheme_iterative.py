"""
Scheme 2: Iterative Two-Stage.

RiskScoreMatrix has no learnable parameters — feedback weights are
provided externally and updated between training rounds based on
sub-model importance (output variance or gradient magnitude).
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple

from .shared import StateCompressionModel, MultiSubModels, DecisionScoringModel
from ..config import Config


class RiskScoreMatrix(nn.Module):
    """Static feedback-weighted aggregation.

    No learnable parameters — weights are supplied externally each round.
    """

    def __init__(self, config: Config):
        super().__init__()

    def forward(
        self,
        sub_model_outputs: torch.Tensor,
        feedback,
    ) -> torch.Tensor:
        """
        Args:
            sub_model_outputs: (batch, seq_len, num_sub_models)
            feedback:          (batch, num_sub_models) numpy or torch
        Returns:
            risk_scores: (batch, seq_len)
        """
        if isinstance(feedback, np.ndarray):
            feedback = torch.FloatTensor(feedback).to(sub_model_outputs.device)
        feedback_expanded = feedback.unsqueeze(1)        # (B, 1, M)
        weighted = sub_model_outputs * feedback_expanded # (B, T, M)
        return weighted.mean(dim=-1)                     # (B, T)


# ── Full Model ───────────────────────────────────────────────

class SelfSupervisedStateTransitionRiskModel(nn.Module):
    """End-to-end model with externally-supplied feedback weights.

    forward() requires a `feedback` tensor as second argument.
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
        feedback: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        compressed = self.state_compression(behavior_sequence)
        sub_out = self.multi_sub_models(compressed)
        risk = self.risk_score_matrix(sub_out, feedback)
        final, per_tp = self.decision_scoring(risk)
        return {
            'final_score': final,
            'per_timepoint_eval': per_tp,
            'risk_scores': risk,
            'sub_model_outputs': sub_out,
            'compressed_states': compressed,
        }
