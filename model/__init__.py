"""
dynamic_seq_modeling.model — network components

  shared.py             — StateCompressionModel, SubModel, MultiSubModels, DecisionScoringModel
  scheme_gating.py      — RiskScoreMatrix with context-aware gate network
  scheme_iterative.py   — RiskScoreMatrix with static feedback weights
  scheme_uncertainty.py — RiskScoreMatrix with learnable log-variance
  builder.py            — factory to assemble full model per scheme
"""

from .shared import (
    StateCompressionModel,
    SubModel,
    MultiSubModels,
    DecisionScoringModel,
)
from .builder import build_model, get_risk_matrix_class, SCHEME_NAMES
