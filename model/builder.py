"""
Factory to build a full SelfSupervisedStateTransitionRiskModel by scheme name.

Schemes:
  'gating'       — context-aware per-timestep gate network
  'iterative'    — static feedback weights, updated between rounds
  'uncertainty'  — learnable log-variance per sub-model (Kendall et al.)
"""

from ..config import Config


SCHEME_NAMES = ('gating', 'iterative', 'uncertainty')


def build_model(scheme: str, config: Config):
    """Build and return the model for the given scheme."""
    if scheme == 'gating':
        from .scheme_gating import SelfSupervisedStateTransitionRiskModel
    elif scheme == 'iterative':
        from .scheme_iterative import SelfSupervisedStateTransitionRiskModel
    elif scheme == 'uncertainty':
        from .scheme_uncertainty import SelfSupervisedStateTransitionRiskModel
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from {SCHEME_NAMES}")
    return SelfSupervisedStateTransitionRiskModel(config)


def get_risk_matrix_class(scheme: str):
    """Return the RiskScoreMatrix class for the given scheme (for inspection)."""
    if scheme == 'gating':
        from .scheme_gating import RiskScoreMatrix
    elif scheme == 'iterative':
        from .scheme_iterative import RiskScoreMatrix
    elif scheme == 'uncertainty':
        from .scheme_uncertainty import RiskScoreMatrix
    else:
        raise ValueError(f"Unknown scheme '{scheme}'. Choose from {SCHEME_NAMES}")
    return RiskScoreMatrix
