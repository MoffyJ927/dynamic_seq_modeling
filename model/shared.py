"""
Shared network components — identical across all three schemes.

  StateCompressionModel  — BiLSTM × 2 encoder
  SubModel / MultiSubModels  — 8 parallel Conv1D detectors
  DecisionScoringModel  — LSTM + Attention for final scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from ..config import Config


# ── State Compression ───────────────────────────────────────

class StateCompressionModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.lstm1 = nn.LSTM(config.feature_dim, config.hidden_dim,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(config.hidden_dim * 2, config.hidden_dim // 2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm1(x); x = self.dropout1(x)
        x, _ = self.lstm2(x); x = self.dropout2(x)
        return self.output_proj(x)


# ── Sub-Models ──────────────────────────────────────────────

class SubModel(nn.Module):
    def __init__(self, config: Config, model_id: int):
        super().__init__()
        self.conv1d = nn.Conv1d(config.hidden_dim, config.state_dim, 3, padding=1)
        self.layer_norm = nn.LayerNorm(config.state_dim)
        self.output_layer = nn.Linear(config.state_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.relu(self.conv1d(x))
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return torch.sigmoid(self.output_layer(x).squeeze(-1))


class MultiSubModels(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.sub_models = nn.ModuleList([
            SubModel(config, i) for i in range(config.num_sub_models)])

    def forward(self, compressed: torch.Tensor) -> torch.Tensor:
        return torch.cat([sm(compressed).unsqueeze(-1) for sm in self.sub_models], dim=-1)


# ── Decision Scoring ────────────────────────────────────────

class DecisionScoringModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(1, 64, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.attention = nn.Linear(64, 1)
        self.final_score_layer = nn.Linear(64, 1)
        self.per_timepoint_layer = nn.Linear(64, config.score_window)

    def forward(self, risk_scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = risk_scores.unsqueeze(-1)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        attn = F.softmax(self.attention(x), dim=1)
        x = x * attn
        pooled = x.mean(dim=1)
        return torch.sigmoid(self.final_score_layer(pooled)), \
               torch.sigmoid(self.per_timepoint_layer(pooled))
