# Self-Learning State Transition Risk Model

Self-supervised architecture for long time-window state transition risk assessment. 

```
原始行为序列 → 状态压缩 → 子模型×8 → 风险评分矩阵 → 决策打分
   Day 1~126      BiLSTM      Conv1D    (Gating Net)    LSTM+Attention
```

---

## Quick Start

```python
from self_learning_modeling.config import Config, generate_synthetic_data
from self_learning_modeling.train import SelfSupervisedTrainer
from self_learning_modeling.predict import Predictor

cfg = Config()  # seq_len=126, feature_dim=32, 8 sub-models, score_window=120

X, _, _ = generate_synthetic_data(1000, cfg)  # replace with your data

trainer = SelfSupervisedTrainer(cfg, device='cuda')
trainer.train(X, _, _, epochs=50, batch_size=64)
trainer.save('model.pt')

pred = Predictor.load('model.pt')
result = pred.predict(X[0])
# result['final_score']         → global risk score (0~1)
# result['risk_scores']          → per-timestep risk (126,)
# result['weights']              → per-timestep sub-model weights (126, 8)
```

---

## Architecture

```
behavior_sequence (B, 126, 32)
        │
        ▼
┌───────────────────┐
│ State Compression │  BiLSTM × 2 → (B, 126, 128)
│    (BiLSTM)       │
└───────────────────┘
        │
        ▼
┌───────────────────
│  Sub-Models × 8   │  Conv1D + LayerNorm + Sigmoid → (B, 126, 8)
│   (Conv1D)        │  each detects different risk patterns
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Risk Score Matrix │  Gating net: compressed_states → softmax weights
│   (Gating Net)    │  weights = σ(gate(state)) → (B, 126, 8)
│                   │  risk = weighted mean → (B, 126)
└───────────────────┘
        │
        ▼
┌───────────────────┐
│ Decision Scoring  │  LSTM + Attention → final_score (B,1)
│ (LSTM+Attention)  │                   → per_timepoint_eval (B,120)
└───────────────────┘
```

---

## Key Design: Context-Aware Gating (Scheme 2)

The original architecture used **static uniform weights** (1/8 per sub-model), meaning the model couldn't adapt "which sub-model to trust" based on context. This version replaces that with a **learnable gating network**:

```python
# model.py — RiskScoreMatrix
class RiskScoreMatrix(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_sub_models),
        )
        self.temperature = nn.Parameter(torch.tensor(2.0))  # learnable sharpness

    def forward(self, sub_outputs, compressed_states):
        logits = self.gate(compressed_states) / self.temperature  # (B,T,M)
        weights = F.softmax(logits, dim=-1)                        # (B,T,M)
        risk = (sub_outputs * weights).mean(dim=-1)               # (B,T)
        return risk, weights  # weights returned for interpretability
```

**What this enables:**

| Capability | Static Weights | Gating Network |
|-----------|----------------|----------------|
| Same weight for all timesteps | ✓ | ✗ (per-timestep) |
| Adapts to current behavior | ✗ | ✓ |
| Learns via backprop | ✗ | ✓ |
| Interpretable per-step weights |  | ✓ (`result['weights']`) |

**Example:** Sub-model 1 might dominate at Day 30 (detecting short-term anomalies), while Sub-model 5 dominates at Day 100 (detecting long-term drift). The gating network learns this automatically.

---

## Self-Supervised Learning

No manual labels needed. Pseudo-labels are derived from **behavior change intensity**:

```
Δx_t = x_{t+1} - x_t                      # adjacent timestep difference
||Δx_t||_2                                  # L2 norm = change strength
z_t = (||Δx_t|| - min) / (max - min)       # normalize to [0,1]

final_label = max(z) > 0.5                 # global: any spike = risk
tp_labels   = z[:120]                       # per-timestep risk
```

High behavioral change → high risk probability. This heuristic works well for fraud/credit risk where anomalies manifest as sudden behavior shifts.

---

## Loss Function

```
L = 0.6 × BCE(final_score, final_label)
  + 0.4 × BCE(per_timepoint_eval, tp_labels)
```

Two objectives trained jointly:
- **Global**: is there any risk in this sequence?
- **Local**: which timesteps are risky?

---

## Modules

| File | Description |
|------|-------------|
| `config.py` | `Config` dataclass + synthetic data / pseudo-label generation |
| `model.py` | All network components (compression, sub-models, gating, scoring) |
| `train.py` | Self-supervised trainer with BCE loss + early stopping |
| `predict.py` | Inference wrapper (single + batch) |
| `draw_presentation.py` | Presentation diagram generator |

---

## Project Structure

```
self_learning_modeling/
├── __init__.py           # package entry + macOS OpenMP fix
── config.py             # Config + data utilities
├── model.py              # Core networks
├── train.py              # Training pipeline
├── predict.py            # Inference wrapper
├── draw_presentation.py  # Diagram generator
└── README.md             # this file
```

---

## Related Packages

| Package | Approach | Key Difference |
|---------|----------|----------------|
| **`self_learning_modeling/`** | **Scheme 2: Context-aware Gating** | Learnable gate network, per-timestep dynamic weights |
| `self_learning_modeling_v3/` | Scheme 3: Iterative Two-Stage | No arch change; train → evaluate importance → update weights → repeat |
| `self_learning_modeling_v4/` | Scheme 4: Uncertainty-Weighted | Learnable log-variance per sub-model; stable models get higher weight |
| `rl_state_modeling/` | RL: PPO Actor-Critic | Feedback weights → RL actions; reward-driven closed loop |

---

## Requirements

```
torch >= 2.0
numpy >= 1.24
```

---

## Output Fields

```python
result = pred.predict(behavior_sequence)

result['final_score']           # float, 0~1, global risk assessment
result['per_timepoint_eval']    # (120,), per-timestep risk evaluation
result['risk_scores']           # (126,), weighted risk per timestep
result['sub_model_outputs']     # (126, 8), each sub-model's raw output
result['weights']               # (126, 8), gating network's learned weights
result['compressed_states']     # (126, 128), BiLSTM hidden states
```
