# Dynamic Sequence Modeling — Self-Learning State Transition Risk

三套子模型加权方案统一包，合并自 `self_learning_modeling` / `v2` / `v3`。

## 目录结构

```
dynamic_seq_modeling/
├── __init__.py              # 包入口
├── config.py                # 超参数 + 数据工具（三方案共用）
├── train.py                 # SelfSupervisedTrainer + IterativeWeightTrainer
├── predict.py               # 统一推理器
├── README.md
└── model/
    ├── __init__.py
    ├── shared.py            # 共用组件: StateCompression, SubModel, MultiSubModels, DecisionScoring
    ├── scheme_gating.py      # 方案1: Gate网络动态加权
    ├── scheme_iterative.py   # 方案2: 轮次更新静态反馈权重
    ├── scheme_uncertainty.py # 方案3: 可学习log-variance加权 (Kendall et al.)
    └── builder.py           # build_model(scheme, config) 工厂函数
```

## 三套方案对比

| 方案 | RiskScoreMatrix | Trainer | 特点 |
|---|---|---|---|
| `gating` | Gate网络 + 温度参数，逐时间步softmax | SelfSupervisedTrainer | 上下文感知的动态权重 |
| `iterative` | 无参数，外部传入feedback | IterativeWeightTrainer | 多轮训练，基于重要性更新权重 |
| `uncertainty` | learnable log_vars，权重∝exp(-log_var) | SelfSupervisedTrainer | 端到端学习，低不确定性子模型得高权重 |

### 共用组件

`StateCompressionModel`、`SubModel`、`MultiSubModels`、`DecisionScoringModel` 三套方案完全相同，定义在 `model/shared.py`。`config.py`、`predict.py` 三套方案共用。

## 快速开始

```python
from dynamic_seq_modeling.config import Config, generate_synthetic_data
from dynamic_seq_modeling.train import SelfSupervisedTrainer, IterativeWeightTrainer
from dynamic_seq_modeling.predict import Predictor

cfg = Config()
X, _, _ = generate_synthetic_data(100, cfg)

# ---- Gating / Uncertainty ----
trainer = SelfSupervisedTrainer(cfg, scheme='uncertainty', device='cpu')
trainer.train(X, epochs=30)
trainer.save('model.pt')

pred = Predictor.load('model.pt')
result = pred.predict(X[0])

# ---- Iterative ----
trainer = IterativeWeightTrainer(cfg, device='cpu')
trainer.train(X, n_rounds=3, epochs_per_round=20)
trainer.save('model.pt')

pred = Predictor.load('model.pt')
result = pred.predict(X[0])  # 自动使用uniform feedback
```
