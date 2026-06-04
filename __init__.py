# Must be set before ANY torch import (macOS OpenMP libomp conflict)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
Dynamic Sequence Modeling — Self-Learning State Transition Risk

Three weighting schemes in one unified package:
  1. 'gating'       — context-aware per-timestep weights via gate network (scheme 1)
  2. 'iterative'    — multi-round training with importance-based weight updates (scheme 2)
  3. 'uncertainty'  — learnable log-variance per sub-model (scheme 3, Kendall et al.)

Quick start:
    from dynamic_seq_modeling.config import Config, generate_synthetic_data
    from dynamic_seq_modeling.train import SelfSupervisedTrainer
    from dynamic_seq_modeling.predict import Predictor

    cfg = Config()
    X, _, _ = generate_synthetic_data(100, cfg)

    trainer = SelfSupervisedTrainer(cfg, scheme='uncertainty', device='cpu')
    trainer.train(X, epochs=30)
    trainer.save('model.pt')

    pred = Predictor.load('model.pt')
    result = pred.predict(X[0])
"""
