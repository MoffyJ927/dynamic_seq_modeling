# Must be set before ANY torch import (macOS OpenMP libomp conflict)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

"""
Self-Supervised State Transition Risk Model

A self-supervised architecture for long time-window state transition risk assessment.

Modules:
  config  – hyperparameters + data utilities
  model   – core networks (compression, sub-models, risk matrix, decision scoring)
  train   – self-supervised training pipeline (pseudo-labels + BCE loss)
  predict – inference wrapper

Quick start:
    from self_learning_modeling.config import Config, generate_synthetic_data
    from self_learning_modeling.train import SelfSupervisedTrainer
    from self_learning_modeling.predict import Predictor

    cfg = Config()
    X, y, tp = generate_synthetic_data(100, cfg)

    trainer = SelfSupervisedTrainer(cfg, device='cpu')
    trainer.train(X, y, tp, epochs=50)
    trainer.save('model.pt')

    pred = Predictor.load('model.pt')
    result = pred.predict(X[0])
"""
