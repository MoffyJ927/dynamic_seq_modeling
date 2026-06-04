"""
Inference / prediction for the Self-Supervised State Transition Risk Model (Scheme 2).

Usage:
    from self_learning_modeling.predict import Predictor
    pred = Predictor.load('checkpoint.pt', device='cpu')
    result = pred.predict(behavior_sequence)
    results = pred.predict_batch(behavior_sequences)
"""

import torch
import numpy as np
from typing import Dict, List

from .config import Config
from .model import SelfSupervisedStateTransitionRiskModel


class Predictor:
    """Wraps a trained model for inference-only use."""

    def __init__(self, model: SelfSupervisedStateTransitionRiskModel,
                 config: Config, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.config = config
        self.device = torch.device(device)

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'Predictor':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config: Config = ckpt['config']
        model = SelfSupervisedStateTransitionRiskModel(config)
        model.load_state_dict(ckpt['model'])
        return cls(model, config, device)

    def predict(self, behavior_sequence: np.ndarray) -> Dict:
        """
        Run inference on a single behavioral sequence.

        Returns dict with:
          final_score, per_timepoint_eval, risk_scores, sub_model_outputs, weights
        """
        with torch.no_grad():
            seq_t = torch.FloatTensor(behavior_sequence).unsqueeze(0).to(self.device)
            outputs = self.model(seq_t)

        return {
            'final_score': outputs['final_score'].squeeze().cpu().numpy(),
            'per_timepoint_eval': outputs['per_timepoint_eval'].squeeze().cpu().numpy(),
            'risk_scores': outputs['risk_scores'].squeeze().cpu().numpy(),
            'sub_model_outputs': outputs['sub_model_outputs'].squeeze().cpu().numpy(),
            'weights': outputs['weights'].squeeze().cpu().numpy(),
        }

    def predict_batch(self, behavior_sequences: np.ndarray) -> List[Dict]:
        with torch.no_grad():
            seq_t = torch.FloatTensor(behavior_sequences).to(self.device)
            outputs = self.model(seq_t)

        results = []
        for i in range(behavior_sequences.shape[0]):
            results.append({
                'final_score': outputs['final_score'][i].squeeze().cpu().numpy(),
                'per_timepoint_eval': outputs['per_timepoint_eval'][i].squeeze().cpu().numpy(),
                'risk_scores': outputs['risk_scores'][i].cpu().numpy(),
                'sub_model_outputs': outputs['sub_model_outputs'][i].cpu().numpy(),
                'weights': outputs['weights'][i].cpu().numpy(),
            })
        return results


# ==========================================
# Quick demo
# ==========================================

def main():
    from . import config as _pkg_init
    from .config import Config, generate_synthetic_data
    from .train import SelfSupervisedTrainer

    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    seq, labels, tp = generate_synthetic_data(50, config)

    trainer = SelfSupervisedTrainer(config, device)
    trainer.train(seq, labels, tp, epochs=20, batch_size=16)

    import tempfile, os
    ckpt_path = os.path.join(tempfile.gettempdir(), 'self_learn_v2.pt')
    trainer.save(ckpt_path)

    predictor = Predictor.load(ckpt_path, device)
    result = predictor.predict(seq[0])

    print(f"\nInference on sample 0:")
    print(f"  Final score     : {result['final_score']:.4f}")
    print(f"  Risk scores mean: {result['risk_scores'].mean():.4f}")
    print(f"  Weights shape   : {result['weights'].shape}")  # (T, M) — per-timestep!

    os.remove(ckpt_path)
    print("\nDemo complete.")


if __name__ == '__main__':
    main()
