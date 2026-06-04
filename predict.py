"""
Unified inference for all three weighting schemes.

Usage:
    from dynamic_seq_modeling.predict import Predictor

    pred = Predictor.load('checkpoint.pt', device='cpu')
    result = pred.predict(sequence)          # single sample
    results = pred.predict_batch(sequences)  # batch
"""

import torch
import numpy as np
from typing import Dict, List, Union

from .config import Config
from .model.builder import build_model


class Predictor:
    """Wraps a trained model for inference-only use.

    Handles all three schemes transparently:
      - gating / uncertainty: forward(behavior_sequence)
      - iterative:            forward(behavior_sequence, feedback)
    For iterative, uniform weights are used as default feedback.
    """

    def __init__(self, model, config: Config, scheme: str, device: str = 'cpu'):
        self.model = model.to(device).eval()
        self.config = config
        self.scheme = scheme
        self.device = torch.device(device)
        self._default_feedback = np.ones(config.num_sub_models, dtype=np.float32) / config.num_sub_models

    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'Predictor':
        ckpt = torch.load(path, map_location=device, weights_only=False)
        config = ckpt.get('config')
        scheme = ckpt.get('scheme', 'gating')
        model = build_model(scheme, config)
        model.load_state_dict(ckpt['model'])
        return cls(model, config, scheme, device)

    def predict(
        self,
        behavior_sequence: np.ndarray,
        feedback: np.ndarray = None,
    ) -> Dict:
        with torch.no_grad():
            seq_t = torch.FloatTensor(behavior_sequence).unsqueeze(0).to(self.device)

            if self.scheme == 'iterative':
                fb = feedback if feedback is not None else self._default_feedback
                fb_t = torch.FloatTensor(fb).unsqueeze(0).to(self.device)
                outputs = self.model(seq_t, fb_t)
            else:
                outputs = self.model(seq_t)

        result = {
            'final_score': outputs['final_score'].squeeze().cpu().numpy(),
            'per_timepoint_eval': outputs['per_timepoint_eval'].squeeze().cpu().numpy(),
            'risk_scores': outputs['risk_scores'].squeeze().cpu().numpy(),
            'sub_model_outputs': outputs['sub_model_outputs'].squeeze().cpu().numpy(),
        }
        if 'weights' in outputs:
            result['weights'] = outputs['weights'].squeeze().cpu().numpy()
        return result

    def predict_batch(
        self,
        behavior_sequences: np.ndarray,
        feedback: np.ndarray = None,
    ) -> List[Dict]:
        with torch.no_grad():
            seq_t = torch.FloatTensor(behavior_sequences).to(self.device)

            if self.scheme == 'iterative':
                N = behavior_sequences.shape[0]
                fb = feedback if feedback is not None else np.tile(self._default_feedback, (N, 1))
                fb_t = torch.FloatTensor(fb).to(self.device)
                outputs = self.model(seq_t, fb_t)
            else:
                outputs = self.model(seq_t)

        results = []
        for i in range(behavior_sequences.shape[0]):
            r = {
                'final_score': outputs['final_score'][i].squeeze().cpu().numpy(),
                'per_timepoint_eval': outputs['per_timepoint_eval'][i].squeeze().cpu().numpy(),
                'risk_scores': outputs['risk_scores'][i].cpu().numpy(),
                'sub_model_outputs': outputs['sub_model_outputs'][i].cpu().numpy(),
            }
            if 'weights' in outputs:
                r['weights'] = outputs['weights'][i].cpu().numpy()
            results.append(r)
        return results


# ── Demo ─────────────────────────────────────────────────────

def main():
    from .config import Config, generate_synthetic_data
    from .train import SelfSupervisedTrainer, IterativeWeightTrainer
    import tempfile, os

    config = Config()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    seq, _, _ = generate_synthetic_data(50, config)

    for scheme, trainer_cls in [('gating', SelfSupervisedTrainer), ('uncertainty', SelfSupervisedTrainer)]:
        print(f"\n--- Testing {scheme} ---")
        trainer = trainer_cls(config, scheme=scheme, device=device)
        trainer.train(seq, epochs=10, batch_size=16)

        ckpt = os.path.join(tempfile.gettempdir(), f'dsm_{scheme}.pt')
        trainer.save(ckpt)
        pred = Predictor.load(ckpt, device)
        r = pred.predict(seq[0])
        print(f"  Final score: {r['final_score']:.4f}, Risk mean: {r['risk_scores'].mean():.4f}")
        if 'weights' in r:
            print(f"  Weights: {np.array(r['weights']).round(3)}")
        os.remove(ckpt)

    print(f"\n--- Testing iterative ---")
    trainer = IterativeWeightTrainer(config, device)
    trainer.train(seq, n_rounds=2, epochs_per_round=10, batch_size=16)
    ckpt = os.path.join(tempfile.gettempdir(), 'dsm_iterative.pt')
    trainer.save(ckpt)
    pred = Predictor.load(ckpt, device)
    r = pred.predict(seq[0])
    print(f"  Final score: {r['final_score']:.4f}, Risk mean: {r['risk_scores'].mean():.4f}")
    os.remove(ckpt)

    print("\nAll schemes tested successfully.")


if __name__ == '__main__':
    main()
