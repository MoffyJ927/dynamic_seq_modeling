"""
Self-supervised training pipeline for the State Transition Risk Model (Scheme 2: Gating).

The key idea: generate pseudo-labels from behavior change intensity,
then train with standard BCE loss. The gating network learns dynamic
sub-model weights — no external feedback needed.

Usage:
    from self_learning_modeling.train import SelfSupervisedTrainer
    trainer = SelfSupervisedTrainer(config, device='cpu')
    history = trainer.train(X, y, tp, epochs=50)
    trainer.save('checkpoint.pt')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple

from .config import Config
from .model import SelfSupervisedStateTransitionRiskModel


class SelfSupervisedTrainer:
    """
    Self-supervised trainer with context-aware gating.

    Pipeline:
      1. Generate pseudo-labels from behavior change intensity
      2. Train with BCE loss: final_score + per_timepoint_eval
         The gating network (inside RiskScoreMatrix) learns dynamic weights
      3. Validate, early-stop, save best
    """

    def __init__(self, config: Config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.model = SelfSupervisedStateTransitionRiskModel(config).to(self.device)

    # ── Training Step ───────────────────────────────────────
    def _train_step(
        self,
        seq: torch.Tensor,
        final_labels: torch.Tensor,
        tp_labels: torch.Tensor,
    ) -> Dict[str, float]:
        self.model.train()
        outputs = self.model(seq)

        final_loss = F.binary_cross_entropy(
            outputs['final_score'].squeeze(-1), final_labels,
        )
        tp_loss = F.binary_cross_entropy(
            outputs['per_timepoint_eval'], tp_labels,
        )
        total_loss = 0.6 * final_loss + 0.4 * tp_loss
        total_loss.backward()

        final_acc = ((outputs['final_score'].squeeze(-1) > 0.5).float() == final_labels).float().mean()
        tp_acc = ((outputs['per_timepoint_eval'] > 0.5).float() == tp_labels).float().mean()

        return {
            'total_loss': total_loss.item(),
            'final_loss': final_loss.item(),
            'tp_loss': tp_loss.item(),
            'final_acc': final_acc.item(),
            'tp_acc': tp_acc.item(),
        }

    # ── Training Loop ───────────────────────────────────────
    def train(
        self,
        behavior_sequences: np.ndarray,
        true_labels: np.ndarray,
        timepoint_labels: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> List[Dict]:
        """
        Run the full self-supervised training loop.

        NOTE: `true_labels` and `timepoint_labels` are kept for API consistency
        but self-supervised pseudo-labels are used instead.
        """
        N = behavior_sequences.shape[0]
        val_size = int(N * validation_split)
        train_size = N - val_size

        # Self-supervised labels from behavior change intensity
        final_labels, tp_labels = self._pseudo_labels(behavior_sequences)

        # Split
        tr_seq = behavior_sequences[:train_size]
        tr_final = final_labels[:train_size]
        tr_tp = tp_labels[:train_size]

        val_seq = behavior_sequences[train_size:]
        val_final = final_labels[train_size:]
        val_tp = tp_labels[train_size:]

        # DataLoader (no feedback column anymore)
        ds = TensorDataset(
            torch.FloatTensor(tr_seq),
            torch.FloatTensor(tr_final),
            torch.FloatTensor(tr_tp),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        # Optimizer + scheduler
        opt = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.l2_reg,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

        history: List[Dict] = []
        best_val_loss = float('inf')
        best_state = None
        patience = 0

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            for b_seq, b_final, b_tp in loader:
                b_seq = b_seq.to(self.device)
                b_final = b_final.to(self.device)
                b_tp = b_tp.to(self.device)

                opt.zero_grad()
                losses = self._train_step(b_seq, b_final, b_tp)
                opt.step()
                epoch_losses.append(losses)

            avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}

            # Validate
            self.model.eval()
            with torch.no_grad():
                vs = torch.FloatTensor(val_seq).to(self.device)
                vfl = torch.FloatTensor(val_final).to(self.device)
                vtp = torch.FloatTensor(val_tp).to(self.device)
                out = self.model(vs)
                val_loss = (
                    0.6 * F.binary_cross_entropy(out['final_score'].squeeze(-1), vfl)
                    + 0.4 * F.binary_cross_entropy(out['per_timepoint_eval'], vtp)
                ).item()

            avg['val_loss'] = val_loss
            history.append(avg)
            sched.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    print(f"Early stopping at epoch {epoch + 1}")
                    self.model.load_state_dict(best_state)
                    break

            if (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs}  |  "
                    f"Loss: {avg['total_loss']:.4f},  "
                    f"Val Loss: {val_loss:.4f},  "
                    f"Final Acc: {avg['final_acc']:.4f}"
                )

        return history

    # ── Persistence ─────────────────────────────────────────
    def save(self, path: str):
        torch.save({
            'model': self.model.state_dict(),
            'config': self.config,
        }, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        print(f"Loaded from {path}")

    # ── Internal ────────────────────────────────────────────
    def _pseudo_labels(self, seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        from .config import generate_self_supervised_labels
        return generate_self_supervised_labels(seq, self.config)
