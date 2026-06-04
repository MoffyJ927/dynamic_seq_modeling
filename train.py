"""
Unified training pipelines for all three weighting schemes.

  SelfSupervisedTrainer  — for 'gating' and 'uncertainty' schemes
  IterativeWeightTrainer — for 'iterative' scheme (multi-round)

All trainers use self-supervised pseudo-labels from behavior change intensity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple

from .config import Config, generate_self_supervised_labels
from .model.builder import build_model


# ═══════════════════════════════════════════════════════════════
# Shared utilities
# ═══════════════════════════════════════════════════════════════

def _compute_metrics(outputs, final_labels, tp_labels):
    final_acc = ((outputs['final_score'].squeeze(-1) > 0.5).float() == final_labels).float().mean()
    tp_acc = ((outputs['per_timepoint_eval'] > 0.5).float() == tp_labels).float().mean()
    return {'final_acc': final_acc.item(), 'tp_acc': tp_acc.item()}


def _train_epoch(model, loader, opt, device, scheme, uniform_fb=None):
    """Generic single-epoch training loop. Returns list of per-batch loss dicts."""
    model.train()
    losses_list = []
    for batch in loader:
        b_seq = batch[0].to(device)
        b_final = batch[-2].to(device) if len(batch) > 2 else batch[1].to(device)
        b_tp = batch[-1].to(device)

        opt.zero_grad()

        if scheme == 'iterative':
            b_fb = batch[1].to(device)
            outputs = model(b_seq, b_fb)
        else:
            outputs = model(b_seq)

        final_loss = F.binary_cross_entropy(outputs['final_score'].squeeze(-1), b_final)
        tp_loss = F.binary_cross_entropy(outputs['per_timepoint_eval'], b_tp)
        total_loss = 0.6 * final_loss + 0.4 * tp_loss
        total_loss.backward()
        opt.step()

        metrics = _compute_metrics(outputs, b_final, b_tp)
        losses_list.append({
            'total_loss': total_loss.item(),
            'final_loss': final_loss.item(),
            'tp_loss': tp_loss.item(),
            **metrics,
        })
    return losses_list


def _validate(model, val_seq, val_final, val_tp, device, scheme, val_fb=None):
    model.eval()
    with torch.no_grad():
        vs = torch.FloatTensor(val_seq).to(device)
        vfl = torch.FloatTensor(val_final).to(device)
        vtp = torch.FloatTensor(val_tp).to(device)
        if scheme == 'iterative':
            vf = torch.FloatTensor(val_fb).to(device)
            out = model(vs, vf)
        else:
            out = model(vs)
        val_loss = (
            0.6 * F.binary_cross_entropy(out['final_score'].squeeze(-1), vfl)
            + 0.4 * F.binary_cross_entropy(out['per_timepoint_eval'], vtp)
        ).item()
    return val_loss


# ═══════════════════════════════════════════════════════════════
# Trainer: Gating & Uncertainty (shared forward API)
# ═══════════════════════════════════════════════════════════════

class SelfSupervisedTrainer:
    """Standard self-supervised trainer for gating and uncertainty schemes.

    Both schemes use the same forward(model, behavior_sequence) API.
    """

    def __init__(self, config: Config, scheme: str = 'gating', device: str = 'cpu'):
        self.config = config
        self.scheme = scheme
        self.device = torch.device(device)
        self.model = build_model(scheme, config).to(self.device)

    def train(
        self,
        behavior_sequences: np.ndarray,
        true_labels: np.ndarray = None,
        timepoint_labels: np.ndarray = None,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> List[Dict]:
        N = behavior_sequences.shape[0]
        val_size = int(N * validation_split)
        tr_size = N - val_size

        final_labels, tp_labels = generate_self_supervised_labels(behavior_sequences, self.config)

        tr_seq, tr_final, tr_tp = behavior_sequences[:tr_size], final_labels[:tr_size], tp_labels[:tr_size]
        val_seq, val_final, val_tp = behavior_sequences[tr_size:], final_labels[tr_size:], tp_labels[tr_size:]

        ds = TensorDataset(
            torch.FloatTensor(tr_seq), torch.FloatTensor(tr_final), torch.FloatTensor(tr_tp))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.config.learning_rate, weight_decay=self.config.l2_reg)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

        history = []
        best_val_loss = float('inf')
        best_state = None
        patience = 0

        for epoch in range(epochs):
            epoch_losses = _train_epoch(self.model, loader, opt, self.device, self.scheme)
            avg = {k: np.mean([l[k] for l in epoch_losses]) for k in epoch_losses[0]}
            val_loss = _validate(self.model, val_seq, val_final, val_tp, self.device, self.scheme)

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
                    print(f"Early stopping at epoch {epoch+1}")
                    self.model.load_state_dict(best_state)
                    break

            if (epoch + 1) % 5 == 0:
                self._log_epoch(epoch + 1, epochs, avg, val_loss)

        return history

    def _log_epoch(self, epoch, total, avg, val_loss):
        parts = [f"Epoch {epoch}/{total}  |  Loss: {avg['total_loss']:.4f},  Val: {val_loss:.4f}"]
        if self.scheme == 'uncertainty':
            w = self.model.risk_score_matrix.log_vars.exp().detach()
            w = (w / w.sum()).cpu().numpy()
            parts.append(f"Weights: {w.round(3)}")
        elif self.scheme == 'gating':
            parts.append(f"Final Acc: {avg['final_acc']:.4f}")
        print("  |  ".join(parts))

    def save(self, path: str):
        torch.save({'model': self.model.state_dict(), 'config': self.config, 'scheme': self.scheme}, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        print(f"Loaded from {path}")


# ═══════════════════════════════════════════════════════════════
# Trainer: Iterative Two-Stage
# ═══════════════════════════════════════════════════════════════

class IterativeWeightTrainer:
    """Multi-round trainer that updates feedback weights between rounds.

    Round 0: uniform weights → train
    Evaluate: compute per-sub-model importance (output variance)
    Round 1+: blend old weights with importance → retrain
    """

    def __init__(self, config: Config, device: str = 'cpu'):
        self.config = config
        self.device = torch.device(device)
        self.scheme = 'iterative'
        self.model = build_model('iterative', config).to(self.device)

    @staticmethod
    def _uniform_feedback(n: int, m: int) -> np.ndarray:
        return np.ones((n, m), dtype=np.float32) / m

    def _compute_importance(
        self,
        behavior_sequences: np.ndarray,
        final_labels: np.ndarray,
        tp_labels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-sub-model importance via output variance."""
        self.model.eval()
        M = self.config.num_sub_models
        importance = np.zeros(M)

        with torch.no_grad():
            seq_t = torch.FloatTensor(behavior_sequences).to(self.device)
            fb = torch.FloatTensor(self._uniform_feedback(1, M)).to(self.device)
            outputs = self.model(seq_t, fb)
            sub_out = outputs['sub_model_outputs']  # (N, T, M)
            for m in range(M):
                importance[m] = float(sub_out[:, :, m].var())

        importance = importance / (importance.sum() + 1e-8)
        return importance

    def train(
        self,
        behavior_sequences: np.ndarray,
        true_labels: np.ndarray = None,
        timepoint_labels: np.ndarray = None,
        n_rounds: int = 3,
        epochs_per_round: int = 20,
        batch_size: int = 32,
        validation_split: float = 0.2,
    ) -> List[Dict]:
        N = behavior_sequences.shape[0]
        val_size = int(N * validation_split)
        tr_size = N - val_size

        final_labels, tp_labels = generate_self_supervised_labels(behavior_sequences, self.config)

        val_seq, val_final, val_tp = behavior_sequences[tr_size:], final_labels[tr_size:], tp_labels[tr_size:]

        # Round 0: uniform
        feedback = self._uniform_feedback(N, self.config.num_sub_models)
        history = []

        for rnd in range(n_rounds):
            print(f"\n{'='*50}")
            print(f"Round {rnd}/{n_rounds-1}  |  feedback[0]: {feedback[0].round(3)}")

            # Train one round
            tr_seq = behavior_sequences[:tr_size]
            tr_final = final_labels[:tr_size]
            tr_tp = tp_labels[:tr_size]
            tr_fb = feedback[:tr_size]
            val_fb = feedback[tr_size:]

            ds = TensorDataset(
                torch.FloatTensor(tr_seq), torch.FloatTensor(tr_fb),
                torch.FloatTensor(tr_final), torch.FloatTensor(tr_tp))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

            opt = torch.optim.Adam(self.model.parameters(),
                                   lr=self.config.learning_rate, weight_decay=self.config.l2_reg)

            for epoch in range(epochs_per_round):
                epoch_losses = _train_epoch(self.model, loader, opt, self.device, self.scheme)

            val_loss = _validate(self.model, val_seq, val_final, val_tp,
                                 self.device, self.scheme, val_fb)
            result = {'round': rnd, 'val_loss': val_loss}
            history.append(result)

            # Evaluate importance
            importance = self._compute_importance(behavior_sequences, final_labels, tp_labels)
            print(f"  Sub-model importance: {importance.round(3)}")

            # Blend: soft update to avoid oscillation
            alpha = 0.5 if rnd > 0 else 1.0
            feedback = (1 - alpha) * feedback + alpha * importance[np.newaxis, :]
            feedback = feedback / feedback.sum(axis=1, keepdims=True)

        print(f"\nFinal feedback weights: {feedback[0].round(4)}")
        return history

    def save(self, path: str):
        torch.save({'model': self.model.state_dict(), 'config': self.config, 'scheme': 'iterative'}, path)
        print(f"Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model'])
        print(f"Loaded from {path}")
