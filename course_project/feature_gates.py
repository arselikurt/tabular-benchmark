"""Trainable numerical input gates for the benchmark's original MLP."""
import torch
from skorch import NeuralNetClassifier

from models.tabular.bin.mlp import MLP


class GatedMLP(MLP):
    def __init__(self, *, d_in, categorical_indicator, **kwargs):
        if categorical_indicator is not None:
            raise ValueError('This extension supports numerical features only.')
        super().__init__(d_in=d_in, categorical_indicator=categorical_indicator, **kwargs)
        self.gate_logits = torch.nn.Parameter(torch.zeros(d_in))

    def gate_weights(self):
        # A zero logit gives a gate of one, so the original MLP starts unscaled.
        return 2 * self.gate_logits.sigmoid()

    def forward(self, x):
        return super().forward(x * self.gate_weights())


class GatedClassifier(NeuralNetClassifier):
    def __init__(self, module, *, gate_penalty=0.01, **kwargs):
        self.gate_penalty = gate_penalty
        super().__init__(module, **kwargs)

    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        if training:
            loss = loss + self.gate_penalty * self.module_.gate_weights().mean()
        return loss
