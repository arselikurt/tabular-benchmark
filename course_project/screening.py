"""Fit a training-only permutation screen for numerical, binary classification."""

import numpy as np
from skorch.dataset import CVSplit


def _mutual_information(codes, labels, bins):
    counts = np.bincount((codes + labels).ravel(), minlength=len(codes) * bins * 2)
    joint = counts.reshape(len(codes), bins, 2) / labels.size
    independent = joint.sum(axis=2, keepdims=True) * joint.sum(axis=1, keepdims=True)
    ratio = np.ones_like(joint)
    np.divide(joint, independent, out=ratio, where=joint > 0)
    return (joint * np.log(ratio)).sum(axis=(1, 2))


def select_inputs(x_train, y_train, seed, *, bins=16, permutations=64, alpha=0.05):
    """Return the mask fitted only on rows used for MLP weight updates."""
    optimization, _ = CVSplit(5, stratified=True)(x_train, y_train)
    features = x_train[optimization.indices]
    labels = y_train[optimization.indices].astype(np.int64)
    cuts = np.quantile(features, np.linspace(0, 1, bins + 1), axis=0)
    codes = np.stack([
        np.searchsorted(np.unique(edges)[1:-1], values, side='right')
        for edges, values in zip(cuts.T, features.T)
    ])
    codes = 2 * (codes + bins * np.arange(len(codes))[:, None])
    observed = _mutual_information(codes, labels, bins)
    rng = np.random.default_rng([seed, 2000])
    null_max = np.array([
        _mutual_information(codes, rng.permutation(labels), bins).max()
        for _ in range(permutations)
    ])
    p_values = (1 + (null_max[:, None] >= observed).sum(axis=0)) / (permutations + 1)
    mask = p_values <= alpha
    return mask if mask.any() else np.ones(features.shape[1], dtype=bool)
