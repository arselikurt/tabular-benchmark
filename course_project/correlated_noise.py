"""Add nuisance columns with a shared per-row Gaussian factor."""
import numpy as np


def add_correlated_noise(x_train, x_val, x_test, y_train, y_val, y_test, *, noise_pct, rng):
    count = noise_pct * x_train.shape[1] // 100
    if count == 0:
        return x_train, x_val, x_test, y_train, y_val, y_test

    rho = 0.8
    source_columns = rng.choice(x_train.shape[1], count)
    means = x_train.mean(axis=0)[source_columns]
    q25, q75 = np.quantile(x_train, [0.25, 0.75], axis=0)
    scales = (q75 - q25)[source_columns] / 1.349
    scales = np.where(scales > 0, scales, 1.0)
    augmented = []
    for x in (x_train, x_val, x_test):
        shared = rng.standard_normal((len(x), 1))
        individual = rng.standard_normal((len(x), count))
        z = np.sqrt(rho) * shared + np.sqrt(1 - rho) * individual
        augmented.append(np.column_stack((x, means + scales * z)))
    return (*augmented, y_train, y_val, y_test)
