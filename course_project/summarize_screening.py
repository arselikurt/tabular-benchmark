"""Check the screening experiment and compute predeclared paired comparisons."""

import argparse
import json
from pathlib import Path

import pandas as pd

DATASETS = ['MagicTelescope', 'pol', 'house_16H']
SEEDS = [3, 4, 5]
NOISE = [0, 50, 100]
MODELS = ['mlp', 'screened_mlp']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = pd.concat([pd.read_csv(path, float_precision='round_trip') for path in args.runs],
                     ignore_index=True)
    keys = ['dataset', 'seed', 'noise_pct', 'model']
    expected = pd.MultiIndex.from_product([DATASETS, SEEDS, NOISE, MODELS], names=keys)
    actual = pd.MultiIndex.from_frame(runs[keys])
    if not actual.is_unique or len(actual.symmetric_difference(expected)):
        raise ValueError('The 54-run grid is incomplete or contains duplicate conditions.')
    if not runs['test_accuracy'].between(0, 1).all():
        raise ValueError('Test accuracies must be finite proportions.')
    if not runs.groupby(['dataset', 'seed'])['test_hash'].nunique().eq(1).all():
        raise ValueError('Test rows differ within a dataset and seed.')
    if not runs['retained_noise_features'].between(0, runs['noise_features']).all():
        raise ValueError('Invalid retained-noise count.')

    paired = runs.pivot(index=keys[:-1], columns='model', values='test_accuracy')
    paired['gain_pp'] = 100 * (paired['screened_mlp'] - paired['mlp'])
    full = paired.xs(100, level='noise_pct')
    clean = paired.xs(0, level='noise_pct')
    selection = runs[runs['model'].eq('screened_mlp') & runs['noise_pct'].eq(100)]
    full_gain = float(full['gain_pp'].mean())
    clean_gain = float(clean['gain_pp'].mean())
    summary = {
        'fit_count': len(runs),
        'full_noise_gain_pp': full_gain,
        'clean_gain_pp': clean_gain,
        'full_noise_positive_pairs': int(full['gain_pp'].gt(0).sum()),
        'full_noise_removed_injected_columns': int((selection['noise_features'] - selection['retained_noise_features']).sum()),
        'full_noise_injected_columns': int(selection['noise_features'].sum()),
        'full_noise_removed_original_columns': int((selection['noise_features'] - selection['retained_original_features']).sum()),
        'h3_supported': full_gain >= 1.0 and clean_gain >= -1.0,
        'by_noise': paired.groupby('noise_pct').mean().to_dict(orient='index'),
        'by_dataset_full_noise': full.groupby('dataset').mean().to_dict(orient='index'),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    runs.sort_values(keys).to_csv(args.output / 'runs.csv', index=False)
    paired.to_csv(args.output / 'paired_accuracy.csv')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
