"""Validate paired runs and extract tables from the benchmark results."""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

DATASETS = ['MagicTelescope', 'pol', 'house_16H']
MODELS = {'rf': 'RandomForest', 'mlp': 'MLP', 'gated_mlp': 'Gated MLP'}
NOISE = [0, 50, 100]


def extract_reference(source):
    with source.open(newline='') as handle:
        rows = [
            {'dataset': row['data__keyword'], 'model': row['model_name'],
             'published_test_accuracy': float(row['mean_test_score']),
             'published_split_accuracies': row['test_scores']}
            for row in csv.DictReader(handle)
            if row['benchmark'] == 'numerical_classification_medium' and row['hp'] == 'default'
            and row['data__keyword'] in DATASETS and row['model_name'] in {'RandomForest', 'MLP'}
        ]
    reference = pd.DataFrame(rows)
    if len(reference) != 6 or reference.duplicated(['dataset', 'model']).any():
        raise ValueError('Expected six unique published default scores.')
    return reference.sort_values(['dataset', 'model'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--published', type=Path, required=True,
                        help='The original 40081681 Figshare CSV or the extracted six-row reference.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = pd.concat([pd.read_csv(path) for path in args.runs], ignore_index=True)
    keys = ['dataset', 'seed', 'noise_pct', 'model']
    expected = pd.MultiIndex.from_product([DATASETS, [0, 1, 2], NOISE, MODELS], names=keys)
    actual = pd.MultiIndex.from_frame(runs[keys])
    if not actual.is_unique or len(actual.symmetric_difference(expected)):
        raise ValueError('Expected exactly one run per dataset, seed, noise level and model.')
    if not runs['test_accuracy'].between(0, 1).all():
        raise ValueError('Test accuracies must be finite proportions.')
    if not runs.groupby(['dataset', 'seed'])['test_hash'].nunique().eq(1).all():
        raise ValueError('A test split differs between paired runs.')

    args.output.mkdir(parents=True, exist_ok=True)
    runs.sort_values(keys).to_csv(args.output / 'runs.csv', index=False)
    if args.published.stat().st_size > 1_000_000:
        reference = extract_reference(args.published)
    else:
        reference = pd.read_csv(args.published)
    reference.to_csv(args.output / 'published_defaults.csv', index=False)
    clean = (runs[runs['noise_pct'].eq(0) & runs['model'].isin(['rf', 'mlp'])]
             .assign(model=lambda df: df['model'].map(MODELS))
             .groupby(['dataset', 'model'])['test_accuracy']
             .agg(reproduced_test_accuracy='mean', split_sd='std').reset_index())
    comparison = reference.merge(clean, on=['dataset', 'model'], validate='one_to_one')
    if len(comparison) != 6:
        raise ValueError('Expected six published versus reproduced comparisons.')
    comparison['delta_pp'] = 100 * (comparison['reproduced_test_accuracy'] - comparison['published_test_accuracy'])
    comparison.to_csv(args.output / 'reproduction.csv', index=False)

    curves = runs.groupby(['model', 'noise_pct'])['test_accuracy'].agg(['mean', 'std']).reset_index()
    curves.to_csv(args.output / 'noise_curve.csv', index=False)
    paired = runs.pivot(index=['dataset', 'seed', 'model'], columns='noise_pct', values='test_accuracy')
    drops = ((paired[0] - paired[100]) * 100).unstack('model')
    drops = drops.rename(columns=lambda name: f'{name}_drop_pp')
    drops['h1_margin_pp'] = drops['mlp_drop_pp'] - drops['rf_drop_pp']
    drops['h2_margin_pp'] = drops['mlp_drop_pp'] - drops['gated_mlp_drop_pp']
    drops.to_csv(args.output / 'paired_drops.csv')
    per_dataset = drops.groupby('dataset').mean()
    mean_clean = (runs[runs['noise_pct'].eq(0)].groupby('model')['test_accuracy'].mean() * 100).to_dict()
    gates = runs[runs['model'].eq('gated_mlp') & runs['noise_pct'].eq(100)]
    summary = {
        'fit_count': len(runs), 'clean_accuracy_pct': mean_clean,
        'mean_drop_pp': drops.mean().to_dict(),
        'h1_positive_pairs': int(drops['h1_margin_pp'].gt(0).sum()),
        'h2_positive_pairs': int(drops['h2_margin_pp'].gt(0).sum()),
        'by_dataset': per_dataset.to_dict(orient='index'),
        'gate_means_full_noise': gates[['mean_original_gate', 'mean_noise_gate']].mean().to_dict(),
        'max_abs_reproduction_delta_pp': float(comparison['delta_pp'].abs().max()),
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
