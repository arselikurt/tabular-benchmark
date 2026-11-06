"""Run the course study with the original benchmark's split, fit and score code."""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('PROJECT_DIR', str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
from scipy.io import arff
import torch

from configs.all_model_configs import total_config
from data_transforms import add_uninformative_features, gaussienize
from generate_dataset_pipeline import data_to_train_test
from train import evaluate_model, train_model

from feature_gates import GatedClassifier, GatedMLP

DATASETS = {
    'MagicTelescope': (44125, 361065, 22103250, '5c42cddeaee233a285dc8cb6334db6a0', 'class'),
    'pol': (44122, 361062, 22103247, 'f4c79b9d56e5eaea043a35cab068d2f5', 'binaryClass'),
    'house_16H': (44123, 361063, 22103248, 'f93bc9053ba80a086ff842a903a7d14f', 'binaryClass'),
}


def load_dataset(name, data_dir):
    dataset_id, task_id, file_id, checksum, target = DATASETS[name]
    path = data_dir / f'{dataset_id}.arff'
    if not path.exists():
        url = f'https://api.openml.org/data/download/{file_id}/dataset'
        print(f'Downloading OpenML task {task_id} from {url}', flush=True)
        with urlopen(url, timeout=45) as response:
            content = response.read()
        if hashlib.md5(content).hexdigest() != checksum:
            raise ValueError(f'Download checksum mismatch for {name}.')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    elif hashlib.md5(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f'Checksum mismatch for {path}.')

    records, metadata = arff.loadarff(path)
    columns = [name for name in metadata.names() if name != target]
    x = np.column_stack([records[column] for column in columns]).astype(np.float32)
    target_values = records[target]
    if metadata[target][0] == 'nominal':
        labels = [label.encode() for label in metadata[target][1]]
        y = np.array([labels.index(value) for value in target_values], dtype=np.int64)
    else:
        y = target_values.astype(np.int64)
    if set(np.unique(y)) != {0, 1} or not np.isfinite(x).all():
        raise ValueError(f'{name} must contain finite numerical features and binary labels.')
    return x, y


def make_config(model, seed, epochs):
    upstream_model = 'mlp' if model == 'gated_mlp' else model
    defaults = total_config[upstream_model]['classif']['default']
    config = {key: entry['value'] if 'value' in entry else entry['values'][0]
              for key, entry in defaults.items()}
    config.update(regression=False, data__categorical=False)
    if model == 'rf':
        config.update(model__n_estimators=100, model__n_jobs=1, model__random_state=seed)
    else:
        config.update(model__device='cpu', model__max_epochs=epochs)
    if model == 'gated_mlp':
        config.update(model__module=GatedMLP, model__net_class=GatedClassifier,
                      model__gate_penalty=0.01)
    return config


def run_dataset(name, args, writer, output):
    x, y = load_dataset(name, args.data_dir)
    print(f'Loaded {name}: {x.shape[0]} rows, {x.shape[1]} original features', flush=True)
    split_config = {'train_prop': 0.7, 'val_test_prop': 0.3, 'max_train_samples': 10000,
                    'max_val_samples': 50000, 'max_test_samples': 50000}
    for seed in args.seeds:
        split_rng = np.random.RandomState(seed)
        # The original experiment runner consumes this draw before splitting.
        split_rng.randn(1)
        split = data_to_train_test(x, y, split_config, split_rng)
        test_hash = hashlib.sha256(split[2].tobytes() + split[5].tobytes()).hexdigest()[:16]
        for noise in args.noise:
            perturbed = add_uninformative_features(*split, multiplier=1 + noise / 100,
                                                  rng=np.random.RandomState(1000 + seed))
            for model in args.models:
                config = make_config(model, seed, args.epochs)
                data = perturbed
                if model != 'rf':
                    data = gaussienize(*data, type='quantile', rng=np.random.RandomState(seed))
                x_train, x_val, x_test = (array.astype(np.float32) for array in data[:3])
                y_train, y_val, y_test = data[3:]
                torch.manual_seed(seed)
                np.random.seed(seed)
                model_id = f'{name}-{seed}-{noise}-{model}'
                print(f'Training {model_id} ({x_train.shape[1]} features)', flush=True)
                start = perf_counter()
                fitted = train_model(seed, x_train, y_train, None, config, model_id)
                fit_seconds = perf_counter() - start
                train_acc, val_acc, test_acc = evaluate_model(
                    fitted, x_train, y_train, x_val, y_val, x_test, y_test, config, model_id)
                if not np.isfinite([train_acc, val_acc, test_acc]).all():
                    raise RuntimeError(f'Non-finite score for {model_id}.')
                gates = None if model != 'gated_mlp' else fitted.module_.gate_weights().detach().numpy()
                writer.writerow({
                    'dataset': name, 'task_id': DATASETS[name][1], 'model': model,
                    'seed': seed, 'noise_pct': noise, 'noise_features': x_train.shape[1] - x.shape[1],
                    'train_rows': len(y_train), 'validation_rows': len(y_val), 'test_rows': len(y_test),
                    'test_hash': test_hash, 'epochs': 0 if model == 'rf' else len(fitted.history),
                    'train_accuracy': train_acc, 'validation_accuracy': val_acc,
                    'test_accuracy': test_acc, 'fit_seconds': fit_seconds,
                    'mean_original_gate': '' if gates is None else gates[:x.shape[1]].mean(),
                    'mean_noise_gate': '' if gates is None or noise == 0 else gates[x.shape[1]:].mean(),
                })
                output.flush()
                print(f'Validation {val_acc:.4%} | Test {test_acc:.4%} | Fit {fit_seconds:.2f}s', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--models', nargs='+', choices=['rf', 'mlp', 'gated_mlp'],
                        default=['rf', 'mlp', 'gated_mlp'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--noise', nargs='+', type=int, choices=[0, 50, 100], default=[0, 50, 100])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'openml_cache' / 'course_project')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.epochs < 1 or any(seed < 0 or seed > 2**32 - 1001 for seed in args.seeds):
        parser.error('Use at least one epoch and nonnegative seeds below 2**32 - 1000.')
    args.data_dir = args.data_dir.resolve()
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    fields = ['dataset', 'task_id', 'model', 'seed', 'noise_pct', 'noise_features', 'train_rows',
              'validation_rows', 'test_rows', 'test_hash', 'epochs', 'train_accuracy',
              'validation_accuracy', 'test_accuracy', 'fit_seconds', 'mean_original_gate', 'mean_noise_gate']
    with TemporaryDirectory(prefix='tabular-checkpoints-') as directory:
        original_directory = Path.cwd()
        os.chdir(directory)
        try:
            with args.output.open('x', newline='') as output:
                writer = csv.DictWriter(output, fieldnames=fields)
                writer.writeheader()
                for name in args.datasets:
                    run_dataset(name, args, writer, output)
        finally:
            os.chdir(original_directory)


if __name__ == '__main__':
    main()
