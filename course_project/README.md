# Tabular benchmark course workflow

This study evaluates the original Random Forest and MLP and two MLP extensions for numerical inputs. It reuses the original split, noise transform, training, and evaluation functions. The local runner loads the authors’ transformed OpenML data from checksummed ARFF files and runs without a Weights & Biases account.

## Reproduce the runs

Run from the repository root. The lockfile was verified on Apple silicon with Python 3.9.6 and macOS 26.5.1. Use one environment and one machine for both MLP variants and the forest when reporting controlled comparisons.

```sh
python3.9 -m venv .venv
.venv/bin/python -m pip install -r course_project/requirements-macos-arm64.txt
.venv/bin/python -m pip check
```

Run a short demo on the first dataset and split. The command downloads and caches the public ARFF on first use. Subsequent runs use the cache. Choose a fresh output path for each run.

```sh
.venv/bin/python -u course_project/run.py --datasets MagicTelescope \
  --models rf mlp --seeds 0 --noise 0 --epochs 3 --output output/demo.csv
```

Run the full grid and rebuild the comparison tables:

```sh
.venv/bin/python -u course_project/run.py --output output/full.csv
.venv/bin/python course_project/summarize.py --runs output/full.csv \
  --published course_project/results/published_defaults.csv --output output/summary
```

`run.py` prints dataset dimensions, model identifiers, and validation and test scores. Each CSV row includes the dataset and task IDs, seed, feature count, split sizes and fingerprint, fit time, and test accuracy. Gate runs also log average weights for original and added columns. To re-extract the authors’ references, pass the [published results CSV](https://api.figshare.com/v2/file/download/40081681?download=1) as `--published`.

## Design

| Dataset | OpenML task | Inputs |
| --- | ---: | ---: |
| MagicTelescope | 361065 | 10 |
| pol | 361062 | 26 |
| house_16H | 361063 | 16 |

We use split seeds 0, 1, and 2 and reproduce the upstream runner’s initial random draw. Each split uses 70% training rows capped at 10,000, then 30% of the remainder for validation and 70% for testing. The two MLPs share the upstream numerical quantile preprocessing and 300-epoch cap with validation-loss checkpoint selection. The forest uses 100 trees. Each fit used one CPU thread on the same Apple M5 Mac with 24 GB RAM. Dataset processes overlapped, so `fit_seconds` is diagnostic only.

We add 0%, 50%, or 100% as many Gaussian features as original columns. The upstream transform uses training-only feature means and interquartile ranges to calibrate those columns. Each model sees the same rows and noise for a given dataset, seed, and noise level. The gate starts at 1 for every input and multiplies that input before the original MLP. Training adds a fixed penalty of 0.01 times the mean gate value to cross-entropy. Validation uses the unpenalized loss to select checkpoints for both MLPs.

- **H1.** The original MLP loses more test accuracy than Random Forest from clean inputs to 100% added features.
- **H2.** The gated MLP loses less test accuracy than the original MLP over the same range.

`results/reproduction.csv` pairs the authors’ default scores with our mean over three splits. `results/paired_drops.csv` compares the clean-to-noisy changes on matching splits. The six published rows were selected by dataset, model, `numerical_classification_medium`, and `hp=default`. The paired changes are descriptive because splits overlap and the study covers three datasets.

## Screening follow-up

[The follow-up protocol](SCREENING_PROTOCOL.md) fixes the permutation screen and the success criteria. The selector learns its mask from the MLP's optimization rows using the maximum mutual information over 64 label shuffles. Both models train with the same settings and the same rows. The follow-up uses seeds 3–5 and the same three datasets.

```sh
.venv/bin/python -u course_project/run.py --models mlp screened_mlp \
  --seeds 3 4 5 --noise 0 50 100 --epochs 300 --output output/screening.csv
.venv/bin/python course_project/summarize_screening.py --runs output/screening.csv \
  --output output/screening-summary
```

The [54-run CSV](results/screening/runs.csv) records the test fingerprint and retained features. The [paired table](results/screening/paired_accuracy.csv) compares the original and screened MLP on identical test rows. The [summary](results/screening/summary.json) shows a 3.39-point mean gain at full noise and a 0.44-point gain on clean inputs. Screening removed 155 of 156 injected columns across the nine full-noise fits. These results cover the tested Gaussian noise and numerical tasks.

## Team checklist

- Review and run the demo on the shared fork. Each teammate should commit substantive work from their own GitHub account and update their completed contributions in the top-level README.
- Use the run log or live demo with the presentation on December 3. Upload the slide file to the [course Drive folder](https://drive.google.com/drive/folders/1AfGhSkamTVjWU2rnpe6s3Gc-VobtCkcW?usp=drive_link) by December 2 at 11:59 pm. Plan for 10–12 minutes and 3–5 minutes of Q&A.
- Attend the presentation session, vote for the presentation you liked most, and give one sentence explaining your choice.
- Check the final report includes links to the upstream code and the team fork. Submit the single four-page IEEE PDF through the course assignment by December 5 at 11:59 pm. The deadline has no grace period.
