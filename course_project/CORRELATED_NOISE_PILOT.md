# Bank Marketing pilot with correlated noise

This pilot applies the fixed screening method to one additional benchmark dataset, Bank Marketing (OpenML dataset 44126, task 361066). Compare the original and screened MLP on seeds 0–2 at 0% and 100% added columns. Both models use the same machine, package versions, train and test rows, noise draws, and upstream training settings. The pilot contains 12 fits.

For each row, draw one shared standard Gaussian and one independent standard Gaussian for each added column. Combine them with weights `sqrt(0.8)` and `sqrt(0.2)`. The injected columns have pairwise correlation 0.8 in expectation. Scale and shift each column using training feature statistics. Noise is generated independently of the rows' values and labels, conditional on those calibration statistics.

Use the prior study's criteria. Mean test accuracy should gain at least 1 percentage point at full noise and lose at most 1 point on clean data. Report the paired gains by seed and retained noise counts. This joint check changes the dataset and noise type. It does not isolate the effect of correlation.

```sh
.venv/bin/python -u course_project/run.py --datasets bank-marketing \
  --models mlp screened_mlp --seeds 0 1 2 --noise 0 100 \
  --noise-kind correlated --output output/bank-correlated.csv
```

## Results

Bank Marketing had 10,578 rows and 7 original numerical inputs. All 12 fits completed and shared the expected test fingerprint within each seed.

| Added columns | Original MLP | Screened MLP | Gain (pp) |
| --- | ---: | ---: | ---: |
| 0% | 78.97% | 78.97% | +0.00 |
| 100% | 78.44% | 78.85% | +0.41 |

The pilot did not meet the predeclared criteria. Screening improved 2 of 3 full-noise pairs and removed 20 of 21 injected columns across the three full-noise fits. See [per-run data](results/correlated_pilot/runs.csv) and [paired results](results/correlated_pilot/paired_accuracy.csv).
