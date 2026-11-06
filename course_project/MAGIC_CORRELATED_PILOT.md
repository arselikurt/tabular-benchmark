# MagicTelescope with correlated noise

Compare the original and screened MLP on MagicTelescope with seeds 3–5 at 0% and 100% added inputs. These match the dataset, split seeds, and noise levels in the earlier independent-noise experiment. Both models use the same machine, environment, training schedule, and test rows. Correlated noise uses a shared Gaussian factor with expected pairwise correlation 0.8, as in the Bank Marketing pilot. The earlier independent Gaussian generator uses a different random construction, so this comparison describes the two noise setups rather than estimating a correlation-only effect.

The same pilot criteria apply: at least 1 percentage point of mean full-noise gain and at most 1 point of mean clean loss.

```sh
.venv/bin/python -u course_project/run.py --datasets MagicTelescope \
  --models mlp screened_mlp --seeds 3 4 5 --noise 0 100 \
  --noise-kind correlated --output output/magic-correlated.csv
```

## Results

The two noise setups share the same test fingerprints for all three seeds. The correlated pilot completed 12 fits.

| Noise setup | Added columns | Original MLP | Screened MLP | Gain (pp) |
| --- | ---: | ---: | ---: | ---: |
| Correlated | 0% | 85.42% | 85.42% | +0.00 |
| Correlated | 100% | 83.71% | 85.28% | +1.57 |
| Independent | 0% | 85.42% | 85.42% | +0.00 |
| Independent | 100% | 83.25% | 85.28% | +2.03 |

The correlated-noise pilot met both criteria. At full noise, screening improved 3 of three pairs and removed 30 of 30 injected columns across the three fits. See [new runs](results/magic_correlated_pilot/runs.csv) and [matched comparison](results/magic_correlated_pilot/paired_accuracy.csv).
