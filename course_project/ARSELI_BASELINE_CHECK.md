# MagicTelescope baseline verification

I reran the original Random Forest and MLP on MagicTelescope (OpenML task 361065) using an Apple M5 Mac, macOS 26.5.1, and Python 3.9.6. I used split seeds 0, 1, and 2 with no added noise. For each seed, both models had the same test split fingerprint and were evaluated on the same 2,810 rows.

| Model | Seed 0 | Seed 1 | Seed 2 | My mean test accuracy | Published default | Difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 84.52% | 86.05% | 85.84% | 85.47% | 85.14% | +0.33 percentage points |
| MLP | 84.66% | 85.84% | 85.09% | 85.20% | 85.20% | 0.00 percentage points |

The MLP matched the published mean. Random Forest was 0.33 percentage points above its published mean and led the MLP by 0.27 points in my rerun. This comparison covers three splits of one dataset with clean inputs.

[My run results](results/verification/arseli-magic-baselines.csv) · [Published reference scores](results/reproduction.csv)