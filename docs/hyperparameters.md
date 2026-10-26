# Hyperparameter choices

How the defaults in `configs/train-mlp.toml` were chosen, and how much to trust each choice.

## Setup

- Data: the 5000-sample dataset from `foilnet generate` (seed 0), split 70/15/15 at random (seed 0): 3500 train, 750 validation, 750 test.
- Every number below is **validation** loss or validation RMSE. The test set was not used to choose anything. It was used once, for the final report, after the settings were fixed.
- "Val loss" is mean squared error in standardised units, averaged over the two targets, at the best epoch. RMSE columns are in physical units.
- Each configuration was trained once (seed 0). The seed check at the end shows how much that matters.
- Reproduce with `python -m foilnet.analysis.hparam_search --data data/naca4-lhs.npz --out docs/hyperparameters.md` (about 5 minutes on 4 workers; it overwrites this file with the raw tables only).

## Result

| setting | value | why |
|---|---|---|
| hidden layers | 128 x 128 | best in the search, but see "what the search cannot tell us" |
| activation | tanh | the solver output is a smooth function of the inputs; SiLU matched it, ReLU was about 2x worse |
| optimiser | AdamW | Adam with decoupled weight decay |
| learning rate | 0.01, cosine decay to 1% | far better than 0.001 or 0.003 at a fixed epoch budget, and a constant rate was about 6x worse |
| weight decay | 1e-4 | no measurable effect; kept because it was the nominal winner |
| epochs | 400 | 200 to 400 roughly halves the loss; 800 adds under 1%; 1600 adds 7% for 3.4x the time |
| batch size | 128 | not searched; a standard value for a dataset this size |
| early stopping | off, restore best epoch | with cosine decay the best epoch is nearly always the last few |

Final test-set result (one run, seed 0): Cl RMSE 7.0e-4 and Cm RMSE 1.65e-4, which are 0.11% and 0.31% of the label spread. Validation gave 6.9e-4 and 1.58e-4, so the choices did not overfit the validation set.

## What the search shows clearly

- **Learning rate and schedule matter most.** At 200 epochs, lr 0.001 is still improving when training stops (its best epoch is the last one), and a constant learning rate plateaus at a loss about 6x higher than a cosine decay.
- **Training length matters up to about 400 epochs**, then returns diminish quickly.
- **ReLU is clearly worse** than tanh or SiLU (val loss 1.07e-5 against 4.9e-6 and 5.1e-6), as expected for a smooth target.

## What the search cannot tell us

Training the same configuration with four different seeds gave validation loss between 4.9e-6 and 6.4e-6 and Cl RMSE between 6.2e-4 and 9.2e-4, a spread of roughly 30%. So:

- The differences between the four architectures at lr 0.01 (val loss 8.1e-6 to 8.7e-6) are **within seed noise**. 128x128 is the nominal winner, not a proven one, and a smaller network would do about as well.
- Weight decay of 0, 1e-5 and 1e-4 gave val loss of 8.09e-6, 8.11e-6 and 8.09e-6: no effect.
- Tanh against SiLU (4.92e-6 against 5.05e-6) is a tie.

Not searched: batch size, depth beyond three layers, width beyond 128, other optimisers.

For scale, the labels differ from a converged (640-panel) solve by an RMS of about 1.6e-4 in Cl and 1.9e-4 in Cm on the test set. That is not noise (the network learns the 160-panel solver, not the converged one), but it is a useful yardstick: against a converged solver, the surrogate's error is about 4x this for Cl and about the same size for Cm.

## The raw search tables

#### Stage 1: architecture x learning rate (200 epochs)

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 128x128 | 0.01 | cosine | 0 | 200 | 199 | 8.09e-06 | 0.000988 | 0.000199 | 22 |
| 64x64 | 0.01 | cosine | 0 | 200 | 200 | 8.7e-06 | 0.00106 | 0.000205 | 15 |
| 64x64x64 | 0.01 | cosine | 0 | 200 | 200 | 8.74e-06 | 0.00116 | 0.000202 | 23 |
| 64x64x64 | 0.003 | cosine | 0 | 200 | 199 | 1.03e-05 | 0.00161 | 0.000202 | 21 |
| 128x128 | 0.003 | cosine | 0 | 200 | 200 | 1.03e-05 | 0.00133 | 0.000217 | 23 |
| 32x32 | 0.01 | cosine | 0 | 200 | 197 | 1.06e-05 | 0.00146 | 0.000215 | 20 |
| 64x64 | 0.003 | cosine | 0 | 200 | 197 | 1.59e-05 | 0.00171 | 0.000267 | 15 |
| 32x32 | 0.003 | cosine | 0 | 200 | 196 | 2.86e-05 | 0.0022 | 0.000362 | 20 |
| 64x64x64 | 0.001 | cosine | 0 | 200 | 199 | 3.58e-05 | 0.00316 | 0.000368 | 20 |
| 128x128 | 0.001 | cosine | 0 | 200 | 199 | 5.13e-05 | 0.00378 | 0.000441 | 22 |
| 64x64 | 0.001 | cosine | 0 | 200 | 199 | 5.43e-05 | 0.00338 | 0.000482 | 22 |
| 32x32 | 0.001 | cosine | 0 | 200 | 200 | 0.000213 | 0.00675 | 0.000953 | 20 |

#### Stage 2: schedule x weight decay (200 epochs)

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 128x128 | 0.01 | cosine | 0.0001 | 200 | 199 | 8.09e-06 | 0.000984 | 0.0002 | 28 |
| 128x128 | 0.01 | cosine | 0 | 200 | 199 | 8.09e-06 | 0.000988 | 0.000199 | 27 |
| 128x128 | 0.01 | cosine | 1e-05 | 200 | 199 | 8.11e-06 | 0.00099 | 0.0002 | 28 |
| 128x128 | 0.01 | constant | 0 | 200 | 163 | 4.9e-05 | 0.00386 | 0.000421 | 27 |
| 128x128 | 0.01 | constant | 0.0001 | 200 | 76 | 4.95e-05 | 0.00279 | 0.00048 | 19 |
| 128x128 | 0.01 | constant | 1e-05 | 200 | 79 | 5.61e-05 | 0.00368 | 0.000477 | 18 |

#### Stage 3: epochs

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 128x128 | 0.01 | cosine | 0.0001 | 1600 | 1529 | 4.57e-06 | 0.000603 | 0.000154 | 148 |
| 128x128 | 0.01 | cosine | 0.0001 | 800 | 780 | 4.89e-06 | 0.000666 | 0.000158 | 76 |
| 128x128 | 0.01 | cosine | 0.0001 | 400 | 395 | 4.92e-06 | 0.00069 | 0.000158 | 44 |
| 128x128 | 0.01 | cosine | 0.0001 | 200 | 199 | 8.09e-06 | 0.000984 | 0.0002 | 25 |

### Activation and seed checks (128x128, lr 0.01, cosine, weight decay 1e-4, 400 epochs)

| check | val loss | val RMSE cl | val RMSE cm |
|---|---|---|---|
| tanh, seed 0 | 4.92e-06 | 0.00069 | 0.000158 |
| silu, seed 0 | 5.05e-06 | 0.000488 | 0.000166 |
| relu, seed 0 | 1.07e-05 | 0.00148 | 0.000215 |
| tanh, seed 1 | 5.14e-06 | 0.000616 | 0.000164 |
| tanh, seed 2 | 6.39e-06 | 0.000875 | 0.000177 |
| tanh, seed 3 | 6.16e-06 | 0.00092 | 0.000172 |

## CST dataset

The nine-input CST data was searched the same way (`configs/dataset-cst.toml`: 10,000 shapes at
320 panels, split 7000 / 1500 / 1500, seed 0; validation set only, one run per configuration).
The grid was widened to larger networks, since the problem has more inputs. The result is saved as
`configs/train-mlp-cst.toml`.

| setting | NACA (`train-mlp.toml`) | CST (`train-mlp-cst.toml`) |
|---|---|---|
| hidden layers | 128 x 128 | 256 x 256 |
| learning rate | 0.01 | 0.003 |
| weight decay | 1e-4 | 0 |
| epochs | 400 | 800 |

What stands out:

- **Epochs mattered much more than architecture.** At the best configuration, validation loss
  fell from 5.2e-6 (200 epochs) to 2.8e-6 (400) to 2.0e-6 (800), with the best epoch still the
  last few. The NACA search gained under 1% from 400 to 800 epochs. I did not try more than 800,
  so the CST loss may still be improvable.
- **Architecture differences at 200 epochs are within seed noise.** The NACA search showed seed
  noise of roughly 30% in validation loss; here the ten configurations span 5.2e-6 to 8.0e-6.
  256 x 256 is the nominal winner, and 128 x 128 is close behind (5.4e-6).
- **Weight decay did nothing** (identical to three digits at 0 and 1e-4).
- Learning rate 0.003 beat 0.01 for the wider networks but not for 128 x 128, again within noise.

#### Stage 1: architecture x learning rate (200 epochs)

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 0.003 | cosine | 0.0001 | 200 | 200 | 5.18e-06 | 0.0012 | 0.000168 | 58 |
| 128x128 | 0.003 | cosine | 0.0001 | 200 | 199 | 5.44e-06 | 0.0012 | 0.000174 | 35 |
| 128x128 | 0.01 | cosine | 0.0001 | 200 | 199 | 5.58e-06 | 0.00132 | 0.000169 | 35 |
| 128x128x128 | 0.003 | cosine | 0.0001 | 200 | 199 | 5.64e-06 | 0.00129 | 0.000172 | 50 |
| 256x256 | 0.01 | cosine | 0.0001 | 200 | 196 | 6.24e-06 | 0.00137 | 0.00018 | 59 |
| 64x64 | 0.01 | cosine | 0.0001 | 200 | 198 | 6.33e-06 | 0.00142 | 0.000178 | 29 |
| 256x256x256 | 0.003 | cosine | 0.0001 | 200 | 200 | 6.64e-06 | 0.00142 | 0.000185 | 64 |
| 128x128x128 | 0.01 | cosine | 0.0001 | 200 | 198 | 7.17e-06 | 0.00155 | 0.000186 | 49 |
| 256x256x256 | 0.01 | cosine | 0.0001 | 200 | 195 | 7.96e-06 | 0.00152 | 0.000206 | 64 |
| 64x64 | 0.003 | cosine | 0.0001 | 200 | 193 | 8.02e-06 | 0.00152 | 0.000207 | 29 |

#### Stage 2: schedule x weight decay (200 epochs)

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 0.003 | cosine | 0 | 200 | 200 | 5.18e-06 | 0.0012 | 0.000168 | 40 |
| 256x256 | 0.003 | cosine | 0.0001 | 200 | 200 | 5.18e-06 | 0.0012 | 0.000168 | 41 |

#### Stage 3: epochs

| hidden | lr | schedule | weight decay | epochs | best epoch | val loss | val RMSE cl | val RMSE cm | seconds |
|---|---|---|---|---|---|---|---|---|---|
| 256x256 | 0.003 | cosine | 0 | 800 | 786 | 1.99e-06 | 0.000675 | 0.000109 | 148 |
| 256x256 | 0.003 | cosine | 0 | 400 | 397 | 2.79e-06 | 0.000807 | 0.000128 | 89 |
| 256x256 | 0.003 | cosine | 0 | 200 | 200 | 5.18e-06 | 0.0012 | 0.000168 | 52 |

