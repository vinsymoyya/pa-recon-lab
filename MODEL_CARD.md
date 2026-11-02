# Model card: PA Recon Lab synthetic pilot

## Access and license

- [PyTorch weights](https://www.kaggle.com/models/vinsymoyya/pa-recon-lab): `unaugmented` and `augmented` variations, each with a checkpoint, hashes, run manifest, scales, splits, and metrics.
- [Public training notebook](https://www.kaggle.com/code/vinsymoyya/pa-recon-robust-synthetic-pilot).
- [Attributed data mirror](https://www.kaggle.com/datasets/vinsymoyya/photoacoustic-skin-phantoms-100-source-mirror).
- Code and released weights: MIT. Original training data: CC BY 4.0; cite Alagappan et al., [Zenodo v2.0.0](https://doi.org/10.5281/zenodo.21785486). The mirror curator is not the original dataset creator.

## Intended use

Research comparison of synthetic detector/time sinogram to initial-pressure (IP) reconstruction in one simulated acquisition setup. Output is an IP image and an uncalibrated per-pixel Gaussian uncertainty scale. This is an engineering baseline with no established architectural novelty. Clinical diagnosis, pathogen/foreign-DNA detection, oxygenation estimation, and deployment on unvalidated scanners are outside the evidence provided.

## Architecture and training

Residual U-Net, width 16, GroupNorm/GELU, three pooling levels, two output heads. Signed input `(1,256,1207)` maps to `(1,256,256)` through a final bilinear feature resize. Log variance is clamped to [-8,4]. Objective: mean Gaussian NLL, omitting its constant, + 0.1 MAE. No explicit acoustic operator or data-consistency term is implemented.

Kaggle notebook version 1 completed two 50-epoch arms on Tesla T4, PyTorch 2.11.0+cu130, Python 3.13.16, NumPy 2.1.3. AdamW: learning rate 0.001, weight decay 0.0001, batch size 4, seed 20261011. Minimum validation NLL+L1 selects epoch 46 for unaugmented and 48 for augmented. Checkpoints contain tensors and primitive metadata; use `torch.load(..., weights_only=True)`.

Augmented training uses normalized Gaussian noise 0.01, detector dropout 0.05, per-detector uniform gain jitter ±0.05, and paired linear mixup Beta(0.4,0.4). The unaugmented arm disables all four. Mixing sinograms and IP targets is motivated by fixed-operator linear superposition; mixed targets are not additional anatomically validated skin phantoms. Stressors are not validated distributions of scanner failures.

## Data provenance

100 synthetic samples, no patient data, 80/10/10 source-disjoint train/validation/test split, seed 20261011. Training-only absolute quantiles give input scale 200.3928680419922 and target scale 366.551239013672. Signals retain their sign. Dataset SHA256: `a25bb58f77f64f5a6b948c2484597cb888f61e6e82179f7f29b423db4cdee7bd`. Split byte SHA256: `9fba8c0491ba3d3c1fba52d9a2200b0dbbfcb6851491f63ff6d770c81dd36a2e`. [Data audit](DATA_AUDIT.md).

## Evaluation: notebook version 1

Pooled pixel metrics in normalized target units, ten development test sources:

| Condition | Unaug. MAE | Aug. MAE | Unaug. RMSE | Aug. RMSE | Aug. 1σ coverage |
|---|---:|---:|---:|---:|---:|
| Clean | 0.053400 | 0.054986 | 0.094284 | 0.094469 | 0.7924 |
| Mild | 0.056003 | 0.054657 | 0.096463 | 0.093939 | 0.7969 |
| Moderate | 0.059904 | 0.057268 | 0.100425 | 0.096362 | 0.7857 |
| Severe | 0.065107 | 0.062507 | 0.107071 | 0.102824 | 0.7488 |

Zero output: MAE 0.065148, RMSE 0.168156 in all conditions. It is a sanity comparator, not a conventional acoustic reconstruction method. Severe stress reduces pooled RMSE by 3.97% relative to unaugmented training; clean MAE increases by 2.97%. Augmentation is not uniformly better.

Paired source-group macro-RMSE delta (augmented minus unaugmented), severe: -0.004451, exploratory 95% percentile interval [-0.006430,-0.002380], 10,000 bootstrap resamples. This endpoint differs from pooled RMSE. Multiple comparisons are not corrected; training-seed variability is unmeasured. [Raw evidence](results/kaggle-v1/README.md).

## Reliability and limitations

The software provides 45 automated checks covering leakage, normalization, numerics, inference, provenance, downloads, preflight validation, and paired analysis. Passing them establishes those software contracts, not diagnostic accuracy, generalization, or calibrated uncertainty.

Sigma is uncalibrated. Coverage alone does not establish useful uncertainty or out-of-distribution detection. The test split was inspected during development; a confirmatory study must reserve a new frozen holdout. Full-release evaluation, multiple seeds, analytical and learned comparators, parameterized physics shifts, and independent measured data remain required. No inference-speed or clinical superiority claim is made.

The old local 10-epoch demo remains separately archived in `results/development/`. Its task-description string was incorrect; actual training was sinogram→IP. It is not the released 50-epoch model.
