# Model card: synthetic reconstruction development baseline

## Intended use

Research comparison of simulated detector/time sinogram to synthetic initial-pressure (IP) reconstruction. Output is an image estimate plus a model-predicted per-pixel uncertainty scale.

## Out of scope

Clinical diagnosis, pathogen or foreign-DNA detection, multispectral oxygenation estimation, or use on unvalidated scanners and populations.

## Training data and labels

The local demo checkpoint, `checkpoints/sinogram_to_ip_demo.pt`, was trained for 10 epochs on 80 samples from the synthetic vascularized digital-skin phantom release by Alagappan et al. ([Zenodo DOI 10.5281/zenodo.21785486](https://doi.org/10.5281/zenodo.21785486), CC BY 4.0). Weights are not distributed in the source repository. It is a pipeline demonstration, not a publication-ready model. The 100-sample subset contains no patient data and cannot establish performance on real subjects, scanners, or pathology.

Training can apply normalized Gaussian noise, independent detector dropout, and per-detector gain perturbations. These stressors are configurable and not validated distributions of real scanner failure. The current test evaluator includes a zero-output baseline; an accepted analytical acoustic reconstruction baseline remains required for a research comparison.

## Evaluation

On the 10-sample held-out subset, clean normalized MAE was 0.0685 versus 0.0651 for a zero-output baseline; normalized RMSE was 0.1146 versus 0.1682. In a fixed-seed stress profile (clean, mild, moderate, severe), MAE ranged from 0.0685 to 0.0737 and remained worse than zero output in every setting; RMSE ranged from 0.1141 to 0.1168 and remained better. Severe corruption reduced empirical one-sigma coverage to 0.768. These small-subset metrics are exploratory, not evidence of superiority or calibrated uncertainty. Final evaluation requires the complete release, frozen partitions, analytical and learned baselines, and independent real-measurement validation.

## Known limitations

The existing development test split has been repeatedly inspected. Reserve a new final holdout before a publication study. The original checkpoint metadata's `claim_boundary` string incorrectly described image-to-image backprojection; its actual training keys and shapes are `sinogram` → `IP`. New checkpoints use a corrected task description. This correction does not change the old weights or reported metrics.

The current model is a residual U-Net mapping the 2-D sinogram representation to an IP image, with a Gaussian heteroscedastic output. Its uncertainty is not guaranteed to be calibrated. It does not identify viruses, DNA, infection, or oxygen saturation. Synthetic data cannot establish skin-tone fairness, scanner robustness, or clinical performance.
