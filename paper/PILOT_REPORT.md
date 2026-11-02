# Acquisition perturbations in synthetic photoacoustic reconstruction: an open reproducibility pilot

**Exploratory technical report draft — 3 November 2026.** Actual authors must supply names, affiliations, ORCID identifiers, contributions, funding and competing-interest declarations before submission. This report has not undergone peer review and does not claim a new state-of-the-art method.

## Abstract

We release an open software and model pilot for photoacoustic sinogram-to-initial-pressure reconstruction on synthetic skin phantoms. Two residual U-Nets were trained for 50 epochs with and without combined source mixing and detector-level perturbations. Source-disjoint partitions contained 80 training, 10 validation, and 10 test samples. Augmentation reduced pooled RMSE under severe synthetic corruption from 0.107071 to 0.102824 in normalized target units, a 3.97% reduction. Clean MAE increased from 0.053400 to 0.054986. The paired source-group macro-RMSE difference under severe corruption was -0.004451, with an exploratory 95% bootstrap interval [-0.006430,-0.002380]. Predicted uncertainty remained uncalibrated. The release includes attributed data access, both checkpoints, exact manifests, training code, and 45 software reliability checks. Results establish a reproducible trade-off in one simulation setup and motivate a larger confirmatory benchmark; they do not establish pathogen detection or clinical utility.

## 1. Introduction

Photoacoustic reconstruction estimates an initial-pressure field from acoustic measurements. This pilot asks a narrow question: under a fixed simulated acquisition, how do combined training perturbations change reconstruction error across declared synthetic corruption conditions? Encoder-decoder networks [2], group normalization [3], and heteroscedastic regression [4] are established components. We use them as an inspectable baseline, without claiming novelty for their combination. Analytical and contemporary learned reconstruction comparators remain prerequisites for a methods paper.

## 2. Materials and methods

### Data and partitions

We use the unmodified 100-sample subset from Alagappan et al., version 2.0.0, CC BY 4.0 [1]. Each sample supplies a `(256,1207)` signed sinogram and `(256,256)` initial-pressure (IP) target. The full source release contains 60,000 synthetic samples, but only the subset was used here. Fixed simulation assumptions include a 0.01 mm grid, 1500 m/s sound speed, 1000 kg/m³ density and top-boundary line sensors. No participant data were collected. Original historical seeds and per-sample scheduled phantom parameters were not retained; released arrays are the authoritative data.

We verify the published subset MD5 and archive its SHA256. Source IDs define disjoint 80/10/10 partitions, seed 20261011. Training-only 0.999 absolute-value quantiles from a bounded value sample yield input scale 200.3928680419922 and target scale 366.551239013672. Pressure sign is retained. Dataset and split hashes are in `results/kaggle-v1/manifest.json`.

### Model and optimization

A residual U-Net of initial width 16 has three average-pooling levels, transposed-convolution upsampling, skip concatenations, GroupNorm and GELU. Inputs are padded to multiples of eight; final bilinear feature resizing produces target dimensions. A two-channel head estimates mean μ and log variance s, clamped to [-8,4]. The loss averages `0.5 exp(-s)(μ-y)^2 + 0.5s + 0.1|μ-y|` over pixels. The omitted Gaussian constant does not affect optimization. No explicit acoustic operator or measurement-consistency step is present.

Both arms use AdamW, learning rate 0.001, weight decay 0.0001, batch size four, 50 epochs, seed 20261011. Minimum clean validation NLL+L1 selects epochs 46 and 48 for unaugmented and augmented arms, respectively. Augmentation combines paired convex mixing Beta(0.4,0.4), normalized Gaussian noise SD 0.01, detector dropout 0.05, and detector gains uniform on [0.95,1.05]. Mixing is motivated by fixed-operator linear superposition; mixed images are not additional anatomically validated phantoms. This experiment cannot isolate the contribution of each augmentation.

### Stress protocol

| Condition | Noise SD | Detector dropout | Gain half-range |
|---|---:|---:|---:|
| Clean | 0 | 0 | 0 |
| Mild | 0.01 | 0.05 | 0.05 |
| Moderate | 0.025 | 0.10 | 0.10 |
| Severe | 0.05 | 0.20 | 0.20 |

Gain precedes dropout; additive noise follows dropout, so dropped rows may contain noise. A common corruption seed and evaluation batch size provide matched inputs across arms. One realization per source/condition was evaluated. Training ran on Tesla T4, Python 3.13.16, PyTorch 2.11.0+cu130, NumPy 2.1.3. Completion around 1,021 seconds includes setup/download/evaluation and is not an inference speed measurement.

### Endpoints and statistics

We report pooled pixel MAE/RMSE and empirical ±sigma coverage against simulated IP in training-normalized units. A zero-output sanity comparator is included. Separately, paired source-group macro metrics subtract unaugmented from augmented endpoints. Ten matched source groups are resampled 10,000 times, seed 20261011, giving exploratory percentile 95% intervals without multiplicity correction. Pixels are not independent observations. Training-seed and simulator-domain variability are not estimated.

## 3. Results

| Condition | Unaug. MAE | Aug. MAE | Unaug. RMSE | Aug. RMSE | Aug. ±sigma coverage |
|---|---:|---:|---:|---:|---:|
| Clean | 0.053400 | 0.054986 | 0.094284 | 0.094469 | 0.7924 |
| Mild | 0.056003 | 0.054657 | 0.096463 | 0.093939 | 0.7969 |
| Moderate | 0.059904 | 0.057268 | 0.100425 | 0.096362 | 0.7857 |
| Severe | 0.065107 | 0.062507 | 0.107071 | 0.102824 | 0.7488 |

Zero output yields MAE 0.065148 and RMSE 0.168156 in every condition; it is not an analytical acoustic baseline. Augmentation lowers severe pooled RMSE by 3.97%, raises clean MAE by 2.97%, and changes clean RMSE by about +0.20%.

| Condition | Paired macro-RMSE delta | Exploratory 95% interval |
|---|---:|---:|
| Clean | +0.000320 | [-0.001075,+0.001673] |
| Mild | -0.002581 | [-0.003654,-0.001474] |
| Moderate | -0.004175 | [-0.005083,-0.003048] |
| Severe | -0.004451 | [-0.006430,-0.002380] |

Macro RMSE averages group RMSE values and differs from pooled RMSE. Results favor augmentation under the tested corruptions while showing a clean-error trade-off. Sigma coverage changes with condition; no calibration was fitted. All raw per-source reports and exact artifact hashes accompany the release.

## 4. Software reliability

There are 45 named executable checks, including real HDF5 fixtures and a small two-arm end-to-end experiment. They cover leakage, train-only normalization, signed data, dimension handling, deterministic corruptions, hand-computed losses, gradients, inference, checksum recovery, provenance, preflight validation, and paired statistics. Eight new negative contracts initially failed. Fixes reject nonfinite evaluation settings, invalid inference scales/indices, unknown training keys, and colliding or unsafe stress names before output writes. Linux/Windows CI and the public notebook execute the suite. Original notebook version 1 preceded the expanded suite; its training evidence remains unchanged.

Passing checks establishes specific software contracts, not biological validity or image quality. The number 45 alone is not a research-quality measure.

## 5. Discussion and limitations

This study uses one training seed, ten development test sources, one corruption realization per condition, and one simulator. The test split was repeatedly inspected during development. Bootstrap intervals describe source sampling within this small set, not total study uncertainty. Sparse foreground structure can affect MAE and RMSE differently; neither metric establishes useful vascular-detail detection. Conventional acoustic baselines, learned comparators, SSIM, task-based endpoints, calibrated uncertainty and independent measurements have not been completed. Real-scanner, skin-tone and pathology generalization remain untested.

A confirmatory study must reserve a new unseen holdout from the full release, validate a conventional reconstruction comparator with the correct measurement operator, include learned baselines and augmentation ablations, run multiple seeds, time inference with synchronization, and evaluate independently acquired phantom data. Fit any calibration on validation data only. Molecular detection would require a target-specific contrast mechanism and paired reference assays; this reconstruction pilot supplies neither.

## 6. Conclusion

An open reproducibility pilot shows lower RMSE under declared synthetic corruptions with a clean-MAE trade-off. The release supports reproduction and further investigation. It does not establish a novel reconstruction method, hardware robustness, or clinical diagnostic performance.

## Availability and declarations

- Code and weights, MIT: [GitHub](https://github.com/vinsymoyya/pa-recon-lab), [Kaggle Models](https://www.kaggle.com/models/vinsymoyya/pa-recon-lab).
- [Training notebook](https://www.kaggle.com/code/vinsymoyya/pa-recon-robust-synthetic-pilot); exact config `configs/pilot.json`.
- Data, CC BY 4.0: [original](https://doi.org/10.5281/zenodo.21785486), [attributed mirror](https://www.kaggle.com/datasets/vinsymoyya/photoacoustic-skin-phantoms-100-source-mirror).
- Evidence: `results/kaggle-v1/`; [software checks](../docs/RELIABILITY.md).
- Ethics: public synthetic arrays only; no new recruitment or clinical measurements.
- Names, affiliations, ORCID, contributions, funding, competing interests and corresponding author must be supplied by the actual authors before submission; none are inferred here.

## References

1. Alagappan G, Low S, Xia Y, Zhen L, Olivo M. Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction. Version 2.0.0. Zenodo; 2026. [doi:10.5281/zenodo.21785486](https://doi.org/10.5281/zenodo.21785486).
2. Ronneberger O, Fischer P, Brox T. U-Net: Convolutional Networks for Biomedical Image Segmentation. 2015. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597). Original task: segmentation; here the general architecture family is adapted to regression.
3. Wu Y, He K. Group Normalization. 2018. [arXiv:1803.08494](https://arxiv.org/abs/1803.08494).
4. Kendall A, Gal Y. What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? 2017. [arXiv:1703.04977](https://arxiv.org/abs/1703.04977).
