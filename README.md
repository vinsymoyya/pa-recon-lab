# PA Recon Lab

**Reproducible photoacoustic reconstruction, acquisition stress tests, and uncertainty evaluation.**

[![CI](https://github.com/vinsymoyya/pa-recon-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/vinsymoyya/pa-recon-lab/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Status](https://img.shields.io/badge/Research-development-orange.svg)](MODEL_CARD.md)

[Quickstart](#quickstart) · [GPU notebook](kaggle_kernel/train.ipynb) · [Results](#measured-results) · [Research roadmap](ROADMAP.md) · [Contribute](CONTRIBUTING.md)

![From signed detector signals to reconstruction and an uncertainty map, evaluated against held-out synthetic initial pressure](docs/assets/pipeline.svg)

PA Recon Lab provides a compact, inspectable PyTorch baseline for mapping detector/time sinograms to initial-pressure images. It keeps data provenance, training splits, checkpoints, corruption settings, and evaluation outputs together so other researchers can reproduce and challenge a result.

The current release is an early research toolkit. Its demo improves RMSE over zero output but loses on MAE. Real measurements, analytical reconstruction baselines, and uncertainty calibration remain open work. Diagnostic and universal virus/DNA detection claims are unsupported by this project.

## What you can do today

- Download a public synthetic subset with checksum verification.
- Train a residual U-Net with optional paired source mixing, noise, detector dropout, and gain perturbations.
- Compare unaugmented and augmented training with one declared configuration.
- Evaluate clean and corrupted measurements with per-sample/group metrics and bootstrap intervals.
- Predict from an input-only HDF5 file using the same checkpoint format locally and on Kaggle.
- Audit the split, dataset hash, configuration, source hash, and environment in each experiment manifest.

## Quickstart

```bash
git clone https://github.com/vinsymoyya/pa-recon-lab.git
cd pa-recon-lab
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e .
pa-recon download-data
```

The download is the 168 MB, 100-sample development subset from [Alagappan et al., Zenodo](https://doi.org/10.5281/zenodo.21785486), licensed CC BY 4.0. Code is MIT; data retains its own license. Install an appropriate CUDA build of PyTorch before training on your GPU.

Run the declared GPU comparison:

```bash
pa-recon run --config configs/pilot.json --output runs/pilot --require-cuda
```

The pilot trains two arms for 50 epochs each, selects each checkpoint by validation loss, and writes clean/stress metrics to a new output directory. It refuses to overwrite an existing run. It is a software and experiment pilot on 80 training, 10 validation, and 10 test sources; the subset cannot support final performance claims.

**Use a hosted GPU:** open the [Kaggle notebook](kaggle_kernel/train.ipynb), or follow the [CLI launch guide](kaggle_kernel/README.md). The notebook embeds the package source and calls the same CLI. GPU and Internet must be enabled; no GPU session has been verified yet.

**Use a CPU or your own data:** see the [manual training and inference guide](docs/USAGE.md). HDF5 inputs are signed `sinogram` arrays, `(N, detectors, time)`; targets are `IP`, `(N, depth, lateral)`. The two spatial dimensions need not match.

## Run inference

After training, the checkpoint works directly with:

```bash
pa-recon predict --file data/example_subset_100_samples.h5 \
  --checkpoint runs/pilot/augmented/model.pt --sample-index 3 \
  --output runs/prediction.npz
```

The NPZ contains `reconstruction`, `predicted_sigma`, and `sinogram` in the dataset's relative units. Inference does not require a target dataset. No pretrained download is released yet; weights from the completed notebook are downloadable as Kaggle outputs. Predicted sigma needs calibration before use as a reliable interval.

## Measured results

The existing local demo used width 8 and 10 epochs on the 100-sample subset. These development results were inspected during implementation; a future study needs a new frozen final holdout.

| Input condition | Model MAE ↓ | Zero MAE ↓ | Model RMSE ↓ | Zero RMSE ↓ | 1σ coverage |
|---|---:|---:|---:|---:|---:|
| Clean | 0.06848 | 0.06515 | 0.11458 | 0.16816 | 0.8289 |
| Mild corruption | 0.06906 | 0.06515 | 0.11415 | 0.16816 | 0.8295 |
| Moderate corruption | 0.07041 | 0.06515 | 0.11447 | 0.16816 | 0.8179 |
| Severe corruption | 0.07372 | 0.06515 | 0.11675 | 0.16816 | 0.7681 |

Metrics are normalized by a fixed training-only scale. Each stress condition uses one corruption realization on the same 10 test sources. These numbers show a metric trade-off and uncertainty sensitivity; the zero baseline is a sanity check, not an acoustic reconstruction comparator. [Raw metrics and provenance](results/development/README.md) · [Data audit](DATA_AUDIT.md) · [Model card](MODEL_CARD.md).

## Research direction

Our target is reconstruction that remains useful as acquisition conditions change: explicitly model the measurement operator, preserve data consistency, quantify failure, and evaluate outside the training setup. The [roadmap](ROADMAP.md) sets evidence gates for physics-conditioned reconstruction, controlled simulation diversity, calibrated uncertainty, and independent measured-data evaluation. These are research hypotheses; novelty and gains must be established experimentally against prior methods.

## Contribute

The most useful first contributions are an analytical baseline, an independent phantom benchmark, uncertainty calibration, and reproducible failure cases. See [CONTRIBUTING.md](CONTRIBUTING.md), [open issues](https://github.com/vinsymoyya/pa-recon-lab/issues), and the [research blueprint](PAI_RESEARCH_BLUEPRINT.md). Negative results with a complete manifest are welcome.

If the toolkit saves you experiment setup time, a star helps others find it. Fork it to add a reproducible method or benchmark; improvements are judged by evidence.

## Cite and reuse

Use [CITATION.cff](CITATION.cff) to cite the software, and cite each dataset and method you use separately. This repository has no associated peer-reviewed paper yet. [License](LICENSE) · [Security](SECURITY.md) · [Community standards](CODE_OF_CONDUCT.md).
