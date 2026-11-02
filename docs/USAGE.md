# Usage guide

Research code for a narrowly defined first task: map simulated photoacoustic detector/time sinograms to synthetic initial-pressure (IP) images, with a per-pixel uncertainty estimate.

**This is not a virus/DNA detector or a medical device.** Photoacoustic frequencies do not uniquely identify arbitrary viruses or foreign DNA through skin. This first model reconstructs simulated pressure images only; it cannot establish biological identity or clinical performance.

## Research status

- Runnable training, evaluation, prediction, split-generation, and scale-estimation commands are implemented.
- A 10-epoch demo checkpoint was trained on 80 of the 100 public synthetic samples. Its 10-sample test results are exploratory and do not establish model superiority or real-world performance; see the model card.
- Checksums, simulation assumptions, the signed-sinogram observation, and demo metrics are recorded in [DATA_AUDIT.md](../DATA_AUDIT.md).
- First dataset: Alagappan et al., *Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction*, Zenodo DOI [10.5281/zenodo.21785486](https://doi.org/10.5281/zenodo.21785486), CC BY 4.0. The 100-sample subset is for pipeline development only; the full release has 60,000 synthetic samples (~101 GB).
- Split units are synthetic sample IDs, not patients. These data support only in-distribution synthetic evaluation and do not demonstrate transfer to real skin or disease detection.
- Training includes configurable sinogram noise, detector dropout, and per-detector gain perturbations as robustness hypotheses; these are not validated physical distributions.
- Training can optionally linearly mix paired training examples (`--mixup-alpha`) as a synthetic superposition augmentation. This follows the linear-acoustics approximation; it does not add new tissue anatomies or acquisition domains and must be checked against held-out data.
- The architecture is a starting baseline. Novelty requires a full literature comparison and controlled ablations of each augmentation.

## Install

Use Python 3.10 or later and install the project dependencies in an isolated environment. Install a PyTorch build appropriate for the machine (CPU or CUDA) before installing this package.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch
python -m pip install -e .
```

On Linux/macOS, activate with `source .venv/bin/activate`.

## Obtain and inspect the synthetic phantom subset

Download `example_subset_100_samples.h5` from the Zenodo record linked above and place it in `data/`. The input is `sinogram` (detector × time, 256 × 1207) and target is `IP` (depth × lateral, 256 × 256). `mua` and `fluence` are additional phantom fields, not model targets in this baseline.

The released sinograms are signed acoustic pressure traces. The loader preserves the sign; scale estimation uses a quantile of absolute values. The paired IP target is nonnegative.

```powershell
pa-recon inspect --file data/example_subset_100_samples.h5 --input-key sinogram --target-key IP
```

Confirm the shapes, axis order, sample count, and values against the downloaded file before proceeding. If samples are not on axis 0, pass `--sample-axis`.

## Make a sample-level synthetic-data split

Generate a deterministic 80/10/10 sample-level split from the HDF5 `sample_id` values:

```powershell
pa-recon make-splits --file data/example_subset_100_samples.h5 --output data/splits.csv --seed 20261011
```

You can also create the CSV manually in this format:

```csv
sample_index,group_id,split
0,1,train
1,2,train
2,3,validation
3,4,test
```

The 100-sample subset is not appropriate for final performance claims. For the released synthetic records, individual generated sample IDs are split units; this does not give patient-level generalization evidence.

Estimate separate 99.9th-percentile scales from a seeded sample of training data only:

```powershell
pa-recon estimate-scales --file data/example_subset_100_samples.h5 --splits data/splits.csv --output data/scales.json
$scales = Get-Content data/scales.json | ConvertFrom-Json
```

Use and record those same scales at training and inference; do not normalize each image independently.

## Train and evaluate

```powershell
pa-recon train --file data/example_subset_100_samples.h5 --input-key sinogram --target-key IP --splits data/splits.csv --input-scale $scales.input_scale --target-scale $scales.target_scale --output checkpoints/sinogram_to_ip.pt

pa-recon evaluate --file data/example_subset_100_samples.h5 --splits data/splits.csv --checkpoint checkpoints/sinogram_to_ip.pt --output runs/test_metrics.json
```

Evaluation reports MAE/RMSE and uncertainty coverage on held-out samples. This 100-sample run is a software demonstration, not a publication result; final work requires the full dataset, independent real-measurement validation, conventional and learned baselines, calibration, runtime, and failure analysis.

For CPU runs, `--num-threads 4` limits PyTorch's thread count. To try paired linear-superposition augmentation on training batches, add `--mixup-alpha 0.4`; it is disabled by default and should be selected using validation only.

Run fixed corruption stress cases by setting normalized Gaussian noise, per-detector dropout, and gain jitter. Evaluate clean and corrupted cases separately using the same held-out split and checkpoint:

```powershell
pa-recon evaluate --file data/example_subset_100_samples.h5 --splits data/splits.csv --checkpoint checkpoints/sinogram_to_ip.pt --output runs/test_corrupt.json --test-noise-std 0.05 --test-detector-dropout 0.2 --test-gain-jitter 0.2 --corruption-seed 20261011
```

The evaluator reports against a zero-output baseline; it does not substitute for a conventional acoustic reconstruction baseline.

## Run one prediction

```powershell
pa-recon predict --file data/example_subset_100_samples.h5 --checkpoint checkpoints/sinogram_to_ip.pt --sample-index 3 --output runs/sample_3.npz
```

Prediction reads only the configured sinogram key; it does not need an `IP` target in the inference HDF5 file. The compressed NPZ contains the sinogram, predicted IP reconstruction, and predicted uncertainty. It does not provide a disease label.

## Manuscript and release checklist

1. Run the complete 60,000-sample synthetic release and freeze sample-level splits/configuration.
2. Compare with established analytical reconstruction and learned baselines.
3. Report normalized and physical-unit image metrics, uncertainty calibration, runtime/hardware, and failure cases.
4. Validate on independently acquired real measurements before making real-world claims.
5. Publish code, weights, configuration, and provenance with compatible licensing; write the paper from measured results.

ResearchGate and ORCID can host or index research outputs; they do not replace peer review or validation.
