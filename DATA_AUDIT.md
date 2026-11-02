# Dataset and demo-run audit

## Source and integrity

- Dataset: Alagappan et al., *Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction*, Zenodo v2.0.0, DOI [10.5281/zenodo.21785486](https://doi.org/10.5281/zenodo.21785486), CC BY 4.0.
- File: `example_subset_100_samples.h5`; 168,316,012 bytes; MD5 `f9cab78d59b66eba53cc8a80bf6cc6a5` (matches the Zenodo record).
- HDF5 inspection: 100 samples; `sinogram` shape `(100, 256, 1207)` and `IP` shape `(100, 256, 256)`; sample axis 0.
- The released simulation script sets grid spacing to 0.01 mm, sound speed to 1500 m/s, density to 1000 kg/m³, and a full-width line sensor at the top grid row. The released set therefore does not test variation in these system parameters.

## Data handling observation

The downloaded README describes all released arrays as nonnegative. The inspected sample 0 sinogram is signed (`p01=-31.1423`, `p50=-3.34281`, `p99=149.287`), as expected for an acoustic pressure trace; `IP` is nonnegative (`p01=0.000852279`, `p50=0.00216607`, `p99=326.991`). The loader preserves sinogram signs. Scale estimation uses the 99.9th percentile of absolute values from a seeded sample of training data.

## Demo protocol and results

- Split: seed `20261011`, 80 train / 10 validation / 10 test; rows are grouped by unique `sample_id`.
- Scales: input `200.3928680419922`, target `366.5512390136719`, 99.9th percentile, sampled from the 80 training examples.
- Demo checkpoint: 10 epochs, CPU, width 8, batch size 2. This small subset and short run are for pipeline verification only.
- Clean held-out test (10 samples): normalized MAE 0.06848 (zero-output baseline 0.06515); normalized RMSE 0.11458 (zero-output baseline 0.16816).
- One corrupted-input case (normalized noise 0.05, detector dropout 0.20, gain jitter 0.20): normalized MAE 0.07380; normalized RMSE 0.11686; same zero-output baseline 0.06515 / 0.16816.
- Empirical one-sigma coverage: 0.829 clean, 0.766 corrupted. This is not a calibration study.

### Fixed-seed synthetic stress profile

The same 10 held-out samples were evaluated with one deterministic corruption realization per level (seed varies by level). This is a descriptive stress check, not independent replication or an estimate of a real scanner's failure distribution.

| Condition | Noise SD (normalized) | Detector dropout | Gain jitter | Model MAE | Zero MAE | Model RMSE | 1σ coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | 0 | 0 | 0 | 0.06848 | 0.06515 | 0.11458 | 0.8289 |
| Mild | 0.01 | 0.05 | 0.05 | 0.06906 | 0.06515 | 0.11415 | 0.8295 |
| Moderate | 0.025 | 0.10 | 0.10 | 0.07041 | 0.06515 | 0.11447 | 0.8179 |
| Severe | 0.05 | 0.20 | 0.20 | 0.07372 | 0.06515 | 0.11675 | 0.7681 |

The model's MAE remains worse than zero output in all four settings, while RMSE is better. The falling coverage at severe corruption suggests uncertainty is not calibrated under these shifts. A visual comparison for training sample index 1 (source ID 607) is saved locally as `runs/diagnostic_sample_1.png` (top-left target, top-right estimate, bottom-left absolute error, bottom-right predicted sigma; each panel independently scaled except target/estimate share a scale). This image is not held-out visual evidence.

The development test sources have been repeatedly inspected. Future publication experiments must reserve a new final holdout. Raw development metrics and normalization/split artifacts are archived in `results/development/`.

These results do not establish superiority, sensor robustness, real-world generalization, or a clinical use. The model has not been compared with an analytical reconstruction baseline or validated on measured data.
