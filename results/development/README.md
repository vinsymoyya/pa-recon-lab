# Archived development evidence

These JSON files are the original local measurements for a width-8, 10-epoch synthetic pipeline demo. They are included to make the README table inspectable; they do not establish a competitive reconstruction method.

| File | Noise SD | Detector dropout | Gain jitter | Corruption seed |
|---|---:|---:|---:|---:|
| `demo_metrics_clean.json` | 0 | 0 | 0 | 20261011 |
| `stress_mild.json` | 0.01 | 0.05 | 0.05 | 20261011 |
| `stress_moderate.json` | 0.025 | 0.10 | 0.10 | 20261012 |
| `stress_severe.json` | 0.05 | 0.20 | 0.20 | 20261013 |

All conditions use the same 10 synthetic test sources. Noise SD is in normalized sinogram units. Each condition is one realization, so differences combine severity and draw variation. The training run used 80 sources, validation used 10, and seed 20261011 generated the included `splits.csv`. Scales in `scales.json` were estimated only from training samples. The checkpoint itself is not bundled in the repository.

Dataset: [Alagappan et al., Synthetic Vascularized Digital Skin Phantoms](https://doi.org/10.5281/zenodo.21785486), CC BY 4.0. Subset file `example_subset_100_samples.h5`, 168,316,012 bytes, MD5 `f9cab78d59b66eba53cc8a80bf6cc6a5`. Source IDs in `splits.csv` refer to that public synthetic release. No patient data is included.

The original demo's uncertainty is not calibrated. MAE loses to zero output while RMSE improves. This development test split has been inspected repeatedly; a final research study needs a new frozen holdout and analytical/learned baselines. New experiments use the package's `run` command to create complete source/environment/configuration manifests.
