# Acquisition investigation evidence — 5 November 2026

- `measured/`: completed metadata audit and diagnostics for two **measured** OpenQPAIData phantom records, with selection/integrity manifests and explicit assumed sound speed/time origin.
- `synthetic/`: completed ten-condition study of the two unchanged v0.2.0 models on ten **synthetic development** source groups. 560 predictions do not create 560 independent sources.
- `paired_comparisons.json`: later paired source-level bootstrap comparisons, after averaging corruption repetitions within each source; no multiplicity correction or training-seed variation.
- `physics_probe.json`: analytic point-arrival focusing and timing-compensation probe; not a full-wave solver validation.
- PNGs: derived figures; measured source imagery remains CC BY 4.0 with credit to Gröhl et al., [OpenQPAIData](https://doi.org/10.5281/zenodo.14044853). Synthetic source data is credited to Alagappan et al., [Zenodo](https://doi.org/10.5281/zenodo.21785486), CC BY 4.0. Code retains MIT.

Raw HDF5 records and detector diagnostic NPZ maps stay in ignored local data/run directories. Reproduce them with the scripts in [the investigation report](../../docs/ACQUISITION_FAILURES.md). Raw values include device offsets and uncalibrated amplitudes; they are not normalized physical-pressure measurements. FBP reconstructions and material labels are not initial-pressure truth.

The initial synthetic run predates the implementation-hash field now written by future studies. Its dataset, checkpoint, split and complete configuration hashes/settings are preserved. No model was retrained or selected using these failures.
