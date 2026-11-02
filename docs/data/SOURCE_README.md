
# Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction

## Summary

This release contains 60,000 two-dimensional synthetic vascularized skin
phantoms and their paired photoacoustic simulation outputs. Each sample includes
an optical absorption map (`mua`), an optical fluence map (`fluence`), an
initial-pressure distribution (`IP`) and an acoustic sinogram (`sinogram`).

Version: 2.0.0  
Version DOI: 10.5281/zenodo.21785486  
Concept DOI (all versions): https://doi.org/10.5281/zenodo.20494833  
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0)

## Primary data files

- `synthetic_skin_phantoms_part01-of-20_samples_00001-03000.h5`: samples 00001–03000
- `synthetic_skin_phantoms_part02-of-20_samples_03001-06000.h5`: samples 03001–06000
- `synthetic_skin_phantoms_part03-of-20_samples_06001-09000.h5`: samples 06001–09000
- `synthetic_skin_phantoms_part04-of-20_samples_09001-12000.h5`: samples 09001–12000
- `synthetic_skin_phantoms_part05-of-20_samples_12001-15000.h5`: samples 12001–15000
- `synthetic_skin_phantoms_part06-of-20_samples_15001-18000.h5`: samples 15001–18000
- `synthetic_skin_phantoms_part07-of-20_samples_18001-21000.h5`: samples 18001–21000
- `synthetic_skin_phantoms_part08-of-20_samples_21001-24000.h5`: samples 21001–24000
- `synthetic_skin_phantoms_part09-of-20_samples_24001-27000.h5`: samples 24001–27000
- `synthetic_skin_phantoms_part10-of-20_samples_27001-30000.h5`: samples 27001–30000
- `synthetic_skin_phantoms_part11-of-20_samples_30001-33000.h5`: samples 30001–33000
- `synthetic_skin_phantoms_part12-of-20_samples_33001-36000.h5`: samples 33001–36000
- `synthetic_skin_phantoms_part13-of-20_samples_36001-39000.h5`: samples 36001–39000
- `synthetic_skin_phantoms_part14-of-20_samples_39001-42000.h5`: samples 39001–42000
- `synthetic_skin_phantoms_part15-of-20_samples_42001-45000.h5`: samples 42001–45000
- `synthetic_skin_phantoms_part16-of-20_samples_45001-48000.h5`: samples 45001–48000
- `synthetic_skin_phantoms_part17-of-20_samples_48001-51000.h5`: samples 48001–51000
- `synthetic_skin_phantoms_part18-of-20_samples_51001-54000.h5`: samples 51001–54000
- `synthetic_skin_phantoms_part19-of-20_samples_54001-57000.h5`: samples 54001–57000
- `synthetic_skin_phantoms_part20-of-20_samples_57001-60000.h5`: samples 57001–60000

Each HDF5 shard is self-contained and uses lossless gzip compression. Files can
be downloaded independently. Every shard contains the following datasets:

| Key | Shape | Type | Unit / meaning |
|---|---:|---|---|
| `sample_id` | `(N,)` | int32 | Stable sample identifier, 1–60000 |
| `mua` | `(N,256,256)` | float32 | Optical absorption coefficient, mm⁻¹ |
| `fluence` | `(N,256,256)` | float64 | Simulated optical fluence, relative units |
| `IP` | `(N,256,256)` | float32 | Initial pressure, relative units |
| `sinogram` | `(N,256,1207)` | float64 | Detector × time acoustic pressure series |

The three image-valued arrays use `(sample, depth row, lateral column)` order;
the sinogram uses `(sample, detector, time)` order. No missing-value code is
used, and all released values are finite and non-negative.

## 100-sample example subset

`example_subset_100_samples.h5` contains 100 sample IDs distributed uniformly
across the complete 1–60000 range. It includes all four scientific variables
(`mua`, `fluence`, `IP` and `sinogram`) with the same shapes, dtypes, units and
orientation as the full shards. Use it to test loading and visualisation before
downloading a full 5 GB shard.

## Index, integrity and code

- `sample_index.csv` maps every stable sample ID to its HDF5 shard and
  zero-based local index, and gives absorber-class pixel counts.
- `generate_skin_data.m` and `PA_simulation.m` are the reviewed,
  deterministic release versions of the phantom and simulation pipeline.

Zenodo reports the size and checksum of every deposited file. To load one
sample without reading the full file into memory:

```python
import h5py

with h5py.File("example_subset_100_samples.h5", "r") as f:
    sample_id = int(f["sample_id"][0])
    mua = f["mua"][0]
    fluence = f["fluence"][0]
    initial_pressure = f["IP"][0]
    sinogram = f["sinogram"][0]
```

## Software requirements and toolbox installation

Reading and visualising the released HDF5 files does **not** require MATLAB,
ValoMC or k-Wave. Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Generating new phantoms and rerunning the optical and acoustic simulations
requires MATLAB (including Parallel Computing Toolbox for the `parfor` loops)
and the following open-source MATLAB toolboxes:

- [ValoMC](https://github.com/InverseLight/ValoMC) performs the Monte Carlo
  optical transport calculation and produces the fluence map.
- [k-Wave](https://www.k-wave.org/) propagates the initial-pressure field and
  records the acoustic sensor time series (sinogram).

Download both toolboxes from the links above. Their installation directories
are supplied as the `valomc_path` and `kwave_path` arguments to
`PA_simulation.m`; the function adds the required folders to the MATLAB path on
the client and parallel workers. `generate_skin_data.m` can be run first to
create new absorption phantoms, followed by `PA_simulation.m` to create matched
`mua`, `fluence`, `IP` and `sinogram` outputs. For reproducible new runs, set a
fixed `BaseSeed` and record the installed toolbox releases with the optional
`ValoMCVersion` and `KWaveVersion` arguments. These toolboxes are required only
to generate new simulations, not to use the deposited arrays.

## Important orientation note

The historical MATLAB records saved `fluence` and `IP` after a 180-degree
rotation but saved `mua` in the ValoMC-native orientation. This release applies
`rot90(mua, 2)` during packaging. Consequently, the released `mua`, `fluence`
and `IP` arrays share the same pixel coordinates, with rows representing depth,
columns representing lateral position, and row 1 on the sensor side. No numeric
values or source dtypes were otherwise changed.

## Reproducibility limitation

The historical run did not retain the random seeds or per-sample scheduled
phantom parameters. The reviewed MATLAB scripts therefore reproduce the stated
pipeline and parameter distributions and deterministically generate new
phantoms, but they cannot regenerate the released 60,000 phantoms byte-for-byte.
The released arrays themselves are the authoritative research data.

## Citation

Alagappan, G., Ann, L. S., Xia, Y., LiangLi, Z. & Olivo, M. (2026).
Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction, version 2.0.0. Zenodo. https://doi.org/10.5281/zenodo.21785486
