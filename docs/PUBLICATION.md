# Reproduce the public release

## Artifact boundaries

Code and weights are MIT. The unmodified synthetic source mirror is CC BY 4.0 and credits Alagappan et al., [Zenodo v2.0.0](https://doi.org/10.5281/zenodo.21785486). The mirror is a convenience copy; the authors' original record remains authoritative. Do not upload a whole checkout, credentials, private logs, or unrelated projects.

1. Download a completed notebook's outputs and verify `manifest.json` has both completed arms and `status=complete`.
2. Archive exact evidence and derive paired comparisons:

   ```bash
   python scripts/summarize_pilot.py --source runs/kaggle-output/pa-recon-results --output results/kaggle-v1
   ```

3. Verify the original data checksum with `pa-recon download-data` and stage an explicitly selected bundle:

   ```bash
   python scripts/prepare_kaggle_release.py --results runs/kaggle-output/pa-recon-results --output runs/publication
   ```

4. Inspect staging metadata, licenses and filenames. Run the 45-check suite and build. For a new release, review model/data cards against that run rather than copying this pilot's measurements.
5. Publish with the Kaggle CLI. On Windows, use native resolved paths; forward slashes in a relative upload path can break the SDK upload-cache filename.

   ```powershell
   kaggle datasets create -p (Resolve-Path runs/publication/data).Path --public
   kaggle models create -p (Resolve-Path runs/publication/model).Path
   kaggle models instances create -p (Resolve-Path runs/publication/unaugmented).Path
   kaggle models instances create -p (Resolve-Path runs/publication/augmented).Path
   ```

These creation commands are for a first publication; existing resources require their version/update commands. Keep credentials in the normal CLI authentication store. Public training notebook publication uses `kaggle kernels push -p kaggle_kernel --timeout 3600`, which starts a run; it is not a metadata-only operation.

## Verify public reuse

Pull public metadata, list files, download each model variation, compare SHA256 with `results/kaggle-v1/artifact_sha256.json`, and run input-only prediction. Check public data visibility and retained source attribution. Do not equate an upload request with a usable published artifact. Track real downloads, reproductions, and contributed benchmarks; no download or star growth is guaranteed.

## Research submission

The [technical report](../paper/PILOT_REPORT.md) is a transparent pilot draft. A final methods paper needs a new frozen holdout, validated conventional and learned comparators, multiple seeds, independent measured data, a completed novelty review, and actual author declarations. No journal acceptance or clinical validation is implied by public hosting or software checks.
