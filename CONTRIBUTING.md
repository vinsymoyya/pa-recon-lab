# Contributing

PA Recon Lab welcomes reproducible improvements to photoacoustic reconstruction. Start with an issue explaining the problem, intended data/task, and evidence that would show an improvement. Small fixes and documentation corrections can go directly to a pull request.

## Set up

```bash
python -m venv .venv
# Activate your environment, then:
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/build_kaggle_notebook.py --check
```

Install the PyTorch build appropriate to your hardware. CI runs small generated fixtures on CPU; it does not download the research dataset or certify research performance.

## Good first contributions

- Improve instructions using a fresh installation and record the exact commands.
- Add a licensed phantom or dataset adapter with axis, units, and group-ID documentation.
- Reproduce an analytical reconstruction baseline from documented sensor/time metadata.
- Evaluate uncertainty calibration or report a failure case with a complete manifest.

## Research contributions

Include source/data rights, immutable configuration, dataset and split hashes, environment, seeds, runtime, and per-source metrics. Keep preprocessing and model selection inside training/validation. Use paired source/subject-level comparisons and report negative results. State the scope of each claim; results on synthetic samples do not establish measured-device or clinical performance.

Paired linear mixing assumes a common linear acoustic operator. Changes in medium, optical transport, detector response, and geometry require an appropriate forward simulation or measured data. Do not describe augmentations as new independent subjects.

## Pull requests

Keep one change focused. Include the trigger, resulting behavior, verification, and limitations. If package or pilot configuration changes, rebuild the generated Kaggle entry points with `python scripts/build_kaggle_notebook.py`. Do not hand-edit the generated notebook's embedded model. Tests should check behavior and independent invariants.

Keep datasets, weights, credentials, private measurements, and local paths out of commits. Large artifacts should be separately versioned with checksums and compatible licenses. Follow the [code of conduct](CODE_OF_CONDUCT.md).
