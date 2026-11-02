# Remote training pilot

This public notebook downloads the 100-sample Zenodo synthetic subset, checks its published MD5, and compares unaugmented and augmented training using `configs/pilot.json`. It requires a CUDA GPU, selects each checkpoint by validation loss, and evaluates clean and fixed corrupted inputs. GPU absence causes an error instead of an unintended CPU training job.

The notebook and portable script are generated from `pa_recon/` with `scripts/build_kaggle_notebook.py`. They embed the same package code and call its CLI. Checkpoints can therefore be loaded by the documented local `pa-recon predict` command.

## Launch

1. Install Kaggle in the local project environment: `python -m pip install kaggle`.
2. On Windows, run `powershell -File scripts/kaggle-login.ps1`. The browser OAuth flow stores credentials outside the repository. If automatic browser opening fails, open the URL displayed in the terminal and enter its verification code in that terminal. Do not paste it into chat or source files.
3. Run `powershell -File scripts/submit-kaggle.ps1 -Owner YOUR_KAGGLE_USERNAME`. Use the actual Kaggle username, which may differ from your GitHub username.

On other systems, use `kaggle auth login`, set the notebook ID to `your-kaggle-username/pa-recon-robust-synthetic-pilot` in `kernel-metadata.json`, and run `kaggle kernels push -p kaggle_kernel --timeout 3600`. The Windows helper uses a temporary `KAGGLE_CONFIG_DIR`; use that same directory for subsequent status/output commands.

## Retrieve the run

Use `kaggle kernels status your-kaggle-username/pa-recon-robust-synthetic-pilot` and `kaggle kernels output your-kaggle-username/pa-recon-robust-synthetic-pilot -p runs/kaggle-output`. Output under `pa-recon-results/` includes source/config/environment manifests, split CSV, normalization scales, both checkpoints, and evaluation JSON. Run paths are unique and existing experiments are preserved.

After editing the package/configuration, rebuild with `python scripts/build_kaggle_notebook.py`. CI checks that these generated files match their sources.

The kernel is public with GPU and internet enabled. Version 1 completed on a Tesla T4; its original metrics are archived in `results/kaggle-v1/`. Version 2 adds the 45-check suite before training. It depends on Kaggle allowing outbound download from Zenodo. Its output remains a development artifact; the 100-sample experiment is not publication-grade.
