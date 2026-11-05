"""Build the acquisition investigation notebook from the package and scripts."""

import argparse
import json
from pathlib import Path

from build_kaggle_notebook import render, ROOT


def content():
    training = json.loads(render()[ROOT / "kaggle_kernel/train.ipynb"])
    setup = "".join(training["cells"][1]["source"])
    extras = {name: (ROOT / name).read_text(encoding="utf-8") for name in [
        "scripts/fetch_measured_examples.py", "scripts/report_acquisition_failures.py", "configs/acquisition_failure_study.json"]}
    setup += f"\nextra_files = {extras!r}\n"
    setup += "for relative, value in extra_files.items():\n    path = project / relative\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(value, encoding='utf-8', newline='')\n"
    run = """import torch
if not torch.cuda.is_available():
    raise RuntimeError('Enable GPU; this investigation requires CUDA for the neural study.')
base = Path('/kaggle/working/acquisition-results')
if base.exists():
    raise FileExistsError('Choose a clean notebook session to preserve existing results.')
base.mkdir()
environment.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
def execute(arguments):
    subprocess.run([sys.executable, *arguments], cwd=project, env=environment, check=True)
execute(['-m', 'unittest', 'discover', '-s', 'tests', '-v'])
source = base / 'measured-source'
execute(['scripts/fetch_measured_examples.py', '--output', str(source)])
measured = base / 'measured-audit'
execute(['-m', 'pa_recon.measured', '--source', str(source), '--output', str(measured),
         '--assumed-sound-speed-m-s', '1500', '--assumed-time-origin-s', '0'])
inputs = Path('/kaggle/input')
models = list(inputs.rglob('model.pt'))
unaugmented = next(p for p in models if '/unaugmented/' in p.as_posix())
augmented = next(p for p in models if '/augmented/' in p.as_posix())
data_file = next(inputs.rglob('example_subset_100_samples.h5'))
synthetic = base / 'synthetic-failures'
execute(['-m', 'pa_recon.acquisition_study', '--file', str(data_file),
         '--splits', str(unaugmented.parent / 'splits.csv'), '--checkpoint', str(unaugmented),
         '--checkpoint', str(augmented), '--device', 'cuda', '--output', str(synthetic)])
report = base / 'evidence'
execute(['scripts/report_acquisition_failures.py', '--synthetic-run', str(synthetic),
         '--measured-run', str(measured), '--measured-source', str(source), '--output', str(report)])
from IPython.display import display, Image
display(Image(filename=str(report / 'measured_sensitivity.png')))
display(Image(filename=str(report / 'synthetic_failures.png')))
print('Measured traces and constructed interventions remain separate. No measured neural accuracy is claimed.')
"""
    cells = [{"cell_type": "markdown", "id": "overview", "metadata": {}, "source": [
        "# PA Recon: acquisition variations and failure cases\n",
        "Audit measured InVision/TROPUS phantom records, then evaluate unchanged synthetic-trained checkpoints on declared synthetic interventions.\n",
        "Measured data: Gröhl et al., https://doi.org/10.5281/zenodo.14044853, CC BY 4.0. Synthetic data: Alagappan et al., https://doi.org/10.5281/zenodo.21785486, CC BY 4.0. Code/weights: MIT.\n",
        "Sound speed 1500 m/s and time origin zero are assumptions, not measured calibration. DAS maps demonstrate sensitivity, not accuracy. Current neural transfer to these curved arrays is blocked.\n",
        "GPU and Internet required. Source-member ZIP CRC32 and SHA256 are recorded; full-archive MD5 is not verified.\n"]}]
    for identifier, code in (("source", setup), ("investigation", run)):
        cells.append({"cell_type": "code", "id": identifier, "metadata": {}, "execution_count": None,
                      "outputs": [], "source": code.splitlines(keepends=True)})
    return json.dumps({**training, "cells": cells}, indent=1)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "kaggle_acquisition/study.ipynb"
    value = content()
    if args.check:
        if not path.exists() or path.read_text(encoding="utf-8") != value:
            raise SystemExit("Rebuild acquisition notebook: python scripts/build_acquisition_notebook.py")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8", newline="\n")
    print("Acquisition notebook is current.")


if __name__ == "__main__":
    main()
