"""Archive a completed pilot and produce paired exploratory comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from pa_recon.analysis import paired_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results/kaggle-v1")
    args = parser.parse_args()
    source, output = Path(args.source), Path(args.output)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete" or manifest["completed_arms"] != ["unaugmented", "augmented"]:
        raise ValueError("Need a completed two-arm pilot")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "splits.csv", "scales.json"):
        shutil.copyfile(source / name, output / name)
    if hashlib.sha256((output / "splits.csv").read_bytes()).hexdigest() != manifest["split_sha256"]:
        raise ValueError("Split hash differs from run manifest")
    comparisons = {}
    for condition in ("clean", "mild", "moderate", "severe"):
        reports = [json.loads((source / arm / f"test_{condition}.json").read_text())
                   for arm in manifest["completed_arms"]]
        comparisons[condition] = [paired_comparison(*reports, metric) for metric in
                                 ("model_mae_normalized", "model_rmse_normalized")]
    for arm in manifest["completed_arms"]:
        (output / arm).mkdir(exist_ok=True)
        for path in (source / arm).glob("*.json"):
            shutil.copyfile(path, output / arm / path.name)
    (output / "paired_comparisons.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    hashes = {path.relative_to(source).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(source.rglob("*")) if path.is_file()}
    (output / "artifact_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    main()
