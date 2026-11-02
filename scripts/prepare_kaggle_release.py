"""Stage explicitly selected public data and model artifacts; never upload a checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/example_subset_100_samples.h5")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/publication")
    args = parser.parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Only completed results can be released")
    if hashlib.sha256(args.dataset.read_bytes()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Dataset does not match the training manifest")
    if args.output.exists():
        raise FileExistsError("Choose a new staging directory")
    data = args.output / "data"
    data.mkdir(parents=True)
    shutil.copyfile(args.dataset, data / args.dataset.name)
    shutil.copyfile(ROOT / "data/zenodo_readme.md", data / "ORIGINAL_README.md")
    attribution = (
        "# Synthetic photoacoustic skin phantom subset\n\n"
        "Unmodified 100-sample subset by Alagappan, G., Low, S., Xia, Y., Zhen, L. and Olivo, M. (2026), "
        "Synthetic Vascularized Digital Skin Phantoms for Photoacoustic Simulation and Reconstruction, v2.0.0. "
        "Source: https://doi.org/10.5281/zenodo.21785486. Original authors retain credit.\n\n"
        "License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/. No array changes. "
        "The mirror curator is not the dataset creator. See ORIGINAL_README.md for simulator details. "
        "Important audit correction: sinogram contains signed pressure values.\n\n"
        "HDF5: sinogram (100,256,1207), IP/mua/fluence (100,256,256), plus sample_id. "
        "No patient data. Relative units; fixed simulated acquisition. Use source IDs to avoid split leakage.\n\n"
        "Training: https://www.kaggle.com/code/vinsymoyya/pa-recon-robust-synthetic-pilot\n"
        "Code: https://github.com/vinsymoyya/pa-recon-lab\n"
        f"SHA256: {manifest['dataset_sha256']}\n"
    )
    (data / "README.md").write_text(attribution, encoding="utf-8")
    write_json(data / "dataset-metadata.json", {
        "id": "vinsymoyya/photoacoustic-skin-phantoms-100-source-mirror",
        "title": "Photoacoustic Skin Phantoms: 100 Source Samples",
        "subtitle": "Attributed Zenodo subset for reproducible sinogram to image training",
        "description": attribution, "licenses": [{"name": "CC-BY-4.0"}],
        "keywords": ["health", "computer vision", "image processing"],
        "resources": [{"path": args.dataset.name, "description": "Unmodified synthetic HDF5 source subset; signed sinograms and IP targets."}]})
    model = args.output / "model"
    model.mkdir()
    card = (ROOT / "MODEL_CARD.md").read_text(encoding="utf-8")
    write_json(model / "model-metadata.json", {
        "ownerSlug": "vinsymoyya", "slug": "pa-recon-lab", "title": "PA Recon Lab",
        "subtitle": "Open synthetic photoacoustic reconstruction pilot with reproducible stress evaluation",
        "isPrivate": False, "description": card,
        "provenanceSources": "https://doi.org/10.5281/zenodo.21785486"})
    for arm in ("unaugmented", "augmented"):
        destination = args.output / arm
        destination.mkdir()
        shutil.copyfile(args.results / arm / "model.pt", destination / "model.pt")
        for name in ("manifest.json", "scales.json", "splits.csv"):
            shutil.copyfile(args.results / name, destination / name)
        for path in (args.results / arm).glob("*.json"):
            shutil.copyfile(path, destination / path.name)
        shutil.copyfile(ROOT / "MODEL_CARD.md", destination / "MODEL_CARD.md")
        shutil.copyfile(ROOT / "LICENSE", destination / "LICENSE")
        hashes = "".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n"
                         for p in sorted(destination.iterdir()) if p.is_file())
        (destination / "SHA256SUMS").write_text(hashes, encoding="utf-8")
        write_json(destination / "model-instance-metadata.json", {
            "ownerSlug": "vinsymoyya", "modelSlug": "pa-recon-lab", "instanceSlug": arm,
            "framework": "pyTorch", "licenseName": "MIT", "fineTunable": True,
            "overview": f"{arm.capitalize()} width-16 residual U-Net; 50-epoch synthetic pilot. Uncalibrated uncertainty; no clinical claims.",
            "trainingData": ["https://doi.org/10.5281/zenodo.21785486"],
            "usage": "Install `pip install git+https://github.com/vinsymoyya/pa-recon-lab.git@v0.2.0`. "
                     "Download this variation and the attributed synthetic subset. Run:\n\n"
                     "```bash\npa-recon predict --file example_subset_100_samples.h5 --checkpoint model.pt "
                     "--sample-index 3 --output prediction.npz --device cpu\n```\n\n"
                     "Use `torch.load(..., weights_only=True)` for the checkpoint. No target is required at inference. "
                     "Read MODEL_CARD.md and compare both arms; augmentation trades clean MAE for lower stress RMSE."})
    print(f"Staged public artifacts at {args.output}")


if __name__ == "__main__":
    main()
