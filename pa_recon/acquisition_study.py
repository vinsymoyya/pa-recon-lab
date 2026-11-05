"""Expose reconstruction failures under declared acquisition interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .acquisition import vary_acquisition, reconstruction_failures
from .data import load_group_splits
from .download import file_digest
from .model import SparseViewRecon


def run_study(config_path: str, file: str, splits: str, checkpoints: list[str], output: str, device: str = "cpu"):
    config = json.loads(Path(config_path).read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version 1")
    names = [case["name"] for case in config["cases"]]
    if not names or len(names) != len(set(names)) or any(not n or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in n) for n in names):
        raise ValueError("Need unique safe case names")
    dt = config["time_sampling"]["dt_s"]
    for case in config["cases"]:
        vary_acquisition(np.zeros((8, 16)), dt, case["settings"])
    seeds = config["corruption_seeds"]
    if not seeds or any(not isinstance(v, int) or v < 0 for v in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("Need unique nonnegative integer corruption seeds")
    root = Path(output)
    if root.exists():
        raise FileExistsError("Use a new output directory to preserve failure evidence")
    split_hash = file_digest(splits)
    models = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint["training"]["split_csv_sha256"] != split_hash:
            raise ValueError("Checkpoint/split provenance mismatch")
        data = checkpoint["data"]
        if data["input_key"] != "sinogram" or data["target_key"] != "IP" or data["sample_axis"] != 0:
            raise ValueError("This study expects the published sample-first sinogram/IP format")
        if any(not np.isfinite(data[key]) or data[key] <= 0 for key in ("input_scale", "target_scale")):
            raise ValueError("Checkpoint scales must be positive and finite")
        cfg = checkpoint["model"]
        model = SparseViewRecon(cfg["in_channels"], cfg["width"]).to(device).eval()
        model.load_state_dict(checkpoint["state_dict"])
        models.append((path, model, data, cfg))
    if not models:
        raise ValueError("At least one checkpoint is required")
    with h5py.File(file) as handle:
        indices = load_group_splits(splits, len(handle["sinogram"]))["test"]
        ids = np.asarray(handle["sample_id"])
    root.mkdir(parents=True)
    manifest = {"config": config, "dataset_sha256": file_digest(file), "split_sha256": split_hash,
        "checkpoints_sha256": [file_digest(v[0]) for v in models], "device": device,
        "torch": str(torch.__version__), "numpy": np.__version__, "status": "running",
        "scope": "Inspected synthetic development sources; not measured acquisition or external validation"}
    manifest["implementation_sha256"] = {p.name: file_digest(p) for p in
        [Path(__file__), Path(__file__).with_name("acquisition.py"), Path(__file__).with_name("model.py")]}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    with h5py.File(file) as handle, torch.no_grad():
        for model_index, (path, model, data, cfg) in enumerate(models):
            for case in config["cases"]:
                # The clean case has no random intervention, so do not duplicate it.
                realizations = seeds if case["settings"] else [seeds[0]]
                for seed in realizations:
                    for index in indices:
                        raw = np.asarray(handle["sinogram"][index])
                        target = np.asarray(handle["IP"][index], dtype=np.float32) / data["target_scale"]
                        source_id = str(ids[index])
                        source_seed = int.from_bytes(hashlib.sha256(f"{seed}:{source_id}".encode()).digest()[:8], "little")
                        transformed, trace = vary_acquisition(raw, dt, case["settings"], source_seed)
                        x = torch.from_numpy(transformed / data["input_scale"])[None, None].to(device)
                        mean, logvar = model(x, tuple(target.shape))
                        metrics = reconstruction_failures(mean[0, 0].cpu().numpy(), torch.exp(.5*logvar[0, 0]).cpu().numpy(), target)
                        zero_rmse = float(np.sqrt(np.mean(target**2)))
                        rows.append({"checkpoint_index": model_index, "case": case["name"], "corruption_seed": seed,
                            "sample_index": index, "source_id": source_id, **metrics,
                            "zero_rmse": zero_rmse, "rmse_worse_than_zero": metrics["rmse"] > zero_rmse,
                            "available_detectors": sum(trace["detector_available"]),
                            "intervention_clipped_fraction": trace["clipped_fraction"]})
                    print(f"Checkpoint {model_index}: {case['name']} seed {seed} complete", flush=True)
    summaries = []
    for model_index in range(len(models)):
        for case in names:
            subset = [r for r in rows if r["checkpoint_index"] == model_index and r["case"] == case]
            # Aggregate corruption repetitions within each source; sources remain the unit.
            by_source = [{key: float(np.mean([r[key] for r in subset if r["source_id"] == source]))
                          for key in ("mae", "rmse", "outside_3sigma_fraction")}
                         for source in sorted({r["source_id"] for r in subset})]
            summaries.append({"checkpoint_index": model_index, "case": case, "source_count": len(by_source),
                "macro_mae": float(np.mean([r["mae"] for r in by_source])),
                "macro_rmse": float(np.mean([r["rmse"] for r in by_source])),
                "macro_outside_3sigma_fraction": float(np.mean([r["outside_3sigma_fraction"] for r in by_source])),
                "worst_source_by_mean_rmse": max(sorted({r["source_id"] for r in subset}),
                    key=lambda source: np.mean([r["rmse"] for r in subset if r["source_id"] == source]))})
    (root / "failures.json").write_text(json.dumps({"summary": summaries, "per_source_realization": rows}, indent=2) + "\n")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/acquisition_failure_study.json")
    parser.add_argument("--file", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=2)
    args = parser.parse_args()
    if args.num_threads < 1:
        raise ValueError("num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    run_study(args.config, args.file, args.splits, args.checkpoint, args.output, args.device)


if __name__ == "__main__":
    main()
