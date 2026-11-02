"""Run a declared comparison through the package's canonical CLI functions."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import torch

from .download import file_digest


def run_experiment(config_path: str, output: str, file_override: str | None = None,
                   require_cuda: bool = False) -> None:
    from .cli import build_parser

    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU was requested but is unavailable. Enable the lab's GPU accelerator.")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected experiment schema_version 1")
    arms = config["arms"]
    names = [arm["name"] for arm in arms]
    if not arms or len(names) != len(set(names)) or any(
        not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in name) for name in names
    ):
        raise ValueError("Arms need unique lowercase names containing letters, digits, underscores or hyphens")
    source = Path(file_override or config["data_file"]).resolve()
    root = Path(output).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a new output directory to preserve existing experiments: {root}")
    root.mkdir(parents=True, exist_ok=True)
    seed = config["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = build_parser()

    def invoke(arguments: list[str]) -> None:
        args = parser.parse_args(arguments)
        if hasattr(args, "num_threads"):
            torch.set_num_threads(args.num_threads)
        args.function(args)

    splits = root / "splits.csv"
    scales_path = root / "scales.json"
    invoke(["make-splits", "--file", str(source), "--output", str(splits), "--seed", str(seed)])
    invoke(["estimate-scales", "--file", str(source), "--splits", str(splits), "--output", str(scales_path),
            "--seed", str(seed)])
    scales = json.loads(scales_path.read_text(encoding="utf-8"))
    manifest = {
        "config": config, "dataset_sha256": file_digest(source), "dataset_bytes": source.stat().st_size,
        "split_sha256": file_digest(splits), "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "python": sys.version, "platform": platform.platform(),
        "torch": str(torch.__version__), "numpy": np.__version__,
        "package_source_sha256": hashlib.sha256(b"".join(
            p.name.encode() + p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py"))
        )).hexdigest(),
        "status": "running", "completed_arms": [],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for arm in arms:
        arm_root = root / arm["name"]
        arm_root.mkdir()
        checkpoint = arm_root / "model.pt"
        parameters = {**config["training"], **arm.get("training", {})}
        arguments = ["train", "--file", str(source), "--input-key", "sinogram", "--target-key", "IP",
                     "--splits", str(splits), "--input-scale", str(scales["input_scale"]),
                     "--target-scale", str(scales["target_scale"]), "--seed", str(seed),
                     "--device", device, "--output", str(checkpoint)]
        allowed = {"epochs", "batch_size", "width", "learning_rate", "noise_std", "detector_dropout",
                   "gain_jitter", "mixup_alpha", "num_threads"}
        if set(parameters) - allowed:
            raise ValueError(f"Unknown training parameters: {sorted(set(parameters) - allowed)}")
        for key, value in parameters.items():
            arguments.extend(["--" + key.replace("_", "-"), str(value)])
        invoke(arguments)
        for split in ("validation", "test"):
            invoke(["evaluate", "--file", str(source), "--splits", str(splits), "--checkpoint", str(checkpoint),
                    "--split", split, "--device", device, "--output", str(arm_root / f"{split}_clean.json")])
        for case in config.get("stress_cases", []):
            case_name = case["name"]
            if not case_name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in case_name):
                raise ValueError("Stress case names must be safe lowercase filenames")
            invoke(["evaluate", "--file", str(source), "--splits", str(splits), "--checkpoint", str(checkpoint),
                    "--device", device, "--output", str(arm_root / f"test_{case_name}.json"),
                    "--test-noise-std", str(case["noise_std"]), "--test-detector-dropout", str(case["detector_dropout"]),
                    "--test-gain-jitter", str(case["gain_jitter"]), "--corruption-seed", str(seed)])
        manifest["completed_arms"].append(arm["name"])
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Experiment complete: {root}", flush=True)
