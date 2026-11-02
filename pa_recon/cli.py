"""Command-line tools for synthetic sinogram-to-image training and evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader

from .data import H5PairInfo, H5PairedImages, _as_chw, _read_sample, inspect_pair, load_group_splits
from .model import SparseViewRecon, gaussian_l1_loss


def download_command(args: argparse.Namespace) -> None:
    from .download import download_subset
    download_subset(args.output)


def run_command(args: argparse.Namespace) -> None:
    from .experiment import run_experiment
    run_experiment(args.config, args.output, args.file, args.require_cuda)


def _print_info(info: H5PairInfo, path: str, sample_axis: int) -> None:
    print(json.dumps({"file": str(path), "input_key": info.input_key,
                      "target_key": info.target_key, "sample_count": info.sample_count,
                      "input_shape": info.input_shape, "target_shape": info.target_shape,
                      "sample_axis": sample_axis}, indent=2))


def inspect_command(args: argparse.Namespace) -> None:
    info = inspect_pair(args.file, args.input_key, args.target_key, args.sample_axis)
    _print_info(info, args.file, args.sample_axis)
    with h5py.File(args.file, "r") as handle:
        for key in (args.input_key, args.target_key):
            sample = _read_sample(handle[key], 0, args.sample_axis)
            finite = sample[np.isfinite(sample)]
            print(f"{key}: sample0 shape={sample.shape} p01={np.percentile(finite, 1):.6g} "
                  f"p50={np.percentile(finite, 50):.6g} p99={np.percentile(finite, 99):.6g}")


def make_splits_command(args: argparse.Namespace) -> None:
    """Create deterministic splits without separating repeated source IDs."""
    if (not math.isfinite(args.train_fraction) or not math.isfinite(args.validation_fraction)
            or min(args.train_fraction, args.validation_fraction) <= 0
            or args.train_fraction + args.validation_fraction >= 1):
        raise ValueError("Train/validation fractions must be positive and sum to less than one")
    info = inspect_pair(args.file, args.input_key, args.target_key, args.sample_axis)
    with h5py.File(args.file, "r") as handle:
        ids = np.asarray(handle[args.id_key]) if args.id_key in handle else np.arange(info.sample_count)
    if ids.shape != (info.sample_count,):
        raise ValueError(f"ID dataset must have shape ({info.sample_count},); got {ids.shape}")
    id_strings = [str(v.item() if hasattr(v, "item") else v) for v in ids]
    unique_ids = list(dict.fromkeys(id_strings))
    order = np.random.default_rng(args.seed).permutation(len(unique_ids))
    n_train = int(len(unique_ids) * args.train_fraction)
    n_val = int(len(unique_ids) * args.validation_fraction)
    if min(n_train, n_val, len(unique_ids) - n_train - n_val) < 1:
        raise ValueError("Fractions must leave at least one source group in each split")
    group_labels = {group: "test" for group in unique_ids}
    group_labels.update({unique_ids[i]: "train" for i in order[:n_train]})
    group_labels.update({unique_ids[i]: "validation" for i in order[n_train:n_train + n_val]})
    labels = [group_labels[group] for group in id_strings]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("sample_index", "group_id", "split"))
        for index, (sample_id, split) in enumerate(zip(ids, labels)):
            writer.writerow((index, str(sample_id.item() if hasattr(sample_id, "item") else sample_id), split))
    print(f"Wrote {info.sample_count} sample-level splits to {output}; seed={args.seed}")


def estimate_scales_command(args: argparse.Namespace) -> None:
    """Estimate independent normalization scales from a bounded train-only sample."""
    if args.max_samples < 1 or args.values_per_sample < 1 or not 0 < args.quantile <= 1:
        raise ValueError("max-samples and values-per-sample must be positive; quantile must be in (0, 1]")
    info = inspect_pair(args.file, args.input_key, args.target_key, args.sample_axis)
    splits = load_group_splits(args.splits, info.sample_count)
    rng = np.random.default_rng(args.seed)
    indices = list(rng.permutation(splits["train"])[:args.max_samples])
    values: dict[str, list[np.ndarray]] = {"input": [], "target": []}
    with h5py.File(args.file, "r") as handle:
        for index in indices:
            for name, key in (("input", args.input_key), ("target", args.target_key)):
                sample = np.asarray(_read_sample(handle[key], index, args.sample_axis), dtype=np.float32).ravel()
                sample = np.abs(sample[np.isfinite(sample)])
                if sample.size:
                    take = min(args.values_per_sample, sample.size)
                    values[name].append(sample[rng.choice(sample.size, take, replace=False)])
    scales = {name: float(np.quantile(np.concatenate(parts), args.quantile)) for name, parts in values.items() if parts}
    if len(scales) != 2 or any(not np.isfinite(v) or v <= 0 for v in scales.values()):
        raise ValueError("Could not estimate positive finite scales from the training split")
    result = {"input_scale": scales["input"], "target_scale": scales["target"],
              "quantile": args.quantile, "train_samples_used": len(indices),
              "values_per_sample": args.values_per_sample, "seed": args.seed,
              "split_csv_sha256": hashlib.sha256(Path(args.splits).read_bytes()).hexdigest()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


def _corrupt_measurements(x: torch.Tensor, noise_std: float, dropout_probability: float,
                         gain_jitter: float) -> torch.Tensor:
    if gain_jitter:
        gain = 1 + torch.empty((x.shape[0], 1, x.shape[-2], 1), device=x.device).uniform_(-gain_jitter, gain_jitter)
        x = x * gain
    if dropout_probability:
        keep = (torch.rand((x.shape[0], 1, x.shape[-2], 1), device=x.device) >= dropout_probability)
        x = x * keep
    if noise_std:
        x = x + torch.randn_like(x) * noise_std
    return x


def _epoch(model: SparseViewRecon, loader: DataLoader, optimizer: torch.optim.Optimizer | None,
           device: torch.device, augmentation: tuple[float, float, float] = (0.0, 0.0, 0.0),
           mixup_alpha: float = 0.0) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    for x, y, _sample_indices in loader:
        x, y = x.to(device), y.to(device)
        if training:
            noise_std, dropout_probability, gain_jitter = augmentation
            if mixup_alpha > 0 and x.shape[0] > 1:
                permutation = torch.randperm(x.shape[0], device=device)
                beta = torch.distributions.Beta(mixup_alpha, mixup_alpha)
                weights = beta.sample((x.shape[0],)).to(device=device, dtype=x.dtype).view(-1, 1, 1, 1)
                x = weights * x + (1 - weights) * x[permutation]
                y = weights * y + (1 - weights) * y[permutation]
            x = _corrupt_measurements(x, noise_std, dropout_probability, gain_jitter)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            mean, logvar = model(x, tuple(y.shape[-2:]))
            loss = gaussian_l1_loss(mean, logvar, y)
            if training:
                loss.backward()
                optimizer.step()
        total += float(loss.detach()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


def train_command(args: argparse.Namespace) -> None:
    values = (args.noise_std, args.gain_jitter, args.mixup_alpha, args.learning_rate, args.detector_dropout)
    if (not all(math.isfinite(v) for v in values) or min(args.noise_std, args.gain_jitter, args.mixup_alpha) < 0
            or not 0 <= args.detector_dropout < 1 or args.learning_rate <= 0):
        raise ValueError("Use finite nonnegative augmentation, dropout in [0, 1), and positive learning rate")
    if min(args.epochs, args.batch_size, args.width, args.num_threads) < 1:
        raise ValueError("epochs, batch-size, width and num-threads must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    info = inspect_pair(args.file, args.input_key, args.target_key, args.sample_axis)
    splits = load_group_splits(args.splits, info.sample_count)
    print(f"Split sizes: { {name: len(values) for name, values in splits.items()} }")

    datasets = {name: H5PairedImages(args.file, args.input_key, args.target_key,
                                     indices, args.input_scale, args.target_scale, args.sample_axis)
                for name, indices in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=(name == "train"), num_workers=0)
               for name, dataset in datasets.items()}
    sample_x, sample_y, _ = datasets["train"][0]
    if sample_x.shape[0] != sample_y.shape[0]:
        raise ValueError("Input and target channel counts must match")
    model = SparseViewRecon(in_channels=sample_x.shape[0], width=args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best_validation = float("inf")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss = _epoch(model, loaders["train"], optimizer, device,
                            (args.noise_std, args.detector_dropout, args.gain_jitter), args.mixup_alpha)
        validation_loss = _epoch(model, loaders["validation"], None, device)
        if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
            raise FloatingPointError("Non-finite training/validation loss; no checkpoint saved for this epoch")
        print(f"epoch={epoch:03d} train_nll_l1={train_loss:.6f} validation_nll_l1={validation_loss:.6f} "
              f"seconds={time.perf_counter() - epoch_start:.2f}", flush=True)
        if validation_loss < best_validation:
            best_validation = validation_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "model": {"name": "SparseViewRecon", "in_channels": sample_x.shape[0], "width": args.width,
                          "output_size": list(sample_y.shape[-2:])},
                "data": {"input_key": args.input_key, "target_key": args.target_key,
                         "input_scale": args.input_scale, "target_scale": args.target_scale,
                         "sample_axis": args.sample_axis},
                "training": {"seed": args.seed, "epoch": epoch,
                             "validation_loss": validation_loss,
                             "batch_size": args.batch_size,
                             "learning_rate": args.learning_rate,
                             "augmentation": {"noise_std_normalized": args.noise_std,
                                             "detector_dropout": args.detector_dropout,
                                              "detector_gain_jitter": args.gain_jitter,
                                              "linear_superposition_mixup_alpha": args.mixup_alpha},
                             "max_epochs": args.epochs,
                             "cpu_threads": args.num_threads,
                             "torch_version": str(torch.__version__),
                             "split_csv_sha256": hashlib.sha256(Path(args.splits).read_bytes()).hexdigest()},
                "claim_boundary": "Synthetic detector-time sinogram to initial-pressure image; simulation-specific research baseline.",
            }
            temporary = output.with_name(output.name + ".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(output)
    print(f"Best validation loss: {best_validation:.6f}; checkpoint: {output}")
    for dataset in datasets.values():
        dataset.close()


def evaluate_command(args: argparse.Namespace) -> None:
    if (not all(math.isfinite(v) for v in (args.test_noise_std, args.test_gain_jitter, args.test_detector_dropout))
            or args.test_noise_std < 0 or args.test_gain_jitter < 0 or not 0 <= args.test_detector_dropout < 1):
        raise ValueError("Use finite nonnegative test noise/gain jitter and detector dropout in [0, 1)")
    torch.manual_seed(args.corruption_seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    split_hash = hashlib.sha256(Path(args.splits).read_bytes()).hexdigest()
    expected_hash = checkpoint.get("training", {}).get("split_csv_sha256")
    if expected_hash is not None and expected_hash != split_hash:
        raise ValueError("Evaluation split CSV differs from the checkpoint's training split manifest")
    data_cfg = checkpoint["data"]
    info = inspect_pair(args.file, data_cfg["input_key"], data_cfg["target_key"], data_cfg["sample_axis"])
    splits = load_group_splits(args.splits, info.sample_count)
    dataset = H5PairedImages(args.file, data_cfg["input_key"], data_cfg["target_key"],
                             splits[args.split], data_cfg["input_scale"], data_cfg["target_scale"], data_cfg["sample_axis"])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model_cfg = checkpoint["model"]
    model = SparseViewRecon(model_cfg["in_channels"], model_cfg["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    sample_groups: dict[int, str] = {}
    with Path(args.splits).open("r", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            sample_groups[int(row["sample_index"])] = row["group_id"].strip()
    accum = {key: 0.0 for key in ("model_mae", "zero_mae", "model_rmse_sq", "zero_rmse_sq", "target_sq", "target_count", "coverage_1sigma", "coverage_count", "uncertainty")}
    per_sample: list[dict[str, float | int | str]] = []
    with torch.no_grad():
        for x, y, sample_indices in loader:
            x, y = x.to(device), y.to(device)
            x = _corrupt_measurements(x, args.test_noise_std, args.test_detector_dropout, args.test_gain_jitter)
            mean, logvar = model(x, tuple(y.shape[-2:]))
            error = mean - y
            zero_error = y
            sigma = torch.exp(0.5 * logvar)
            count = y.numel()
            accum["model_mae"] += error.abs().sum().item()
            accum["zero_mae"] += zero_error.abs().sum().item()
            accum["model_rmse_sq"] += error.square().sum().item()
            accum["zero_rmse_sq"] += zero_error.square().sum().item()
            accum["target_sq"] += y.square().sum().item()
            accum["target_count"] += count
            accum["coverage_1sigma"] += ((error.abs() <= sigma).float()).sum().item()
            accum["coverage_count"] += count
            accum["uncertainty"] += sigma.sum().item()
            for item in range(x.shape[0]):
                item_error = error[item]
                item_zero_error = y[item]
                item_target = y[item]
                index = int(sample_indices[item])
                per_sample.append({
                    "sample_index": index,
                    "group_id": sample_groups[index],
                    "model_mae_normalized": item_error.abs().mean().item(),
                    "model_rmse_normalized": item_error.square().mean().sqrt().item(),
                    "zero_baseline_mae_normalized": item_zero_error.abs().mean().item(),
                    "zero_baseline_rmse_normalized": item_zero_error.square().mean().sqrt().item(),
                    "target_rms_normalized": item_target.square().mean().sqrt().item(),
                    "one_sigma_coverage": (item_error.abs() <= sigma[item]).float().mean().item(),
                    "mean_predicted_sigma_normalized": sigma[item].mean().item(),
                })
    n = accum["target_count"]
    result = {
        f"{args.split}_samples": len(dataset),
        "evaluation_split": args.split,
        "split_csv_sha256": split_hash,
        "model_mae_normalized": accum["model_mae"] / n,
        "zero_baseline_mae_normalized": accum["zero_mae"] / n,
        "model_rmse_normalized": (accum["model_rmse_sq"] / n) ** 0.5,
        "zero_baseline_rmse_normalized": (accum["zero_rmse_sq"] / n) ** 0.5,
        "target_rms_normalized": (accum["target_sq"] / n) ** 0.5,
        "predicted_1sigma_coverage": accum["coverage_1sigma"] / accum["coverage_count"],
        "mean_predicted_sigma_normalized": accum["uncertainty"] / n,
        "input_scale": data_cfg["input_scale"], "target_scale": data_cfg["target_scale"],
        "test_corruption": {"noise_std_normalized": args.test_noise_std,
                            "detector_dropout": args.test_detector_dropout,
                            "detector_gain_jitter": args.test_gain_jitter,
                            "seed": args.corruption_seed},
        "reference": "synthetic initial pressure (IP) from digital-skin phantom dataset",
    }
    result["model_mae_target_units"] = result["model_mae_normalized"] * data_cfg["target_scale"]
    result["zero_baseline_mae_target_units"] = result["zero_baseline_mae_normalized"] * data_cfg["target_scale"]
    groups = sorted({str(row["group_id"]) for row in per_sample})
    group_metrics = []
    for group in groups:
        rows = [row for row in per_sample if row["group_id"] == group]
        group_metrics.append({
            "group_id": group,
            "samples": len(rows),
            "model_mae_normalized": float(np.mean([row["model_mae_normalized"] for row in rows])),
            "model_rmse_normalized": float(np.mean([row["model_rmse_normalized"] for row in rows])),
            "zero_baseline_mae_normalized": float(np.mean([row["zero_baseline_mae_normalized"] for row in rows])),
            "zero_baseline_rmse_normalized": float(np.mean([row["zero_baseline_rmse_normalized"] for row in rows])),
            "one_sigma_coverage": float(np.mean([row["one_sigma_coverage"] for row in rows])),
            "mean_predicted_sigma_normalized": float(np.mean([row["mean_predicted_sigma_normalized"] for row in rows])),
        })
    if len(group_metrics) >= 2:
        rng = np.random.default_rng(20261011)
        for metric in ("model_mae_normalized", "model_rmse_normalized", "zero_baseline_mae_normalized", "zero_baseline_rmse_normalized"):
            values = np.asarray([row[metric] for row in group_metrics])
            draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            result[f"group_macro_{metric}"] = float(values.mean())
            result[f"group_macro_{metric}_bootstrap_95ci"] = [float(v) for v in np.quantile(draws, [0.025, 0.975])]
    result["group_count"] = len(group_metrics)
    result["per_group_metrics"] = group_metrics
    result["per_sample_metrics"] = per_sample
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    dataset.close()


def predict_command(args: argparse.Namespace) -> None:
    if args.sample_index < 0:
        raise ValueError("sample-index must be nonnegative")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg, data_cfg = checkpoint["model"], checkpoint["data"]
    if any(not math.isfinite(data_cfg[key]) or data_cfg[key] <= 0 for key in ("input_scale", "target_scale")):
        raise ValueError("Checkpoint normalization scales must be positive and finite")
    with h5py.File(args.file, "r") as handle:
        if data_cfg["input_key"] not in handle:
            raise KeyError(f"Input HDF5 key missing: {data_cfg['input_key']}")
        raw_input = _read_sample(handle[data_cfg["input_key"]], args.sample_index, data_cfg["sample_axis"])
    x_chw = _as_chw(raw_input, data_cfg["input_key"])
    if not np.isfinite(x_chw).all():
        raise ValueError(f"Non-finite values at sample {args.sample_index}")
    x = torch.from_numpy(np.ascontiguousarray(x_chw / data_cfg["input_scale"]))
    model = SparseViewRecon(model_cfg["in_channels"], model_cfg["width"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        mean, logvar = model(x.unsqueeze(0).to(device), tuple(model_cfg["output_size"]))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output,
                        sinogram=x.numpy() * data_cfg["input_scale"],
                        reconstruction=(mean[0].cpu().numpy() * data_cfg["target_scale"]),
                        predicted_sigma=(torch.exp(0.5 * logvar[0]).cpu().numpy() * data_cfg["target_scale"]),
                        sample_index=args.sample_index,
                        reference_type="synthetic IP phantom reference; not clinical ground truth")
    print(f"Wrote reconstruction and predicted uncertainty to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pa-recon", description="Research tools for sinogram-to-image PAI reconstruction")
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download-data", help="Download and checksum-verify the public 100-sample subset")
    download_parser.add_argument("--output", default="data/example_subset_100_samples.h5")
    download_parser.set_defaults(function=download_command)
    run_parser = sub.add_parser("run", help="Run a configured comparison with reproducibility manifests")
    run_parser.add_argument("--config", default="configs/pilot.json")
    run_parser.add_argument("--file", help="Override the dataset location declared in the config")
    run_parser.add_argument("--output", default="runs/pilot")
    run_parser.add_argument("--require-cuda", action="store_true")
    run_parser.set_defaults(function=run_command)
    split_parser = sub.add_parser("make-splits", help="Create deterministic sample-level synthetic-data splits")
    split_parser.add_argument("--file", required=True)
    split_parser.add_argument("--input-key", default="sinogram")
    split_parser.add_argument("--target-key", default="IP")
    split_parser.add_argument("--id-key", default="sample_id")
    split_parser.add_argument("--sample-axis", type=int, default=0)
    split_parser.add_argument("--seed", type=int, default=20261011)
    split_parser.add_argument("--train-fraction", type=float, default=0.8)
    split_parser.add_argument("--validation-fraction", type=float, default=0.1)
    split_parser.add_argument("--output", required=True)
    split_parser.set_defaults(function=make_splits_command)
    scale_parser = sub.add_parser("estimate-scales", help="Estimate normalization from training samples only")
    scale_parser.add_argument("--file", required=True)
    scale_parser.add_argument("--input-key", default="sinogram")
    scale_parser.add_argument("--target-key", default="IP")
    scale_parser.add_argument("--sample-axis", type=int, default=0)
    scale_parser.add_argument("--splits", required=True)
    scale_parser.add_argument("--max-samples", type=int, default=512)
    scale_parser.add_argument("--values-per-sample", type=int, default=4096)
    scale_parser.add_argument("--quantile", type=float, default=0.999)
    scale_parser.add_argument("--seed", type=int, default=20261011)
    scale_parser.add_argument("--output", required=True)
    scale_parser.set_defaults(function=estimate_scales_command)
    for name, function in (("inspect", inspect_command), ("train", train_command),
                           ("evaluate", evaluate_command), ("predict", predict_command)):
        child = sub.add_parser(name)
        child.add_argument("--file", required=True, help="Photoacoustic HDF5 dataset file")
        child.set_defaults(function=function)
        if name in ("train", "evaluate", "predict"):
            child.add_argument("--num-threads", type=int, default=4,
                               help="PyTorch CPU thread limit; set to 1 or 2 for small machines")
        if name in ("inspect", "train"):
            child.add_argument("--input-key", required=True, help="Input dataset key, e.g. sinogram")
            child.add_argument("--target-key", required=True, help="Reference dataset key, e.g. IP")
            child.add_argument("--sample-axis", type=int, default=0)
        if name == "train":
            child.add_argument("--splits", required=True, help="CSV: sample_index,group_id,split; synthetic samples use sample_id groups")
            child.add_argument("--input-scale", type=float, required=True, help="Fixed sinogram scale from training data")
            child.add_argument("--target-scale", type=float, required=True, help="Fixed IP scale from training data")
            child.add_argument("--output", required=True, help="Checkpoint path (.pt)")
            child.add_argument("--epochs", type=int, default=50)
            child.add_argument("--batch-size", type=int, default=2)
            child.add_argument("--width", type=int, default=16)
            child.add_argument("--learning-rate", type=float, default=1e-3)
            child.add_argument("--noise-std", type=float, default=0.01, help="Normalized Gaussian sinogram noise augmentation")
            child.add_argument("--detector-dropout", type=float, default=0.05, help="Probability of dropping a detector per sample")
            child.add_argument("--gain-jitter", type=float, default=0.1, help="Uniform per-detector gain perturbation")
            child.add_argument("--mixup-alpha", type=float, default=0.0,
                               help="Beta-distributed pair mixing for linear-acoustics superposition (0 disables)")
            child.add_argument("--seed", type=int, default=20261011)
            child.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        elif name == "evaluate":
            child.add_argument("--splits", required=True)
            child.add_argument("--split", choices=("validation", "test"), default="test")
            child.add_argument("--checkpoint", required=True)
            child.add_argument("--output", required=True, help="Write aggregate held-out metrics to JSON")
            child.add_argument("--batch-size", type=int, default=8)
            child.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            child.add_argument("--test-noise-std", type=float, default=0.0)
            child.add_argument("--test-detector-dropout", type=float, default=0.0)
            child.add_argument("--test-gain-jitter", type=float, default=0.0)
            child.add_argument("--corruption-seed", type=int, default=20261011)
        elif name == "predict":
            child.add_argument("--checkpoint", required=True)
            child.add_argument("--sample-index", type=int, required=True)
            child.add_argument("--output", required=True, help="Compressed NPZ with reconstruction and uncertainty")
            child.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "num_threads"):
        if args.num_threads < 1:
            raise ValueError("num-threads must be positive")
        torch.set_num_threads(args.num_threads)
    args.function(args)


if __name__ == "__main__":
    main()
