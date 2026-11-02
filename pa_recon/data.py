"""HDF5 input/target pairs and explicit sample-group split handling."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class H5PairInfo:
    input_key: str
    target_key: str
    sample_count: int
    input_shape: tuple[int, ...]
    target_shape: tuple[int, ...]


def inspect_pair(path: str | Path, input_key: str, target_key: str, sample_axis: int = 0) -> H5PairInfo:
    """Inspect paired datasets without loading their contents into memory."""
    with h5py.File(path, "r") as handle:
        missing = [key for key in (input_key, target_key) if key not in handle]
        if missing:
            available = sorted(handle.keys())
            raise KeyError(f"Missing HDF5 key(s) {missing}. Top-level keys: {available}")
        x, y = handle[input_key], handle[target_key]
        if x.ndim < 3 or y.ndim < 3:
            raise ValueError(f"Expected batched image arrays; got {input_key}={x.shape}, {target_key}={y.shape}")
        axis_x = sample_axis % x.ndim
        axis_y = sample_axis % y.ndim
        if x.shape[axis_x] != y.shape[axis_y]:
            raise ValueError(f"Pair sample counts differ: {x.shape} vs {y.shape}")
        return H5PairInfo(input_key, target_key, int(x.shape[axis_x]), tuple(x.shape), tuple(y.shape))


def load_group_splits(path: str | Path, sample_count: int) -> dict[Split, list[int]]:
    """Read a CSV with sample_index,group_id,split and enforce group isolation."""
    grouped: dict[str, set[str]] = {}
    splits: dict[Split, list[int]] = {"train": [], "validation": [], "test": []}
    seen: set[int] = set()
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"sample_index", "group_id", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("Split CSV must contain columns: sample_index,group_id,split")
        for row in reader:
            index = int(row["sample_index"])
            group = row["group_id"].strip()
            split = row["split"].strip().lower()
            if not group or split not in splits:
                raise ValueError(f"Invalid group or split at sample {index}")
            if index < 0 or index >= sample_count or index in seen:
                raise ValueError(f"Invalid or duplicate sample_index {index}")
            seen.add(index)
            grouped.setdefault(group, set()).add(split)
            splits[split].append(index)  # type: ignore[index]
    if seen != set(range(sample_count)):
        missing = sorted(set(range(sample_count)) - seen)
        raise ValueError(f"Split CSV must map every sample exactly once; missing {len(missing)} samples")
    leaking = [group for group, labels in grouped.items() if len(labels) != 1]
    if leaking:
        raise ValueError(f"Volunteer/session leakage across splits for group(s): {leaking[:8]}")
    empty = [name for name, indices in splits.items() if not indices]
    if empty:
        raise ValueError(f"Each split must contain samples; empty: {empty}")
    return splits


def _as_chw(array: np.ndarray, key: str) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 2:
        return array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"Each sample in {key!r} must be HxW, CxHxW, or HxWxC; got {array.shape}")
    if array.shape[0] <= 4:
        return array
    if array.shape[-1] <= 4:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Cannot infer channel axis for {key!r} sample shape {array.shape}")


def _read_sample(dataset: h5py.Dataset, index: int, sample_axis: int) -> np.ndarray:
    axis = sample_axis % dataset.ndim
    selection = [slice(None)] * dataset.ndim
    selection[axis] = index
    return np.asarray(dataset[tuple(selection)])


class H5PairedImages(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """Read paired HDF5 arrays by sample index with a lazy per-process file handle."""

    def __init__(self, path: str | Path, input_key: str, target_key: str,
                 indices: list[int], input_scale: float, target_scale: float, sample_axis: int = 0):
        if not np.isfinite(input_scale) or not np.isfinite(target_scale) or input_scale <= 0 or target_scale <= 0:
            raise ValueError("input and target scales must be positive and fixed from training data")
        self.path = str(path)
        self.input_key = input_key
        self.target_key = target_key
        self.indices = list(indices)
        self.input_scale = float(input_scale)
        self.target_scale = float(target_scale)
        self.sample_axis = sample_axis
        self._handle: h5py.File | None = None

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample_index = self.indices[position]
        handle = self._file()
        x = _read_sample(handle[self.input_key], sample_index, self.sample_axis)
        y = _read_sample(handle[self.target_key], sample_index, self.sample_axis)
        x_chw = _as_chw(x, self.input_key)
        y_chw = _as_chw(y, self.target_key)
        if x_chw.shape[0] != y_chw.shape[0]:
            raise ValueError(f"Input/target channel mismatch at sample {sample_index}: {x_chw.shape} vs {y_chw.shape}")
        if not np.isfinite(x_chw).all() or not np.isfinite(y_chw).all():
            raise ValueError(f"Non-finite values at sample {sample_index}")
        return (torch.from_numpy(np.ascontiguousarray(x_chw / self.input_scale)),
                torch.from_numpy(np.ascontiguousarray(y_chw / self.target_scale)), sample_index)
