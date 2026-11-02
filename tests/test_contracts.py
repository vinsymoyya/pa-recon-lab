"""Small independent contracts; no research-data download or performance claim."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from pa_recon.cli import build_parser
from pa_recon.data import load_group_splits
from pa_recon.download import download_subset
from pa_recon.experiment import run_experiment
from pa_recon.model import SparseViewRecon


def invoke(*arguments):
    args = build_parser().parse_args(list(map(str, arguments)))
    with contextlib.redirect_stdout(io.StringIO()):
        args.function(args)


class ResearchContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.file = self.root / "pairs.h5"
        rng = np.random.default_rng(104)
        with h5py.File(self.file, "w") as handle:
            x = rng.normal(size=(12, 16, 24)).astype(np.float32)
            handle["sinogram"] = x
            handle["IP"] = np.abs(x[:, :, :16])
            handle["sample_id"] = np.arange(12)
        self.splits = self.root / "splits.csv"
        torch.set_num_threads(2)

    def tearDown(self):
        self.directory.cleanup()

    def make_split(self):
        invoke("make-splits", "--file", self.file, "--output", self.splits)

    def test_repeated_source_ids_are_never_split(self):
        with h5py.File(self.file, "a") as handle:
            del handle["sample_id"]
            handle["sample_id"] = np.repeat(np.arange(6), 2)
        invoke("make-splits", "--file", self.file, "--output", self.splits,
               "--train-fraction", 0.5, "--validation-fraction", 0.25)
        load_group_splits(self.splits, 12)
        with self.splits.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        groups = {}
        for row in rows:
            groups.setdefault(row["group_id"], set()).add(row["split"])
        self.assertEqual({len(v) for v in groups.values()}, {1})
        first = self.splits.read_bytes()
        invoke("make-splits", "--file", self.file, "--output", self.splits,
               "--train-fraction", 0.5, "--validation-fraction", 0.25)
        self.assertEqual(self.splits.read_bytes(), first)

    def test_split_loader_rejects_group_leakage(self):
        self.make_split()
        rows = list(csv.DictReader(io.StringIO(self.splits.read_text())))
        for row in rows:
            row["group_id"] = "same-source"
        with self.splits.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_index", "group_id", "split"))
            writer.writeheader(); writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, "leakage"):
            load_group_splits(self.splits, 12)

    def test_normalization_ignores_validation_and_test_values(self):
        self.make_split()
        output = self.root / "scales.json"
        invoke("estimate-scales", "--file", self.file, "--splits", self.splits, "--output", output)
        before = json.loads(output.read_text())
        groups = load_group_splits(self.splits, 12)
        with h5py.File(self.file, "a") as handle:
            for i in groups["validation"] + groups["test"]:
                handle["sinogram"][i] = 1e9
                handle["IP"][i] = 1e9
        invoke("estimate-scales", "--file", self.file, "--splits", self.splits, "--output", output)
        self.assertEqual(before, json.loads(output.read_text()))

    def test_explicit_output_size_wins_over_padding_crop(self):
        model = SparseViewRecon(width=2).eval()
        with torch.no_grad():
            x = torch.zeros(1, 1, 17, 19)
            for output_size in ((24, 24), (16, 20)):
                mean, logvar = model(x, output_size)
                self.assertEqual(mean.shape[-2:], output_size)
                self.assertTrue(torch.isfinite(mean).all())
                self.assertEqual(mean.shape, logvar.shape)

    def test_shared_experiment_and_input_only_checkpoint_inference(self):
        config = {"schema_version": 1, "data_file": str(self.file), "seed": 12,
                  "training": {"epochs": 1, "width": 2, "batch_size": 2, "num_threads": 2},
                  "arms": [{"name": "baseline", "training": {"mixup_alpha": 0}},
                           {"name": "mixed", "training": {"mixup_alpha": 0.4}}],
                  "stress_cases": [{"name": "noise", "noise_std": 0.01, "detector_dropout": 0.1, "gain_jitter": 0.1}]}
        config_path = self.root / "config.json"
        config_path.write_text(json.dumps(config))
        output = self.root / "experiment"
        with contextlib.redirect_stdout(io.StringIO()):
            run_experiment(str(config_path), str(output))
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed_arms"], ["baseline", "mixed"])
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        checkpoint = output / "mixed/model.pt"
        metrics = json.loads((output / "mixed/test_clean.json").read_text())
        self.assertEqual(metrics["evaluation_split"], "test")
        self.assertEqual(metrics["test_samples"], 2)
        val = json.loads((output / "mixed/validation_clean.json").read_text())
        self.assertEqual(val["validation_samples"], 1)
        # An inference file deliberately has no target key at all.
        input_only = self.root / "input_only.h5"
        with h5py.File(input_only, "w") as handle:
            handle["sinogram"] = np.zeros((1, 16, 24), dtype=np.float32)
        prediction = self.root / "new_directory/prediction.npz"
        invoke("predict", "--file", input_only, "--checkpoint", checkpoint,
               "--sample-index", 0, "--output", prediction, "--device", "cpu")
        with np.load(prediction) as result:
            self.assertEqual(result["reconstruction"].shape, (1, 16, 16))
            self.assertTrue(np.isfinite(result["predicted_sigma"]).all())
        # Even a semantically equivalent CSV with a changed hash needs deliberate review.
        changed_split = self.root / "changed.csv"
        changed_split.write_bytes((output / "splits.csv").read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            invoke("evaluate", "--file", self.file, "--splits", changed_split,
                   "--checkpoint", checkpoint, "--output", self.root / "bad.json")
        with self.assertRaises(FileExistsError):
            run_experiment(str(config_path), str(output))


class ArchiveContracts(unittest.TestCase):
    def test_archived_split_preserves_original_provenance(self):
        archive = Path(__file__).resolve().parents[1] / "results/development"
        expected = json.loads((archive / "scales.json").read_text())["split_csv_sha256"]
        actual = hashlib.sha256((archive / "splits.csv").read_bytes()).hexdigest()
        provenance = json.loads((archive / "checkpoint_provenance.json").read_text())
        self.assertEqual(actual, expected)
        self.assertEqual(actual, provenance["training"]["split_csv_sha256"])


class DownloadContracts(unittest.TestCase):
    def test_bad_download_never_becomes_the_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.h5"
            with mock.patch("urllib.request.urlopen", return_value=io.BytesIO(b"not the dataset")):
                with self.assertRaisesRegex(ValueError, "published MD5"):
                    download_subset(path)
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".h5.part").exists())

    def test_existing_wrong_file_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subset.h5"
            path.write_bytes(b"user-owned")
            with self.assertRaisesRegex(ValueError, "Existing file"):
                download_subset(path)
            self.assertEqual(path.read_bytes(), b"user-owned")


if __name__ == "__main__":
    unittest.main()
