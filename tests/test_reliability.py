"""37 additional independent reliability contracts; 45 in the complete suite."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch

from pa_recon.analysis import paired_comparison
from pa_recon.cli import _corrupt_measurements
from pa_recon.data import H5PairedImages, inspect_pair, load_group_splits
from pa_recon.experiment import run_experiment
from pa_recon.model import SparseViewRecon, gaussian_l1_loss
from test_contracts import invoke


class FileContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.file = self.root / "data.h5"
        with h5py.File(self.file, "w") as h:
            h["sinogram"] = np.full((12, 16, 24), -6, dtype=np.float32)
            h["IP"] = np.full((12, 16, 16), 8, dtype=np.float32)
        self.splits = self.root / "split.csv"
        invoke("make-splits", "--file", self.file, "--output", self.splits)

    def tearDown(self):
        self.directory.cleanup()

    def dataset(self, **kwargs):
        return H5PairedImages(self.file, "sinogram", "IP", [0], **{
            "input_scale": 2, "target_scale": 4, **kwargs})

    def replace(self, key, value):
        with h5py.File(self.file, "a") as h:
            del h[key]
            h[key] = value

    def edit_rows(self, change):
        rows = list(csv.DictReader(io.StringIO(self.splits.read_text())))
        change(rows)
        with self.splits.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("sample_index", "group_id", "split"))
            writer.writeheader()
            writer.writerows(rows)

    def test_missing_hdf5_key(self):
        with self.assertRaisesRegex(KeyError, "Missing"):
            inspect_pair(self.file, "absent", "IP")

    def test_unequal_pair_counts(self):
        self.replace("IP", np.zeros((11, 16, 16)))
        with self.assertRaisesRegex(ValueError, "counts differ"):
            inspect_pair(self.file, "sinogram", "IP")

    def test_unbatched_arrays_rejected(self):
        self.replace("IP", np.zeros((16, 16)))
        with self.assertRaisesRegex(ValueError, "batched"):
            inspect_pair(self.file, "sinogram", "IP")

    def test_signed_input_and_independent_scales(self):
        d = self.dataset()
        try:
            x, y, index = d[0]
            self.assertTrue(torch.equal(x, torch.full((1, 16, 24), -3.)))
            self.assertTrue(torch.equal(y, torch.full((1, 16, 16), 2.)))
            self.assertEqual(index, 0)
        finally:
            d.close()

    def test_last_sample_axis(self):
        self.replace("sinogram", np.full((16, 24, 12), -6))
        self.replace("IP", np.full((16, 16, 12), 8))
        d = self.dataset(sample_axis=-1)
        try:
            self.assertEqual(inspect_pair(self.file, "sinogram", "IP", -1).sample_count, 12)
            self.assertEqual(tuple(d[0][0].shape), (1, 16, 24))
            self.assertEqual(float(d[0][0].mean()), -3.)
        finally:
            d.close()

    def test_channel_mismatch(self):
        self.replace("IP", np.zeros((12, 2, 16, 16)))
        d = self.dataset()
        try:
            with self.assertRaisesRegex(ValueError, "channel mismatch"):
                d[0]
        finally:
            d.close()

    def test_nonfinite_sample(self):
        with h5py.File(self.file, "a") as h:
            h["IP"][0, 0, 0] = np.nan
        d = self.dataset()
        try:
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                d[0]
        finally:
            d.close()

    def test_invalid_normalization_scale(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "scales"):
                self.dataset(input_scale=value)

    def test_duplicate_sample_index(self):
        self.edit_rows(lambda rows: rows.append(rows[0].copy()))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_group_splits(self.splits, 12)

    def test_missing_sample_assignment(self):
        self.edit_rows(lambda rows: rows.pop())
        with self.assertRaisesRegex(ValueError, "every sample"):
            load_group_splits(self.splits, 12)

    def test_unknown_split_label(self):
        self.edit_rows(lambda rows: rows[0].update(split="holdout_typo"))
        with self.assertRaisesRegex(ValueError, "Invalid group or split"):
            load_group_splits(self.splits, 12)

    def test_empty_split(self):
        self.edit_rows(lambda rows: [r.update(split="train") for r in rows])
        with self.assertRaisesRegex(ValueError, "empty"):
            load_group_splits(self.splits, 12)


class NumericalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_corruption_is_identity(self):
        x = torch.randn(2, 1, 16, 24)
        self.assertTrue(torch.equal(x, _corrupt_measurements(x, 0, 0, 0)))

    def test_dropout_removes_whole_detector_rows(self):
        torch.manual_seed(42)
        x = _corrupt_measurements(torch.ones(8, 1, 16, 24), 0, .5, 0)
        self.assertEqual(set(x.unique().tolist()), {0., 1.})
        self.assertTrue(torch.equal(x, x[..., :1].expand_as(x)))

    def test_corruption_reproducible_from_seed(self):
        x = torch.ones(4, 1, 16, 24)
        torch.manual_seed(12)
        first = _corrupt_measurements(x, .1, .2, .3)
        torch.manual_seed(12)
        self.assertTrue(torch.equal(first, _corrupt_measurements(x, .1, .2, .3)))
        torch.manual_seed(13)
        self.assertFalse(torch.equal(first, _corrupt_measurements(x, .1, .2, .3)))

    def test_gain_is_bounded_and_constant_over_time(self):
        x = _corrupt_measurements(torch.ones(4, 1, 16, 24), 0, 0, .2)
        self.assertGreaterEqual(x.min().item(), .8)
        self.assertLessEqual(x.max().item(), 1.2)
        self.assertTrue(torch.equal(x, x[..., :1].expand_as(x)))

    def test_model_restores_unpadded_dimensions(self):
        with torch.no_grad():
            mean, _ = SparseViewRecon(width=2).eval()(torch.zeros(1, 1, 17, 19))
        self.assertEqual(tuple(mean.shape), (1, 1, 17, 19))

    def test_model_preserves_batch_and_channels(self):
        with torch.no_grad():
            mean, variance = SparseViewRecon(in_channels=2, width=2).eval()(torch.zeros(3, 2, 16, 24), (16, 16))
        self.assertEqual(tuple(mean.shape), (3, 2, 16, 16))
        self.assertEqual(mean.shape, variance.shape)

    def test_log_variance_clamped_at_both_limits(self):
        model = SparseViewRecon(width=2).eval()
        with torch.no_grad():
            model.head.weight.zero_()
            for bias, expected in ((-100, -8), (100, 4)):
                model.head.bias.fill_(bias)
                _, logvar = model(torch.zeros(1, 1, 16, 16))
                self.assertTrue(torch.equal(logvar, torch.full_like(logvar, expected)))

    def test_finite_nonzero_training_gradients(self):
        model = SparseViewRecon(width=2)
        mean, logvar = model(torch.randn(2, 1, 16, 24), (16, 16))
        gaussian_l1_loss(mean, logvar, torch.ones_like(mean)).backward()
        gradients = [p.grad for p in model.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_loss_matches_hand_calculation(self):
        # Errors 1 and 3, variance 4: NLL=.5*(5/4+log(4)); L1=.1*2.
        value = gaussian_l1_loss(torch.tensor([1., 3.]), torch.log(torch.tensor([4., 4.])), torch.zeros(2))
        self.assertAlmostEqual(value.item(), .5 * (1.25 + np.log(4)) + .2, places=6)

    def test_perfect_unit_variance_prediction_zero_loss(self):
        self.assertEqual(gaussian_l1_loss(torch.ones(2), torch.zeros(2), torch.ones(2)).item(), 0)


class PreflightContracts(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.config = {"schema_version": 1, "seed": 1, "data_file": "not-needed.h5",
                       "training": {"epochs": 1}, "arms": [{"name": "base"}], "stress_cases": []}

    def tearDown(self):
        self.directory.cleanup()

    def run_invalid(self, pattern):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config))
        with self.assertRaisesRegex(ValueError, pattern):
            run_experiment(str(path), str(self.root / "output"))
        self.assertFalse((self.root / "output").exists())

    def test_unknown_training_key_fails_before_writes(self):
        self.config["training"]["misspelled_rate"] = .1
        self.run_invalid("Unknown training")

    def test_duplicate_stress_names_rejected(self):
        self.config["stress_cases"] = [{"name": "noise"}, {"name": "noise"}]
        self.run_invalid("Stress")

    def test_clean_stress_name_cannot_overwrite_clean_metrics(self):
        self.config["stress_cases"] = [{"name": "clean"}]
        self.run_invalid("Stress")

    def test_stress_path_traversal_rejected(self):
        self.config["stress_cases"] = [{"name": "../outside"}]
        self.run_invalid("Stress")

    def test_required_gpu_failure_creates_no_output(self):
        with mock.patch("torch.cuda.is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                run_experiment("not-needed.json", str(self.root / "output"), require_cuda=True)
        self.assertFalse((self.root / "output").exists())


class PairedStatisticsContracts(unittest.TestCase):
    def reports(self):
        report = {"evaluation_split": "test", "split_csv_sha256": "fixed", "input_scale": 2,
                  "target_scale": 3, "test_corruption": {"seed": 42},
                  "per_group_metrics": [{"group_id": str(i), "mae": float(i)} for i in range(3)]}
        return report, copy.deepcopy(report)

    def test_paired_groups_must_match(self):
        a, b = self.reports()
        b["per_group_metrics"].pop()
        with self.assertRaisesRegex(ValueError, "same"):
            paired_comparison(a, b, "mae")

    def test_paired_delta_matches_independent_calculation(self):
        a, b = self.reports()
        b["per_group_metrics"] = [{"group_id": "2", "mae": 0}, {"group_id": "0", "mae": 1}, {"group_id": "1", "mae": 0}]
        self.assertAlmostEqual(paired_comparison(a, b, "mae")["mean_paired_delta"], -2/3)

    def test_constant_improvement_interval(self):
        a, b = self.reports()
        for row in b["per_group_metrics"]:
            row["mae"] -= .5
        result = paired_comparison(a, b, "mae")
        self.assertEqual(result["paired_bootstrap_95ci"], [-.5, -.5])

    def test_bootstrap_seed_is_reproducible(self):
        a, b = self.reports()
        b["per_group_metrics"][0]["mae"] = 4
        self.assertEqual(paired_comparison(a, b, "mae", seed=7), paired_comparison(a, b, "mae", seed=7))

    def test_nonfinite_group_metrics_rejected(self):
        a, b = self.reports()
        b["per_group_metrics"][0]["mae"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            paired_comparison(a, b, "mae")

    def test_duplicate_groups_rejected(self):
        a, b = self.reports()
        b["per_group_metrics"].append(b["per_group_metrics"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            paired_comparison(a, b, "mae")


class InvalidInferenceContracts(unittest.TestCase):
    def test_nan_evaluation_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            invoke("evaluate", "--file", "absent", "--splits", "absent", "--checkpoint", "absent",
                   "--output", "absent", "--test-noise-std", "nan")

    def test_infinite_evaluation_corruption_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            invoke("evaluate", "--file", "absent", "--splits", "absent", "--checkpoint", "absent",
                   "--output", "absent", "--test-gain-jitter", "inf")

    def test_negative_prediction_index_rejected(self):
        with self.assertRaisesRegex(ValueError, "sample-index"):
            invoke("predict", "--file", "absent", "--checkpoint", "absent", "--output", "absent", "--sample-index", -1)

    def test_invalid_checkpoint_normalization_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.pt"
            torch.save({"model": {}, "data": {"input_scale": 0, "target_scale": 1}}, path)
            with self.assertRaisesRegex(ValueError, "scale"):
                invoke("predict", "--file", "absent", "--checkpoint", path, "--output", "absent", "--sample-index", 0)
