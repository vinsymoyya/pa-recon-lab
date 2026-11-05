"""Metadata audit and explicitly assumed delay-and-sum diagnostics for measured IPASC data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .acquisition import signal_summary, vary_acquisition
from .download import file_digest


def read_ipasc(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read a single-wavelength, single-frame IPASC record without axis guessing."""
    with h5py.File(path) as handle:
        dataset = handle["binary_time_series_data"]
        if dataset.ndim != 4 or dataset.shape[2:] != (1, 1):
            raise ValueError("Expected detector,time,1 wavelength,1 frame; select/export frames explicitly")
        raw = np.asarray(dataset[:, :, 0, 0], dtype=np.float64)
        meta = handle["meta_data"]
        frequency = float(np.asarray(meta["ad_sampling_rate"]).squeeze())
        if not np.isfinite(frequency) or frequency <= 0:
            raise ValueError("Sampling rate must be positive finite Hz")
        detectors = handle["meta_data_device/detectors"]
        positions = np.stack([np.asarray(detectors[key]["detector_position"]) for key in sorted(detectors)])
        if positions.shape != (raw.shape[0], 3) or not np.isfinite(positions).all() or not np.isfinite(raw).all():
            raise ValueError("Detector coordinates or signals are invalid")
        fov = np.asarray(handle["meta_data_device/general/field_of_view"]).tolist()
        pulse_energy = np.asarray(meta["pulse_energy"]).ravel().tolist() if "pulse_energy" in meta else None
        info = {"file_sha256": file_digest(path), "raw_shape": list(dataset.shape),
            "sample_rate_hz": frequency, "dt_s": 1/frequency,
            "wavelength_raw": np.asarray(meta["acquisition_wavelengths"]).tolist(),
            "field_of_view_m": fov, "pulse_energy_raw": pulse_energy,
            "pulse_energy_unit": "not recorded in this file; not used for cross-device amplitude comparison",
            "acquisition_metadata_keys": sorted(meta.keys()),
            "sound_speed_m_s": None, "time_origin_s": None,
            "missing_calibration": ["sound speed", "time origin/trigger offset", "ADC full-scale calibration", "detector impulse response"]}
    centered = positions - positions.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    info["geometry"] = {"classification": "nonlinear planar array" if singular[1] > singular[0]*1e-3 and singular[2] < singular[0]*1e-3 else "other",
        "coordinate_min_m": positions.min(axis=0).tolist(), "coordinate_max_m": positions.max(axis=0).tolist(),
        "centered_singular_values_m": singular.tolist(),
        "positions_sha256_float64": hashlib.sha256(positions.astype("<f8").tobytes()).hexdigest()}
    info["neural_transfer"] = {"status": "blocked", "reasons": [
        "Published checkpoint was trained on a top-boundary line array, not this curved device geometry",
        "Training dt is inferred as 2 ns; recorded device sampling differs",
        "No calibrated mapping of measured signal units to synthetic pressure normalization",
        "Provided FBP reconstruction and material labels are not initial-pressure ground truth"]}
    return raw, positions, info


def delay_and_sum(signal: np.ndarray, positions_m: np.ndarray, dt_s: float, sound_speed_m_s: float,
                  time_origin_s: float, fov_m: list, size: int = 96) -> np.ndarray:
    """2-D x-z diagnostic beamformer, not a quantitative inverse or full UBP.

    Point receivers, constant sound speed, linear time interpolation; omits
    detector directivity, impulse-response inversion and solid-angle weighting.
    Out-of-window samples contribute zero, without circular wrap.
    """
    signal = np.asarray(signal, dtype=np.float64)
    positions_m = np.asarray(positions_m, dtype=np.float64)
    if signal.ndim != 2 or positions_m.shape != (len(signal), 3) or not np.isfinite(signal).all() or not np.isfinite(positions_m).all():
        raise ValueError("Invalid detector signals/coordinates")
    if not np.isfinite([dt_s, sound_speed_m_s, time_origin_s]).all() or dt_s <= 0 or sound_speed_m_s <= 0 or size < 2:
        raise ValueError("Need finite timing, positive sampling/sound speed, and size >=2")
    if len(fov_m) != 6 or not np.isfinite(fov_m).all() or not (fov_m[0] < fov_m[1] and fov_m[4] < fov_m[5]):
        raise ValueError("Need finite nonempty x-z field of view")
    x, z = np.meshgrid(np.linspace(fov_m[0], fov_m[1], size), np.linspace(fov_m[4], fov_m[5], size))
    y = .5*(fov_m[2]+fov_m[3])
    result = np.zeros_like(x)
    times = np.arange(signal.shape[-1])
    for row, position in zip(signal, positions_m):
        distance = np.sqrt((x-position[0])**2 + (y-position[1])**2 + (z-position[2])**2)
        index = (distance / sound_speed_m_s - time_origin_s) / dt_s
        result += np.interp(index.ravel(), times, row, left=0, right=0).reshape(x.shape)
    return result / len(signal)


def run_audit(source: Path, output: Path, sound_speed: float, time_origin: float):
    if output.exists():
        raise FileExistsError("Use a new output directory to preserve evidence")
    paths = sorted(source.glob("*/raw/*_ipasc.hdf5"))
    if not paths:
        raise ValueError("No measured examples; run scripts/fetch_measured_examples.py first")
    output.mkdir(parents=True)
    manifest = {"source": "https://doi.org/10.5281/zenodo.14044853", "license": "CC-BY-4.0",
        "selection_manifest": json.loads((source / "manifest.json").read_text()),
        "status": "running", "evidence_boundary": "Measured raw data; post-acquisition/operator interventions are constructed scenarios",
        "beamformer_assumptions": {"sound_speed_m_s": sound_speed, "time_origin_s": time_origin,
            "calibration_status": "User-declared assumptions, NOT measured calibration", "image_plane": "x-z", "image_size": 96}}
    manifest["implementation_sha256"] = {p.name: file_digest(p) for p in
        [Path(__file__), Path(__file__).with_name("acquisition.py")]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    reports = []
    for path in paths:
        raw, positions, info = read_ipasc(path)
        device = path.parent.parent.name
        dt = info["dt_s"]
        baseline = raw - raw.mean(axis=-1, keepdims=True)
        nominal = delay_and_sum(baseline, positions, dt, sound_speed, time_origin, info["field_of_view_m"])
        cases = {"nominal": {}, "sparse_half": {"keep_every": 2}, "contiguous_outage": {"dead_fraction": .25},
                 "trigger_two_samples": {"delay_s": 2*dt}, "channel_jitter_half_sample": {"jitter_std_s": .5*dt},
                 "bandlimit_fs_eighth": {"lowpass_hz": 1/(8*dt)}, "clipping_3rms": {"clip_rms": 3}}
        maps = {"nominal": nominal}
        diagnostics = []
        for name, settings in cases.items():
            modified, trace = vary_acquisition(baseline, dt, settings, seed=20261105)
            image = delay_and_sum(modified, positions, dt, sound_speed, time_origin, info["field_of_view_m"])
            maps[name] = image
            denominator = np.linalg.norm(nominal)
            diagnostics.append({"case": name, "settings": settings,
                "relative_change_from_nominal_das": float(np.linalg.norm(image-nominal)/denominator) if denominator else None,
                "available_detectors": sum(trace["detector_available"]), "clipped_fraction": trace["clipped_fraction"]})
        for name, c, origin, geometry in (
            ("sound_speed_minus_5pct", .95*sound_speed, time_origin, positions),
            ("sound_speed_plus_5pct", 1.05*sound_speed, time_origin, positions),
            ("assumed_origin_plus_two_samples", sound_speed, time_origin+2*dt, positions),
            ("assumed_positions_jitter_0p1mm", sound_speed, time_origin,
                positions + np.random.default_rng(20261105).normal(0, 1e-4, positions.shape))):
            image = delay_and_sum(baseline, geometry, dt, c, origin, info["field_of_view_m"])
            maps[name] = image
            diagnostics.append({"case": name, "sound_speed_m_s": c, "time_origin_s": origin,
                "relative_change_from_nominal_das": float(np.linalg.norm(image-nominal)/np.linalg.norm(nominal)) if np.linalg.norm(nominal) else None,
                "evidence_kind": "Reconstruction assumption changed; measured trace unchanged"})
        np.savez_compressed(output / f"{device}_diagnostics.npz", **maps)
        info.update({"device": device, "source_member": path.relative_to(source).as_posix(),
            "raw_signal": signal_summary(raw, dt), "cases": diagnostics,
            "comparison_boundary": "Relative changes are sensitivity, NOT accuracy against true initial pressure"})
        reports.append(info)
        print(f"Measured {device}: {raw.shape}, {1/dt:g} Hz; diagnostics complete", flush=True)
    (output / "audit.json").write_text(json.dumps(reports, indent=2)+"\n")
    manifest["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/measured/openqpai/examples"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assumed-sound-speed-m-s", type=float, required=True)
    parser.add_argument("--assumed-time-origin-s", type=float, required=True)
    args = parser.parse_args()
    run_audit(args.source, args.output, args.assumed_sound_speed_m_s, args.assumed_time_origin_s)


if __name__ == "__main__":
    main()
