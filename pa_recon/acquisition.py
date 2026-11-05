"""Explicit post-acquisition interventions in physical time units.

These manipulate recorded traces; they do not resimulate a changed medium,
transducer directivity, fluence, or sensor position.
"""

from __future__ import annotations

import math
import numpy as np


def vary_acquisition(signal: np.ndarray, dt_s: float, settings: dict, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Apply declared interventions to a signed detector x time array.

    Delay uses zero extension rather than circular wrap. Band limiting uses a
    zero-padded Gaussian magnitude response with its -3 dB point at cutoff_hz.
    The added filter is zero phase, not a measured detector impulse response.
    """
    source = np.asarray(signal, dtype=np.float64)
    if source.ndim != 2 or min(source.shape) < 2 or not np.isfinite(source).all():
        raise ValueError("Need a finite detector x time matrix with at least two rows and columns")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive and finite")
    allowed = {"delay_s", "jitter_std_s", "lowpass_hz", "keep_every", "dead_fraction", "clip_rms", "gain_slope", "noise_rms_fraction"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown acquisition settings: {sorted(set(settings) - allowed)}")
    for key, value in settings.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite numeric")
        if key not in {"delay_s", "gain_slope"} and value < 0:
            raise ValueError(f"{key} must be nonnegative")
    if "keep_every" in settings and (int(settings["keep_every"]) != settings["keep_every"] or settings["keep_every"] < 1):
        raise ValueError("keep_every must be a positive integer")
    if not 0 <= settings.get("dead_fraction", 0) < 1:
        raise ValueError("dead_fraction must lie in [0,1)")
    if abs(settings.get("gain_slope", 0)) > 1:
        raise ValueError("gain_slope must lie in [-1,1]")
    cutoff = settings.get("lowpass_hz")
    if cutoff is not None and not 0 < cutoff < .5 / dt_s:
        raise ValueError("lowpass_hz must lie strictly between zero and Nyquist")
    if "clip_rms" in settings and settings["clip_rms"] <= 0:
        raise ValueError("clip_rms must be positive")
    rng = np.random.default_rng(seed)
    detectors, samples = source.shape
    result = source.copy()
    if cutoff is not None:
        length = 1 << (2 * samples - 1).bit_length()
        frequencies = np.fft.rfftfreq(length, dt_s)
        response = np.exp(-.5 * np.log(2) * (frequencies / cutoff) ** 2)
        result = np.fft.irfft(np.fft.rfft(result, n=length, axis=-1) * response, n=length, axis=-1)[..., :samples]
    delays = np.full(detectors, settings.get("delay_s", 0.), dtype=np.float64)
    if settings.get("jitter_std_s", 0):
        delays += rng.normal(0, settings["jitter_std_s"], detectors)
    if np.any(delays):
        positions = np.arange(samples)
        result = np.stack([np.interp(positions - delay / dt_s, positions, row, left=0., right=0.)
                           for row, delay in zip(result, delays)])
    if settings.get("gain_slope", 0):
        result *= (1 + settings["gain_slope"] * np.linspace(-1, 1, detectors))[:, None]
    available = np.arange(detectors) % int(settings.get("keep_every", 1)) == 0
    dead_count = int(detectors * settings.get("dead_fraction", 0))
    if dead_count:
        start = (detectors - dead_count) // 2
        available[start:start + dead_count] = False
    result[~available] = 0
    source_rms = float(np.sqrt(np.mean(source ** 2)))
    if settings.get("noise_rms_fraction", 0):
        result += rng.normal(0, source_rms * settings["noise_rms_fraction"], source.shape)
    clipped_fraction = 0.
    if "clip_rms" in settings:
        limit = settings["clip_rms"] * source_rms
        clipped_fraction = float(np.mean(np.abs(result) > limit))
        result = np.clip(result, -limit, limit)
    return result.astype(np.float32), {"settings": settings, "seed": seed, "dt_s": dt_s,
        "nyquist_hz": .5 / dt_s, "detector_available": available.tolist(),
        "realized_delays_s": delays.tolist(), "source_rms": source_rms,
        "clipped_fraction": clipped_fraction,
        "evidence_kind": "post-acquisition intervention, not newly acquired or resimulated data"}


def signal_summary(signal: np.ndarray, dt_s: float | None = None) -> dict:
    """Describe signal health; flat extrema are flags, not proof of ADC clipping."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) < 2 or not np.isfinite(x).all():
        raise ValueError("Need finite detector x time data")
    rms = np.sqrt(np.mean(x*x, axis=-1))
    median_rms = float(np.median(rms))
    result = {"detectors": x.shape[0], "time_samples": x.shape[1], "rms": float(np.sqrt(np.mean(x*x))),
        "minimum": float(x.min()), "maximum": float(x.max()), "negative_fraction": float(np.mean(x < 0)),
        "exact_zero_detectors": int(np.sum(rms == 0)),
        "low_rms_detectors": int(np.sum(rms < median_rms * .01)) if median_rms else None,
        "channel_rms_q05_q50_q95": np.quantile(rms, [.05, .5, .95]).tolist(),
        "mean_channel_dc_absolute": float(np.mean(np.abs(x.mean(axis=-1)))),
        "centered_rms": float(np.sqrt(np.mean((x-x.mean(axis=-1, keepdims=True))**2))),
        "samples_equal_65535": int(np.sum(x == 65535)),
        "extrema_fraction": float(np.mean((x == x.min()) | (x == x.max()))),
        "interpretation": "Signal diagnostics only; low RMS and repeated extrema are not confirmed hardware faults"}
    if dt_s is not None:
        if not math.isfinite(dt_s) or dt_s <= 0:
            raise ValueError("dt_s must be positive and finite")
        centered = x - x.mean(axis=-1, keepdims=True)
        power = np.mean(np.abs(np.fft.rfft(centered, axis=-1)) ** 2, axis=0)
        frequencies = np.fft.rfftfreq(x.shape[-1], dt_s)
        total = power.sum()
        result.update({"dt_s": dt_s, "nyquist_hz": .5 / dt_s,
            "spectral_centroid_hz": float(np.sum(frequencies * power) / total) if total else None,
            "spectral_energy_95pct_hz": float(frequencies[np.searchsorted(np.cumsum(power), .95 * total)]) if total else None})
    return result


def reconstruction_failures(mean: np.ndarray, sigma: np.ndarray, target: np.ndarray) -> dict:
    """Depth and edge error diagnostics, not a vessel-segmentation endpoint."""
    mean, sigma, target = [np.asarray(v, dtype=np.float64) for v in (mean, sigma, target)]
    if mean.ndim != 2 or min(mean.shape) < 2 or mean.shape != target.shape or sigma.shape != target.shape:
        raise ValueError("Need matching depth x lateral arrays")
    if not all(np.isfinite(v).all() for v in (mean, sigma, target)) or np.any(sigma <= 0):
        raise ValueError("Need finite arrays and positive sigma")
    error = mean - target
    out = {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))),
           "outside_3sigma_fraction": float(np.mean(np.abs(error) > 3*sigma)), "depth_quartiles": []}
    for rows in np.array_split(np.arange(target.shape[0]), 4):
        if not len(rows):
            continue
        truth, pred = target[rows], mean[rows]
        out["depth_quartiles"].append({"row_start": int(rows[0]), "row_stop_exclusive": int(rows[-1])+1,
            "target_rms": float(np.sqrt(np.mean(truth**2))), "prediction_rms": float(np.sqrt(np.mean(pred**2))),
            "rmse": float(np.sqrt(np.mean((pred-truth)**2))),
            "outside_3sigma_fraction": float(np.mean(np.abs(pred-truth) > 3*sigma[rows]))})
    ref_gradient = np.concatenate([v.ravel() for v in np.gradient(target)])
    pred_gradient = np.concatenate([v.ravel() for v in np.gradient(mean)])
    denominator = np.linalg.norm(ref_gradient) * np.linalg.norm(pred_gradient)
    out["gradient_cosine_similarity"] = float(np.dot(ref_gradient, pred_gradient) / denominator) if denominator else None
    out["interpretation"] = "Depth and gradient diagnostics; no vessel labels, calibrated coverage, or clinical endpoint"
    return out
