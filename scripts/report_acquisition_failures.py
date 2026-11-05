"""Archive measured and constructed acquisition evidence with scientific figures."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pa_recon.acquisition import vary_acquisition
from pa_recon.analysis import paired_comparison
from pa_recon.measured import delay_and_sum


def physics_probe():
    # Analytic Gaussian arrivals from a known center point, not a wave solver.
    angle = np.linspace(0, 2*np.pi, 96, endpoint=False)
    positions = np.stack([.04*np.cos(angle), np.zeros(96), .04*np.sin(angle)], axis=1)
    dt, c = 25e-9, 1500.
    time = np.arange(2200)*dt
    arrival = np.linalg.norm(positions, axis=1)/c
    traces = np.exp(-.5*((time[None, :]-arrival[:, None])/50e-9)**2)
    fov = [-.016, .016, 0, 0, -.016, .016]
    image = delay_and_sum(traces, positions, dt, c, 0, fov, 65)
    shifted, _ = vary_acquisition(traces, dt, {"delay_s": 2*dt})
    compensated = delay_and_sum(shifted, positions, dt, c, -2*dt, fov, 65)
    peak = [int(v) for v in np.unravel_index(np.argmax(image), image.shape)]
    relative_error = float(np.linalg.norm(image-compensated)/np.linalg.norm(image))
    if peak != [32, 32] or relative_error > 1e-6:
        raise ValueError("Analytic arrival probe did not focus/compensate as expected")
    return {"source": "Analytic center-point arrival experiment, not measured data or full-wave simulation",
        "expected_peak": [32, 32], "observed_peak": peak, "delay_samples": 2,
        "compensated_relative_image_error": relative_error,
        "boundary": "Checks interpolation, time sign, and geometry units only; not quantitative PAI inverse validity"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-run", type=Path, required=True)
    parser.add_argument("--measured-run", type=Path, required=True)
    parser.add_argument("--measured-source", type=Path, default=Path("data/measured/openqpai/examples"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise FileExistsError("Choose a new evidence archive")
    for run in (args.synthetic_run, args.measured_run):
        if json.loads((run/"manifest.json").read_text())["status"] != "complete":
            raise ValueError("Only completed runs can be archived")
    root.mkdir(parents=True)
    for kind, run in (("synthetic", args.synthetic_run), ("measured", args.measured_run)):
        (root/kind).mkdir()
        for path in run.glob("*.json"):
            shutil.copyfile(path, root/kind/path.name)
    data = json.loads((args.synthetic_run/"failures.json").read_text())
    manifest = json.loads((args.synthetic_run/"manifest.json").read_text())
    rows, summary = data["per_source_realization"], data["summary"]
    comparisons = {}
    for case in manifest["config"]["cases"]:
        reports = []
        for model in (0, 1):
            subset = [r for r in rows if r["checkpoint_index"] == model and r["case"] == case["name"]]
            sources = sorted({r["source_id"] for r in subset})
            reports.append({"evaluation_split": "test", "split_csv_sha256": manifest["split_sha256"],
                "input_scale": 200.3928680419922, "target_scale": 366.551239013672,
                "test_corruption": case, "per_group_metrics": [
                    {"group_id": source, "rmse": float(np.mean([r["rmse"] for r in subset if r["source_id"] == source]))}
                    for source in sources]})
        comparisons[case["name"]] = paired_comparison(*reports, "rmse", seed=20261105)
    (root/"paired_comparisons.json").write_text(json.dumps(comparisons, indent=2)+"\n")
    (root/"physics_probe.json").write_text(json.dumps(physics_probe(), indent=2)+"\n")
    cases = [c["name"] for c in manifest["config"]["cases"]]
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for model, label, color in ((0, "Unaugmented", "#3b6ea8"), (1, "Augmented", "#b7532c")):
        values = {s["case"]: s for s in summary if s["checkpoint_index"] == model}
        for axis, field in zip(axes, ("macro_rmse", "macro_outside_3sigma_fraction")):
            axis.bar(np.arange(len(cases))+(model-.5)*.35, [values[c][field] for c in cases], width=.35, label=label, color=color)
    for axis in axes:
        axis.set_xticks(np.arange(len(cases)), [c.replace("_", " ") for c in cases], rotation=55, ha="right", fontsize=9)
        axis.legend(); axis.grid(axis="y", alpha=.2); axis.set_axisbelow(True)
    axes[0].set_ylabel("Source-macro normalized RMSE")
    axes[1].set_ylabel("Mean fraction outside predicted +/-3 sigma")
    figure.suptitle("Constructed interventions on 10 synthetic development sources; sigma is uncalibrated")
    figure.savefig(root/"synthetic_failures.png", dpi=180)
    plt.close(figure)
    audit = json.loads((args.measured_run/"audit.json").read_text())
    figure, axes = plt.subplots(len(audit), 4, figsize=(16, 4*len(audit)), constrained_layout=True, squeeze=False)
    for axes_row, report in zip(axes, audit):
        device = report["device"]
        path = args.measured_source/report["source_member"]
        reference = np.load(path.parent.parent/"recon"/path.name.replace("_ipasc.hdf5", ".npy"), allow_pickle=False)
        with np.load(args.measured_run/f"{device}_diagnostics.npz") as maps:
            nominal, changed = maps["nominal"], maps["sound_speed_plus_5pct"]
        panels = [(reference, f"{device}: supplied FBP\nnot IP ground truth"),
                  (nominal, "Diagnostic delay-and-sum\nc=1500 m/s, t0=0 assumed"),
                  (changed, "Same measured traces\nassumed c +5%"),
                  (changed-nominal, "Change from diagnostic nominal\nnot reconstruction accuracy")]
        for axis, (values, title) in zip(axes_row, panels):
            limit = max(float(np.quantile(np.abs(values), .99)), 1e-12)
            artist = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Image column");axis.set_ylabel("Image row")
            figure.colorbar(artist, ax=axis, fraction=.046)
    figure.suptitle("Measured OpenQPAIData phantom P.5.6.2, 750 nm | Independent panel scales; FBP/DAS registration unverified")
    figure.savefig(root/"measured_sensitivity.png", dpi=180)
    plt.close(figure)
    print(f"Archived acquisition failure evidence: {root}")


if __name__ == "__main__":
    main()
