# Research blueprint: robust photoacoustic reconstruction under skin and acquisition shift

## Scope

This project starts with a constrained open-data problem: reconstruct a synthetic initial-pressure image from detector/time sinograms while measuring sensitivity to sensor gain, dropout, and noise. This dataset does not vary hardware or tissue parameters in a way that justifies broad robustness claims. The project does **not** identify viruses or foreign DNA from a transcutaneous scan.

## Why the original target needs a different measurement path

PA imaging uses pulsed optical excitation. Absorbed optical energy produces an initial pressure field, which propagates as ultrasound and is measured by transducers. A reconstruction model estimates the source distribution from those measurements. The signal's acoustic spectrum depends on source geometry, tissue, propagation, and instrumentation; it is not a universal pathogen-identity barcode.

Molecular specificity would require a validated contrast mechanism (for example, a target-binding probe with suitable optical absorption), a defined pathogen or nucleic-acid target, and paired ground-truth measurements. Published work describes molecular probes and viral imaging as a research opportunity, not a validated all-virus/all-DNA non-invasive diagnostic. Without that evidence, a model could learn confounders and produce convincing but unsupported labels.

## Proposed first paper and model

**Working title:** *Uncertainty-aware photoacoustic reconstruction from synthetic skin-phantom sinograms: a reproducible benchmark*.

**Candidate hypothesis (novelty unconfirmed):** Train a direct sinogram-to-IP baseline with detector gain, dropout, and additive-noise perturbations, then measure performance across prespecified corruption severities against the unaugmented baseline. Report uncertainty calibration and failure modes. The released 100-sample subset is development only; final claims require the full 60,000-sample release and independent real measurements. This is an experiment plan, not an established contribution.

**First implementation:** a residual U-Net maps the public synthetic detector/time sinogram to the corresponding initial-pressure (IP) field and predicts per-pixel aleatoric uncertainty. This is a learned inverse mapping for one released simulation setup; distribution shift and physical validity remain open. Compare with analytical reconstruction and compatible learned baselines. Do not advertise speedups or diagnostic capability without direct measurements against named baselines on specified hardware and data.

**Stress conditions:** test only synthetic sample-level generalization and prespecified detector dropout, gain, and noise sweeps on this release. The released simulator uses a 0.01 mm grid, 1500 m/s sound speed, 1000 kg/m³ density, and a fixed full-width line sensor at the top boundary; it does not establish robustness to acoustic speed, geometry, bandwidth, skin pigmentation, or real hardware. Those shifts require parameterized new simulations or paired measured data. Synthetic sample splits are not patient-held-out validation.

## Related evidence that constrains novelty

- [PA OmniNet (2025)](https://doi.org/10.1016/j.pacs.2025.100740) already targets reconstruction robustness across system configurations, using a small context set for reconstructed images. It is an important comparator, but its image input is not a drop-in raw-sinogram baseline.
- [Godefroy et al. (2021)](https://doi.org/10.1016/j.pacs.2020.100218) studied limited-view/bandwidth artifact correction and Monte Carlo dropout uncertainty, with simulation and experimental leaf data. Uncertainty prediction alone is not a novelty claim.
- [Hybrid Bayesian reconstruction and error prediction (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11424948/) further shows uncertainty-aware reconstruction is an active direction.
- [Synthetic-to-real domain adaptation (2025)](https://doi.org/10.1016/j.pacs.2025.100736) directly identifies simulation-to-experiment shift as a barrier in quantitative PAI.
- [Photoacoustic device fingerprints and model bias (2026)](https://www.nature.com/articles/s41598-026-53468-6) motivates device-held-out evaluation; ordinary random sample splits cannot support claims of hardware robustness.

## Public starting points to audit

- [PAID public-dataset catalogue](https://github.com/CbinHu/Photoacoustic-Imaging-Open-Datasets): a useful index, not itself a guarantee of availability, license compatibility, or ground-truth quality.
- [Cambridge dataset: data-driven quantitative PAI using tissue-mimicking phantoms](https://doi.org/10.17863/CAM.96644): provides phantom, blood-flow phantom, and mouse data plus model weights; verify the license, acquisition details, and intended reuse before selecting it.
- [PACT reconstruction benchmark preprint (2026)](https://arxiv.org/abs/2601.17165): proposes anatomically plausible synthetic breast data and task-based assessment; reports that image similarity can look good while clinically relevant lesions are missed. Treat as a comparator and avoid claiming novelty over it without a detailed literature review.
- [OADAT: experimental and synthetic clinical optoacoustic data](https://openreview.net/forum?id=BVi6MhKO0G): includes human forearm signals and synthetic data, plus benchmark code. The repository reports the dataset license as CC BY-NC 4.0 and its code as MIT; verify the authoritative dataset record and terms before training or redistributing weights. It is not a virus/DNA dataset.
- [PA OmniNet code and paper](https://github.com/olivierstam4/PA_OmniNet): an existing generalizable reconstruction approach that must be considered before proposing a new architecture.
- [Molecular imaging of viral pathogenesis and opportunities for the future](https://www.nature.com/articles/s44303-024-00056-w): supports the distinction between promising molecular imaging research and an established non-invasive virus test.

## Evaluation contract

1. Verify HDF5 integrity and reproduce the authors' reconstruction or an accepted analytical sinogram baseline before performance claims.
2. Lock sample-level train/validation/test assignments; separately obtain an external real-data set for any real-world claim.
3. Compare unaugmented and augmented training under identical held-out clean and corrupted inputs; include zero-output and analytical baselines.
4. Report NRMSE/SSIM, uncertainty calibration, per-corruption results, failure examples, and runtime/hardware. Use paired bootstrap intervals over test samples.
5. Add ablations for noise, dropout, and gain augmentation; do not equate these corruption tests with tissue/device domain generalization.
6. Release code, locked environment, configs, model card, data provenance/license, weights only where compatible, and exact commands.

## Virus/DNA research track (separate and later)

Before training a pathogen classifier, select one pathogen and tissue, specify the intended clinical decision, and identify a molecular target and probe with evidence for target binding, in-vivo safety, optical absorption, and detectability at the required depth. Obtain ethics and biosafety approvals, a protocol, and paired reference tests (e.g. validated molecular assay) across positives, negatives, and relevant confounders. A viable early study could assess whether PAI detects a probe-associated signal in controlled phantoms or ex-vivo samples; this is not equivalent to diagnosing infection through intact human skin. No dataset identified in this initial search supplies the paired transcutaneous PAI plus viral/DNA ground truth needed to train or validate the requested universal detector.

## Publication and release boundary

ResearchGate and ORCID are dissemination/profile services, not substitutes for peer review. The manuscript becomes publishable only after the study has been run, results are independently checked, data and code rights are cleared, limitations are reported, and authorship/ethics requirements are met. This blueprint is a proposed protocol, not a completed paper or validated medical model.

## Immediate gates

- Completed development gate: the Zenodo subset download/checksum and keys/shapes are recorded in the data audit.
- Establish a conventional reconstruction baseline from the released simulation setup before any paper claim.
- Complete the hosted comparison from `configs/pilot.json`; then reserve a new final holdout before research claims, because the existing development test set has been inspected.
- Treat the current U-Net as an engineering baseline; do a novelty review before naming a new method.
- Keep pathogen detection out of scope until a target-specific probe and paired reference dataset exist.
