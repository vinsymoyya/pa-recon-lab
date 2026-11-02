# Research roadmap

The ambition is robust, inspectable reconstruction under acquisition shift. The current U-Net and synthetic pilot are starting baselines. A new method earns a claim through comparisons, ablations, and independently held-out measurements.

## 0. Reproducibility foundation — in progress

- Shared package and checkpoint format for CLI and hosted GPU experiments.
- Verified downloads, source-group splits, train-only scaling, declared comparison arms, and run manifests.
- Executable checks for leakage prevention, checkpoint inference, and failed-download recovery.
- Complete and archive the first hosted GPU comparison before advertising a remote result.

## 1. Establish the inverse-problem baseline

Implement an accepted analytical reconstruction using documented geometry, time sampling, and preprocessing. Compare it with the unaugmented network before adding architecture complexity. Report runtime, peak memory, relative-unit errors, structural metrics, and vessel localization performance.

**Exit evidence:** reproducible point-source and vessel phantoms, an independently checked implementation, complete hardware/configuration information, and paired comparisons. The zero-output baseline remains a sanity check.

## 2. Build controlled simulation diversity

Use a documented acoustic solver such as [k-Wave](https://www.k-wave.org/documentation.php) to vary geometry, sound-speed fields, attenuation, and detector responses. Generate source anatomy independently of the acquisition configuration. Separate anatomy families and acquisition configurations between training and final evaluation. Validate numerical stability and grid convergence before scaling generation.

**Exit evidence:** source and solver versions, parameter ranges with citations, sensor/time metadata, parameterized seed manifests, convergence checks, and held-out configurations. Paired linear source mixing is a training augmentation under a fixed linear operator; it does not create new medium or device physics.

## 3. Investigate physics-conditioned reconstruction

Candidate: an iterative model that combines a learned image prior with residuals from the forward operator and its adjoint, conditioned on geometry and detector response metadata. Missing metadata needs an explicit estimation or uncertainty path. Learned recovery of invisible structures must be assessed using false-positive and missed-vessel rates.

**Exit evidence:** verified forward/adjoint consistency, comparison with analytical and learned baselines at matched compute, and ablations for operator conditioning, data consistency, and training diversity. This direction is not a novelty claim: [PA OmniNet](https://doi.org/10.1016/j.pacs.2025.100740) and existing model-based/learned reconstruction work constrain it.

## 4. Calibrate uncertainty and measure failure

Fit calibration only on validation data. Evaluate interval coverage, width, error-versus-coverage trade-offs, and behavior under unseen acquisition settings. Include failures where uncertainty remains low despite large reconstruction error. Account for spatial dependence by resampling sources/subjects rather than pixels.

**Exit evidence:** prespecified calibration protocol, external calibration assessment, and comparison with established uncertainty methods, including [Godefroy et al.](https://doi.org/10.1016/j.pacs.2020.100218).

## 5. Validate independently and publish

Use licensed measured data with subject/device separation. Treat algorithmic full-view backprojection as a reference reconstruction, not verified initial-pressure truth. Freeze the final holdout before tuning and retain a clear record of development-set inspection. Select a journal and report all comparisons, uncertainty, negative results, limitations, author contributions, data rights, and compute costs.

**Exit evidence:** independent measurements, sufficient sample size justified for the claims, reproducible baselines, archived code/weights/manifests, and a manuscript written from completed results. ResearchGate and ORCID support dissemination and identity; journal requirements and scientific validation determine publication readiness.

## Community growth

Make the first experiment easy to reproduce. Publish complete benchmark artifacts, document failures, and keep small contribution opportunities open. Invite feedback through issues and accept compatible third-party baselines. Track reproducible external runs and useful contributions alongside stars and forks; those counts cannot establish scientific quality.
