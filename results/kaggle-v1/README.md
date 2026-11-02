# Completed Kaggle pilot: notebook version 1

Actual downloaded metrics from version 1 of the [Kaggle notebook](https://www.kaggle.com/code/vinsymoyya/pa-recon-robust-synthetic-pilot); select version 1 in the notebook history. Newer runs may appear by default. Immutable checksums here identify the downloaded evidence and weights.

`manifest.json` records Tesla T4, CUDA, environment, dataset/source hashes, exact configuration, both completed arms, and `status=complete`. `artifact_sha256.json` covers original files including both weights. Weights are distributed through [Kaggle Models](https://www.kaggle.com/models/vinsymoyya/pa-recon-lab), not committed to Git. Split bytes, including CRLF, preserve the original provenance hash.

Each arm contains validation and four test-condition reports with aggregate, sample, and group metrics. `paired_comparisons.json` is a later analysis of unchanged reports. Reproduce:

```bash
python scripts/summarize_pilot.py --source runs/kaggle-output/pa-recon-results --output results/kaggle-v1
```

Augmentation lowers pooled severe RMSE from 0.107071 to 0.102824 (3.97%), while increasing clean MAE from 0.053400 to 0.054986 (2.97%). Paired group macro-RMSE severe delta: -0.004451, exploratory 95% interval [-0.006430,-0.002380]. Macro RMSE averages source RMSE values and is not pooled pixel RMSE.

Intervals use 10,000 paired resamples of ten source groups, seed 20261011, percentile endpoints, no multiplicity correction. All conditions use the same inspected development set, one corruption seed and one training seed per arm. Training random-number paths differ between arms. This comparison cannot isolate noise, gain, dropout, and mixup effects, or establish patient/scanner generalization.

The log places experiment completion around 1,021 seconds after process start, including setup/download/evaluation; this is not inference latency. No new physics simulation was generated. The 45-check suite was added after this version's training; subsequent notebook versions run it before training. See [model card](../../MODEL_CARD.md) and [pilot report](../../paper/PILOT_REPORT.md).
