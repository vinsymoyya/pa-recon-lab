# Reliability evidence

Run `python scripts/check_reliability.py --expected 45`. It fails on missing checks, failures, errors, or skips, and writes `runs/verification.json`. Ordinary discovery works too: `python -m unittest discover -s tests -v`.

These are 45 named test methods; related boundary values use subtests. Tests use small HDF5 fixtures and a real two-arm train/evaluate/predict experiment, with no downloads or GPU requirement. The public notebook runs the same suite before training. CI covers Linux/Python 3.10 and Windows/Python 3.12. Generated-notebook freshness and package build are checked separately, outside the 45-method count.

Eight new negative checks initially failed: nonfinite evaluation noise/gain, invalid checkpoint scales, negative prediction index, unknown training options, duplicate stress names, a reserved clean-name collision, and unsafe stress paths. Production validation fixes make them pass. The original failure log stays locally in `runs/checks-before.txt` because it contains workstation paths.

## Meaning of a pass

The suite establishes specific executable software contracts. It does not establish scanner generalization, calibrated uncertainty, patient safety, pathogen specificity, novelty, or publication acceptance. Scientific evidence is reported separately in the [pilot report](../paper/PILOT_REPORT.md), with ten inspected development test sources and explicit limitations.

## Inventory

| # | Check | Source |
|---:|---|---|
| 1 | repeated source ids are never split | [ResearchContracts](../tests/test_contracts.py#L52) |
| 2 | split loader rejects group leakage | [ResearchContracts](../tests/test_contracts.py#L70) |
| 3 | normalization ignores validation and test values | [ResearchContracts](../tests/test_contracts.py#L81) |
| 4 | explicit output size wins over padding crop | [ResearchContracts](../tests/test_contracts.py#L94) |
| 5 | shared experiment and input only checkpoint inference | [ResearchContracts](../tests/test_contracts.py#L104) |
| 6 | archived split preserves original provenance | [ArchiveContracts](../tests/test_contracts.py#L146) |
| 7 | bad download never becomes the dataset | [DownloadContracts](../tests/test_contracts.py#L156) |
| 8 | existing wrong file is preserved | [DownloadContracts](../tests/test_contracts.py#L165) |
| 9 | missing hdf5 key | [FileContracts](../tests/test_reliability.py#L55) |
| 10 | unequal pair counts | [FileContracts](../tests/test_reliability.py#L59) |
| 11 | unbatched arrays rejected | [FileContracts](../tests/test_reliability.py#L64) |
| 12 | signed input and independent scales | [FileContracts](../tests/test_reliability.py#L69) |
| 13 | last sample axis | [FileContracts](../tests/test_reliability.py#L79) |
| 14 | channel mismatch | [FileContracts](../tests/test_reliability.py#L90) |
| 15 | nonfinite sample | [FileContracts](../tests/test_reliability.py#L99) |
| 16 | invalid normalization scale | [FileContracts](../tests/test_reliability.py#L109) |
| 17 | duplicate sample index | [FileContracts](../tests/test_reliability.py#L114) |
| 18 | missing sample assignment | [FileContracts](../tests/test_reliability.py#L119) |
| 19 | unknown split label | [FileContracts](../tests/test_reliability.py#L124) |
| 20 | empty split | [FileContracts](../tests/test_reliability.py#L129) |
| 21 | zero corruption is identity | [NumericalContracts](../tests/test_reliability.py#L140) |
| 22 | dropout removes whole detector rows | [NumericalContracts](../tests/test_reliability.py#L144) |
| 23 | corruption reproducible from seed | [NumericalContracts](../tests/test_reliability.py#L150) |
| 24 | gain is bounded and constant over time | [NumericalContracts](../tests/test_reliability.py#L159) |
| 25 | model restores unpadded dimensions | [NumericalContracts](../tests/test_reliability.py#L165) |
| 26 | model preserves batch and channels | [NumericalContracts](../tests/test_reliability.py#L170) |
| 27 | log variance clamped at both limits | [NumericalContracts](../tests/test_reliability.py#L176) |
| 28 | finite nonzero training gradients | [NumericalContracts](../tests/test_reliability.py#L185) |
| 29 | loss matches hand calculation | [NumericalContracts](../tests/test_reliability.py#L193) |
| 30 | perfect unit variance prediction zero loss | [NumericalContracts](../tests/test_reliability.py#L198) |
| 31 | unknown training key fails before writes | [PreflightContracts](../tests/test_reliability.py#L219) |
| 32 | duplicate stress names rejected | [PreflightContracts](../tests/test_reliability.py#L223) |
| 33 | clean stress name cannot overwrite clean metrics | [PreflightContracts](../tests/test_reliability.py#L227) |
| 34 | stress path traversal rejected | [PreflightContracts](../tests/test_reliability.py#L231) |
| 35 | required gpu failure creates no output | [PreflightContracts](../tests/test_reliability.py#L235) |
| 36 | paired groups must match | [PairedStatisticsContracts](../tests/test_reliability.py#L249) |
| 37 | paired delta matches independent calculation | [PairedStatisticsContracts](../tests/test_reliability.py#L255) |
| 38 | constant improvement interval | [PairedStatisticsContracts](../tests/test_reliability.py#L260) |
| 39 | bootstrap seed is reproducible | [PairedStatisticsContracts](../tests/test_reliability.py#L267) |
| 40 | nonfinite group metrics rejected | [PairedStatisticsContracts](../tests/test_reliability.py#L272) |
| 41 | duplicate groups rejected | [PairedStatisticsContracts](../tests/test_reliability.py#L278) |
| 42 | nan evaluation corruption rejected | [InvalidInferenceContracts](../tests/test_reliability.py#L286) |
| 43 | infinite evaluation corruption rejected | [InvalidInferenceContracts](../tests/test_reliability.py#L291) |
| 44 | negative prediction index rejected | [InvalidInferenceContracts](../tests/test_reliability.py#L296) |
| 45 | invalid checkpoint normalization rejected | [InvalidInferenceContracts](../tests/test_reliability.py#L300) |
