# Dose2QALY

Monte Carlo lifetime cancer-risk simulation for diagnostic
medical-imaging examination records.

## What it does

Dose2QALY reads imaging examination records (Excel) and
estimates, per exam, the absorbed dose to each radiosensitive organ.
It then draws Monte Carlo realisations of lifetime cancer outcomes
using the ICRP Publication 152 REIC model. One iteration simulates
one year of the cohort.

Outputs include induced cancer cases, fatal cancers, ICRP
severity-weighted quality-adjusted
life-years (QALYs) lost, life-years lost, organ doses, per-exam effective dose, and
collective effective dose (person Sv) per year.

Dose models exist for:

- CT: direct per-organ doses from the dose-tracking
  system's "(CT) mSv" export columns.
- Mammography (MG): the mean glandular dose is taken as the breast
  absorbed dose.
- Radiography (CR/DX/DR): DAP x PCXMC v2.0 radiography coefficients.
- CBCT: dental CBCT only (DAP x PCXMC v2.0 dental coefficients).

Other modalities, and non-dental CBCT, are counted as exams but have
no dose model: they contribute no modelled cancer risk.

## Status

Research code accompanying a manuscript. Outputs have been validated
only against the authors' own data. Open items:

- Constants without an in-code source are marked "source needed".

## Install

```powershell
pip install -e .
```

Optional extras:

- `pip install -e ".[gpu]"` — CuPy for GPU-accelerated iterations.
  Install the CuPy wheel matching your CUDA toolkit instead, e.g.
  `pip install cupy-cuda12x`.
- `pip install -e ".[perf]"` — psutil, threadpoolctl, zstandard
  (CPU tuning and faster preprocess cache).
- `pip install -e ".[dev]"` — openpyxl (Excel writing), ruff.

The GUI uses Tkinter (stdlib). On Linux install the `python3-tk`
system package.

## Quick start

GUI:

```powershell
python -m dose2qaly
```


## Input format

Rows are imaging exams; columns recognised by the engine are defined
in `excel_io.NEEDED_COLS_ORDERED`. Equivalent headers are merged via
`COLUMN_ALTERNATIVES` (`Gender` / `Sex`; the filtration-column
spellings). The "Input File Columns" tab of the setup dialog shows what
each column is used for, by modality.

| Group | Columns |
|---|---|
| Patient | `A#` (examination), `Patient MRN` (person), `Gender` (or `Sex`), `Patient Weight kg`, `Population` |
| Exam metadata | `Age at Exam yrs`, `Date Performed`, `Device`, `Equipment` (last three optional) |
| Descriptions | `Description`, `Protocol` |
| Region classification | `Scan Regions` (CT only), `Body Part Examined (DX)`, `Body Part Examined (CBCT)` |
| Modality | `Acquisition Modality` (`Examination Modalities` and `Modality` accepted as fallbacks) |
| Radiography / angiography dose | `DAP (Total) Gy-cm2`, `kVp kV`, `Xray Filter Max Thickness (DX)` (DICOM added-filter thickness set) |
| Mammography dose | `Mean Glandular Dose mGy` |
| Pre-calculated effective dose | `ICRP 103 mSv` |
| CT organ doses | the 28 `"... (CT) mSv"` per-organ columns, plus `Remainder ICRP103 mSv` |

Exam-description parsing in `region_resolver` is tuned to Finnish RIS
description strings and NCSP codes, with English fallback tokens;
CT region classification uses only the DICOM `Scan Regions` field and
is language-independent.

## Outputs

`run_simulation` returns `(stats, iter_cases, iter_deaths, events,
exam_data)`, where `iter_cases` and `iter_deaths` are the per-iteration
count lists. Key `stats` fields — the totals are **sums over all
simulated iterations (years)**; divide by `stats["iterations"]` for the
annual mean:

- `cancer_cases`, `cancer_deaths` — induced and fatal cancers.
- `QALY_lost`, `life_lost` — QALYs and life-years lost.
- `total_sv` — collective effective dose (person Sv).
- `total_detriment` — total detriment.
- `expected_incidence`, `expected_deaths` — model expectation
  accumulated per iteration (no Monte Carlo noise in each term).
- `iter_QALY_list`, `iter_life_list` — per-iteration QALY and
  life-year series.
- `heritable_risk`, `heritable_detriment` — heritable component.
- `modality_stats` — the same quantities per modality.
- `_convergence` — stopping-rule metadata: stop reason
  (`stability_and_precision` / `stability_only` / `max_iterations` /
  `not_converged`), thresholds, final relative SEs.

Stopping rule: the run stops once the per-age-bin risk-per-Sv
estimates are stable (relative change of the 95 % CI half-width below
`convergence_pct` for `CONVERGENCE_WINDOW` consecutive iterations,
after the minimum iteration count) and, when `precision_stop_enabled`
is on (default), the relative standard error of both annual cancer
count and annual QALYs is at or below `precision_target_rel_se`
(default 0.01). `n_iterations` is a hard cap; reaching it is reported
in `_convergence`, not hidden.

## Reproducibility and verification

`seed` makes `run_simulation` reproducible. The risk engine was
validated against the ICRP Publication 152 adult working-population
reference scenario (synthetic cohort, uniform 0.1 Gy organ dose); the
validation inputs and results are provided in the `validation/`
directory.

## Limitations

- Bernoulli draws use float32; measured effect on the synthetic cohort
  is +0.05 % expected hits overall, up to about +1 % for
  low-probability modalities.
- The preprocess cache fingerprints only the first 4 MB of each input
  file.
- Per-year batch runs (`run_yearly_simulation`) are unseeded.
- The GUI is closure-based and covered only by a smoke test.
- Exam-description parsing is tuned to Finnish RIS strings (see
  Input format).

## Citing

If you use Dose2QALY, please cite the article it accompanies:

Hyvärinen T, Brix M, Liimatainen T, Hanni M. Quality-adjusted life-years
lost to radiation from diagnostic imaging. (Reference to be updated on
publication; machine-readable form in `CITATION.cff`.)

## License

MIT (see `LICENSE`).
