# Dose2QALY

Monte Carlo estimation of lifetime cancer risk and quality-adjusted
life-years (QALYs) lost from diagnostic medical-imaging examination
records.

## What it does

Dose2QALY reads a hospital's imaging examination records (Excel exports
from a dose-tracking system) and estimates, per exam, the absorbed dose
to each radiosensitive organ. It then draws Monte Carlo realisations of
lifetime cancer outcomes using the ICRP Publication 152 REIC model. One
iteration simulates one year of the cohort.

Outputs include induced cancer cases, fatal cancers, ICRP
severity-weighted quality-adjusted life-years (QALYs) lost, life-years
lost, per-exam effective dose, and collective effective dose
(person Sv) per year, by modality, imaging region, age group and cancer
site.

Organ doses are obtained as follows:

- CT: per-organ doses are read directly from the dose-tracking
  system's "(CT) mSv" export columns.
- Mammography (MG): the mean glandular dose is taken as the breast
  absorbed dose.
- Radiography (CR/DX/DR) and dental CBCT: organ doses are DAP
  multiplied by conversion coefficients derived with PCXMC v2.0 (STUK).
  Coefficient sets are included for:
  - radiography: chest, cervical/thoracic/lumbar spine, scoliosis,
    skull, pelvis, abdomen and extremities, in AP, PA, lateral and
    oblique projections, for adult and paediatric phantoms and for the
    beam qualities (kVp, filtration) in the input;
  - dental CBCT: 60-120 kVp.

Fluoroscopy and angiography (XA/RF), dental panoramic, intraoral and
cephalometric radiography, non-dental CBCT and other modalities have no
dose model; the coefficient tables for fluoroscopy/angiography and for
dental radiography are zero-valued placeholders. These exams are
reported as excluded examinations and contribute no modelled cancer
risk.

Two adjustments are applied by default and can be switched off in the
setup dialog:

- Weight adjustment (Golikov and Druzhinina) for adult chest and
  abdomen CT, from `Patient Weight kg`.
- Competing-mortality correction (Mataac et al.) for patients whose
  cumulative CT effective dose within a calendar year is 10 mSv or
  more. It needs a per-person identifier (`Patient MRN`); without one
  every exam is treated as a separate patient and the correction is
  under-applied.

"ICRP 152 reference assumptions" turns both off, so the engine runs
with ICRP's own survival and dose assumptions.

## Status

Research code accompanying a manuscript. The risk engine has been
checked against the ICRP Publication 152 nominal risk coefficients (see
`validation/`). The dose models and adjustments have been used only on
the authors' own data.

## Install

Python 3.10 or newer.

```powershell
git clone https://github.com/thyvarinen/dose2qaly.git
cd dose2qaly
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
dose2qaly
```

(or `python -m dose2qaly`). The setup dialog takes the input files,
grouped by year, the run mode and the simulation parameters. "Average
across years" (default) runs each year on its own and reports the mean
annual burden; "Combine all years into one run" pools every file into a
single cohort. Results are shown in a tabbed window.

Script:

```python
from dose2qaly import run_simulation

if __name__ == "__main__":
    stats, iter_cases, iter_deaths, events, exam_data = run_simulation(
        ["exams_2024.xlsx"],
        n_iterations=1000,
        min_latency_params={"Leukaemia": 5.0, "Solid": 5.0},
        population="Euro-American",
        seed=1,
    )
    n = stats["iterations"]
    print("cancers per year:", stats["cancer_cases"] / n)
    print("QALYs lost per year:", stats["QALY_lost"] / n)
```

## Data and privacy

Dose2QALY runs locally and sends nothing anywhere. To speed up re-runs
it caches copies of the input files and the preprocessed exam records,
including per-patient cumulative doses, under the output directory
(default `~/dose2qaly_output`, selectable in the setup dialog). The
cache is not encrypted. Pseudonymise the input before use, keep the
output directory in a secure, access-controlled location, and never
commit it to a repository. Deleting the cache does not change results.

## Input format

Rows are imaging exams; columns recognised by the engine are defined
in `excel_io.NEEDED_COLS_ORDERED`. Equivalent headers are merged via
`COLUMN_ALTERNATIVES` (`Gender` / `Sex`; `Acquisition Modality` /
`Examination Modalities` / `Modality`). The "Input File Columns" tab of
the setup dialog shows what each column is used for, by modality.

| Group | Columns |
|---|---|
| Patient | `A#` (examination), `Patient MRN` (person), `Gender` (or `Sex`), `Patient Weight kg`, `Population` |
| Exam metadata | `Age at Exam yrs`, `Date Performed`, `Device`, `Equipment` (last three optional) |
| Descriptions | `Description`, `Protocol` |
| Region classification | `Scan Regions` (CT), `Body Part Examined (DX)`, `Body Part Examined (CBCT)` |
| Modality | `Acquisition Modality` |
| Radiography / dental CBCT dose | `DAP (Total) Gy-cm2`, `kVp kV`, `Xray Filter Max Thickness (DX)` (DICOM added-filter thickness set) |
| Mammography dose | `Mean Glandular Dose mGy` |
| Pre-calculated effective dose | `ICRP 103 mSv` |
| CT organ doses | the 28 `"... (CT) mSv"` per-organ columns, plus `Remainder ICRP103 mSv` |

`Population` (Euro-American / Asian) is optional per row; rows without
it take the population chosen in the setup dialog.

The imaging region comes from DICOM fields: `Scan Regions` for CT,
`Body Part Examined (DX)` for radiography and
`Body Part Examined (CBCT)` for CBCT. `Description` and `Protocol` are
used to recognise exposures with no patient in the beam (quality
control, phantoms), dental examinations, scoliosis imaging and the
radiographic projection. This text matching is tuned to Finnish RIS
description strings and NCSP codes, with English fallback tokens.

The two workbooks in `validation/` are complete examples of the layout.

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
`convergence_pct` for 20 consecutive iterations, after the minimum
iteration count) and, when `precision_stop_enabled` is on (default),
the relative standard error of both annual cancer count and annual
QALYs is at or below `precision_target_rel_se` (default 0.01).
`n_iterations` is a hard cap; reaching it is reported in
`_convergence`, not hidden.

Domain terms (REIC, DDREF, QALY, detriment, collective effective dose,
...) are defined in `docs/GLOSSARY.md`.

## Reproducibility and verification

`run_simulation(..., seed=...)` is reproducible on a given backend; CPU
and GPU use different random streams. Runs started from the GUI are not
seeded.

The risk engine was checked against the ICRP Publication 152 nominal
risk coefficients with a synthetic two-million-person cohort (uniform
0.1 Gy organ dose). The cohort files and step-by-step instructions are
in `validation/`.

## Repository layout

- `dose2qaly/` — the Python package.
- `validation/` — synthetic ICRP 152 validation cohorts and
  instructions.
- `docs/GLOSSARY.md` — definitions of the domain terms used in the
  code.

## Citing

If you use Dose2QALY, please cite the article it accompanies:

Hyvärinen T, Brix M, Liimatainen T, Hanni M. Quality-adjusted life-years
lost to radiation from diagnostic imaging. (Reference to be updated on
publication; machine-readable form in `CITATION.cff`.)

## License

MIT (see `LICENSE`).
