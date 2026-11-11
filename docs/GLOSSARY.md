# Glossary

Domain terms used throughout `dose2qaly`. Definitions describe
how the code uses each term. Where a definition comes from an ICRP
publication the code cites, the publication is named; otherwise terms are
defined from the code's own usage.

## Risk model terms

**ICRP 103 / ICRP 152** — International Commission on Radiological
Protection publications. ICRP 103 (2007) defines effective dose and the
severity-adjusted detriment formula used here; ICRP 152 (2022) defines the
lifetime cancer-risk model (risk coefficients per cancer site, lethality
fractions, life-lost factors) used here.

**LAR (Lifetime Attributable Risk)** — cancer incidence risk integrated
from exposure age to age 94, weighted by background survival only
(ICRP 152 §52 footnote).

**REIC (Risk of Exposure-Induced Cancer)** — like LAR, but the integral
uses cancer-free survival S_cf(a|e) instead of background survival
(ICRP 152 §46). Default risk model (`risk_model="REIC"`); the difference
from LAR is that a case already counted as radiation-induced cannot be
counted again, so REIC integrates the probability of being the *first*
cancer.

**ERR (Excess Relative Risk)** — risk model in which the radiation-induced
hazard is proportional to the baseline incidence rate. Site-specific
per-Gy coefficient; see `risk_models`.

**EAR (Excess Absolute Rate)** — risk model in which the radiation-induced
hazard is an additive rate per 10 000 person-years per Gy, independent of
the baseline rate. Leukaemia uses an EAR kernel.

**DDREF (Dose and Dose-Rate Effectiveness Factor)** — low-dose /
low-dose-rate adjustment, applied as a divisor on the linear-in-dose term
of solid-cancer kernels (default 2.0) and leukaemia (default 1.0).
Configurable for sensitivity testing.

**Latency** — minimum number of years after exposure before a
radiation-induced cancer is counted. The site-specific coefficients define
a latency plateau (e.g. leukaemia differs from solid cancers); risk before
the latency end is zero or ramped, as coded in `risk_models`.

**Detriment (ICRP 103)** — severity-weighted measure of radiation harm per
unit dose. Computed per site as `D = R·(k + q·(1−k))·l` where R is nominal
risk, k the lethality fraction, q the quality-of-life factor and l the
relative life-loss factor (ICRP 103 p.51 paras 78/83; parameters from
ICRP 152 Table A.4.1). Reported in `detriment_arr` and `total_detriment`.

**Severity weighting / q_min** — `q = k + q_min·(1−k)`: the
quality-of-life factor attached to a cancer for detriment/QALY
accounting. `q_min` is the minimum severity weight for a non-fatal case;
a fatal case carries weight 1.

**l_rel** — relative life-loss factor per site (ICRP 152 Table 3.9 /
A.4.1): the site's mean years-of-life-lost relative to the all-site
average, used in detriment and the `"icrp152"` QALY mode.

**Lethality fraction k** — probability that a radiation-induced cancer of a
given site is fatal (ICRP 152 Table A.4.1). Used both in the detriment
formula and in the Monte Carlo case-fatality draw.

**QALYs (as used here)** — "ICRP severity-weighted QALYs":
quality-adjusted life-years lost where the quality adjustment is the
ICRP-152 cancer-severity weighting. `sim_config["QALY_mode"]` defaults to
`"full"` (per-cancer `q_individual × YLL`, with q = 1 if the drawn case
was fatal else `q_severity`); `"icrp152"` uses
`q_severity × l_rel × 15 y`. Display text says "QALYs lost" / "QALYs
gained"; never "QALYs saved".

**YLL / life-years lost** — years from a modelled cancer's onset age to
that patient's sampled natural lifespan (competing-mortality-adjusted).

**Competing mortality** — the chance a patient dies of another cause
before a radiation-induced cancer would appear. Lifespans are sampled per
cancer hit, conditioned on the hit's onset age, so a hit can only be
fatal/loss-bearing within the patient's remaining life.

**Mataac frailty multiplier** — empirical additive excess-hazard
correction (Mataac et al. 2024) applied in `preprocess` for exams whose
cumulative effective dose places the patient in a frail stratum (≥ 10 mSv
yearly CT CED). Patients with high prior dose die sooner of other causes,
which lowers their modelled radiation risk. See `mataac_survival_correction`.

## Dose terms

**Organ absorbed dose** — energy deposited per unit organ mass, in gray
(Gy); the quantity the risk model consumes per organ/site.

**Effective dose (E)** — single-number dose metric in sievert (Sv;
input columns and `ed_msv` are mSv). It is a *per-exam* quantity. For CT
it is read from the input file's effective-dose column; for radiography,
mammography and dental CBCT exams it comes from the modality's DAP- or
MGD-based coefficient tables (see `exam_doses`).

**Collective effective dose** — the sum of effective doses over a
population or a year, in person-Sv (displayed "person Sv/yr"). Never
abbreviated to bare "effective dose".

**CED (cumulative effective dose)** — running per-patient total of
effective dose (Sv), used to assign Mataac frailty strata.

**DAP / KAP (dose-area product / kerma-area product)** — machine-reported
radiography exposure quantity in Gy·cm²; multiplied by per-organ
coefficients (mGy per Gy·cm²) to get organ doses.

**MGD (mean glandular dose)** — the mammography-specific dose quantity
reported by the device (mGy), converted to breast absorbed dose and
effective dose in `mammography_doses`.

**Golikov size correction** — body-size scaling (Golikov/Druzhinina
coefficient model) applied to CT doses of adult patients (age
≥ 18) only: doses are scaled relative to the 70 kg reference phantom via
`exam_doses._golikov_relative_scale`. Disabled with `use_golikov=False`
in the simulation config.

## Simulation terms

**Monte Carlo iteration / Bernoulli draw** — one simulated "year" of the
cohort: for every exam × cancer site, a Bernoulli draw with the site's
per-exam lifetime-risk probability decides whether that cancer occurs in
this iteration. Iterations are independent realisations; results are
averaged over them.

**Precision stopping rule** — the run stops when the stability
rule (95 % CI half-width change < `convergence_pct` for 20 consecutive
iterations) is met *and* the relative standard error of both the annual
cancer count and annual QALYs is ≤ `precision_target_rel_se` (default
0.02). `n_iterations` is the hard cap. See `_stopping_decision` in
`main.py`.

**Welford** — numerically stable online mean/variance algorithm used to
track the convergence statistics without storing all iterations.
