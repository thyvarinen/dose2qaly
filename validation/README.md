# Validation against ICRP Publication 152

The two workbooks in this directory are synthetic cohorts used to check that
the Dose2QALY risk engine reproduces the ICRP Publication 152 nominal risk
coefficients. They contain no patient data.

| File | Rows | Population |
|---|---|---|
| `ICRP152_AgeDistribution_CT_EuroAmerican.xlsx` | 1 000 000 | Euro-American |
| `ICRP152_AgeDistribution_CT_Asian.xlsx` | 1 000 000 | Asian |

Each row is one person receiving a uniform absorbed dose of 0.1 Gy
(100 mSv) to every organ, with age at exposure (0-85 years) and sex
following the ICRP 152 Annex A Table A.7 reference population. All
organ-dose columns, including sex-specific organs, are filled for every
row; the engine applies each cancer site to the appropriate sex. Only the
columns that define the scenario carry values (modality, age, sex,
population and the CT organ, remainder and effective doses); the remaining
columns of the Dose2QALY input whitelist are present as empty headers.

## Reproducing the check

1. Start the GUI (`dose2qaly` or `python -m dose2qaly`) and add both
   workbooks as input files. Each row carries its own `Population`
   value, so no population needs to be selected in the setup.
2. In the simulation setup keep risk model **REIC** and set
   **ICRP 152 reference assumptions: Enabled**. This switches off the
   weight (Golikov and Druzhinina) and competing-mortality (Mataac et al.)
   adjustments so the engine runs with ICRP's own survival and dose
   assumptions.
3. Run the simulation. Leave the precision stop at its default so the run
   continues until the relative standard error of the annual cancer count
   and QALYs is at or below 1 %.
4. Open **By Cancer Type -> Cancers per Sv**. The table lists, for the
   simulated cohort (all ages), cases per 10 000 persons per Sv for each
   cancer site and the total, in the layout of ICRP 152 Table 3.9.
5. Click **Compare with ICRP152 values** on the same tab. The window shows
   the nominal risk coefficients for the working-age population (age at
   exposure 18-64 years, Table A.7 reference demographic, averaged over
   sexes and over the populations present in the input) next to the
   ICRP 152 Table 3.8 values, with the deviation per cancer site and in
   total. These coefficients are computed from the engine's risk kernels
   and survival functions. This is the comparison reported in the
   supplementary appendix of the accompanying article.

Runtime for the two million rows is of the order of minutes on a modern
desktop; GPU acceleration (`pip install -e ".[gpu]"`) is optional.
