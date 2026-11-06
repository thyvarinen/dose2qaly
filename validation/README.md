# Validation against ICRP Publication 152

The two workbooks in this directory are synthetic cohorts used to check that
the Dose2QALY risk engine reproduces the ICRP Publication 152 nominal risk
coefficients. They contain no patient data.

| File | Rows | Population |
|---|---|---|
| `ICRP152_AgeDistribution_CT_EuroAmerican.xlsx` | 1 000 000 | Euro-American |
| `ICRP152_AgeDistribution_CT_Asian.xlsx` | 1 000 000 | Asian |

Each row is one person receiving a uniform absorbed dose of 0.1 Gy
(100 mSv) to every organ, with age at exposure drawn from the ICRP 152
Annex A Table A.7 age distribution and sex assigned 50/50. All organ-dose
columns, including sex-specific organs, are filled for every row; the
engine applies each cancer site to the appropriate sex. Only the columns
that define the scenario carry values (modality, age, sex, population and
the CT organ, remainder and effective doses); the remaining columns of the
Dose2QALY input whitelist are present as empty headers.

## Reproducing the check

1. Start the GUI (`dose2qaly` or `python -m dose2qaly`) and load one of the
   workbooks.
2. In the simulation setup choose the matching **Population**
   (Euro-American or Asian), risk model **REIC**, and set
   **ICRP 152 reference assumptions: Enabled**. This switches off the
   body-size (Golikov) and competing-mortality (Mataac) adjustments so the
   engine runs with ICRP's own survival and dose assumptions.
3. Run the simulation. Leave the precision target at its default so the run
   continues until the relative standard error of the totals is below 1 %.
4. Open **By Cancer Type -> Cancers per Sv**. The table lists cases per
   10 000 persons per Sv for each cancer site and the total, in the layout of
   ICRP 152 Table 3.9.
5. Compare the site-specific values and the total with ICRP 152 Table 3.9.
   The expected agreement is reported in the supplementary appendix of the
   accompanying article (Table S5).

Runtime for one million rows is of the order of minutes on a modern
desktop; GPU acceleration (`pip install -e ".[gpu]"`) is optional.
