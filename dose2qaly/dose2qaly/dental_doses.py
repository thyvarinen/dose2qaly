"""Dental imaging organ-dose coefficients.

Provides per-organ absorbed dose coefficients (mGy per Gy·cm² reference DAP)
for dental CBCT, panoramic, intraoral, and cephalometric (kefalometry) imaging.

The coefficients scale linearly with the actual DAP recorded in the Excel row.
This module is consumed by ``exam_doses.compute_exam_organ_doses``.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Dental CBCT -- per-kVp coefficients (60-120 kV).
#
# Source: PCXMC v2.0, one file per tube voltage. Each file simulates 500
# projections (a full CBCT rotation), each projection run at
# "Input dose quantity"='DAP',
# "Input dose value"=1.0. NOTE: per the PCXMC user's guide, the DAP input
# quantity is in mGy*cm^2 (NOT Gy*cm^2) -- so each projection is modelled at
# DAP = 1 mGy*cm^2 = 0.001 Gy*cm^2. Row 510 ("Examination data" sheet) is the
# PCXMC-computed sum of organ doses / ED over all 500 projections; dividing
# by 500 normalises that sum to a coefficient per unit of total DAP
# (reference DAP = 0.001 Gy*cm^2), consistent with the mGy-per-Gy*cm^2
# convention used elsewhere in this module.
# Filtration is fixed at 0 mm Al + 0.2 mm Cu additional filter for every kVp.
# ---------------------------------------------------------------------------

DENTAL_CBCT_REFERENCE_DAP_GY_CM2 = 0.001  # PCXMC input DAP used to generate the table above (1 mGy*cm^2)

DENTAL_CBCT_DOSES_BY_KVP = {
    60: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 9.3186e-05,
        "coeffs": {
            "Active_Bone_Marrow": 0.000107974,
            "Adrenals": 8e-08,
            "Brain": 0.000284184,
            "Breast": 1.224e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.000855296,
            "Gallbladder": 6e-09,
            "Heart": 7.86e-07,
            "Kidneys": 1e-08,
            "Liver": 2.6e-08,
            "Lungs": 5.796e-06,
            "Lymph_Nodes": 0.000396576,
            "Muscle": 6.7186e-05,
            "Oesophagus": 3.326e-06,
            "Oral_Mucosa": 0.002045922,
            "Ovaries": 0,
            "Pancreas": 8e-08,
            "Prostate": 0,
            "Salivary_Glands": 0.00340855,
            "Skeleton": 0.000520324,
            "Skin": 0.000149588,
            "Small_Intestine": 0,
            "Spleen": 1.72e-07,
            "Stomach": 3.4e-08,
            "Testicles": 0,
            "Thymus": 3.012e-06,
            "Thyroid": 0.00011228,
            "Urinary_Bladder": 0,
            "Uterus": 0,
            "Average_Total_Body": 0.000129958,
        },
    },
    70: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.00010952,
        "coeffs": {
            "Active_Bone_Marrow": 0.000132994,
            "Adrenals": 1.14e-07,
            "Brain": 0.000376366,
            "Breast": 1.78e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.001091556,
            "Gallbladder": 2.2e-08,
            "Heart": 1.208e-06,
            "Kidneys": 1.2e-08,
            "Liver": 2.36e-07,
            "Lungs": 8.032e-06,
            "Lymph_Nodes": 0.000457006,
            "Muscle": 7.5296e-05,
            "Oesophagus": 5.122e-06,
            "Oral_Mucosa": 0.002409216,
            "Ovaries": 0,
            "Pancreas": 1.8e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.003740324,
            "Skeleton": 0.000607746,
            "Skin": 0.000158288,
            "Small_Intestine": 0,
            "Spleen": 2.96e-07,
            "Stomach": 7.8e-08,
            "Testicles": 0,
            "Thymus": 4.682e-06,
            "Thyroid": 0.000150434,
            "Urinary_Bladder": 0,
            "Uterus": 0,
            "Average_Total_Body": 0.000149598,
        },
    },
    80: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.000124438,
        "coeffs": {
            "Active_Bone_Marrow": 0.000156384,
            "Adrenals": 2.62e-07,
            "Brain": 0.000463652,
            "Breast": 2.432e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.001314384,
            "Gallbladder": 6.2e-08,
            "Heart": 1.824e-06,
            "Kidneys": 1.6e-08,
            "Liver": 5e-07,
            "Lungs": 1.0302e-05,
            "Lymph_Nodes": 0.000511668,
            "Muscle": 8.2458e-05,
            "Oesophagus": 7.282e-06,
            "Oral_Mucosa": 0.002741094,
            "Ovaries": 0,
            "Pancreas": 3.64e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.004024998,
            "Skeleton": 0.000675618,
            "Skin": 0.000166112,
            "Small_Intestine": 0,
            "Spleen": 4.64e-07,
            "Stomach": 1.62e-07,
            "Testicles": 0,
            "Thymus": 6.586e-06,
            "Thyroid": 0.000186782,
            "Urinary_Bladder": 0,
            "Uterus": 0,
            "Average_Total_Body": 0.000166022,
        },
    },
    90: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.000136672,
        "coeffs": {
            "Active_Bone_Marrow": 0.000175838,
            "Adrenals": 3.44e-07,
            "Brain": 0.000536338,
            "Breast": 2.99e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.001500328,
            "Gallbladder": 7.4e-08,
            "Heart": 2.4e-06,
            "Kidneys": 2.2e-08,
            "Liver": 7.02e-07,
            "Lungs": 1.2296e-05,
            "Lymph_Nodes": 0.000556596,
            "Muscle": 8.8294e-05,
            "Oesophagus": 9.34e-06,
            "Oral_Mucosa": 0.003014112,
            "Ovaries": 1.2e-08,
            "Pancreas": 4.94e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.004254236,
            "Skeleton": 0.000722684,
            "Skin": 0.000172484,
            "Small_Intestine": 0,
            "Spleen": 6.32e-07,
            "Stomach": 3.04e-07,
            "Testicles": 0,
            "Thymus": 8.388e-06,
            "Thyroid": 0.00021711,
            "Urinary_Bladder": 0,
            "Uterus": 2e-09,
            "Average_Total_Body": 0.000178376,
        },
    },
    100: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.000146442,
        "coeffs": {
            "Active_Bone_Marrow": 0.000191494,
            "Adrenals": 4.24e-07,
            "Brain": 0.000594976,
            "Breast": 3.52e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.001651448,
            "Gallbladder": 1.12e-07,
            "Heart": 2.9e-06,
            "Kidneys": 3.4e-08,
            "Liver": 8.8e-07,
            "Lungs": 1.3946e-05,
            "Lymph_Nodes": 0.00059267,
            "Muscle": 9.2978e-05,
            "Oesophagus": 1.1048e-05,
            "Oral_Mucosa": 0.003232878,
            "Ovaries": 1.2e-08,
            "Pancreas": 6.06e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.004436088,
            "Skeleton": 0.000753394,
            "Skin": 0.000177526,
            "Small_Intestine": 0,
            "Spleen": 7.66e-07,
            "Stomach": 4.38e-07,
            "Testicles": 0,
            "Thymus": 9.956e-06,
            "Thyroid": 0.000241278,
            "Urinary_Bladder": 0,
            "Uterus": 2e-09,
            "Average_Total_Body": 0.000187456,
        },
    },
    110: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.000154346,
        "coeffs": {
            "Active_Bone_Marrow": 0.000204244,
            "Adrenals": 5.52e-07,
            "Brain": 0.000642444,
            "Breast": 3.922e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.001775986,
            "Gallbladder": 1.46e-07,
            "Heart": 3.31e-06,
            "Kidneys": 5.2e-08,
            "Liver": 1.024e-06,
            "Lungs": 1.535e-05,
            "Lymph_Nodes": 0.000621992,
            "Muscle": 9.6714e-05,
            "Oesophagus": 1.2552e-05,
            "Oral_Mucosa": 0.003412722,
            "Ovaries": 1.2e-08,
            "Pancreas": 7.06e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.004580972,
            "Skeleton": 0.00077287,
            "Skin": 0.000181574,
            "Small_Intestine": 0,
            "Spleen": 8.84e-07,
            "Stomach": 5.52e-07,
            "Testicles": 0,
            "Thymus": 1.128e-05,
            "Thyroid": 0.000260776,
            "Urinary_Bladder": 0,
            "Uterus": 2e-09,
            "Average_Total_Body": 0.000194094,
        },
    },
    120: {
        "filtration_al": 0.0,
        "filtration_cu": 0.2,
        "effective_dose_msv": 0.00016081200000000024,
        "coeffs": {
            "Active_Bone_Marrow": 0.00021475800000000076,
            "Adrenals": 6.54e-07,
            "Brain": 0.00068107,
            "Breast": 4.322e-06,
            "Colon": 0,
            "Extrathoracic_Airways": 0.00188098,
            "Gallbladder": 1.76e-07,
            "Heart": 3.72e-06,
            "Kidneys": 8.2e-08,
            "Liver": 1.164e-06,
            "Lungs": 1.647e-05,
            "Lymph_Nodes": 0.0006462519999999997,
            "Muscle": 9.98e-05,
            "Oesophagus": 1.3852e-05,
            "Oral_Mucosa": 0.0035638,
            "Ovaries": 1.2e-08,
            "Pancreas": 7.92e-07,
            "Prostate": 0,
            "Salivary_Glands": 0.004698706,
            "Skeleton": 0.00078448,
            "Skin": 0.000184868,
            "Small_Intestine": 0,
            "Spleen": 1.014e-06,
            "Stomach": 6.34e-07,
            "Testicles": 0,
            "Thymus": 1.2362e-05,
            "Thyroid": 0.000276466,
            "Urinary_Bladder": 0,
            "Uterus": 2e-09,
            "Average_Total_Body": 0.000199004,
        },
    },
}

_DENTAL_CBCT_KVPS = sorted(DENTAL_CBCT_DOSES_BY_KVP)


def nearest_dental_cbct_kvp(kvp) -> int:
    """Return the nearest available Dental CBCT kVp bin (60-120 kV).

    Falls back to 80 kV (mid-range) when ``kvp`` is missing or unparseable.
    """
    try:
        kv = float(kvp)
    except (TypeError, ValueError):
        kv = 80.0
    return min(_DENTAL_CBCT_KVPS, key=lambda k: abs(k - kv))


KEFALOMETRY_REFERENCE_DAP_GY_CM2 = 1.0

KEFALOMETRY_IMAGING_DOSES = {
    "Active_Bone_Marrow": 0.0,
    "Adrenals": 0.0,
    "Brain": 0.0,
    "Breast": 0.0,
    "Colon": 0.0,
    "Upper_Large_Intestine": 0.0,
    "Lower_Large_Intestine": 0.0,
    "Extrathoracic_Airways": 0.0,
    "Gallbladder": 0.0,
    "Heart": 0.0,
    "Kidneys": 0.0,
    "Liver": 0.0,
    "Lungs": 0.0,
    "Lymph_Nodes": 0.0,
    "Muscle": 0.0,
    "Oesophagus": 0.0,
    "Oral_Mucosa": 0.0,
    "Ovaries": 0.0,
    "Pancreas": 0.0,
    "Prostate": 0.0,
    "Salivary_Glands": 0.0,
    "Skeleton": 0.0,
    "Pelvis": 0.0,
    "Skin": 0.0,
    "Small_Intestine": 0.0,
    "Spleen": 0.0,
    "Stomach": 0.0,
    "Testicles": 0.0,
    "Thymus": 0.0,
    "Thyroid": 0.0,
    "Urinary_Bladder": 0.0,
    "Uterus": 0.0,
    "Average_Total_Body": 0.0,
}


# Placeholders -- per-organ coefficients for Dental Panoramic and Intraoral
# are not yet modelled.  All values are 0.0 so these modalities yield zero
# organ dose and zero effective dose until proper PCXMC coefficients are
# available.

DENTAL_PANORAMIC_REFERENCE_DAP_GY_CM2 = 1.0

DENTAL_PANORAMIC_DOSES = {
    "Active_Bone_Marrow": 0.0,
    "Adrenals": 0.0,
    "Brain": 0.0,
    "Breast": 0.0,
    "Colon": 0.0,
    "Extrathoracic_Airways": 0.0,
    "Gallbladder": 0.0,
    "Heart": 0.0,
    "Kidneys": 0.0,
    "Liver": 0.0,
    "Lungs": 0.0,
    "Lymph_Nodes": 0.0,
    "Muscle": 0.0,
    "Oesophagus": 0.0,
    "Oral_Mucosa": 0.0,
    "Ovaries": 0.0,
    "Pancreas": 0.0,
    "Prostate": 0.0,
    "Salivary_Glands": 0.0,
    "Skeleton": 0.0,
    "Skin": 0.0,
    "Small_Intestine": 0.0,
    "Spleen": 0.0,
    "Stomach": 0.0,
    "Testicles": 0.0,
    "Thymus": 0.0,
    "Thyroid": 0.0,
    "Urinary_Bladder": 0.0,
    "Uterus": 0.0,
    "Average_Total_Body": 0.0,
}

INTRAORAL_REFERENCE_DAP_GY_CM2 = 1.0

INTRAORAL_IMAGING_DOSES = {
    "Active_Bone_Marrow": 0.0,
    "Adrenals": 0.0,
    "Brain": 0.0,
    "Breast": 0.0,
    "Colon": 0.0,
    "Upper_Large_Intestine": 0.0,
    "Lower_Large_Intestine": 0.0,
    "Extrathoracic_Airways": 0.0,
    "Gallbladder": 0.0,
    "Heart": 0.0,
    "Kidneys": 0.0,
    "Liver": 0.0,
    "Lungs": 0.0,
    "Lymph_Nodes": 0.0,
    "Muscle": 0.0,
    "Oesophagus": 0.0,
    "Oral_Mucosa": 0.0,
    "Ovaries": 0.0,
    "Pancreas": 0.0,
    "Prostate": 0.0,
    "Salivary_Glands": 0.0,
    "Skeleton": 0.0,
    "Skull": 0.0,
    "Upper_Spine": 0.0,
    "Middle_Spine": 0.0,
    "Lower_Spine": 0.0,
    "Scapulae": 0.0,
    "Clavicles": 0.0,
    "Ribs": 0.0,
    "Upper_Arm_Bones": 0.0,
    "Middle_Arm_Bones": 0.0,
    "Lower_Arm_Bones": 0.0,
    "Pelvis": 0.0,
    "Upper_Leg_Bones": 0.0,
    "Middle_Leg_Bones": 0.0,
    "Lower_Leg_Bones": 0.0,
    "Skin": 0.0,
    "Small_Intestine": 0.0,
    "Spleen": 0.0,
    "Stomach": 0.0,
    "Testicles": 0.0,
    "Thymus": 0.0,
    "Thyroid": 0.0,
    "Urinary_Bladder": 0.0,
    "Uterus": 0.0,
    "Average_Total_Body": 0.0,
}

# Mapping from dental region/subtype to dose coefficients and reference DAP.
# NOTE: "Dental CBCT" is intentionally absent here -- it is handled via the
# per-kVp DENTAL_CBCT_DOSES_BY_KVP table above instead of a single fixed
# coefficient set.
DENTAL_DOSE_MAPPING = {
    "Dental Panoramic": {
        "coeffs": DENTAL_PANORAMIC_DOSES,
        "reference_dap_gy_cm2": DENTAL_PANORAMIC_REFERENCE_DAP_GY_CM2,
    },
    "Dental_intraoral": {
        "coeffs": INTRAORAL_IMAGING_DOSES,
        "reference_dap_gy_cm2": INTRAORAL_REFERENCE_DAP_GY_CM2,
    },
    "Kefalometry": {
        "coeffs": KEFALOMETRY_IMAGING_DOSES,
        "reference_dap_gy_cm2": KEFALOMETRY_REFERENCE_DAP_GY_CM2,
    },
}

