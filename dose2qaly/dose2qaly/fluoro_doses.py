"""Fluoroscopy / angiography organ-dose coefficients.

Sources
-------
* PCXMC v2.0

PCXMC was run at a DAP of 1 Gy·cm².
Structure mirrors ``radiography_doses.py`` (one large nested dict).

Units
-----
* ``coefficients_fluoro`` — organ doses in mGy per Gy·cm²
* ``eff_coefficients_fluoro`` — effective dose in mSv per Gy·cm²

Lookup hierarchy
----------------
  exam name → sex (F / M / U) → (kVp, mmAl, mmCu) or "default" → organ dict / float

Exams with PCXMC data
----------------------
  Coronary Angiography, Coronary Angioplasty, Carotid Angiography,
  ERCP, Peripheral Angiography.

Exams without PCXMC data (fallback placeholders)
----------------------------------------------
  Abdominal Angiography, Cystography, Unknown.

This module is consumed by ``exam_doses.compute_exam_organ_doses``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

from ._parse_utils import to_float as _xa_to_float


coefficients_fluoro: Dict = {

    # ── Coronary Angiography ──
    "Coronary Angiography": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.197, "Adrenals": 0.336, "Brain": 0.001, "Breast": 2.696, "Colon": 0.011, "Endosteum": 0.26, "Extrathoracic_Airways": 0.009, "Gallbladder": 0.103, "Heart": 3.766, "Kidneys": 0.065, "Liver": 0.406, "Lungs": 0.817, "Lymph_Nodes": 0.324, "Muscle": 0.184, "Oesophagus": 0.624, "Oral_Mucosa": 0.005, "Ovaries": 0.002, "Pancreas": 0.348, "Salivary_Glands": 0.005, "Skin": 0.178, "Small_Intestine": 0.013, "Spleen": 0.161, "Stomach": 0.365, "Testicles": 0.001, "Thymus": 2.326, "Thyroid": 0.028, "Urinary_Bladder": 0.001, "Uterus": 0.004},
            "default": {"Active_Bone_Marrow": 0.197, "Adrenals": 0.336, "Brain": 0.001, "Breast": 2.696, "Colon": 0.011, "Endosteum": 0.26, "Extrathoracic_Airways": 0.009, "Gallbladder": 0.103, "Heart": 3.766, "Kidneys": 0.065, "Liver": 0.406, "Lungs": 0.817, "Lymph_Nodes": 0.324, "Muscle": 0.184, "Oesophagus": 0.624, "Oral_Mucosa": 0.005, "Ovaries": 0.002, "Pancreas": 0.348, "Salivary_Glands": 0.005, "Skin": 0.178, "Small_Intestine": 0.013, "Spleen": 0.161, "Stomach": 0.365, "Testicles": 0.001, "Thymus": 2.326, "Thyroid": 0.028, "Urinary_Bladder": 0.001, "Uterus": 0.004},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.235, "Adrenals": 0.44, "Brain": 0.001, "Breast": 2.879, "Colon": 0.015, "Endosteum": 0.285, "Extrathoracic_Airways": 0.012, "Gallbladder": 0.137, "Heart": 4.37, "Kidneys": 0.087, "Liver": 0.494, "Lungs": 0.977, "Lymph_Nodes": 0.39, "Muscle": 0.217, "Oesophagus": 0.797, "Oral_Mucosa": 0.006, "Ovaries": 0.003, "Pancreas": 0.442, "Salivary_Glands": 0.007, "Skin": 0.19, "Small_Intestine": 0.018, "Spleen": 0.209, "Stomach": 0.44, "Testicles": 0.001, "Thymus": 2.612, "Thyroid": 0.036, "Urinary_Bladder": 0.002, "Uterus": 0.005},
            "default": {"Active_Bone_Marrow": 0.235, "Adrenals": 0.44, "Brain": 0.001, "Breast": 2.879, "Colon": 0.015, "Endosteum": 0.285, "Extrathoracic_Airways": 0.012, "Gallbladder": 0.137, "Heart": 4.37, "Kidneys": 0.087, "Liver": 0.494, "Lungs": 0.977, "Lymph_Nodes": 0.39, "Muscle": 0.217, "Oesophagus": 0.797, "Oral_Mucosa": 0.006, "Ovaries": 0.003, "Pancreas": 0.442, "Salivary_Glands": 0.007, "Skin": 0.19, "Small_Intestine": 0.018, "Spleen": 0.209, "Stomach": 0.44, "Testicles": 0.001, "Thymus": 2.612, "Thyroid": 0.036, "Urinary_Bladder": 0.002, "Uterus": 0.005},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.115, "Adrenals": 0.11, "Breast": 2.26, "Colon": 0.002, "Endosteum": 0.197, "Extrathoracic_Airways": 0.001, "Gallbladder": 0.03, "Heart": 2.373, "Kidneys": 0.019, "Liver": 0.211, "Lungs": 0.452, "Lymph_Nodes": 0.174, "Muscle": 0.109, "Oesophagus": 0.245, "Oral_Mucosa": 0.001, "Pancreas": 0.135, "Salivary_Glands": 0.001, "Skin": 0.15, "Small_Intestine": 0.003, "Spleen": 0.053, "Stomach": 0.189, "Thymus": 1.671, "Thyroid": 0.009},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.149, "Adrenals": 0.179, "Breast": 2.595, "Colon": 0.004, "Endosteum": 0.243, "Extrathoracic_Airways": 0.001, "Gallbladder": 0.051, "Heart": 3.15, "Kidneys": 0.033, "Liver": 0.299, "Lungs": 0.636, "Lymph_Nodes": 0.243, "Muscle": 0.146, "Oesophagus": 0.381, "Oral_Mucosa": 0.001, "Pancreas": 0.213, "Salivary_Glands": 0.001, "Skin": 0.165, "Small_Intestine": 0.005, "Spleen": 0.085, "Stomach": 0.274, "Thymus": 2.1, "Thyroid": 0.018, "Uterus": 0.001},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.173, "Adrenals": 0.237, "Breast": 2.767, "Colon": 0.005, "Endosteum": 0.273, "Extrathoracic_Airways": 0.002, "Gallbladder": 0.072, "Heart": 3.647, "Kidneys": 0.045, "Liver": 0.36, "Lungs": 0.764, "Lymph_Nodes": 0.293, "Muscle": 0.171, "Oesophagus": 0.488, "Oral_Mucosa": 0.002, "Pancreas": 0.28, "Salivary_Glands": 0.002, "Skin": 0.174, "Small_Intestine": 0.007, "Spleen": 0.112, "Stomach": 0.34, "Thymus": 2.311, "Thyroid": 0.027, "Urinary_Bladder": 0.001, "Uterus": 0.001},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.151, "Adrenals": 0.204, "Breast": 2.498, "Colon": 0.005, "Endosteum": 0.234, "Extrathoracic_Airways": 0.004, "Gallbladder": 0.06, "Heart": 3.073, "Kidneys": 0.038, "Liver": 0.3, "Lungs": 0.629, "Lymph_Nodes": 0.245, "Muscle": 0.145, "Oesophagus": 0.41, "Oral_Mucosa": 0.003, "Pancreas": 0.234, "Salivary_Glands": 0.002, "Skin": 0.163, "Small_Intestine": 0.007, "Spleen": 0.1, "Stomach": 0.277, "Thymus": 2.003, "Thyroid": 0.018, "Urinary_Bladder": 0.001, "Uterus": 0.002},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.193, "Adrenals": 0.308, "Brain": 0.001, "Breast": 2.796, "Colon": 0.009, "Endosteum": 0.28, "Extrathoracic_Airways": 0.007, "Gallbladder": 0.094, "Heart": 3.899, "Kidneys": 0.059, "Liver": 0.404, "Lungs": 0.834, "Lymph_Nodes": 0.327, "Muscle": 0.186, "Oesophagus": 0.594, "Oral_Mucosa": 0.004, "Pancreas": 0.343, "Salivary_Glands": 0.004, "Skin": 0.18, "Small_Intestine": 0.011, "Spleen": 0.151, "Stomach": 0.378, "Thymus": 2.414, "Thyroid": 0.028, "Urinary_Bladder": 0.002, "Uterus": 0.003},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.22, "Adrenals": 0.388, "Brain": 0.001, "Breast": 2.922, "Colon": 0.012, "Endosteum": 0.302, "Extrathoracic_Airways": 0.011, "Gallbladder": 0.119, "Heart": 4.36, "Kidneys": 0.075, "Liver": 0.47, "Lungs": 0.956, "Lymph_Nodes": 0.379, "Muscle": 0.211, "Oesophagus": 0.723, "Oral_Mucosa": 0.006, "Pancreas": 0.421, "Salivary_Glands": 0.006, "Skin": 0.189, "Small_Intestine": 0.015, "Spleen": 0.191, "Stomach": 0.441, "Thymus": 2.619, "Thyroid": 0.034, "Urinary_Bladder": 0.003, "Uterus": 0.004},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.179, "Adrenals": 0.286, "Brain": 0.001, "Breast": 2.627, "Colon": 0.008, "Endosteum": 0.253, "Extrathoracic_Airways": 0.007, "Gallbladder": 0.087, "Heart": 3.505, "Kidneys": 0.054, "Liver": 0.365, "Lungs": 0.745, "Lymph_Nodes": 0.294, "Muscle": 0.169, "Oesophagus": 0.536, "Oral_Mucosa": 0.004, "Ovaries": 0.001, "Pancreas": 0.306, "Salivary_Glands": 0.004, "Skin": 0.172, "Small_Intestine": 0.011, "Spleen": 0.138, "Stomach": 0.333, "Testicles": 0.001, "Thymus": 2.22, "Thyroid": 0.023, "Urinary_Bladder": 0.001, "Uterus": 0.003},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.22, "Adrenals": 0.399, "Brain": 0.001, "Breast": 2.862, "Colon": 0.013, "Endosteum": 0.288, "Extrathoracic_Airways": 0.011, "Gallbladder": 0.123, "Heart": 4.225, "Kidneys": 0.075, "Liver": 0.465, "Lungs": 0.931, "Lymph_Nodes": 0.37, "Muscle": 0.207, "Oesophagus": 0.721, "Oral_Mucosa": 0.005, "Ovaries": 0.001, "Pancreas": 0.414, "Salivary_Glands": 0.006, "Skin": 0.187, "Small_Intestine": 0.015, "Spleen": 0.191, "Stomach": 0.423, "Testicles": 0.001, "Thymus": 2.578, "Thyroid": 0.033, "Urinary_Bladder": 0.002, "Uterus": 0.004},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.245, "Adrenals": 0.48, "Brain": 0.001, "Breast": 2.942, "Colon": 0.016, "Endosteum": 0.301, "Extrathoracic_Airways": 0.015, "Gallbladder": 0.149, "Heart": 4.572, "Kidneys": 0.09, "Liver": 0.523, "Lungs": 1.029, "Lymph_Nodes": 0.413, "Muscle": 0.228, "Oesophagus": 0.841, "Oral_Mucosa": 0.007, "Ovaries": 0.002, "Pancreas": 0.481, "Salivary_Glands": 0.008, "Skin": 0.194, "Small_Intestine": 0.019, "Spleen": 0.228, "Stomach": 0.47, "Testicles": 0.001, "Thymus": 2.744, "Thyroid": 0.038, "Urinary_Bladder": 0.002, "Uterus": 0.006},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.257, "Adrenals": 0.508, "Brain": 0.001, "Breast": 2.931, "Colon": 0.018, "Endosteum": 0.289, "Extrathoracic_Airways": 0.015, "Gallbladder": 0.159, "Heart": 4.636, "Kidneys": 0.101, "Liver": 0.542, "Lungs": 1.056, "Lymph_Nodes": 0.424, "Muscle": 0.234, "Oesophagus": 0.905, "Oral_Mucosa": 0.007, "Ovaries": 0.005, "Pancreas": 0.494, "Salivary_Glands": 0.009, "Skin": 0.196, "Small_Intestine": 0.022, "Spleen": 0.24, "Stomach": 0.475, "Testicles": 0.001, "Thymus": 2.725, "Thyroid": 0.041, "Urinary_Bladder": 0.002, "Uterus": 0.006},
            "default": {"Active_Bone_Marrow": 0.193, "Adrenals": 0.308, "Brain": 0.001, "Breast": 2.796, "Colon": 0.009, "Endosteum": 0.28, "Extrathoracic_Airways": 0.007, "Gallbladder": 0.094, "Heart": 3.899, "Kidneys": 0.059, "Liver": 0.404, "Lungs": 0.834, "Lymph_Nodes": 0.327, "Muscle": 0.186, "Oesophagus": 0.594, "Oral_Mucosa": 0.004, "Pancreas": 0.343, "Salivary_Glands": 0.004, "Skin": 0.18, "Small_Intestine": 0.011, "Spleen": 0.151, "Stomach": 0.378, "Thymus": 2.414, "Thyroid": 0.028, "Urinary_Bladder": 0.002, "Uterus": 0.003},
        },
    },

    # ── Coronary Angioplasty ──
    "Coronary Angioplasty": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.202, "Adrenals": 0.407, "Brain": 0.001, "Breast": 2.174, "Colon": 0.014, "Endosteum": 0.284, "Extrathoracic_Airways": 0.01, "Gallbladder": 0.125, "Heart": 2.397, "Kidneys": 0.074, "Liver": 0.549, "Lungs": 1.005, "Lymph_Nodes": 0.31, "Muscle": 0.181, "Oesophagus": 0.655, "Oral_Mucosa": 0.005, "Pancreas": 0.455, "Salivary_Glands": 0.008, "Skin": 0.174, "Small_Intestine": 0.019, "Spleen": 0.195, "Stomach": 0.554, "Thymus": 2.636, "Thyroid": 0.034, "Uterus": 0.002},
            "default": {"Active_Bone_Marrow": 0.202, "Adrenals": 0.407, "Brain": 0.001, "Breast": 2.174, "Colon": 0.014, "Endosteum": 0.284, "Extrathoracic_Airways": 0.01, "Gallbladder": 0.125, "Heart": 2.397, "Kidneys": 0.074, "Liver": 0.549, "Lungs": 1.005, "Lymph_Nodes": 0.31, "Muscle": 0.181, "Oesophagus": 0.655, "Oral_Mucosa": 0.005, "Pancreas": 0.455, "Salivary_Glands": 0.008, "Skin": 0.174, "Small_Intestine": 0.019, "Spleen": 0.195, "Stomach": 0.554, "Thymus": 2.636, "Thyroid": 0.034, "Uterus": 0.002},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.239, "Adrenals": 0.519, "Brain": 0.001, "Breast": 2.331, "Colon": 0.019, "Endosteum": 0.313, "Extrathoracic_Airways": 0.015, "Gallbladder": 0.165, "Heart": 2.797, "Kidneys": 0.098, "Liver": 0.658, "Lungs": 1.181, "Lymph_Nodes": 0.372, "Muscle": 0.213, "Oesophagus": 0.824, "Oral_Mucosa": 0.007, "Pancreas": 0.562, "Salivary_Glands": 0.012, "Skin": 0.188, "Small_Intestine": 0.025, "Spleen": 0.245, "Stomach": 0.65, "Thymus": 2.898, "Thyroid": 0.046, "Urinary_Bladder": 0.001, "Uterus": 0.003},
            "default": {"Active_Bone_Marrow": 0.239, "Adrenals": 0.519, "Brain": 0.001, "Breast": 2.331, "Colon": 0.019, "Endosteum": 0.313, "Extrathoracic_Airways": 0.015, "Gallbladder": 0.165, "Heart": 2.797, "Kidneys": 0.098, "Liver": 0.658, "Lungs": 1.181, "Lymph_Nodes": 0.372, "Muscle": 0.213, "Oesophagus": 0.824, "Oral_Mucosa": 0.007, "Pancreas": 0.562, "Salivary_Glands": 0.012, "Skin": 0.188, "Small_Intestine": 0.025, "Spleen": 0.245, "Stomach": 0.65, "Thymus": 2.898, "Thyroid": 0.046, "Urinary_Bladder": 0.001, "Uterus": 0.003},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.119, "Adrenals": 0.168, "Breast": 1.791, "Colon": 0.004, "Endosteum": 0.211, "Gallbladder": 0.038, "Heart": 1.475, "Kidneys": 0.022, "Liver": 0.301, "Lungs": 0.607, "Lymph_Nodes": 0.167, "Muscle": 0.108, "Oesophagus": 0.271, "Oral_Mucosa": 0.001, "Pancreas": 0.217, "Salivary_Glands": 0.001, "Skin": 0.145, "Small_Intestine": 0.004, "Spleen": 0.081, "Stomach": 0.334, "Thymus": 2.05, "Thyroid": 0.008},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.153, "Adrenals": 0.277, "Breast": 2.063, "Colon": 0.007, "Endosteum": 0.262, "Extrathoracic_Airways": 0.001, "Gallbladder": 0.061, "Heart": 1.971, "Kidneys": 0.038, "Liver": 0.419, "Lungs": 0.824, "Lymph_Nodes": 0.235, "Muscle": 0.145, "Oesophagus": 0.415, "Oral_Mucosa": 0.001, "Pancreas": 0.335, "Salivary_Glands": 0.002, "Skin": 0.16, "Small_Intestine": 0.008, "Spleen": 0.129, "Stomach": 0.451, "Thymus": 2.54, "Thyroid": 0.015},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.177, "Adrenals": 0.366, "Breast": 2.219, "Colon": 0.011, "Endosteum": 0.294, "Extrathoracic_Airways": 0.001, "Gallbladder": 0.077, "Heart": 2.293, "Kidneys": 0.053, "Liver": 0.5, "Lungs": 0.965, "Lymph_Nodes": 0.283, "Muscle": 0.169, "Oesophagus": 0.528, "Oral_Mucosa": 0.001, "Pancreas": 0.424, "Salivary_Glands": 0.003, "Skin": 0.171, "Small_Intestine": 0.011, "Spleen": 0.168, "Stomach": 0.525, "Thymus": 2.741, "Thyroid": 0.023},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.138, "Adrenals": 0.222, "Breast": 1.911, "Colon": 0.006, "Endosteum": 0.235, "Extrathoracic_Airways": 0.001, "Gallbladder": 0.054, "Heart": 1.722, "Kidneys": 0.033, "Liver": 0.364, "Lungs": 0.71, "Lymph_Nodes": 0.203, "Muscle": 0.127, "Oesophagus": 0.364, "Oral_Mucosa": 0.001, "Pancreas": 0.277, "Salivary_Glands": 0.003, "Skin": 0.152, "Small_Intestine": 0.007, "Spleen": 0.109, "Stomach": 0.391, "Thymus": 2.208, "Thyroid": 0.015},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.177, "Adrenals": 0.341, "Breast": 2.186, "Colon": 0.01, "Endosteum": 0.289, "Extrathoracic_Airways": 0.002, "Gallbladder": 0.084, "Heart": 2.258, "Kidneys": 0.055, "Liver": 0.496, "Lungs": 0.941, "Lymph_Nodes": 0.279, "Muscle": 0.166, "Oesophagus": 0.546, "Oral_Mucosa": 0.002, "Pancreas": 0.407, "Salivary_Glands": 0.005, "Skin": 0.169, "Small_Intestine": 0.012, "Spleen": 0.166, "Stomach": 0.518, "Thymus": 2.636, "Thyroid": 0.026},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.205, "Adrenals": 0.422, "Breast": 2.335, "Colon": 0.013, "Endosteum": 0.321, "Extrathoracic_Airways": 0.004, "Gallbladder": 0.108, "Heart": 2.592, "Kidneys": 0.073, "Liver": 0.584, "Lungs": 1.08, "Lymph_Nodes": 0.329, "Muscle": 0.192, "Oesophagus": 0.688, "Oral_Mucosa": 0.003, "Pancreas": 0.493, "Salivary_Glands": 0.008, "Skin": 0.179, "Small_Intestine": 0.016, "Spleen": 0.208, "Stomach": 0.596, "Thymus": 2.778, "Thyroid": 0.037, "Uterus": 0.001},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.156, "Adrenals": 0.27, "Breast": 2.005, "Colon": 0.008, "Endosteum": 0.254, "Extrathoracic_Airways": 0.003, "Gallbladder": 0.072, "Heart": 1.934, "Kidneys": 0.044, "Liver": 0.42, "Lungs": 0.8, "Lymph_Nodes": 0.235, "Muscle": 0.143, "Oesophagus": 0.451, "Oral_Mucosa": 0.002, "Pancreas": 0.328, "Salivary_Glands": 0.004, "Skin": 0.159, "Small_Intestine": 0.01, "Spleen": 0.136, "Stomach": 0.444, "Thymus": 2.335, "Thyroid": 0.02},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.392, "Breast": 2.263, "Colon": 0.013, "Endosteum": 0.306, "Extrathoracic_Airways": 0.005, "Gallbladder": 0.109, "Heart": 2.472, "Kidneys": 0.068, "Liver": 0.556, "Lungs": 1.029, "Lymph_Nodes": 0.312, "Muscle": 0.183, "Oesophagus": 0.65, "Oral_Mucosa": 0.003, "Pancreas": 0.46, "Salivary_Glands": 0.007, "Skin": 0.176, "Small_Intestine": 0.016, "Spleen": 0.198, "Stomach": 0.573, "Thymus": 2.718, "Thyroid": 0.032, "Uterus": 0.001},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.226, "Adrenals": 0.466, "Brain": 0.001, "Breast": 2.386, "Colon": 0.017, "Endosteum": 0.333, "Extrathoracic_Airways": 0.007, "Gallbladder": 0.139, "Heart": 2.78, "Kidneys": 0.086, "Liver": 0.64, "Lungs": 1.157, "Lymph_Nodes": 0.36, "Muscle": 0.207, "Oesophagus": 0.789, "Oral_Mucosa": 0.005, "Pancreas": 0.537, "Salivary_Glands": 0.01, "Skin": 0.186, "Small_Intestine": 0.021, "Spleen": 0.241, "Stomach": 0.649, "Thymus": 2.844, "Thyroid": 0.041, "Uterus": 0.001},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.171, "Adrenals": 0.311, "Breast": 2.07, "Colon": 0.011, "Endosteum": 0.267, "Extrathoracic_Airways": 0.005, "Gallbladder": 0.09, "Heart": 2.1, "Kidneys": 0.053, "Liver": 0.464, "Lungs": 0.871, "Lymph_Nodes": 0.26, "Muscle": 0.156, "Oesophagus": 0.52, "Oral_Mucosa": 0.003, "Pancreas": 0.371, "Salivary_Glands": 0.005, "Skin": 0.164, "Small_Intestine": 0.013, "Spleen": 0.157, "Stomach": 0.484, "Thymus": 2.444, "Thyroid": 0.025, "Uterus": 0.001},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.214, "Adrenals": 0.431, "Brain": 0.001, "Breast": 2.302, "Colon": 0.016, "Endosteum": 0.314, "Extrathoracic_Airways": 0.008, "Gallbladder": 0.131, "Heart": 2.614, "Kidneys": 0.078, "Liver": 0.597, "Lungs": 1.09, "Lymph_Nodes": 0.336, "Muscle": 0.195, "Oesophagus": 0.716, "Oral_Mucosa": 0.005, "Pancreas": 0.498, "Salivary_Glands": 0.008, "Skin": 0.181, "Small_Intestine": 0.019, "Spleen": 0.218, "Stomach": 0.607, "Thymus": 2.799, "Thyroid": 0.036, "Uterus": 0.001},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.24, "Adrenals": 0.502, "Brain": 0.001, "Breast": 2.398, "Colon": 0.02, "Endosteum": 0.335, "Extrathoracic_Airways": 0.012, "Gallbladder": 0.163, "Heart": 2.886, "Kidneys": 0.095, "Liver": 0.674, "Lungs": 1.204, "Lymph_Nodes": 0.379, "Muscle": 0.217, "Oesophagus": 0.841, "Oral_Mucosa": 0.007, "Pancreas": 0.568, "Salivary_Glands": 0.011, "Skin": 0.19, "Small_Intestine": 0.025, "Spleen": 0.257, "Stomach": 0.674, "Thymus": 2.928, "Thyroid": 0.044, "Uterus": 0.001},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.183, "Adrenals": 0.345, "Breast": 2.117, "Colon": 0.012, "Endosteum": 0.276, "Extrathoracic_Airways": 0.007, "Gallbladder": 0.104, "Heart": 2.226, "Kidneys": 0.061, "Liver": 0.499, "Lungs": 0.926, "Lymph_Nodes": 0.28, "Muscle": 0.166, "Oesophagus": 0.573, "Oral_Mucosa": 0.004, "Pancreas": 0.404, "Salivary_Glands": 0.006, "Skin": 0.169, "Small_Intestine": 0.015, "Spleen": 0.172, "Stomach": 0.515, "Thymus": 2.532, "Thyroid": 0.029, "Uterus": 0.001},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.224, "Adrenals": 0.46, "Brain": 0.001, "Breast": 2.322, "Colon": 0.017, "Endosteum": 0.317, "Extrathoracic_Airways": 0.012, "Gallbladder": 0.147, "Heart": 2.704, "Kidneys": 0.086, "Liver": 0.625, "Lungs": 1.131, "Lymph_Nodes": 0.353, "Muscle": 0.203, "Oesophagus": 0.762, "Oral_Mucosa": 0.006, "Pancreas": 0.525, "Salivary_Glands": 0.009, "Skin": 0.184, "Small_Intestine": 0.022, "Spleen": 0.23, "Stomach": 0.63, "Thymus": 2.86, "Thyroid": 0.041, "Uterus": 0.002},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.249, "Adrenals": 0.526, "Brain": 0.001, "Breast": 2.396, "Colon": 0.021, "Endosteum": 0.332, "Extrathoracic_Airways": 0.017, "Gallbladder": 0.179, "Heart": 2.942, "Kidneys": 0.103, "Liver": 0.693, "Lungs": 1.233, "Lymph_Nodes": 0.392, "Muscle": 0.223, "Oesophagus": 0.874, "Oral_Mucosa": 0.008, "Pancreas": 0.589, "Salivary_Glands": 0.011, "Skin": 0.192, "Small_Intestine": 0.027, "Spleen": 0.264, "Stomach": 0.69, "Thymus": 2.987, "Thyroid": 0.048, "Urinary_Bladder": 0.001, "Uterus": 0.003},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.261, "Adrenals": 0.588, "Brain": 0.001, "Breast": 2.374, "Colon": 0.022, "Endosteum": 0.319, "Extrathoracic_Airways": 0.019, "Gallbladder": 0.193, "Heart": 2.978, "Kidneys": 0.114, "Liver": 0.714, "Lungs": 1.265, "Lymph_Nodes": 0.405, "Muscle": 0.229, "Oesophagus": 0.92, "Oral_Mucosa": 0.008, "Pancreas": 0.618, "Salivary_Glands": 0.014, "Skin": 0.194, "Small_Intestine": 0.03, "Spleen": 0.272, "Stomach": 0.694, "Thymus": 2.993, "Thyroid": 0.052, "Urinary_Bladder": 0.001, "Uterus": 0.005},
            "default": {"Active_Bone_Marrow": 0.198, "Adrenals": 0.392, "Breast": 2.263, "Colon": 0.013, "Endosteum": 0.306, "Extrathoracic_Airways": 0.005, "Gallbladder": 0.109, "Heart": 2.472, "Kidneys": 0.068, "Liver": 0.556, "Lungs": 1.029, "Lymph_Nodes": 0.312, "Muscle": 0.183, "Oesophagus": 0.65, "Oral_Mucosa": 0.003, "Pancreas": 0.46, "Salivary_Glands": 0.007, "Skin": 0.176, "Small_Intestine": 0.016, "Spleen": 0.198, "Stomach": 0.573, "Thymus": 2.718, "Thyroid": 0.032, "Uterus": 0.001},
        },
    },

    # ── Carotid Angiography ──
    "Carotid Angiography": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.153, "Adrenals": 0.003, "Brain": 0.053, "Breast": 0.009, "Endosteum": 0.485, "Extrathoracic_Airways": 5.265, "Gallbladder": 0.001, "Heart": 0.014, "Liver": 0.003, "Lungs": 0.061, "Lymph_Nodes": 1.45, "Muscle": 0.143, "Oesophagus": 0.161, "Oral_Mucosa": 0.504, "Pancreas": 0.001, "Salivary_Glands": 1.488, "Skin": 0.235, "Spleen": 0.003, "Stomach": 0.002, "Thymus": 0.065, "Thyroid": 12.518},
            "default": {"Active_Bone_Marrow": 0.153, "Adrenals": 0.003, "Brain": 0.053, "Breast": 0.009, "Endosteum": 0.485, "Extrathoracic_Airways": 5.265, "Gallbladder": 0.001, "Heart": 0.014, "Liver": 0.003, "Lungs": 0.061, "Lymph_Nodes": 1.45, "Muscle": 0.143, "Oesophagus": 0.161, "Oral_Mucosa": 0.504, "Pancreas": 0.001, "Salivary_Glands": 1.488, "Skin": 0.235, "Spleen": 0.003, "Stomach": 0.002, "Thymus": 0.065, "Thyroid": 12.518},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.189, "Adrenals": 0.004, "Brain": 0.071, "Breast": 0.012, "Endosteum": 0.554, "Extrathoracic_Airways": 5.828, "Gallbladder": 0.002, "Heart": 0.019, "Kidneys": 0.001, "Liver": 0.004, "Lungs": 0.077, "Lymph_Nodes": 1.591, "Muscle": 0.161, "Oesophagus": 0.207, "Oral_Mucosa": 0.616, "Pancreas": 0.002, "Salivary_Glands": 1.747, "Skin": 0.253, "Spleen": 0.004, "Stomach": 0.003, "Thymus": 0.086, "Thyroid": 13.36},
            "default": {"Active_Bone_Marrow": 0.189, "Adrenals": 0.004, "Brain": 0.071, "Breast": 0.012, "Endosteum": 0.554, "Extrathoracic_Airways": 5.828, "Gallbladder": 0.002, "Heart": 0.019, "Kidneys": 0.001, "Liver": 0.004, "Lungs": 0.077, "Lymph_Nodes": 1.591, "Muscle": 0.161, "Oesophagus": 0.207, "Oral_Mucosa": 0.616, "Pancreas": 0.002, "Salivary_Glands": 1.747, "Skin": 0.253, "Spleen": 0.004, "Stomach": 0.003, "Thymus": 0.086, "Thyroid": 13.36},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.074, "Brain": 0.014, "Breast": 0.003, "Endosteum": 0.315, "Extrathoracic_Airways": 3.978, "Heart": 0.004, "Lungs": 0.025, "Lymph_Nodes": 1.127, "Muscle": 0.101, "Oesophagus": 0.061, "Oral_Mucosa": 0.257, "Salivary_Glands": 0.897, "Skin": 0.194, "Spleen": 0.001, "Thymus": 0.019, "Thyroid": 10.544},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.107, "Brain": 0.024, "Breast": 0.004, "Endosteum": 0.435, "Extrathoracic_Airways": 4.761, "Heart": 0.008, "Liver": 0.001, "Lungs": 0.039, "Lymph_Nodes": 1.331, "Muscle": 0.125, "Oesophagus": 0.093, "Oral_Mucosa": 0.377, "Salivary_Glands": 1.223, "Skin": 0.219, "Spleen": 0.002, "Thymus": 0.034, "Thyroid": 11.987},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.131, "Brain": 0.033, "Breast": 0.006, "Endosteum": 0.514, "Extrathoracic_Airways": 5.174, "Heart": 0.011, "Liver": 0.001, "Lungs": 0.051, "Lymph_Nodes": 1.439, "Muscle": 0.139, "Oesophagus": 0.117, "Oral_Mucosa": 0.457, "Salivary_Glands": 1.434, "Skin": 0.234, "Spleen": 0.003, "Thymus": 0.05, "Thyroid": 12.684},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.092, "Brain": 0.021, "Breast": 0.004, "Endosteum": 0.371, "Extrathoracic_Airways": 4.313, "Heart": 0.006, "Liver": 0.001, "Lungs": 0.034, "Lymph_Nodes": 1.213, "Muscle": 0.112, "Oesophagus": 0.08, "Oral_Mucosa": 0.315, "Salivary_Glands": 1.055, "Skin": 0.205, "Spleen": 0.002, "Thymus": 0.03, "Thyroid": 11.107},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.131, "Brain": 0.035, "Breast": 0.007, "Endosteum": 0.496, "Extrathoracic_Airways": 5.092, "Heart": 0.01, "Liver": 0.001, "Lungs": 0.051, "Lymph_Nodes": 1.416, "Muscle": 0.137, "Oesophagus": 0.119, "Oral_Mucosa": 0.444, "Salivary_Glands": 1.402, "Skin": 0.231, "Spleen": 0.003, "Thymus": 0.052, "Thyroid": 12.473},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.158, "Brain": 0.047, "Breast": 0.009, "Endosteum": 0.57, "Extrathoracic_Airways": 5.494, "Gallbladder": 0.001, "Heart": 0.014, "Liver": 0.002, "Lungs": 0.064, "Lymph_Nodes": 1.518, "Muscle": 0.151, "Oesophagus": 0.148, "Oral_Mucosa": 0.524, "Salivary_Glands": 1.613, "Skin": 0.245, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.07, "Thyroid": 13.083},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.11, "Brain": 0.029, "Breast": 0.005, "Endosteum": 0.414, "Extrathoracic_Airways": 4.614, "Heart": 0.008, "Liver": 0.001, "Lungs": 0.042, "Lymph_Nodes": 1.288, "Muscle": 0.122, "Oesophagus": 0.102, "Oral_Mucosa": 0.374, "Salivary_Glands": 1.187, "Skin": 0.215, "Spleen": 0.002, "Thymus": 0.04, "Thyroid": 11.553},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.15, "Brain": 0.046, "Breast": 0.008, "Endosteum": 0.534, "Extrathoracic_Airways": 5.387, "Gallbladder": 0.001, "Heart": 0.013, "Liver": 0.002, "Lungs": 0.061, "Lymph_Nodes": 1.484, "Muscle": 0.146, "Oesophagus": 0.148, "Oral_Mucosa": 0.513, "Pancreas": 0.001, "Salivary_Glands": 1.531, "Skin": 0.24, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.062, "Thyroid": 12.816},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.177, "Adrenals": 0.001, "Brain": 0.06, "Breast": 0.011, "Endosteum": 0.594, "Extrathoracic_Airways": 5.787, "Gallbladder": 0.001, "Heart": 0.016, "Liver": 0.002, "Lungs": 0.074, "Lymph_Nodes": 1.581, "Muscle": 0.159, "Oesophagus": 0.183, "Oral_Mucosa": 0.601, "Pancreas": 0.001, "Salivary_Glands": 1.721, "Skin": 0.252, "Spleen": 0.004, "Stomach": 0.001, "Thymus": 0.079, "Thyroid": 13.345},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.124, "Adrenals": 0.001, "Brain": 0.037, "Breast": 0.007, "Endosteum": 0.445, "Extrathoracic_Airways": 4.851, "Gallbladder": 0.001, "Heart": 0.01, "Liver": 0.002, "Lungs": 0.049, "Lymph_Nodes": 1.347, "Muscle": 0.129, "Oesophagus": 0.122, "Oral_Mucosa": 0.421, "Pancreas": 0.001, "Salivary_Glands": 1.29, "Skin": 0.222, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.047, "Thyroid": 11.915},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.165, "Adrenals": 0.002, "Brain": 0.055, "Breast": 0.01, "Endosteum": 0.552, "Extrathoracic_Airways": 5.584, "Gallbladder": 0.001, "Heart": 0.015, "Liver": 0.002, "Lungs": 0.068, "Lymph_Nodes": 1.532, "Muscle": 0.152, "Oesophagus": 0.173, "Oral_Mucosa": 0.558, "Pancreas": 0.001, "Salivary_Glands": 1.614, "Skin": 0.245, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.07, "Thyroid": 13.083},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.19, "Adrenals": 0.003, "Brain": 0.069, "Breast": 0.012, "Endosteum": 0.599, "Extrathoracic_Airways": 5.95, "Gallbladder": 0.001, "Heart": 0.018, "Kidneys": 0.001, "Liver": 0.003, "Lungs": 0.08, "Lymph_Nodes": 1.62, "Muscle": 0.164, "Oesophagus": 0.21, "Oral_Mucosa": 0.641, "Pancreas": 0.001, "Salivary_Glands": 1.779, "Skin": 0.256, "Spleen": 0.004, "Stomach": 0.002, "Thymus": 0.084, "Thyroid": 13.569},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.136, "Adrenals": 0.001, "Brain": 0.043, "Breast": 0.008, "Endosteum": 0.465, "Extrathoracic_Airways": 5.029, "Gallbladder": 0.001, "Heart": 0.012, "Liver": 0.002, "Lungs": 0.054, "Lymph_Nodes": 1.391, "Muscle": 0.135, "Oesophagus": 0.138, "Oral_Mucosa": 0.453, "Pancreas": 0.001, "Salivary_Glands": 1.37, "Skin": 0.227, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.055, "Thyroid": 12.19},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.175, "Adrenals": 0.003, "Brain": 0.062, "Breast": 0.011, "Endosteum": 0.559, "Extrathoracic_Airways": 5.707, "Gallbladder": 0.001, "Heart": 0.016, "Liver": 0.003, "Lungs": 0.072, "Lymph_Nodes": 1.562, "Muscle": 0.157, "Oesophagus": 0.188, "Oral_Mucosa": 0.581, "Pancreas": 0.002, "Salivary_Glands": 1.672, "Skin": 0.249, "Spleen": 0.004, "Stomach": 0.002, "Thymus": 0.077, "Thyroid": 13.255},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.199, "Adrenals": 0.004, "Brain": 0.076, "Breast": 0.013, "Endosteum": 0.593, "Extrathoracic_Airways": 6.034, "Gallbladder": 0.001, "Heart": 0.019, "Kidneys": 0.001, "Liver": 0.004, "Lungs": 0.083, "Lymph_Nodes": 1.641, "Muscle": 0.167, "Oesophagus": 0.224, "Oral_Mucosa": 0.652, "Pancreas": 0.002, "Salivary_Glands": 1.818, "Skin": 0.258, "Spleen": 0.004, "Stomach": 0.002, "Thymus": 0.091, "Thyroid": 13.688},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.209, "Adrenals": 0.006, "Brain": 0.084, "Breast": 0.015, "Endosteum": 0.568, "Extrathoracic_Airways": 6.077, "Gallbladder": 0.002, "Heart": 0.022, "Kidneys": 0.001, "Liver": 0.005, "Lungs": 0.087, "Lymph_Nodes": 1.65, "Muscle": 0.17, "Oesophagus": 0.238, "Oral_Mucosa": 0.676, "Pancreas": 0.003, "Salivary_Glands": 1.868, "Skin": 0.26, "Spleen": 0.005, "Stomach": 0.003, "Thymus": 0.098, "Thyroid": 13.656},
            "default": {"Active_Bone_Marrow": 0.15, "Brain": 0.046, "Breast": 0.008, "Endosteum": 0.534, "Extrathoracic_Airways": 5.387, "Gallbladder": 0.001, "Heart": 0.013, "Liver": 0.002, "Lungs": 0.061, "Lymph_Nodes": 1.484, "Muscle": 0.146, "Oesophagus": 0.148, "Oral_Mucosa": 0.513, "Pancreas": 0.001, "Salivary_Glands": 1.531, "Skin": 0.24, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.062, "Thyroid": 12.816},
        },
    },

    # ── ERCP ──
    "ERCP": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
            "default": {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
            "default": {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.08, "Adrenals": 0.217, "Breast": 0.024, "Colon": 0.105, "Endosteum": 0.13, "Gallbladder": 0.728, "Heart": 0.141, "Kidneys": 0.218, "Liver": 0.295, "Lungs": 0.055, "Lymph_Nodes": 0.375, "Muscle": 0.147, "Oesophagus": 0.283, "Ovaries": 0.01, "Pancreas": 1.013, "Skin": 0.108, "Small_Intestine": 0.1, "Spleen": 0.429, "Stomach": 3.219, "Thymus": 0.008, "Urinary_Bladder": 0.002, "Uterus": 0.009},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.34, "Breast": 0.036, "Colon": 0.161, "Endosteum": 0.165, "Gallbladder": 1.036, "Heart": 0.22, "Kidneys": 0.354, "Liver": 0.419, "Lungs": 0.086, "Lymph_Nodes": 0.532, "Muscle": 0.193, "Oesophagus": 0.426, "Ovaries": 0.02, "Pancreas": 1.491, "Skin": 0.118, "Small_Intestine": 0.157, "Spleen": 0.643, "Stomach": 4.129, "Thymus": 0.015, "Urinary_Bladder": 0.003, "Uterus": 0.015},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.452, "Breast": 0.044, "Colon": 0.205, "Endosteum": 0.19, "Gallbladder": 1.233, "Heart": 0.282, "Kidneys": 0.466, "Liver": 0.509, "Lungs": 0.11, "Lymph_Nodes": 0.643, "Muscle": 0.223, "Oesophagus": 0.531, "Ovaries": 0.034, "Pancreas": 1.845, "Prostate": 0.001, "Skin": 0.124, "Small_Intestine": 0.204, "Spleen": 0.797, "Stomach": 4.672, "Thymus": 0.02, "Thyroid": 0.001, "Urinary_Bladder": 0.005, "Uterus": 0.023},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.304, "Breast": 0.032, "Colon": 0.138, "Endosteum": 0.148, "Gallbladder": 0.867, "Heart": 0.185, "Kidneys": 0.297, "Liver": 0.362, "Lungs": 0.074, "Lymph_Nodes": 0.456, "Muscle": 0.17, "Oesophagus": 0.358, "Ovaries": 0.024, "Pancreas": 1.273, "Prostate": 0.001, "Skin": 0.113, "Small_Intestine": 0.136, "Spleen": 0.545, "Stomach": 3.638, "Thymus": 0.013, "Thyroid": 0.001, "Urinary_Bladder": 0.004, "Uterus": 0.015},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.131, "Adrenals": 0.468, "Breast": 0.046, "Colon": 0.205, "Endosteum": 0.188, "Gallbladder": 1.184, "Heart": 0.277, "Kidneys": 0.458, "Liver": 0.502, "Lungs": 0.11, "Lymph_Nodes": 0.631, "Muscle": 0.221, "Oesophagus": 0.517, "Ovaries": 0.045, "Pancreas": 1.817, "Prostate": 0.003, "Skin": 0.124, "Small_Intestine": 0.206, "Spleen": 0.789, "Stomach": 4.582, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.007, "Uterus": 0.025},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.157, "Adrenals": 0.607, "Breast": 0.057, "Colon": 0.255, "Endosteum": 0.214, "Gallbladder": 1.367, "Heart": 0.342, "Kidneys": 0.579, "Liver": 0.598, "Lungs": 0.138, "Lymph_Nodes": 0.746, "Muscle": 0.253, "Oesophagus": 0.624, "Ovaries": 0.07, "Pancreas": 2.192, "Prostate": 0.005, "Skin": 0.132, "Small_Intestine": 0.261, "Spleen": 0.957, "Stomach": 5.12, "Thymus": 0.03, "Thyroid": 0.002, "Urinary_Bladder": 0.01, "Uterus": 0.035},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.114, "Adrenals": 0.388, "Breast": 0.039, "Colon": 0.169, "Endosteum": 0.164, "Gallbladder": 0.99, "Heart": 0.225, "Kidneys": 0.375, "Liver": 0.422, "Lungs": 0.091, "Lymph_Nodes": 0.53, "Muscle": 0.191, "Oesophagus": 0.423, "Ovaries": 0.038, "Pancreas": 1.51, "Prostate": 0.004, "Skin": 0.118, "Small_Intestine": 0.171, "Spleen": 0.653, "Stomach": 3.989, "Thymus": 0.018, "Thyroid": 0.001, "Urinary_Bladder": 0.006, "Uterus": 0.021},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.18, "Adrenals": 0.715, "Breast": 0.065, "Colon": 0.288, "Endosteum": 0.229, "Gallbladder": 1.479, "Heart": 0.383, "Kidneys": 0.678, "Liver": 0.66, "Lungs": 0.157, "Lymph_Nodes": 0.821, "Muscle": 0.273, "Oesophagus": 0.683, "Ovaries": 0.09, "Pancreas": 2.445, "Prostate": 0.014, "Skin": 0.137, "Small_Intestine": 0.308, "Spleen": 1.074, "Stomach": 5.394, "Thymus": 0.037, "Thyroid": 0.002, "Urinary_Bladder": 0.015, "Uterus": 0.043},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.467, "Breast": 0.045, "Colon": 0.193, "Endosteum": 0.176, "Gallbladder": 1.095, "Heart": 0.257, "Kidneys": 0.443, "Liver": 0.472, "Lungs": 0.105, "Lymph_Nodes": 0.591, "Muscle": 0.208, "Oesophagus": 0.474, "Ovaries": 0.05, "Pancreas": 1.709, "Prostate": 0.007, "Skin": 0.122, "Small_Intestine": 0.201, "Spleen": 0.739, "Stomach": 4.257, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.008, "Uterus": 0.026},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.169, "Adrenals": 0.666, "Breast": 0.061, "Colon": 0.264, "Endosteum": 0.215, "Gallbladder": 1.404, "Heart": 0.352, "Kidneys": 0.625, "Liver": 0.616, "Lungs": 0.146, "Lymph_Nodes": 0.768, "Muscle": 0.258, "Oesophagus": 0.631, "Ovaries": 0.08, "Pancreas": 2.276, "Prostate": 0.012, "Skin": 0.133, "Small_Intestine": 0.281, "Spleen": 0.993, "Stomach": 5.119, "Testicles": 0.001, "Thymus": 0.033, "Thyroid": 0.001, "Urinary_Bladder": 0.014, "Uterus": 0.039},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.196, "Adrenals": 0.81, "Breast": 0.071, "Colon": 0.309, "Endosteum": 0.235, "Gallbladder": 1.571, "Heart": 0.408, "Kidneys": 0.747, "Liver": 0.701, "Lungs": 0.172, "Lymph_Nodes": 0.872, "Muscle": 0.286, "Oesophagus": 0.718, "Ovaries": 0.105, "Pancreas": 2.62, "Prostate": 0.018, "Salivary_Glands": 0.001, "Skin": 0.14, "Small_Intestine": 0.335, "Spleen": 1.144, "Stomach": 5.536, "Testicles": 0.001, "Thymus": 0.041, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.049},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.141, "Adrenals": 0.537, "Breast": 0.05, "Colon": 0.212, "Endosteum": 0.185, "Gallbladder": 1.182, "Heart": 0.282, "Kidneys": 0.498, "Liver": 0.512, "Lungs": 0.117, "Lymph_Nodes": 0.639, "Muscle": 0.221, "Oesophagus": 0.513, "Ovaries": 0.06, "Pancreas": 1.865, "Prostate": 0.009, "Skin": 0.125, "Small_Intestine": 0.223, "Spleen": 0.807, "Stomach": 4.461, "Thymus": 0.025, "Thyroid": 0.001, "Urinary_Bladder": 0.01, "Uterus": 0.031},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.181, "Adrenals": 0.742, "Breast": 0.066, "Colon": 0.28, "Endosteum": 0.22, "Gallbladder": 1.479, "Heart": 0.372, "Kidneys": 0.678, "Liver": 0.651, "Lungs": 0.156, "Lymph_Nodes": 0.808, "Muscle": 0.268, "Oesophagus": 0.659, "Ovaries": 0.09, "Pancreas": 2.409, "Prostate": 0.014, "Salivary_Glands": 0.001, "Skin": 0.136, "Small_Intestine": 0.301, "Spleen": 1.048, "Stomach": 5.254, "Testicles": 0.001, "Thymus": 0.036, "Thyroid": 0.001, "Urinary_Bladder": 0.015, "Uterus": 0.045},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.207, "Adrenals": 0.887, "Breast": 0.077, "Colon": 0.32, "Endosteum": 0.236, "Gallbladder": 1.641, "Heart": 0.422, "Kidneys": 0.795, "Liver": 0.73, "Lungs": 0.181, "Lymph_Nodes": 0.903, "Muscle": 0.293, "Oesophagus": 0.736, "Ovaries": 0.112, "Pancreas": 2.725, "Prostate": 0.02, "Salivary_Glands": 0.001, "Skin": 0.142, "Small_Intestine": 0.351, "Spleen": 1.186, "Stomach": 5.616, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.001, "Urinary_Bladder": 0.019, "Uterus": 0.055},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.222, "Adrenals": 0.952, "Breast": 0.081, "Colon": 0.33, "Endosteum": 0.232, "Gallbladder": 1.754, "Heart": 0.435, "Kidneys": 0.852, "Liver": 0.759, "Lungs": 0.189, "Lymph_Nodes": 0.939, "Muscle": 0.3, "Oesophagus": 0.765, "Ovaries": 0.112, "Pancreas": 2.839, "Prostate": 0.022, "Salivary_Glands": 0.001, "Skin": 0.145, "Small_Intestine": 0.368, "Spleen": 1.221, "Stomach": 5.642, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.002, "Urinary_Bladder": 0.022, "Uterus": 0.063},
            "default": {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
        },
    },

    # ── Peripheral Angiography ──
    "Peripheral Angiography": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.009, "Colon": 0.017, "Endosteum": 0.246, "Lymph_Nodes": 0.01, "Muscle": 0.111, "Prostate": 0.077, "Skin": 0.164, "Small_Intestine": 0.001, "Testicles": 0.428, "Urinary_Bladder": 0.025, "Uterus": 0.008},
            "default": {"Active_Bone_Marrow": 0.009, "Colon": 0.017, "Endosteum": 0.246, "Lymph_Nodes": 0.01, "Muscle": 0.111, "Prostate": 0.077, "Skin": 0.164, "Small_Intestine": 0.001, "Testicles": 0.428, "Urinary_Bladder": 0.025, "Uterus": 0.008},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.287, "Lymph_Nodes": 0.012, "Muscle": 0.126, "Prostate": 0.1, "Skin": 0.176, "Small_Intestine": 0.002, "Testicles": 0.475, "Urinary_Bladder": 0.032, "Uterus": 0.011},
            "default": {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.287, "Lymph_Nodes": 0.012, "Muscle": 0.126, "Prostate": 0.1, "Skin": 0.176, "Small_Intestine": 0.002, "Testicles": 0.475, "Urinary_Bladder": 0.032, "Uterus": 0.011},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.003, "Colon": 0.008, "Endosteum": 0.147, "Lymph_Nodes": 0.007, "Muscle": 0.078, "Prostate": 0.03, "Skin": 0.139, "Testicles": 0.31, "Urinary_Bladder": 0.01, "Uterus": 0.002},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.005, "Colon": 0.011, "Endosteum": 0.208, "Lymph_Nodes": 0.009, "Muscle": 0.096, "Prostate": 0.048, "Skin": 0.153, "Testicles": 0.365, "Urinary_Bladder": 0.013, "Uterus": 0.005},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.006, "Colon": 0.014, "Endosteum": 0.25, "Lymph_Nodes": 0.01, "Muscle": 0.107, "Prostate": 0.062, "Skin": 0.162, "Testicles": 0.392, "Urinary_Bladder": 0.015, "Uterus": 0.007},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.004, "Colon": 0.01, "Endosteum": 0.177, "Lymph_Nodes": 0.008, "Muscle": 0.087, "Prostate": 0.039, "Skin": 0.145, "Testicles": 0.341, "Urinary_Bladder": 0.012, "Uterus": 0.004},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.006, "Colon": 0.014, "Endosteum": 0.242, "Lymph_Nodes": 0.01, "Muscle": 0.106, "Prostate": 0.058, "Skin": 0.16, "Testicles": 0.405, "Urinary_Bladder": 0.017, "Uterus": 0.006},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.008, "Colon": 0.017, "Endosteum": 0.283, "Lymph_Nodes": 0.011, "Muscle": 0.117, "Prostate": 0.072, "Skin": 0.169, "Testicles": 0.447, "Urinary_Bladder": 0.021, "Uterus": 0.008},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.005, "Colon": 0.012, "Endosteum": 0.201, "Lymph_Nodes": 0.009, "Muscle": 0.094, "Prostate": 0.05, "Skin": 0.151, "Testicles": 0.378, "Urinary_Bladder": 0.015, "Uterus": 0.005},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.008, "Colon": 0.016, "Endosteum": 0.266, "Lymph_Nodes": 0.011, "Muscle": 0.113, "Prostate": 0.072, "Skin": 0.166, "Small_Intestine": 0.001, "Testicles": 0.454, "Urinary_Bladder": 0.022, "Uterus": 0.007},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.01, "Colon": 0.019, "Endosteum": 0.301, "Lymph_Nodes": 0.012, "Muscle": 0.124, "Prostate": 0.088, "Skin": 0.174, "Small_Intestine": 0.001, "Testicles": 0.509, "Urinary_Bladder": 0.027, "Uterus": 0.009},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.006, "Colon": 0.013, "Endosteum": 0.219, "Lymph_Nodes": 0.009, "Muscle": 0.1, "Prostate": 0.058, "Skin": 0.156, "Small_Intestine": 0.001, "Testicles": 0.402, "Urinary_Bladder": 0.018, "Uterus": 0.006},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.009, "Colon": 0.018, "Endosteum": 0.279, "Lymph_Nodes": 0.011, "Muscle": 0.118, "Prostate": 0.081, "Skin": 0.17, "Small_Intestine": 0.001, "Testicles": 0.475, "Urinary_Bladder": 0.025, "Uterus": 0.009},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.308, "Lymph_Nodes": 0.012, "Muscle": 0.128, "Prostate": 0.098, "Skin": 0.178, "Small_Intestine": 0.001, "Testicles": 0.522, "Urinary_Bladder": 0.03, "Uterus": 0.011},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.007, "Colon": 0.015, "Endosteum": 0.232, "Lymph_Nodes": 0.01, "Muscle": 0.105, "Prostate": 0.065, "Skin": 0.16, "Small_Intestine": 0.001, "Testicles": 0.415, "Urinary_Bladder": 0.021, "Uterus": 0.007},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.01, "Colon": 0.019, "Endosteum": 0.285, "Lymph_Nodes": 0.012, "Muscle": 0.122, "Prostate": 0.088, "Skin": 0.173, "Small_Intestine": 0.001, "Testicles": 0.48, "Urinary_Bladder": 0.028, "Uterus": 0.011},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.012, "Colon": 0.022, "Endosteum": 0.309, "Lymph_Nodes": 0.013, "Muscle": 0.13, "Prostate": 0.103, "Skin": 0.18, "Small_Intestine": 0.002, "Testicles": 0.516, "Urinary_Bladder": 0.034, "Uterus": 0.013},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.013, "Colon": 0.024, "Endosteum": 0.3, "Lymph_Nodes": 0.012, "Muscle": 0.133, "Prostate": 0.115, "Skin": 0.181, "Small_Intestine": 0.002, "Testicles": 0.493, "Urinary_Bladder": 0.037, "Uterus": 0.013},
            "default": {"Active_Bone_Marrow": 0.008, "Colon": 0.016, "Endosteum": 0.266, "Lymph_Nodes": 0.011, "Muscle": 0.113, "Prostate": 0.072, "Skin": 0.166, "Small_Intestine": 0.001, "Testicles": 0.454, "Urinary_Bladder": 0.022, "Uterus": 0.007},
        },
    },

    # ── Abdominal Angiography ──
    # DUMMY: no PCXMC data. Uses ERCP coefficients as proxy (similar abdominal field).
    "Abdominal Angiography": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
            "default": {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
            "default": {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.08, "Adrenals": 0.217, "Breast": 0.024, "Colon": 0.105, "Endosteum": 0.13, "Gallbladder": 0.728, "Heart": 0.141, "Kidneys": 0.218, "Liver": 0.295, "Lungs": 0.055, "Lymph_Nodes": 0.375, "Muscle": 0.147, "Oesophagus": 0.283, "Ovaries": 0.01, "Pancreas": 1.013, "Skin": 0.108, "Small_Intestine": 0.1, "Spleen": 0.429, "Stomach": 3.219, "Thymus": 0.008, "Urinary_Bladder": 0.002, "Uterus": 0.009},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.34, "Breast": 0.036, "Colon": 0.161, "Endosteum": 0.165, "Gallbladder": 1.036, "Heart": 0.22, "Kidneys": 0.354, "Liver": 0.419, "Lungs": 0.086, "Lymph_Nodes": 0.532, "Muscle": 0.193, "Oesophagus": 0.426, "Ovaries": 0.02, "Pancreas": 1.491, "Skin": 0.118, "Small_Intestine": 0.157, "Spleen": 0.643, "Stomach": 4.129, "Thymus": 0.015, "Urinary_Bladder": 0.003, "Uterus": 0.015},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.452, "Breast": 0.044, "Colon": 0.205, "Endosteum": 0.19, "Gallbladder": 1.233, "Heart": 0.282, "Kidneys": 0.466, "Liver": 0.509, "Lungs": 0.11, "Lymph_Nodes": 0.643, "Muscle": 0.223, "Oesophagus": 0.531, "Ovaries": 0.034, "Pancreas": 1.845, "Prostate": 0.001, "Skin": 0.124, "Small_Intestine": 0.204, "Spleen": 0.797, "Stomach": 4.672, "Thymus": 0.02, "Thyroid": 0.001, "Urinary_Bladder": 0.005, "Uterus": 0.023},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.304, "Breast": 0.032, "Colon": 0.138, "Endosteum": 0.148, "Gallbladder": 0.867, "Heart": 0.185, "Kidneys": 0.297, "Liver": 0.362, "Lungs": 0.074, "Lymph_Nodes": 0.456, "Muscle": 0.17, "Oesophagus": 0.358, "Ovaries": 0.024, "Pancreas": 1.273, "Prostate": 0.001, "Skin": 0.113, "Small_Intestine": 0.136, "Spleen": 0.545, "Stomach": 3.638, "Thymus": 0.013, "Thyroid": 0.001, "Urinary_Bladder": 0.004, "Uterus": 0.015},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.131, "Adrenals": 0.468, "Breast": 0.046, "Colon": 0.205, "Endosteum": 0.188, "Gallbladder": 1.184, "Heart": 0.277, "Kidneys": 0.458, "Liver": 0.502, "Lungs": 0.11, "Lymph_Nodes": 0.631, "Muscle": 0.221, "Oesophagus": 0.517, "Ovaries": 0.045, "Pancreas": 1.817, "Prostate": 0.003, "Skin": 0.124, "Small_Intestine": 0.206, "Spleen": 0.789, "Stomach": 4.582, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.007, "Uterus": 0.025},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.157, "Adrenals": 0.607, "Breast": 0.057, "Colon": 0.255, "Endosteum": 0.214, "Gallbladder": 1.367, "Heart": 0.342, "Kidneys": 0.579, "Liver": 0.598, "Lungs": 0.138, "Lymph_Nodes": 0.746, "Muscle": 0.253, "Oesophagus": 0.624, "Ovaries": 0.07, "Pancreas": 2.192, "Prostate": 0.005, "Skin": 0.132, "Small_Intestine": 0.261, "Spleen": 0.957, "Stomach": 5.12, "Thymus": 0.03, "Thyroid": 0.002, "Urinary_Bladder": 0.01, "Uterus": 0.035},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.114, "Adrenals": 0.388, "Breast": 0.039, "Colon": 0.169, "Endosteum": 0.164, "Gallbladder": 0.99, "Heart": 0.225, "Kidneys": 0.375, "Liver": 0.422, "Lungs": 0.091, "Lymph_Nodes": 0.53, "Muscle": 0.191, "Oesophagus": 0.423, "Ovaries": 0.038, "Pancreas": 1.51, "Prostate": 0.004, "Skin": 0.118, "Small_Intestine": 0.171, "Spleen": 0.653, "Stomach": 3.989, "Thymus": 0.018, "Thyroid": 0.001, "Urinary_Bladder": 0.006, "Uterus": 0.021},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.18, "Adrenals": 0.715, "Breast": 0.065, "Colon": 0.288, "Endosteum": 0.229, "Gallbladder": 1.479, "Heart": 0.383, "Kidneys": 0.678, "Liver": 0.66, "Lungs": 0.157, "Lymph_Nodes": 0.821, "Muscle": 0.273, "Oesophagus": 0.683, "Ovaries": 0.09, "Pancreas": 2.445, "Prostate": 0.014, "Skin": 0.137, "Small_Intestine": 0.308, "Spleen": 1.074, "Stomach": 5.394, "Thymus": 0.037, "Thyroid": 0.002, "Urinary_Bladder": 0.015, "Uterus": 0.043},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.467, "Breast": 0.045, "Colon": 0.193, "Endosteum": 0.176, "Gallbladder": 1.095, "Heart": 0.257, "Kidneys": 0.443, "Liver": 0.472, "Lungs": 0.105, "Lymph_Nodes": 0.591, "Muscle": 0.208, "Oesophagus": 0.474, "Ovaries": 0.05, "Pancreas": 1.709, "Prostate": 0.007, "Skin": 0.122, "Small_Intestine": 0.201, "Spleen": 0.739, "Stomach": 4.257, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.008, "Uterus": 0.026},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.169, "Adrenals": 0.666, "Breast": 0.061, "Colon": 0.264, "Endosteum": 0.215, "Gallbladder": 1.404, "Heart": 0.352, "Kidneys": 0.625, "Liver": 0.616, "Lungs": 0.146, "Lymph_Nodes": 0.768, "Muscle": 0.258, "Oesophagus": 0.631, "Ovaries": 0.08, "Pancreas": 2.276, "Prostate": 0.012, "Skin": 0.133, "Small_Intestine": 0.281, "Spleen": 0.993, "Stomach": 5.119, "Testicles": 0.001, "Thymus": 0.033, "Thyroid": 0.001, "Urinary_Bladder": 0.014, "Uterus": 0.039},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.196, "Adrenals": 0.81, "Breast": 0.071, "Colon": 0.309, "Endosteum": 0.235, "Gallbladder": 1.571, "Heart": 0.408, "Kidneys": 0.747, "Liver": 0.701, "Lungs": 0.172, "Lymph_Nodes": 0.872, "Muscle": 0.286, "Oesophagus": 0.718, "Ovaries": 0.105, "Pancreas": 2.62, "Prostate": 0.018, "Salivary_Glands": 0.001, "Skin": 0.14, "Small_Intestine": 0.335, "Spleen": 1.144, "Stomach": 5.536, "Testicles": 0.001, "Thymus": 0.041, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.049},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.141, "Adrenals": 0.537, "Breast": 0.05, "Colon": 0.212, "Endosteum": 0.185, "Gallbladder": 1.182, "Heart": 0.282, "Kidneys": 0.498, "Liver": 0.512, "Lungs": 0.117, "Lymph_Nodes": 0.639, "Muscle": 0.221, "Oesophagus": 0.513, "Ovaries": 0.06, "Pancreas": 1.865, "Prostate": 0.009, "Skin": 0.125, "Small_Intestine": 0.223, "Spleen": 0.807, "Stomach": 4.461, "Thymus": 0.025, "Thyroid": 0.001, "Urinary_Bladder": 0.01, "Uterus": 0.031},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.181, "Adrenals": 0.742, "Breast": 0.066, "Colon": 0.28, "Endosteum": 0.22, "Gallbladder": 1.479, "Heart": 0.372, "Kidneys": 0.678, "Liver": 0.651, "Lungs": 0.156, "Lymph_Nodes": 0.808, "Muscle": 0.268, "Oesophagus": 0.659, "Ovaries": 0.09, "Pancreas": 2.409, "Prostate": 0.014, "Salivary_Glands": 0.001, "Skin": 0.136, "Small_Intestine": 0.301, "Spleen": 1.048, "Stomach": 5.254, "Testicles": 0.001, "Thymus": 0.036, "Thyroid": 0.001, "Urinary_Bladder": 0.015, "Uterus": 0.045},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.207, "Adrenals": 0.887, "Breast": 0.077, "Colon": 0.32, "Endosteum": 0.236, "Gallbladder": 1.641, "Heart": 0.422, "Kidneys": 0.795, "Liver": 0.73, "Lungs": 0.181, "Lymph_Nodes": 0.903, "Muscle": 0.293, "Oesophagus": 0.736, "Ovaries": 0.112, "Pancreas": 2.725, "Prostate": 0.02, "Salivary_Glands": 0.001, "Skin": 0.142, "Small_Intestine": 0.351, "Spleen": 1.186, "Stomach": 5.616, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.001, "Urinary_Bladder": 0.019, "Uterus": 0.055},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.222, "Adrenals": 0.952, "Breast": 0.081, "Colon": 0.33, "Endosteum": 0.232, "Gallbladder": 1.754, "Heart": 0.435, "Kidneys": 0.852, "Liver": 0.759, "Lungs": 0.189, "Lymph_Nodes": 0.939, "Muscle": 0.3, "Oesophagus": 0.765, "Ovaries": 0.112, "Pancreas": 2.839, "Prostate": 0.022, "Salivary_Glands": 0.001, "Skin": 0.145, "Small_Intestine": 0.368, "Spleen": 1.221, "Stomach": 5.642, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.002, "Urinary_Bladder": 0.022, "Uterus": 0.063},
            "default": {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
        },
    },

    # ── Cystography ──
    # DUMMY: no PCXMC data. Pelvic region (combines cystography + pelvis procedures).
    # Placeholder uses Peripheral Angiography coefficients as a starting point.
    "Cystography": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.009, "Colon": 0.017, "Endosteum": 0.246, "Lymph_Nodes": 0.01, "Muscle": 0.111, "Prostate": 0.077, "Skin": 0.164, "Small_Intestine": 0.001, "Testicles": 0.428, "Urinary_Bladder": 0.025, "Uterus": 0.008},
            "default": {"Active_Bone_Marrow": 0.009, "Colon": 0.017, "Endosteum": 0.246, "Lymph_Nodes": 0.01, "Muscle": 0.111, "Prostate": 0.077, "Skin": 0.164, "Small_Intestine": 0.001, "Testicles": 0.428, "Urinary_Bladder": 0.025, "Uterus": 0.008},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.287, "Lymph_Nodes": 0.012, "Muscle": 0.126, "Prostate": 0.1, "Skin": 0.176, "Small_Intestine": 0.002, "Testicles": 0.475, "Urinary_Bladder": 0.032, "Uterus": 0.011},
            "default": {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.287, "Lymph_Nodes": 0.012, "Muscle": 0.126, "Prostate": 0.1, "Skin": 0.176, "Small_Intestine": 0.002, "Testicles": 0.475, "Urinary_Bladder": 0.032, "Uterus": 0.011},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.003, "Colon": 0.008, "Endosteum": 0.147, "Lymph_Nodes": 0.007, "Muscle": 0.078, "Prostate": 0.03, "Skin": 0.139, "Testicles": 0.31, "Urinary_Bladder": 0.01, "Uterus": 0.002},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.005, "Colon": 0.011, "Endosteum": 0.208, "Lymph_Nodes": 0.009, "Muscle": 0.096, "Prostate": 0.048, "Skin": 0.153, "Testicles": 0.365, "Urinary_Bladder": 0.013, "Uterus": 0.005},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.006, "Colon": 0.014, "Endosteum": 0.25, "Lymph_Nodes": 0.01, "Muscle": 0.107, "Prostate": 0.062, "Skin": 0.162, "Testicles": 0.392, "Urinary_Bladder": 0.015, "Uterus": 0.007},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.004, "Colon": 0.01, "Endosteum": 0.177, "Lymph_Nodes": 0.008, "Muscle": 0.087, "Prostate": 0.039, "Skin": 0.145, "Testicles": 0.341, "Urinary_Bladder": 0.012, "Uterus": 0.004},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.006, "Colon": 0.014, "Endosteum": 0.242, "Lymph_Nodes": 0.01, "Muscle": 0.106, "Prostate": 0.058, "Skin": 0.16, "Testicles": 0.405, "Urinary_Bladder": 0.017, "Uterus": 0.006},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.008, "Colon": 0.017, "Endosteum": 0.283, "Lymph_Nodes": 0.011, "Muscle": 0.117, "Prostate": 0.072, "Skin": 0.169, "Testicles": 0.447, "Urinary_Bladder": 0.021, "Uterus": 0.008},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.005, "Colon": 0.012, "Endosteum": 0.201, "Lymph_Nodes": 0.009, "Muscle": 0.094, "Prostate": 0.05, "Skin": 0.151, "Testicles": 0.378, "Urinary_Bladder": 0.015, "Uterus": 0.005},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.008, "Colon": 0.016, "Endosteum": 0.266, "Lymph_Nodes": 0.011, "Muscle": 0.113, "Prostate": 0.072, "Skin": 0.166, "Small_Intestine": 0.001, "Testicles": 0.454, "Urinary_Bladder": 0.022, "Uterus": 0.007},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.01, "Colon": 0.019, "Endosteum": 0.301, "Lymph_Nodes": 0.012, "Muscle": 0.124, "Prostate": 0.088, "Skin": 0.174, "Small_Intestine": 0.001, "Testicles": 0.509, "Urinary_Bladder": 0.027, "Uterus": 0.009},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.006, "Colon": 0.013, "Endosteum": 0.219, "Lymph_Nodes": 0.009, "Muscle": 0.1, "Prostate": 0.058, "Skin": 0.156, "Small_Intestine": 0.001, "Testicles": 0.402, "Urinary_Bladder": 0.018, "Uterus": 0.006},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.009, "Colon": 0.018, "Endosteum": 0.279, "Lymph_Nodes": 0.011, "Muscle": 0.118, "Prostate": 0.081, "Skin": 0.17, "Small_Intestine": 0.001, "Testicles": 0.475, "Urinary_Bladder": 0.025, "Uterus": 0.009},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.011, "Colon": 0.021, "Endosteum": 0.308, "Lymph_Nodes": 0.012, "Muscle": 0.128, "Prostate": 0.098, "Skin": 0.178, "Small_Intestine": 0.001, "Testicles": 0.522, "Urinary_Bladder": 0.03, "Uterus": 0.011},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.007, "Colon": 0.015, "Endosteum": 0.232, "Lymph_Nodes": 0.01, "Muscle": 0.105, "Prostate": 0.065, "Skin": 0.16, "Small_Intestine": 0.001, "Testicles": 0.415, "Urinary_Bladder": 0.021, "Uterus": 0.007},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.01, "Colon": 0.019, "Endosteum": 0.285, "Lymph_Nodes": 0.012, "Muscle": 0.122, "Prostate": 0.088, "Skin": 0.173, "Small_Intestine": 0.001, "Testicles": 0.48, "Urinary_Bladder": 0.028, "Uterus": 0.011},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.012, "Colon": 0.022, "Endosteum": 0.309, "Lymph_Nodes": 0.013, "Muscle": 0.13, "Prostate": 0.103, "Skin": 0.18, "Small_Intestine": 0.002, "Testicles": 0.516, "Urinary_Bladder": 0.034, "Uterus": 0.013},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.013, "Colon": 0.024, "Endosteum": 0.3, "Lymph_Nodes": 0.012, "Muscle": 0.133, "Prostate": 0.115, "Skin": 0.181, "Small_Intestine": 0.002, "Testicles": 0.493, "Urinary_Bladder": 0.037, "Uterus": 0.013},
            "default": {"Active_Bone_Marrow": 0.008, "Colon": 0.016, "Endosteum": 0.266, "Lymph_Nodes": 0.011, "Muscle": 0.113, "Prostate": 0.072, "Skin": 0.166, "Small_Intestine": 0.001, "Testicles": 0.454, "Urinary_Bladder": 0.022, "Uterus": 0.007},
        },
    },

    # ── Unknown ──
    # Fallback for unrecognized angiography/fluoroscopy procedures.
    # Uses Abdominal Angiography (= ERCP proxy) coefficients.
    # Kept as a separate named category.
    "Unknown": {
        "F": {
            (120, 4.0, 0.1): {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
            "default": {"Active_Bone_Marrow": 0.161, "Adrenals": 0.634, "Breast": 0.057, "Colon": 0.239, "Endosteum": 0.194, "Gallbladder": 1.323, "Heart": 0.317, "Kidneys": 0.581, "Liver": 0.569, "Lungs": 0.133, "Lymph_Nodes": 0.709, "Muscle": 0.239, "Oesophagus": 0.573, "Ovaries": 0.071, "Pancreas": 2.091, "Prostate": 0.011, "Salivary_Glands": 0.001, "Skin": 0.13, "Small_Intestine": 0.256, "Spleen": 0.901, "Stomach": 4.729, "Testicles": 0.001, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.013, "Uterus": 0.039},
        },
        "M": {
            (120, 4.0, 0.4): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
            "default": {"Active_Bone_Marrow": 0.198, "Adrenals": 0.825, "Breast": 0.072, "Colon": 0.298, "Endosteum": 0.222, "Gallbladder": 1.598, "Heart": 0.395, "Kidneys": 0.747, "Liver": 0.692, "Lungs": 0.168, "Lymph_Nodes": 0.859, "Muscle": 0.28, "Oesophagus": 0.701, "Ovaries": 0.096, "Pancreas": 2.573, "Prostate": 0.017, "Salivary_Glands": 0.001, "Skin": 0.139, "Small_Intestine": 0.326, "Spleen": 1.111, "Stomach": 5.378, "Testicles": 0.001, "Thymus": 0.038, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.053},
        },
        "U": {
            (60, 4.0, 0.1): {"Active_Bone_Marrow": 0.08, "Adrenals": 0.217, "Breast": 0.024, "Colon": 0.105, "Endosteum": 0.13, "Gallbladder": 0.728, "Heart": 0.141, "Kidneys": 0.218, "Liver": 0.295, "Lungs": 0.055, "Lymph_Nodes": 0.375, "Muscle": 0.147, "Oesophagus": 0.283, "Ovaries": 0.01, "Pancreas": 1.013, "Skin": 0.108, "Small_Intestine": 0.1, "Spleen": 0.429, "Stomach": 3.219, "Thymus": 0.008, "Urinary_Bladder": 0.002, "Uterus": 0.009},
            (60, 4.0, 0.4): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.34, "Breast": 0.036, "Colon": 0.161, "Endosteum": 0.165, "Gallbladder": 1.036, "Heart": 0.22, "Kidneys": 0.354, "Liver": 0.419, "Lungs": 0.086, "Lymph_Nodes": 0.532, "Muscle": 0.193, "Oesophagus": 0.426, "Ovaries": 0.02, "Pancreas": 1.491, "Skin": 0.118, "Small_Intestine": 0.157, "Spleen": 0.643, "Stomach": 4.129, "Thymus": 0.015, "Urinary_Bladder": 0.003, "Uterus": 0.015},
            (60, 4.0, 0.9): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.452, "Breast": 0.044, "Colon": 0.205, "Endosteum": 0.19, "Gallbladder": 1.233, "Heart": 0.282, "Kidneys": 0.466, "Liver": 0.509, "Lungs": 0.11, "Lymph_Nodes": 0.643, "Muscle": 0.223, "Oesophagus": 0.531, "Ovaries": 0.034, "Pancreas": 1.845, "Prostate": 0.001, "Skin": 0.124, "Small_Intestine": 0.204, "Spleen": 0.797, "Stomach": 4.672, "Thymus": 0.02, "Thyroid": 0.001, "Urinary_Bladder": 0.005, "Uterus": 0.023},
            (70, 4.0, 0.1): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.304, "Breast": 0.032, "Colon": 0.138, "Endosteum": 0.148, "Gallbladder": 0.867, "Heart": 0.185, "Kidneys": 0.297, "Liver": 0.362, "Lungs": 0.074, "Lymph_Nodes": 0.456, "Muscle": 0.17, "Oesophagus": 0.358, "Ovaries": 0.024, "Pancreas": 1.273, "Prostate": 0.001, "Skin": 0.113, "Small_Intestine": 0.136, "Spleen": 0.545, "Stomach": 3.638, "Thymus": 0.013, "Thyroid": 0.001, "Urinary_Bladder": 0.004, "Uterus": 0.015},
            (70, 4.0, 0.4): {"Active_Bone_Marrow": 0.131, "Adrenals": 0.468, "Breast": 0.046, "Colon": 0.205, "Endosteum": 0.188, "Gallbladder": 1.184, "Heart": 0.277, "Kidneys": 0.458, "Liver": 0.502, "Lungs": 0.11, "Lymph_Nodes": 0.631, "Muscle": 0.221, "Oesophagus": 0.517, "Ovaries": 0.045, "Pancreas": 1.817, "Prostate": 0.003, "Skin": 0.124, "Small_Intestine": 0.206, "Spleen": 0.789, "Stomach": 4.582, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.007, "Uterus": 0.025},
            (70, 4.0, 0.9): {"Active_Bone_Marrow": 0.157, "Adrenals": 0.607, "Breast": 0.057, "Colon": 0.255, "Endosteum": 0.214, "Gallbladder": 1.367, "Heart": 0.342, "Kidneys": 0.579, "Liver": 0.598, "Lungs": 0.138, "Lymph_Nodes": 0.746, "Muscle": 0.253, "Oesophagus": 0.624, "Ovaries": 0.07, "Pancreas": 2.192, "Prostate": 0.005, "Skin": 0.132, "Small_Intestine": 0.261, "Spleen": 0.957, "Stomach": 5.12, "Thymus": 0.03, "Thyroid": 0.002, "Urinary_Bladder": 0.01, "Uterus": 0.035},
            (80, 4.0, 0.1): {"Active_Bone_Marrow": 0.114, "Adrenals": 0.388, "Breast": 0.039, "Colon": 0.169, "Endosteum": 0.164, "Gallbladder": 0.99, "Heart": 0.225, "Kidneys": 0.375, "Liver": 0.422, "Lungs": 0.091, "Lymph_Nodes": 0.53, "Muscle": 0.191, "Oesophagus": 0.423, "Ovaries": 0.038, "Pancreas": 1.51, "Prostate": 0.004, "Skin": 0.118, "Small_Intestine": 0.171, "Spleen": 0.653, "Stomach": 3.989, "Thymus": 0.018, "Thyroid": 0.001, "Urinary_Bladder": 0.006, "Uterus": 0.021},
            (80, 4.0, 0.4): {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
            (80, 4.0, 0.9): {"Active_Bone_Marrow": 0.18, "Adrenals": 0.715, "Breast": 0.065, "Colon": 0.288, "Endosteum": 0.229, "Gallbladder": 1.479, "Heart": 0.383, "Kidneys": 0.678, "Liver": 0.66, "Lungs": 0.157, "Lymph_Nodes": 0.821, "Muscle": 0.273, "Oesophagus": 0.683, "Ovaries": 0.09, "Pancreas": 2.445, "Prostate": 0.014, "Skin": 0.137, "Small_Intestine": 0.308, "Spleen": 1.074, "Stomach": 5.394, "Thymus": 0.037, "Thyroid": 0.002, "Urinary_Bladder": 0.015, "Uterus": 0.043},
            (90, 4.0, 0.1): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.467, "Breast": 0.045, "Colon": 0.193, "Endosteum": 0.176, "Gallbladder": 1.095, "Heart": 0.257, "Kidneys": 0.443, "Liver": 0.472, "Lungs": 0.105, "Lymph_Nodes": 0.591, "Muscle": 0.208, "Oesophagus": 0.474, "Ovaries": 0.05, "Pancreas": 1.709, "Prostate": 0.007, "Skin": 0.122, "Small_Intestine": 0.201, "Spleen": 0.739, "Stomach": 4.257, "Thymus": 0.022, "Thyroid": 0.001, "Urinary_Bladder": 0.008, "Uterus": 0.026},
            (90, 4.0, 0.4): {"Active_Bone_Marrow": 0.169, "Adrenals": 0.666, "Breast": 0.061, "Colon": 0.264, "Endosteum": 0.215, "Gallbladder": 1.404, "Heart": 0.352, "Kidneys": 0.625, "Liver": 0.616, "Lungs": 0.146, "Lymph_Nodes": 0.768, "Muscle": 0.258, "Oesophagus": 0.631, "Ovaries": 0.08, "Pancreas": 2.276, "Prostate": 0.012, "Skin": 0.133, "Small_Intestine": 0.281, "Spleen": 0.993, "Stomach": 5.119, "Testicles": 0.001, "Thymus": 0.033, "Thyroid": 0.001, "Urinary_Bladder": 0.014, "Uterus": 0.039},
            (90, 4.0, 0.9): {"Active_Bone_Marrow": 0.196, "Adrenals": 0.81, "Breast": 0.071, "Colon": 0.309, "Endosteum": 0.235, "Gallbladder": 1.571, "Heart": 0.408, "Kidneys": 0.747, "Liver": 0.701, "Lungs": 0.172, "Lymph_Nodes": 0.872, "Muscle": 0.286, "Oesophagus": 0.718, "Ovaries": 0.105, "Pancreas": 2.62, "Prostate": 0.018, "Salivary_Glands": 0.001, "Skin": 0.14, "Small_Intestine": 0.335, "Spleen": 1.144, "Stomach": 5.536, "Testicles": 0.001, "Thymus": 0.041, "Thyroid": 0.002, "Urinary_Bladder": 0.018, "Uterus": 0.049},
            (100, 4.0, 0.1): {"Active_Bone_Marrow": 0.141, "Adrenals": 0.537, "Breast": 0.05, "Colon": 0.212, "Endosteum": 0.185, "Gallbladder": 1.182, "Heart": 0.282, "Kidneys": 0.498, "Liver": 0.512, "Lungs": 0.117, "Lymph_Nodes": 0.639, "Muscle": 0.221, "Oesophagus": 0.513, "Ovaries": 0.06, "Pancreas": 1.865, "Prostate": 0.009, "Skin": 0.125, "Small_Intestine": 0.223, "Spleen": 0.807, "Stomach": 4.461, "Thymus": 0.025, "Thyroid": 0.001, "Urinary_Bladder": 0.01, "Uterus": 0.031},
            (100, 4.0, 0.4): {"Active_Bone_Marrow": 0.181, "Adrenals": 0.742, "Breast": 0.066, "Colon": 0.28, "Endosteum": 0.22, "Gallbladder": 1.479, "Heart": 0.372, "Kidneys": 0.678, "Liver": 0.651, "Lungs": 0.156, "Lymph_Nodes": 0.808, "Muscle": 0.268, "Oesophagus": 0.659, "Ovaries": 0.09, "Pancreas": 2.409, "Prostate": 0.014, "Salivary_Glands": 0.001, "Skin": 0.136, "Small_Intestine": 0.301, "Spleen": 1.048, "Stomach": 5.254, "Testicles": 0.001, "Thymus": 0.036, "Thyroid": 0.001, "Urinary_Bladder": 0.015, "Uterus": 0.045},
            (100, 4.0, 0.9): {"Active_Bone_Marrow": 0.207, "Adrenals": 0.887, "Breast": 0.077, "Colon": 0.32, "Endosteum": 0.236, "Gallbladder": 1.641, "Heart": 0.422, "Kidneys": 0.795, "Liver": 0.73, "Lungs": 0.181, "Lymph_Nodes": 0.903, "Muscle": 0.293, "Oesophagus": 0.736, "Ovaries": 0.112, "Pancreas": 2.725, "Prostate": 0.02, "Salivary_Glands": 0.001, "Skin": 0.142, "Small_Intestine": 0.351, "Spleen": 1.186, "Stomach": 5.616, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.002, "Urinary_Bladder": 0.019, "Uterus": 0.055},
            (120, 4.0, 0.9): {"Active_Bone_Marrow": 0.222, "Adrenals": 0.952, "Breast": 0.081, "Colon": 0.33, "Endosteum": 0.232, "Gallbladder": 1.754, "Heart": 0.435, "Kidneys": 0.852, "Liver": 0.759, "Lungs": 0.189, "Lymph_Nodes": 0.939, "Muscle": 0.3, "Oesophagus": 0.765, "Ovaries": 0.112, "Pancreas": 2.839, "Prostate": 0.022, "Salivary_Glands": 0.001, "Skin": 0.145, "Small_Intestine": 0.368, "Spleen": 1.221, "Stomach": 5.642, "Testicles": 0.001, "Thymus": 0.043, "Thyroid": 0.002, "Urinary_Bladder": 0.022, "Uterus": 0.063},
            "default": {"Active_Bone_Marrow": 0.152, "Adrenals": 0.574, "Breast": 0.054, "Colon": 0.24, "Endosteum": 0.205, "Gallbladder": 1.306, "Heart": 0.321, "Kidneys": 0.553, "Liver": 0.568, "Lungs": 0.13, "Lymph_Nodes": 0.711, "Muscle": 0.243, "Oesophagus": 0.585, "Ovaries": 0.065, "Pancreas": 2.08, "Prostate": 0.009, "Skin": 0.13, "Small_Intestine": 0.251, "Spleen": 0.91, "Stomach": 4.911, "Thymus": 0.029, "Thyroid": 0.001, "Urinary_Bladder": 0.011, "Uterus": 0.033},
        },
    },
}

# ---------------------------------------------------------------------------

eff_coefficients_fluoro: Dict = {

    # Coronary Angiography
    "Coronary Angiography": {
        "F": {
            (120, 4.0, 0.1): 0.000608,
            "default": 0.000608,
        },
        "M": {
            (120, 4.0, 0.4): 0.000686,
            "default": 0.000686,
        },
        "U": {
            (60, 4.0, 0.1): 0.000427,
            (60, 4.0, 0.4): 0.000528,
            (60, 4.0, 0.9): 0.000592,
            (80, 4.0, 0.1): 0.000517,
            (80, 4.0, 0.4): 0.000623,
            (80, 4.0, 0.9): 0.000682,
            (100, 4.0, 0.1): 0.000574,
            (100, 4.0, 0.4): 0.000667,
            (100, 4.0, 0.9): 0.000713,
            (120, 4.0, 0.9): 0.000722,
            "default": 0.000623,
        },
    },
    # Coronary Angioplasty
    "Coronary Angioplasty": {
        "F": {
            (120, 4.0, 0.1): 0.000591,
            "default": 0.000591,
        },
        "M": {
            (120, 4.0, 0.4): 0.00067,
            "default": 0.00067,
        },
        "U": {
            (60, 4.0, 0.1): 0.000409,
            (60, 4.0, 0.4): 0.000511,
            (60, 4.0, 0.9): 0.000575,
            (70, 4.0, 0.1): 0.000458,
            (70, 4.0, 0.4): 0.000566,
            (70, 4.0, 0.9): 0.000632,
            (80, 4.0, 0.1): 0.0005,
            (80, 4.0, 0.4): 0.000608,
            (80, 4.0, 0.9): 0.000667,
            (90, 4.0, 0.1): 0.000533,
            (90, 4.0, 0.4): 0.000634,
            (90, 4.0, 0.9): 0.000686,
            (100, 4.0, 0.1): 0.000557,
            (100, 4.0, 0.4): 0.000651,
            (100, 4.0, 0.9): 0.000697,
            (120, 4.0, 0.9): 0.000705,
            "default": 0.000608,
        },
    },
    # Carotid Angiography
    "Carotid Angiography": {
        "F": {
            (120, 4.0, 0.1): 0.000626,
            "default": 0.000626,
        },
        "M": {
            (120, 4.0, 0.4): 0.00068,
            "default": 0.00068,
        },
        "U": {
            (60, 4.0, 0.1): 0.000501,
            (60, 4.0, 0.4): 0.000582,
            (60, 4.0, 0.9): 0.000624,
            (70, 4.0, 0.1): 0.000535,
            (70, 4.0, 0.4): 0.000614,
            (70, 4.0, 0.9): 0.000654,
            (80, 4.0, 0.1): 0.000563,
            (80, 4.0, 0.4): 0.000639,
            (80, 4.0, 0.9): 0.000675,
            (90, 4.0, 0.1): 0.000586,
            (90, 4.0, 0.4): 0.000657,
            (90, 4.0, 0.9): 0.000691,
            (100, 4.0, 0.1): 0.000603,
            (100, 4.0, 0.4): 0.000669,
            (100, 4.0, 0.9): 0.000699,
            (120, 4.0, 0.9): 0.000702,
            "default": 0.000639,
        },
    },
    # ERCP
    "ERCP": {
        "F": {
            (120, 4.0, 0.1): 0.000756,
            "default": 0.000756,
        },
        "M": {
            (120, 4.0, 0.4): 0.000879,
            "default": 0.000879,
        },
        "U": {
            (60, 4.0, 0.1): 0.000475,
            (60, 4.0, 0.4): 0.000626,
            (60, 4.0, 0.9): 0.000723,
            (70, 4.0, 0.1): 0.000549,
            (70, 4.0, 0.4): 0.000711,
            (70, 4.0, 0.9): 0.000811,
            (80, 4.0, 0.1): 0.000613,
            (80, 4.0, 0.4): 0.000776,
            (80, 4.0, 0.9): 0.000868,
            (90, 4.0, 0.1): 0.000664,
            (90, 4.0, 0.4): 0.000819,
            (90, 4.0, 0.9): 0.000901,
            (100, 4.0, 0.1): 0.000703,
            (100, 4.0, 0.4): 0.000848,
            (100, 4.0, 0.9): 0.000922,
            (120, 4.0, 0.9): 0.000936,
            "default": 0.000776,
        },
    },
    # Peripheral Angiography
    "Peripheral Angiography": {
        "F": {
            (120, 4.0, 0.1): 2.7e-05,
            "default": 2.7e-05,
        },
        "M": {
            (120, 4.0, 0.4): 3.1e-05,
            "default": 3.1e-05,
        },
        "U": {
            (60, 4.0, 0.1): 1.8e-05,
            (60, 4.0, 0.4): 2.2e-05,
            (60, 4.0, 0.9): 2.4e-05,
            (70, 4.0, 0.1): 2e-05,
            (70, 4.0, 0.4): 2.5e-05,
            (70, 4.0, 0.9): 2.8e-05,
            (80, 4.0, 0.1): 2.2e-05,
            (80, 4.0, 0.4): 2.8e-05,
            (80, 4.0, 0.9): 3.1e-05,
            (90, 4.0, 0.1): 2.4e-05,
            (90, 4.0, 0.4): 2.9e-05,
            (90, 4.0, 0.9): 3.3e-05,
            (100, 4.0, 0.1): 2.5e-05,
            (100, 4.0, 0.4): 3e-05,
            (100, 4.0, 0.9): 3.3e-05,
            (120, 4.0, 0.9): 3.3e-05,
            "default": 2.8e-05,
        },
    },
    # Abdominal Angiography (ERCP proxy)
    "Abdominal Angiography": {
        "F": {
            (120, 4.0, 0.1): 0.000756,
            "default": 0.000756,
        },
        "M": {
            (120, 4.0, 0.4): 0.000879,
            "default": 0.000879,
        },
        "U": {
            (60, 4.0, 0.1): 0.000475,
            (60, 4.0, 0.4): 0.000626,
            (60, 4.0, 0.9): 0.000723,
            (70, 4.0, 0.1): 0.000549,
            (70, 4.0, 0.4): 0.000711,
            (70, 4.0, 0.9): 0.000811,
            (80, 4.0, 0.1): 0.000613,
            (80, 4.0, 0.4): 0.000776,
            (80, 4.0, 0.9): 0.000868,
            (90, 4.0, 0.1): 0.000664,
            (90, 4.0, 0.4): 0.000819,
            (90, 4.0, 0.9): 0.000901,
            (100, 4.0, 0.1): 0.000703,
            (100, 4.0, 0.4): 0.000848,
            (100, 4.0, 0.9): 0.000922,
            (120, 4.0, 0.9): 0.000936,
            "default": 0.000776,
        },
    },
    # Cystography (Peripheral proxy)
    "Cystography": {
        "F": {
            (120, 4.0, 0.1): 2.7e-05,
            "default": 2.7e-05,
        },
        "M": {
            (120, 4.0, 0.4): 3.1e-05,
            "default": 3.1e-05,
        },
        "U": {
            (60, 4.0, 0.1): 1.8e-05,
            (60, 4.0, 0.4): 2.2e-05,
            (60, 4.0, 0.9): 2.4e-05,
            (70, 4.0, 0.1): 2e-05,
            (70, 4.0, 0.4): 2.5e-05,
            (70, 4.0, 0.9): 2.8e-05,
            (80, 4.0, 0.1): 2.2e-05,
            (80, 4.0, 0.4): 2.8e-05,
            (80, 4.0, 0.9): 3.1e-05,
            (90, 4.0, 0.1): 2.4e-05,
            (90, 4.0, 0.4): 2.9e-05,
            (90, 4.0, 0.9): 3.3e-05,
            (100, 4.0, 0.1): 2.5e-05,
            (100, 4.0, 0.4): 3e-05,
            (100, 4.0, 0.9): 3.3e-05,
            (120, 4.0, 0.9): 3.3e-05,
            "default": 2.8e-05,
        },
    },
    # Unknown (ERCP proxy)
    "Unknown": {
        "F": {
            (120, 4.0, 0.1): 0.000756,
            "default": 0.000756,
        },
        "M": {
            (120, 4.0, 0.4): 0.000879,
            "default": 0.000879,
        },
        "U": {
            (60, 4.0, 0.1): 0.000475,
            (60, 4.0, 0.4): 0.000626,
            (60, 4.0, 0.9): 0.000723,
            (70, 4.0, 0.1): 0.000549,
            (70, 4.0, 0.4): 0.000711,
            (70, 4.0, 0.9): 0.000811,
            (80, 4.0, 0.1): 0.000613,
            (80, 4.0, 0.4): 0.000776,
            (80, 4.0, 0.9): 0.000868,
            (90, 4.0, 0.1): 0.000664,
            (90, 4.0, 0.4): 0.000819,
            (90, 4.0, 0.9): 0.000901,
            (100, 4.0, 0.1): 0.000703,
            (100, 4.0, 0.4): 0.000848,
            (100, 4.0, 0.9): 0.000922,
            (120, 4.0, 0.9): 0.000936,
            "default": 0.000776,
        },
    },
}

# ---------------------------------------------------------------------------
# Scale raw PCXMC effective-dose values  Sv / Gy·cm²  →  mSv / Gy·cm²
# Organ-dose coefficients are already stored as mGy / Gy·cm² above.
# ---------------------------------------------------------------------------
for _exam in eff_coefficients_fluoro:
    for _sex in eff_coefficients_fluoro[_exam]:
        for _bq in list(eff_coefficients_fluoro[_exam][_sex]):
            _val = eff_coefficients_fluoro[_exam][_sex][_bq]
            if isinstance(_val, (int, float)):
                eff_coefficients_fluoro[_exam][_sex][_bq] = _val * 1000.0



# ---------------------------------------------------------------------------
# Inject "Validation" unity exam into fluoroscopy coefficient tables
# ---------------------------------------------------------------------------
# Mirrors the radiography validation pattern
# (radiography_doses.py, section "Validation: unity coefficient
# set").  A validation row with
#     Modality               = "RF" (or "XA")
#     Fluoro DAP (Total) Gy-cm2 = 1000.0
#     Protocol / Description = text containing "validation"
# traverses the full production lookup chain
#     _resolve_fluoro_exam → coefficients_fluoro[exam][sex][beam_key]
# and produces exactly 0.1 Gy to every organ and 0.1 Sv effective dose,
# directly comparable to the CT and radiography pathway validations.
#
#   dose_Gy = DAP × coeff_mGy_per_Gy·cm² / 1000
#           = 1000 × 0.1 / 1000    = 0.1 Gy per organ ✓
#   E_mSv   = DAP × coeff_mSv_per_Gy·cm²
#           = 1000 × 0.1           = 100 mSv = 0.1 Sv ✓
#
# The coefficient magnitudes 0.1 mGy/(Gy·cm²) and 0.1 mSv/(Gy·cm²) sit inside
# the physically plausible fluoroscopy range (real PCXMC values here
# span ~0.001-5), so an accidental Validation lookup on a real clinical DAP
# (~1-500 Gy·cm²) produces realistic-looking doses rather than catastrophic
# ones.
#
# The injection happens AFTER the Sv→mSv scaling loop above so the unity
# 0.1 mSv/(Gy·cm²) is not multiplied by 1000 a second time.

_VALIDATION_UNITY_COEFF_MGY_PER_GYCM2: float = 0.1
_VALIDATION_UNITY_COEFF_MSV_PER_GYCM2: float = 0.1

# Organ names use the fluoro-table spellings (Kidneys, Lungs, Ovaries,
# Testicles, Urinary_Bladder, …) so the full pipeline — including downstream
# organ-name normalisation in exam_doses._normalize_organ —
# behaves identically to a real clinical fluoroscopy row.
_VALIDATION_UNITY_FLUORO_ORGANS: Dict[str, float] = {
    "Active_Bone_Marrow":    _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Adrenals":              _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Brain":                 _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Breast":                _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Colon":                 _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Endosteum":             _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Extrathoracic_Airways": _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Gallbladder":           _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Heart":                 _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Kidneys":               _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Liver":                 _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Lungs":                 _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Lymph_Nodes":           _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Muscle":                _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Oesophagus":            _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Oral_Mucosa":           _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Ovaries":               _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Pancreas":              _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Prostate":              _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Salivary_Glands":       _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Skin":                  _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Small_Intestine":       _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Spleen":                _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Stomach":               _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Testicles":             _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Thymus":                _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Thyroid":               _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Urinary_Bladder":       _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
    "Uterus":                _VALIDATION_UNITY_COEFF_MGY_PER_GYCM2,
}


coefficients_fluoro["Validation"] = {
    _s: {"default": dict(_VALIDATION_UNITY_FLUORO_ORGANS)}
    for _s in ("F", "M", "U")
}
eff_coefficients_fluoro["Validation"] = {
    _s: {"default": _VALIDATION_UNITY_COEFF_MSV_PER_GYCM2}
    for _s in ("F", "M", "U")
}


_REGION_TO_EXAM = {
    "Cystography": "Cystography",
    "CarotidAngiography": "Carotid Angiography",
    "CoronaryAngiography": "Coronary Angiography",
    "CoronaryAngioplasty": "Coronary Angioplasty",
    "PeripheralAngiography": "Peripheral Angiography",
    "AbdominalAngiography": "Abdominal Angiography",
    "ERCP": "ERCP",
    # Generic region labels
    "Chest": "Coronary Angiography",
    "Neck": "Carotid Angiography",
    "Abdomen": "Abdominal Angiography",
    "Pelvis": "Cystography",
    "Extremities": "Peripheral Angiography",
    "Unknown": "Unknown",
    "Default": "Unknown",
    # Internal-consistency validation exam (unity coefficients)
    "Validation": "Validation",
}

_EXAM_KEYWORDS = [
    ("ptca",          "Coronary Angioplasty"),
    ("angioplast",    "Coronary Angioplasty"),
    ("pci",           "Coronary Angioplasty"),
    ("coron",         "Coronary Angiography"),
    ("cardiac",       "Coronary Angiography"),
    ("cardio",        "Coronary Angiography"),
    ("ercp",          "ERCP"),
    ("erc ",          "ERCP"),
    ("endoscop",      "ERCP"),
    ("cysto",         "Cystography"),
    ("voiding",       "Cystography"),
    ("carotid",       "Carotid Angiography"),
    ("aorto",         "Abdominal Angiography"),
    ("aorta",         "Abdominal Angiography"),
    ("peripher",      "Peripheral Angiography"),
    ("femoral",       "Peripheral Angiography"),
    ("lower limb",    "Peripheral Angiography"),
    ("iliac",         "Cystography"),
    ("pelvic",        "Cystography"),
    ("pelvis",        "Cystography"),
    ("thorax",        "Coronary Angiography"),
    ("heart",         "Coronary Angiography"),
    # Internal-consistency validation exam (unity coefficients)
    ("validation",    "Validation"),
]


@lru_cache(maxsize=8192)
def _resolve_fluoro_exam_cached(text: Optional[str], region: str) -> str:
    """Pure, memoised core of :func:`_resolve_fluoro_exam`.

    ``text`` is the already-lowercased ``"{Protocol} {Description}"`` string
    when a row was supplied, else ``None`` (reproducing the row-absent path).
    The fluoroscopy coefficient tables and ``_EXAM_KEYWORDS`` are static after
    import, so memoising on ``(text, region)`` is identical.
    """
    if text is not None:
        for keyword, exam in _EXAM_KEYWORDS:
            if keyword in text:
                return exam
    rl = str(region or "").lower()
    for keyword, exam in _EXAM_KEYWORDS:
        if keyword in rl:
            return exam
    mapped = _REGION_TO_EXAM.get(region)
    if mapped:
        return mapped
    return "Unknown"


def _resolve_fluoro_exam(region: str, row: dict = None) -> str:
    """Map Protocol/Description text or region string to a fluoroscopy exam name."""
    # Build the lookup text as an f-string of the raw row values (so a
    # present-but-None field still stringifies to "None"), then
    # delegate to the memoised core.
    text = (f"{row.get('Protocol', '')} {row.get('Description', '')}".lower()
            if row else None)
    return _resolve_fluoro_exam_cached(text, region)


@lru_cache(maxsize=2048)
def _xa_organ_unit_coeffs(exam: str, sex: str):
    """Per-(exam, sex) tuple of ``(organ, coeff_mGy_per_Gy_cm2)`` with the
    positive-coefficient filtering done once (coefficients are static).

    Pre-filters positive coefficients once so the caller only applies
    ``dap * c / 1000.0`` per organ without re-checking each entry.
    """
    exam_table = coefficients_fluoro.get(exam, {})
    sex_table = exam_table.get(sex) or exam_table.get("U") or {}
    coeffs = sex_table.get("default")
    if not isinstance(coeffs, dict) or not coeffs:
        return None
    out = []
    for organ, coeff in coeffs.items():
        c = _xa_to_float(coeff)
        if c and c > 0:
            out.append((organ, c))
    return tuple(out) if out else None


@lru_cache(maxsize=2048)
def _xa_eff_unit_coeff(exam: str, sex: str) -> Optional[float]:
    """Per-(exam, sex) effective-dose coefficient (mSv per Gy*cm2), memoised."""
    exam_eff = eff_coefficients_fluoro.get(exam, {})
    sex_eff = exam_eff.get(sex) or exam_eff.get("U") or {}
    coeff = sex_eff.get("default")
    c = _xa_to_float(coeff)
    return c if (c is not None and c > 0) else None


def _sex_key(sex_code: str) -> str:
    s = str(sex_code or "U").strip().upper()
    return s if s in ("F", "M") else "U"


def xa_organ_doses_gy_from_dap(
    dap_gy_cm2: float,
    region: str,
    sex_code: str = "U",
    row: dict = None,
) -> Optional[Dict[str, float]]:
    """Return organ absorbed doses in Gy from DAP (Gy*cm2).

    Uses exam-based PCXMC fluoroscopy conversion coefficients.
    dose_Gy = DAP_Gy_cm2 * coeff_mGy_per_Gy_cm2 / 1000
    """
    dap = _xa_to_float(dap_gy_cm2)
    if dap is None or dap <= 0:
        return None
    exam = _resolve_fluoro_exam(region, row)
    sex = _sex_key(sex_code)
    unit = _xa_organ_unit_coeffs(exam, sex)
    if not unit:
        return None
    out: Dict[str, float] = {}
    for organ, c in unit:
        dose_gy = dap * c / 1000.0
        if dose_gy > 0:
            out[organ] = dose_gy
    return out if out else None


def xa_effective_dose_msv_from_dap(
    dap_gy_cm2: float,
    region: str,
    sex_code: str = "U",
    row: dict = None,
) -> Optional[float]:
    """Return effective dose in mSv from DAP (Gy*cm2).

    ed_mSv = DAP_Gy_cm2 * coeff_mSv_per_Gy_cm2
    """
    dap = _xa_to_float(dap_gy_cm2)
    if dap is None or dap <= 0:
        return None
    exam = _resolve_fluoro_exam(region, row)
    sex = _sex_key(sex_code)
    c = _xa_eff_unit_coeff(exam, sex)
    if c is None:
        return None
    return dap * c



