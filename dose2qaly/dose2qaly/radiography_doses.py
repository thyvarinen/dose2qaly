"""Organ dose conversion coefficients for radiography.

Coefficient tables provide mGy / (Gy·cm²) (organ dose) and mSv / (Gy·cm²)
(effective dose) per imaging region/projection/age-group/beam-quality.

Sources
-------
* PCXMC v2.0

PCXMC was run at a DAP of 1.0 mGy·cm², so organ doses (mGy) and effective
doses (mSv) are multiplied by 1000 to give coefficients per Gy·cm².
Abdomen keeps separate F/M/U sex blocks; the other regions use a single
``"U"`` block.

Projection-specific tables (PA, LATR, OBLIQUE) cover only the
region/projection combinations for which a dedicated PCXMC export exists.
For all other combinations the public lookup functions
(get_radiography_organ_coeffs, get_radiography_eff_dose) fall back to the
AP table automatically.

This module's tables were generated from PCXMC exports by an
internal script not included in this release; contact the maintainer to
regenerate — do not hand-edit the tables below.
"""
from __future__ import annotations
from typing import Dict, Mapping, Optional, Tuple, Any
from functools import lru_cache
from .projection_orientation import (
    get_projection_orientation,
    infer_projection_orientation_from_text,
    normalize_projection_orientation,
)
from ._parse_utils import to_float as _to_float


_RADIO_BEAM_QUALITIES: Tuple = (
    (60, 2.5, 0.0),  (60, 3.0, 0.0),  (60, 3.5, 0.0),
    (70, 2.5, 0.0),  (70, 3.0, 0.0),  (70, 3.5, 0.0),
    (80, 2.5, 0.0),  (80, 3.0, 0.0),  (80, 3.5, 0.0),
    (90, 2.5, 0.0),  (90, 3.0, 0.0),  (90, 3.5, 0.0),
    (100, 2.5, 0.0), (100, 3.0, 0.0), (100, 3.5, 0.0),
    (110, 2.5, 0.0), (110, 3.0, 0.0), (110, 3.5, 0.0),
    (120, 2.5, 0.0), (120, 3.0, 0.0), (120, 3.5, 0.0),
)


def _pb(organs: dict) -> dict:
    """Expand organ dict to all beam qualities."""
    return {bk: dict(organs) for bk in _RADIO_BEAM_QUALITIES}


def _pbe(coeff: float) -> dict:
    """Expand effective-dose coefficient to all beam qualities."""
    return {bk: float(coeff) for bk in _RADIO_BEAM_QUALITIES}


# Validation region (unity coefficients)
_VALIDATION_ORGANS: Dict[str, float] = {
    "Active_Bone_Marrow": 0.1, "Adrenals": 0.1, "Brain": 0.1, "Breast": 0.1,
    "Colon": 0.1, "Endosteum": 0.1, "Extrathoracic_Airways": 0.1,
    "Gallbladder": 0.1, "Gonads": 0.1, "Heart": 0.1, "Kidneys": 0.1,
    "Lens": 0.1, "Liver": 0.1, "Lungs": 0.1, "Lymph_Nodes": 0.1,
    "Muscle": 0.1, "Oesophagus": 0.1, "Oral_Mucosa": 0.1, "Pancreas": 0.1,
    "Prostate": 0.1, "Salivary_Glands": 0.1, "Skin": 0.1,
    "Small_Intestine": 0.1, "Spleen": 0.1, "Stomach": 0.1, "Thymus": 0.1,
    "Thyroid": 0.1, "Urinary_Bladder": 0.1, "Uterus": 0.1,
}
_VALIDATION_ORGAN_REGION: Dict = {
    "F": _pb(_VALIDATION_ORGANS),
    "M": _pb(_VALIDATION_ORGANS),
    "U": _pb(_VALIDATION_ORGANS),
}
_VALIDATION_EFF_REGION: Dict = {
    "F": _pbe(0.1),
    "M": _pbe(0.1),
    "U": _pbe(0.1),
}



# AP organ coefficients (mGy / Gy·cm²)

coefficients_AP: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
            'F': {
                (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.067881, 'Adrenals': 0.07648, 'Brain': 1.9e-05, 'Breast': 0.028277, 'Colon': 0.414058, 'Endosteum': 0.04668, 'Extrathoracic_Airways': 0.000382, 'Gallbladder': 0.202652, 'Gonads': 0.119289, 'Heart': 0.090458, 'Kidneys': 0.112926, 'Lens': 0.000749, 'Liver': 0.261069, 'Lungs': 0.045847, 'Lymph_Nodes': 0.140173, 'Muscle': 0.077002, 'Oesophagus': 0.026855, 'Oral_Mucosa': 0.000416, 'Pancreas': 0.276713, 'Salivary_Glands': 0.000216, 'Skin': 0.106175, 'Small_Intestine': 0.341646, 'Spleen': 0.074324, 'Stomach': 0.347134, 'Thymus': 0.00382, 'Thyroid': 0.001, 'Urinary_Bladder': 0.540122, 'Uterus': 0.144309},
                (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.075459, 'Adrenals': 0.086188, 'Brain': 2.2e-05, 'Breast': 0.030324, 'Colon': 0.44919, 'Endosteum': 0.052207, 'Extrathoracic_Airways': 0.000428, 'Gallbladder': 0.225152, 'Gonads': 0.133816, 'Heart': 0.09911, 'Kidneys': 0.12662, 'Lens': 0.000823, 'Liver': 0.28585, 'Lungs': 0.050377, 'Lymph_Nodes': 0.153605, 'Muscle': 0.083359, 'Oesophagus': 0.030145, 'Oral_Mucosa': 0.00047, 'Pancreas': 0.305863, 'Salivary_Glands': 0.000247, 'Skin': 0.108895, 'Small_Intestine': 0.37356, 'Spleen': 0.083372, 'Stomach': 0.377719, 'Thymus': 0.00434, 'Thyroid': 0.001144, 'Urinary_Bladder': 0.576874, 'Uterus': 0.161142},
                (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.082406, 'Adrenals': 0.095223, 'Brain': 2.5e-05, 'Breast': 0.032154, 'Colon': 0.479994, 'Endosteum': 0.057314, 'Extrathoracic_Airways': 0.00047, 'Gallbladder': 0.245597, 'Gonads': 0.147229, 'Heart': 0.10683, 'Kidneys': 0.139253, 'Lens': 0.000888, 'Liver': 0.307889, 'Lungs': 0.05445, 'Lymph_Nodes': 0.165591, 'Muscle': 0.088971, 'Oesophagus': 0.033188, 'Oral_Mucosa': 0.000521, 'Pancreas': 0.332104, 'Salivary_Glands': 0.000276, 'Skin': 0.111228, 'Small_Intestine': 0.401869, 'Spleen': 0.091729, 'Stomach': 0.404655, 'Thymus': 0.004831, 'Thyroid': 0.001283, 'Urinary_Bladder': 0.608687, 'Uterus': 0.176562},
                (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.107614, 'Adrenals': 0.130669, 'Brain': 5.9e-05, 'Breast': 0.037229, 'Colon': 0.531962, 'Endosteum': 0.077004, 'Extrathoracic_Airways': 0.000698, 'Gallbladder': 0.299403, 'Gonads': 0.193024, 'Heart': 0.124367, 'Kidneys': 0.180805, 'Lens': 0.001004, 'Liver': 0.351649, 'Lungs': 0.065235, 'Lymph_Nodes': 0.192035, 'Muscle': 0.101473, 'Oesophagus': 0.043938, 'Oral_Mucosa': 0.000853, 'Pancreas': 0.391717, 'Salivary_Glands': 0.000513, 'Skin': 0.115103, 'Small_Intestine': 0.457339, 'Spleen': 0.119788, 'Stomach': 0.452049, 'Thymus': 0.007586, 'Thyroid': 0.002391, 'Urinary_Bladder': 0.65659, 'Uterus': 0.223925},
                (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.118157, 'Adrenals': 0.144667, 'Brain': 6.7e-05, 'Breast': 0.039818, 'Colon': 0.571311, 'Endosteum': 0.084889, 'Extrathoracic_Airways': 0.000773, 'Gallbladder': 0.327733, 'Gonads': 0.212926, 'Heart': 0.134766, 'Kidneys': 0.199339, 'Lens': 0.001088, 'Liver': 0.380513, 'Lungs': 0.070913, 'Lymph_Nodes': 0.208072, 'Muscle': 0.109036, 'Oesophagus': 0.0485, 'Oral_Mucosa': 0.000949, 'Pancreas': 0.426923, 'Salivary_Glands': 0.000575, 'Skin': 0.118108, 'Small_Intestine': 0.494358, 'Spleen': 0.132123, 'Stomach': 0.486651, 'Thymus': 0.008456, 'Thyroid': 0.002681, 'Urinary_Bladder': 0.696697, 'Uterus': 0.246151},
                (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.127736, 'Adrenals': 0.157501, 'Brain': 7.4e-05, 'Breast': 0.042126, 'Colon': 0.605582, 'Endosteum': 0.092092, 'Extrathoracic_Airways': 0.000841, 'Gallbladder': 0.353166, 'Gonads': 0.231047, 'Heart': 0.143977, 'Kidneys': 0.216192, 'Lens': 0.001161, 'Liver': 0.405955, 'Lungs': 0.075982, 'Lymph_Nodes': 0.222269, 'Muscle': 0.115693, 'Oesophagus': 0.05266, 'Oral_Mucosa': 0.001039, 'Pancreas': 0.458263, 'Salivary_Glands': 0.000633, 'Skin': 0.120691, 'Small_Intestine': 0.526928, 'Spleen': 0.143351, 'Stomach': 0.516896, 'Thymus': 0.009266, 'Thyroid': 0.002954, 'Urinary_Bladder': 0.731274, 'Uterus': 0.266258},
                (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.145729, 'Adrenals': 0.182483, 'Brain': 0.000115, 'Breast': 0.0453, 'Colon': 0.629, 'Endosteum': 0.105229, 'Extrathoracic_Airways': 0.001046, 'Gallbladder': 0.383943, 'Gonads': 0.261033, 'Heart': 0.153664, 'Kidneys': 0.242691, 'Lens': 0.001362, 'Liver': 0.427717, 'Lungs': 0.082536, 'Lymph_Nodes': 0.236613, 'Muscle': 0.122886, 'Oesophagus': 0.060057, 'Oral_Mucosa': 0.001329, 'Pancreas': 0.489863, 'Salivary_Glands': 0.00085, 'Skin': 0.122607, 'Small_Intestine': 0.554433, 'Spleen': 0.161408, 'Stomach': 0.538636, 'Thymus': 0.011414, 'Thyroid': 0.003907, 'Urinary_Bladder': 0.751059, 'Uterus': 0.296588},
                (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.157841, 'Adrenals': 0.198734, 'Brain': 0.000127, 'Breast': 0.048112, 'Colon': 0.668542, 'Endosteum': 0.114264, 'Extrathoracic_Airways': 0.00114, 'Gallbladder': 0.414327, 'Gonads': 0.283445, 'Heart': 0.16458, 'Kidneys': 0.263388, 'Lens': 0.001463, 'Liver': 0.457371, 'Lungs': 0.088659, 'Lymph_Nodes': 0.253374, 'Muscle': 0.130827, 'Oesophagus': 0.065258, 'Oral_Mucosa': 0.001455, 'Pancreas': 0.526737, 'Salivary_Glands': 0.000935, 'Skin': 0.12563, 'Small_Intestine': 0.592395, 'Spleen': 0.175237, 'Stomach': 0.573582, 'Thymus': 0.012498, 'Thyroid': 0.004298, 'Urinary_Bladder': 0.790755, 'Uterus': 0.321202},
                (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.168682, 'Adrenals': 0.213371, 'Brain': 0.000139, 'Breast': 0.050589, 'Colon': 0.70256, 'Endosteum': 0.122378, 'Extrathoracic_Airways': 0.001226, 'Gallbladder': 0.441175, 'Gonads': 0.303503, 'Heart': 0.174122, 'Kidneys': 0.281886, 'Lens': 0.001551, 'Liver': 0.483152, 'Lungs': 0.094053, 'Lymph_Nodes': 0.268013, 'Muscle': 0.137741, 'Oesophagus': 0.069921, 'Oral_Mucosa': 0.001571, 'Pancreas': 0.559068, 'Salivary_Glands': 0.001013, 'Skin': 0.128208, 'Small_Intestine': 0.625352, 'Spleen': 0.18761, 'Stomach': 0.603738, 'Thymus': 0.013487, 'Thyroid': 0.004657, 'Urinary_Bladder': 0.824611, 'Uterus': 0.343115},
                (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.17766, 'Adrenals': 0.225501, 'Brain': 0.000174, 'Breast': 0.051709, 'Colon': 0.702684, 'Endosteum': 0.127726, 'Extrathoracic_Airways': 0.00136, 'Gallbladder': 0.450828, 'Gonads': 0.316248, 'Heart': 0.176563, 'Kidneys': 0.292633, 'Lens': 0.001675, 'Liver': 0.486363, 'Lungs': 0.096368, 'Lymph_Nodes': 0.271263, 'Muscle': 0.13984, 'Oesophagus': 0.073366, 'Oral_Mucosa': 0.001746, 'Pancreas': 0.566065, 'Salivary_Glands': 0.00116, 'Skin': 0.128385, 'Small_Intestine': 0.62897, 'Spleen': 0.195265, 'Stomach': 0.604646, 'Thymus': 0.014634, 'Thyroid': 0.005255, 'Urinary_Bladder': 0.821901, 'Uterus': 0.355683},
                (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.190109, 'Adrenals': 0.242244, 'Brain': 0.000189, 'Breast': 0.05449, 'Colon': 0.739921, 'Endosteum': 0.136887, 'Extrathoracic_Airways': 0.001465, 'Gallbladder': 0.480646, 'Gonads': 0.33889, 'Heart': 0.187136, 'Kidneys': 0.313432, 'Lens': 0.001782, 'Liver': 0.514691, 'Lungs': 0.102408, 'Lymph_Nodes': 0.287441, 'Muscle': 0.147556, 'Oesophagus': 0.078673, 'Oral_Mucosa': 0.001885, 'Pancreas': 0.601682, 'Salivary_Glands': 0.001256, 'Skin': 0.131242, 'Small_Intestine': 0.665169, 'Spleen': 0.209216, 'Stomach': 0.637666, 'Thymus': 0.015789, 'Thyroid': 0.005691, 'Urinary_Bladder': 0.858902, 'Uterus': 0.38037},
                (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.201103, 'Adrenals': 0.257096, 'Brain': 0.000203, 'Breast': 0.056908, 'Colon': 0.771563, 'Endosteum': 0.144986, 'Extrathoracic_Airways': 0.001558, 'Gallbladder': 0.506622, 'Gonads': 0.358853, 'Heart': 0.19626, 'Kidneys': 0.331743, 'Lens': 0.001874, 'Liver': 0.538998, 'Lungs': 0.107661, 'Lymph_Nodes': 0.301388, 'Muscle': 0.154197, 'Oesophagus': 0.083361, 'Oral_Mucosa': 0.00201, 'Pancreas': 0.632479, 'Salivary_Glands': 0.001343, 'Skin': 0.133656, 'Small_Intestine': 0.69619, 'Spleen': 0.221511, 'Stomach': 0.6658, 'Thymus': 0.016826, 'Thyroid': 0.006087, 'Urinary_Bladder': 0.890095, 'Uterus': 0.402038},
            },
            'M': {
                (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.048067, 'Adrenals': 0.029291, 'Brain': 1.2e-05, 'Breast': 0.035447, 'Colon': 0.276332, 'Endosteum': 0.031414, 'Extrathoracic_Airways': 0.000315, 'Gallbladder': 0.147322, 'Gonads': 0.014989, 'Heart': 0.063171, 'Kidneys': 0.049005, 'Lens': 0.00011, 'Liver': 0.164537, 'Lungs': 0.034766, 'Lymph_Nodes': 0.162581, 'Muscle': 0.073533, 'Oesophagus': 0.017352, 'Oral_Mucosa': 0.000219, 'Pancreas': 0.155633, 'Prostate': 0.094642, 'Salivary_Glands': 0.000207, 'Skin': 0.094077, 'Small_Intestine': 0.288608, 'Spleen': 0.032943, 'Stomach': 0.254333, 'Thymus': 0.002769, 'Thyroid': 0.000842, 'Urinary_Bladder': 0.277557},
                (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.053521, 'Adrenals': 0.033283, 'Brain': 1.4e-05, 'Breast': 0.038707, 'Colon': 0.302145, 'Endosteum': 0.035192, 'Extrathoracic_Airways': 0.00035, 'Gallbladder': 0.164476, 'Gonads': 0.016878, 'Heart': 0.069973, 'Kidneys': 0.055435, 'Lens': 0.000122, 'Liver': 0.181515, 'Lungs': 0.038393, 'Lymph_Nodes': 0.175539, 'Muscle': 0.079118, 'Oesophagus': 0.019626, 'Oral_Mucosa': 0.000248, 'Pancreas': 0.173818, 'Prostate': 0.106312, 'Salivary_Glands': 0.000235, 'Skin': 0.09651, 'Small_Intestine': 0.316249, 'Spleen': 0.03726, 'Stomach': 0.278794, 'Thymus': 0.003154, 'Thyroid': 0.000966, 'Urinary_Bladder': 0.303729},
                (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.058534, 'Adrenals': 0.037056, 'Brain': 1.6e-05, 'Breast': 0.041611, 'Colon': 0.325061, 'Endosteum': 0.038694, 'Extrathoracic_Airways': 0.000382, 'Gallbladder': 0.180199, 'Gonads': 0.018636, 'Heart': 0.07614, 'Kidneys': 0.061461, 'Lens': 0.000132, 'Liver': 0.196805, 'Lungs': 0.041675, 'Lymph_Nodes': 0.187015, 'Muscle': 0.084013, 'Oesophagus': 0.021758, 'Oral_Mucosa': 0.000276, 'Pancreas': 0.190483, 'Prostate': 0.117123, 'Salivary_Glands': 0.000261, 'Skin': 0.098589, 'Small_Intestine': 0.340843, 'Spleen': 0.041306, 'Stomach': 0.300566, 'Thymus': 0.00352, 'Thyroid': 0.001084, 'Urinary_Bladder': 0.326972},
                (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.07809, 'Adrenals': 0.056854, 'Brain': 4.1e-05, 'Breast': 0.048389, 'Colon': 0.370087, 'Endosteum': 0.053339, 'Extrathoracic_Airways': 0.000572, 'Gallbladder': 0.227237, 'Gonads': 0.025828, 'Heart': 0.093516, 'Kidneys': 0.087904, 'Lens': 0.00021, 'Liver': 0.234201, 'Lungs': 0.051489, 'Lymph_Nodes': 0.211253, 'Muscle': 0.094288, 'Oesophagus': 0.031468, 'Oral_Mucosa': 0.000482, 'Pancreas': 0.240649, 'Prostate': 0.157139, 'Salivary_Glands': 0.000452, 'Skin': 0.102067, 'Small_Intestine': 0.39057, 'Spleen': 0.059244, 'Stomach': 0.344515, 'Thymus': 0.005841, 'Thyroid': 0.002103, 'Urinary_Bladder': 0.375084},
                (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.085878, 'Adrenals': 0.063354, 'Brain': 4.6e-05, 'Breast': 0.052329, 'Colon': 0.40006, 'Endosteum': 0.058897, 'Extrathoracic_Airways': 0.000631, 'Gallbladder': 0.24972, 'Gonads': 0.028592, 'Heart': 0.102227, 'Kidneys': 0.097601, 'Lens': 0.000231, 'Liver': 0.255078, 'Lungs': 0.056207, 'Lymph_Nodes': 0.226513, 'Muscle': 0.10082, 'Oesophagus': 0.034947, 'Oral_Mucosa': 0.000538, 'Pancreas': 0.264524, 'Prostate': 0.173585, 'Salivary_Glands': 0.000505, 'Skin': 0.104754, 'Small_Intestine': 0.422896, 'Spleen': 0.06578, 'Stomach': 0.373117, 'Thymus': 0.006526, 'Thyroid': 0.002361, 'Urinary_Bladder': 0.405815},
                (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.092975, 'Adrenals': 0.06939, 'Brain': 5.1e-05, 'Breast': 0.055819, 'Colon': 0.426445, 'Endosteum': 0.063991, 'Extrathoracic_Airways': 0.000685, 'Gallbladder': 0.270055, 'Gonads': 0.031129, 'Heart': 0.110049, 'Kidneys': 0.106542, 'Lens': 0.00025, 'Liver': 0.273698, 'Lungs': 0.060445, 'Lymph_Nodes': 0.239952, 'Muscle': 0.106538, 'Oesophagus': 0.038157, 'Oral_Mucosa': 0.00059, 'Pancreas': 0.286118, 'Prostate': 0.18861, 'Salivary_Glands': 0.000555, 'Skin': 0.107059, 'Small_Intestine': 0.451408, 'Spleen': 0.071808, 'Stomach': 0.398348, 'Thymus': 0.007166, 'Thyroid': 0.002604, 'Urinary_Bladder': 0.432905},
                (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.107538, 'Adrenals': 0.084655, 'Brain': 8.1e-05, 'Breast': 0.059705, 'Colon': 0.448759, 'Endosteum': 0.074246, 'Extrathoracic_Airways': 0.000858, 'Gallbladder': 0.298776, 'Gonads': 0.035945, 'Heart': 0.120751, 'Kidneys': 0.12545, 'Lens': 0.000309, 'Liver': 0.294773, 'Lungs': 0.066871, 'Lymph_Nodes': 0.252858, 'Muscle': 0.112412, 'Oesophagus': 0.045364, 'Oral_Mucosa': 0.000783, 'Pancreas': 0.316705, 'Prostate': 0.215773, 'Salivary_Glands': 0.00073, 'Skin': 0.108853, 'Small_Intestine': 0.476322, 'Spleen': 0.084802, 'Stomach': 0.420382, 'Thymus': 0.009077, 'Thyroid': 0.003584, 'Urinary_Bladder': 0.458046},
                (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.11664, 'Adrenals': 0.092658, 'Brain': 8.9e-05, 'Breast': 0.063866, 'Colon': 0.479505, 'Endosteum': 0.080737, 'Extrathoracic_Airways': 0.000934, 'Gallbladder': 0.323442, 'Gonads': 0.039141, 'Heart': 0.130237, 'Kidneys': 0.136914, 'Lens': 0.000336, 'Liver': 0.316926, 'Lungs': 0.072083, 'Lymph_Nodes': 0.268696, 'Muscle': 0.11922, 'Oesophagus': 0.049527, 'Oral_Mucosa': 0.00086, 'Pancreas': 0.342909, 'Prostate': 0.234589, 'Salivary_Glands': 0.000802, 'Skin': 0.111563, 'Small_Intestine': 0.509602, 'Spleen': 0.092559, 'Stomach': 0.449828, 'Thymus': 0.009957, 'Thyroid': 0.003946, 'Urinary_Bladder': 0.489848},
                (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.124813, 'Adrenals': 0.099947, 'Brain': 9.7e-05, 'Breast': 0.067505, 'Colon': 0.506209, 'Endosteum': 0.086584, 'Extrathoracic_Airways': 0.001003, 'Gallbladder': 0.345383, 'Gonads': 0.042022, 'Heart': 0.138631, 'Kidneys': 0.147287, 'Lens': 0.000361, 'Liver': 0.336399, 'Lungs': 0.076702, 'Lymph_Nodes': 0.282473, 'Muscle': 0.125121, 'Oesophagus': 0.053298, 'Oral_Mucosa': 0.00093, 'Pancreas': 0.366218, 'Prostate': 0.251483, 'Salivary_Glands': 0.000867, 'Skin': 0.113873, 'Small_Intestine': 0.538555, 'Spleen': 0.09958, 'Stomach': 0.475447, 'Thymus': 0.010763, 'Thyroid': 0.004281, 'Urinary_Bladder': 0.517521},
                (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.132581, 'Adrenals': 0.108488, 'Brain': 0.000124, 'Breast': 0.068501, 'Colon': 0.509237, 'Endosteum': 0.091191, 'Extrathoracic_Airways': 0.001113, 'Gallbladder': 0.355728, 'Gonads': 0.04405, 'Heart': 0.142557, 'Kidneys': 0.156675, 'Lens': 0.000379, 'Liver': 0.342417, 'Lungs': 0.07943, 'Lymph_Nodes': 0.285084, 'Muscle': 0.126784, 'Oesophagus': 0.057092, 'Oral_Mucosa': 0.001069, 'Pancreas': 0.377113, 'Prostate': 0.264051, 'Salivary_Glands': 0.000987, 'Skin': 0.114128, 'Small_Intestine': 0.542145, 'Spleen': 0.106266, 'Stomach': 0.478753, 'Thymus': 0.011869, 'Thyroid': 0.004925, 'Urinary_Bladder': 0.522215},
                (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.142049, 'Adrenals': 0.117021, 'Brain': 0.000135, 'Breast': 0.072539, 'Colon': 0.538569, 'Endosteum': 0.097855, 'Extrathoracic_Airways': 0.001197, 'Gallbladder': 0.380232, 'Gonads': 0.047308, 'Heart': 0.151961, 'Kidneys': 0.168587, 'Lens': 0.000407, 'Liver': 0.364017, 'Lungs': 0.084657, 'Lymph_Nodes': 0.300308, 'Muscle': 0.133376, 'Oesophagus': 0.061457, 'Oral_Mucosa': 0.001156, 'Pancreas': 0.403136, 'Prostate': 0.283266, 'Salivary_Glands': 0.001068, 'Skin': 0.116699, 'Small_Intestine': 0.573942, 'Spleen': 0.114361, 'Stomach': 0.5069, 'Thymus': 0.012826, 'Thyroid': 0.005339, 'Urinary_Bladder': 0.552731},
                (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.150435, 'Adrenals': 0.124669, 'Brain': 0.000145, 'Breast': 0.076025, 'Colon': 0.563715, 'Endosteum': 0.103768, 'Extrathoracic_Airways': 0.001273, 'Gallbladder': 0.401707, 'Gonads': 0.050198, 'Heart': 0.160169, 'Kidneys': 0.179196, 'Lens': 0.000432, 'Liver': 0.382747, 'Lungs': 0.08923, 'Lymph_Nodes': 0.313388, 'Muscle': 0.13903, 'Oesophagus': 0.065351, 'Oral_Mucosa': 0.001235, 'Pancreas': 0.425943, 'Prostate': 0.30026, 'Salivary_Glands': 0.001141, 'Skin': 0.118871, 'Small_Intestine': 0.601241, 'Spleen': 0.121575, 'Stomach': 0.531068, 'Thymus': 0.013688, 'Thyroid': 0.005714, 'Urinary_Bladder': 0.578947},
            },
            'U': {
                (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.057974, 'Adrenals': 0.0528855, 'Brain': 1.55e-05, 'Breast': 0.031862, 'Colon': 0.345195, 'Endosteum': 0.039047, 'Extrathoracic_Airways': 0.0003485, 'Gallbladder': 0.174987, 'Gonads': 0.067139, 'Heart': 0.0768145, 'Kidneys': 0.0809655, 'Lens': 0.0004295, 'Liver': 0.212803, 'Lungs': 0.0403065, 'Lymph_Nodes': 0.151377, 'Muscle': 0.0752675, 'Oesophagus': 0.0221035, 'Oral_Mucosa': 0.0003175, 'Pancreas': 0.216173, 'Prostate': 0.047321, 'Salivary_Glands': 0.0002115, 'Skin': 0.100126, 'Small_Intestine': 0.315127, 'Spleen': 0.0536335, 'Stomach': 0.300733, 'Thymus': 0.0032945, 'Thyroid': 0.000921, 'Urinary_Bladder': 0.40884, 'Uterus': 0.0721545},
                (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.06449, 'Adrenals': 0.0597355, 'Brain': 1.8e-05, 'Breast': 0.0345155, 'Colon': 0.375667, 'Endosteum': 0.0436995, 'Extrathoracic_Airways': 0.000389, 'Gallbladder': 0.194814, 'Gonads': 0.075347, 'Heart': 0.0845415, 'Kidneys': 0.0910275, 'Lens': 0.0004725, 'Liver': 0.233683, 'Lungs': 0.044385, 'Lymph_Nodes': 0.164572, 'Muscle': 0.0812385, 'Oesophagus': 0.0248855, 'Oral_Mucosa': 0.000359, 'Pancreas': 0.239841, 'Prostate': 0.053156, 'Salivary_Glands': 0.000241, 'Skin': 0.102703, 'Small_Intestine': 0.344905, 'Spleen': 0.060316, 'Stomach': 0.328257, 'Thymus': 0.003747, 'Thyroid': 0.001055, 'Urinary_Bladder': 0.440302, 'Uterus': 0.080571},
                (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.07047, 'Adrenals': 0.0661395, 'Brain': 2.05e-05, 'Breast': 0.0368825, 'Colon': 0.402527, 'Endosteum': 0.048004, 'Extrathoracic_Airways': 0.000426, 'Gallbladder': 0.212898, 'Gonads': 0.0829325, 'Heart': 0.091485, 'Kidneys': 0.100357, 'Lens': 0.00051, 'Liver': 0.252347, 'Lungs': 0.0480625, 'Lymph_Nodes': 0.176303, 'Muscle': 0.086492, 'Oesophagus': 0.027473, 'Oral_Mucosa': 0.0003985, 'Pancreas': 0.261294, 'Prostate': 0.0585615, 'Salivary_Glands': 0.0002685, 'Skin': 0.104908, 'Small_Intestine': 0.371356, 'Spleen': 0.0665175, 'Stomach': 0.35261, 'Thymus': 0.0041755, 'Thyroid': 0.0011835, 'Urinary_Bladder': 0.46783, 'Uterus': 0.088281},
                (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.092852, 'Adrenals': 0.0937615, 'Brain': 5e-05, 'Breast': 0.042809, 'Colon': 0.451025, 'Endosteum': 0.0651715, 'Extrathoracic_Airways': 0.000635, 'Gallbladder': 0.26332, 'Gonads': 0.109426, 'Heart': 0.108941, 'Kidneys': 0.134354, 'Lens': 0.000607, 'Liver': 0.292925, 'Lungs': 0.058362, 'Lymph_Nodes': 0.201644, 'Muscle': 0.0978805, 'Oesophagus': 0.037703, 'Oral_Mucosa': 0.0006675, 'Pancreas': 0.316183, 'Prostate': 0.0785695, 'Salivary_Glands': 0.0004825, 'Skin': 0.108585, 'Small_Intestine': 0.423955, 'Spleen': 0.089516, 'Stomach': 0.398282, 'Thymus': 0.0067135, 'Thyroid': 0.002247, 'Urinary_Bladder': 0.515837, 'Uterus': 0.111963},
                (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.102017, 'Adrenals': 0.10401, 'Brain': 5.65e-05, 'Breast': 0.0460735, 'Colon': 0.485685, 'Endosteum': 0.071893, 'Extrathoracic_Airways': 0.000702, 'Gallbladder': 0.288726, 'Gonads': 0.120759, 'Heart': 0.118497, 'Kidneys': 0.14847, 'Lens': 0.0006595, 'Liver': 0.317796, 'Lungs': 0.06356, 'Lymph_Nodes': 0.217292, 'Muscle': 0.104928, 'Oesophagus': 0.0417235, 'Oral_Mucosa': 0.0007435, 'Pancreas': 0.345723, 'Prostate': 0.0867925, 'Salivary_Glands': 0.00054, 'Skin': 0.111431, 'Small_Intestine': 0.458627, 'Spleen': 0.0989515, 'Stomach': 0.429884, 'Thymus': 0.007491, 'Thyroid': 0.002521, 'Urinary_Bladder': 0.551256, 'Uterus': 0.123076},
                (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.110355, 'Adrenals': 0.113446, 'Brain': 6.25e-05, 'Breast': 0.0489725, 'Colon': 0.516014, 'Endosteum': 0.0780415, 'Extrathoracic_Airways': 0.000763, 'Gallbladder': 0.311611, 'Gonads': 0.131088, 'Heart': 0.127013, 'Kidneys': 0.161367, 'Lens': 0.0007055, 'Liver': 0.339827, 'Lungs': 0.0682135, 'Lymph_Nodes': 0.23111, 'Muscle': 0.111116, 'Oesophagus': 0.0454085, 'Oral_Mucosa': 0.0008145, 'Pancreas': 0.37219, 'Prostate': 0.094305, 'Salivary_Glands': 0.000594, 'Skin': 0.113875, 'Small_Intestine': 0.489168, 'Spleen': 0.107579, 'Stomach': 0.457622, 'Thymus': 0.008216, 'Thyroid': 0.002779, 'Urinary_Bladder': 0.582089, 'Uterus': 0.133129},
                (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.126634, 'Adrenals': 0.133569, 'Brain': 9.8e-05, 'Breast': 0.0525025, 'Colon': 0.538879, 'Endosteum': 0.0897375, 'Extrathoracic_Airways': 0.000952, 'Gallbladder': 0.341359, 'Gonads': 0.148489, 'Heart': 0.137207, 'Kidneys': 0.18407, 'Lens': 0.0008355, 'Liver': 0.361245, 'Lungs': 0.0747035, 'Lymph_Nodes': 0.244735, 'Muscle': 0.117649, 'Oesophagus': 0.0527105, 'Oral_Mucosa': 0.001056, 'Pancreas': 0.403284, 'Prostate': 0.107886, 'Salivary_Glands': 0.00079, 'Skin': 0.11573, 'Small_Intestine': 0.515378, 'Spleen': 0.123105, 'Stomach': 0.479509, 'Thymus': 0.0102455, 'Thyroid': 0.0037455, 'Urinary_Bladder': 0.604553, 'Uterus': 0.148294},
                (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.13724, 'Adrenals': 0.145696, 'Brain': 0.000108, 'Breast': 0.055989, 'Colon': 0.574024, 'Endosteum': 0.0975005, 'Extrathoracic_Airways': 0.001037, 'Gallbladder': 0.368885, 'Gonads': 0.161293, 'Heart': 0.147408, 'Kidneys': 0.200151, 'Lens': 0.0008995, 'Liver': 0.387149, 'Lungs': 0.080371, 'Lymph_Nodes': 0.261035, 'Muscle': 0.125024, 'Oesophagus': 0.0573925, 'Oral_Mucosa': 0.0011575, 'Pancreas': 0.434823, 'Prostate': 0.117294, 'Salivary_Glands': 0.0008685, 'Skin': 0.118596, 'Small_Intestine': 0.550998, 'Spleen': 0.133898, 'Stomach': 0.511705, 'Thymus': 0.0112275, 'Thyroid': 0.004122, 'Urinary_Bladder': 0.640301, 'Uterus': 0.160601},
                (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.146748, 'Adrenals': 0.156659, 'Brain': 0.000118, 'Breast': 0.059047, 'Colon': 0.604384, 'Endosteum': 0.104481, 'Extrathoracic_Airways': 0.0011145, 'Gallbladder': 0.393279, 'Gonads': 0.172763, 'Heart': 0.156377, 'Kidneys': 0.214587, 'Lens': 0.000956, 'Liver': 0.409776, 'Lungs': 0.0853775, 'Lymph_Nodes': 0.275243, 'Muscle': 0.131431, 'Oesophagus': 0.0616095, 'Oral_Mucosa': 0.0012505, 'Pancreas': 0.462643, 'Prostate': 0.125742, 'Salivary_Glands': 0.00094, 'Skin': 0.12104, 'Small_Intestine': 0.581954, 'Spleen': 0.143595, 'Stomach': 0.539593, 'Thymus': 0.012125, 'Thyroid': 0.004469, 'Urinary_Bladder': 0.671066, 'Uterus': 0.171558},
                (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.15512, 'Adrenals': 0.166994, 'Brain': 0.000149, 'Breast': 0.060105, 'Colon': 0.605961, 'Endosteum': 0.109459, 'Extrathoracic_Airways': 0.0012365, 'Gallbladder': 0.403278, 'Gonads': 0.180149, 'Heart': 0.15956, 'Kidneys': 0.224654, 'Lens': 0.001027, 'Liver': 0.41439, 'Lungs': 0.087899, 'Lymph_Nodes': 0.278173, 'Muscle': 0.133312, 'Oesophagus': 0.065229, 'Oral_Mucosa': 0.0014075, 'Pancreas': 0.471589, 'Prostate': 0.132025, 'Salivary_Glands': 0.0010735, 'Skin': 0.121256, 'Small_Intestine': 0.585557, 'Spleen': 0.150765, 'Stomach': 0.5417, 'Thymus': 0.0132515, 'Thyroid': 0.00509, 'Urinary_Bladder': 0.672058, 'Uterus': 0.177842},
                (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.166079, 'Adrenals': 0.179633, 'Brain': 0.000162, 'Breast': 0.0635145, 'Colon': 0.639245, 'Endosteum': 0.117371, 'Extrathoracic_Airways': 0.001331, 'Gallbladder': 0.430439, 'Gonads': 0.193099, 'Heart': 0.169548, 'Kidneys': 0.241009, 'Lens': 0.0010945, 'Liver': 0.439354, 'Lungs': 0.0935325, 'Lymph_Nodes': 0.293875, 'Muscle': 0.140466, 'Oesophagus': 0.070065, 'Oral_Mucosa': 0.0015205, 'Pancreas': 0.502409, 'Prostate': 0.141633, 'Salivary_Glands': 0.001162, 'Skin': 0.12397, 'Small_Intestine': 0.619555, 'Spleen': 0.161789, 'Stomach': 0.572283, 'Thymus': 0.0143075, 'Thyroid': 0.005515, 'Urinary_Bladder': 0.705817, 'Uterus': 0.190185},
                (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.175769, 'Adrenals': 0.190883, 'Brain': 0.000174, 'Breast': 0.0664665, 'Colon': 0.667639, 'Endosteum': 0.124377, 'Extrathoracic_Airways': 0.0014155, 'Gallbladder': 0.454164, 'Gonads': 0.204525, 'Heart': 0.178214, 'Kidneys': 0.25547, 'Lens': 0.001153, 'Liver': 0.460873, 'Lungs': 0.0984455, 'Lymph_Nodes': 0.307388, 'Muscle': 0.146614, 'Oesophagus': 0.074356, 'Oral_Mucosa': 0.0016225, 'Pancreas': 0.529211, 'Prostate': 0.15013, 'Salivary_Glands': 0.001242, 'Skin': 0.126264, 'Small_Intestine': 0.648716, 'Spleen': 0.171543, 'Stomach': 0.598434, 'Thymus': 0.015257, 'Thyroid': 0.0059005, 'Urinary_Bladder': 0.734521, 'Uterus': 0.201019},
            },
        },
            "Chest": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.059, "Adrenals": 0.022, "Brain": 0.0, "Breast": 0.595, "Colon": 0.004, "Endosteum": 0.112, "Extrathoracic_Airways": 0.069, "Gallbladder": 0.073, "Heart": 0.278, "Kidneys": 0.018, "Liver": 0.16, "Lungs": 0.194, "Lymph_Nodes": 0.084, "Muscle": 0.058, "Oesophagus": 0.07, "Oral_Mucosa": 0.003, "Ovaries": 0.0, "Pancreas": 0.103, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.102, "Small_Intestine": 0.004, "Spleen": 0.055, "Stomach": 0.189, "Testicles": 0.0, "Thymus": 0.468, "Thyroid": 0.291, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.064, "Adrenals": 0.025, "Brain": 0.0, "Breast": 0.626, "Colon": 0.004, "Endosteum": 0.12, "Extrathoracic_Airways": 0.073, "Gallbladder": 0.08, "Heart": 0.303, "Kidneys": 0.021, "Liver": 0.175, "Lungs": 0.212, "Lymph_Nodes": 0.092, "Muscle": 0.062, "Oesophagus": 0.078, "Oral_Mucosa": 0.003, "Ovaries": 0.0, "Pancreas": 0.115, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.104, "Small_Intestine": 0.004, "Spleen": 0.061, "Stomach": 0.207, "Testicles": 0.0, "Thymus": 0.503, "Thyroid": 0.31, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.067, "Adrenals": 0.028, "Brain": 0.0, "Breast": 0.651, "Colon": 0.005, "Endosteum": 0.127, "Extrathoracic_Airways": 0.077, "Gallbladder": 0.087, "Heart": 0.326, "Kidneys": 0.023, "Liver": 0.189, "Lungs": 0.229, "Lymph_Nodes": 0.099, "Muscle": 0.066, "Oesophagus": 0.086, "Oral_Mucosa": 0.003, "Ovaries": 0.0, "Pancreas": 0.126, "Prostate": 0.0, "Salivary_Glands": 0.007, "Skin": 0.106, "Small_Intestine": 0.005, "Spleen": 0.067, "Stomach": 0.223, "Testicles": 0.0, "Thymus": 0.533, "Thyroid": 0.327, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.059, "Brain": 0.001, "Breast": 0.685, "Colon": 0.008, "Endosteum": 0.139, "Extrathoracic_Airways": 0.085, "Gallbladder": 0.119, "Heart": 0.387, "Kidneys": 0.036, "Liver": 0.23, "Lungs": 0.275, "Lymph_Nodes": 0.122, "Muscle": 0.078, "Oesophagus": 0.121, "Oral_Mucosa": 0.005, "Ovaries": 0.0, "Pancreas": 0.166, "Prostate": 0.0, "Salivary_Glands": 0.009, "Skin": 0.111, "Small_Intestine": 0.008, "Spleen": 0.096, "Stomach": 0.265, "Testicles": 0.0, "Thymus": 0.605, "Thyroid": 0.375, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.065, "Brain": 0.001, "Breast": 0.715, "Colon": 0.009, "Endosteum": 0.147, "Extrathoracic_Airways": 0.09, "Gallbladder": 0.13, "Heart": 0.417, "Kidneys": 0.04, "Liver": 0.249, "Lungs": 0.297, "Lymph_Nodes": 0.131, "Muscle": 0.083, "Oesophagus": 0.132, "Oral_Mucosa": 0.005, "Ovaries": 0.0, "Pancreas": 0.181, "Prostate": 0.0, "Salivary_Glands": 0.01, "Skin": 0.114, "Small_Intestine": 0.009, "Spleen": 0.106, "Stomach": 0.286, "Testicles": 0.0, "Thymus": 0.644, "Thyroid": 0.398, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.071, "Brain": 0.001, "Breast": 0.741, "Colon": 0.01, "Endosteum": 0.154, "Extrathoracic_Airways": 0.094, "Gallbladder": 0.14, "Heart": 0.444, "Kidneys": 0.044, "Liver": 0.265, "Lungs": 0.317, "Lymph_Nodes": 0.14, "Muscle": 0.088, "Oesophagus": 0.143, "Oral_Mucosa": 0.006, "Ovaries": 0.0, "Pancreas": 0.196, "Prostate": 0.0, "Salivary_Glands": 0.011, "Skin": 0.116, "Small_Intestine": 0.01, "Spleen": 0.115, "Stomach": 0.305, "Testicles": 0.0, "Thymus": 0.678, "Thyroid": 0.418, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.092, "Brain": 0.001, "Breast": 0.754, "Colon": 0.012, "Endosteum": 0.158, "Extrathoracic_Airways": 0.096, "Gallbladder": 0.162, "Heart": 0.48, "Kidneys": 0.054, "Liver": 0.29, "Lungs": 0.342, "Lymph_Nodes": 0.154, "Muscle": 0.094, "Oesophagus": 0.17, "Oral_Mucosa": 0.008, "Ovaries": 0.002, "Pancreas": 0.224, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.119, "Small_Intestine": 0.013, "Spleen": 0.135, "Stomach": 0.326, "Testicles": 0.0, "Thymus": 0.726, "Thyroid": 0.437, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.103, "Adrenals": 0.1, "Brain": 0.002, "Breast": 0.782, "Colon": 0.013, "Endosteum": 0.165, "Extrathoracic_Airways": 0.101, "Gallbladder": 0.175, "Heart": 0.511, "Kidneys": 0.059, "Liver": 0.309, "Lungs": 0.365, "Lymph_Nodes": 0.164, "Muscle": 0.1, "Oesophagus": 0.183, "Oral_Mucosa": 0.009, "Ovaries": 0.002, "Pancreas": 0.242, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.122, "Small_Intestine": 0.014, "Spleen": 0.145, "Stomach": 0.347, "Testicles": 0.0, "Thymus": 0.766, "Thyroid": 0.459, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.108, "Brain": 0.002, "Breast": 0.805, "Colon": 0.014, "Endosteum": 0.172, "Extrathoracic_Airways": 0.105, "Gallbladder": 0.186, "Heart": 0.538, "Kidneys": 0.063, "Liver": 0.326, "Lungs": 0.384, "Lymph_Nodes": 0.173, "Muscle": 0.105, "Oesophagus": 0.195, "Oral_Mucosa": 0.01, "Ovaries": 0.002, "Pancreas": 0.257, "Prostate": 0.0, "Salivary_Glands": 0.014, "Skin": 0.124, "Small_Intestine": 0.015, "Spleen": 0.155, "Stomach": 0.365, "Testicles": 0.0, "Thymus": 0.801, "Thyroid": 0.479, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.112, "Adrenals": 0.116, "Brain": 0.002, "Breast": 0.801, "Colon": 0.015, "Endosteum": 0.17, "Extrathoracic_Airways": 0.104, "Gallbladder": 0.194, "Heart": 0.551, "Kidneys": 0.069, "Liver": 0.335, "Lungs": 0.393, "Lymph_Nodes": 0.179, "Muscle": 0.108, "Oesophagus": 0.209, "Oral_Mucosa": 0.011, "Ovaries": 0.004, "Pancreas": 0.273, "Prostate": 0.0, "Salivary_Glands": 0.016, "Skin": 0.125, "Small_Intestine": 0.016, "Spleen": 0.167, "Stomach": 0.374, "Testicles": 0.0, "Thymus": 0.813, "Thyroid": 0.477, "Urinary_Bladder": 0.002, "Uterus": 0.003},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.117, "Adrenals": 0.124, "Brain": 0.002, "Breast": 0.826, "Colon": 0.016, "Endosteum": 0.176, "Extrathoracic_Airways": 0.108, "Gallbladder": 0.206, "Heart": 0.58, "Kidneys": 0.074, "Liver": 0.354, "Lungs": 0.414, "Lymph_Nodes": 0.188, "Muscle": 0.113, "Oesophagus": 0.222, "Oral_Mucosa": 0.012, "Ovaries": 0.004, "Pancreas": 0.291, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.128, "Small_Intestine": 0.018, "Spleen": 0.178, "Stomach": 0.394, "Testicles": 0.0, "Thymus": 0.85, "Thyroid": 0.497, "Urinary_Bladder": 0.002, "Uterus": 0.003},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.122, "Adrenals": 0.132, "Brain": 0.002, "Breast": 0.846, "Colon": 0.017, "Endosteum": 0.182, "Extrathoracic_Airways": 0.111, "Gallbladder": 0.216, "Heart": 0.605, "Kidneys": 0.078, "Liver": 0.37, "Lungs": 0.433, "Lymph_Nodes": 0.197, "Muscle": 0.117, "Oesophagus": 0.235, "Oral_Mucosa": 0.013, "Ovaries": 0.005, "Pancreas": 0.306, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.13, "Small_Intestine": 0.019, "Spleen": 0.188, "Stomach": 0.411, "Testicles": 0.0, "Thymus": 0.882, "Thyroid": 0.514, "Urinary_Bladder": 0.002, "Uterus": 0.003},
                },
            },
            "Pelvis": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.027, "Adrenals": 0.0, "Brain": 0.0, "Breast": 0.0, "Colon": 0.189, "Endosteum": 0.037, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.018, "Heart": 0.0, "Kidneys": 0.004, "Liver": 0.005, "Lungs": 0.0, "Lymph_Nodes": 0.061, "Muscle": 0.094, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.17, "Pancreas": 0.002, "Prostate": 0.297, "Salivary_Glands": 0.0, "Skin": 0.101, "Small_Intestine": 0.187, "Spleen": 0.002, "Stomach": 0.008, "Testicles": 0.817, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.486, "Uterus": 0.23},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.031, "Adrenals": 0.0, "Brain": 0.0, "Breast": 0.0, "Colon": 0.208, "Endosteum": 0.041, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.02, "Heart": 0.0, "Kidneys": 0.004, "Liver": 0.005, "Lungs": 0.0, "Lymph_Nodes": 0.067, "Muscle": 0.101, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.187, "Pancreas": 0.002, "Prostate": 0.33, "Salivary_Glands": 0.0, "Skin": 0.103, "Small_Intestine": 0.206, "Spleen": 0.003, "Stomach": 0.008, "Testicles": 0.86, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.529, "Uterus": 0.255},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.034, "Adrenals": 0.0, "Brain": 0.0, "Breast": 0.0, "Colon": 0.226, "Endosteum": 0.045, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.022, "Heart": 0.0, "Kidneys": 0.005, "Liver": 0.006, "Lungs": 0.0, "Lymph_Nodes": 0.072, "Muscle": 0.107, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.203, "Pancreas": 0.002, "Prostate": 0.36, "Salivary_Glands": 0.0, "Skin": 0.105, "Small_Intestine": 0.223, "Spleen": 0.003, "Stomach": 0.009, "Testicles": 0.896, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.567, "Uterus": 0.278},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.05, "Adrenals": 0.001, "Brain": 0.0, "Breast": 0.001, "Colon": 0.285, "Endosteum": 0.062, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.031, "Heart": 0.0, "Kidneys": 0.009, "Liver": 0.009, "Lungs": 0.0, "Lymph_Nodes": 0.09, "Muscle": 0.123, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.266, "Pancreas": 0.005, "Prostate": 0.45, "Salivary_Glands": 0.0, "Skin": 0.109, "Small_Intestine": 0.282, "Spleen": 0.005, "Stomach": 0.015, "Testicles": 0.942, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.652, "Uterus": 0.358},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.055, "Adrenals": 0.002, "Brain": 0.0, "Breast": 0.001, "Colon": 0.31, "Endosteum": 0.069, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.035, "Heart": 0.0, "Kidneys": 0.01, "Liver": 0.01, "Lungs": 0.0, "Lymph_Nodes": 0.097, "Muscle": 0.131, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.29, "Pancreas": 0.006, "Prostate": 0.491, "Salivary_Glands": 0.0, "Skin": 0.112, "Small_Intestine": 0.307, "Spleen": 0.005, "Stomach": 0.016, "Testicles": 0.985, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.701, "Uterus": 0.391},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.059, "Adrenals": 0.002, "Brain": 0.0, "Breast": 0.001, "Colon": 0.332, "Endosteum": 0.074, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.037, "Heart": 0.0, "Kidneys": 0.011, "Liver": 0.011, "Lungs": 0.0, "Lymph_Nodes": 0.104, "Muscle": 0.138, "Oesophagus": 0.0, "Oral_Mucosa": 0.0, "Ovaries": 0.311, "Pancreas": 0.006, "Prostate": 0.527, "Salivary_Glands": 0.0, "Skin": 0.114, "Small_Intestine": 0.329, "Spleen": 0.006, "Stomach": 0.018, "Testicles": 1.021, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.744, "Uterus": 0.42},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.072, "Adrenals": 0.003, "Brain": 0.0, "Breast": 0.001, "Colon": 0.369, "Endosteum": 0.085, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.043, "Heart": 0.001, "Kidneys": 0.015, "Liver": 0.013, "Lungs": 0.001, "Lymph_Nodes": 0.115, "Muscle": 0.148, "Oesophagus": 0.001, "Oral_Mucosa": 0.0, "Ovaries": 0.35, "Pancreas": 0.008, "Prostate": 0.58, "Salivary_Glands": 0.0, "Skin": 0.117, "Small_Intestine": 0.365, "Spleen": 0.007, "Stomach": 0.022, "Testicles": 1.039, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.793, "Uterus": 0.47},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.078, "Adrenals": 0.003, "Brain": 0.0, "Breast": 0.001, "Colon": 0.396, "Endosteum": 0.092, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.047, "Heart": 0.001, "Kidneys": 0.016, "Liver": 0.014, "Lungs": 0.001, "Lymph_Nodes": 0.122, "Muscle": 0.156, "Oesophagus": 0.001, "Oral_Mucosa": 0.0, "Ovaries": 0.376, "Pancreas": 0.009, "Prostate": 0.623, "Salivary_Glands": 0.0, "Skin": 0.119, "Small_Intestine": 0.391, "Spleen": 0.008, "Stomach": 0.024, "Testicles": 1.079, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.841, "Uterus": 0.504},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.083, "Adrenals": 0.004, "Brain": 0.0, "Breast": 0.001, "Colon": 0.419, "Endosteum": 0.098, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.05, "Heart": 0.001, "Kidneys": 0.017, "Liver": 0.015, "Lungs": 0.001, "Lymph_Nodes": 0.129, "Muscle": 0.163, "Oesophagus": 0.001, "Oral_Mucosa": 0.0, "Ovaries": 0.398, "Pancreas": 0.01, "Prostate": 0.661, "Salivary_Glands": 0.0, "Skin": 0.121, "Small_Intestine": 0.414, "Spleen": 0.009, "Stomach": 0.026, "Testicles": 1.112, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.884, "Uterus": 0.535},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.09, "Adrenals": 0.004, "Brain": 0.0, "Breast": 0.002, "Colon": 0.436, "Endosteum": 0.101, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.053, "Heart": 0.001, "Kidneys": 0.02, "Liver": 0.016, "Lungs": 0.001, "Lymph_Nodes": 0.134, "Muscle": 0.167, "Oesophagus": 0.001, "Oral_Mucosa": 0.0, "Ovaries": 0.422, "Pancreas": 0.012, "Prostate": 0.677, "Salivary_Glands": 0.0, "Skin": 0.122, "Small_Intestine": 0.431, "Spleen": 0.01, "Stomach": 0.028, "Testicles": 1.109, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.899, "Uterus": 0.556},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.096, "Adrenals": 0.005, "Brain": 0.0, "Breast": 0.002, "Colon": 0.461, "Endosteum": 0.108, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.057, "Heart": 0.001, "Kidneys": 0.021, "Liver": 0.017, "Lungs": 0.001, "Lymph_Nodes": 0.142, "Muscle": 0.175, "Oesophagus": 0.002, "Oral_Mucosa": 0.0, "Ovaries": 0.447, "Pancreas": 0.013, "Prostate": 0.717, "Salivary_Glands": 0.0, "Skin": 0.124, "Small_Intestine": 0.456, "Spleen": 0.011, "Stomach": 0.03, "Testicles": 1.144, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.944, "Uterus": 0.589},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.102, "Adrenals": 0.005, "Brain": 0.0, "Breast": 0.002, "Colon": 0.483, "Endosteum": 0.114, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.06, "Heart": 0.001, "Kidneys": 0.022, "Liver": 0.018, "Lungs": 0.001, "Lymph_Nodes": 0.148, "Muscle": 0.182, "Oesophagus": 0.002, "Oral_Mucosa": 0.0, "Ovaries": 0.469, "Pancreas": 0.014, "Prostate": 0.752, "Salivary_Glands": 0.0, "Skin": 0.126, "Small_Intestine": 0.478, "Spleen": 0.012, "Stomach": 0.032, "Testicles": 1.172, "Thymus": 0.0, "Thyroid": 0.0, "Urinary_Bladder": 0.983, "Uterus": 0.618},
                },
            },
            "Lumbar Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.041, "Adrenals": 0.069, "Brain": 0.0, "Breast": 0.007, "Colon": 0.322, "Endosteum": 0.056, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.5, "Heart": 0.034, "Kidneys": 0.061, "Liver": 0.31, "Lungs": 0.018, "Lymph_Nodes": 0.193, "Muscle": 0.093, "Oesophagus": 0.059, "Oral_Mucosa": 0.0, "Ovaries": 0.26, "Pancreas": 0.207, "Prostate": 0.016, "Salivary_Glands": 0.0, "Skin": 0.097, "Small_Intestine": 0.344, "Spleen": 0.068, "Stomach": 0.531, "Testicles": 0.002, "Thymus": 0.001, "Thyroid": 0.0, "Urinary_Bladder": 0.1, "Uterus": 0.336},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.044, "Adrenals": 0.077, "Brain": 0.0, "Breast": 0.008, "Colon": 0.353, "Endosteum": 0.06, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.549, "Heart": 0.038, "Kidneys": 0.068, "Liver": 0.339, "Lungs": 0.02, "Lymph_Nodes": 0.213, "Muscle": 0.1, "Oesophagus": 0.065, "Oral_Mucosa": 0.0, "Ovaries": 0.29, "Pancreas": 0.231, "Prostate": 0.018, "Salivary_Glands": 0.0, "Skin": 0.099, "Small_Intestine": 0.38, "Spleen": 0.076, "Stomach": 0.578, "Testicles": 0.002, "Thymus": 0.001, "Thyroid": 0.0, "Urinary_Bladder": 0.111, "Uterus": 0.371},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.085, "Brain": 0.0, "Breast": 0.009, "Colon": 0.382, "Endosteum": 0.064, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.594, "Heart": 0.042, "Kidneys": 0.076, "Liver": 0.364, "Lungs": 0.022, "Lymph_Nodes": 0.231, "Muscle": 0.106, "Oesophagus": 0.071, "Oral_Mucosa": 0.0, "Ovaries": 0.317, "Pancreas": 0.253, "Prostate": 0.02, "Salivary_Glands": 0.0, "Skin": 0.1, "Small_Intestine": 0.412, "Spleen": 0.083, "Stomach": 0.62, "Testicles": 0.002, "Thymus": 0.001, "Thyroid": 0.0, "Urinary_Bladder": 0.121, "Uterus": 0.403},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.066, "Adrenals": 0.117, "Brain": 0.0, "Breast": 0.013, "Colon": 0.466, "Endosteum": 0.078, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.724, "Heart": 0.063, "Kidneys": 0.116, "Liver": 0.432, "Lungs": 0.032, "Lymph_Nodes": 0.29, "Muscle": 0.124, "Oesophagus": 0.094, "Oral_Mucosa": 0.0, "Ovaries": 0.404, "Pancreas": 0.336, "Prostate": 0.032, "Salivary_Glands": 0.0, "Skin": 0.105, "Small_Intestine": 0.522, "Spleen": 0.119, "Stomach": 0.726, "Testicles": 0.005, "Thymus": 0.003, "Thyroid": 0.0, "Urinary_Bladder": 0.151, "Uterus": 0.492},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.072, "Adrenals": 0.128, "Brain": 0.0, "Breast": 0.014, "Colon": 0.505, "Endosteum": 0.083, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.785, "Heart": 0.069, "Kidneys": 0.128, "Liver": 0.466, "Lungs": 0.035, "Lymph_Nodes": 0.315, "Muscle": 0.132, "Oesophagus": 0.103, "Oral_Mucosa": 0.0, "Ovaries": 0.441, "Pancreas": 0.368, "Prostate": 0.036, "Salivary_Glands": 0.0, "Skin": 0.107, "Small_Intestine": 0.568, "Spleen": 0.13, "Stomach": 0.781, "Testicles": 0.006, "Thymus": 0.004, "Thyroid": 0.0, "Urinary_Bladder": 0.164, "Uterus": 0.535},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.077, "Adrenals": 0.139, "Brain": 0.0, "Breast": 0.015, "Colon": 0.539, "Endosteum": 0.088, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.838, "Heart": 0.075, "Kidneys": 0.139, "Liver": 0.496, "Lungs": 0.038, "Lymph_Nodes": 0.338, "Muscle": 0.139, "Oesophagus": 0.111, "Oral_Mucosa": 0.0, "Ovaries": 0.475, "Pancreas": 0.397, "Prostate": 0.039, "Salivary_Glands": 0.0, "Skin": 0.109, "Small_Intestine": 0.609, "Spleen": 0.141, "Stomach": 0.829, "Testicles": 0.006, "Thymus": 0.004, "Thyroid": 0.0, "Urinary_Bladder": 0.176, "Uterus": 0.573},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.092, "Adrenals": 0.165, "Brain": 0.0, "Breast": 0.018, "Colon": 0.591, "Endosteum": 0.096, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.925, "Heart": 0.09, "Kidneys": 0.171, "Liver": 0.537, "Lungs": 0.046, "Lymph_Nodes": 0.376, "Muscle": 0.15, "Oesophagus": 0.132, "Oral_Mucosa": 0.0, "Ovaries": 0.522, "Pancreas": 0.453, "Prostate": 0.048, "Salivary_Glands": 0.0, "Skin": 0.111, "Small_Intestine": 0.675, "Spleen": 0.168, "Stomach": 0.892, "Testicles": 0.008, "Thymus": 0.007, "Thyroid": 0.0, "Urinary_Bladder": 0.195, "Uterus": 0.631},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.098, "Adrenals": 0.179, "Brain": 0.0, "Breast": 0.019, "Colon": 0.632, "Endosteum": 0.101, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.988, "Heart": 0.097, "Kidneys": 0.186, "Liver": 0.571, "Lungs": 0.049, "Lymph_Nodes": 0.403, "Muscle": 0.159, "Oesophagus": 0.142, "Oral_Mucosa": 0.0, "Ovaries": 0.561, "Pancreas": 0.487, "Prostate": 0.052, "Salivary_Glands": 0.0, "Skin": 0.114, "Small_Intestine": 0.723, "Spleen": 0.182, "Stomach": 0.948, "Testicles": 0.009, "Thymus": 0.008, "Thyroid": 0.0, "Urinary_Bladder": 0.21, "Uterus": 0.676},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.105, "Adrenals": 0.191, "Brain": 0.0, "Breast": 0.02, "Colon": 0.667, "Endosteum": 0.106, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.043, "Heart": 0.104, "Kidneys": 0.199, "Liver": 0.602, "Lungs": 0.052, "Lymph_Nodes": 0.426, "Muscle": 0.167, "Oesophagus": 0.151, "Oral_Mucosa": 0.0, "Ovaries": 0.596, "Pancreas": 0.518, "Prostate": 0.056, "Salivary_Glands": 0.0, "Skin": 0.115, "Small_Intestine": 0.766, "Spleen": 0.194, "Stomach": 0.997, "Testicles": 0.009, "Thymus": 0.008, "Thyroid": 0.0, "Urinary_Bladder": 0.222, "Uterus": 0.715},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.114, "Adrenals": 0.207, "Brain": 0.0, "Breast": 0.022, "Colon": 0.692, "Endosteum": 0.109, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.076, "Heart": 0.11, "Kidneys": 0.217, "Liver": 0.618, "Lungs": 0.056, "Lymph_Nodes": 0.444, "Muscle": 0.171, "Oesophagus": 0.163, "Oral_Mucosa": 0.0, "Ovaries": 0.622, "Pancreas": 0.551, "Prostate": 0.059, "Salivary_Glands": 0.0, "Skin": 0.117, "Small_Intestine": 0.794, "Spleen": 0.211, "Stomach": 1.017, "Testicles": 0.011, "Thymus": 0.011, "Thyroid": 0.0, "Urinary_Bladder": 0.227, "Uterus": 0.737},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.121, "Adrenals": 0.221, "Brain": 0.0, "Breast": 0.023, "Colon": 0.731, "Endosteum": 0.114, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.136, "Heart": 0.118, "Kidneys": 0.232, "Liver": 0.65, "Lungs": 0.06, "Lymph_Nodes": 0.469, "Muscle": 0.18, "Oesophagus": 0.174, "Oral_Mucosa": 0.0, "Ovaries": 0.659, "Pancreas": 0.586, "Prostate": 0.063, "Salivary_Glands": 0.0, "Skin": 0.119, "Small_Intestine": 0.841, "Spleen": 0.224, "Stomach": 1.069, "Testicles": 0.011, "Thymus": 0.011, "Thyroid": 0.0, "Urinary_Bladder": 0.241, "Uterus": 0.779},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.127, "Adrenals": 0.233, "Brain": 0.0, "Breast": 0.024, "Colon": 0.764, "Endosteum": 0.119, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.188, "Heart": 0.124, "Kidneys": 0.245, "Liver": 0.678, "Lungs": 0.063, "Lymph_Nodes": 0.492, "Muscle": 0.187, "Oesophagus": 0.183, "Oral_Mucosa": 0.0, "Ovaries": 0.693, "Pancreas": 0.616, "Prostate": 0.067, "Salivary_Glands": 0.0, "Skin": 0.12, "Small_Intestine": 0.881, "Spleen": 0.237, "Stomach": 1.114, "Testicles": 0.012, "Thymus": 0.012, "Thyroid": 0.0, "Urinary_Bladder": 0.252, "Uterus": 0.816},
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.085, "Adrenals": 0.056, "Brain": 0.0, "Breast": 0.598, "Colon": 0.002, "Endosteum": 0.158, "Extrathoracic_Airways": 0.017, "Gallbladder": 0.045, "Heart": 0.59, "Kidneys": 0.016, "Liver": 0.207, "Lungs": 0.305, "Lymph_Nodes": 0.097, "Muscle": 0.075, "Oesophagus": 0.138, "Oral_Mucosa": 0.003, "Ovaries": 0.0, "Pancreas": 0.165, "Prostate": 0.0, "Salivary_Glands": 0.005, "Skin": 0.106, "Small_Intestine": 0.002, "Spleen": 0.051, "Stomach": 0.221, "Testicles": 0.0, "Thymus": 1.052, "Thyroid": 0.047, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.091, "Adrenals": 0.063, "Brain": 0.0, "Breast": 0.629, "Colon": 0.002, "Endosteum": 0.168, "Extrathoracic_Airways": 0.019, "Gallbladder": 0.05, "Heart": 0.642, "Kidneys": 0.018, "Liver": 0.226, "Lungs": 0.335, "Lymph_Nodes": 0.107, "Muscle": 0.081, "Oesophagus": 0.155, "Oral_Mucosa": 0.004, "Ovaries": 0.0, "Pancreas": 0.184, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.109, "Small_Intestine": 0.003, "Spleen": 0.056, "Stomach": 0.241, "Testicles": 0.0, "Thymus": 1.134, "Thyroid": 0.052, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.095, "Adrenals": 0.07, "Brain": 0.0, "Breast": 0.655, "Colon": 0.003, "Endosteum": 0.176, "Extrathoracic_Airways": 0.021, "Gallbladder": 0.055, "Heart": 0.689, "Kidneys": 0.02, "Liver": 0.244, "Lungs": 0.361, "Lymph_Nodes": 0.115, "Muscle": 0.087, "Oesophagus": 0.17, "Oral_Mucosa": 0.004, "Ovaries": 0.0, "Pancreas": 0.201, "Prostate": 0.0, "Salivary_Glands": 0.007, "Skin": 0.111, "Small_Intestine": 0.003, "Spleen": 0.062, "Stomach": 0.259, "Testicles": 0.0, "Thymus": 1.205, "Thyroid": 0.057, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.111, "Adrenals": 0.111, "Brain": 0.001, "Breast": 0.701, "Colon": 0.005, "Endosteum": 0.191, "Extrathoracic_Airways": 0.029, "Gallbladder": 0.08, "Heart": 0.812, "Kidneys": 0.035, "Liver": 0.297, "Lungs": 0.433, "Lymph_Nodes": 0.147, "Muscle": 0.103, "Oesophagus": 0.241, "Oral_Mucosa": 0.006, "Ovaries": 0.0, "Pancreas": 0.272, "Prostate": 0.0, "Salivary_Glands": 0.011, "Skin": 0.117, "Small_Intestine": 0.006, "Spleen": 0.087, "Stomach": 0.311, "Testicles": 0.0, "Thymus": 1.349, "Thyroid": 0.087, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.118, "Adrenals": 0.123, "Brain": 0.001, "Breast": 0.733, "Colon": 0.005, "Endosteum": 0.202, "Extrathoracic_Airways": 0.032, "Gallbladder": 0.087, "Heart": 0.874, "Kidneys": 0.039, "Liver": 0.321, "Lungs": 0.468, "Lymph_Nodes": 0.16, "Muscle": 0.11, "Oesophagus": 0.264, "Oral_Mucosa": 0.007, "Ovaries": 0.0, "Pancreas": 0.298, "Prostate": 0.0, "Salivary_Glands": 0.012, "Skin": 0.12, "Small_Intestine": 0.006, "Spleen": 0.095, "Stomach": 0.335, "Testicles": 0.0, "Thymus": 1.439, "Thyroid": 0.095, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.124, "Adrenals": 0.134, "Brain": 0.001, "Breast": 0.761, "Colon": 0.006, "Endosteum": 0.211, "Extrathoracic_Airways": 0.034, "Gallbladder": 0.094, "Heart": 0.929, "Kidneys": 0.042, "Liver": 0.342, "Lungs": 0.5, "Lymph_Nodes": 0.171, "Muscle": 0.117, "Oesophagus": 0.286, "Oral_Mucosa": 0.007, "Ovaries": 0.0, "Pancreas": 0.322, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.122, "Small_Intestine": 0.007, "Spleen": 0.103, "Stomach": 0.357, "Testicles": 0.0, "Thymus": 1.517, "Thyroid": 0.102, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.135, "Adrenals": 0.158, "Brain": 0.001, "Breast": 0.78, "Colon": 0.008, "Endosteum": 0.215, "Extrathoracic_Airways": 0.041, "Gallbladder": 0.11, "Heart": 0.998, "Kidneys": 0.055, "Liver": 0.373, "Lungs": 0.542, "Lymph_Nodes": 0.192, "Muscle": 0.126, "Oesophagus": 0.34, "Oral_Mucosa": 0.01, "Ovaries": 0.0, "Pancreas": 0.367, "Prostate": 0.0, "Salivary_Glands": 0.016, "Skin": 0.126, "Small_Intestine": 0.01, "Spleen": 0.123, "Stomach": 0.384, "Testicles": 0.0, "Thymus": 1.582, "Thyroid": 0.124, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.142, "Adrenals": 0.172, "Brain": 0.002, "Breast": 0.81, "Colon": 0.008, "Endosteum": 0.225, "Extrathoracic_Airways": 0.044, "Gallbladder": 0.119, "Heart": 1.061, "Kidneys": 0.059, "Liver": 0.397, "Lungs": 0.578, "Lymph_Nodes": 0.205, "Muscle": 0.134, "Oesophagus": 0.367, "Oral_Mucosa": 0.011, "Ovaries": 0.0, "Pancreas": 0.395, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.129, "Small_Intestine": 0.01, "Spleen": 0.132, "Stomach": 0.409, "Testicles": 0.0, "Thymus": 1.669, "Thyroid": 0.134, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.149, "Adrenals": 0.184, "Brain": 0.002, "Breast": 0.836, "Colon": 0.009, "Endosteum": 0.233, "Extrathoracic_Airways": 0.047, "Gallbladder": 0.127, "Heart": 1.116, "Kidneys": 0.064, "Liver": 0.419, "Lungs": 0.61, "Lymph_Nodes": 0.218, "Muscle": 0.141, "Oesophagus": 0.392, "Oral_Mucosa": 0.011, "Ovaries": 0.0, "Pancreas": 0.421, "Prostate": 0.0, "Salivary_Glands": 0.018, "Skin": 0.131, "Small_Intestine": 0.011, "Spleen": 0.141, "Stomach": 0.431, "Testicles": 0.0, "Thymus": 1.743, "Thyroid": 0.142, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.155, "Adrenals": 0.203, "Brain": 0.002, "Breast": 0.836, "Colon": 0.01, "Endosteum": 0.231, "Extrathoracic_Airways": 0.051, "Gallbladder": 0.135, "Heart": 1.139, "Kidneys": 0.071, "Liver": 0.432, "Lungs": 0.627, "Lymph_Nodes": 0.227, "Muscle": 0.145, "Oesophagus": 0.418, "Oral_Mucosa": 0.013, "Ovaries": 0.001, "Pancreas": 0.441, "Prostate": 0.0, "Salivary_Glands": 0.02, "Skin": 0.133, "Small_Intestine": 0.013, "Spleen": 0.154, "Stomach": 0.44, "Testicles": 0.0, "Thymus": 1.749, "Thyroid": 0.156, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.162, "Adrenals": 0.217, "Brain": 0.002, "Breast": 0.863, "Colon": 0.011, "Endosteum": 0.239, "Extrathoracic_Airways": 0.054, "Gallbladder": 0.144, "Heart": 1.197, "Kidneys": 0.076, "Liver": 0.456, "Lungs": 0.661, "Lymph_Nodes": 0.24, "Muscle": 0.152, "Oesophagus": 0.446, "Oral_Mucosa": 0.014, "Ovaries": 0.001, "Pancreas": 0.469, "Prostate": 0.0, "Salivary_Glands": 0.022, "Skin": 0.136, "Small_Intestine": 0.014, "Spleen": 0.164, "Stomach": 0.463, "Testicles": 0.0, "Thymus": 1.826, "Thyroid": 0.166, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.168, "Adrenals": 0.23, "Brain": 0.002, "Breast": 0.886, "Colon": 0.012, "Endosteum": 0.246, "Extrathoracic_Airways": 0.057, "Gallbladder": 0.152, "Heart": 1.248, "Kidneys": 0.081, "Liver": 0.476, "Lungs": 0.69, "Lymph_Nodes": 0.252, "Muscle": 0.158, "Oesophagus": 0.47, "Oral_Mucosa": 0.014, "Ovaries": 0.001, "Pancreas": 0.493, "Prostate": 0.0, "Salivary_Glands": 0.023, "Skin": 0.138, "Small_Intestine": 0.015, "Spleen": 0.173, "Stomach": 0.483, "Testicles": 0.0, "Thymus": 1.892, "Thyroid": 0.175, "Urinary_Bladder": 0.001, "Uterus": 0.002},
                },
            },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.039, "Adrenals": 0.0, "Brain": 0.031, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.185, "Extrathoracic_Airways": 0.99, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.009, "Lymph_Nodes": 0.315, "Muscle": 0.041, "Oesophagus": 0.022, "Oral_Mucosa": 0.967, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.748, "Skin": 0.144, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.005, "Thyroid": 2.431, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.042, "Adrenals": 0.0, "Brain": 0.035, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.202, "Extrathoracic_Airways": 1.064, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.01, "Lymph_Nodes": 0.337, "Muscle": 0.043, "Oesophagus": 0.025, "Oral_Mucosa": 1.033, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.811, "Skin": 0.148, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.006, "Thyroid": 2.565, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.046, "Adrenals": 0.0, "Brain": 0.039, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.217, "Extrathoracic_Airways": 1.129, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.355, "Muscle": 0.046, "Oesophagus": 0.027, "Oral_Mucosa": 1.09, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.867, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.006, "Thyroid": 2.678, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.058, "Adrenals": 0.0, "Brain": 0.066, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.252, "Extrathoracic_Airways": 1.295, "Gallbladder": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.015, "Lymph_Nodes": 0.392, "Muscle": 0.052, "Oesophagus": 0.038, "Oral_Mucosa": 1.234, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.999, "Skin": 0.157, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.009, "Thyroid": 2.744, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.063, "Adrenals": 0.0, "Brain": 0.073, "Breast": 0.004, "Colon": 0.0, "Endosteum": 0.271, "Extrathoracic_Airways": 1.381, "Gallbladder": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.016, "Lymph_Nodes": 0.416, "Muscle": 0.055, "Oesophagus": 0.041, "Oral_Mucosa": 1.31, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.071, "Skin": 0.162, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.01, "Thyroid": 2.868, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.068, "Adrenals": 0.0, "Brain": 0.08, "Breast": 0.004, "Colon": 0.0, "Endosteum": 0.288, "Extrathoracic_Airways": 1.456, "Gallbladder": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.018, "Lymph_Nodes": 0.436, "Muscle": 0.058, "Oesophagus": 0.045, "Oral_Mucosa": 1.376, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.135, "Skin": 0.165, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.011, "Thyroid": 2.97, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.077, "Adrenals": 0.0, "Brain": 0.101, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.302, "Extrathoracic_Airways": 1.559, "Gallbladder": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.457, "Muscle": 0.061, "Oesophagus": 0.052, "Oral_Mucosa": 1.477, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.218, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.014, "Thyroid": 2.967, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.082, "Adrenals": 0.0, "Brain": 0.109, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.32, "Extrathoracic_Airways": 1.646, "Gallbladder": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.48, "Muscle": 0.064, "Oesophagus": 0.056, "Oral_Mucosa": 1.555, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.291, "Skin": 0.173, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.015, "Thyroid": 3.073, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.087, "Adrenals": 0.001, "Brain": 0.117, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.336, "Extrathoracic_Airways": 1.723, "Gallbladder": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.024, "Lymph_Nodes": 0.5, "Muscle": 0.067, "Oesophagus": 0.06, "Oral_Mucosa": 1.622, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.354, "Skin": 0.176, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.017, "Thyroid": 3.161, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.092, "Adrenals": 0.001, "Brain": 0.129, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.335, "Extrathoracic_Airways": 1.756, "Gallbladder": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.506, "Muscle": 0.068, "Oesophagus": 0.063, "Oral_Mucosa": 1.671, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.395, "Skin": 0.177, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.02, "Thyroid": 3.125, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.001, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.351, "Extrathoracic_Airways": 1.838, "Gallbladder": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.027, "Lymph_Nodes": 0.527, "Muscle": 0.071, "Oesophagus": 0.067, "Oral_Mucosa": 1.745, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.464, "Skin": 0.181, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.022, "Thyroid": 3.215, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.102, "Adrenals": 0.001, "Brain": 0.147, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.365, "Extrathoracic_Airways": 1.908, "Gallbladder": 0.0, "Heart": 0.007, "Kidneys": 0.001, "Liver": 0.001, "Lungs": 0.029, "Lymph_Nodes": 0.545, "Muscle": 0.074, "Oesophagus": 0.071, "Oral_Mucosa": 1.809, "Ovaries": 0.0, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.524, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.001, "Testicles": 0.0, "Thymus": 0.023, "Thyroid": 3.288, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.058, 'Adrenals': 0, 'Brain': 0.355, 'Breast': 0.001, 'Colon': 0, 'Extrathoracic_Airways': 0.291, 'Gallbladder': 0, 'Heart': 0, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.001, 'Lymph_Nodes': 0.097, 'Muscle': 0.022, 'Oesophagus': 0, 'Oral_Mucosa': 0.978, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 0.674, 'Endosteum': 0.307, 'Skin': 0.123, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.001, 'Thyroid': 0.028, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.064, 'Adrenals': 0, 'Brain': 0.394, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.323, 'Gallbladder': 0, 'Heart': 0, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.002, 'Lymph_Nodes': 0.106, 'Muscle': 0.023, 'Oesophagus': 0, 'Oral_Mucosa': 1.048, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 0.729, 'Endosteum': 0.332, 'Skin': 0.126, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.031, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.068, 'Adrenals': 0, 'Brain': 0.43, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.352, 'Gallbladder': 0, 'Heart': 0, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.002, 'Lymph_Nodes': 0.114, 'Muscle': 0.024, 'Oesophagus': 0, 'Oral_Mucosa': 1.109, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 0.777, 'Endosteum': 0.354, 'Skin': 0.128, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.034, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.082, 'Adrenals': 0, 'Brain': 0.569, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.465, 'Gallbladder': 0, 'Heart': 0, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.003, 'Lymph_Nodes': 0.139, 'Muscle': 0.027, 'Oesophagus': 0.001, 'Oral_Mucosa': 1.266, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 0.899, 'Endosteum': 0.395, 'Skin': 0.133, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.001, 'Thyroid': 0.041, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.089, 'Adrenals': 0, 'Brain': 0.622, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.508, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.003, 'Lymph_Nodes': 0.15, 'Muscle': 0.029, 'Oesophagus': 0.001, 'Oral_Mucosa': 1.347, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 0.962, 'Endosteum': 0.423, 'Skin': 0.136, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.044, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.094, 'Adrenals': 0, 'Brain': 0.67, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.547, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.003, 'Lymph_Nodes': 0.16, 'Muscle': 0.03, 'Oesophagus': 0.001, 'Oral_Mucosa': 1.419, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.018, 'Endosteum': 0.447, 'Skin': 0.139, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.047, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.104, 'Adrenals': 0, 'Brain': 0.763, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.628, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.004, 'Lymph_Nodes': 0.178, 'Muscle': 0.032, 'Oesophagus': 0.002, 'Oral_Mucosa': 1.514, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.098, 'Endosteum': 0.458, 'Skin': 0.142, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.056, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.111, 'Adrenals': 0, 'Brain': 0.821, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.675, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.005, 'Lymph_Nodes': 0.19, 'Muscle': 0.033, 'Oesophagus': 0.002, 'Oral_Mucosa': 1.596, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.163, 'Endosteum': 0.483, 'Skin': 0.145, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.06, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.116, 'Adrenals': 0, 'Brain': 0.872, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.718, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.005, 'Lymph_Nodes': 0.2, 'Muscle': 0.035, 'Oesophagus': 0.002, 'Oral_Mucosa': 1.668, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.219, 'Endosteum': 0.505, 'Skin': 0.148, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.064, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.121, 'Adrenals': 0.001, 'Brain': 0.923, 'Breast': 0.002, 'Colon': 0, 'Extrathoracic_Airways': 0.759, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.006, 'Lymph_Nodes': 0.209, 'Muscle': 0.036, 'Oesophagus': 0.003, 'Oral_Mucosa': 1.721, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.258, 'Endosteum': 0.498, 'Skin': 0.15, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.002, 'Thyroid': 0.069, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.128, 'Adrenals': 0.001, 'Brain': 0.98, 'Breast': 0.003, 'Colon': 0, 'Extrathoracic_Airways': 0.806, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.006, 'Lymph_Nodes': 0.221, 'Muscle': 0.037, 'Oesophagus': 0.004, 'Oral_Mucosa': 1.799, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.32, 'Endosteum': 0.52, 'Skin': 0.153, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.003, 'Thyroid': 0.073, 'Urinary_Bladder': 0, 'Uterus': 0},
                    (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.133, 'Adrenals': 0.001, 'Brain': 1.03, 'Breast': 0.003, 'Colon': 0, 'Extrathoracic_Airways': 0.847, 'Gallbladder': 0, 'Heart': 0.001, 'Kidneys': 0, 'Liver': 0, 'Lungs': 0.006, 'Lymph_Nodes': 0.231, 'Muscle': 0.038, 'Oesophagus': 0.004, 'Oral_Mucosa': 1.867, 'Ovaries': 0, 'Pancreas': 0, 'Prostate': 0, 'Salivary_Glands': 1.374, 'Endosteum': 0.538, 'Skin': 0.155, 'Small_Intestine': 0, 'Spleen': 0, 'Stomach': 0, 'Testicles': 0, 'Thymus': 0.003, 'Thyroid': 0.077, 'Urinary_Bladder': 0, 'Uterus': 0},
                },
            },
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
                'F': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.495296, 'Adrenals': 1.19799, 'Brain': 0.00620518, 'Breast': 5.54038, 'Colon': 4.33797, 'Endosteum': 0.507461, 'Extrathoracic_Airways': 0.030536, 'Gallbladder': 2.82782, 'Gonads': 2.73599, 'Heart': 2.2098, 'Kidneys': 1.81875, 'Lens': 0.0231146, 'Liver': 2.70908, 'Lungs': 1.36956, 'Lymph_Nodes': 1.32651, 'Muscle': 1.26585, 'Oesophagus': 0.745846, 'Oral_Mucosa': 0.0773897, 'Pancreas': 2.38281, 'Salivary_Glands': 0.0542352, 'Skin': 1.24304, 'Small_Intestine': 3.69283, 'Spleen': 1.81893, 'Stomach': 2.96169, 'Thymus': 0.271143, 'Thyroid': 0.0938613, 'Urinary_Bladder': 4.77657, 'Uterus': 2.67375},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.545978, 'Adrenals': 1.3205, 'Brain': 0.00702532, 'Breast': 5.64402, 'Colon': 4.58755, 'Endosteum': 0.560124, 'Extrathoracic_Airways': 0.0339022, 'Gallbladder': 3.05802, 'Gonads': 2.95742, 'Heart': 2.3749, 'Kidneys': 1.99207, 'Lens': 0.0255042, 'Liver': 2.92858, 'Lungs': 1.48376, 'Lymph_Nodes': 1.42938, 'Muscle': 1.33604, 'Oesophagus': 0.820906, 'Oral_Mucosa': 0.0849817, 'Pancreas': 2.58654, 'Salivary_Glands': 0.0603867, 'Skin': 1.27944, 'Small_Intestine': 3.95374, 'Spleen': 1.98508, 'Stomach': 3.19783, 'Thymus': 0.299437, 'Thyroid': 0.104971, 'Urinary_Bladder': 5.02135, 'Uterus': 2.88475},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.797858, 'Adrenals': 1.91751, 'Brain': 0.0118006, 'Breast': 6.0492, 'Colon': 5.59641, 'Endosteum': 0.823943, 'Extrathoracic_Airways': 0.0518086, 'Gallbladder': 4.07511, 'Gonads': 3.92868, 'Heart': 3.08451, 'Kidneys': 2.80487, 'Lens': 0.037699, 'Liver': 3.89703, 'Lungs': 1.99553, 'Lymph_Nodes': 1.90415, 'Muscle': 1.63181, 'Oesophagus': 1.1838, 'Oral_Mucosa': 0.122885, 'Pancreas': 3.50227, 'Salivary_Glands': 0.092801, 'Skin': 1.44003, 'Small_Intestine': 5.0555, 'Spleen': 2.75085, 'Stomach': 4.23413, 'Thymus': 0.440059, 'Thyroid': 0.164532, 'Urinary_Bladder': 5.98438, 'Uterus': 3.80226},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.591611, 'Adrenals': 1.43028, 'Brain': 0.00779976, 'Breast': 5.72909, 'Colon': 4.79697, 'Endosteum': 0.60766, 'Extrathoracic_Airways': 0.0369937, 'Gallbladder': 3.25758, 'Gonads': 3.14897, 'Heart': 2.51665, 'Kidneys': 2.14552, 'Lens': 0.0276696, 'Liver': 3.11884, 'Lungs': 1.58324, 'Lymph_Nodes': 1.51975, 'Muscle': 1.39568, 'Oesophagus': 0.887995, 'Oral_Mucosa': 0.0918192, 'Pancreas': 2.76425, 'Salivary_Glands': 0.066027, 'Skin': 1.31087, 'Small_Intestine': 4.17637, 'Spleen': 2.13131, 'Stomach': 3.4021, 'Thymus': 0.324936, 'Thyroid': 0.115217, 'Urinary_Bladder': 5.22484, 'Uterus': 3.0667},
                },
                'M': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.496029, 'Adrenals': 1.19821, 'Brain': 0.00618251, 'Breast': 5.53954, 'Colon': 4.33438, 'Endosteum': 0.506519, 'Extrathoracic_Airways': 0.0305365, 'Gallbladder': 2.82766, 'Gonads': 0.394756, 'Heart': 2.2122, 'Kidneys': 1.81874, 'Lens': 0.0231836, 'Liver': 2.71454, 'Lungs': 1.37312, 'Lymph_Nodes': 1.31226, 'Muscle': 1.28083, 'Oesophagus': 0.749651, 'Oral_Mucosa': 0.0771453, 'Pancreas': 2.38335, 'Salivary_Glands': 0.0541488, 'Skin': 1.23869, 'Small_Intestine': 3.69344, 'Spleen': 1.81778, 'Stomach': 2.96116, 'Thymus': 0.270768, 'Thyroid': 0.0933761, 'Urinary_Bladder': 3.93084, 'Prostate': 3.40315},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.546773, 'Adrenals': 1.3207, 'Brain': 0.00699972, 'Breast': 5.64343, 'Colon': 4.5838, 'Endosteum': 0.559051, 'Extrathoracic_Airways': 0.0339019, 'Gallbladder': 3.0578, 'Gonads': 0.428296, 'Heart': 2.37727, 'Kidneys': 1.99203, 'Lens': 0.0255887, 'Liver': 2.93421, 'Lungs': 1.48743, 'Lymph_Nodes': 1.41797, 'Muscle': 1.35135, 'Oesophagus': 0.824973, 'Oral_Mucosa': 0.0847192, 'Pancreas': 2.58707, 'Salivary_Glands': 0.0602922, 'Skin': 1.27496, 'Small_Intestine': 3.95434, 'Spleen': 1.98376, 'Stomach': 3.19721, 'Thymus': 0.299013, 'Thyroid': 0.104422, 'Urinary_Bladder': 4.18453, 'Prostate': 3.58892},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.79894, 'Adrenals': 1.9174, 'Brain': 0.0117584, 'Breast': 6.05136, 'Colon': 5.59231, 'Endosteum': 0.822159, 'Extrathoracic_Airways': 0.0517488, 'Gallbladder': 4.07414, 'Gonads': 0.581868, 'Heart': 3.08638, 'Kidneys': 2.80467, 'Lens': 0.0378724, 'Liver': 3.90295, 'Lungs': 1.99936, 'Lymph_Nodes': 1.90376, 'Muscle': 1.64818, 'Oesophagus': 1.18872, 'Oral_Mucosa': 0.122578, 'Pancreas': 3.50241, 'Salivary_Glands': 0.092672, 'Skin': 1.43506, 'Small_Intestine': 5.05609, 'Spleen': 2.74865, 'Stomach': 4.23283, 'Thymus': 0.43937, 'Thyroid': 0.163608, 'Urinary_Bladder': 5.22701, 'Prostate': 4.32799},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.59246, 'Adrenals': 1.43045, 'Brain': 0.00777141, 'Breast': 5.72883, 'Colon': 4.79311, 'Endosteum': 0.606464, 'Extrathoracic_Airways': 0.0369906, 'Gallbladder': 3.25729, 'Gonads': 0.457703, 'Heart': 2.51897, 'Kidneys': 2.14546, 'Lens': 0.0277698, 'Liver': 3.12458, 'Lungs': 1.58698, 'Lymph_Nodes': 1.51074, 'Muscle': 1.41125, 'Oesophagus': 0.892275, 'Oral_Mucosa': 0.091543, 'Pancreas': 2.76477, 'Salivary_Glands': 0.0659254, 'Skin': 1.30628, 'Small_Intestine': 4.17696, 'Spleen': 2.12982, 'Stomach': 3.40138, 'Thymus': 0.324466, 'Thyroid': 0.114609, 'Urinary_Bladder': 4.39893, 'Prostate': 3.74376},
                },
                'U': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.495663, 'Adrenals': 1.1981, 'Brain': 0.00619385, 'Breast': 5.53996, 'Colon': 4.33617, 'Endosteum': 0.50699, 'Extrathoracic_Airways': 0.0305362, 'Gallbladder': 2.82774, 'Gonads': 1.56537, 'Heart': 2.211, 'Kidneys': 1.81875, 'Lens': 0.0231491, 'Liver': 2.71181, 'Lungs': 1.37134, 'Lymph_Nodes': 1.31939, 'Muscle': 1.27334, 'Oesophagus': 0.747748, 'Oral_Mucosa': 0.0772675, 'Pancreas': 2.38308, 'Salivary_Glands': 0.054192, 'Skin': 1.24086, 'Small_Intestine': 3.69313, 'Spleen': 1.81836, 'Stomach': 2.96143, 'Thymus': 0.270956, 'Thyroid': 0.0936187, 'Urinary_Bladder': 4.3537, 'Uterus': 1.33687, 'Prostate': 1.70157},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.546375, 'Adrenals': 1.3206, 'Brain': 0.00701252, 'Breast': 5.64372, 'Colon': 4.58568, 'Endosteum': 0.559587, 'Extrathoracic_Airways': 0.033902, 'Gallbladder': 3.05791, 'Gonads': 1.69286, 'Heart': 2.37608, 'Kidneys': 1.99205, 'Lens': 0.0255464, 'Liver': 2.9314, 'Lungs': 1.4856, 'Lymph_Nodes': 1.42368, 'Muscle': 1.3437, 'Oesophagus': 0.82294, 'Oral_Mucosa': 0.0848505, 'Pancreas': 2.5868, 'Salivary_Glands': 0.0603394, 'Skin': 1.2772, 'Small_Intestine': 3.95404, 'Spleen': 1.98442, 'Stomach': 3.19752, 'Thymus': 0.299225, 'Thyroid': 0.104697, 'Urinary_Bladder': 4.60294, 'Uterus': 1.44237, 'Prostate': 1.79446},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.798399, 'Adrenals': 1.91745, 'Brain': 0.0117795, 'Breast': 6.05028, 'Colon': 5.59436, 'Endosteum': 0.823051, 'Extrathoracic_Airways': 0.0517787, 'Gallbladder': 4.07463, 'Gonads': 2.25528, 'Heart': 3.08545, 'Kidneys': 2.80477, 'Lens': 0.0377857, 'Liver': 3.89999, 'Lungs': 1.99745, 'Lymph_Nodes': 1.90395, 'Muscle': 1.64, 'Oesophagus': 1.18626, 'Oral_Mucosa': 0.122731, 'Pancreas': 3.50234, 'Salivary_Glands': 0.0927365, 'Skin': 1.43754, 'Small_Intestine': 5.0558, 'Spleen': 2.74975, 'Stomach': 4.23348, 'Thymus': 0.439714, 'Thyroid': 0.16407, 'Urinary_Bladder': 5.60569, 'Uterus': 1.90113, 'Prostate': 2.164},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.592035, 'Adrenals': 1.43037, 'Brain': 0.00778559, 'Breast': 5.72896, 'Colon': 4.79504, 'Endosteum': 0.607062, 'Extrathoracic_Airways': 0.0369921, 'Gallbladder': 3.25743, 'Gonads': 1.80334, 'Heart': 2.51781, 'Kidneys': 2.14549, 'Lens': 0.0277197, 'Liver': 3.12171, 'Lungs': 1.58511, 'Lymph_Nodes': 1.51525, 'Muscle': 1.40347, 'Oesophagus': 0.890135, 'Oral_Mucosa': 0.0916811, 'Pancreas': 2.76451, 'Salivary_Glands': 0.0659762, 'Skin': 1.30857, 'Small_Intestine': 4.17667, 'Spleen': 2.13057, 'Stomach': 3.40174, 'Thymus': 0.324701, 'Thyroid': 0.114913, 'Urinary_Bladder': 4.81189, 'Uterus': 1.53335, 'Prostate': 1.87188},
                },
            },
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.877, "Adrenals": 1.45, "Brain": 0.031, "Breast": 12.885, "Colon": 0.196, "Endosteum": 3.63, "Extrathoracic_Airways": 0.271, "Gallbladder": 2.379, "Heart": 5.562, "Kidneys": 0.836, "Liver": 3.528, "Lungs": 4.395, "Lymph_Nodes": 1.644, "Muscle": 1.334, "Oesophagus": 2.484, "Oral_Mucosa": 0.234, "Ovaries": 0.172, "Pancreas": 2.758, "Prostate": 0.0, "Salivary_Glands": 0.298, "Skin": 1.328, "Small_Intestine": 0.243, "Spleen": 1.984, "Stomach": 3.513, "Testicles": 0.006, "Thymus": 6.947, "Thyroid": 1.202, "Urinary_Bladder": 0.051, "Uterus": 0.109},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.935, "Adrenals": 1.583, "Brain": 0.035, "Breast": 13.002, "Colon": 0.215, "Endosteum": 3.848, "Extrathoracic_Airways": 0.297, "Gallbladder": 2.565, "Heart": 5.908, "Kidneys": 0.918, "Liver": 3.753, "Lungs": 4.678, "Lymph_Nodes": 1.764, "Muscle": 1.412, "Oesophagus": 2.692, "Oral_Mucosa": 0.256, "Ovaries": 0.19, "Pancreas": 2.979, "Prostate": 0.0, "Salivary_Glands": 0.324, "Skin": 1.361, "Small_Intestine": 0.268, "Spleen": 2.153, "Stomach": 3.732, "Testicles": 0.007, "Thymus": 7.262, "Thyroid": 1.291, "Urinary_Bladder": 0.057, "Uterus": 0.122},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 1.202, "Adrenals": 2.233, "Brain": 0.054, "Breast": 13.111, "Colon": 0.316, "Endosteum": 4.796, "Extrathoracic_Airways": 0.429, "Gallbladder": 3.392, "Heart": 7.375, "Kidneys": 1.332, "Liver": 4.731, "Lungs": 5.904, "Lymph_Nodes": 2.314, "Muscle": 1.762, "Oesophagus": 3.68, "Oral_Mucosa": 0.369, "Ovaries": 0.277, "Pancreas": 3.99, "Prostate": 0.0, "Salivary_Glands": 0.454, "Skin": 1.515, "Small_Intestine": 0.397, "Spleen": 2.952, "Stomach": 4.671, "Testicles": 0.014, "Thymus": 8.53, "Thyroid": 1.742, "Urinary_Bladder": 0.094, "Uterus": 0.194},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.985, "Adrenals": 1.702, "Brain": 0.038, "Breast": 13.075, "Colon": 0.233, "Endosteum": 4.035, "Extrathoracic_Airways": 0.321, "Gallbladder": 2.726, "Heart": 6.203, "Kidneys": 0.992, "Liver": 3.946, "Lungs": 4.92, "Lymph_Nodes": 1.87, "Muscle": 1.48, "Oesophagus": 2.875, "Oral_Mucosa": 0.276, "Ovaries": 0.206, "Pancreas": 3.172, "Prostate": 0.0, "Salivary_Glands": 0.347, "Skin": 1.391, "Small_Intestine": 0.29, "Spleen": 2.303, "Stomach": 3.918, "Testicles": 0.008, "Thymus": 7.524, "Thyroid": 1.37, "Urinary_Bladder": 0.063, "Uterus": 0.134},
                },
            },
            "Pelvis": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.406, "Adrenals": 0.114, "Brain": 0.001, "Breast": 0.0, "Colon": 4.981, "Endosteum": 1.198, "Extrathoracic_Airways": 0.003, "Gallbladder": 0.706, "Heart": 0.062, "Kidneys": 0.358, "Liver": 0.376, "Lungs": 0.052, "Lymph_Nodes": 1.465, "Muscle": 1.782, "Oesophagus": 0.068, "Oral_Mucosa": 0.001, "Ovaries": 5.362, "Pancreas": 0.331, "Prostate": 6.532, "Salivary_Glands": 0.007, "Skin": 1.308, "Small_Intestine": 4.94, "Spleen": 0.249, "Stomach": 0.456, "Testicles": 1.235, "Thymus": 0.011, "Thyroid": 0.012, "Urinary_Bladder": 8.639, "Uterus": 5.71},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.444, "Adrenals": 0.126, "Brain": 0.001, "Breast": 0.0, "Colon": 5.335, "Endosteum": 1.305, "Extrathoracic_Airways": 0.003, "Gallbladder": 0.775, "Heart": 0.068, "Kidneys": 0.395, "Liver": 0.411, "Lungs": 0.058, "Lymph_Nodes": 1.576, "Muscle": 1.885, "Oesophagus": 0.076, "Oral_Mucosa": 0.002, "Ovaries": 5.774, "Pancreas": 0.364, "Prostate": 7.02, "Salivary_Glands": 0.008, "Skin": 1.341, "Small_Intestine": 5.307, "Spleen": 0.276, "Stomach": 0.498, "Testicles": 1.331, "Thymus": 0.012, "Thyroid": 0.013, "Urinary_Bladder": 9.127, "Uterus": 6.165},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.625, "Adrenals": 0.187, "Brain": 0.001, "Breast": 0.0, "Colon": 6.905, "Endosteum": 1.818, "Extrathoracic_Airways": 0.006, "Gallbladder": 1.129, "Heart": 0.102, "Kidneys": 0.588, "Liver": 0.587, "Lungs": 0.091, "Lymph_Nodes": 2.087, "Muscle": 2.336, "Oesophagus": 0.121, "Oral_Mucosa": 0.003, "Ovaries": 7.61, "Pancreas": 0.534, "Prostate": 9.201, "Salivary_Glands": 0.014, "Skin": 1.489, "Small_Intestine": 6.955, "Spleen": 0.42, "Stomach": 0.716, "Testicles": 1.794, "Thymus": 0.019, "Thyroid": 0.023, "Urinary_Bladder": 11.12, "Uterus": 8.254},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.477, "Adrenals": 0.137, "Brain": 0.001, "Breast": 0.0, "Colon": 5.642, "Endosteum": 1.4, "Extrathoracic_Airways": 0.004, "Gallbladder": 0.837, "Heart": 0.074, "Kidneys": 0.429, "Liver": 0.442, "Lungs": 0.064, "Lymph_Nodes": 1.674, "Muscle": 1.973, "Oesophagus": 0.084, "Oral_Mucosa": 0.002, "Ovaries": 6.13, "Pancreas": 0.395, "Prostate": 7.444, "Salivary_Glands": 0.009, "Skin": 1.369, "Small_Intestine": 5.627, "Spleen": 0.301, "Stomach": 0.537, "Testicles": 1.416, "Thymus": 0.013, "Thyroid": 0.015, "Urinary_Bladder": 9.539, "Uterus": 6.563},
                },
            },
            "Lumbar Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.467, 'Adrenals': 2.815, 'Brain': 0.004, 'Breast': 0.28600000000000003, 'Colon': 5.758, 'Extrathoracic_Airways': 0.012, 'Gallbladder': 11.593, 'Heart': 0.795, 'Kidneys': 3.0260000000000002, 'Liver': 8.328999999999999, 'Lungs': 0.524, 'Lymph_Nodes': 4.829, 'Muscle': 1.555, 'Oesophagus': 1.8760000000000001, 'Oral_Mucosa': 0.031, 'Ovaries': 3.021, 'Pancreas': 6.949999999999999, 'Prostate': 0.319, 'Salivary_Glands': 0.024, 'Endosteum': 1.637, 'Skin': 1.004, 'Small_Intestine': 8.447, 'Spleen': 1.935, 'Stomach': 9.328, 'Testicles': 0.148, 'Thymus': 0.066, 'Thyroid': 0.008, 'Urinary_Bladder': 0.726, 'Uterus': 2.005},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.506, 'Adrenals': 3.076, 'Brain': 0.004, 'Breast': 0.32, 'Colon': 6.169, 'Extrathoracic_Airways': 0.014, 'Gallbladder': 12.491, 'Heart': 0.866, 'Kidneys': 3.311, 'Liver': 8.808, 'Lungs': 0.574, 'Lymph_Nodes': 5.195, 'Muscle': 1.65, 'Oesophagus': 2.052, 'Oral_Mucosa': 0.034, 'Ovaries': 3.278, 'Pancreas': 7.484, 'Prostate': 0.352, 'Salivary_Glands': 0.027, 'Endosteum': 1.7630000000000001, 'Skin': 1.0250000000000001, 'Small_Intestine': 9.078, 'Spleen': 2.111, 'Stomach': 9.927, 'Testicles': 0.165, 'Thymus': 0.073, 'Thyroid': 0.01, 'Urinary_Bladder': 0.791, 'Uterus': 2.1870000000000003},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.7010000000000001, 'Adrenals': 4.352, 'Brain': 0.007, 'Breast': 0.51, 'Colon': 8.004000000000001, 'Extrathoracic_Airways': 0.023, 'Gallbladder': 16.598000000000003, 'Heart': 1.211, 'Kidneys': 4.712, 'Liver': 10.828000000000001, 'Lungs': 0.8220000000000001, 'Lymph_Nodes': 6.845000000000001, 'Muscle': 2.081, 'Oesophagus': 2.922, 'Oral_Mucosa': 0.054, 'Ovaries': 4.392, 'Pancreas': 9.882, 'Prostate': 0.503, 'Salivary_Glands': 0.046, 'Endosteum': 2.372, 'Skin': 1.142, 'Small_Intestine': 11.901, 'Spleen': 2.96, 'Stomach': 12.577, 'Testicles': 0.258, 'Thymus': 0.11299999999999999, 'Thyroid': 0.018000000000000002, 'Urinary_Bladder': 1.089, 'Uterus': 3.065},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.54, 'Adrenals': 3.3089999999999997, 'Brain': 0.004, 'Breast': 0.352, 'Colon': 6.525, 'Extrathoracic_Airways': 0.015000000000000001, 'Gallbladder': 13.277000000000001, 'Heart': 0.93, 'Kidneys': 3.5660000000000003, 'Liver': 9.215, 'Lungs': 0.619, 'Lymph_Nodes': 5.513, 'Muscle': 1.7329999999999999, 'Oesophagus': 2.209, 'Oral_Mucosa': 0.037, 'Ovaries': 3.5010000000000003, 'Pancreas': 7.948, 'Prostate': 0.381, 'Salivary_Glands': 0.030000000000000002, 'Endosteum': 1.875, 'Skin': 1.046, 'Small_Intestine': 9.626, 'Spleen': 2.267, 'Stomach': 10.445, 'Testicles': 0.18100000000000002, 'Thymus': 0.08, 'Thyroid': 0.011, 'Urinary_Bladder': 0.847, 'Uterus': 2.348},
                },},
            "Thoracic Spine": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.79, "Adrenals": 2.195, "Brain": 0.037, "Breast": 2.198, "Colon": 0.244, "Endosteum": 2.925, "Extrathoracic_Airways": 0.245, "Gallbladder": 3.514, "Heart": 10.298, "Kidneys": 1.25, "Liver": 4.747, "Lungs": 3.899, "Lymph_Nodes": 2.357, "Muscle": 1.46, "Oesophagus": 4.285, "Oral_Mucosa": 0.26, "Ovaries": 0.021, "Pancreas": 4.262, "Prostate": 0.006, "Salivary_Glands": 0.36, "Skin": 1.157, "Small_Intestine": 0.312, "Spleen": 1.237, "Stomach": 4.614, "Testicles": 0.006, "Thymus": 14.058, "Thyroid": 2.998, "Urinary_Bladder": 0.041, "Uterus": 0.092},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.841, "Adrenals": 2.405, "Brain": 0.042, "Breast": 2.372, "Colon": 0.268, "Endosteum": 3.092, "Extrathoracic_Airways": 0.269, "Gallbladder": 3.789, "Heart": 10.919, "Kidneys": 1.376, "Liver": 5.031, "Lungs": 4.175, "Lymph_Nodes": 2.532, "Muscle": 1.552, "Oesophagus": 4.63, "Oral_Mucosa": 0.285, "Ovaries": 0.024, "Pancreas": 4.612, "Prostate": 0.007, "Salivary_Glands": 0.395, "Skin": 1.188, "Small_Intestine": 0.344, "Spleen": 1.351, "Stomach": 4.917, "Testicles": 0.007, "Thymus": 14.665, "Thyroid": 3.18, "Urinary_Bladder": 0.047, "Uterus": 0.102},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 1.085, "Adrenals": 3.458, "Brain": 0.065, "Breast": 3.197, "Colon": 0.395, "Endosteum": 3.851, "Extrathoracic_Airways": 0.393, "Gallbladder": 5.046, "Heart": 13.504, "Kidneys": 2.016, "Liver": 6.236, "Lungs": 5.421, "Lymph_Nodes": 3.332, "Muscle": 1.97, "Oesophagus": 6.177, "Oral_Mucosa": 0.405, "Ovaries": 0.044, "Pancreas": 6.268, "Prostate": 0.013, "Salivary_Glands": 0.582, "Skin": 1.337, "Small_Intestine": 0.514, "Spleen": 1.911, "Stomach": 6.252, "Testicles": 0.014, "Thymus": 17.043, "Thyroid": 3.915, "Urinary_Bladder": 0.076, "Uterus": 0.162},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.886, "Adrenals": 2.594, "Brain": 0.046, "Breast": 2.525, "Colon": 0.29, "Endosteum": 3.237, "Extrathoracic_Airways": 0.291, "Gallbladder": 4.029, "Heart": 11.446, "Kidneys": 1.489, "Liver": 5.273, "Lungs": 4.416, "Lymph_Nodes": 2.685, "Muscle": 1.633, "Oesophagus": 4.93, "Oral_Mucosa": 0.306, "Ovaries": 0.027, "Pancreas": 4.922, "Prostate": 0.008, "Salivary_Glands": 0.427, "Skin": 1.216, "Small_Intestine": 0.374, "Spleen": 1.453, "Stomach": 5.178, "Testicles": 0.008, "Thymus": 15.166, "Thyroid": 3.334, "Urinary_Bladder": 0.051, "Uterus": 0.112},
                },
            },
                "Scoliosis": {
                    "U": {
                        (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.621, 'Adrenals': 0.966, 'Brain': 0.036, 'Breast': 4.435, 'Colon': 2.081, 'Extrathoracic_Airways': 0.652, 'Gallbladder': 2.861, 'Heart': 3.535, 'Kidneys': 1.005, 'Liver': 3.324, 'Lungs': 2.867, 'Lymph_Nodes': 2.269, 'Muscle': 1.423, 'Oesophagus': 1.913, 'Oral_Mucosa': 0.248, 'Ovaries': 1.871, 'Pancreas': 2.017, 'Prostate': 0.23, 'Salivary_Glands': 0.963, 'Endosteum': 2.221, 'Skin': 1.426, 'Small_Intestine': 2.5, 'Spleen': 1.669, 'Stomach': 3.853, 'Testicles': 0.066, 'Thymus': 4.771, 'Thyroid': 5.702, 'Urinary_Bladder': 0.456, 'Uterus': 1.795},
                        (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.664, 'Adrenals': 1.062, 'Brain': 0.04, 'Breast': 4.542, 'Colon': 2.23, 'Extrathoracic_Airways': 0.69, 'Gallbladder': 3.067, 'Heart': 3.758, 'Kidneys': 1.101, 'Liver': 3.534, 'Lungs': 3.05, 'Lymph_Nodes': 2.427, 'Muscle': 1.506, 'Oesophagus': 2.089, 'Oral_Mucosa': 0.271, 'Ovaries': 2.035, 'Pancreas': 2.193, 'Prostate': 0.253, 'Salivary_Glands': 1.036, 'Endosteum': 2.361, 'Skin': 1.464, 'Small_Intestine': 2.691, 'Spleen': 1.813, 'Stomach': 4.104, 'Testicles': 0.071, 'Thymus': 4.997, 'Thyroid': 5.938, 'Urinary_Bladder': 0.491, 'Uterus': 1.942},
                        (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.866, 'Adrenals': 1.543, 'Brain': 0.061, 'Breast': 4.943, 'Colon': 2.894, 'Extrathoracic_Airways': 0.861, 'Gallbladder': 3.961, 'Heart': 4.727, 'Kidneys': 1.578, 'Liver': 4.435, 'Lungs': 3.835, 'Lymph_Nodes': 3.139, 'Muscle': 1.872, 'Oesophagus': 2.961, 'Oral_Mucosa': 0.388, 'Ovaries': 2.8, 'Pancreas': 3.059, 'Prostate': 0.369, 'Salivary_Glands': 1.376, 'Endosteum': 2.977, 'Skin': 1.636, 'Small_Intestine': 3.555, 'Spleen': 2.489, 'Stomach': 5.216, 'Testicles': 0.098, 'Thymus': 5.896, 'Thyroid': 6.844, 'Urinary_Bladder': 0.647, 'Uterus': 2.636},
                        (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.702, 'Adrenals': 1.148, 'Brain': 0.043, 'Breast': 4.632, 'Colon': 2.359, 'Extrathoracic_Airways': 0.722, 'Gallbladder': 3.245, 'Heart': 3.949, 'Kidneys': 1.187, 'Liver': 3.713, 'Lungs': 3.207, 'Lymph_Nodes': 2.565, 'Muscle': 1.577, 'Oesophagus': 2.246, 'Oral_Mucosa': 0.292, 'Ovaries': 2.18, 'Pancreas': 2.351, 'Prostate': 0.274, 'Salivary_Glands': 1.101, 'Endosteum': 2.481, 'Skin': 1.497, 'Small_Intestine': 2.857, 'Spleen': 1.94, 'Stomach': 4.32, 'Testicles': 0.075, 'Thymus': 5.184, 'Thyroid': 6.133, 'Urinary_Bladder': 0.522, 'Uterus': 2.071},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.914, "Adrenals": 0.067, "Brain": 0.363, "Breast": 0.14, "Colon": 0.011, "Endosteum": 2.793, "Extrathoracic_Airways": 9.116, "Gallbladder": 0.035, "Heart": 0.387, "Kidneys": 0.055, "Liver": 0.072, "Lungs": 0.977, "Lymph_Nodes": 3.072, "Muscle": 1.548, "Oesophagus": 1.207, "Oral_Mucosa": 13.727, "Ovaries": 0.0, "Pancreas": 0.064, "Prostate": 0.0, "Salivary_Glands": 6.74, "Skin": 1.805, "Small_Intestine": 0.019, "Spleen": 0.092, "Stomach": 0.063, "Testicles": 0.0, "Thymus": 8.377, "Thyroid": 22.429, "Urinary_Bladder": 0.004, "Uterus": 0.005},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.973, "Adrenals": 0.075, "Brain": 0.399, "Breast": 0.159, "Colon": 0.013, "Endosteum": 2.971, "Extrathoracic_Airways": 9.654, "Gallbladder": 0.04, "Heart": 0.424, "Kidneys": 0.061, "Liver": 0.08, "Lungs": 1.054, "Lymph_Nodes": 3.231, "Muscle": 1.638, "Oesophagus": 1.309, "Oral_Mucosa": 14.379, "Ovaries": 0.0, "Pancreas": 0.071, "Prostate": 0.0, "Salivary_Glands": 7.144, "Skin": 1.86, "Small_Intestine": 0.021, "Spleen": 0.1, "Stomach": 0.07, "Testicles": 0.0, "Thymus": 8.774, "Thyroid": 23.229, "Urinary_Bladder": 0.004, "Uterus": 0.006},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 1.247, "Adrenals": 0.12, "Brain": 0.583, "Breast": 0.28, "Colon": 0.021, "Endosteum": 3.77, "Extrathoracic_Airways": 11.978, "Gallbladder": 0.066, "Heart": 0.612, "Kidneys": 0.098, "Liver": 0.123, "Lungs": 1.404, "Lymph_Nodes": 3.903, "Muscle": 2.035, "Oesophagus": 1.788, "Oral_Mucosa": 17.103, "Ovaries": 0.0, "Pancreas": 0.111, "Prostate": 0.0, "Salivary_Glands": 8.858, "Skin": 2.097, "Small_Intestine": 0.031, "Spleen": 0.14, "Stomach": 0.113, "Testicles": 0.0, "Thymus": 10.371, "Thyroid": 26.212, "Urinary_Bladder": 0.007, "Uterus": 0.011},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 1.024, "Adrenals": 0.082, "Brain": 0.431, "Breast": 0.177, "Colon": 0.014, "Endosteum": 3.126, "Extrathoracic_Airways": 10.115, "Gallbladder": 0.044, "Heart": 0.457, "Kidneys": 0.067, "Liver": 0.087, "Lungs": 1.121, "Lymph_Nodes": 3.367, "Muscle": 1.716, "Oesophagus": 1.399, "Oral_Mucosa": 14.928, "Ovaries": 0.0, "Pancreas": 0.078, "Prostate": 0.0, "Salivary_Glands": 7.487, "Skin": 1.906, "Small_Intestine": 0.023, "Spleen": 0.107, "Stomach": 0.077, "Testicles": 0.0, "Thymus": 9.106, "Thyroid": 23.876, "Urinary_Bladder": 0.005, "Uterus": 0.007},
                },
            },
            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 1.184, 'Adrenals': 0.030000000000000002, 'Brain': 2.508, 'Breast': 0.017, 'Colon': 0.003, 'Extrathoracic_Airways': 4.495, 'Gallbladder': 0.004, 'Heart': 0.102, 'Kidneys': 0.015000000000000001, 'Liver': 0.023, 'Lungs': 0.254, 'Lymph_Nodes': 1.2830000000000001, 'Muscle': 0.6669999999999999, 'Oesophagus': 0.373, 'Oral_Mucosa': 4.2940000000000005, 'Ovaries': 0.0, 'Pancreas': 0.015000000000000001, 'Prostate': 0.0, 'Salivary_Glands': 4.012, 'Endosteum': 2.527, 'Skin': 1.41, 'Small_Intestine': 0.008, 'Spleen': 0.039, 'Stomach': 0.025, 'Testicles': 0.0, 'Thymus': 1.22, 'Thyroid': 5.695, 'Urinary_Bladder': 0.012, 'Uterus': 0.007},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 1.251, 'Adrenals': 0.032, 'Brain': 2.689, 'Breast': 0.02, 'Colon': 0.003, 'Extrathoracic_Airways': 4.712, 'Gallbladder': 0.004, 'Heart': 0.112, 'Kidneys': 0.017, 'Liver': 0.026, 'Lungs': 0.27399999999999997, 'Lymph_Nodes': 1.345, 'Muscle': 0.7020000000000001, 'Oesophagus': 0.41, 'Oral_Mucosa': 4.517, 'Ovaries': 0.0, 'Pancreas': 0.017, 'Prostate': 0.0, 'Salivary_Glands': 4.229, 'Endosteum': 2.66, 'Skin': 1.441, 'Small_Intestine': 0.009000000000000001, 'Spleen': 0.043000000000000003, 'Stomach': 0.028, 'Testicles': 0.0, 'Thymus': 1.2910000000000001, 'Thyroid': 5.8919999999999995, 'Urinary_Bladder': 0.013, 'Uterus': 0.008},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 1.554, 'Adrenals': 0.046, 'Brain': 3.5140000000000002, 'Breast': 0.039, 'Colon': 0.005, 'Extrathoracic_Airways': 5.603000000000001, 'Gallbladder': 0.004, 'Heart': 0.163, 'Kidneys': 0.026, 'Liver': 0.038, 'Lungs': 0.365, 'Lymph_Nodes': 1.609, 'Muscle': 0.852, 'Oesophagus': 0.594, 'Oral_Mucosa': 5.444, 'Ovaries': 0.0, 'Pancreas': 0.028, 'Prostate': 0.0, 'Salivary_Glands': 5.132000000000001, 'Endosteum': 3.22, 'Skin': 1.585, 'Small_Intestine': 0.014, 'Spleen': 0.061, 'Stomach': 0.045000000000000005, 'Testicles': 0.0, 'Thymus': 1.588, 'Thyroid': 6.781, 'Urinary_Bladder': 0.018000000000000002, 'Uterus': 0.014},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 1.309, 'Adrenals': 0.034999999999999996, 'Brain': 2.847, 'Breast': 0.023, 'Colon': 0.003, 'Extrathoracic_Airways': 4.894, 'Gallbladder': 0.004, 'Heart': 0.121, 'Kidneys': 0.018000000000000002, 'Liver': 0.028, 'Lungs': 0.29100000000000004, 'Lymph_Nodes': 1.398, 'Muscle': 0.731, 'Oesophagus': 0.443, 'Oral_Mucosa': 4.706, 'Ovaries': 0.0, 'Pancreas': 0.018000000000000002, 'Prostate': 0.0, 'Salivary_Glands': 4.412, 'Endosteum': 2.7729999999999997, 'Skin': 1.469, 'Small_Intestine': 0.01, 'Spleen': 0.046, 'Stomach': 0.030000000000000002, 'Testicles': 0.0, 'Thymus': 1.35, 'Thyroid': 6.0600000000000005, 'Urinary_Bladder': 0.014, 'Uterus': 0.009000000000000001},
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.149, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.183, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.078, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.159, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08399999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.166, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.23900000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.262, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.094, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.177, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.28200000000000003, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.098, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.316, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.106, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.189, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        1: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
                'F': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.2206, 'Adrenals': 0.336022, 'Brain': 0.00102336, 'Breast': 1.06584, 'Colon': 1.7196, 'Endosteum': 0.296001, 'Extrathoracic_Airways': 0.0052539, 'Gallbladder': 1.08739, 'Gonads': 1.34518, 'Heart': 0.553078, 'Kidneys': 0.5488, 'Lens': 0.00370935, 'Liver': 1.19471, 'Lungs': 0.413991, 'Lymph_Nodes': 1.02861, 'Muscle': 0.435561, 'Oesophagus': 0.138817, 'Oral_Mucosa': 0.0177554, 'Pancreas': 0.995259, 'Salivary_Glands': 0.0118535, 'Skin': 0.563889, 'Small_Intestine': 1.49288, 'Spleen': 0.759272, 'Stomach': 1.13876, 'Thymus': 0.0734256, 'Thyroid': 0.0365784, 'Urinary_Bladder': 1.49899, 'Uterus': 0.96455},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.244317, 'Adrenals': 0.374597, 'Brain': 0.00116572, 'Breast': 1.09911, 'Colon': 1.85295, 'Endosteum': 0.328059, 'Extrathoracic_Airways': 0.00588458, 'Gallbladder': 1.19064, 'Gonads': 1.46029, 'Heart': 0.60204, 'Kidneys': 0.608841, 'Lens': 0.00412533, 'Liver': 1.29941, 'Lungs': 0.45068, 'Lymph_Nodes': 1.10146, 'Muscle': 0.46197, 'Oesophagus': 0.154102, 'Oral_Mucosa': 0.0195537, 'Pancreas': 1.09115, 'Salivary_Glands': 0.0133192, 'Skin': 0.580007, 'Small_Intestine': 1.6199, 'Spleen': 0.831096, 'Stomach': 1.24248, 'Thymus': 0.0821063, 'Thyroid': 0.0411245, 'Urinary_Bladder': 1.62227, 'Uterus': 1.05562},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.366193, 'Adrenals': 0.576516, 'Brain': 0.00203179, 'Breast': 1.24336, 'Colon': 2.43901, 'Endosteum': 0.493374, 'Extrathoracic_Airways': 0.00943503, 'Gallbladder': 1.68094, 'Gonads': 1.9821, 'Heart': 0.830073, 'Kidneys': 0.914046, 'Lens': 0.00634378, 'Liver': 1.7815, 'Lungs': 0.621989, 'Lymph_Nodes': 1.42382, 'Muscle': 0.575156, 'Oesophagus': 0.232621, 'Oral_Mucosa': 0.0287441, 'Pancreas': 1.54812, 'Salivary_Glands': 0.0214819, 'Skin': 0.651537, 'Small_Intestine': 2.19685, 'Spleen': 1.17123, 'Stomach': 1.72607, 'Thymus': 0.128703, 'Thyroid': 0.0663931, 'Urinary_Bladder': 2.17301, 'Uterus': 1.48557},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.265883, 'Adrenals': 0.409931, 'Brain': 0.00130175, 'Breast': 1.12747, 'Colon': 1.96824, 'Endosteum': 0.357243, 'Extrathoracic_Airways': 0.00647332, 'Gallbladder': 1.28231, 'Gonads': 1.5609, 'Heart': 0.645186, 'Kidneys': 0.663346, 'Lens': 0.00450759, 'Liver': 1.39139, 'Lungs': 0.483033, 'Lymph_Nodes': 1.16448, 'Muscle': 0.48453, 'Oesophagus': 0.168014, 'Oral_Mucosa': 0.0211834, 'Pancreas': 1.1764, 'Salivary_Glands': 0.0146853, 'Skin': 0.593926, 'Small_Intestine': 1.73098, 'Spleen': 0.894811, 'Stomach': 1.33401, 'Thymus': 0.0901139, 'Thyroid': 0.0453609, 'Urinary_Bladder': 1.72949, 'Uterus': 1.13632},
                },
                'M': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.220837, 'Adrenals': 0.33754, 'Brain': 0.00102471, 'Breast': 1.0544, 'Colon': 1.71968, 'Endosteum': 0.296306, 'Extrathoracic_Airways': 0.00525405, 'Gallbladder': 1.08738, 'Gonads': 0.735247, 'Heart': 0.553148, 'Kidneys': 0.549049, 'Lens': 0.00363492, 'Liver': 1.19745, 'Lungs': 0.414074, 'Lymph_Nodes': 1.0409, 'Muscle': 0.438768, 'Oesophagus': 0.139401, 'Oral_Mucosa': 0.0176251, 'Pancreas': 0.994934, 'Salivary_Glands': 0.0118419, 'Skin': 0.566711, 'Small_Intestine': 1.49228, 'Spleen': 0.76224, 'Stomach': 1.14054, 'Thymus': 0.0734932, 'Thyroid': 0.0365951, 'Urinary_Bladder': 1.4538, 'Prostate': 1.30305},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.244573, 'Adrenals': 0.37625, 'Brain': 0.00116725, 'Breast': 1.08748, 'Colon': 1.85307, 'Endosteum': 0.328389, 'Extrathoracic_Airways': 0.00588247, 'Gallbladder': 1.19059, 'Gonads': 0.775705, 'Heart': 0.602111, 'Kidneys': 0.609114, 'Lens': 0.00403809, 'Liver': 1.30229, 'Lungs': 0.450761, 'Lymph_Nodes': 1.11399, 'Muscle': 0.465296, 'Oesophagus': 0.154746, 'Oral_Mucosa': 0.0194121, 'Pancreas': 1.09081, 'Salivary_Glands': 0.0133081, 'Skin': 0.582988, 'Small_Intestine': 1.61927, 'Spleen': 0.834271, 'Stomach': 1.24438, 'Thymus': 0.0821792, 'Thyroid': 0.0411416, 'Urinary_Bladder': 1.57439, 'Prostate': 1.40575},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.366533, 'Adrenals': 0.578749, 'Brain': 0.00203439, 'Breast': 1.23133, 'Colon': 2.43932, 'Endosteum': 0.493816, 'Extrathoracic_Airways': 0.00941054, 'Gallbladder': 1.68039, 'Gonads': 0.946623, 'Heart': 0.830153, 'Kidneys': 0.914428, 'Lens': 0.00616662, 'Liver': 1.78488, 'Lungs': 0.622052, 'Lymph_Nodes': 1.43671, 'Muscle': 0.578937, 'Oesophagus': 0.23354, 'Oral_Mucosa': 0.0285485, 'Pancreas': 1.54781, 'Salivary_Glands': 0.0214792, 'Skin': 0.655145, 'Small_Intestine': 2.1962, 'Spleen': 1.17527, 'Stomach': 1.72841, 'Thymus': 0.128791, 'Thyroid': 0.066419, 'Urinary_Bladder': 2.11507, 'Prostate': 1.85718},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.266155, 'Adrenals': 0.411701, 'Brain': 0.00130346, 'Breast': 1.1157, 'Colon': 1.96838, 'Endosteum': 0.357594, 'Extrathoracic_Airways': 0.00646863, 'Gallbladder': 1.2822, 'Gonads': 0.810127, 'Heart': 0.645258, 'Kidneys': 0.663641, 'Lens': 0.0044074, 'Liver': 1.39438, 'Lungs': 0.483112, 'Lymph_Nodes': 1.17716, 'Muscle': 0.487954, 'Oesophagus': 0.16871, 'Oral_Mucosa': 0.021032, 'Pancreas': 1.17606, 'Salivary_Glands': 0.0146748, 'Skin': 0.597039, 'Small_Intestine': 1.73034, 'Spleen': 0.898162, 'Stomach': 1.336, 'Thymus': 0.0901911, 'Thyroid': 0.0453784, 'Urinary_Bladder': 1.6794, 'Prostate': 1.49455},
                },
                'U': {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.220719, 'Adrenals': 0.336781, 'Brain': 0.00102403, 'Breast': 1.06012, 'Colon': 1.71964, 'Endosteum': 0.296154, 'Extrathoracic_Airways': 0.00525397, 'Gallbladder': 1.08738, 'Gonads': 1.04021, 'Heart': 0.553113, 'Kidneys': 0.548924, 'Lens': 0.00367214, 'Liver': 1.19608, 'Lungs': 0.414033, 'Lymph_Nodes': 1.03476, 'Muscle': 0.437164, 'Oesophagus': 0.139109, 'Oral_Mucosa': 0.0176903, 'Pancreas': 0.995096, 'Salivary_Glands': 0.0118477, 'Skin': 0.5653, 'Small_Intestine': 1.49258, 'Spleen': 0.760756, 'Stomach': 1.13965, 'Thymus': 0.0734594, 'Thyroid': 0.0365867, 'Urinary_Bladder': 1.47639, 'Uterus': 0.482275, 'Prostate': 0.651524},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.244445, 'Adrenals': 0.375424, 'Brain': 0.00116649, 'Breast': 1.09329, 'Colon': 1.85301, 'Endosteum': 0.328224, 'Extrathoracic_Airways': 0.00588352, 'Gallbladder': 1.19062, 'Gonads': 1.118, 'Heart': 0.602076, 'Kidneys': 0.608978, 'Lens': 0.00408171, 'Liver': 1.30085, 'Lungs': 0.450721, 'Lymph_Nodes': 1.10772, 'Muscle': 0.463633, 'Oesophagus': 0.154424, 'Oral_Mucosa': 0.0194829, 'Pancreas': 1.09098, 'Salivary_Glands': 0.0133137, 'Skin': 0.581498, 'Small_Intestine': 1.61959, 'Spleen': 0.832683, 'Stomach': 1.24343, 'Thymus': 0.0821428, 'Thyroid': 0.041133, 'Urinary_Bladder': 1.59833, 'Uterus': 0.52781, 'Prostate': 0.702876},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.366363, 'Adrenals': 0.577632, 'Brain': 0.00203309, 'Breast': 1.23735, 'Colon': 2.43917, 'Endosteum': 0.493595, 'Extrathoracic_Airways': 0.00942279, 'Gallbladder': 1.68066, 'Gonads': 1.46436, 'Heart': 0.830113, 'Kidneys': 0.914237, 'Lens': 0.0062552, 'Liver': 1.78319, 'Lungs': 0.622021, 'Lymph_Nodes': 1.43026, 'Muscle': 0.577046, 'Oesophagus': 0.233081, 'Oral_Mucosa': 0.0286463, 'Pancreas': 1.54796, 'Salivary_Glands': 0.0214806, 'Skin': 0.653341, 'Small_Intestine': 2.19653, 'Spleen': 1.17325, 'Stomach': 1.72724, 'Thymus': 0.128747, 'Thyroid': 0.066406, 'Urinary_Bladder': 2.14404, 'Uterus': 0.742787, 'Prostate': 0.92859},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.266019, 'Adrenals': 0.410816, 'Brain': 0.0013026, 'Breast': 1.12158, 'Colon': 1.96831, 'Endosteum': 0.357419, 'Extrathoracic_Airways': 0.00647098, 'Gallbladder': 1.28225, 'Gonads': 1.18551, 'Heart': 0.645222, 'Kidneys': 0.663494, 'Lens': 0.00445749, 'Liver': 1.39288, 'Lungs': 0.483072, 'Lymph_Nodes': 1.17082, 'Muscle': 0.486242, 'Oesophagus': 0.168362, 'Oral_Mucosa': 0.0211077, 'Pancreas': 1.17623, 'Salivary_Glands': 0.0146801, 'Skin': 0.595483, 'Small_Intestine': 1.73066, 'Spleen': 0.896487, 'Stomach': 1.335, 'Thymus': 0.0901525, 'Thyroid': 0.0453696, 'Urinary_Bladder': 1.70444, 'Uterus': 0.568159, 'Prostate': 0.747276},
                },
            },
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.333, "Adrenals": 0.432, "Brain": 0.009, "Breast": 4.236, "Colon": 0.09, "Endosteum": 1.571, "Extrathoracic_Airways": 0.09, "Gallbladder": 0.662, "Heart": 2.193, "Kidneys": 0.261, "Liver": 1.328, "Lungs": 1.704, "Lymph_Nodes": 0.655, "Muscle": 0.503, "Oesophagus": 0.795, "Oral_Mucosa": 0.066, "Ovaries": 0.046, "Pancreas": 0.959, "Prostate": 0.0, "Salivary_Glands": 0.129, "Skin": 0.627, "Small_Intestine": 0.083, "Spleen": 0.697, "Stomach": 1.439, "Testicles": 0.001, "Thymus": 3.117, "Thyroid": 1.781, "Urinary_Bladder": 0.014, "Uterus": 0.01},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.357, "Adrenals": 0.478, "Brain": 0.01, "Breast": 4.336, "Colon": 0.1, "Endosteum": 1.675, "Extrathoracic_Airways": 0.099, "Gallbladder": 0.717, "Heart": 2.35, "Kidneys": 0.288, "Liver": 1.427, "Lungs": 1.828, "Lymph_Nodes": 0.706, "Muscle": 0.536, "Oesophagus": 0.874, "Oral_Mucosa": 0.073, "Ovaries": 0.052, "Pancreas": 1.049, "Prostate": 0.0, "Salivary_Glands": 0.141, "Skin": 0.642, "Small_Intestine": 0.092, "Spleen": 0.765, "Stomach": 1.536, "Testicles": 0.002, "Thymus": 3.293, "Thyroid": 1.86, "Urinary_Bladder": 0.016, "Uterus": 0.011},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.472, "Adrenals": 0.735, "Brain": 0.016, "Breast": 4.691, "Colon": 0.154, "Endosteum": 2.152, "Extrathoracic_Airways": 0.145, "Gallbladder": 0.965, "Heart": 3.059, "Kidneys": 0.432, "Liver": 1.884, "Lungs": 2.39, "Lymph_Nodes": 0.944, "Muscle": 0.684, "Oesophagus": 1.273, "Oral_Mucosa": 0.112, "Ovaries": 0.088, "Pancreas": 1.5, "Prostate": 0.0, "Salivary_Glands": 0.197, "Skin": 0.715, "Small_Intestine": 0.144, "Spleen": 1.101, "Stomach": 1.965, "Testicles": 0.003, "Thymus": 4.035, "Thyroid": 2.204, "Urinary_Bladder": 0.026, "Uterus": 0.02},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.378, "Adrenals": 0.521, "Brain": 0.011, "Breast": 4.415, "Colon": 0.109, "Endosteum": 1.766, "Extrathoracic_Airways": 0.107, "Gallbladder": 0.764, "Heart": 2.487, "Kidneys": 0.313, "Liver": 1.514, "Lungs": 1.936, "Lymph_Nodes": 0.75, "Muscle": 0.564, "Oesophagus": 0.945, "Oral_Mucosa": 0.08, "Ovaries": 0.058, "Pancreas": 1.13, "Prostate": 0.0, "Salivary_Glands": 0.151, "Skin": 0.656, "Small_Intestine": 0.101, "Spleen": 0.825, "Stomach": 1.62, "Testicles": 0.002, "Thymus": 3.442, "Thyroid": 1.927, "Urinary_Bladder": 0.018, "Uterus": 0.012},
                },
            },
            "Pelvis": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.214, "Adrenals": 0.047, "Brain": 0.0, "Breast": 0.0, "Colon": 2.136, "Endosteum": 0.576, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.323, "Heart": 0.022, "Kidneys": 0.114, "Liver": 0.139, "Lungs": 0.017, "Lymph_Nodes": 0.682, "Muscle": 0.823, "Oesophagus": 0.017, "Oral_Mucosa": 0.001, "Ovaries": 2.319, "Pancreas": 0.112, "Prostate": 2.831, "Salivary_Glands": 0.001, "Skin": 0.513, "Small_Intestine": 2.158, "Spleen": 0.073, "Stomach": 0.17, "Testicles": 5.15, "Thymus": 0.002, "Thyroid": 0.0, "Urinary_Bladder": 4.22, "Uterus": 2.259},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.236, "Adrenals": 0.053, "Brain": 0.0, "Breast": 0.0, "Colon": 2.312, "Endosteum": 0.632, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.357, "Heart": 0.024, "Kidneys": 0.126, "Liver": 0.153, "Lungs": 0.019, "Lymph_Nodes": 0.737, "Muscle": 0.874, "Oesophagus": 0.019, "Oral_Mucosa": 0.001, "Ovaries": 2.527, "Pancreas": 0.125, "Prostate": 3.06, "Salivary_Glands": 0.001, "Skin": 0.525, "Small_Intestine": 2.338, "Spleen": 0.081, "Stomach": 0.188, "Testicles": 5.33, "Thymus": 0.002, "Thyroid": 0.0, "Urinary_Bladder": 4.51, "Uterus": 2.466},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.345, "Adrenals": 0.085, "Brain": 0.0, "Breast": 0.0, "Colon": 3.146, "Endosteum": 0.912, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.539, "Heart": 0.038, "Kidneys": 0.196, "Liver": 0.227, "Lungs": 0.03, "Lymph_Nodes": 1, "Muscle": 1.109, "Oesophagus": 0.034, "Oral_Mucosa": 0.001, "Ovaries": 3.513, "Pancreas": 0.198, "Prostate": 4.126, "Salivary_Glands": 0.002, "Skin": 0.586, "Small_Intestine": 3.185, "Spleen": 0.123, "Stomach": 0.285, "Testicles": 5.965, "Thymus": 0.003, "Thyroid": 0.0, "Urinary_Bladder": 5.845, "Uterus": 3.48},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.255, "Adrenals": 0.058, "Brain": 0.0, "Breast": 0.0, "Colon": 2.468, "Endosteum": 0.682, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.388, "Heart": 0.027, "Kidneys": 0.138, "Liver": 0.166, "Lungs": 0.021, "Lymph_Nodes": 0.786, "Muscle": 0.919, "Oesophagus": 0.022, "Oral_Mucosa": 0.001, "Ovaries": 2.711, "Pancreas": 0.137, "Prostate": 3.26, "Salivary_Glands": 0.001, "Skin": 0.536, "Small_Intestine": 2.496, "Spleen": 0.088, "Stomach": 0.205, "Testicles": 5.474, "Thymus": 0.002, "Thyroid": 0.0, "Urinary_Bladder": 4.763, "Uterus": 2.652},
                },
            },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.213, 'Adrenals': 1.056, 'Brain': 0.001, 'Breast': 0.055, 'Colon': 2.6679999999999997, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 5.611, 'Heart': 0.414, 'Kidneys': 1.137, 'Liver': 3.686, 'Lungs': 0.25, 'Lymph_Nodes': 2.088, 'Muscle': 0.696, 'Oesophagus': 0.853, 'Oral_Mucosa': 0.003, 'Ovaries': 0.7649999999999999, 'Pancreas': 2.6870000000000003, 'Prostate': 0.177, 'Salivary_Glands': 0.004, 'Endosteum': 0.666, 'Skin': 0.483, 'Small_Intestine': 3.474, 'Spleen': 1.016, 'Stomach': 4.917999999999999, 'Testicles': 0.107, 'Thymus': 0.05, 'Thyroid': 0.005, 'Urinary_Bladder': 0.294, 'Uterus': 0.889},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.231, 'Adrenals': 1.163, 'Brain': 0.001, 'Breast': 0.062, 'Colon': 2.87, 'Extrathoracic_Airways': 0.002, 'Gallbladder': 6.041, 'Heart': 0.453, 'Kidneys': 1.2489999999999999, 'Liver': 3.927, 'Lungs': 0.27399999999999997, 'Lymph_Nodes': 2.255, 'Muscle': 0.74, 'Oesophagus': 0.9309999999999999, 'Oral_Mucosa': 0.003, 'Ovaries': 0.8450000000000001, 'Pancreas': 2.926, 'Prostate': 0.19699999999999998, 'Salivary_Glands': 0.004, 'Endosteum': 0.717, 'Skin': 0.494, 'Small_Intestine': 3.7529999999999997, 'Spleen': 1.113, 'Stomach': 5.234, 'Testicles': 0.116, 'Thymus': 0.056, 'Thyroid': 0.005, 'Urinary_Bladder': 0.324, 'Uterus': 0.9700000000000001},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.32899999999999996, 'Adrenals': 1.721, 'Brain': 0.002, 'Breast': 0.097, 'Colon': 3.798, 'Extrathoracic_Airways': 0.003, 'Gallbladder': 8.013, 'Heart': 0.646, 'Kidneys': 1.828, 'Liver': 5.002000000000001, 'Lungs': 0.40099999999999997, 'Lymph_Nodes': 3.044, 'Muscle': 0.9460000000000001, 'Oesophagus': 1.323, 'Oral_Mucosa': 0.005, 'Ovaries': 1.27, 'Pancreas': 4.104, 'Prostate': 0.318, 'Salivary_Glands': 0.006, 'Endosteum': 0.972, 'Skin': 0.55, 'Small_Intestine': 5.068, 'Spleen': 1.601, 'Stomach': 6.587, 'Testicles': 0.155, 'Thymus': 0.09000000000000001, 'Thyroid': 0.009000000000000001, 'Urinary_Bladder': 0.483, 'Uterus': 1.383},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.248, 'Adrenals': 1.259, 'Brain': 0.001, 'Breast': 0.068, 'Colon': 3.0469999999999997, 'Extrathoracic_Airways': 0.002, 'Gallbladder': 6.418, 'Heart': 0.48700000000000004, 'Kidneys': 1.351, 'Liver': 4.135, 'Lungs': 0.296, 'Lymph_Nodes': 2.403, 'Muscle': 0.778, 'Oesophagus': 1.001, 'Oral_Mucosa': 0.003, 'Ovaries': 0.919, 'Pancreas': 3.14, 'Prostate': 0.216, 'Salivary_Glands': 0.004, 'Endosteum': 0.762, 'Skin': 0.504, 'Small_Intestine': 3.9979999999999998, 'Spleen': 1.2, 'Stomach': 5.505000000000001, 'Testicles': 0.12300000000000001, 'Thymus': 0.061, 'Thyroid': 0.006, 'Urinary_Bladder': 0.35100000000000003, 'Uterus': 1.044},
                },},
            "Thoracic Spine": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.313, "Adrenals": 0.684, "Brain": 0.014, "Breast": 0.577, "Colon": 0.108, "Endosteum": 1.287, "Extrathoracic_Airways": 0.159, "Gallbladder": 1.235, "Heart": 4.516, "Kidneys": 0.412, "Liver": 1.961, "Lungs": 1.631, "Lymph_Nodes": 1.102, "Muscle": 0.604, "Oesophagus": 1.551, "Oral_Mucosa": 0.134, "Ovaries": 0.095, "Pancreas": 1.773, "Prostate": 0.005, "Salivary_Glands": 0.176, "Skin": 0.541, "Small_Intestine": 0.11, "Spleen": 0.47, "Stomach": 1.84, "Testicles": 0.0, "Thymus": 6.497, "Thyroid": 4.066, "Urinary_Bladder": 0.014, "Uterus": 0.026},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.334, "Adrenals": 0.755, "Brain": 0.016, "Breast": 0.63, "Colon": 0.12, "Endosteum": 1.365, "Extrathoracic_Airways": 0.174, "Gallbladder": 1.344, "Heart": 4.827, "Kidneys": 0.456, "Liver": 2.1, "Lungs": 1.764, "Lymph_Nodes": 1.186, "Muscle": 0.645, "Oesophagus": 1.693, "Oral_Mucosa": 0.147, "Ovaries": 0.105, "Pancreas": 1.937, "Prostate": 0.006, "Salivary_Glands": 0.192, "Skin": 0.555, "Small_Intestine": 0.122, "Spleen": 0.518, "Stomach": 1.975, "Testicles": 0.0, "Thymus": 6.854, "Thyroid": 4.232, "Urinary_Bladder": 0.016, "Uterus": 0.03},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.438, "Adrenals": 1.123, "Brain": 0.025, "Breast": 0.883, "Colon": 0.181, "Endosteum": 1.732, "Extrathoracic_Airways": 0.246, "Gallbladder": 1.873, "Heart": 6.218, "Kidneys": 0.689, "Liver": 2.735, "Lungs": 2.391, "Lymph_Nodes": 1.576, "Muscle": 0.84, "Oesophagus": 2.376, "Oral_Mucosa": 0.21, "Ovaries": 0.153, "Pancreas": 2.73, "Prostate": 0.012, "Salivary_Glands": 0.268, "Skin": 0.627, "Small_Intestine": 0.186, "Spleen": 0.765, "Stomach": 2.588, "Testicles": 0.0, "Thymus": 8.323, "Thyroid": 4.849, "Urinary_Bladder": 0.026, "Uterus": 0.055},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.353, "Adrenals": 0.82, "Brain": 0.017, "Breast": 0.677, "Colon": 0.13, "Endosteum": 1.434, "Extrathoracic_Airways": 0.187, "Gallbladder": 1.441, "Heart": 5.097, "Kidneys": 0.496, "Liver": 2.222, "Lungs": 1.881, "Lymph_Nodes": 1.259, "Muscle": 0.682, "Oesophagus": 1.819, "Oral_Mucosa": 0.159, "Ovaries": 0.113, "Pancreas": 2.082, "Prostate": 0.007, "Salivary_Glands": 0.205, "Skin": 0.568, "Small_Intestine": 0.133, "Spleen": 0.561, "Stomach": 2.093, "Testicles": 0.0, "Thymus": 7.156, "Thyroid": 4.368, "Urinary_Bladder": 0.017, "Uterus": 0.033},
                },
            },
                "Scoliosis": {
                    "U": {
                        (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.257, 'Adrenals': 0.355, 'Brain': 0.015, 'Breast': 2.755, 'Colon': 0.906, 'Extrathoracic_Airways': 0.294, 'Gallbladder': 1.272, 'Heart': 1.487, 'Kidneys': 0.34, 'Liver': 1.348, 'Lungs': 1.242, 'Lymph_Nodes': 0.957, 'Muscle': 0.596, 'Oesophagus': 0.678, 'Oral_Mucosa': 0.187, 'Ovaries': 0.773, 'Pancreas': 0.819, 'Prostate': 0.068, 'Salivary_Glands': 0.505, 'Endosteum': 0.929, 'Skin': 0.659, 'Small_Intestine': 1.014, 'Spleen': 0.635, 'Stomach': 1.675, 'Testicles': 0.037, 'Thymus': 1.988, 'Thyroid': 2.377, 'Urinary_Bladder': 0.223, 'Uterus': 0.713},
                        (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.276, 'Adrenals': 0.392, 'Brain': 0.017, 'Breast': 2.79, 'Colon': 0.978, 'Extrathoracic_Airways': 0.312, 'Gallbladder': 1.378, 'Heart': 1.584, 'Kidneys': 0.374, 'Liver': 1.442, 'Lungs': 1.33, 'Lymph_Nodes': 1.026, 'Muscle': 0.633, 'Oesophagus': 0.738, 'Oral_Mucosa': 0.204, 'Ovaries': 0.85, 'Pancreas': 0.896, 'Prostate': 0.074, 'Salivary_Glands': 0.541, 'Endosteum': 0.989, 'Skin': 0.675, 'Small_Intestine': 1.095, 'Spleen': 0.693, 'Stomach': 1.787, 'Testicles': 0.041, 'Thymus': 2.096, 'Thyroid': 2.467, 'Urinary_Bladder': 0.243, 'Uterus': 0.775},
                        (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.369, 'Adrenals': 0.582, 'Brain': 0.026, 'Breast': 2.909, 'Colon': 1.317, 'Extrathoracic_Airways': 0.387, 'Gallbladder': 1.892, 'Heart': 2.014, 'Kidneys': 0.559, 'Liver': 1.863, 'Lungs': 1.722, 'Lymph_Nodes': 1.354, 'Muscle': 0.799, 'Oesophagus': 1.025, 'Oral_Mucosa': 0.282, 'Ovaries': 1.25, 'Pancreas': 1.286, 'Prostate': 0.108, 'Salivary_Glands': 0.705, 'Endosteum': 1.265, 'Skin': 0.753, 'Small_Intestine': 1.481, 'Spleen': 0.976, 'Stomach': 2.283, 'Testicles': 0.061, 'Thymus': 2.556, 'Thyroid': 2.84, 'Urinary_Bladder': 0.338, 'Uterus': 1.099},
                        (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.293, 'Adrenals': 0.425, 'Brain': 0.019, 'Breast': 2.818, 'Colon': 1.041, 'Extrathoracic_Airways': 0.328, 'Gallbladder': 1.472, 'Heart': 1.667, 'Kidneys': 0.406, 'Liver': 1.522, 'Lungs': 1.405, 'Lymph_Nodes': 1.088, 'Muscle': 0.664, 'Oesophagus': 0.792, 'Oral_Mucosa': 0.218, 'Ovaries': 0.921, 'Pancreas': 0.964, 'Prostate': 0.08, 'Salivary_Glands': 0.573, 'Endosteum': 1.041, 'Skin': 0.69, 'Small_Intestine': 1.167, 'Spleen': 0.744, 'Stomach': 1.884, 'Testicles': 0.045, 'Thymus': 2.189, 'Thyroid': 2.543, 'Urinary_Bladder': 0.261, 'Uterus': 0.832},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.461, "Adrenals": 0.048, "Brain": 0.311, "Breast": 0.085, "Colon": 0.003, "Endosteum": 1.526, "Extrathoracic_Airways": 6.204, "Gallbladder": 0.015, "Heart": 0.144, "Kidneys": 0.012, "Liver": 0.024, "Lungs": 0.391, "Lymph_Nodes": 1.616, "Muscle": 0.597, "Oesophagus": 0.429, "Oral_Mucosa": 5.35, "Ovaries": 0.0, "Pancreas": 0.022, "Prostate": 0.0, "Salivary_Glands": 2.335, "Skin": 0.755, "Small_Intestine": 0.003, "Spleen": 0.026, "Stomach": 0.017, "Testicles": 0.0, "Thymus": 2.898, "Thyroid": 10.684, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.493, "Adrenals": 0.054, "Brain": 0.343, "Breast": 0.094, "Colon": 0.004, "Endosteum": 1.629, "Extrathoracic_Airways": 6.577, "Gallbladder": 0.016, "Heart": 0.159, "Kidneys": 0.014, "Liver": 0.027, "Lungs": 0.427, "Lymph_Nodes": 1.705, "Muscle": 0.636, "Oesophagus": 0.469, "Oral_Mucosa": 5.655, "Ovaries": 0.0, "Pancreas": 0.025, "Prostate": 0.0, "Salivary_Glands": 2.517, "Skin": 0.777, "Small_Intestine": 0.003, "Spleen": 0.029, "Stomach": 0.019, "Testicles": 0.0, "Thymus": 3.07, "Thyroid": 11.053, "Urinary_Bladder": 0.001, "Uterus": 0.0},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.646, "Adrenals": 0.088, "Brain": 0.514, "Breast": 0.139, "Colon": 0.006, "Endosteum": 2.104, "Extrathoracic_Airways": 8.196, "Gallbladder": 0.027, "Heart": 0.236, "Kidneys": 0.023, "Liver": 0.041, "Lungs": 0.6, "Lymph_Nodes": 2.091, "Muscle": 0.811, "Oesophagus": 0.666, "Oral_Mucosa": 6.996, "Ovaries": 0.0, "Pancreas": 0.04, "Prostate": 0.0, "Salivary_Glands": 3.361, "Skin": 0.877, "Small_Intestine": 0.006, "Spleen": 0.045, "Stomach": 0.031, "Testicles": 0.0, "Thymus": 3.798, "Thyroid": 12.464, "Urinary_Bladder": 0.001, "Uterus": 0.0},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.521, "Adrenals": 0.06, "Brain": 0.372, "Breast": 0.102, "Colon": 0.004, "Endosteum": 1.719, "Extrathoracic_Airways": 6.897, "Gallbladder": 0.018, "Heart": 0.172, "Kidneys": 0.015, "Liver": 0.029, "Lungs": 0.458, "Lymph_Nodes": 1.781, "Muscle": 0.669, "Oesophagus": 0.504, "Oral_Mucosa": 5.916, "Ovaries": 0.0, "Pancreas": 0.027, "Prostate": 0.0, "Salivary_Glands": 2.676, "Skin": 0.796, "Small_Intestine": 0.004, "Spleen": 0.032, "Stomach": 0.021, "Testicles": 0.0, "Thymus": 3.216, "Thyroid": 11.353, "Urinary_Bladder": 0.001, "Uterus": 0.0},
                },
            },
            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.549, 'Adrenals': 0.004, 'Brain': 1.1520000000000001, 'Breast': 0.015000000000000001, 'Colon': 0.001, 'Extrathoracic_Airways': 2.6679999999999997, 'Gallbladder': 0.0, 'Heart': 0.034999999999999996, 'Kidneys': 0.005, 'Liver': 0.009000000000000001, 'Lungs': 0.097, 'Lymph_Nodes': 0.777, 'Muscle': 0.301, 'Oesophagus': 0.148, 'Oral_Mucosa': 2.5799999999999996, 'Ovaries': 0.0, 'Pancreas': 0.013, 'Prostate': 0.0, 'Salivary_Glands': 2.088, 'Endosteum': 1.496, 'Skin': 0.734, 'Small_Intestine': 0.001, 'Spleen': 0.007, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.32, 'Thyroid': 4.432, 'Urinary_Bladder': 0.0, 'Uterus': 0.011},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.585, 'Adrenals': 0.005, 'Brain': 1.247, 'Breast': 0.016, 'Colon': 0.001, 'Extrathoracic_Airways': 2.823, 'Gallbladder': 0.0, 'Heart': 0.039, 'Kidneys': 0.005, 'Liver': 0.009000000000000001, 'Lungs': 0.106, 'Lymph_Nodes': 0.819, 'Muscle': 0.319, 'Oesophagus': 0.163, 'Oral_Mucosa': 2.726, 'Ovaries': 0.0, 'Pancreas': 0.014, 'Prostate': 0.0, 'Salivary_Glands': 2.227, 'Endosteum': 1.585, 'Skin': 0.751, 'Small_Intestine': 0.001, 'Spleen': 0.008, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.345, 'Thyroid': 4.58, 'Urinary_Bladder': 0.0, 'Uterus': 0.012},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.751, 'Adrenals': 0.008, 'Brain': 1.709, 'Breast': 0.025, 'Colon': 0.001, 'Extrathoracic_Airways': 3.496, 'Gallbladder': 0.001, 'Heart': 0.059, 'Kidneys': 0.008, 'Liver': 0.015000000000000001, 'Lungs': 0.152, 'Lymph_Nodes': 0.9959999999999999, 'Muscle': 0.398, 'Oesophagus': 0.242, 'Oral_Mucosa': 3.3579999999999997, 'Ovaries': 0.0, 'Pancreas': 0.020999999999999998, 'Prostate': 0.0, 'Salivary_Glands': 2.8449999999999998, 'Endosteum': 1.9819999999999998, 'Skin': 0.8260000000000001, 'Small_Intestine': 0.002, 'Spleen': 0.013, 'Stomach': 0.004, 'Testicles': 0.0, 'Thymus': 0.46799999999999997, 'Thyroid': 5.066, 'Urinary_Bladder': 0.001, 'Uterus': 0.016},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.616, 'Adrenals': 0.005, 'Brain': 1.332, 'Breast': 0.017, 'Colon': 0.001, 'Extrathoracic_Airways': 2.956, 'Gallbladder': 0.0, 'Heart': 0.041999999999999996, 'Kidneys': 0.006, 'Liver': 0.01, 'Lungs': 0.114, 'Lymph_Nodes': 0.855, 'Muscle': 0.33399999999999996, 'Oesophagus': 0.176, 'Oral_Mucosa': 2.85, 'Ovaries': 0.0, 'Pancreas': 0.016, 'Prostate': 0.0, 'Salivary_Glands': 2.347, 'Endosteum': 1.663, 'Skin': 0.7649999999999999, 'Small_Intestine': 0.001, 'Spleen': 0.009000000000000001, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.367, 'Thyroid': 4.697, 'Urinary_Bladder': 0.0, 'Uterus': 0.013},
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.149, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.183, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.078, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.159, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08399999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.166, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.23900000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.262, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.094, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.177, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.28200000000000003, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.098, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.316, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.106, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.189, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        5: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
                'F': {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.104089, 'Adrenals': 0.196499, 'Brain': 0.000390944, 'Breast': 0.604535, 'Colon': 1.24174, 'Endosteum': 0.158874, 'Extrathoracic_Airways': 0.00147957, 'Gallbladder': 0.676644, 'Gonads': 0.725414, 'Heart': 0.365, 'Kidneys': 0.342458, 'Lens': 0.00117658, 'Liver': 0.702437, 'Lungs': 0.113642, 'Lymph_Nodes': 0.641272, 'Muscle': 0.232648, 'Oesophagus': 0.114579, 'Oral_Mucosa': 0.00522715, 'Pancreas': 0.570631, 'Salivary_Glands': 0.00348055, 'Skin': 0.286017, 'Small_Intestine': 1.06133, 'Spleen': 0.323549, 'Stomach': 0.821699, 'Thymus': 0.0314682, 'Thyroid': 0.00942435, 'Urinary_Bladder': 1.23494, 'Uterus': 0.578295},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.114934, 'Adrenals': 0.218753, 'Brain': 0.000444176, 'Breast': 0.625191, 'Colon': 1.31989, 'Endosteum': 0.176002, 'Extrathoracic_Airways': 0.0016579, 'Gallbladder': 0.739027, 'Gonads': 0.790185, 'Heart': 0.397074, 'Kidneys': 0.378742, 'Lens': 0.00130189, 'Liver': 0.763189, 'Lungs': 0.124802, 'Lymph_Nodes': 0.681733, 'Muscle': 0.246051, 'Oesophagus': 0.126997, 'Oral_Mucosa': 0.00578188, 'Pancreas': 0.625077, 'Salivary_Glands': 0.00390738, 'Skin': 0.29329, 'Small_Intestine': 1.14483, 'Spleen': 0.356344, 'Stomach': 0.888988, 'Thymus': 0.0351993, 'Thyroid': 0.0106068, 'Urinary_Bladder': 1.3151, 'Uterus': 0.633939},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.170685, 'Adrenals': 0.336663, 'Brain': 0.000770264, 'Breast': 0.715584, 'Colon': 1.65358, 'Endosteum': 0.265725, 'Extrathoracic_Airways': 0.00268073, 'Gallbladder': 1.0358, 'Gonads': 1.0926, 'Heart': 0.547527, 'Kidneys': 0.563197, 'Lens': 0.00196722, 'Liver': 1.04557, 'Lungs': 0.180011, 'Lymph_Nodes': 0.859495, 'Muscle': 0.304924, 'Oesophagus': 0.191063, 'Oral_Mucosa': 0.00870454, 'Pancreas': 0.887316, 'Salivary_Glands': 0.00633155, 'Skin': 0.325317, 'Small_Intestine': 1.51786, 'Spleen': 0.51922, 'Stomach': 1.1946, 'Thymus': 0.0556463, 'Thyroid': 0.0173585, 'Urinary_Bladder': 1.66051, 'Uterus': 0.902324},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.124786, 'Adrenals': 0.239185, 'Brain': 0.000495065, 'Breast': 0.64284, 'Colon': 1.38664, 'Endosteum': 0.191653, 'Extrathoracic_Airways': 0.00182508, 'Gallbladder': 0.79438, 'Gonads': 0.847317, 'Heart': 0.425385, 'Kidneys': 0.411641, 'Lens': 0.00141662, 'Liver': 0.81667, 'Lungs': 0.134822, 'Lymph_Nodes': 0.716586, 'Muscle': 0.257581, 'Oesophagus': 0.138305, 'Oral_Mucosa': 0.00628907, 'Pancreas': 0.673597, 'Salivary_Glands': 0.00430682, 'Skin': 0.299539, 'Small_Intestine': 1.21737, 'Spleen': 0.385862, 'Stomach': 0.947771, 'Thymus': 0.0386585, 'Thyroid': 0.0117159, 'Urinary_Bladder': 1.3838, 'Uterus': 0.683559},
                },
                'M': {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.104447, 'Adrenals': 0.197058, 'Brain': 0.000392228, 'Breast': 0.599087, 'Colon': 1.24111, 'Endosteum': 0.159537, 'Extrathoracic_Airways': 0.00148448, 'Gallbladder': 0.676702, 'Gonads': 1.28862, 'Heart': 0.365167, 'Kidneys': 0.342566, 'Lens': 0.00120622, 'Liver': 0.702531, 'Lungs': 0.113686, 'Lymph_Nodes': 0.639803, 'Muscle': 0.236758, 'Oesophagus': 0.116161, 'Oral_Mucosa': 0.00517682, 'Pancreas': 0.570553, 'Salivary_Glands': 0.0034866, 'Skin': 0.287046, 'Small_Intestine': 1.06427, 'Spleen': 0.323606, 'Stomach': 0.822537, 'Thymus': 0.0314947, 'Thyroid': 0.00951921, 'Urinary_Bladder': 0.888923, 'Prostate': 1.15676},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.115327, 'Adrenals': 0.21936, 'Brain': 0.000445633, 'Breast': 0.619655, 'Colon': 1.31928, 'Endosteum': 0.176736, 'Extrathoracic_Airways': 0.00166362, 'Gallbladder': 0.739103, 'Gonads': 1.3493, 'Heart': 0.397252, 'Kidneys': 0.378857, 'Lens': 0.00133204, 'Liver': 0.763288, 'Lungs': 0.124849, 'Lymph_Nodes': 0.679702, 'Muscle': 0.250331, 'Oesophagus': 0.12875, 'Oral_Mucosa': 0.00572559, 'Pancreas': 0.624986, 'Salivary_Glands': 0.00391394, 'Skin': 0.294416, 'Small_Intestine': 1.14791, 'Spleen': 0.356402, 'Stomach': 0.889895, 'Thymus': 0.0352297, 'Thyroid': 0.0107133, 'Urinary_Bladder': 0.959853, 'Prostate': 1.22652},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.171253, 'Adrenals': 0.33746, 'Brain': 0.000772735, 'Breast': 0.709934, 'Colon': 1.65315, 'Endosteum': 0.266822, 'Extrathoracic_Airways': 0.0026924, 'Gallbladder': 1.03595, 'Gonads': 1.58802, 'Heart': 0.547763, 'Kidneys': 0.56333, 'Lens': 0.0020005, 'Liver': 1.04568, 'Lungs': 0.180073, 'Lymph_Nodes': 0.854528, 'Muscle': 0.309878, 'Oesophagus': 0.193696, 'Oral_Mucosa': 0.00862339, 'Pancreas': 0.887147, 'Salivary_Glands': 0.00633942, 'Skin': 0.326853, 'Small_Intestine': 1.52137, 'Spleen': 0.519275, 'Stomach': 1.19583, 'Thymus': 0.0556981, 'Thyroid': 0.0175314, 'Urinary_Bladder': 1.27945, 'Prostate': 1.51839},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.12521, 'Adrenals': 0.239834, 'Brain': 0.000496684, 'Breast': 0.637247, 'Colon': 1.38605, 'Endosteum': 0.19245, 'Extrathoracic_Airways': 0.00183161, 'Gallbladder': 0.794474, 'Gonads': 1.3996, 'Heart': 0.425574, 'Kidneys': 0.411761, 'Lens': 0.00144718, 'Liver': 0.816773, 'Lungs': 0.134873, 'Lymph_Nodes': 0.714037, 'Muscle': 0.262002, 'Oesophagus': 0.140213, 'Oral_Mucosa': 0.00622753, 'Pancreas': 0.673495, 'Salivary_Glands': 0.00431379, 'Skin': 0.300749, 'Small_Intestine': 1.22056, 'Spleen': 0.385921, 'Stomach': 0.948738, 'Thymus': 0.0386925, 'Thyroid': 0.0118333, 'Urinary_Bladder': 1.02164, 'Prostate': 1.28561},
                },
                'U': {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.104268, 'Adrenals': 0.196779, 'Brain': 0.000391586, 'Breast': 0.601811, 'Colon': 1.24142, 'Endosteum': 0.159205, 'Extrathoracic_Airways': 0.00148202, 'Gallbladder': 0.676673, 'Gonads': 1.00702, 'Heart': 0.365084, 'Kidneys': 0.342512, 'Lens': 0.0011914, 'Liver': 0.702484, 'Lungs': 0.113664, 'Lymph_Nodes': 0.640538, 'Muscle': 0.234703, 'Oesophagus': 0.11537, 'Oral_Mucosa': 0.00520198, 'Pancreas': 0.570592, 'Salivary_Glands': 0.00348358, 'Skin': 0.286532, 'Small_Intestine': 1.0628, 'Spleen': 0.323578, 'Stomach': 0.822118, 'Thymus': 0.0314814, 'Thyroid': 0.00947178, 'Urinary_Bladder': 1.06193, 'Uterus': 0.289147, 'Prostate': 0.578382},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.11513, 'Adrenals': 0.219057, 'Brain': 0.000444904, 'Breast': 0.622423, 'Colon': 1.31958, 'Endosteum': 0.176369, 'Extrathoracic_Airways': 0.00166076, 'Gallbladder': 0.739065, 'Gonads': 1.06974, 'Heart': 0.397163, 'Kidneys': 0.378799, 'Lens': 0.00131697, 'Liver': 0.763239, 'Lungs': 0.124825, 'Lymph_Nodes': 0.680718, 'Muscle': 0.248191, 'Oesophagus': 0.127873, 'Oral_Mucosa': 0.00575373, 'Pancreas': 0.625031, 'Salivary_Glands': 0.00391066, 'Skin': 0.293853, 'Small_Intestine': 1.14637, 'Spleen': 0.356373, 'Stomach': 0.889441, 'Thymus': 0.0352145, 'Thyroid': 0.01066, 'Urinary_Bladder': 1.13748, 'Uterus': 0.316969, 'Prostate': 0.613259},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.170969, 'Adrenals': 0.337062, 'Brain': 0.0007715, 'Breast': 0.712759, 'Colon': 1.65336, 'Endosteum': 0.266274, 'Extrathoracic_Airways': 0.00268657, 'Gallbladder': 1.03588, 'Gonads': 1.34031, 'Heart': 0.547645, 'Kidneys': 0.563264, 'Lens': 0.00198386, 'Liver': 1.04562, 'Lungs': 0.180042, 'Lymph_Nodes': 0.857011, 'Muscle': 0.307401, 'Oesophagus': 0.19238, 'Oral_Mucosa': 0.00866397, 'Pancreas': 0.887232, 'Salivary_Glands': 0.00633548, 'Skin': 0.326085, 'Small_Intestine': 1.51961, 'Spleen': 0.519247, 'Stomach': 1.19522, 'Thymus': 0.0556722, 'Thyroid': 0.0174449, 'Urinary_Bladder': 1.46998, 'Uterus': 0.451162, 'Prostate': 0.759194},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.124998, 'Adrenals': 0.239509, 'Brain': 0.000495874, 'Breast': 0.640044, 'Colon': 1.38634, 'Endosteum': 0.192052, 'Extrathoracic_Airways': 0.00182834, 'Gallbladder': 0.794427, 'Gonads': 1.12346, 'Heart': 0.425479, 'Kidneys': 0.411701, 'Lens': 0.0014319, 'Liver': 0.816721, 'Lungs': 0.134847, 'Lymph_Nodes': 0.715312, 'Muscle': 0.259791, 'Oesophagus': 0.139259, 'Oral_Mucosa': 0.0062583, 'Pancreas': 0.673546, 'Salivary_Glands': 0.0043103, 'Skin': 0.300144, 'Small_Intestine': 1.21897, 'Spleen': 0.385892, 'Stomach': 0.948255, 'Thymus': 0.0386755, 'Thyroid': 0.0117746, 'Urinary_Bladder': 1.20272, 'Uterus': 0.341779, 'Prostate': 0.642805},
                },
            },
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.292, "Brain": 0.005, "Breast": 2.604, "Colon": 0.033, "Endosteum": 0.704, "Extrathoracic_Airways": 0.108, "Gallbladder": 0.333, "Heart": 1.333, "Kidneys": 0.128, "Liver": 0.775, "Lungs": 1.044, "Lymph_Nodes": 0.4, "Muscle": 0.284, "Oesophagus": 0.504, "Oral_Mucosa": 0.058, "Ovaries": 0.01, "Pancreas": 0.57, "Prostate": 0.009, "Salivary_Glands": 0.08, "Skin": 0.314, "Small_Intestine": 0.037, "Spleen": 0.371, "Stomach": 0.86, "Testicles": 0.0, "Thymus": 1.97, "Thyroid": 1.236, "Urinary_Bladder": 0.006, "Uterus": 0.02},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.212, "Adrenals": 0.322, "Brain": 0.006, "Breast": 2.658, "Colon": 0.036, "Endosteum": 0.751, "Extrathoracic_Airways": 0.118, "Gallbladder": 0.364, "Heart": 1.433, "Kidneys": 0.142, "Liver": 0.836, "Lungs": 1.122, "Lymph_Nodes": 0.432, "Muscle": 0.302, "Oesophagus": 0.554, "Oral_Mucosa": 0.064, "Ovaries": 0.011, "Pancreas": 0.624, "Prostate": 0.01, "Salivary_Glands": 0.087, "Skin": 0.321, "Small_Intestine": 0.041, "Spleen": 0.406, "Stomach": 0.923, "Testicles": 0.0, "Thymus": 2.088, "Thyroid": 1.298, "Urinary_Bladder": 0.007, "Uterus": 0.022},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.28, "Adrenals": 0.477, "Brain": 0.009, "Breast": 2.848, "Colon": 0.057, "Endosteum": 0.968, "Extrathoracic_Airways": 0.169, "Gallbladder": 0.52, "Heart": 1.903, "Kidneys": 0.219, "Liver": 1.127, "Lungs": 1.484, "Lymph_Nodes": 0.588, "Muscle": 0.388, "Oesophagus": 0.812, "Oral_Mucosa": 0.094, "Ovaries": 0.015, "Pancreas": 0.9, "Prostate": 0.017, "Salivary_Glands": 0.121, "Skin": 0.357, "Small_Intestine": 0.064, "Spleen": 0.583, "Stomach": 1.225, "Testicles": 0.0, "Thymus": 2.597, "Thyroid": 1.61, "Urinary_Bladder": 0.011, "Uterus": 0.031},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.224, "Adrenals": 0.349, "Brain": 0.006, "Breast": 2.7, "Colon": 0.04, "Endosteum": 0.792, "Extrathoracic_Airways": 0.126, "Gallbladder": 0.391, "Heart": 1.521, "Kidneys": 0.155, "Liver": 0.89, "Lungs": 1.191, "Lymph_Nodes": 0.46, "Muscle": 0.318, "Oesophagus": 0.599, "Oral_Mucosa": 0.069, "Ovaries": 0.011, "Pancreas": 0.673, "Prostate": 0.011, "Salivary_Glands": 0.093, "Skin": 0.328, "Small_Intestine": 0.045, "Spleen": 0.438, "Stomach": 0.979, "Testicles": 0.0, "Thymus": 2.189, "Thyroid": 1.354, "Urinary_Bladder": 0.008, "Uterus": 0.024},
                },
            },
            "Pelvis": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.136, "Adrenals": 0.01, "Brain": 0.0, "Breast": 0.001, "Colon": 0.995, "Endosteum": 0.248, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.156, "Heart": 0.006, "Kidneys": 0.046, "Liver": 0.057, "Lungs": 0.005, "Lymph_Nodes": 0.312, "Muscle": 0.4, "Oesophagus": 0.007, "Oral_Mucosa": 0.0, "Ovaries": 1.058, "Pancreas": 0.047, "Prostate": 1.135, "Salivary_Glands": 0.0, "Skin": 0.293, "Small_Intestine": 0.956, "Spleen": 0.029, "Stomach": 0.074, "Testicles": 3.047, "Thymus": 0.003, "Thyroid": 0.0, "Urinary_Bladder": 1.839, "Uterus": 1.161},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.15, "Adrenals": 0.011, "Brain": 0.0, "Breast": 0.001, "Colon": 1.079, "Endosteum": 0.272, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.172, "Heart": 0.007, "Kidneys": 0.051, "Liver": 0.063, "Lungs": 0.006, "Lymph_Nodes": 0.337, "Muscle": 0.425, "Oesophagus": 0.008, "Oral_Mucosa": 0.0, "Ovaries": 1.151, "Pancreas": 0.052, "Prostate": 1.247, "Salivary_Glands": 0.0, "Skin": 0.299, "Small_Intestine": 1.037, "Spleen": 0.031, "Stomach": 0.081, "Testicles": 3.148, "Thymus": 0.003, "Thyroid": 0.0, "Urinary_Bladder": 1.971, "Uterus": 1.264},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.222, "Adrenals": 0.019, "Brain": 0.0, "Breast": 0.001, "Colon": 1.482, "Endosteum": 0.395, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.251, "Heart": 0.011, "Kidneys": 0.08, "Liver": 0.095, "Lungs": 0.009, "Lymph_Nodes": 0.46, "Muscle": 0.542, "Oesophagus": 0.013, "Oral_Mucosa": 0.0, "Ovaries": 1.612, "Pancreas": 0.083, "Prostate": 1.808, "Salivary_Glands": 0.001, "Skin": 0.327, "Small_Intestine": 1.434, "Spleen": 0.046, "Stomach": 0.12, "Testicles": 3.559, "Thymus": 0.005, "Thyroid": 0.0, "Urinary_Bladder": 2.59, "Uterus": 1.771},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.163, "Adrenals": 0.012, "Brain": 0.0, "Breast": 0.001, "Colon": 1.153, "Endosteum": 0.293, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.186, "Heart": 0.007, "Kidneys": 0.056, "Liver": 0.069, "Lungs": 0.006, "Lymph_Nodes": 0.36, "Muscle": 0.447, "Oesophagus": 0.009, "Oral_Mucosa": 0.0, "Ovaries": 1.235, "Pancreas": 0.057, "Prostate": 1.348, "Salivary_Glands": 0.0, "Skin": 0.304, "Small_Intestine": 1.11, "Spleen": 0.034, "Stomach": 0.088, "Testicles": 3.232, "Thymus": 0.003, "Thyroid": 0.0, "Urinary_Bladder": 2.087, "Uterus": 1.355},
                },
            },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.135, 'Adrenals': 0.705, 'Brain': 0.0, 'Breast': 0.043000000000000003, 'Colon': 1.2730000000000001, 'Extrathoracic_Airways': 0.002, 'Gallbladder': 3.614, 'Heart': 0.283, 'Kidneys': 0.575, 'Liver': 2.195, 'Lungs': 0.13799999999999998, 'Lymph_Nodes': 1.186, 'Muscle': 0.345, 'Oesophagus': 0.501, 'Oral_Mucosa': 0.001, 'Ovaries': 0.25999999999999995, 'Pancreas': 1.693, 'Prostate': 0.028, 'Salivary_Glands': 0.002, 'Endosteum': 0.297, 'Skin': 0.259, 'Small_Intestine': 1.721, 'Spleen': 0.391, 'Stomach': 2.608, 'Testicles': 0.030000000000000002, 'Thymus': 0.023, 'Thyroid': 0.014, 'Urinary_Bladder': 0.103, 'Uterus': 0.304},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.147, 'Adrenals': 0.78, 'Brain': 0.0, 'Breast': 0.046, 'Colon': 1.38, 'Extrathoracic_Airways': 0.002, 'Gallbladder': 3.9090000000000003, 'Heart': 0.311, 'Kidneys': 0.635, 'Liver': 2.351, 'Lungs': 0.152, 'Lymph_Nodes': 1.287, 'Muscle': 0.368, 'Oesophagus': 0.549, 'Oral_Mucosa': 0.001, 'Ovaries': 0.28800000000000003, 'Pancreas': 1.848, 'Prostate': 0.031, 'Salivary_Glands': 0.002, 'Endosteum': 0.319, 'Skin': 0.264, 'Small_Intestine': 1.868, 'Spleen': 0.431, 'Stomach': 2.798, 'Testicles': 0.033, 'Thymus': 0.026, 'Thyroid': 0.016, 'Urinary_Bladder': 0.114, 'Uterus': 0.337},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.21000000000000002, 'Adrenals': 1.177, 'Brain': 0.0, 'Breast': 0.060000000000000005, 'Colon': 1.899, 'Extrathoracic_Airways': 0.003, 'Gallbladder': 5.303999999999999, 'Heart': 0.457, 'Kidneys': 0.9600000000000001, 'Liver': 3.068, 'Lungs': 0.228, 'Lymph_Nodes': 1.774, 'Muscle': 0.47800000000000004, 'Oesophagus': 0.791, 'Oral_Mucosa': 0.002, 'Ovaries': 0.444, 'Pancreas': 2.612, 'Prostate': 0.052, 'Salivary_Glands': 0.003, 'Endosteum': 0.428, 'Skin': 0.28800000000000003, 'Small_Intestine': 2.5799999999999996, 'Spleen': 0.644, 'Stomach': 3.6740000000000004, 'Testicles': 0.051, 'Thymus': 0.041999999999999996, 'Thyroid': 0.026, 'Urinary_Bladder': 0.178, 'Uterus': 0.524},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.158, 'Adrenals': 0.848, 'Brain': 0.0, 'Breast': 0.049, 'Colon': 1.4760000000000002, 'Extrathoracic_Airways': 0.002, 'Gallbladder': 4.169, 'Heart': 0.33599999999999997, 'Kidneys': 0.691, 'Liver': 2.487, 'Lungs': 0.165, 'Lymph_Nodes': 1.377, 'Muscle': 0.389, 'Oesophagus': 0.592, 'Oral_Mucosa': 0.001, 'Ovaries': 0.314, 'Pancreas': 1.986, 'Prostate': 0.034, 'Salivary_Glands': 0.002, 'Endosteum': 0.33799999999999997, 'Skin': 0.268, 'Small_Intestine': 1.9989999999999999, 'Spleen': 0.46799999999999997, 'Stomach': 2.9650000000000003, 'Testicles': 0.036000000000000004, 'Thymus': 0.029, 'Thyroid': 0.017, 'Urinary_Bladder': 0.125, 'Uterus': 0.368},
                },},
            "Thoracic Spine": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.191, "Adrenals": 0.443, "Brain": 0.005, "Breast": 1.497, "Colon": 0.058, "Endosteum": 0.601, "Extrathoracic_Airways": 0.084, "Gallbladder": 0.873, "Heart": 2.249, "Kidneys": 0.226, "Liver": 1.137, "Lungs": 1.168, "Lymph_Nodes": 0.577, "Muscle": 0.317, "Oesophagus": 0.765, "Oral_Mucosa": 0.057, "Ovaries": 0.051, "Pancreas": 0.856, "Prostate": 0.0, "Salivary_Glands": 0.088, "Skin": 0.296, "Small_Intestine": 0.066, "Spleen": 0.356, "Stomach": 1.277, "Testicles": 0.0, "Thymus": 3.295, "Thyroid": 1.347, "Urinary_Bladder": 0.011, "Uterus": 0.007},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.204, "Adrenals": 0.489, "Brain": 0.006, "Breast": 1.548, "Colon": 0.064, "Endosteum": 0.637, "Extrathoracic_Airways": 0.092, "Gallbladder": 0.947, "Heart": 2.411, "Kidneys": 0.25, "Liver": 1.221, "Lungs": 1.257, "Lymph_Nodes": 0.624, "Muscle": 0.339, "Oesophagus": 0.84, "Oral_Mucosa": 0.062, "Ovaries": 0.056, "Pancreas": 0.939, "Prostate": 0.0, "Salivary_Glands": 0.096, "Skin": 0.304, "Small_Intestine": 0.073, "Spleen": 0.39, "Stomach": 1.369, "Testicles": 0.0, "Thymus": 3.489, "Thyroid": 1.42, "Urinary_Bladder": 0.013, "Uterus": 0.008},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.265, "Adrenals": 0.732, "Brain": 0.009, "Breast": 1.764, "Colon": 0.096, "Endosteum": 0.805, "Extrathoracic_Airways": 0.136, "Gallbladder": 1.303, "Heart": 3.157, "Kidneys": 0.381, "Liver": 1.615, "Lungs": 1.675, "Lymph_Nodes": 0.851, "Muscle": 0.443, "Oesophagus": 1.225, "Oral_Mucosa": 0.094, "Ovaries": 0.084, "Pancreas": 1.362, "Prostate": 0.0, "Salivary_Glands": 0.14, "Skin": 0.343, "Small_Intestine": 0.113, "Spleen": 0.567, "Stomach": 1.787, "Testicles": 0.0, "Thymus": 4.32, "Thyroid": 1.747, "Urinary_Bladder": 0.021, "Uterus": 0.014},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.215, "Adrenals": 0.531, "Brain": 0.006, "Breast": 1.591, "Colon": 0.069, "Endosteum": 0.668, "Extrathoracic_Airways": 0.1, "Gallbladder": 1.013, "Heart": 2.552, "Kidneys": 0.272, "Liver": 1.295, "Lungs": 1.335, "Lymph_Nodes": 0.665, "Muscle": 0.358, "Oesophagus": 0.907, "Oral_Mucosa": 0.068, "Ovaries": 0.061, "Pancreas": 1.014, "Prostate": 0.0, "Salivary_Glands": 0.104, "Skin": 0.311, "Small_Intestine": 0.079, "Spleen": 0.421, "Stomach": 1.449, "Testicles": 0.0, "Thymus": 3.654, "Thyroid": 1.483, "Urinary_Bladder": 0.014, "Uterus": 0.009},
                },
            },

            "Cervical Spine": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.227, "Adrenals": 0.013, "Brain": 0.085, "Breast": 0.077, "Colon": 0.001, "Endosteum": 0.974, "Extrathoracic_Airways": 2.662, "Gallbladder": 0.005, "Heart": 0.079, "Kidneys": 0.004, "Liver": 0.011, "Lungs": 0.272, "Lymph_Nodes": 0.873, "Muscle": 0.279, "Oesophagus": 0.342, "Oral_Mucosa": 3.198, "Ovaries": 0.0, "Pancreas": 0.011, "Prostate": 0.0, "Salivary_Glands": 1.256, "Skin": 0.415, "Small_Intestine": 0.001, "Spleen": 0.012, "Stomach": 0.006, "Testicles": 0.0, "Thymus": 1.567, "Thyroid": 7.414, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.243, "Adrenals": 0.014, "Brain": 0.095, "Breast": 0.086, "Colon": 0.001, "Endosteum": 1.039, "Extrathoracic_Airways": 2.852, "Gallbladder": 0.006, "Heart": 0.087, "Kidneys": 0.004, "Liver": 0.013, "Lungs": 0.296, "Lymph_Nodes": 0.927, "Muscle": 0.297, "Oesophagus": 0.376, "Oral_Mucosa": 3.382, "Ovaries": 0.0, "Pancreas": 0.012, "Prostate": 0.0, "Salivary_Glands": 1.365, "Skin": 0.427, "Small_Intestine": 0.001, "Spleen": 0.014, "Stomach": 0.007, "Testicles": 0.0, "Thymus": 1.671, "Thyroid": 7.733, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.321, "Adrenals": 0.024, "Brain": 0.148, "Breast": 0.138, "Colon": 0.001, "Endosteum": 1.338, "Extrathoracic_Airways": 3.736, "Gallbladder": 0.01, "Heart": 0.133, "Kidneys": 0.007, "Liver": 0.02, "Lungs": 0.417, "Lymph_Nodes": 1.175, "Muscle": 0.386, "Oesophagus": 0.552, "Oral_Mucosa": 4.225, "Ovaries": 0.0, "Pancreas": 0.021, "Prostate": 0.0, "Salivary_Glands": 1.891, "Skin": 0.483, "Small_Intestine": 0.002, "Spleen": 0.022, "Stomach": 0.011, "Testicles": 0.0, "Thymus": 2.142, "Thyroid": 9.047, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.258, "Adrenals": 0.016, "Brain": 0.104, "Breast": 0.094, "Colon": 0.001, "Endosteum": 1.096, "Extrathoracic_Airways": 3.019, "Gallbladder": 0.007, "Heart": 0.095, "Kidneys": 0.005, "Liver": 0.014, "Lungs": 0.318, "Lymph_Nodes": 0.974, "Muscle": 0.314, "Oesophagus": 0.406, "Oral_Mucosa": 3.542, "Ovaries": 0.0, "Pancreas": 0.014, "Prostate": 0.0, "Salivary_Glands": 1.461, "Skin": 0.437, "Small_Intestine": 0.001, "Spleen": 0.015, "Stomach": 0.007, "Testicles": 0.0, "Thymus": 1.762, "Thyroid": 8.001, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.256, 'Adrenals': 0.001, 'Brain': 0.839, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.536, 'Gallbladder': 0.001, 'Heart': 0.008, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.02, 'Lymph_Nodes': 0.397, 'Muscle': 0.111, 'Oesophagus': 0.018000000000000002, 'Oral_Mucosa': 1.746, 'Ovaries': 0.0, 'Pancreas': 0.003, 'Prostate': 0.0, 'Salivary_Glands': 1.426, 'Endosteum': 1.161, 'Skin': 0.371, 'Small_Intestine': 0.0, 'Spleen': 0.002, 'Stomach': 0.002, 'Testicles': 0.0, 'Thymus': 0.034, 'Thyroid': 1.3010000000000002, 'Urinary_Bladder': 0.001, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.273, 'Adrenals': 0.002, 'Brain': 0.914, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.645, 'Gallbladder': 0.001, 'Heart': 0.009000000000000001, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.022, 'Lymph_Nodes': 0.423, 'Muscle': 0.118, 'Oesophagus': 0.02, 'Oral_Mucosa': 1.848, 'Ovaries': 0.0, 'Pancreas': 0.003, 'Prostate': 0.0, 'Salivary_Glands': 1.522, 'Endosteum': 1.232, 'Skin': 0.379, 'Small_Intestine': 0.0, 'Spleen': 0.002, 'Stomach': 0.002, 'Testicles': 0.0, 'Thymus': 0.037, 'Thyroid': 1.361, 'Urinary_Bladder': 0.001, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.358, 'Adrenals': 0.003, 'Brain': 1.2899999999999998, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 2.142, 'Gallbladder': 0.001, 'Heart': 0.014, 'Kidneys': 0.001, 'Liver': 0.003, 'Lungs': 0.033, 'Lymph_Nodes': 0.5429999999999999, 'Muscle': 0.149, 'Oesophagus': 0.032, 'Oral_Mucosa': 2.339, 'Ovaries': 0.0, 'Pancreas': 0.005, 'Prostate': 0.0, 'Salivary_Glands': 1.9589999999999999, 'Endosteum': 1.5510000000000002, 'Skin': 0.417, 'Small_Intestine': 0.001, 'Spleen': 0.004, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.058, 'Thyroid': 1.655, 'Urinary_Bladder': 0.001, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.289, 'Adrenals': 0.002, 'Brain': 0.9810000000000001, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 1.74, 'Gallbladder': 0.001, 'Heart': 0.009000000000000001, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.024, 'Lymph_Nodes': 0.446, 'Muscle': 0.124, 'Oesophagus': 0.022, 'Oral_Mucosa': 1.938, 'Ovaries': 0.0, 'Pancreas': 0.004, 'Prostate': 0.0, 'Salivary_Glands': 1.605, 'Endosteum': 1.294, 'Skin': 0.386, 'Small_Intestine': 0.0, 'Spleen': 0.002, 'Stomach': 0.002, 'Testicles': 0.0, 'Thymus': 0.041, 'Thyroid': 1.415, 'Urinary_Bladder': 0.001, 'Uterus': 0.0},
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.149, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.183, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.078, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.159, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08399999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.166, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.23900000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.262, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.094, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.177, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.28200000000000003, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.098, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.316, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.106, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.189, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        10: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
                'F': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.0904324, 'Adrenals': 0.114119, 'Brain': 0.000151939, 'Breast': 0.0535778, 'Colon': 0.825727, 'Endosteum': 0.101279, 'Extrathoracic_Airways': 0.000490324, 'Gallbladder': 0.426075, 'Gonads': 0.390498, 'Heart': 0.151012, 'Kidneys': 0.157409, 'Lens': 0.000287943, 'Liver': 0.40517, 'Lungs': 0.0610279, 'Lymph_Nodes': 0.221812, 'Muscle': 0.178535, 'Oesophagus': 0.0571517, 'Oral_Mucosa': 0.0016869, 'Pancreas': 0.330741, 'Salivary_Glands': 0.00118749, 'Skin': 0.199486, 'Small_Intestine': 0.677608, 'Spleen': 0.191593, 'Stomach': 0.533783, 'Thymus': 0.0126613, 'Thyroid': 0.00365753, 'Urinary_Bladder': 0.608174, 'Uterus': 0.316118},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0996436, 'Adrenals': 0.12689, 'Brain': 0.000172184, 'Breast': 0.0582372, 'Colon': 0.881974, 'Endosteum': 0.112058, 'Extrathoracic_Airways': 0.000551775, 'Gallbladder': 0.4662, 'Gonads': 0.427769, 'Heart': 0.165411, 'Kidneys': 0.174555, 'Lens': 0.000322398, 'Liver': 0.441324, 'Lungs': 0.0673147, 'Lymph_Nodes': 0.240519, 'Muscle': 0.18861, 'Oesophagus': 0.0635326, 'Oral_Mucosa': 0.0018802, 'Pancreas': 0.364177, 'Salivary_Glands': 0.00133706, 'Skin': 0.204506, 'Small_Intestine': 0.733304, 'Spleen': 0.210866, 'Stomach': 0.579661, 'Thymus': 0.0141977, 'Thyroid': 0.0041188, 'Urinary_Bladder': 0.658782, 'Uterus': 0.348066},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.147445, 'Adrenals': 0.19547, 'Brain': 0.000297101, 'Breast': 0.0809126, 'Colon': 1.1272, 'Endosteum': 0.169146, 'Extrathoracic_Airways': 0.000915218, 'Gallbladder': 0.66102, 'Gonads': 0.610157, 'Heart': 0.236746, 'Kidneys': 0.264842, 'Lens': 0.000518828, 'Liver': 0.614175, 'Lungs': 0.0996755, 'Lymph_Nodes': 0.331379, 'Muscle': 0.232988, 'Oesophagus': 0.097725, 'Oral_Mucosa': 0.0029521, 'Pancreas': 0.532493, 'Salivary_Glands': 0.0022166, 'Skin': 0.22678, 'Small_Intestine': 0.989549, 'Spleen': 0.307733, 'Stomach': 0.794631, 'Thymus': 0.0229486, 'Thyroid': 0.00682688, 'Urinary_Bladder': 0.892544, 'Uterus': 0.508812},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.108025, 'Adrenals': 0.138648, 'Brain': 0.000191551, 'Breast': 0.0623903, 'Colon': 0.930356, 'Endosteum': 0.121931, 'Extrathoracic_Airways': 0.00060986, 'Gallbladder': 0.502027, 'Gonads': 0.461124, 'Heart': 0.178334, 'Kidneys': 0.190254, 'Lens': 0.000354625, 'Liver': 0.473421, 'Lungs': 0.0730263, 'Lymph_Nodes': 0.257188, 'Muscle': 0.197277, 'Oesophagus': 0.0694044, 'Oral_Mucosa': 0.00205963, 'Pancreas': 0.394376, 'Salivary_Glands': 0.00147832, 'Skin': 0.208827, 'Small_Intestine': 0.782143, 'Spleen': 0.228263, 'Stomach': 0.620133, 'Thymus': 0.0156364, 'Thyroid': 0.00455435, 'Urinary_Bladder': 0.703218, 'Uterus': 0.376915},
                },
                'M': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.0906135, 'Adrenals': 0.114059, 'Brain': 0.000152326, 'Breast': 0.053109, 'Colon': 0.827425, 'Endosteum': 0.101578, 'Extrathoracic_Airways': 0.000491848, 'Gallbladder': 0.426426, 'Gonads': 0.940982, 'Heart': 0.151018, 'Kidneys': 0.157627, 'Lens': 0.000261826, 'Liver': 0.405252, 'Lungs': 0.0610114, 'Lymph_Nodes': 0.198441, 'Muscle': 0.179273, 'Oesophagus': 0.0547216, 'Oral_Mucosa': 0.00170683, 'Pancreas': 0.331017, 'Salivary_Glands': 0.00118815, 'Skin': 0.200667, 'Small_Intestine': 0.679468, 'Spleen': 0.191741, 'Stomach': 0.533799, 'Thymus': 0.0126337, 'Thyroid': 0.00365139, 'Urinary_Bladder': 0.46764, 'Prostate': 0.57529},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0998434, 'Adrenals': 0.12682, 'Brain': 0.000172606, 'Breast': 0.0577346, 'Colon': 0.883682, 'Endosteum': 0.112391, 'Extrathoracic_Airways': 0.000553714, 'Gallbladder': 0.466572, 'Gonads': 0.98228, 'Heart': 0.16542, 'Kidneys': 0.174791, 'Lens': 0.000294341, 'Liver': 0.44141, 'Lungs': 0.0672968, 'Lymph_Nodes': 0.215927, 'Muscle': 0.18938, 'Oesophagus': 0.0608441, 'Oral_Mucosa': 0.00190207, 'Pancreas': 0.364471, 'Salivary_Glands': 0.00133774, 'Skin': 0.205744, 'Small_Intestine': 0.735273, 'Spleen': 0.211026, 'Stomach': 0.579674, 'Thymus': 0.0141668, 'Thyroid': 0.00411188, 'Urinary_Bladder': 0.509682, 'Prostate': 0.619536},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.147738, 'Adrenals': 0.195333, 'Brain': 0.000297654, 'Breast': 0.0802682, 'Colon': 1.12885, 'Endosteum': 0.16966, 'Extrathoracic_Airways': 0.00092062, 'Gallbladder': 0.661481, 'Gonads': 1.14388, 'Heart': 0.236778, 'Kidneys': 0.265157, 'Lens': 0.000486249, 'Liver': 0.614274, 'Lungs': 0.0996501, 'Lymph_Nodes': 0.301577, 'Muscle': 0.233894, 'Oesophagus': 0.0937003, 'Oral_Mucosa': 0.00298208, 'Pancreas': 0.532852, 'Salivary_Glands': 0.00221731, 'Skin': 0.228252, 'Small_Intestine': 0.991897, 'Spleen': 0.307931, 'Stomach': 0.794612, 'Thymus': 0.0228966, 'Thyroid': 0.00681584, 'Urinary_Bladder': 0.710163, 'Prostate': 0.819803},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.108242, 'Adrenals': 0.138569, 'Brain': 0.000192003, 'Breast': 0.0618588, 'Colon': 0.932066, 'Endosteum': 0.122295, 'Extrathoracic_Airways': 0.000612242, 'Gallbladder': 0.502416, 'Gonads': 1.01643, 'Heart': 0.178347, 'Kidneys': 0.190507, 'Lens': 0.000325042, 'Liver': 0.47351, 'Lungs': 0.073007, 'Lymph_Nodes': 0.231555, 'Muscle': 0.198075, 'Oesophagus': 0.0664806, 'Oral_Mucosa': 0.00208318, 'Pancreas': 0.394685, 'Salivary_Glands': 0.00147901, 'Skin': 0.210114, 'Small_Intestine': 0.7842, 'Spleen': 0.228433, 'Stomach': 0.620143, 'Thymus': 0.0156022, 'Thyroid': 0.0045467, 'Urinary_Bladder': 0.54699, 'Prostate': 0.658077},
                },
                'U': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.090523, 'Adrenals': 0.114089, 'Brain': 0.000152132, 'Breast': 0.0533434, 'Colon': 0.826576, 'Endosteum': 0.101429, 'Extrathoracic_Airways': 0.000491086, 'Gallbladder': 0.42625, 'Gonads': 0.66574, 'Heart': 0.151015, 'Kidneys': 0.157518, 'Lens': 0.000274884, 'Liver': 0.405211, 'Lungs': 0.0610196, 'Lymph_Nodes': 0.210126, 'Muscle': 0.178904, 'Oesophagus': 0.0559367, 'Oral_Mucosa': 0.00169686, 'Pancreas': 0.330879, 'Salivary_Glands': 0.00118782, 'Skin': 0.200077, 'Small_Intestine': 0.678538, 'Spleen': 0.191667, 'Stomach': 0.533791, 'Thymus': 0.0126475, 'Thyroid': 0.00365446, 'Urinary_Bladder': 0.537907, 'Uterus': 0.158059, 'Prostate': 0.287645},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0997435, 'Adrenals': 0.126855, 'Brain': 0.000172395, 'Breast': 0.0579859, 'Colon': 0.882828, 'Endosteum': 0.112225, 'Extrathoracic_Airways': 0.000552745, 'Gallbladder': 0.466386, 'Gonads': 0.705024, 'Heart': 0.165415, 'Kidneys': 0.174673, 'Lens': 0.000308369, 'Liver': 0.441367, 'Lungs': 0.0673058, 'Lymph_Nodes': 0.228223, 'Muscle': 0.188995, 'Oesophagus': 0.0621883, 'Oral_Mucosa': 0.00189114, 'Pancreas': 0.364324, 'Salivary_Glands': 0.0013374, 'Skin': 0.205125, 'Small_Intestine': 0.734288, 'Spleen': 0.210946, 'Stomach': 0.579667, 'Thymus': 0.0141823, 'Thyroid': 0.00411534, 'Urinary_Bladder': 0.584232, 'Uterus': 0.174033, 'Prostate': 0.309768},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.147592, 'Adrenals': 0.195402, 'Brain': 0.000297378, 'Breast': 0.0805904, 'Colon': 1.12802, 'Endosteum': 0.169403, 'Extrathoracic_Airways': 0.000917919, 'Gallbladder': 0.661251, 'Gonads': 0.877019, 'Heart': 0.236762, 'Kidneys': 0.264999, 'Lens': 0.000502539, 'Liver': 0.614224, 'Lungs': 0.0996628, 'Lymph_Nodes': 0.316478, 'Muscle': 0.233441, 'Oesophagus': 0.0957127, 'Oral_Mucosa': 0.00296709, 'Pancreas': 0.532673, 'Salivary_Glands': 0.00221696, 'Skin': 0.227516, 'Small_Intestine': 0.990723, 'Spleen': 0.307832, 'Stomach': 0.794621, 'Thymus': 0.0229226, 'Thyroid': 0.00682136, 'Urinary_Bladder': 0.801353, 'Uterus': 0.254406, 'Prostate': 0.409902},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.108134, 'Adrenals': 0.138608, 'Brain': 0.000191777, 'Breast': 0.0621246, 'Colon': 0.931211, 'Endosteum': 0.122113, 'Extrathoracic_Airways': 0.000611051, 'Gallbladder': 0.502221, 'Gonads': 0.738775, 'Heart': 0.178341, 'Kidneys': 0.19038, 'Lens': 0.000339833, 'Liver': 0.473465, 'Lungs': 0.0730166, 'Lymph_Nodes': 0.244372, 'Muscle': 0.197676, 'Oesophagus': 0.0679425, 'Oral_Mucosa': 0.00207141, 'Pancreas': 0.394531, 'Salivary_Glands': 0.00147867, 'Skin': 0.209471, 'Small_Intestine': 0.783172, 'Spleen': 0.228348, 'Stomach': 0.620138, 'Thymus': 0.0156193, 'Thyroid': 0.00455052, 'Urinary_Bladder': 0.625104, 'Uterus': 0.188458, 'Prostate': 0.329039},
                },
            },
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.135, "Adrenals": 0.153, "Brain": 0.002, "Breast": 1.656, "Colon": 0.027, "Endosteum": 0.418, "Extrathoracic_Airways": 0.112, "Gallbladder": 0.336, "Heart": 0.743, "Kidneys": 0.096, "Liver": 0.51, "Lungs": 0.6, "Lymph_Nodes": 0.252, "Muscle": 0.17, "Oesophagus": 0.2, "Oral_Mucosa": 0.025, "Ovaries": 0.018, "Pancreas": 0.317, "Prostate": 0.002, "Salivary_Glands": 0.032, "Skin": 0.222, "Small_Intestine": 0.03, "Spleen": 0.202, "Stomach": 0.645, "Testicles": 0.0, "Thymus": 1.067, "Thyroid": 0.727, "Urinary_Bladder": 0.001, "Uterus": 0.01},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.145, "Adrenals": 0.17, "Brain": 0.003, "Breast": 1.717, "Colon": 0.03, "Endosteum": 0.446, "Extrathoracic_Airways": 0.12, "Gallbladder": 0.365, "Heart": 0.802, "Kidneys": 0.106, "Liver": 0.551, "Lungs": 0.648, "Lymph_Nodes": 0.272, "Muscle": 0.181, "Oesophagus": 0.22, "Oral_Mucosa": 0.027, "Ovaries": 0.021, "Pancreas": 0.348, "Prostate": 0.002, "Salivary_Glands": 0.035, "Skin": 0.227, "Small_Intestine": 0.033, "Spleen": 0.223, "Stomach": 0.694, "Testicles": 0.0, "Thymus": 1.138, "Thyroid": 0.757, "Urinary_Bladder": 0.001, "Uterus": 0.011},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.192, "Adrenals": 0.258, "Brain": 0.005, "Breast": 1.987, "Colon": 0.045, "Endosteum": 0.578, "Extrathoracic_Airways": 0.161, "Gallbladder": 0.509, "Heart": 1.081, "Kidneys": 0.162, "Liver": 0.749, "Lungs": 0.875, "Lymph_Nodes": 0.369, "Muscle": 0.235, "Oesophagus": 0.329, "Oral_Mucosa": 0.041, "Ovaries": 0.035, "Pancreas": 0.512, "Prostate": 0.004, "Salivary_Glands": 0.051, "Skin": 0.251, "Small_Intestine": 0.052, "Spleen": 0.336, "Stomach": 0.924, "Testicles": 0.0, "Thymus": 1.452, "Thyroid": 0.878, "Urinary_Bladder": 0.002, "Uterus": 0.019},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.153, "Adrenals": 0.185, "Brain": 0.003, "Breast": 1.77, "Colon": 0.033, "Endosteum": 0.471, "Extrathoracic_Airways": 0.128, "Gallbladder": 0.392, "Heart": 0.853, "Kidneys": 0.115, "Liver": 0.587, "Lungs": 0.69, "Lymph_Nodes": 0.289, "Muscle": 0.191, "Oesophagus": 0.239, "Oral_Mucosa": 0.03, "Ovaries": 0.023, "Pancreas": 0.377, "Prostate": 0.002, "Salivary_Glands": 0.038, "Skin": 0.232, "Small_Intestine": 0.036, "Spleen": 0.242, "Stomach": 0.737, "Testicles": 0.0, "Thymus": 1.199, "Thyroid": 0.782, "Urinary_Bladder": 0.001, "Uterus": 0.012},
                },
            },
            "Pelvis": {
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.109, 'Adrenals': 0.006, 'Brain': 0, 'Breast': 0, 'Colon': 0.629, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.077, 'Heart': 0.002, 'Kidneys': 0.029, 'Liver': 0.028, 'Lungs': 0.002, 'Lymph_Nodes': 0.203, 'Muscle': 0.261, 'Oesophagus': 0.003, 'Oral_Mucosa': 0, 'Ovaries': 0.59, 'Pancreas': 0.018, 'Prostate': 0.94, 'Salivary_Glands': 0, 'Endosteum': 0.156, 'Skin': 0.204, 'Small_Intestine': 0.622, 'Spleen': 0.014, 'Stomach': 0.042, 'Testicles': 2.377, 'Thymus': 0, 'Thyroid': 0, 'Urinary_Bladder': 1.406, 'Uterus': 0.872},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.12, 'Adrenals': 0.007, 'Brain': 0, 'Breast': 0, 'Colon': 0.683, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.085, 'Heart': 0.002, 'Kidneys': 0.032, 'Liver': 0.03, 'Lungs': 0.002, 'Lymph_Nodes': 0.219, 'Muscle': 0.277, 'Oesophagus': 0.003, 'Oral_Mucosa': 0, 'Ovaries': 0.646, 'Pancreas': 0.02, 'Prostate': 1.027, 'Salivary_Glands': 0, 'Endosteum': 0.171, 'Skin': 0.208, 'Small_Intestine': 0.675, 'Spleen': 0.016, 'Stomach': 0.046, 'Testicles': 2.445, 'Thymus': 0.001, 'Thyroid': 0, 'Urinary_Bladder': 1.508, 'Uterus': 0.956},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.178, 'Adrenals': 0.011, 'Brain': 0, 'Breast': 0, 'Colon': 0.947, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.126, 'Heart': 0.003, 'Kidneys': 0.051, 'Liver': 0.046, 'Lungs': 0.004, 'Lymph_Nodes': 0.297, 'Muscle': 0.356, 'Oesophagus': 0.005, 'Oral_Mucosa': 0, 'Ovaries': 0.946, 'Pancreas': 0.031, 'Prostate': 1.487, 'Salivary_Glands': 0, 'Endosteum': 0.249, 'Skin': 0.229, 'Small_Intestine': 0.94, 'Spleen': 0.024, 'Stomach': 0.068, 'Testicles': 2.654, 'Thymus': 0.001, 'Thyroid': 0, 'Urinary_Bladder': 1.985, 'Uterus': 1.387},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.129, 'Adrenals': 0.007, 'Brain': 0, 'Breast': 0, 'Colon': 0.731, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.092, 'Heart': 0.002, 'Kidneys': 0.035, 'Liver': 0.033, 'Lungs': 0.003, 'Lymph_Nodes': 0.233, 'Muscle': 0.292, 'Oesophagus': 0.004, 'Oral_Mucosa': 0, 'Ovaries': 0.698, 'Pancreas': 0.022, 'Prostate': 1.106, 'Salivary_Glands': 0, 'Endosteum': 0.185, 'Skin': 0.212, 'Small_Intestine': 0.723, 'Spleen': 0.017, 'Stomach': 0.05, 'Testicles': 2.498, 'Thymus': 0.001, 'Thyroid': 0, 'Urinary_Bladder': 1.597, 'Uterus': 1.032},
                },
            },
            "Lumbar Spine": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.102, 'Adrenals': 0.303, 'Brain': 0.0, 'Breast': 0.039, 'Colon': 0.971, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 2.09, 'Heart': 0.11900000000000001, 'Kidneys': 0.33599999999999997, 'Liver': 1.1, 'Lungs': 0.061, 'Lymph_Nodes': 0.758, 'Muscle': 0.237, 'Oesophagus': 0.155, 'Oral_Mucosa': 0.001, 'Ovaries': 0.289, 'Pancreas': 0.9420000000000001, 'Prostate': 0.020999999999999998, 'Salivary_Glands': 0.001, 'Endosteum': 0.154, 'Skin': 0.17300000000000001, 'Small_Intestine': 1.317, 'Spleen': 0.305, 'Stomach': 1.6689999999999998, 'Testicles': 0.013, 'Thymus': 0.014, 'Thyroid': 0.001, 'Urinary_Bladder': 0.091, 'Uterus': 0.359},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.112, 'Adrenals': 0.33599999999999997, 'Brain': 0.0, 'Breast': 0.043000000000000003, 'Colon': 1.0510000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 2.262, 'Heart': 0.131, 'Kidneys': 0.371, 'Liver': 1.181, 'Lungs': 0.067, 'Lymph_Nodes': 0.8220000000000001, 'Muscle': 0.253, 'Oesophagus': 0.16899999999999998, 'Oral_Mucosa': 0.001, 'Ovaries': 0.317, 'Pancreas': 1.031, 'Prostate': 0.024, 'Salivary_Glands': 0.001, 'Endosteum': 0.166, 'Skin': 0.177, 'Small_Intestine': 1.428, 'Spleen': 0.335, 'Stomach': 1.788, 'Testicles': 0.015000000000000001, 'Thymus': 0.015000000000000001, 'Thyroid': 0.001, 'Urinary_Bladder': 0.101, 'Uterus': 0.397},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.163, 'Adrenals': 0.519, 'Brain': 0.0, 'Breast': 0.066, 'Colon': 1.442, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 3.0959999999999996, 'Heart': 0.196, 'Kidneys': 0.5599999999999999, 'Liver': 1.563, 'Lungs': 0.102, 'Lymph_Nodes': 1.139, 'Muscle': 0.32699999999999996, 'Oesophagus': 0.23800000000000002, 'Oral_Mucosa': 0.001, 'Ovaries': 0.47300000000000003, 'Pancreas': 1.4929999999999999, 'Prostate': 0.039, 'Salivary_Glands': 0.001, 'Endosteum': 0.22699999999999998, 'Skin': 0.194, 'Small_Intestine': 1.97, 'Spleen': 0.49700000000000005, 'Stomach': 2.342, 'Testicles': 0.024, 'Thymus': 0.024, 'Thyroid': 0.001, 'Urinary_Bladder': 0.15100000000000002, 'Uterus': 0.609},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.12000000000000001, 'Adrenals': 0.366, 'Brain': 0.0, 'Breast': 0.047, 'Colon': 1.123, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 2.415, 'Heart': 0.14200000000000002, 'Kidneys': 0.40299999999999997, 'Liver': 1.252, 'Lungs': 0.073, 'Lymph_Nodes': 0.88, 'Muscle': 0.26699999999999996, 'Oesophagus': 0.182, 'Oral_Mucosa': 0.001, 'Ovaries': 0.34299999999999997, 'Pancreas': 1.1119999999999999, 'Prostate': 0.026, 'Salivary_Glands': 0.001, 'Endosteum': 0.177, 'Skin': 0.18000000000000002, 'Small_Intestine': 1.527, 'Spleen': 0.363, 'Stomach': 1.8920000000000001, 'Testicles': 0.016, 'Thymus': 0.017, 'Thyroid': 0.001, 'Urinary_Bladder': 0.10900000000000001, 'Uterus': 0.432},
                },},
            "Thoracic Spine": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.145, "Adrenals": 0.214, "Brain": 0.002, "Breast": 2.447, "Colon": 0.032, "Endosteum": 0.363, "Extrathoracic_Airways": 0.069, "Gallbladder": 0.528, "Heart": 1.283, "Kidneys": 0.142, "Liver": 0.674, "Lungs": 0.805, "Lymph_Nodes": 0.339, "Muscle": 0.192, "Oesophagus": 0.337, "Oral_Mucosa": 0.028, "Ovaries": 0.017, "Pancreas": 0.573, "Prostate": 0.01, "Salivary_Glands": 0.045, "Skin": 0.19, "Small_Intestine": 0.036, "Spleen": 0.229, "Stomach": 0.89, "Testicles": 0.0, "Thymus": 2.018, "Thyroid": 0.394, "Urinary_Bladder": 0.002, "Uterus": 0.004},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.154, "Adrenals": 0.238, "Brain": 0.003, "Breast": 2.493, "Colon": 0.036, "Endosteum": 0.385, "Extrathoracic_Airways": 0.076, "Gallbladder": 0.577, "Heart": 1.383, "Kidneys": 0.157, "Liver": 0.727, "Lungs": 0.869, "Lymph_Nodes": 0.369, "Muscle": 0.206, "Oesophagus": 0.371, "Oral_Mucosa": 0.031, "Ovaries": 0.019, "Pancreas": 0.627, "Prostate": 0.012, "Salivary_Glands": 0.049, "Skin": 0.195, "Small_Intestine": 0.04, "Spleen": 0.252, "Stomach": 0.955, "Testicles": 0.0, "Thymus": 2.149, "Thyroid": 0.425, "Urinary_Bladder": 0.003, "Uterus": 0.004},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.202, "Adrenals": 0.368, "Brain": 0.004, "Breast": 2.659, "Colon": 0.055, "Endosteum": 0.488, "Extrathoracic_Airways": 0.112, "Gallbladder": 0.822, "Heart": 1.86, "Kidneys": 0.244, "Liver": 0.983, "Lungs": 1.17, "Lymph_Nodes": 0.514, "Muscle": 0.271, "Oesophagus": 0.545, "Oral_Mucosa": 0.043, "Ovaries": 0.032, "Pancreas": 0.898, "Prostate": 0.019, "Salivary_Glands": 0.067, "Skin": 0.217, "Small_Intestine": 0.062, "Spleen": 0.38, "Stomach": 1.254, "Testicles": 0.0, "Thymus": 2.733, "Thyroid": 0.589, "Urinary_Bladder": 0.004, "Uterus": 0.006},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.163, "Adrenals": 0.259, "Brain": 0.003, "Breast": 2.529, "Colon": 0.039, "Endosteum": 0.404, "Extrathoracic_Airways": 0.082, "Gallbladder": 0.621, "Heart": 1.472, "Kidneys": 0.172, "Liver": 0.775, "Lungs": 0.925, "Lymph_Nodes": 0.395, "Muscle": 0.218, "Oesophagus": 0.401, "Oral_Mucosa": 0.033, "Ovaries": 0.021, "Pancreas": 0.676, "Prostate": 0.013, "Salivary_Glands": 0.053, "Skin": 0.199, "Small_Intestine": 0.043, "Spleen": 0.274, "Stomach": 1.011, "Testicles": 0.0, "Thymus": 2.262, "Thyroid": 0.453, "Urinary_Bladder": 0.003, "Uterus": 0.005},
                },
            },

            "Cervical Spine": {
                "U": {
                    (70, 2.5, 0.0): {"Active_Bone_Marrow": 0.15, "Adrenals": 0.004, "Brain": 0.063, "Breast": 0.012, "Colon": 0.0, "Endosteum": 0.61, "Extrathoracic_Airways": 2.212, "Gallbladder": 0.001, "Heart": 0.031, "Kidneys": 0.001, "Liver": 0.005, "Lungs": 0.151, "Lymph_Nodes": 0.624, "Muscle": 0.163, "Oesophagus": 0.162, "Oral_Mucosa": 2.326, "Ovaries": 0.0, "Pancreas": 0.004, "Prostate": 0.0, "Salivary_Glands": 0.775, "Skin": 0.272, "Small_Intestine": 0.0, "Spleen": 0.005, "Stomach": 0.002, "Testicles": 0.0, "Thymus": 0.872, "Thyroid": 4.885, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.0, 0.0): {"Active_Bone_Marrow": 0.162, "Adrenals": 0.004, "Brain": 0.071, "Breast": 0.014, "Colon": 0.0, "Endosteum": 0.653, "Extrathoracic_Airways": 2.369, "Gallbladder": 0.001, "Heart": 0.035, "Kidneys": 0.001, "Liver": 0.006, "Lungs": 0.166, "Lymph_Nodes": 0.664, "Muscle": 0.175, "Oesophagus": 0.179, "Oral_Mucosa": 2.473, "Ovaries": 0.0, "Pancreas": 0.005, "Prostate": 0.0, "Salivary_Glands": 0.844, "Skin": 0.279, "Small_Intestine": 0.0, "Spleen": 0.006, "Stomach": 0.002, "Testicles": 0.0, "Thymus": 0.93, "Thyroid": 5.122, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.0, 0.1): {"Active_Bone_Marrow": 0.217, "Adrenals": 0.006, "Brain": 0.113, "Breast": 0.023, "Colon": 0.0, "Endosteum": 0.854, "Extrathoracic_Airways": 3.091, "Gallbladder": 0.002, "Heart": 0.056, "Kidneys": 0.002, "Liver": 0.01, "Lungs": 0.239, "Lymph_Nodes": 0.848, "Muscle": 0.228, "Oesophagus": 0.27, "Oral_Mucosa": 3.144, "Ovaries": 0.0, "Pancreas": 0.008, "Prostate": 0.0, "Salivary_Glands": 1.189, "Skin": 0.31, "Small_Intestine": 0.0, "Spleen": 0.009, "Stomach": 0.003, "Testicles": 0.0, "Thymus": 1.189, "Thyroid": 6.108, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (70, 3.5, 0.0): {"Active_Bone_Marrow": 0.172, "Adrenals": 0.004, "Brain": 0.077, "Breast": 0.015, "Colon": 0.0, "Endosteum": 0.691, "Extrathoracic_Airways": 2.506, "Gallbladder": 0.001, "Heart": 0.038, "Kidneys": 0.001, "Liver": 0.006, "Lungs": 0.179, "Lymph_Nodes": 0.699, "Muscle": 0.185, "Oesophagus": 0.194, "Oral_Mucosa": 2.6, "Ovaries": 0.0, "Pancreas": 0.005, "Prostate": 0.0, "Salivary_Glands": 0.906, "Skin": 0.285, "Small_Intestine": 0.0, "Spleen": 0.006, "Stomach": 0.002, "Testicles": 0.0, "Thymus": 0.981, "Thyroid": 5.321, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.148, 'Adrenals': 0.001, 'Brain': 0.655, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.0410000000000001, 'Gallbladder': 0.0, 'Heart': 0.004, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.008, 'Lymph_Nodes': 0.243, 'Muscle': 0.055, 'Oesophagus': 0.01, 'Oral_Mucosa': 1.503, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.099, 'Endosteum': 0.703, 'Skin': 0.22599999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.012, 'Thyroid': 0.2, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.159, 'Adrenals': 0.001, 'Brain': 0.7150000000000001, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.124, 'Gallbladder': 0.0, 'Heart': 0.004, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.008, 'Lymph_Nodes': 0.261, 'Muscle': 0.058, 'Oesophagus': 0.011, 'Oral_Mucosa': 1.591, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.174, 'Endosteum': 0.748, 'Skin': 0.23, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.013, 'Thyroid': 0.219, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.20900000000000002, 'Adrenals': 0.001, 'Brain': 1.024, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.5250000000000001, 'Gallbladder': 0.001, 'Heart': 0.007, 'Kidneys': 0.0, 'Liver': 0.001, 'Lungs': 0.013, 'Lymph_Nodes': 0.34900000000000003, 'Muscle': 0.074, 'Oesophagus': 0.016, 'Oral_Mucosa': 2.0140000000000002, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.526, 'Endosteum': 0.949, 'Skin': 0.252, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.020999999999999998, 'Thyroid': 0.316, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.16799999999999998, 'Adrenals': 0.001, 'Brain': 0.7699999999999999, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 1.198, 'Gallbladder': 0.0, 'Heart': 0.005, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.009000000000000001, 'Lymph_Nodes': 0.277, 'Muscle': 0.061, 'Oesophagus': 0.012, 'Oral_Mucosa': 1.6689999999999998, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.24, 'Endosteum': 0.786, 'Skin': 0.23399999999999999, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.015000000000000001, 'Thyroid': 0.235, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.149, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.183, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.078, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.159, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08399999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.166, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.23900000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.262, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.094, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.177, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.28200000000000003, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.098, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.316, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.106, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.189, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        15: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Abdomen": {
                'F': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.09415, 'Adrenals': 0.0931176, 'Brain': 7.21749e-05, 'Breast': 0.0667851, 'Colon': 0.755613, 'Endosteum': 0.0874595, 'Extrathoracic_Airways': 0.000270857, 'Gallbladder': 0.293179, 'Gonads': 0.305446, 'Heart': 0.0842788, 'Kidneys': 0.0960834, 'Lens': 0.000154311, 'Liver': 0.363822, 'Lungs': 0.0417949, 'Lymph_Nodes': 0.314308, 'Muscle': 0.0947161, 'Oesophagus': 0.041199, 'Oral_Mucosa': 0.00079033, 'Pancreas': 0.198245, 'Salivary_Glands': 0.000592136, 'Skin': 0.137478, 'Small_Intestine': 0.504294, 'Spleen': 0.171706, 'Stomach': 0.454259, 'Thymus': 0.00553117, 'Thyroid': 0.00264772, 'Urinary_Bladder': 0.392689, 'Uterus': 0.198402},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.103917, 'Adrenals': 0.103721, 'Brain': 8.18704e-05, 'Breast': 0.0715044, 'Colon': 0.805057, 'Endosteum': 0.0967933, 'Extrathoracic_Airways': 0.000305464, 'Gallbladder': 0.323235, 'Gonads': 0.335222, 'Heart': 0.0927618, 'Kidneys': 0.106984, 'Lens': 0.000173577, 'Liver': 0.396699, 'Lungs': 0.046195, 'Lymph_Nodes': 0.3358, 'Muscle': 0.100553, 'Oesophagus': 0.0458695, 'Oral_Mucosa': 0.000888964, 'Pancreas': 0.219546, 'Salivary_Glands': 0.00066751, 'Skin': 0.140898, 'Small_Intestine': 0.546907, 'Spleen': 0.188874, 'Stomach': 0.493154, 'Thymus': 0.00621406, 'Thyroid': 0.0029803, 'Urinary_Bladder': 0.428834, 'Uterus': 0.219502},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.154982, 'Adrenals': 0.16134, 'Brain': 0.000142164, 'Breast': 0.0940195, 'Colon': 1.02104, 'Endosteum': 0.146255, 'Extrathoracic_Airways': 0.000512182, 'Gallbladder': 0.476041, 'Gonads': 0.483212, 'Heart': 0.136052, 'Kidneys': 0.166012, 'Lens': 0.000288253, 'Liver': 0.554834, 'Lungs': 0.0691299, 'Lymph_Nodes': 0.433066, 'Muscle': 0.126853, 'Oesophagus': 0.0711964, 'Oral_Mucosa': 0.00146598, 'Pancreas': 0.330803, 'Salivary_Glands': 0.0011149, 'Skin': 0.156136, 'Small_Intestine': 0.746821, 'Spleen': 0.275145, 'Stomach': 0.675444, 'Thymus': 0.0101618, 'Thyroid': 0.00492988, 'Urinary_Bladder': 0.603889, 'Uterus': 0.329295},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.112824, 'Adrenals': 0.113519, 'Brain': 9.1164e-05, 'Breast': 0.0756749, 'Colon': 0.847584, 'Endosteum': 0.105343, 'Extrathoracic_Airways': 0.000338275, 'Gallbladder': 0.350459, 'Gonads': 0.361987, 'Heart': 0.100446, 'Kidneys': 0.117046, 'Lens': 0.000191808, 'Liver': 0.42594, 'Lungs': 0.0502078, 'Lymph_Nodes': 0.354484, 'Muscle': 0.10561, 'Oesophagus': 0.0501819, 'Oral_Mucosa': 0.000981966, 'Pancreas': 0.239, 'Salivary_Glands': 0.000738871, 'Skin': 0.143845, 'Small_Intestine': 0.584495, 'Spleen': 0.204367, 'Stomach': 0.52746, 'Thymus': 0.00685605, 'Thyroid': 0.00329415, 'Urinary_Bladder': 0.461057, 'Uterus': 0.238746},
                },
                'M': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.0736894, 'Adrenals': 0.166448, 'Brain': 3.79712e-05, 'Breast': 0.0333614, 'Colon': 0.725004, 'Endosteum': 0.0636256, 'Extrathoracic_Airways': 0.000195221, 'Gallbladder': 0.377816, 'Gonads': 0.477813, 'Heart': 0.0521692, 'Kidneys': 0.130711, 'Lens': 0.000313386, 'Liver': 0.46028, 'Lungs': 0.038102, 'Lymph_Nodes': 0.181159, 'Muscle': 0.0947204, 'Oesophagus': 0.0415101, 'Oral_Mucosa': 0.000480724, 'Pancreas': 0.250235, 'Salivary_Glands': 0.000304013, 'Skin': 0.125521, 'Small_Intestine': 0.506708, 'Spleen': 0.234709, 'Stomach': 0.484913, 'Thymus': 0.00422933, 'Thyroid': 0.00175606, 'Urinary_Bladder': 0.258577, 'Prostate': 0.234029},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0815452, 'Adrenals': 0.183656, 'Brain': 4.31224e-05, 'Breast': 0.0366175, 'Colon': 0.770882, 'Endosteum': 0.0705985, 'Extrathoracic_Airways': 0.000218908, 'Gallbladder': 0.415027, 'Gonads': 0.506153, 'Heart': 0.0576964, 'Kidneys': 0.14517, 'Lens': 0.00034844, 'Liver': 0.499522, 'Lungs': 0.042153, 'Lymph_Nodes': 0.196297, 'Muscle': 0.100892, 'Oesophagus': 0.0461183, 'Oral_Mucosa': 0.00053984, 'Pancreas': 0.276301, 'Salivary_Glands': 0.000343644, 'Skin': 0.128701, 'Small_Intestine': 0.549537, 'Spleen': 0.256793, 'Stomach': 0.525089, 'Thymus': 0.00476212, 'Thyroid': 0.00198131, 'Urinary_Bladder': 0.284517, 'Prostate': 0.255942},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.123134, 'Adrenals': 0.272275, 'Brain': 7.54654e-05, 'Breast': 0.0531178, 'Colon': 0.971214, 'Endosteum': 0.108035, 'Extrathoracic_Airways': 0.000357182, 'Gallbladder': 0.600155, 'Gonads': 0.627402, 'Heart': 0.0867187, 'Kidneys': 0.222264, 'Lens': 0.000539356, 'Liver': 0.684078, 'Lungs': 0.0633923, 'Lymph_Nodes': 0.270169, 'Muscle': 0.129136, 'Oesophagus': 0.0707946, 'Oral_Mucosa': 0.000884362, 'Pancreas': 0.409984, 'Salivary_Glands': 0.000583433, 'Skin': 0.143021, 'Small_Intestine': 0.750061, 'Spleen': 0.364013, 'Stomach': 0.711229, 'Thymus': 0.00789029, 'Thyroid': 0.00332452, 'Urinary_Bladder': 0.415372, 'Prostate': 0.362536},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.0887384, 'Adrenals': 0.199294, 'Brain': 4.80721e-05, 'Breast': 0.0395589, 'Colon': 0.810322, 'Endosteum': 0.0770119, 'Extrathoracic_Airways': 0.0002412, 'Gallbladder': 0.448504, 'Gonads': 0.530341, 'Heart': 0.0627472, 'Kidneys': 0.158454, 'Lens': 0.000380799, 'Liver': 0.534163, 'Lungs': 0.045854, 'Lymph_Nodes': 0.209798, 'Muscle': 0.106271, 'Oesophagus': 0.0503571, 'Oral_Mucosa': 0.000595492, 'Pancreas': 0.299976, 'Salivary_Glands': 0.000381362, 'Skin': 0.131452, 'Small_Intestine': 0.587292, 'Spleen': 0.276508, 'Stomach': 0.560385, 'Thymus': 0.00526515, 'Thyroid': 0.00219486, 'Urinary_Bladder': 0.307948, 'Prostate': 0.275506},
                },
                'U': {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.0839197, 'Adrenals': 0.129783, 'Brain': 5.5073e-05, 'Breast': 0.0500732, 'Colon': 0.740308, 'Endosteum': 0.0755425, 'Extrathoracic_Airways': 0.000233039, 'Gallbladder': 0.335497, 'Gonads': 0.391629, 'Heart': 0.068224, 'Kidneys': 0.113397, 'Lens': 0.000233849, 'Liver': 0.412051, 'Lungs': 0.0399484, 'Lymph_Nodes': 0.247733, 'Muscle': 0.0947182, 'Oesophagus': 0.0413545, 'Oral_Mucosa': 0.000635527, 'Pancreas': 0.22424, 'Salivary_Glands': 0.000448074, 'Skin': 0.1315, 'Small_Intestine': 0.505501, 'Spleen': 0.203208, 'Stomach': 0.469586, 'Thymus': 0.00488025, 'Thyroid': 0.00220189, 'Urinary_Bladder': 0.325633, 'Uterus': 0.099201, 'Prostate': 0.117014},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0927309, 'Adrenals': 0.143689, 'Brain': 6.24964e-05, 'Breast': 0.0540609, 'Colon': 0.78797, 'Endosteum': 0.0836959, 'Extrathoracic_Airways': 0.000262186, 'Gallbladder': 0.369131, 'Gonads': 0.420687, 'Heart': 0.0752291, 'Kidneys': 0.126077, 'Lens': 0.000261008, 'Liver': 0.44811, 'Lungs': 0.044174, 'Lymph_Nodes': 0.266048, 'Muscle': 0.100722, 'Oesophagus': 0.0459939, 'Oral_Mucosa': 0.000714402, 'Pancreas': 0.247923, 'Salivary_Glands': 0.000505577, 'Skin': 0.1348, 'Small_Intestine': 0.548222, 'Spleen': 0.222833, 'Stomach': 0.509121, 'Thymus': 0.00548809, 'Thyroid': 0.0024808, 'Urinary_Bladder': 0.356675, 'Uterus': 0.109751, 'Prostate': 0.127971},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.139058, 'Adrenals': 0.216807, 'Brain': 0.000108814, 'Breast': 0.0735687, 'Colon': 0.996128, 'Endosteum': 0.127145, 'Extrathoracic_Airways': 0.000434682, 'Gallbladder': 0.538098, 'Gonads': 0.555307, 'Heart': 0.111385, 'Kidneys': 0.194138, 'Lens': 0.000413804, 'Liver': 0.619456, 'Lungs': 0.0662611, 'Lymph_Nodes': 0.351617, 'Muscle': 0.127995, 'Oesophagus': 0.0709955, 'Oral_Mucosa': 0.00117517, 'Pancreas': 0.370393, 'Salivary_Glands': 0.000849168, 'Skin': 0.149578, 'Small_Intestine': 0.748441, 'Spleen': 0.319579, 'Stomach': 0.693337, 'Thymus': 0.00902604, 'Thyroid': 0.0041272, 'Urinary_Bladder': 0.509631, 'Uterus': 0.164648, 'Prostate': 0.181268},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.100781, 'Adrenals': 0.156406, 'Brain': 6.9618e-05, 'Breast': 0.0576169, 'Colon': 0.828953, 'Endosteum': 0.0911774, 'Extrathoracic_Airways': 0.000289737, 'Gallbladder': 0.399481, 'Gonads': 0.446164, 'Heart': 0.0815965, 'Kidneys': 0.13775, 'Lens': 0.000286304, 'Liver': 0.480052, 'Lungs': 0.0480309, 'Lymph_Nodes': 0.282141, 'Muscle': 0.10594, 'Oesophagus': 0.0502695, 'Oral_Mucosa': 0.000788729, 'Pancreas': 0.269488, 'Salivary_Glands': 0.000560117, 'Skin': 0.137649, 'Small_Intestine': 0.585894, 'Spleen': 0.240437, 'Stomach': 0.543922, 'Thymus': 0.0060606, 'Thyroid': 0.00274451, 'Urinary_Bladder': 0.384502, 'Uterus': 0.119373, 'Prostate': 0.137753},
                },
            },
            "Chest": {
                "U": {
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.131, "Adrenals": 0.145, "Brain": 0.003, "Breast": 1.063, "Colon": 0.02, "Endosteum": 0.292, "Extrathoracic_Airways": 0.114, "Gallbladder": 0.203, "Heart": 0.72, "Kidneys": 0.075, "Liver": 0.433, "Lungs": 0.464, "Lymph_Nodes": 0.216, "Muscle": 0.148, "Oesophagus": 0.249, "Oral_Mucosa": 0.015, "Ovaries": 0.006, "Pancreas": 0.32, "Prostate": 0.0, "Salivary_Glands": 0.024, "Skin": 0.17, "Small_Intestine": 0.02, "Spleen": 0.208, "Stomach": 0.499, "Testicles": 0.0, "Thymus": 1.058, "Thyroid": 0.62, "Urinary_Bladder": 0.002, "Uterus": 0.004},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.139, "Adrenals": 0.157, "Brain": 0.003, "Breast": 1.103, "Colon": 0.022, "Endosteum": 0.308, "Extrathoracic_Airways": 0.121, "Gallbladder": 0.218, "Heart": 0.765, "Kidneys": 0.081, "Liver": 0.461, "Lungs": 0.495, "Lymph_Nodes": 0.23, "Muscle": 0.157, "Oesophagus": 0.269, "Oral_Mucosa": 0.017, "Ovaries": 0.006, "Pancreas": 0.344, "Prostate": 0.0, "Salivary_Glands": 0.026, "Skin": 0.174, "Small_Intestine": 0.021, "Spleen": 0.224, "Stomach": 0.531, "Testicles": 0.0, "Thymus": 1.116, "Thyroid": 0.648, "Urinary_Bladder": 0.002, "Uterus": 0.005},
                    (100, 3.0, 0.1): {"Active_Bone_Marrow": 0.177, "Adrenals": 0.222, "Brain": 0.005, "Breast": 1.266, "Colon": 0.031, "Endosteum": 0.378, "Extrathoracic_Airways": 0.153, "Gallbladder": 0.293, "Heart": 0.977, "Kidneys": 0.114, "Liver": 0.595, "Lungs": 0.644, "Lymph_Nodes": 0.3, "Muscle": 0.198, "Oesophagus": 0.369, "Oral_Mucosa": 0.022, "Ovaries": 0.01, "Pancreas": 0.471, "Prostate": 0.0, "Salivary_Glands": 0.037, "Skin": 0.192, "Small_Intestine": 0.031, "Spleen": 0.309, "Stomach": 0.677, "Testicles": 0.0, "Thymus": 1.372, "Thyroid": 0.767, "Urinary_Bladder": 0.003, "Uterus": 0.007},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.146, "Adrenals": 0.168, "Brain": 0.003, "Breast": 1.136, "Colon": 0.023, "Endosteum": 0.321, "Extrathoracic_Airways": 0.127, "Gallbladder": 0.231, "Heart": 0.805, "Kidneys": 0.087, "Liver": 0.486, "Lungs": 0.523, "Lymph_Nodes": 0.243, "Muscle": 0.165, "Oesophagus": 0.286, "Oral_Mucosa": 0.017, "Ovaries": 0.007, "Pancreas": 0.367, "Prostate": 0.0, "Salivary_Glands": 0.028, "Skin": 0.177, "Small_Intestine": 0.023, "Spleen": 0.239, "Stomach": 0.558, "Testicles": 0.0, "Thymus": 1.166, "Thyroid": 0.672, "Urinary_Bladder": 0.002, "Uterus": 0.005},
                },
            },
            "Pelvis": {
                "U": {
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.078, 'Adrenals': 0.004, 'Brain': 0, 'Breast': 0.001, 'Colon': 0.4, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.049, 'Heart': 0.001, 'Kidneys': 0.014, 'Liver': 0.011, 'Lungs': 0.001, 'Lymph_Nodes': 0.133, 'Muscle': 0.175, 'Oesophagus': 0.002, 'Oral_Mucosa': 0, 'Ovaries': 0.443, 'Pancreas': 0.008, 'Prostate': 0.641, 'Salivary_Glands': 0, 'Endosteum': 0.094, 'Skin': 0.141, 'Small_Intestine': 0.409, 'Spleen': 0.007, 'Stomach': 0.024, 'Testicles': 1.577, 'Thymus': 0, 'Thyroid': 0, 'Urinary_Bladder': 1.017, 'Uterus': 0.58},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.085, 'Adrenals': 0.005, 'Brain': 0, 'Breast': 0.002, 'Colon': 0.435, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.054, 'Heart': 0.001, 'Kidneys': 0.015, 'Liver': 0.013, 'Lungs': 0.001, 'Lymph_Nodes': 0.144, 'Muscle': 0.186, 'Oesophagus': 0.002, 'Oral_Mucosa': 0, 'Ovaries': 0.482, 'Pancreas': 0.009, 'Prostate': 0.69, 'Salivary_Glands': 0, 'Endosteum': 0.103, 'Skin': 0.144, 'Small_Intestine': 0.444, 'Spleen': 0.008, 'Stomach': 0.027, 'Testicles': 1.646, 'Thymus': 0, 'Thyroid': 0, 'Urinary_Bladder': 1.092, 'Uterus': 0.631},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.127, 'Adrenals': 0.008, 'Brain': 0, 'Breast': 0.002, 'Colon': 0.612, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.082, 'Heart': 0.002, 'Kidneys': 0.024, 'Liver': 0.019, 'Lungs': 0.002, 'Lymph_Nodes': 0.198, 'Muscle': 0.24, 'Oesophagus': 0.004, 'Oral_Mucosa': 0, 'Ovaries': 0.679, 'Pancreas': 0.014, 'Prostate': 0.923, 'Salivary_Glands': 0, 'Endosteum': 0.15, 'Skin': 0.161, 'Small_Intestine': 0.624, 'Spleen': 0.013, 'Stomach': 0.04, 'Testicles': 1.945, 'Thymus': 0.001, 'Thyroid': 0, 'Urinary_Bladder': 1.448, 'Uterus': 0.891},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.092, 'Adrenals': 0.005, 'Brain': 0, 'Breast': 0.002, 'Colon': 0.466, 'Extrathoracic_Airways': 0, 'Gallbladder': 0.059, 'Heart': 0.001, 'Kidneys': 0.017, 'Liver': 0.014, 'Lungs': 0.001, 'Lymph_Nodes': 0.153, 'Muscle': 0.196, 'Oesophagus': 0.003, 'Oral_Mucosa': 0, 'Ovaries': 0.516, 'Pancreas': 0.01, 'Prostate': 0.732, 'Salivary_Glands': 0, 'Endosteum': 0.111, 'Skin': 0.147, 'Small_Intestine': 0.476, 'Spleen': 0.008, 'Stomach': 0.029, 'Testicles': 1.704, 'Thymus': 0, 'Thyroid': 0, 'Urinary_Bladder': 1.158, 'Uterus': 0.677},
                },
            },
            "Lumbar Spine": {                "U": {
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.077, 'Adrenals': 0.146, 'Brain': 0.0, 'Breast': 0.015000000000000001, 'Colon': 0.629, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 1.026, 'Heart': 0.06899999999999999, 'Kidneys': 0.159, 'Liver': 0.555, 'Lungs': 0.038, 'Lymph_Nodes': 0.40700000000000003, 'Muscle': 0.16, 'Oesophagus': 0.127, 'Oral_Mucosa': 0.0, 'Ovaries': 0.426, 'Pancreas': 0.477, 'Prostate': 0.031, 'Salivary_Glands': 0.001, 'Endosteum': 0.097, 'Skin': 0.12300000000000001, 'Small_Intestine': 0.7290000000000001, 'Spleen': 0.192, 'Stomach': 1.0399999999999998, 'Testicles': 0.006, 'Thymus': 0.012, 'Thyroid': 0.0, 'Urinary_Bladder': 0.08600000000000001, 'Uterus': 0.455},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.083, 'Adrenals': 0.16, 'Brain': 0.0, 'Breast': 0.017, 'Colon': 0.6809999999999999, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 1.1119999999999999, 'Heart': 0.076, 'Kidneys': 0.176, 'Liver': 0.598, 'Lungs': 0.041999999999999996, 'Lymph_Nodes': 0.441, 'Muscle': 0.171, 'Oesophagus': 0.13799999999999998, 'Oral_Mucosa': 0.0, 'Ovaries': 0.465, 'Pancreas': 0.521, 'Prostate': 0.034, 'Salivary_Glands': 0.001, 'Endosteum': 0.104, 'Skin': 0.126, 'Small_Intestine': 0.7919999999999999, 'Spleen': 0.211, 'Stomach': 1.1150000000000002, 'Testicles': 0.007, 'Thymus': 0.014, 'Thyroid': 0.0, 'Urinary_Bladder': 0.094, 'Uterus': 0.496},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.12000000000000001, 'Adrenals': 0.236, 'Brain': 0.0, 'Breast': 0.025, 'Colon': 0.937, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 1.5430000000000001, 'Heart': 0.11299999999999999, 'Kidneys': 0.266, 'Liver': 0.805, 'Lungs': 0.062, 'Lymph_Nodes': 0.616, 'Muscle': 0.222, 'Oesophagus': 0.191, 'Oral_Mucosa': 0.0, 'Ovaries': 0.6669999999999999, 'Pancreas': 0.754, 'Prostate': 0.054, 'Salivary_Glands': 0.001, 'Endosteum': 0.13899999999999998, 'Skin': 0.13899999999999998, 'Small_Intestine': 1.1039999999999999, 'Spleen': 0.312, 'Stomach': 1.472, 'Testicles': 0.011, 'Thymus': 0.022, 'Thyroid': 0.0, 'Urinary_Bladder': 0.13899999999999998, 'Uterus': 0.7110000000000001},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.09000000000000001, 'Adrenals': 0.17200000000000001, 'Brain': 0.0, 'Breast': 0.018000000000000002, 'Colon': 0.727, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 1.1900000000000002, 'Heart': 0.082, 'Kidneys': 0.191, 'Liver': 0.636, 'Lungs': 0.045000000000000005, 'Lymph_Nodes': 0.472, 'Muscle': 0.18000000000000002, 'Oesophagus': 0.147, 'Oral_Mucosa': 0.0, 'Ovaries': 0.501, 'Pancreas': 0.562, 'Prostate': 0.037, 'Salivary_Glands': 0.001, 'Endosteum': 0.11, 'Skin': 0.128, 'Small_Intestine': 0.847, 'Spleen': 0.228, 'Stomach': 1.181, 'Testicles': 0.008, 'Thymus': 0.015000000000000001, 'Thyroid': 0.0, 'Urinary_Bladder': 0.102, 'Uterus': 0.5339999999999999},
                },},
            "Thoracic Spine": {
                "U": {
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.138, "Adrenals": 0.174, "Brain": 0.003, "Breast": 0.887, "Colon": 0.023, "Endosteum": 0.274, "Extrathoracic_Airways": 0.203, "Gallbladder": 0.314, "Heart": 1.041, "Kidneys": 0.097, "Liver": 0.499, "Lungs": 0.566, "Lymph_Nodes": 0.306, "Muscle": 0.159, "Oesophagus": 0.38, "Oral_Mucosa": 0.025, "Ovaries": 0.006, "Pancreas": 0.436, "Prostate": 0.0, "Salivary_Glands": 0.03, "Skin": 0.158, "Small_Intestine": 0.025, "Spleen": 0.183, "Stomach": 0.625, "Testicles": 0.0, "Thymus": 1.594, "Thyroid": 0.99, "Urinary_Bladder": 0.002, "Uterus": 0.004},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.145, "Adrenals": 0.189, "Brain": 0.003, "Breast": 0.919, "Colon": 0.025, "Endosteum": 0.287, "Extrathoracic_Airways": 0.214, "Gallbladder": 0.337, "Heart": 1.105, "Kidneys": 0.106, "Liver": 0.532, "Lungs": 0.604, "Lymph_Nodes": 0.326, "Muscle": 0.169, "Oesophagus": 0.411, "Oral_Mucosa": 0.027, "Ovaries": 0.006, "Pancreas": 0.469, "Prostate": 0.0, "Salivary_Glands": 0.032, "Skin": 0.162, "Small_Intestine": 0.027, "Spleen": 0.197, "Stomach": 0.663, "Testicles": 0.0, "Thymus": 1.682, "Thyroid": 1.03, "Urinary_Bladder": 0.002, "Uterus": 0.004},
                    (100, 3.0, 0.1): {"Active_Bone_Marrow": 0.183, "Adrenals": 0.266, "Brain": 0.005, "Breast": 1.056, "Colon": 0.036, "Endosteum": 0.346, "Extrathoracic_Airways": 0.267, "Gallbladder": 0.452, "Heart": 1.403, "Kidneys": 0.15, "Liver": 0.684, "Lungs": 0.785, "Lymph_Nodes": 0.421, "Muscle": 0.214, "Oesophagus": 0.568, "Oral_Mucosa": 0.038, "Ovaries": 0.009, "Pancreas": 0.64, "Prostate": 0.0, "Salivary_Glands": 0.044, "Skin": 0.179, "Small_Intestine": 0.039, "Spleen": 0.274, "Stomach": 0.839, "Testicles": 0.0, "Thymus": 2.075, "Thyroid": 1.206, "Urinary_Bladder": 0.003, "Uterus": 0.006},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.152, "Adrenals": 0.202, "Brain": 0.004, "Breast": 0.947, "Colon": 0.027, "Endosteum": 0.298, "Extrathoracic_Airways": 0.224, "Gallbladder": 0.358, "Heart": 1.161, "Kidneys": 0.113, "Liver": 0.56, "Lungs": 0.637, "Lymph_Nodes": 0.343, "Muscle": 0.177, "Oesophagus": 0.438, "Oral_Mucosa": 0.029, "Ovaries": 0.007, "Pancreas": 0.5, "Prostate": 0.0, "Salivary_Glands": 0.035, "Skin": 0.165, "Small_Intestine": 0.03, "Spleen": 0.211, "Stomach": 0.696, "Testicles": 0.0, "Thymus": 1.758, "Thyroid": 1.065, "Urinary_Bladder": 0.002, "Uterus": 0.004},
                },
            },

            "Cervical Spine": {
                "U": {
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.126, "Adrenals": 0.005, "Brain": 0.044, "Breast": 0.023, "Colon": 0.0, "Endosteum": 0.403, "Extrathoracic_Airways": 1.842, "Gallbladder": 0.001, "Heart": 0.029, "Kidneys": 0.001, "Liver": 0.005, "Lungs": 0.104, "Lymph_Nodes": 0.554, "Muscle": 0.132, "Oesophagus": 0.169, "Oral_Mucosa": 1.576, "Ovaries": 0.0, "Pancreas": 0.003, "Prostate": 0.0, "Salivary_Glands": 1.059, "Skin": 0.204, "Small_Intestine": 0.0, "Spleen": 0.006, "Stomach": 0.003, "Testicles": 0.0, "Thymus": 0.267, "Thyroid": 4.124, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.133, "Adrenals": 0.006, "Brain": 0.048, "Breast": 0.025, "Colon": 0.0, "Endosteum": 0.426, "Extrathoracic_Airways": 1.936, "Gallbladder": 0.001, "Heart": 0.032, "Kidneys": 0.002, "Liver": 0.006, "Lungs": 0.112, "Lymph_Nodes": 0.581, "Muscle": 0.139, "Oesophagus": 0.183, "Oral_Mucosa": 1.66, "Ovaries": 0.0, "Pancreas": 0.003, "Prostate": 0.0, "Salivary_Glands": 1.127, "Skin": 0.209, "Small_Intestine": 0.0, "Spleen": 0.006, "Stomach": 0.004, "Testicles": 0.0, "Thymus": 0.286, "Thyroid": 4.264, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.1): {"Active_Bone_Marrow": 0.172, "Adrenals": 0.008, "Brain": 0.069, "Breast": 0.034, "Colon": 0.0, "Endosteum": 0.529, "Extrathoracic_Airways": 2.356, "Gallbladder": 0.001, "Heart": 0.045, "Kidneys": 0.002, "Liver": 0.009, "Lungs": 0.153, "Lymph_Nodes": 0.698, "Muscle": 0.174, "Oesophagus": 0.253, "Oral_Mucosa": 2.042, "Ovaries": 0.0, "Pancreas": 0.004, "Prostate": 0.0, "Salivary_Glands": 1.449, "Skin": 0.232, "Small_Intestine": 0.0, "Spleen": 0.009, "Stomach": 0.005, "Testicles": 0.0, "Thymus": 0.376, "Thyroid": 4.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.14, "Adrenals": 0.006, "Brain": 0.051, "Breast": 0.027, "Colon": 0.0, "Endosteum": 0.446, "Extrathoracic_Airways": 2.017, "Gallbladder": 0.001, "Heart": 0.034, "Kidneys": 0.002, "Liver": 0.006, "Lungs": 0.119, "Lymph_Nodes": 0.603, "Muscle": 0.146, "Oesophagus": 0.195, "Oral_Mucosa": 1.732, "Ovaries": 0.0, "Pancreas": 0.003, "Prostate": 0.0, "Salivary_Glands": 1.187, "Skin": 0.214, "Small_Intestine": 0.0, "Spleen": 0.007, "Stomach": 0.004, "Testicles": 0.0, "Thymus": 0.303, "Thyroid": 4.382, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.108, 'Adrenals': 0.0, 'Brain': 0.602, 'Breast': 0.002, 'Colon': 0.0, 'Extrathoracic_Airways': 0.651, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.003, 'Lymph_Nodes': 0.17200000000000001, 'Muscle': 0.034999999999999996, 'Oesophagus': 0.005, 'Oral_Mucosa': 1.357, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.977, 'Endosteum': 0.475, 'Skin': 0.171, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.004, 'Thyroid': 0.075, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.116, 'Adrenals': 0.0, 'Brain': 0.6579999999999999, 'Breast': 0.002, 'Colon': 0.0, 'Extrathoracic_Airways': 0.709, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.003, 'Lymph_Nodes': 0.186, 'Muscle': 0.037, 'Oesophagus': 0.006, 'Oral_Mucosa': 1.442, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.048, 'Endosteum': 0.508, 'Skin': 0.175, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.004, 'Thyroid': 0.082, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.157, 'Adrenals': 0.0, 'Brain': 0.9510000000000001, 'Breast': 0.003, 'Colon': 0.0, 'Extrathoracic_Airways': 1.004, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.005, 'Lymph_Nodes': 0.256, 'Muscle': 0.047, 'Oesophagus': 0.008, 'Oral_Mucosa': 1.856, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.3860000000000001, 'Endosteum': 0.662, 'Skin': 0.192, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.006, 'Thyroid': 0.118, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.124, 'Adrenals': 0.0, 'Brain': 0.71, 'Breast': 0.002, 'Colon': 0.0, 'Extrathoracic_Airways': 0.761, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.004, 'Lymph_Nodes': 0.199, 'Muscle': 0.039, 'Oesophagus': 0.006, 'Oral_Mucosa': 1.518, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.11, 'Endosteum': 0.537, 'Skin': 0.178, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.004, 'Thyroid': 0.088, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.053, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.133, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.098, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.056, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.059, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13899999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (45, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.149, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.058, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.122, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.062, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.132, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.146, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (50, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.183, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.078, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.159, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.003, 'Muscle': 0.063, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.14300000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.147, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.156, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.06999999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (55, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08399999999999999, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.166, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.148, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.067, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.163, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.177, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.23900000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.07100000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.152, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.196, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.262, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.094, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.177, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.18100000000000002, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.075, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.19799999999999998, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.004, 'Muscle': 0.079, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.213, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.083, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.164, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.28200000000000003, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.098, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.211, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.082, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.163, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.229, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.08600000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.245, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.005, 'Muscle': 0.09000000000000001, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.17200000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.316, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.006, 'Muscle': 0.106, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.189, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
    },
}


# AP effective dose coefficients (mSv / Gy·cm²)

eff_coefficients_AP: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
            'F': {
                (60, 2.5, 0.0): 0.166847,
                (60, 3.0, 0.0): 0.18177,
                (60, 3.5, 0.0): 0.195021,
                (80, 2.5, 0.0): 0.223433,
                (80, 3.0, 0.0): 0.24109,
                (80, 3.5, 0.0): 0.256667,
                (100, 2.5, 0.0): 0.272025,
                (100, 3.0, 0.0): 0.290409,
                (100, 3.5, 0.0): 0.306425,
                (120, 2.5, 0.0): 0.309967,
                (120, 3.0, 0.0): 0.327698,
                (120, 3.5, 0.0): 0.342954,
            },
            'M': {
                (60, 2.5, 0.0): 0.108895,
                (60, 3.0, 0.0): 0.119508,
                (60, 3.5, 0.0): 0.129008,
                (80, 2.5, 0.0): 0.151453,
                (80, 3.0, 0.0): 0.16435,
                (80, 3.5, 0.0): 0.175806,
                (100, 2.5, 0.0): 0.188553,
                (100, 3.0, 0.0): 0.202189,
                (100, 3.5, 0.0): 0.214139,
                (120, 2.5, 0.0): 0.217757,
                (120, 3.0, 0.0): 0.231031,
                (120, 3.5, 0.0): 0.242515,
            },
            'U': {
                (60, 2.5, 0.0): 0.137871,
                (60, 3.0, 0.0): 0.150639,
                (60, 3.5, 0.0): 0.162015,
                (80, 2.5, 0.0): 0.187443,
                (80, 3.0, 0.0): 0.20272,
                (80, 3.5, 0.0): 0.216236,
                (100, 2.5, 0.0): 0.230289,
                (100, 3.0, 0.0): 0.246299,
                (100, 3.5, 0.0): 0.260282,
                (120, 2.5, 0.0): 0.263862,
                (120, 3.0, 0.0): 0.279365,
                (120, 3.5, 0.0): 0.292735,
            },
        },
            "Chest": {
                "U": {
                    (60, 2.5, 0.0): 0.159,
                    (60, 3.0, 0.0): 0.171,
                    (60, 3.5, 0.0): 0.181,
                    (80, 2.5, 0.0): 0.205,
                    (80, 3.0, 0.0): 0.219,
                    (80, 3.5, 0.0): 0.23,
                    (100, 2.5, 0.0): 0.243,
                    (100, 3.0, 0.0): 0.256,
                    (100, 3.5, 0.0): 0.268,
                    (120, 2.5, 0.0): 0.272,
                    (120, 3.0, 0.0): 0.284,
                    (120, 3.5, 0.0): 0.294,
                },
            },
            "Pelvis": {
                "U": {
                    (60, 2.5, 0.0): 0.093,
                    (60, 3.0, 0.0): 0.101,
                    (60, 3.5, 0.0): 0.108,
                    (80, 2.5, 0.0): 0.127,
                    (80, 3.0, 0.0): 0.137,
                    (80, 3.5, 0.0): 0.145,
                    (100, 2.5, 0.0): 0.157,
                    (100, 3.0, 0.0): 0.167,
                    (100, 3.5, 0.0): 0.175,
                    (120, 2.5, 0.0): 0.18,
                    (120, 3.0, 0.0): 0.189,
                    (120, 3.5, 0.0): 0.197,
                },
            },
            "Lumbar Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.157,
                    (60, 3.0, 0.0): 0.172,
                    (60, 3.5, 0.0): 0.186,
                    (80, 2.5, 0.0): 0.226,
                    (80, 3.0, 0.0): 0.245,
                    (80, 3.5, 0.0): 0.261,
                    (100, 2.5, 0.0): 0.287,
                    (100, 3.0, 0.0): 0.306,
                    (100, 3.5, 0.0): 0.323,
                    (120, 2.5, 0.0): 0.335,
                    (120, 3.0, 0.0): 0.353,
                    (120, 3.5, 0.0): 0.369,
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.184,
                    (60, 3.0, 0.0): 0.198,
                    (60, 3.5, 0.0): 0.21,
                    (80, 2.5, 0.0): 0.243,
                    (80, 3.0, 0.0): 0.26,
                    (80, 3.5, 0.0): 0.274,
                    (100, 2.5, 0.0): 0.294,
                    (100, 3.0, 0.0): 0.31,
                    (100, 3.5, 0.0): 0.325,
                    (120, 2.5, 0.0): 0.332,
                    (120, 3.0, 0.0): 0.348,
                    (120, 3.5, 0.0): 0.362,
                },
            },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.137,
                    (60, 3.0, 0.0): 0.145,
                    (60, 3.5, 0.0): 0.152,
                    (80, 2.5, 0.0): 0.163,
                    (80, 3.0, 0.0): 0.172,
                    (80, 3.5, 0.0): 0.179,
                    (100, 2.5, 0.0): 0.184,
                    (100, 3.0, 0.0): 0.192,
                    (100, 3.5, 0.0): 0.199,
                    (120, 2.5, 0.0): 0.2,
                    (120, 3.0, 0.0): 0.207,
                    (120, 3.5, 0.0): 0.214,
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): 0.036,
                    (60, 3.0, 0.0): 0.039,
                    (60, 3.5, 0.0): 0.042,
                    (80, 2.5, 0.0): 0.05,
                    (80, 3.0, 0.0): 0.053,
                    (80, 3.5, 0.0): 0.057,
                    (100, 2.5, 0.0): 0.062,
                    (100, 3.0, 0.0): 0.066,
                    (100, 3.5, 0.0): 0.069,
                    (120, 2.5, 0.0): 0.072,
                    (120, 3.0, 0.0): 0.076,
                    (120, 3.5, 0.0): 0.079,
                },
            },
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
                'F': {
                    (65, 2.5, 0.0): 2.53392,
                    (65, 3.0, 0.0): 2.68123,
                    (65, 3.0, 0.1): 3.31543,
                    (65, 3.5, 0.0): 2.80754,
                },
                'M': {
                    (65, 2.5, 0.0): 2.31977,
                    (65, 3.0, 0.0): 2.45223,
                    (65, 3.0, 0.1): 3.0229,
                    (65, 3.5, 0.0): 2.5658,
                },
                'U': {
                    (65, 2.5, 0.0): 2.42684,
                    (65, 3.0, 0.0): 2.56673,
                    (65, 3.0, 0.1): 3.16916,
                    (65, 3.5, 0.0): 2.68667,
                },
            },
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): 3.212,
                    (65, 3.0, 0.0): 3.336,
                    (65, 3.0, 0.1): 3.839,
                    (65, 3.5, 0.0): 3.44,
                },
            },
            "Pelvis": {
                "U": {
                    (65, 2.5, 0.0): 1.509,
                    (65, 3.0, 0.0): 1.617,
                    (65, 3.0, 0.1): 2.097,
                    (65, 3.5, 0.0): 1.71,
                },
            },
            "Lumbar Spine": {                "U": {
                    (65, 2.5, 0.0): 2.953,
                    (65, 3.0, 0.0): 3.163,
                    (65, 3.0, 0.1): 4.107,
                    (65, 3.5, 0.0): 3.3440000000000003,
                },},
            "Thoracic Spine": {
                "U": {
                    (65, 2.5, 0.0): 2.321,
                    (65, 3.0, 0.0): 2.48,
                    (65, 3.0, 0.1): 3.195,
                    (65, 3.5, 0.0): 2.618,
                },
            },
                "Scoliosis": {
                    "U": {
                        (65, 2.5, 0.0): 2.473,
                        (65, 3.0, 0.0): 2.612,
                        (65, 3.0, 0.1): 3.221,
                        (65, 3.5, 0.0): 2.733,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (65, 2.5, 0.0): 1.656,
                    (65, 3.0, 0.0): 1.736,
                    (65, 3.0, 0.1): 2.076,
                    (65, 3.5, 0.0): 1.804,
                },
            },
            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): 0.639,
                    (70, 3.0, 0.0): 0.671,
                    (70, 3.0, 0.1): 0.815,
                    (70, 3.5, 0.0): 0.699,
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (45, 3.0, 0.1): 0.004,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (50, 3.0, 0.1): 0.004,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (55, 3.0, 0.1): 0.005,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (60, 3.0, 0.1): 0.005,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (65, 3.0, 0.1): 0.005,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (70, 3.0, 0.1): 0.006,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                    (80, 3.0, 0.1): 0.006,
                },
            },
        },
        1: {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
                'F': {
                    (65, 2.5, 0.0): 0.854756,
                    (65, 3.0, 0.0): 0.920961,
                    (65, 3.0, 0.1): 1.22401,
                    (65, 3.5, 0.0): 0.978943,
                },
                'M': {
                    (65, 2.5, 0.0): 0.80651,
                    (65, 3.0, 0.0): 0.866755,
                    (65, 3.0, 0.1): 1.14162,
                    (65, 3.5, 0.0): 0.919442,
                },
                'U': {
                    (65, 2.5, 0.0): 0.830633,
                    (65, 3.0, 0.0): 0.893858,
                    (65, 3.0, 0.1): 1.18282,
                    (65, 3.5, 0.0): 0.949192,
                },
            },
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): 1.208,
                    (65, 3.0, 0.0): 1.269,
                    (65, 3.0, 0.1): 1.539,
                    (65, 3.5, 0.0): 1.322,
                },
            },
            "Pelvis": {
                "U": {
                    (65, 2.5, 0.0): 0.853,
                    (65, 3.0, 0.0): 0.913,
                    (65, 3.0, 0.1): 1.19,
                    (65, 3.5, 0.0): 0.965,
                },
            },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): 1.3860000000000001,
                    (70, 3.0, 0.0): 1.486,
                    (70, 3.0, 0.1): 1.9480000000000002,
                    (70, 3.5, 0.0): 1.574,
                },},
            "Thoracic Spine": {
                "U": {
                    (65, 2.5, 0.0): 1.027,
                    (65, 3.0, 0.0): 1.101,
                    (65, 3.0, 0.1): 1.439,
                    (65, 3.5, 0.0): 1.165,
                },
            },
                "Scoliosis": {
                    "U": {
                        (70, 2.5, 0.0): 1.154,
                        (70, 3.0, 0.0): 1.215,
                        (70, 3.0, 0.1): 1.491,
                        (70, 3.5, 0.0): 1.268,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (65, 2.5, 0.0): 0.766,
                    (65, 3.0, 0.0): 0.805,
                    (65, 3.0, 0.1): 0.972,
                    (65, 3.5, 0.0): 0.838,
                },
            },
            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): 0.38,
                    (70, 3.0, 0.0): 0.399,
                    (70, 3.0, 0.1): 0.48000000000000004,
                    (70, 3.5, 0.0): 0.415,
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (45, 3.0, 0.1): 0.004,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (50, 3.0, 0.1): 0.004,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (55, 3.0, 0.1): 0.005,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (60, 3.0, 0.1): 0.005,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (65, 3.0, 0.1): 0.005,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (70, 3.0, 0.1): 0.006,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                    (80, 3.0, 0.1): 0.006,
                },
            },
        },
        5: {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
                'F': {
                    (70, 2.5, 0.0): 0.537658,
                    (70, 3.0, 0.0): 0.575882,
                    (70, 3.0, 0.1): 0.749317,
                    (70, 3.5, 0.0): 0.609206,
                },
                'M': {
                    (70, 2.5, 0.0): 0.573781,
                    (70, 3.0, 0.0): 0.61145,
                    (70, 3.0, 0.1): 0.779075,
                    (70, 3.5, 0.0): 0.644052,
                },
                'U': {
                    (70, 2.5, 0.0): 0.55572,
                    (70, 3.0, 0.0): 0.593666,
                    (70, 3.0, 0.1): 0.764196,
                    (70, 3.5, 0.0): 0.626629,
                },
            },
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): 0.735,
                    (70, 3.0, 0.0): 0.773,
                    (70, 3.0, 0.1): 0.943,
                    (70, 3.5, 0.0): 0.805,
                },
            },
            "Pelvis": {
                "U": {
                    (70, 2.5, 0.0): 0.42,
                    (70, 3.0, 0.0): 0.449,
                    (70, 3.0, 0.1): 0.586,
                    (70, 3.5, 0.0): 0.474,
                },
            },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): 0.732,
                    (70, 3.0, 0.0): 0.79,
                    (70, 3.0, 0.1): 1.0690000000000002,
                    (70, 3.5, 0.0): 0.842,
                },},
            "Thoracic Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.732,
                    (70, 3.0, 0.0): 0.779,
                    (70, 3.0, 0.1): 0.995,
                    (70, 3.5, 0.0): 0.82,
                },
            },

            "Cervical Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.488,
                    (70, 3.0, 0.0): 0.516,
                    (70, 3.0, 0.1): 0.64,
                    (70, 3.5, 0.0): 0.539,
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.16,
                    (75, 3.0, 0.0): 0.16899999999999998,
                    (75, 3.0, 0.1): 0.216,
                    (75, 3.5, 0.0): 0.178,
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (45, 3.0, 0.1): 0.004,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (50, 3.0, 0.1): 0.004,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (55, 3.0, 0.1): 0.005,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (60, 3.0, 0.1): 0.005,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (65, 3.0, 0.1): 0.005,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (70, 3.0, 0.1): 0.006,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                    (80, 3.0, 0.1): 0.006,
                },
            },
        },
        10: {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
                'F': {
                    (75, 2.5, 0.0): 0.290633,
                    (75, 3.0, 0.0): 0.31455,
                    (75, 3.0, 0.1): 0.426811,
                    (75, 3.5, 0.0): 0.335637,
                },
                'M': {
                    (75, 2.5, 0.0): 0.331349,
                    (75, 3.0, 0.0): 0.355339,
                    (75, 3.0, 0.1): 0.464864,
                    (75, 3.5, 0.0): 0.376275,
                },
                'U': {
                    (75, 2.5, 0.0): 0.310991,
                    (75, 3.0, 0.0): 0.334945,
                    (75, 3.0, 0.1): 0.445838,
                    (75, 3.5, 0.0): 0.355956,
                },
            },
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): 0.465,
                    (70, 3.0, 0.0): 0.492,
                    (70, 3.0, 0.1): 0.619,
                    (70, 3.5, 0.0): 0.516,
                },
            },
            "Pelvis": {
                "U": {
                    (75, 2.5, 0.0): 0.293,
                    (75, 3.0, 0.0): 0.313,
                    (75, 3.0, 0.1): 0.404,
                    (75, 3.5, 0.0): 0.33,
                },
            },
            "Lumbar Spine": {                "U": {
                    (75, 2.5, 0.0): 0.47100000000000003,
                    (75, 3.0, 0.0): 0.508,
                    (75, 3.0, 0.1): 0.69,
                    (75, 3.5, 0.0): 0.5419999999999999,
                },},
            "Thoracic Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.634,
                    (70, 3.0, 0.0): 0.665,
                    (70, 3.0, 0.1): 0.812,
                    (70, 3.5, 0.0): 0.693,
                },
            },

            "Cervical Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.315,
                    (70, 3.0, 0.0): 0.333,
                    (70, 3.0, 0.1): 0.417,
                    (70, 3.5, 0.0): 0.349,
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.08,
                    (75, 3.0, 0.0): 0.08600000000000001,
                    (75, 3.0, 0.1): 0.114,
                    (75, 3.5, 0.0): 0.091,
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (45, 3.0, 0.1): 0.004,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (50, 3.0, 0.1): 0.004,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (55, 3.0, 0.1): 0.005,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (60, 3.0, 0.1): 0.005,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (65, 3.0, 0.1): 0.005,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (70, 3.0, 0.1): 0.006,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                    (80, 3.0, 0.1): 0.006,
                },
            },
        },
        15: {
            "Validation": _VALIDATION_EFF_REGION,
            "Abdomen": {
                'F': {
                    (75, 2.5, 0.0): 0.247187,
                    (75, 3.0, 0.0): 0.267283,
                    (75, 3.0, 0.1): 0.362321,
                    (75, 3.5, 0.0): 0.285033,
                },
                'M': {
                    (75, 2.5, 0.0): 0.253831,
                    (75, 3.0, 0.0): 0.273078,
                    (75, 3.0, 0.1): 0.362393,
                    (75, 3.5, 0.0): 0.289967,
                },
                'U': {
                    (75, 2.5, 0.0): 0.250509,
                    (75, 3.0, 0.0): 0.27018,
                    (75, 3.0, 0.1): 0.362357,
                    (75, 3.5, 0.0): 0.2875,
                },
            },
            "Chest": {
                "U": {
                    (100, 2.5, 0.0): 0.349,
                    (100, 3.0, 0.0): 0.367,
                    (100, 3.0, 0.1): 0.452,
                    (100, 3.5, 0.0): 0.383,
                },
            },
            "Pelvis": {
                "U": {
                    (80, 2.5, 0.0): 0.198,
                    (80, 3.0, 0.0): 0.212,
                    (80, 3.0, 0.1): 0.28,
                    (80, 3.5, 0.0): 0.224,
                },
            },
            "Lumbar Spine": {                "U": {
                    (80, 2.5, 0.0): 0.299,
                    (80, 3.0, 0.0): 0.323,
                    (80, 3.0, 0.1): 0.441,
                    (80, 3.5, 0.0): 0.345,
                },},
            "Thoracic Spine": {
                "U": {
                    (100, 2.5, 0.0): 0.39,
                    (100, 3.0, 0.0): 0.412,
                    (100, 3.0, 0.1): 0.51,
                    (100, 3.5, 0.0): 0.43,
                },
            },

            "Cervical Spine": {
                "U": {
                    (100, 2.5, 0.0): 0.261,
                    (100, 3.0, 0.0): 0.272,
                    (100, 3.0, 0.1): 0.323,
                    (100, 3.5, 0.0): 0.282,
                },
            },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.059,
                    (75, 3.0, 0.0): 0.064,
                    (75, 3.0, 0.1): 0.08600000000000001,
                    (75, 3.5, 0.0): 0.068,
                },},
            "Extremity": {
                "U": {
                    (45, 2.5, 0.0): 0.003,
                    (45, 3.0, 0.0): 0.003,
                    (45, 3.5, 0.0): 0.003,
                    (45, 3.0, 0.1): 0.004,
                    (50, 2.5, 0.0): 0.003,
                    (50, 3.0, 0.0): 0.003,
                    (50, 3.5, 0.0): 0.003,
                    (50, 3.0, 0.1): 0.004,
                    (55, 2.5, 0.0): 0.003,
                    (55, 3.0, 0.0): 0.004,
                    (55, 3.5, 0.0): 0.004,
                    (55, 3.0, 0.1): 0.005,
                    (60, 2.5, 0.0): 0.004,
                    (60, 3.0, 0.0): 0.004,
                    (60, 3.5, 0.0): 0.004,
                    (60, 3.0, 0.1): 0.005,
                    (65, 2.5, 0.0): 0.004,
                    (65, 3.0, 0.0): 0.004,
                    (65, 3.5, 0.0): 0.004,
                    (65, 3.0, 0.1): 0.005,
                    (70, 2.5, 0.0): 0.004,
                    (70, 3.0, 0.0): 0.004,
                    (70, 3.5, 0.0): 0.005,
                    (70, 3.0, 0.1): 0.006,
                    (80, 2.5, 0.0): 0.005,
                    (80, 3.0, 0.0): 0.005,
                    (80, 3.5, 0.0): 0.005,
                    (80, 3.0, 0.1): 0.006,
                },
            },
        },
    },
}


# PA organ coefficients (mGy / Gy·cm²)

coefficients_PA: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.092, "Adrenals": 0.234, "Brain": 0.0, "Breast": 0.039, "Colon": 0.003, "Endosteum": 0.178, "Extrathoracic_Airways": 0.008, "Gallbladder": 0.023, "Heart": 0.053, "Kidneys": 0.207, "Liver": 0.094, "Lungs": 0.227, "Lymph_Nodes": 0.043, "Muscle": 0.052, "Oesophagus": 0.085, "Oral_Mucosa": 0.004, "Ovaries": 0.0, "Pancreas": 0.069, "Prostate": 0.0, "Salivary_Glands": 0.004, "Skin": 0.088, "Small_Intestine": 0.003, "Spleen": 0.213, "Stomach": 0.041, "Testicles": 0.0, "Thymus": 0.026, "Thyroid": 0.031, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.099, "Adrenals": 0.255, "Brain": 0.001, "Breast": 0.044, "Colon": 0.003, "Endosteum": 0.191, "Extrathoracic_Airways": 0.009, "Gallbladder": 0.026, "Heart": 0.059, "Kidneys": 0.224, "Liver": 0.103, "Lungs": 0.247, "Lymph_Nodes": 0.047, "Muscle": 0.056, "Oesophagus": 0.095, "Oral_Mucosa": 0.004, "Ovaries": 0.0, "Pancreas": 0.078, "Prostate": 0.0, "Salivary_Glands": 0.005, "Skin": 0.09, "Small_Intestine": 0.004, "Spleen": 0.234, "Stomach": 0.046, "Testicles": 0.0, "Thymus": 0.03, "Thyroid": 0.035, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.105, "Adrenals": 0.274, "Brain": 0.001, "Breast": 0.048, "Colon": 0.003, "Endosteum": 0.202, "Extrathoracic_Airways": 0.01, "Gallbladder": 0.029, "Heart": 0.065, "Kidneys": 0.239, "Liver": 0.112, "Lungs": 0.265, "Lymph_Nodes": 0.051, "Muscle": 0.059, "Oesophagus": 0.105, "Oral_Mucosa": 0.005, "Ovaries": 0.0, "Pancreas": 0.085, "Prostate": 0.0, "Salivary_Glands": 0.005, "Skin": 0.092, "Small_Intestine": 0.004, "Spleen": 0.252, "Stomach": 0.05, "Testicles": 0.0, "Thymus": 0.033, "Thyroid": 0.038, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.126, "Adrenals": 0.327, "Brain": 0.001, "Breast": 0.066, "Colon": 0.006, "Endosteum": 0.222, "Extrathoracic_Airways": 0.016, "Gallbladder": 0.051, "Heart": 0.098, "Kidneys": 0.269, "Liver": 0.143, "Lungs": 0.309, "Lymph_Nodes": 0.07, "Muscle": 0.07, "Oesophagus": 0.149, "Oral_Mucosa": 0.006, "Ovaries": 0.0, "Pancreas": 0.126, "Prostate": 0.0, "Salivary_Glands": 0.008, "Skin": 0.096, "Small_Intestine": 0.008, "Spleen": 0.305, "Stomach": 0.071, "Testicles": 0.0, "Thymus": 0.05, "Thyroid": 0.052, "Urinary_Bladder": 0.0, "Uterus": 0.001},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.135, "Adrenals": 0.352, "Brain": 0.001, "Breast": 0.073, "Colon": 0.007, "Endosteum": 0.235, "Extrathoracic_Airways": 0.018, "Gallbladder": 0.057, "Heart": 0.108, "Kidneys": 0.288, "Liver": 0.155, "Lungs": 0.332, "Lymph_Nodes": 0.076, "Muscle": 0.075, "Oesophagus": 0.164, "Oral_Mucosa": 0.007, "Ovaries": 0.0, "Pancreas": 0.138, "Prostate": 0.0, "Salivary_Glands": 0.009, "Skin": 0.098, "Small_Intestine": 0.008, "Spleen": 0.33, "Stomach": 0.078, "Testicles": 0.0, "Thymus": 0.055, "Thyroid": 0.057, "Urinary_Bladder": 0.0, "Uterus": 0.002},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.143, "Adrenals": 0.375, "Brain": 0.001, "Breast": 0.078, "Colon": 0.007, "Endosteum": 0.247, "Extrathoracic_Airways": 0.019, "Gallbladder": 0.062, "Heart": 0.117, "Kidneys": 0.304, "Liver": 0.166, "Lungs": 0.353, "Lymph_Nodes": 0.082, "Muscle": 0.079, "Oesophagus": 0.178, "Oral_Mucosa": 0.008, "Ovaries": 0.0, "Pancreas": 0.15, "Prostate": 0.0, "Salivary_Glands": 0.01, "Skin": 0.1, "Small_Intestine": 0.009, "Spleen": 0.352, "Stomach": 0.084, "Testicles": 0.0, "Thymus": 0.06, "Thyroid": 0.062, "Urinary_Bladder": 0.0, "Uterus": 0.002},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.156, "Adrenals": 0.412, "Brain": 0.002, "Breast": 0.091, "Colon": 0.009, "Endosteum": 0.251, "Extrathoracic_Airways": 0.024, "Gallbladder": 0.078, "Heart": 0.141, "Kidneys": 0.321, "Liver": 0.187, "Lungs": 0.377, "Lymph_Nodes": 0.095, "Muscle": 0.086, "Oesophagus": 0.21, "Oral_Mucosa": 0.009, "Ovaries": 0.0, "Pancreas": 0.178, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.103, "Small_Intestine": 0.012, "Spleen": 0.379, "Stomach": 0.1, "Testicles": 0.0, "Thymus": 0.073, "Thyroid": 0.077, "Urinary_Bladder": 0.0, "Uterus": 0.002},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.166, "Adrenals": 0.439, "Brain": 0.002, "Breast": 0.099, "Colon": 0.01, "Endosteum": 0.264, "Extrathoracic_Airways": 0.026, "Gallbladder": 0.084, "Heart": 0.152, "Kidneys": 0.339, "Liver": 0.2, "Lungs": 0.4, "Lymph_Nodes": 0.102, "Muscle": 0.091, "Oesophagus": 0.227, "Oral_Mucosa": 0.01, "Ovaries": 0.0, "Pancreas": 0.193, "Prostate": 0.0, "Salivary_Glands": 0.014, "Skin": 0.106, "Small_Intestine": 0.014, "Spleen": 0.405, "Stomach": 0.108, "Testicles": 0.0, "Thymus": 0.08, "Thyroid": 0.084, "Urinary_Bladder": 0.0, "Uterus": 0.002},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.174, "Adrenals": 0.462, "Brain": 0.002, "Breast": 0.105, "Colon": 0.011, "Endosteum": 0.274, "Extrathoracic_Airways": 0.028, "Gallbladder": 0.091, "Heart": 0.163, "Kidneys": 0.355, "Liver": 0.212, "Lungs": 0.421, "Lymph_Nodes": 0.108, "Muscle": 0.095, "Oesophagus": 0.242, "Oral_Mucosa": 0.01, "Ovaries": 0.0, "Pancreas": 0.206, "Prostate": 0.0, "Salivary_Glands": 0.015, "Skin": 0.108, "Small_Intestine": 0.015, "Spleen": 0.427, "Stomach": 0.115, "Testicles": 0.0, "Thymus": 0.085, "Thyroid": 0.089, "Urinary_Bladder": 0.0, "Uterus": 0.002},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.181, "Adrenals": 0.477, "Brain": 0.002, "Breast": 0.112, "Colon": 0.012, "Endosteum": 0.269, "Extrathoracic_Airways": 0.031, "Gallbladder": 0.098, "Heart": 0.177, "Kidneys": 0.36, "Liver": 0.222, "Lungs": 0.43, "Lymph_Nodes": 0.115, "Muscle": 0.098, "Oesophagus": 0.26, "Oral_Mucosa": 0.011, "Ovaries": 0.0, "Pancreas": 0.219, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.109, "Small_Intestine": 0.016, "Spleen": 0.44, "Stomach": 0.125, "Testicles": 0.0, "Thymus": 0.093, "Thyroid": 0.098, "Urinary_Bladder": 0.001, "Uterus": 0.003},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.19, "Adrenals": 0.502, "Brain": 0.003, "Breast": 0.119, "Colon": 0.013, "Endosteum": 0.28, "Extrathoracic_Airways": 0.033, "Gallbladder": 0.105, "Heart": 0.189, "Kidneys": 0.376, "Liver": 0.235, "Lungs": 0.452, "Lymph_Nodes": 0.122, "Muscle": 0.103, "Oesophagus": 0.277, "Oral_Mucosa": 0.011, "Ovaries": 0.0, "Pancreas": 0.234, "Prostate": 0.0, "Salivary_Glands": 0.018, "Skin": 0.111, "Small_Intestine": 0.017, "Spleen": 0.464, "Stomach": 0.133, "Testicles": 0.0, "Thymus": 0.099, "Thyroid": 0.104, "Urinary_Bladder": 0.001, "Uterus": 0.003},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.198, "Adrenals": 0.524, "Brain": 0.003, "Breast": 0.126, "Colon": 0.014, "Endosteum": 0.289, "Extrathoracic_Airways": 0.035, "Gallbladder": 0.111, "Heart": 0.199, "Kidneys": 0.39, "Liver": 0.246, "Lungs": 0.471, "Lymph_Nodes": 0.128, "Muscle": 0.107, "Oesophagus": 0.292, "Oral_Mucosa": 0.012, "Ovaries": 0.0, "Pancreas": 0.247, "Prostate": 0.0, "Salivary_Glands": 0.019, "Skin": 0.113, "Small_Intestine": 0.019, "Spleen": 0.485, "Stomach": 0.141, "Testicles": 0.0, "Thymus": 0.105, "Thyroid": 0.11, "Urinary_Bladder": 0.001, "Uterus": 0.003},
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.087, "Adrenals": 0.049, "Brain": 0.0, "Breast": 0.511, "Colon": 0.002, "Endosteum": 0.157, "Extrathoracic_Airways": 0.022, "Gallbladder": 0.036, "Gonads": 0.0, "Heart": 0.688, "Kidneys": 0.015, "Liver": 0.191, "Lungs": 0.299, "Lymph_Nodes": 0.094, "Muscle": 0.076, "Oesophagus": 0.127, "Oral_Mucosa": 0.003, "Pancreas": 0.139, "Prostate": 0.0, "Salivary_Glands": 0.005, "Skin": 0.106, "Small_Intestine": 0.002, "Spleen": 0.05, "Stomach": 0.178, "Thymus": 1.343, "Thyroid": 0.036, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.093, "Adrenals": 0.056, "Brain": 0.001, "Breast": 0.537, "Colon": 0.002, "Endosteum": 0.167, "Extrathoracic_Airways": 0.024, "Gallbladder": 0.041, "Gonads": 0.0, "Heart": 0.748, "Kidneys": 0.017, "Liver": 0.209, "Lungs": 0.328, "Lymph_Nodes": 0.103, "Muscle": 0.082, "Oesophagus": 0.142, "Oral_Mucosa": 0.003, "Pancreas": 0.155, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.108, "Small_Intestine": 0.003, "Spleen": 0.057, "Stomach": 0.195, "Thymus": 1.45, "Thyroid": 0.04, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.098, "Adrenals": 0.062, "Brain": 0.001, "Breast": 0.56, "Colon": 0.002, "Endosteum": 0.175, "Extrathoracic_Airways": 0.026, "Gallbladder": 0.045, "Gonads": 0.0, "Heart": 0.802, "Kidneys": 0.019, "Liver": 0.226, "Lungs": 0.354, "Lymph_Nodes": 0.112, "Muscle": 0.088, "Oesophagus": 0.156, "Oral_Mucosa": 0.004, "Pancreas": 0.169, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.11, "Small_Intestine": 0.003, "Spleen": 0.062, "Stomach": 0.211, "Thymus": 1.543, "Thyroid": 0.044, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.113, "Adrenals": 0.098, "Brain": 0.001, "Breast": 0.596, "Colon": 0.004, "Endosteum": 0.191, "Extrathoracic_Airways": 0.036, "Gallbladder": 0.071, "Gonads": 0.0, "Heart": 0.935, "Kidneys": 0.034, "Liver": 0.276, "Lungs": 0.429, "Lymph_Nodes": 0.143, "Muscle": 0.104, "Oesophagus": 0.23, "Oral_Mucosa": 0.006, "Pancreas": 0.233, "Prostate": 0.0, "Salivary_Glands": 0.01, "Skin": 0.116, "Small_Intestine": 0.005, "Spleen": 0.089, "Stomach": 0.255, "Thymus": 1.675, "Thyroid": 0.067, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.12, "Adrenals": 0.109, "Brain": 0.001, "Breast": 0.623, "Colon": 0.005, "Endosteum": 0.201, "Extrathoracic_Airways": 0.039, "Gallbladder": 0.078, "Gonads": 0.0, "Heart": 1.005, "Kidneys": 0.038, "Liver": 0.299, "Lungs": 0.464, "Lymph_Nodes": 0.155, "Muscle": 0.111, "Oesophagus": 0.252, "Oral_Mucosa": 0.007, "Pancreas": 0.255, "Prostate": 0.0, "Salivary_Glands": 0.012, "Skin": 0.119, "Small_Intestine": 0.006, "Spleen": 0.097, "Stomach": 0.277, "Thymus": 1.785, "Thyroid": 0.074, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.126, "Adrenals": 0.118, "Brain": 0.001, "Breast": 0.647, "Colon": 0.005, "Endosteum": 0.21, "Extrathoracic_Airways": 0.042, "Gallbladder": 0.085, "Gonads": 0.0, "Heart": 1.067, "Kidneys": 0.041, "Liver": 0.319, "Lungs": 0.496, "Lymph_Nodes": 0.166, "Muscle": 0.118, "Oesophagus": 0.273, "Oral_Mucosa": 0.008, "Pancreas": 0.276, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.122, "Small_Intestine": 0.007, "Spleen": 0.106, "Stomach": 0.295, "Thymus": 1.879, "Thyroid": 0.081, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.138, "Adrenals": 0.14, "Brain": 0.001, "Breast": 0.662, "Colon": 0.007, "Endosteum": 0.214, "Extrathoracic_Airways": 0.046, "Gallbladder": 0.101, "Gonads": 0.0, "Heart": 1.141, "Kidneys": 0.053, "Liver": 0.35, "Lungs": 0.538, "Lymph_Nodes": 0.186, "Muscle": 0.128, "Oesophagus": 0.325, "Oral_Mucosa": 0.009, "Pancreas": 0.32, "Prostate": 0.0, "Salivary_Glands": 0.016, "Skin": 0.125, "Small_Intestine": 0.009, "Spleen": 0.125, "Stomach": 0.324, "Thymus": 1.943, "Thyroid": 0.098, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.145, "Adrenals": 0.152, "Brain": 0.002, "Breast": 0.688, "Colon": 0.008, "Endosteum": 0.224, "Extrathoracic_Airways": 0.05, "Gallbladder": 0.109, "Gonads": 0.0, "Heart": 1.211, "Kidneys": 0.058, "Liver": 0.374, "Lungs": 0.574, "Lymph_Nodes": 0.2, "Muscle": 0.136, "Oesophagus": 0.352, "Oral_Mucosa": 0.01, "Pancreas": 0.345, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.128, "Small_Intestine": 0.01, "Spleen": 0.135, "Stomach": 0.346, "Thymus": 2.047, "Thyroid": 0.106, "Urinary_Bladder": 0.001, "Uterus": 0.0005},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.151, "Adrenals": 0.163, "Brain": 0.002, "Breast": 0.71, "Colon": 0.009, "Endosteum": 0.232, "Extrathoracic_Airways": 0.053, "Gallbladder": 0.117, "Gonads": 0.0, "Heart": 1.273, "Kidneys": 0.062, "Liver": 0.394, "Lungs": 0.606, "Lymph_Nodes": 0.211, "Muscle": 0.143, "Oesophagus": 0.375, "Oral_Mucosa": 0.011, "Pancreas": 0.368, "Prostate": 0.0, "Salivary_Glands": 0.019, "Skin": 0.131, "Small_Intestine": 0.011, "Spleen": 0.145, "Stomach": 0.365, "Thymus": 2.135, "Thyroid": 0.114, "Urinary_Bladder": 0.001, "Uterus": 0.0005},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.157, "Adrenals": 0.177, "Brain": 0.002, "Breast": 0.708, "Colon": 0.01, "Endosteum": 0.229, "Extrathoracic_Airways": 0.055, "Gallbladder": 0.126, "Gonads": 0.0005, "Heart": 1.299, "Kidneys": 0.07, "Liver": 0.408, "Lungs": 0.624, "Lymph_Nodes": 0.221, "Muscle": 0.147, "Oesophagus": 0.403, "Oral_Mucosa": 0.012, "Pancreas": 0.391, "Prostate": 0.0, "Salivary_Glands": 0.02, "Skin": 0.133, "Small_Intestine": 0.012, "Spleen": 0.157, "Stomach": 0.375, "Thymus": 2.149, "Thyroid": 0.124, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.164, "Adrenals": 0.192, "Brain": 0.002, "Breast": 0.73, "Colon": 0.011, "Endosteum": 0.238, "Extrathoracic_Airways": 0.058, "Gallbladder": 0.134, "Gonads": 0.0005, "Heart": 1.364, "Kidneys": 0.075, "Liver": 0.431, "Lungs": 0.658, "Lymph_Nodes": 0.234, "Muscle": 0.154, "Oesophagus": 0.427, "Oral_Mucosa": 0.013, "Pancreas": 0.417, "Prostate": 0.0, "Salivary_Glands": 0.02, "Skin": 0.135, "Small_Intestine": 0.013, "Spleen": 0.168, "Stomach": 0.395, "Thymus": 2.243, "Thyroid": 0.133, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.171, "Adrenals": 0.203, "Brain": 0.002, "Breast": 0.749, "Colon": 0.012, "Endosteum": 0.245, "Extrathoracic_Airways": 0.061, "Gallbladder": 0.142, "Gonads": 0.0005, "Heart": 1.42, "Kidneys": 0.08, "Liver": 0.451, "Lungs": 0.687, "Lymph_Nodes": 0.245, "Muscle": 0.161, "Oesophagus": 0.45, "Oral_Mucosa": 0.014, "Pancreas": 0.439, "Prostate": 0.0, "Salivary_Glands": 0.022, "Skin": 0.138, "Small_Intestine": 0.014, "Spleen": 0.177, "Stomach": 0.413, "Thymus": 2.321, "Thyroid": 0.14, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                },
            },
            "Scoliosis": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.173, "Brain": 0.001, "Breast": 0.027, "Colon": 0.031, "Endosteum": 0.176, "Extrathoracic_Airways": 0.024, "Gallbladder": 0.045, "Heart": 0.045, "Kidneys": 0.298, "Liver": 0.083, "Lungs": 0.176, "Lymph_Nodes": 0.058, "Muscle": 0.063, "Oesophagus": 0.064, "Oral_Mucosa": 0.012, "Ovaries": 0.017, "Pancreas": 0.064, "Prostate": 0.001, "Salivary_Glands": 0.05, "Skin": 0.099, "Small_Intestine": 0.053, "Spleen": 0.192, "Stomach": 0.042, "Testicles": 0.0, "Thymus": 0.022, "Thyroid": 0.027, "Urinary_Bladder": 0.002, "Uterus": 0.013},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.116, "Adrenals": 0.191, "Brain": 0.001, "Breast": 0.031, "Colon": 0.035, "Endosteum": 0.19, "Extrathoracic_Airways": 0.026, "Gallbladder": 0.05, "Heart": 0.05, "Kidneys": 0.321, "Liver": 0.091, "Lungs": 0.193, "Lymph_Nodes": 0.064, "Muscle": 0.067, "Oesophagus": 0.072, "Oral_Mucosa": 0.013, "Ovaries": 0.02, "Pancreas": 0.072, "Prostate": 0.001, "Salivary_Glands": 0.054, "Skin": 0.101, "Small_Intestine": 0.059, "Spleen": 0.211, "Stomach": 0.047, "Testicles": 0.0, "Thymus": 0.025, "Thyroid": 0.03, "Urinary_Bladder": 0.002, "Uterus": 0.015},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.124, "Adrenals": 0.207, "Brain": 0.001, "Breast": 0.034, "Colon": 0.038, "Endosteum": 0.202, "Extrathoracic_Airways": 0.029, "Gallbladder": 0.055, "Heart": 0.055, "Kidneys": 0.342, "Liver": 0.099, "Lungs": 0.207, "Lymph_Nodes": 0.07, "Muscle": 0.071, "Oesophagus": 0.079, "Oral_Mucosa": 0.014, "Ovaries": 0.022, "Pancreas": 0.079, "Prostate": 0.001, "Salivary_Glands": 0.058, "Skin": 0.102, "Small_Intestine": 0.065, "Spleen": 0.227, "Stomach": 0.052, "Testicles": 0.0, "Thymus": 0.027, "Thyroid": 0.033, "Urinary_Bladder": 0.002, "Uterus": 0.017},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.15, "Adrenals": 0.246, "Brain": 0.002, "Breast": 0.047, "Colon": 0.055, "Endosteum": 0.223, "Extrathoracic_Airways": 0.04, "Gallbladder": 0.078, "Heart": 0.079, "Kidneys": 0.383, "Liver": 0.129, "Lungs": 0.242, "Lymph_Nodes": 0.095, "Muscle": 0.083, "Oesophagus": 0.116, "Oral_Mucosa": 0.021, "Ovaries": 0.044, "Pancreas": 0.12, "Prostate": 0.003, "Salivary_Glands": 0.068, "Skin": 0.107, "Small_Intestine": 0.092, "Spleen": 0.267, "Stomach": 0.075, "Testicles": 0.001, "Thymus": 0.046, "Thyroid": 0.058, "Urinary_Bladder": 0.005, "Uterus": 0.027},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.161, "Adrenals": 0.266, "Brain": 0.002, "Breast": 0.051, "Colon": 0.061, "Endosteum": 0.238, "Extrathoracic_Airways": 0.044, "Gallbladder": 0.086, "Heart": 0.087, "Kidneys": 0.408, "Liver": 0.14, "Lungs": 0.261, "Lymph_Nodes": 0.103, "Muscle": 0.089, "Oesophagus": 0.127, "Oral_Mucosa": 0.023, "Ovaries": 0.049, "Pancreas": 0.132, "Prostate": 0.003, "Salivary_Glands": 0.073, "Skin": 0.109, "Small_Intestine": 0.101, "Spleen": 0.288, "Stomach": 0.082, "Testicles": 0.001, "Thymus": 0.051, "Thyroid": 0.064, "Urinary_Bladder": 0.006, "Uterus": 0.03},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.171, "Adrenals": 0.285, "Brain": 0.003, "Breast": 0.055, "Colon": 0.066, "Endosteum": 0.251, "Extrathoracic_Airways": 0.048, "Gallbladder": 0.093, "Heart": 0.094, "Kidneys": 0.43, "Liver": 0.151, "Lungs": 0.277, "Lymph_Nodes": 0.111, "Muscle": 0.094, "Oesophagus": 0.138, "Oral_Mucosa": 0.025, "Ovaries": 0.054, "Pancreas": 0.143, "Prostate": 0.003, "Salivary_Glands": 0.077, "Skin": 0.111, "Small_Intestine": 0.109, "Spleen": 0.307, "Stomach": 0.089, "Testicles": 0.001, "Thymus": 0.056, "Thyroid": 0.07, "Urinary_Bladder": 0.006, "Uterus": 0.033},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.188, "Adrenals": 0.307, "Brain": 0.003, "Breast": 0.066, "Colon": 0.079, "Endosteum": 0.256, "Extrathoracic_Airways": 0.057, "Gallbladder": 0.111, "Heart": 0.112, "Kidneys": 0.452, "Liver": 0.17, "Lungs": 0.296, "Lymph_Nodes": 0.129, "Muscle": 0.102, "Oesophagus": 0.161, "Oral_Mucosa": 0.031, "Ovaries": 0.073, "Pancreas": 0.17, "Prostate": 0.004, "Salivary_Glands": 0.083, "Skin": 0.114, "Small_Intestine": 0.13, "Spleen": 0.327, "Stomach": 0.104, "Testicles": 0.001, "Thymus": 0.067, "Thyroid": 0.089, "Urinary_Bladder": 0.009, "Uterus": 0.041},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.2, "Adrenals": 0.328, "Brain": 0.004, "Breast": 0.071, "Colon": 0.085, "Endosteum": 0.27, "Extrathoracic_Airways": 0.061, "Gallbladder": 0.12, "Heart": 0.121, "Kidneys": 0.477, "Liver": 0.183, "Lungs": 0.315, "Lymph_Nodes": 0.138, "Muscle": 0.108, "Oesophagus": 0.174, "Oral_Mucosa": 0.033, "Ovaries": 0.079, "Pancreas": 0.184, "Prostate": 0.005, "Salivary_Glands": 0.088, "Skin": 0.117, "Small_Intestine": 0.14, "Spleen": 0.348, "Stomach": 0.113, "Testicles": 0.002, "Thymus": 0.073, "Thyroid": 0.096, "Urinary_Bladder": 0.01, "Uterus": 0.044},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.21, "Adrenals": 0.347, "Brain": 0.004, "Breast": 0.076, "Colon": 0.091, "Endosteum": 0.281, "Extrathoracic_Airways": 0.065, "Gallbladder": 0.128, "Heart": 0.129, "Kidneys": 0.499, "Liver": 0.194, "Lungs": 0.331, "Lymph_Nodes": 0.147, "Muscle": 0.113, "Oesophagus": 0.186, "Oral_Mucosa": 0.035, "Ovaries": 0.085, "Pancreas": 0.197, "Prostate": 0.005, "Salivary_Glands": 0.092, "Skin": 0.119, "Small_Intestine": 0.15, "Spleen": 0.367, "Stomach": 0.121, "Testicles": 0.002, "Thymus": 0.078, "Thyroid": 0.103, "Urinary_Bladder": 0.011, "Uterus": 0.048},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.219, "Adrenals": 0.363, "Brain": 0.005, "Breast": 0.082, "Colon": 0.098, "Endosteum": 0.277, "Extrathoracic_Airways": 0.071, "Gallbladder": 0.14, "Heart": 0.139, "Kidneys": 0.508, "Liver": 0.204, "Lungs": 0.338, "Lymph_Nodes": 0.156, "Muscle": 0.116, "Oesophagus": 0.199, "Oral_Mucosa": 0.037, "Ovaries": 0.092, "Pancreas": 0.209, "Prostate": 0.005, "Salivary_Glands": 0.096, "Skin": 0.12, "Small_Intestine": 0.16, "Spleen": 0.375, "Stomach": 0.129, "Testicles": 0.002, "Thymus": 0.087, "Thyroid": 0.114, "Urinary_Bladder": 0.013, "Uterus": 0.052},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.231, "Adrenals": 0.384, "Brain": 0.005, "Breast": 0.088, "Colon": 0.105, "Endosteum": 0.288, "Extrathoracic_Airways": 0.076, "Gallbladder": 0.149, "Heart": 0.149, "Kidneys": 0.531, "Liver": 0.216, "Lungs": 0.356, "Lymph_Nodes": 0.166, "Muscle": 0.122, "Oesophagus": 0.212, "Oral_Mucosa": 0.04, "Ovaries": 0.098, "Pancreas": 0.223, "Prostate": 0.006, "Salivary_Glands": 0.101, "Skin": 0.122, "Small_Intestine": 0.171, "Spleen": 0.395, "Stomach": 0.138, "Testicles": 0.002, "Thymus": 0.093, "Thyroid": 0.122, "Urinary_Bladder": 0.014, "Uterus": 0.056},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.241, "Adrenals": 0.402, "Brain": 0.005, "Breast": 0.093, "Colon": 0.111, "Endosteum": 0.298, "Extrathoracic_Airways": 0.08, "Gallbladder": 0.157, "Heart": 0.157, "Kidneys": 0.551, "Liver": 0.227, "Lungs": 0.371, "Lymph_Nodes": 0.175, "Muscle": 0.127, "Oesophagus": 0.224, "Oral_Mucosa": 0.042, "Ovaries": 0.104, "Pancreas": 0.236, "Prostate": 0.006, "Salivary_Glands": 0.105, "Skin": 0.124, "Small_Intestine": 0.18, "Spleen": 0.412, "Stomach": 0.145, "Testicles": 0.003, "Thymus": 0.098, "Thyroid": 0.129, "Urinary_Bladder": 0.015, "Uterus": 0.059},
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.052, "Adrenals": 0.0, "Brain": 0.392, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.269, "Extrathoracic_Airways": 0.094, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.001, "Lymph_Nodes": 0.096, "Muscle": 0.025, "Oesophagus": 0.001, "Oral_Mucosa": 0.266, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.989, "Skin": 0.123, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.017, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.057, "Adrenals": 0.0, "Brain": 0.435, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.291, "Extrathoracic_Airways": 0.105, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.002, "Lymph_Nodes": 0.102, "Muscle": 0.026, "Oesophagus": 0.001, "Oral_Mucosa": 0.292, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.052, "Skin": 0.125, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.019, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.061, "Adrenals": 0.0, "Brain": 0.474, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.311, "Extrathoracic_Airways": 0.116, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.002, "Lymph_Nodes": 0.108, "Muscle": 0.027, "Oesophagus": 0.001, "Oral_Mucosa": 0.315, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.106, "Skin": 0.128, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.021, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.075, "Adrenals": 0.0, "Brain": 0.619, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.35, "Extrathoracic_Airways": 0.176, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.127, "Muscle": 0.03, "Oesophagus": 0.001, "Oral_Mucosa": 0.418, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.218, "Skin": 0.133, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.005, "Thyroid": 0.037, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.081, "Adrenals": 0.0, "Brain": 0.676, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.375, "Extrathoracic_Airways": 0.194, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.135, "Muscle": 0.032, "Oesophagus": 0.002, "Oral_Mucosa": 0.453, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.287, "Skin": 0.136, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.005, "Thyroid": 0.041, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.086, "Adrenals": 0.0, "Brain": 0.727, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.397, "Extrathoracic_Airways": 0.21, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.142, "Muscle": 0.033, "Oesophagus": 0.002, "Oral_Mucosa": 0.486, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.347, "Skin": 0.138, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.006, "Thyroid": 0.044, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.095, "Adrenals": 0.0, "Brain": 0.824, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.408, "Extrathoracic_Airways": 0.26, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.004, "Lymph_Nodes": 0.155, "Muscle": 0.035, "Oesophagus": 0.003, "Oral_Mucosa": 0.552, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.414, "Skin": 0.142, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.007, "Thyroid": 0.056, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.101, "Adrenals": 0.0, "Brain": 0.885, "Breast": 0.0, "Colon": 0.0, "Endosteum": 0.432, "Extrathoracic_Airways": 0.282, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.004, "Lymph_Nodes": 0.164, "Muscle": 0.036, "Oesophagus": 0.003, "Oral_Mucosa": 0.592, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.482, "Skin": 0.145, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.008, "Thyroid": 0.061, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.0, "Brain": 0.94, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.452, "Extrathoracic_Airways": 0.302, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.005, "Lymph_Nodes": 0.172, "Muscle": 0.038, "Oesophagus": 0.003, "Oral_Mucosa": 0.627, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.54, "Skin": 0.147, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.008, "Thyroid": 0.065, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.111, "Adrenals": 0.0, "Brain": 0.991, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.446, "Extrathoracic_Airways": 0.33, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.005, "Lymph_Nodes": 0.178, "Muscle": 0.038, "Oesophagus": 0.004, "Oral_Mucosa": 0.667, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.565, "Skin": 0.149, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.008, "Thyroid": 0.071, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.117, "Adrenals": 0.0, "Brain": 1.052, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.466, "Extrathoracic_Airways": 0.353, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.006, "Lymph_Nodes": 0.186, "Muscle": 0.04, "Oesophagus": 0.004, "Oral_Mucosa": 0.706, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.628, "Skin": 0.152, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.009, "Thyroid": 0.076, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.122, "Adrenals": 0.0, "Brain": 1.105, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.483, "Extrathoracic_Airways": 0.373, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.006, "Lymph_Nodes": 0.194, "Muscle": 0.041, "Oesophagus": 0.004, "Oral_Mucosa": 0.741, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.682, "Skin": 0.154, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.01, "Thyroid": 0.08, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 1.302, "Adrenals": 4.784, "Brain": 0.035, "Breast": 3.486, "Colon": 0.171, "Endosteum": 5.028, "Extrathoracic_Airways": 0.202, "Gallbladder": 1.047, "Heart": 2.175, "Kidneys": 3.294, "Liver": 2.092, "Lungs": 4.897, "Lymph_Nodes": 1.266, "Muscle": 1.272, "Oesophagus": 2.423, "Oral_Mucosa": 0.15, "Ovaries": 0.172, "Pancreas": 2.528, "Prostate": 0.056, "Salivary_Glands": 0.231, "Skin": 1.325, "Small_Intestine": 0.214, "Spleen": 4.414, "Stomach": 1.363, "Testicles": 0.0, "Thymus": 0.995, "Thyroid": 0.758, "Urinary_Bladder": 0.052, "Uterus": 0.087},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 1.384, "Adrenals": 5.073, "Brain": 0.038, "Breast": 3.695, "Colon": 0.189, "Endosteum": 5.316, "Extrathoracic_Airways": 0.222, "Gallbladder": 1.157, "Heart": 2.365, "Kidneys": 3.473, "Liver": 2.257, "Lungs": 5.192, "Lymph_Nodes": 1.366, "Muscle": 1.348, "Oesophagus": 2.639, "Oral_Mucosa": 0.165, "Ovaries": 0.189, "Pancreas": 2.733, "Prostate": 0.064, "Salivary_Glands": 0.254, "Skin": 1.36, "Small_Intestine": 0.237, "Spleen": 4.708, "Stomach": 1.489, "Testicles": 0.0, "Thymus": 1.09, "Thyroid": 0.819, "Urinary_Bladder": 0.058, "Uterus": 0.098},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 1.753, "Adrenals": 6.314, "Brain": 0.058, "Breast": 4.3, "Colon": 0.286, "Endosteum": 6.537, "Extrathoracic_Airways": 0.323, "Gallbladder": 1.734, "Heart": 3.271, "Kidneys": 4.228, "Liver": 3.029, "Lungs": 6.441, "Lymph_Nodes": 1.836, "Muscle": 1.691, "Oesophagus": 3.671, "Oral_Mucosa": 0.242, "Ovaries": 0.271, "Pancreas": 3.669, "Prostate": 0.113, "Salivary_Glands": 0.378, "Skin": 1.521, "Small_Intestine": 0.359, "Spleen": 5.995, "Stomach": 2.106, "Testicles": 0.0, "Thymus": 1.565, "Thyroid": 1.1, "Urinary_Bladder": 0.085, "Uterus": 0.162},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 1.455, "Adrenals": 5.32, "Brain": 0.042, "Breast": 3.859, "Colon": 0.206, "Endosteum": 5.562, "Extrathoracic_Airways": 0.239, "Gallbladder": 1.257, "Heart": 2.534, "Kidneys": 3.624, "Liver": 2.402, "Lungs": 5.443, "Lymph_Nodes": 1.454, "Muscle": 1.414, "Oesophagus": 2.83, "Oral_Mucosa": 0.179, "Ovaries": 0.204, "Pancreas": 2.911, "Prostate": 0.072, "Salivary_Glands": 0.276, "Skin": 1.39, "Small_Intestine": 0.258, "Spleen": 4.962, "Stomach": 1.602, "Testicles": 0.0, "Thymus": 1.176, "Thyroid": 0.872, "Urinary_Bladder": 0.062, "Uterus": 0.108},
                },
            },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 1.3860000000000001, 'Adrenals': 8.927, 'Brain': 0.044, 'Breast': 0.6539999999999999, 'Colon': 0.213, 'Endosteum': 4.718, 'Extrathoracic_Airways': 0.18000000000000002, 'Gallbladder': 1.739, 'Heart': 3.7239999999999998, 'Kidneys': 4.866, 'Liver': 1.89, 'Lungs': 5.167, 'Lymph_Nodes': 1.792, 'Muscle': 1.468, 'Oesophagus': 4.6080000000000005, 'Oral_Mucosa': 0.184, 'Ovaries': 0.077, 'Pancreas': 4.153, 'Prostate': 0.012, 'Salivary_Glands': 0.255, 'Skin': 1.203, 'Small_Intestine': 0.275, 'Spleen': 2.1470000000000002, 'Stomach': 1.676, 'Testicles': 0.0, 'Thymus': 1.56, 'Thyroid': 1.0579999999999998, 'Urinary_Bladder': 0.034999999999999996, 'Uterus': 0.082},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 1.478, 'Adrenals': 9.445, 'Brain': 0.049, 'Breast': 0.726, 'Colon': 0.235, 'Endosteum': 5.0, 'Extrathoracic_Airways': 0.201, 'Gallbladder': 1.9200000000000002, 'Heart': 4.054, 'Kidneys': 5.12, 'Liver': 2.058, 'Lungs': 5.492999999999999, 'Lymph_Nodes': 1.9480000000000002, 'Muscle': 1.5590000000000002, 'Oesophagus': 5.0489999999999995, 'Oral_Mucosa': 0.204, 'Ovaries': 0.087, 'Pancreas': 4.5360000000000005, 'Prostate': 0.014, 'Salivary_Glands': 0.27999999999999997, 'Skin': 1.232, 'Small_Intestine': 0.304, 'Spleen': 2.312, 'Stomach': 1.828, 'Testicles': 0.0, 'Thymus': 1.719, 'Thyroid': 1.158, 'Urinary_Bladder': 0.04, 'Uterus': 0.092},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 1.5579999999999998, 'Adrenals': 9.883, 'Brain': 0.053, 'Breast': 0.791, 'Colon': 0.256, 'Endosteum': 5.242, 'Extrathoracic_Airways': 0.22, 'Gallbladder': 2.084, 'Heart': 4.347, 'Kidneys': 5.333, 'Liver': 2.2079999999999997, 'Lungs': 5.772, 'Lymph_Nodes': 2.0869999999999997, 'Muscle': 1.638, 'Oesophagus': 5.446000000000001, 'Oral_Mucosa': 0.223, 'Ovaries': 0.098, 'Pancreas': 4.879, 'Prostate': 0.016, 'Salivary_Glands': 0.303, 'Skin': 1.2570000000000001, 'Small_Intestine': 0.331, 'Spleen': 2.457, 'Stomach': 1.9629999999999999, 'Testicles': 0.0, 'Thymus': 1.863, 'Thyroid': 1.248, 'Urinary_Bladder': 0.045000000000000005, 'Uterus': 0.1},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 1.896, 'Adrenals': 11.586, 'Brain': 0.075, 'Breast': 1.0759999999999998, 'Colon': 0.352, 'Endosteum': 6.2170000000000005, 'Extrathoracic_Airways': 0.317, 'Gallbladder': 2.856, 'Heart': 5.64, 'Kidneys': 6.155, 'Liver': 2.886, 'Lungs': 6.901, 'Lymph_Nodes': 2.7070000000000003, 'Muscle': 1.9689999999999999, 'Oesophagus': 7.2509999999999994, 'Oral_Mucosa': 0.314, 'Ovaries': 0.154, 'Pancreas': 6.4079999999999995, 'Prostate': 0.025, 'Salivary_Glands': 0.40900000000000003, 'Skin': 1.368, 'Small_Intestine': 0.46, 'Spleen': 3.0709999999999997, 'Stomach': 2.541, 'Testicles': 0.0, 'Thymus': 2.534, 'Thyroid': 1.655, 'Urinary_Bladder': 0.07100000000000001, 'Uterus': 0.14300000000000002},
                },},

            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 1.206, 'Adrenals': 0.053, 'Brain': 2.67, 'Breast': 0.0, 'Colon': 0.005, 'Extrathoracic_Airways': 1.031, 'Gallbladder': 0.009000000000000001, 'Heart': 0.097, 'Kidneys': 0.016, 'Liver': 0.031, 'Lungs': 0.23800000000000002, 'Lymph_Nodes': 0.678, 'Muscle': 0.616, 'Oesophagus': 0.305, 'Oral_Mucosa': 2.098, 'Ovaries': 0.0, 'Pancreas': 0.060000000000000005, 'Prostate': 0.0, 'Salivary_Glands': 4.469, 'Endosteum': 2.715, 'Skin': 1.418, 'Small_Intestine': 0.007, 'Spleen': 0.045000000000000005, 'Stomach': 0.019, 'Testicles': 0.007, 'Thymus': 0.347, 'Thyroid': 1.835, 'Urinary_Bladder': 0.006, 'Uterus': 0.007},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 1.274, 'Adrenals': 0.059, 'Brain': 2.856, 'Breast': 0.0, 'Colon': 0.006, 'Extrathoracic_Airways': 1.129, 'Gallbladder': 0.011, 'Heart': 0.106, 'Kidneys': 0.018000000000000002, 'Liver': 0.034999999999999996, 'Lungs': 0.256, 'Lymph_Nodes': 0.7200000000000001, 'Muscle': 0.6489999999999999, 'Oesophagus': 0.331, 'Oral_Mucosa': 2.245, 'Ovaries': 0.0, 'Pancreas': 0.067, 'Prostate': 0.0, 'Salivary_Glands': 4.639, 'Endosteum': 2.857, 'Skin': 1.451, 'Small_Intestine': 0.007, 'Spleen': 0.049, 'Stomach': 0.020999999999999998, 'Testicles': 0.007, 'Thymus': 0.38, 'Thyroid': 1.997, 'Urinary_Bladder': 0.007, 'Uterus': 0.008},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 1.577, 'Adrenals': 0.095, 'Brain': 3.692, 'Breast': 0.0, 'Colon': 0.01, 'Extrathoracic_Airways': 1.647, 'Gallbladder': 0.02, 'Heart': 0.148, 'Kidneys': 0.029, 'Liver': 0.054, 'Lungs': 0.342, 'Lymph_Nodes': 0.914, 'Muscle': 0.789, 'Oesophagus': 0.46599999999999997, 'Oral_Mucosa': 2.931, 'Ovaries': 0.0, 'Pancreas': 0.098, 'Prostate': 0.0, 'Salivary_Glands': 5.316000000000001, 'Endosteum': 3.448, 'Skin': 1.6, 'Small_Intestine': 0.011, 'Spleen': 0.075, 'Stomach': 0.032, 'Testicles': 0.014, 'Thymus': 0.5509999999999999, 'Thyroid': 2.799, 'Urinary_Bladder': 0.012, 'Uterus': 0.014},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 1.333, 'Adrenals': 0.06499999999999999, 'Brain': 3.018, 'Breast': 0.0, 'Colon': 0.007, 'Extrathoracic_Airways': 1.219, 'Gallbladder': 0.012, 'Heart': 0.11299999999999999, 'Kidneys': 0.02, 'Liver': 0.038, 'Lungs': 0.273, 'Lymph_Nodes': 0.756, 'Muscle': 0.6759999999999999, 'Oesophagus': 0.355, 'Oral_Mucosa': 2.375, 'Ovaries': 0.0, 'Pancreas': 0.07200000000000001, 'Prostate': 0.0, 'Salivary_Glands': 4.779, 'Endosteum': 2.978, 'Skin': 1.48, 'Small_Intestine': 0.008, 'Spleen': 0.054, 'Stomach': 0.023, 'Testicles': 0.008, 'Thymus': 0.40900000000000003, 'Thyroid': 2.142, 'Urinary_Bladder': 0.007, 'Uterus': 0.009000000000000001},
                },},
        },
        1: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): {"Active_Bone_Marrow": 0.471, "Adrenals": 1.72, "Brain": 0.012, "Breast": 0.795, "Colon": 0.07, "Endosteum": 2.077, "Extrathoracic_Airways": 0.038, "Gallbladder": 0.253, "Heart": 0.673, "Kidneys": 1.382, "Liver": 0.739, "Lungs": 1.842, "Lymph_Nodes": 0.423, "Muscle": 0.469, "Oesophagus": 0.777, "Oral_Mucosa": 0.057, "Ovaries": 0.046, "Pancreas": 0.815, "Prostate": 0.0, "Salivary_Glands": 0.086, "Skin": 0.616, "Small_Intestine": 0.073, "Spleen": 1.679, "Stomach": 0.447, "Testicles": 0.003, "Thymus": 0.3, "Thyroid": 0.268, "Urinary_Bladder": 0.012, "Uterus": 0.011},
                    (65, 3.0, 0.0): {"Active_Bone_Marrow": 0.505, "Adrenals": 1.851, "Brain": 0.013, "Breast": 0.886, "Colon": 0.078, "Endosteum": 2.213, "Extrathoracic_Airways": 0.042, "Gallbladder": 0.281, "Heart": 0.74, "Kidneys": 1.469, "Liver": 0.804, "Lungs": 1.975, "Lymph_Nodes": 0.461, "Muscle": 0.499, "Oesophagus": 0.856, "Oral_Mucosa": 0.062, "Ovaries": 0.053, "Pancreas": 0.894, "Prostate": 0.0, "Salivary_Glands": 0.094, "Skin": 0.63, "Small_Intestine": 0.081, "Spleen": 1.806, "Stomach": 0.493, "Testicles": 0.003, "Thymus": 0.334, "Thyroid": 0.295, "Urinary_Bladder": 0.013, "Uterus": 0.012},
                    (65, 3.0, 0.1): {"Active_Bone_Marrow": 0.666, "Adrenals": 2.45, "Brain": 0.021, "Breast": 1.36, "Colon": 0.122, "Endosteum": 2.822, "Extrathoracic_Airways": 0.066, "Gallbladder": 0.437, "Heart": 1.077, "Kidneys": 1.839, "Liver": 1.12, "Lungs": 2.572, "Lymph_Nodes": 0.651, "Muscle": 0.641, "Oesophagus": 1.27, "Oral_Mucosa": 0.088, "Ovaries": 0.089, "Pancreas": 1.294, "Prostate": 0.0, "Salivary_Glands": 0.135, "Skin": 0.697, "Small_Intestine": 0.131, "Spleen": 2.39, "Stomach": 0.739, "Testicles": 0.006, "Thymus": 0.518, "Thyroid": 0.435, "Urinary_Bladder": 0.023, "Uterus": 0.021},
                    (65, 3.5, 0.0): {"Active_Bone_Marrow": 0.535, "Adrenals": 1.966, "Brain": 0.014, "Breast": 0.969, "Colon": 0.085, "Endosteum": 2.331, "Extrathoracic_Airways": 0.046, "Gallbladder": 0.307, "Heart": 0.799, "Kidneys": 1.543, "Liver": 0.861, "Lungs": 2.09, "Lymph_Nodes": 0.495, "Muscle": 0.526, "Oesophagus": 0.928, "Oral_Mucosa": 0.067, "Ovaries": 0.058, "Pancreas": 0.965, "Prostate": 0.0, "Salivary_Glands": 0.102, "Skin": 0.642, "Small_Intestine": 0.089, "Spleen": 1.917, "Stomach": 0.536, "Testicles": 0.004, "Thymus": 0.364, "Thyroid": 0.32, "Urinary_Bladder": 0.015, "Uterus": 0.014},
                },
            },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.54, 'Adrenals': 3.935, 'Brain': 0.016, 'Breast': 0.446, 'Colon': 0.07200000000000001, 'Endosteum': 2.041, 'Extrathoracic_Airways': 0.039, 'Gallbladder': 0.41800000000000004, 'Heart': 1.141, 'Kidneys': 2.045, 'Liver': 0.675, 'Lungs': 2.0500000000000003, 'Lymph_Nodes': 0.601, 'Muscle': 0.583, 'Oesophagus': 1.438, 'Oral_Mucosa': 0.062, 'Ovaries': 0.037, 'Pancreas': 1.349, 'Prostate': 0.008, 'Salivary_Glands': 0.1, 'Skin': 0.544, 'Small_Intestine': 0.088, 'Spleen': 0.763, 'Stomach': 0.566, 'Testicles': 0.01, 'Thymus': 0.515, 'Thyroid': 0.45399999999999996, 'Urinary_Bladder': 0.01, 'Uterus': 0.02},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.5810000000000001, 'Adrenals': 4.229, 'Brain': 0.018000000000000002, 'Breast': 0.481, 'Colon': 0.08, 'Endosteum': 2.179, 'Extrathoracic_Airways': 0.043000000000000003, 'Gallbladder': 0.467, 'Heart': 1.253, 'Kidneys': 2.178, 'Liver': 0.741, 'Lungs': 2.202, 'Lymph_Nodes': 0.657, 'Muscle': 0.622, 'Oesophagus': 1.5839999999999999, 'Oral_Mucosa': 0.06899999999999999, 'Ovaries': 0.041, 'Pancreas': 1.479, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.11, 'Skin': 0.558, 'Small_Intestine': 0.098, 'Spleen': 0.8270000000000001, 'Stomach': 0.625, 'Testicles': 0.012, 'Thymus': 0.5720000000000001, 'Thyroid': 0.506, 'Urinary_Bladder': 0.011, 'Uterus': 0.022},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.616, 'Adrenals': 4.486000000000001, 'Brain': 0.019, 'Breast': 0.5109999999999999, 'Colon': 0.088, 'Endosteum': 2.299, 'Extrathoracic_Airways': 0.048, 'Gallbladder': 0.513, 'Heart': 1.355, 'Kidneys': 2.294, 'Liver': 0.8009999999999999, 'Lungs': 2.334, 'Lymph_Nodes': 0.708, 'Muscle': 0.656, 'Oesophagus': 1.717, 'Oral_Mucosa': 0.075, 'Ovaries': 0.045000000000000005, 'Pancreas': 1.595, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.12000000000000001, 'Skin': 0.571, 'Small_Intestine': 0.108, 'Spleen': 0.884, 'Stomach': 0.679, 'Testicles': 0.013, 'Thymus': 0.624, 'Thyroid': 0.555, 'Urinary_Bladder': 0.012, 'Uterus': 0.025},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.7739999999999999, 'Adrenals': 5.569, 'Brain': 0.028, 'Breast': 0.636, 'Colon': 0.127, 'Endosteum': 2.8089999999999997, 'Extrathoracic_Airways': 0.07100000000000001, 'Gallbladder': 0.745, 'Heart': 1.828, 'Kidneys': 2.774, 'Liver': 1.081, 'Lungs': 2.887, 'Lymph_Nodes': 0.9420000000000001, 'Muscle': 0.803, 'Oesophagus': 2.3440000000000003, 'Oral_Mucosa': 0.104, 'Ovaries': 0.066, 'Pancreas': 2.13, 'Prostate': 0.012, 'Salivary_Glands': 0.16699999999999998, 'Skin': 0.629, 'Small_Intestine': 0.154, 'Spleen': 1.1360000000000001, 'Stomach': 0.934, 'Testicles': 0.023, 'Thymus': 0.8809999999999999, 'Thyroid': 0.8059999999999999, 'Urinary_Bladder': 0.018000000000000002, 'Uterus': 0.04},
                },},

            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.53, 'Adrenals': 0.009000000000000001, 'Brain': 1.267, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 0.45899999999999996, 'Gallbladder': 0.0, 'Heart': 0.04, 'Kidneys': 0.005, 'Liver': 0.008, 'Lungs': 0.092, 'Lymph_Nodes': 0.292, 'Muscle': 0.28200000000000003, 'Oesophagus': 0.177, 'Oral_Mucosa': 1.098, 'Ovaries': 0.0, 'Pancreas': 0.013, 'Prostate': 0.0, 'Salivary_Glands': 2.042, 'Endosteum': 1.496, 'Skin': 0.73, 'Small_Intestine': 0.001, 'Spleen': 0.011, 'Stomach': 0.006, 'Testicles': 0.0, 'Thymus': 0.14200000000000002, 'Thyroid': 0.525, 'Urinary_Bladder': 0.0, 'Uterus': 0.004},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.5640000000000001, 'Adrenals': 0.011, 'Brain': 1.3699999999999999, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 0.505, 'Gallbladder': 0.0, 'Heart': 0.044, 'Kidneys': 0.006, 'Liver': 0.009000000000000001, 'Lungs': 0.1, 'Lymph_Nodes': 0.313, 'Muscle': 0.298, 'Oesophagus': 0.194, 'Oral_Mucosa': 1.186, 'Ovaries': 0.0, 'Pancreas': 0.014, 'Prostate': 0.0, 'Salivary_Glands': 2.149, 'Endosteum': 1.5870000000000002, 'Skin': 0.746, 'Small_Intestine': 0.001, 'Spleen': 0.012, 'Stomach': 0.007, 'Testicles': 0.0, 'Thymus': 0.157, 'Thyroid': 0.584, 'Urinary_Bladder': 0.0, 'Uterus': 0.004},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.7240000000000001, 'Adrenals': 0.02, 'Brain': 1.8699999999999999, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 0.742, 'Gallbladder': 0.001, 'Heart': 0.06999999999999999, 'Kidneys': 0.009000000000000001, 'Liver': 0.014, 'Lungs': 0.14300000000000002, 'Lymph_Nodes': 0.41300000000000003, 'Muscle': 0.368, 'Oesophagus': 0.28800000000000003, 'Oral_Mucosa': 1.616, 'Ovaries': 0.0, 'Pancreas': 0.020999999999999998, 'Prostate': 0.0, 'Salivary_Glands': 2.608, 'Endosteum': 1.9849999999999999, 'Skin': 0.8140000000000001, 'Small_Intestine': 0.002, 'Spleen': 0.02, 'Stomach': 0.011, 'Testicles': 0.0, 'Thymus': 0.235, 'Thyroid': 0.916, 'Urinary_Bladder': 0.0, 'Uterus': 0.008},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.594, 'Adrenals': 0.012, 'Brain': 1.462, 'Breast': 0.0, 'Colon': 0.001, 'Extrathoracic_Airways': 0.5469999999999999, 'Gallbladder': 0.001, 'Heart': 0.048, 'Kidneys': 0.007, 'Liver': 0.01, 'Lungs': 0.107, 'Lymph_Nodes': 0.331, 'Muscle': 0.311, 'Oesophagus': 0.21000000000000002, 'Oral_Mucosa': 1.264, 'Ovaries': 0.0, 'Pancreas': 0.015000000000000001, 'Prostate': 0.0, 'Salivary_Glands': 2.2390000000000003, 'Endosteum': 1.665, 'Skin': 0.759, 'Small_Intestine': 0.001, 'Spleen': 0.014, 'Stomach': 0.007, 'Testicles': 0.0, 'Thymus': 0.17, 'Thyroid': 0.638, 'Urinary_Bladder': 0.0, 'Uterus': 0.005},
                },},
        },
        5: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.286, 'Adrenals': 1.203, 'Brain': 0.007, 'Breast': 0.362, 'Colon': 0.026, 'Extrathoracic_Airways': 0.069, 'Gallbladder': 0.128, 'Heart': 0.406, 'Kidneys': 0.688, 'Liver': 0.443, 'Lungs': 1.086, 'Lymph_Nodes': 0.257, 'Muscle': 0.259, 'Oesophagus': 0.492, 'Oral_Mucosa': 0.033, 'Ovaries': 0.01, 'Pancreas': 0.498, 'Prostate': 0.009, 'Salivary_Glands': 0.047, 'Endosteum': 0.983, 'Skin': 0.307, 'Small_Intestine': 0.034, 'Spleen': 0.962, 'Stomach': 0.252, 'Testicles': 0, 'Thymus': 0.16, 'Thyroid': 0.23, 'Urinary_Bladder': 0.002, 'Uterus': 0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.306, 'Adrenals': 1.294, 'Brain': 0.008, 'Breast': 0.395, 'Colon': 0.029, 'Extrathoracic_Airways': 0.075, 'Gallbladder': 0.142, 'Heart': 0.446, 'Kidneys': 0.734, 'Liver': 0.481, 'Lungs': 1.166, 'Lymph_Nodes': 0.281, 'Muscle': 0.276, 'Oesophagus': 0.543, 'Oral_Mucosa': 0.037, 'Ovaries': 0.011, 'Pancreas': 0.548, 'Prostate': 0.01, 'Salivary_Glands': 0.052, 'Endosteum': 1.047, 'Skin': 0.314, 'Small_Intestine': 0.038, 'Spleen': 1.039, 'Stomach': 0.277, 'Testicles': 0, 'Thymus': 0.177, 'Thyroid': 0.256, 'Urinary_Bladder': 0.002, 'Uterus': 0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.405, 'Adrenals': 1.698, 'Brain': 0.013, 'Breast': 0.558, 'Colon': 0.047, 'Extrathoracic_Airways': 0.109, 'Gallbladder': 0.224, 'Heart': 0.655, 'Kidneys': 0.947, 'Liver': 0.669, 'Lungs': 1.538, 'Lymph_Nodes': 0.403, 'Muscle': 0.357, 'Oesophagus': 0.821, 'Oral_Mucosa': 0.058, 'Ovaries': 0.02, 'Pancreas': 0.814, 'Prostate': 0.019, 'Salivary_Glands': 0.075, 'Endosteum': 1.341, 'Skin': 0.349, 'Small_Intestine': 0.06, 'Spleen': 1.404, 'Stomach': 0.413, 'Testicles': 0, 'Thymus': 0.272, 'Thyroid': 0.4, 'Urinary_Bladder': 0.003, 'Uterus': 0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.324, 'Adrenals': 1.373, 'Brain': 0.009, 'Breast': 0.425, 'Colon': 0.032, 'Extrathoracic_Airways': 0.081, 'Gallbladder': 0.156, 'Heart': 0.483, 'Kidneys': 0.775, 'Liver': 0.515, 'Lungs': 1.237, 'Lymph_Nodes': 0.302, 'Muscle': 0.291, 'Oesophagus': 0.59, 'Oral_Mucosa': 0.04, 'Ovaries': 0.013, 'Pancreas': 0.594, 'Prostate': 0.012, 'Salivary_Glands': 0.056, 'Endosteum': 1.104, 'Skin': 0.32, 'Small_Intestine': 0.042, 'Spleen': 1.106, 'Stomach': 0.301, 'Testicles': 0, 'Thymus': 0.193, 'Thyroid': 0.279, 'Urinary_Bladder': 0.002, 'Uterus': 0},
                },
            },
                        "Thoracic Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.308, 'Adrenals': 1.996, 'Brain': 0.008, 'Breast': 0.453, 'Colon': 0.04, 'Endosteum': 0.919, 'Extrathoracic_Airways': 0.034, 'Gallbladder': 0.281, 'Heart': 0.5900000000000001, 'Kidneys': 1.552, 'Liver': 0.45199999999999996, 'Lungs': 1.333, 'Lymph_Nodes': 0.358, 'Muscle': 0.301, 'Oesophagus': 0.728, 'Oral_Mucosa': 0.029, 'Ovaries': 0.027, 'Pancreas': 0.778, 'Prostate': 0.005, 'Salivary_Glands': 0.046, 'Skin': 0.28400000000000003, 'Small_Intestine': 0.052, 'Spleen': 0.704, 'Stomach': 0.36000000000000004, 'Testicles': 0.0, 'Thymus': 0.256, 'Thyroid': 0.311, 'Urinary_Bladder': 0.005, 'Uterus': 0.029},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.33, 'Adrenals': 2.139, 'Brain': 0.009000000000000001, 'Breast': 0.5, 'Colon': 0.044, 'Endosteum': 0.9810000000000001, 'Extrathoracic_Airways': 0.037, 'Gallbladder': 0.313, 'Heart': 0.647, 'Kidneys': 1.649, 'Liver': 0.494, 'Lungs': 1.4300000000000002, 'Lymph_Nodes': 0.39, 'Muscle': 0.321, 'Oesophagus': 0.804, 'Oral_Mucosa': 0.032, 'Ovaries': 0.030000000000000002, 'Pancreas': 0.852, 'Prostate': 0.005, 'Salivary_Glands': 0.051, 'Skin': 0.29100000000000004, 'Small_Intestine': 0.058, 'Spleen': 0.762, 'Stomach': 0.397, 'Testicles': 0.0, 'Thymus': 0.28400000000000003, 'Thyroid': 0.34400000000000003, 'Urinary_Bladder': 0.005, 'Uterus': 0.033},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.35, 'Adrenals': 2.263, 'Brain': 0.01, 'Breast': 0.5419999999999999, 'Colon': 0.049, 'Endosteum': 1.035, 'Extrathoracic_Airways': 0.041, 'Gallbladder': 0.34299999999999997, 'Heart': 0.6980000000000001, 'Kidneys': 1.734, 'Liver': 0.533, 'Lungs': 1.516, 'Lymph_Nodes': 0.42000000000000004, 'Muscle': 0.339, 'Oesophagus': 0.874, 'Oral_Mucosa': 0.034999999999999996, 'Ovaries': 0.033, 'Pancreas': 0.919, 'Prostate': 0.006, 'Salivary_Glands': 0.055, 'Skin': 0.297, 'Small_Intestine': 0.064, 'Spleen': 0.8140000000000001, 'Stomach': 0.431, 'Testicles': 0.0, 'Thymus': 0.31, 'Thyroid': 0.375, 'Urinary_Bladder': 0.006, 'Uterus': 0.036000000000000004},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.441, 'Adrenals': 2.782, 'Brain': 0.014, 'Breast': 0.746, 'Colon': 0.06999999999999999, 'Endosteum': 1.268, 'Extrathoracic_Airways': 0.058, 'Gallbladder': 0.49299999999999994, 'Heart': 0.9430000000000001, 'Kidneys': 2.083, 'Liver': 0.713, 'Lungs': 1.878, 'Lymph_Nodes': 0.559, 'Muscle': 0.417, 'Oesophagus': 1.216, 'Oral_Mucosa': 0.049, 'Ovaries': 0.05, 'Pancreas': 1.228, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.076, 'Skin': 0.325, 'Small_Intestine': 0.093, 'Spleen': 1.046, 'Stomach': 0.593, 'Testicles': 0.0, 'Thymus': 0.438, 'Thyroid': 0.53, 'Urinary_Bladder': 0.008, 'Uterus': 0.05},
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.314, 'Adrenals': 0.909, 'Brain': 0.008, 'Breast': 0.31, 'Colon': 0.235, 'Extrathoracic_Airways': 0.06, 'Gallbladder': 0.242, 'Heart': 0.293, 'Kidneys': 1.138, 'Liver': 0.407, 'Lungs': 0.738, 'Lymph_Nodes': 0.297, 'Muscle': 0.277, 'Oesophagus': 0.362, 'Oral_Mucosa': 0.051, 'Ovaries': 0.323, 'Pancreas': 0.371, 'Prostate': 0.055, 'Salivary_Glands': 0.124, 'Endosteum': 0.714, 'Skin': 0.289, 'Small_Intestine': 0.349, 'Spleen': 0.798, 'Stomach': 0.259, 'Testicles': 0.007, 'Thymus': 0.135, 'Thyroid': 0.192, 'Urinary_Bladder': 0.055, 'Uterus': 0.127},
                        (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.336, 'Adrenals': 0.975, 'Brain': 0.009, 'Breast': 0.336, 'Colon': 0.258, 'Extrathoracic_Airways': 0.067, 'Gallbladder': 0.268, 'Heart': 0.321, 'Kidneys': 1.206, 'Liver': 0.442, 'Lungs': 0.79, 'Lymph_Nodes': 0.323, 'Muscle': 0.295, 'Oesophagus': 0.398, 'Oral_Mucosa': 0.056, 'Ovaries': 0.353, 'Pancreas': 0.406, 'Prostate': 0.062, 'Salivary_Glands': 0.134, 'Endosteum': 0.759, 'Skin': 0.296, 'Small_Intestine': 0.382, 'Spleen': 0.858, 'Stomach': 0.283, 'Testicles': 0.007, 'Thymus': 0.148, 'Thyroid': 0.209, 'Urinary_Bladder': 0.061, 'Uterus': 0.141},
                        (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.443, 'Adrenals': 1.281, 'Brain': 0.013, 'Breast': 0.463, 'Colon': 0.377, 'Extrathoracic_Airways': 0.105, 'Gallbladder': 0.408, 'Heart': 0.469, 'Kidneys': 1.507, 'Liver': 0.611, 'Lungs': 1.032, 'Lymph_Nodes': 0.459, 'Muscle': 0.378, 'Oesophagus': 0.585, 'Oral_Mucosa': 0.083, 'Ovaries': 0.505, 'Pancreas': 0.586, 'Prostate': 0.099, 'Salivary_Glands': 0.183, 'Endosteum': 0.96, 'Skin': 0.328, 'Small_Intestine': 0.551, 'Spleen': 1.136, 'Stomach': 0.409, 'Testicles': 0.011, 'Thymus': 0.216, 'Thyroid': 0.294, 'Urinary_Bladder': 0.093, 'Uterus': 0.218},
                        (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.356, 'Adrenals': 1.033, 'Brain': 0.009, 'Breast': 0.359, 'Colon': 0.278, 'Extrathoracic_Airways': 0.073, 'Gallbladder': 0.291, 'Heart': 0.346, 'Kidneys': 1.265, 'Liver': 0.472, 'Lungs': 0.836, 'Lymph_Nodes': 0.347, 'Muscle': 0.31, 'Oesophagus': 0.43, 'Oral_Mucosa': 0.061, 'Ovaries': 0.38, 'Pancreas': 0.437, 'Prostate': 0.068, 'Salivary_Glands': 0.143, 'Endosteum': 0.798, 'Skin': 0.301, 'Small_Intestine': 0.411, 'Spleen': 0.91, 'Stomach': 0.305, 'Testicles': 0.008, 'Thymus': 0.16, 'Thyroid': 0.224, 'Urinary_Bladder': 0.066, 'Uterus': 0.154},
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.229, 'Adrenals': 0.003, 'Brain': 0.93, 'Breast': 0.031, 'Colon': 0.0, 'Extrathoracic_Airways': 0.39, 'Gallbladder': 0.002, 'Heart': 0.006, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.024, 'Lymph_Nodes': 0.23299999999999998, 'Muscle': 0.111, 'Oesophagus': 0.031, 'Oral_Mucosa': 0.8270000000000001, 'Ovaries': 0.0, 'Pancreas': 0.002, 'Prostate': 0.0, 'Salivary_Glands': 1.769, 'Endosteum': 1.0610000000000002, 'Skin': 0.375, 'Small_Intestine': 0.0, 'Spleen': 0.003, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.029, 'Thyroid': 0.443, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.245, 'Adrenals': 0.003, 'Brain': 1.011, 'Breast': 0.034, 'Colon': 0.0, 'Extrathoracic_Airways': 0.43, 'Gallbladder': 0.002, 'Heart': 0.006, 'Kidneys': 0.001, 'Liver': 0.003, 'Lungs': 0.027, 'Lymph_Nodes': 0.248, 'Muscle': 0.11699999999999999, 'Oesophagus': 0.034, 'Oral_Mucosa': 0.889, 'Ovaries': 0.0, 'Pancreas': 0.002, 'Prostate': 0.0, 'Salivary_Glands': 1.857, 'Endosteum': 1.1280000000000001, 'Skin': 0.383, 'Small_Intestine': 0.0, 'Spleen': 0.003, 'Stomach': 0.004, 'Testicles': 0.0, 'Thymus': 0.032, 'Thyroid': 0.486, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.322, 'Adrenals': 0.005, 'Brain': 1.418, 'Breast': 0.049, 'Colon': 0.001, 'Extrathoracic_Airways': 0.648, 'Gallbladder': 0.004, 'Heart': 0.01, 'Kidneys': 0.002, 'Liver': 0.004, 'Lungs': 0.04, 'Lymph_Nodes': 0.325, 'Muscle': 0.14400000000000002, 'Oesophagus': 0.049, 'Oral_Mucosa': 1.184, 'Ovaries': 0.0, 'Pancreas': 0.003, 'Prostate': 0.0, 'Salivary_Glands': 2.241, 'Endosteum': 1.426, 'Skin': 0.419, 'Small_Intestine': 0.0, 'Spleen': 0.005, 'Stomach': 0.006, 'Testicles': 0.0, 'Thymus': 0.047, 'Thyroid': 0.714, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.259, 'Adrenals': 0.004, 'Brain': 1.0839999999999999, 'Breast': 0.037, 'Colon': 0.0, 'Extrathoracic_Airways': 0.467, 'Gallbladder': 0.003, 'Heart': 0.007, 'Kidneys': 0.001, 'Liver': 0.003, 'Lungs': 0.029, 'Lymph_Nodes': 0.262, 'Muscle': 0.122, 'Oesophagus': 0.036000000000000004, 'Oral_Mucosa': 0.944, 'Ovaries': 0.0, 'Pancreas': 0.002, 'Prostate': 0.0, 'Salivary_Glands': 1.933, 'Endosteum': 1.186, 'Skin': 0.39, 'Small_Intestine': 0.0, 'Spleen': 0.003, 'Stomach': 0.004, 'Testicles': 0.0, 'Thymus': 0.034, 'Thyroid': 0.526, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
        10: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.203, 'Adrenals': 0.679, 'Brain': 0.003, 'Breast': 0.152, 'Colon': 0.019, 'Extrathoracic_Airways': 0.026, 'Gallbladder': 0.12, 'Heart': 0.223, 'Kidneys': 0.703, 'Liver': 0.296, 'Lungs': 0.609, 'Lymph_Nodes': 0.145, 'Muscle': 0.153, 'Oesophagus': 0.242, 'Oral_Mucosa': 0.011, 'Ovaries': 0.011, 'Pancreas': 0.268, 'Prostate': 0, 'Salivary_Glands': 0.025, 'Endosteum': 0.578, 'Skin': 0.216, 'Small_Intestine': 0.026, 'Spleen': 0.6, 'Stomach': 0.14, 'Testicles': 0, 'Thymus': 0.09, 'Thyroid': 0.1, 'Urinary_Bladder': 0.002, 'Uterus': 0.002},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.218, 'Adrenals': 0.729, 'Brain': 0.004, 'Breast': 0.167, 'Colon': 0.021, 'Extrathoracic_Airways': 0.029, 'Gallbladder': 0.133, 'Heart': 0.246, 'Kidneys': 0.75, 'Liver': 0.323, 'Lungs': 0.657, 'Lymph_Nodes': 0.158, 'Muscle': 0.163, 'Oesophagus': 0.267, 'Oral_Mucosa': 0.012, 'Ovaries': 0.012, 'Pancreas': 0.294, 'Prostate': 0, 'Salivary_Glands': 0.028, 'Endosteum': 0.617, 'Skin': 0.221, 'Small_Intestine': 0.028, 'Spleen': 0.647, 'Stomach': 0.154, 'Testicles': 0, 'Thymus': 0.1, 'Thyroid': 0.111, 'Urinary_Bladder': 0.002, 'Uterus': 0.002},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.292, 'Adrenals': 0.967, 'Brain': 0.006, 'Breast': 0.234, 'Colon': 0.034, 'Extrathoracic_Airways': 0.042, 'Gallbladder': 0.208, 'Heart': 0.37, 'Kidneys': 0.961, 'Liver': 0.459, 'Lungs': 0.881, 'Lymph_Nodes': 0.228, 'Muscle': 0.213, 'Oesophagus': 0.399, 'Oral_Mucosa': 0.019, 'Ovaries': 0.022, 'Pancreas': 0.431, 'Prostate': 0, 'Salivary_Glands': 0.041, 'Endosteum': 0.799, 'Skin': 0.243, 'Small_Intestine': 0.045, 'Spleen': 0.871, 'Stomach': 0.23, 'Testicles': 0, 'Thymus': 0.16, 'Thyroid': 0.167, 'Urinary_Bladder': 0.003, 'Uterus': 0.004},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.232, 'Adrenals': 0.772, 'Brain': 0.004, 'Breast': 0.18, 'Colon': 0.023, 'Extrathoracic_Airways': 0.031, 'Gallbladder': 0.145, 'Heart': 0.267, 'Kidneys': 0.791, 'Liver': 0.347, 'Lungs': 0.698, 'Lymph_Nodes': 0.171, 'Muscle': 0.173, 'Oesophagus': 0.29, 'Oral_Mucosa': 0.013, 'Ovaries': 0.013, 'Pancreas': 0.317, 'Prostate': 0, 'Salivary_Glands': 0.03, 'Endosteum': 0.652, 'Skin': 0.225, 'Small_Intestine': 0.031, 'Spleen': 0.689, 'Stomach': 0.167, 'Testicles': 0, 'Thymus': 0.11, 'Thyroid': 0.12, 'Urinary_Bladder': 0.002, 'Uterus': 0.002},
                },
            },
                        "Thoracic Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.23900000000000002, 'Adrenals': 1.2329999999999999, 'Brain': 0.003, 'Breast': 0.419, 'Colon': 0.024, 'Endosteum': 0.541, 'Extrathoracic_Airways': 0.034, 'Gallbladder': 0.192, 'Heart': 0.326, 'Kidneys': 1.048, 'Liver': 0.303, 'Lungs': 0.891, 'Lymph_Nodes': 0.222, 'Muscle': 0.179, 'Oesophagus': 0.373, 'Oral_Mucosa': 0.02, 'Ovaries': 0.024, 'Pancreas': 0.488, 'Prostate': 0.007, 'Salivary_Glands': 0.024, 'Skin': 0.187, 'Small_Intestine': 0.029, 'Spleen': 0.5680000000000001, 'Stomach': 0.223, 'Testicles': 0.001, 'Thymus': 0.121, 'Thyroid': 0.13999999999999999, 'Urinary_Bladder': 0.002, 'Uterus': 0.003},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.257, 'Adrenals': 1.333, 'Brain': 0.004, 'Breast': 0.457, 'Colon': 0.027, 'Endosteum': 0.5770000000000001, 'Extrathoracic_Airways': 0.038, 'Gallbladder': 0.212, 'Heart': 0.36000000000000004, 'Kidneys': 1.118, 'Liver': 0.332, 'Lungs': 0.959, 'Lymph_Nodes': 0.243, 'Muscle': 0.192, 'Oesophagus': 0.41100000000000003, 'Oral_Mucosa': 0.023, 'Ovaries': 0.027, 'Pancreas': 0.5379999999999999, 'Prostate': 0.008, 'Salivary_Glands': 0.026, 'Skin': 0.191, 'Small_Intestine': 0.032, 'Spleen': 0.616, 'Stomach': 0.247, 'Testicles': 0.001, 'Thymus': 0.136, 'Thyroid': 0.153, 'Urinary_Bladder': 0.002, 'Uterus': 0.003},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.273, 'Adrenals': 1.4220000000000002, 'Brain': 0.004, 'Breast': 0.49, 'Colon': 0.029, 'Endosteum': 0.61, 'Extrathoracic_Airways': 0.041999999999999996, 'Gallbladder': 0.231, 'Heart': 0.39, 'Kidneys': 1.178, 'Liver': 0.357, 'Lungs': 1.019, 'Lymph_Nodes': 0.261, 'Muscle': 0.203, 'Oesophagus': 0.446, 'Oral_Mucosa': 0.025, 'Ovaries': 0.029, 'Pancreas': 0.583, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.029, 'Skin': 0.195, 'Small_Intestine': 0.034999999999999996, 'Spleen': 0.6589999999999999, 'Stomach': 0.268, 'Testicles': 0.001, 'Thymus': 0.149, 'Thyroid': 0.165, 'Urinary_Bladder': 0.002, 'Uterus': 0.003},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.34600000000000003, 'Adrenals': 1.8159999999999998, 'Brain': 0.006, 'Breast': 0.641, 'Colon': 0.041999999999999996, 'Endosteum': 0.749, 'Extrathoracic_Airways': 0.060000000000000005, 'Gallbladder': 0.32, 'Heart': 0.539, 'Kidneys': 1.432, 'Liver': 0.47800000000000004, 'Lungs': 1.279, 'Lymph_Nodes': 0.34900000000000003, 'Muscle': 0.252, 'Oesophagus': 0.614, 'Oral_Mucosa': 0.034999999999999996, 'Ovaries': 0.041, 'Pancreas': 0.7969999999999999, 'Prostate': 0.015000000000000001, 'Salivary_Glands': 0.039, 'Skin': 0.213, 'Small_Intestine': 0.051, 'Spleen': 0.8490000000000001, 'Stomach': 0.372, 'Testicles': 0.002, 'Thymus': 0.219, 'Thyroid': 0.216, 'Urinary_Bladder': 0.004, 'Uterus': 0.005},
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.231, 'Adrenals': 0.505, 'Brain': 0.003, 'Breast': 0.116, 'Colon': 0.127, 'Extrathoracic_Airways': 0.055, 'Gallbladder': 0.144, 'Heart': 0.147, 'Kidneys': 0.679, 'Liver': 0.238, 'Lungs': 0.431, 'Lymph_Nodes': 0.167, 'Muscle': 0.159, 'Oesophagus': 0.191, 'Oral_Mucosa': 0.014, 'Ovaries': 0.2, 'Pancreas': 0.224, 'Prostate': 0.008, 'Salivary_Glands': 0.04, 'Endosteum': 0.409, 'Skin': 0.176, 'Small_Intestine': 0.186, 'Spleen': 0.447, 'Stomach': 0.136, 'Testicles': 0, 'Thymus': 0.066, 'Thyroid': 0.093, 'Urinary_Bladder': 0.035, 'Uterus': 0.139},
                        (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.248, 'Adrenals': 0.54, 'Brain': 0.003, 'Breast': 0.126, 'Colon': 0.14, 'Extrathoracic_Airways': 0.061, 'Gallbladder': 0.159, 'Heart': 0.161, 'Kidneys': 0.721, 'Liver': 0.259, 'Lungs': 0.463, 'Lymph_Nodes': 0.182, 'Muscle': 0.169, 'Oesophagus': 0.209, 'Oral_Mucosa': 0.015, 'Ovaries': 0.221, 'Pancreas': 0.247, 'Prostate': 0.009, 'Salivary_Glands': 0.043, 'Endosteum': 0.435, 'Skin': 0.179, 'Small_Intestine': 0.204, 'Spleen': 0.483, 'Stomach': 0.149, 'Testicles': 0, 'Thymus': 0.073, 'Thyroid': 0.102, 'Urinary_Bladder': 0.038, 'Uterus': 0.154},
                        (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.331, 'Adrenals': 0.705, 'Brain': 0.005, 'Breast': 0.169, 'Colon': 0.208, 'Extrathoracic_Airways': 0.092, 'Gallbladder': 0.25, 'Heart': 0.24, 'Kidneys': 0.911, 'Liver': 0.367, 'Lungs': 0.613, 'Lymph_Nodes': 0.264, 'Muscle': 0.219, 'Oesophagus': 0.3, 'Oral_Mucosa': 0.019, 'Ovaries': 0.327, 'Pancreas': 0.366, 'Prostate': 0.017, 'Salivary_Glands': 0.062, 'Endosteum': 0.554, 'Skin': 0.193, 'Small_Intestine': 0.302, 'Spleen': 0.661, 'Stomach': 0.224, 'Testicles': 0, 'Thymus': 0.114, 'Thyroid': 0.151, 'Urinary_Bladder': 0.06, 'Uterus': 0.233},
                        (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.263, 'Adrenals': 0.57, 'Brain': 0.003, 'Breast': 0.134, 'Colon': 0.151, 'Extrathoracic_Airways': 0.066, 'Gallbladder': 0.174, 'Heart': 0.175, 'Kidneys': 0.758, 'Liver': 0.278, 'Lungs': 0.492, 'Lymph_Nodes': 0.197, 'Muscle': 0.178, 'Oesophagus': 0.225, 'Oral_Mucosa': 0.016, 'Ovaries': 0.239, 'Pancreas': 0.267, 'Prostate': 0.01, 'Salivary_Glands': 0.046, 'Endosteum': 0.458, 'Skin': 0.182, 'Small_Intestine': 0.221, 'Spleen': 0.515, 'Stomach': 0.162, 'Testicles': 0, 'Thymus': 0.08, 'Thyroid': 0.11, 'Urinary_Bladder': 0.042, 'Uterus': 0.167},
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.12999999999999998, 'Adrenals': 0.0, 'Brain': 0.716, 'Breast': 0.016, 'Colon': 0.0, 'Extrathoracic_Airways': 0.303, 'Gallbladder': 0.0, 'Heart': 0.003, 'Kidneys': 0.0, 'Liver': 0.001, 'Lungs': 0.009000000000000001, 'Lymph_Nodes': 0.171, 'Muscle': 0.058, 'Oesophagus': 0.01, 'Oral_Mucosa': 0.5469999999999999, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.472, 'Endosteum': 0.623, 'Skin': 0.22, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.005, 'Thyroid': 0.087, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.13999999999999999, 'Adrenals': 0.0, 'Brain': 0.781, 'Breast': 0.018000000000000002, 'Colon': 0.0, 'Extrathoracic_Airways': 0.33399999999999996, 'Gallbladder': 0.0, 'Heart': 0.004, 'Kidneys': 0.0, 'Liver': 0.001, 'Lungs': 0.01, 'Lymph_Nodes': 0.182, 'Muscle': 0.061, 'Oesophagus': 0.011, 'Oral_Mucosa': 0.594, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.545, 'Endosteum': 0.664, 'Skin': 0.225, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.006, 'Thyroid': 0.096, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.186, 'Adrenals': 0.0, 'Brain': 1.108, 'Breast': 0.025, 'Colon': 0.0, 'Extrathoracic_Airways': 0.49799999999999994, 'Gallbladder': 0.0, 'Heart': 0.006, 'Kidneys': 0.0, 'Liver': 0.002, 'Lungs': 0.016, 'Lymph_Nodes': 0.23399999999999999, 'Muscle': 0.076, 'Oesophagus': 0.017, 'Oral_Mucosa': 0.839, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.867, 'Endosteum': 0.85, 'Skin': 0.247, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.009000000000000001, 'Thyroid': 0.13899999999999998, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.148, 'Adrenals': 0.0, 'Brain': 0.839, 'Breast': 0.019, 'Colon': 0.0, 'Extrathoracic_Airways': 0.36200000000000004, 'Gallbladder': 0.0, 'Heart': 0.004, 'Kidneys': 0.0, 'Liver': 0.001, 'Lungs': 0.011, 'Lymph_Nodes': 0.191, 'Muscle': 0.064, 'Oesophagus': 0.012, 'Oral_Mucosa': 0.637, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.608, 'Endosteum': 0.699, 'Skin': 0.229, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.007, 'Thyroid': 0.103, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
        15: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.216, 'Adrenals': 0.665, 'Brain': 0.003, 'Breast': 0.113, 'Colon': 0.016, 'Extrathoracic_Airways': 0.046, 'Gallbladder': 0.11, 'Heart': 0.217, 'Kidneys': 0.466, 'Liver': 0.278, 'Lungs': 0.561, 'Lymph_Nodes': 0.148, 'Muscle': 0.139, 'Oesophagus': 0.271, 'Oral_Mucosa': 0.009, 'Ovaries': 0.007, 'Pancreas': 0.301, 'Prostate': 0, 'Salivary_Glands': 0.016, 'Endosteum': 0.449, 'Skin': 0.144, 'Small_Intestine': 0.019, 'Spleen': 0.561, 'Stomach': 0.145, 'Testicles': 0, 'Thymus': 0.096, 'Thyroid': 0.144, 'Urinary_Bladder': 0.002, 'Uterus': 0.003},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.229, 'Adrenals': 0.709, 'Brain': 0.004, 'Breast': 0.123, 'Colon': 0.017, 'Extrathoracic_Airways': 0.049, 'Gallbladder': 0.119, 'Heart': 0.235, 'Kidneys': 0.492, 'Liver': 0.298, 'Lungs': 0.597, 'Lymph_Nodes': 0.159, 'Muscle': 0.147, 'Oesophagus': 0.293, 'Oral_Mucosa': 0.01, 'Ovaries': 0.007, 'Pancreas': 0.325, 'Prostate': 0, 'Salivary_Glands': 0.017, 'Endosteum': 0.472, 'Skin': 0.148, 'Small_Intestine': 0.021, 'Spleen': 0.597, 'Stomach': 0.156, 'Testicles': 0, 'Thymus': 0.104, 'Thyroid': 0.156, 'Urinary_Bladder': 0.002, 'Uterus': 0.003},
                    (100, 3.0, 0.1): {'Active_Bone_Marrow': 0.292, 'Adrenals': 0.92, 'Brain': 0.005, 'Breast': 0.169, 'Colon': 0.025, 'Extrathoracic_Airways': 0.069, 'Gallbladder': 0.168, 'Heart': 0.328, 'Kidneys': 0.609, 'Liver': 0.396, 'Lungs': 0.765, 'Lymph_Nodes': 0.215, 'Muscle': 0.185, 'Oesophagus': 0.408, 'Oral_Mucosa': 0.014, 'Ovaries': 0.01, 'Pancreas': 0.449, 'Prostate': 0, 'Salivary_Glands': 0.024, 'Endosteum': 0.575, 'Skin': 0.164, 'Small_Intestine': 0.031, 'Spleen': 0.768, 'Stomach': 0.218, 'Testicles': 0, 'Thymus': 0.149, 'Thyroid': 0.218, 'Urinary_Bladder': 0.003, 'Uterus': 0.005},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.24, 'Adrenals': 0.748, 'Brain': 0.004, 'Breast': 0.131, 'Colon': 0.018, 'Extrathoracic_Airways': 0.053, 'Gallbladder': 0.127, 'Heart': 0.251, 'Kidneys': 0.515, 'Liver': 0.316, 'Lungs': 0.628, 'Lymph_Nodes': 0.169, 'Muscle': 0.154, 'Oesophagus': 0.313, 'Oral_Mucosa': 0.01, 'Ovaries': 0.008, 'Pancreas': 0.347, 'Prostate': 0, 'Salivary_Glands': 0.018, 'Endosteum': 0.492, 'Skin': 0.151, 'Small_Intestine': 0.023, 'Spleen': 0.629, 'Stomach': 0.167, 'Testicles': 0, 'Thymus': 0.112, 'Thyroid': 0.167, 'Urinary_Bladder': 0.002, 'Uterus': 0.004},
                },
            },
                        "Thoracic Spine": {                "U": {
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.24600000000000002, 'Adrenals': 1.019, 'Brain': 0.004, 'Breast': 0.135, 'Colon': 0.018000000000000002, 'Endosteum': 0.433, 'Extrathoracic_Airways': 0.041999999999999996, 'Gallbladder': 0.13899999999999998, 'Heart': 0.281, 'Kidneys': 0.588, 'Liver': 0.257, 'Lungs': 0.717, 'Lymph_Nodes': 0.186, 'Muscle': 0.152, 'Oesophagus': 0.451, 'Oral_Mucosa': 0.009000000000000001, 'Ovaries': 0.001, 'Pancreas': 0.389, 'Prostate': 0.002, 'Salivary_Glands': 0.02, 'Skin': 0.14200000000000002, 'Small_Intestine': 0.02, 'Spleen': 0.465, 'Stomach': 0.176, 'Testicles': 0.0, 'Thymus': 0.12999999999999998, 'Thyroid': 0.135, 'Urinary_Bladder': 0.001, 'Uterus': 0.005},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.261, 'Adrenals': 1.082, 'Brain': 0.004, 'Breast': 0.146, 'Colon': 0.019, 'Endosteum': 0.455, 'Extrathoracic_Airways': 0.046, 'Gallbladder': 0.15100000000000002, 'Heart': 0.304, 'Kidneys': 0.622, 'Liver': 0.27599999999999997, 'Lungs': 0.761, 'Lymph_Nodes': 0.199, 'Muscle': 0.161, 'Oesophagus': 0.488, 'Oral_Mucosa': 0.01, 'Ovaries': 0.001, 'Pancreas': 0.42000000000000004, 'Prostate': 0.003, 'Salivary_Glands': 0.020999999999999998, 'Skin': 0.145, 'Small_Intestine': 0.022, 'Spleen': 0.496, 'Stomach': 0.19, 'Testicles': 0.0, 'Thymus': 0.14200000000000002, 'Thyroid': 0.146, 'Urinary_Bladder': 0.001, 'Uterus': 0.005},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.27399999999999997, 'Adrenals': 1.138, 'Brain': 0.005, 'Breast': 0.157, 'Colon': 0.020999999999999998, 'Endosteum': 0.475, 'Extrathoracic_Airways': 0.049, 'Gallbladder': 0.162, 'Heart': 0.324, 'Kidneys': 0.651, 'Liver': 0.29300000000000004, 'Lungs': 0.8, 'Lymph_Nodes': 0.212, 'Muscle': 0.16799999999999998, 'Oesophagus': 0.521, 'Oral_Mucosa': 0.01, 'Ovaries': 0.001, 'Pancreas': 0.448, 'Prostate': 0.003, 'Salivary_Glands': 0.022, 'Skin': 0.148, 'Small_Intestine': 0.024, 'Spleen': 0.523, 'Stomach': 0.203, 'Testicles': 0.0, 'Thymus': 0.152, 'Thyroid': 0.156, 'Urinary_Bladder': 0.001, 'Uterus': 0.005},
                    (100, 3.0, 0.1): {'Active_Bone_Marrow': 0.33399999999999996, 'Adrenals': 1.3760000000000001, 'Brain': 0.006, 'Breast': 0.205, 'Colon': 0.028, 'Endosteum': 0.5569999999999999, 'Extrathoracic_Airways': 0.06499999999999999, 'Gallbladder': 0.214, 'Heart': 0.421, 'Kidneys': 0.7739999999999999, 'Liver': 0.372, 'Lungs': 0.968, 'Lymph_Nodes': 0.27, 'Muscle': 0.203, 'Oesophagus': 0.68, 'Oral_Mucosa': 0.014, 'Ovaries': 0.001, 'Pancreas': 0.578, 'Prostate': 0.004, 'Salivary_Glands': 0.029, 'Skin': 0.161, 'Small_Intestine': 0.032, 'Spleen': 0.643, 'Stomach': 0.263, 'Testicles': 0.001, 'Thymus': 0.203, 'Thyroid': 0.206, 'Urinary_Bladder': 0.001, 'Uterus': 0.007},
                },},
                "Scoliosis": {
                    "U": {
                        (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.163, 'Adrenals': 0.385, 'Brain': 0.003, 'Breast': 0.051, 'Colon': 0.062, 'Extrathoracic_Airways': 0.046, 'Gallbladder': 0.115, 'Heart': 0.099, 'Kidneys': 0.494, 'Liver': 0.163, 'Lungs': 0.295, 'Lymph_Nodes': 0.116, 'Muscle': 0.106, 'Oesophagus': 0.136, 'Oral_Mucosa': 0.021, 'Ovaries': 0.048, 'Pancreas': 0.146, 'Prostate': 0, 'Salivary_Glands': 0.077, 'Endosteum': 0.284, 'Skin': 0.132, 'Small_Intestine': 0.108, 'Spleen': 0.316, 'Stomach': 0.094, 'Testicles': 0.001, 'Thymus': 0.043, 'Thyroid': 0.077, 'Urinary_Bladder': 0.008, 'Uterus': 0.027},
                        (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.175, 'Adrenals': 0.412, 'Brain': 0.003, 'Breast': 0.055, 'Colon': 0.067, 'Extrathoracic_Airways': 0.05, 'Gallbladder': 0.127, 'Heart': 0.108, 'Kidneys': 0.527, 'Liver': 0.177, 'Lungs': 0.318, 'Lymph_Nodes': 0.126, 'Muscle': 0.113, 'Oesophagus': 0.15, 'Oral_Mucosa': 0.023, 'Ovaries': 0.053, 'Pancreas': 0.16, 'Prostate': 0, 'Salivary_Glands': 0.082, 'Endosteum': 0.302, 'Skin': 0.135, 'Small_Intestine': 0.118, 'Spleen': 0.341, 'Stomach': 0.104, 'Testicles': 0.001, 'Thymus': 0.048, 'Thyroid': 0.085, 'Urinary_Bladder': 0.009, 'Uterus': 0.03},
                        (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.235, 'Adrenals': 0.535, 'Brain': 0.005, 'Breast': 0.082, 'Colon': 0.099, 'Extrathoracic_Airways': 0.072, 'Gallbladder': 0.188, 'Heart': 0.161, 'Kidneys': 0.679, 'Liver': 0.25, 'Lungs': 0.427, 'Lymph_Nodes': 0.181, 'Muscle': 0.147, 'Oesophagus': 0.224, 'Oral_Mucosa': 0.035, 'Ovaries': 0.084, 'Pancreas': 0.237, 'Prostate': 0, 'Salivary_Glands': 0.109, 'Endosteum': 0.385, 'Skin': 0.151, 'Small_Intestine': 0.175, 'Spleen': 0.469, 'Stomach': 0.155, 'Testicles': 0.002, 'Thymus': 0.076, 'Thyroid': 0.126, 'Urinary_Bladder': 0.014, 'Uterus': 0.048},
                        (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.186, 'Adrenals': 0.436, 'Brain': 0.004, 'Breast': 0.06, 'Colon': 0.073, 'Extrathoracic_Airways': 0.054, 'Gallbladder': 0.137, 'Heart': 0.117, 'Kidneys': 0.556, 'Liver': 0.19, 'Lungs': 0.338, 'Lymph_Nodes': 0.136, 'Muscle': 0.119, 'Oesophagus': 0.162, 'Oral_Mucosa': 0.025, 'Ovaries': 0.058, 'Pancreas': 0.173, 'Prostate': 0, 'Salivary_Glands': 0.087, 'Endosteum': 0.317, 'Skin': 0.138, 'Small_Intestine': 0.128, 'Spleen': 0.364, 'Stomach': 0.112, 'Testicles': 0.001, 'Thymus': 0.052, 'Thyroid': 0.092, 'Urinary_Bladder': 0.01, 'Uterus': 0.033},
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.094, 'Adrenals': 0.0, 'Brain': 0.662, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.221, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.003, 'Lymph_Nodes': 0.14100000000000001, 'Muscle': 0.04, 'Oesophagus': 0.001, 'Oral_Mucosa': 0.46900000000000003, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.3159999999999998, 'Endosteum': 0.419, 'Skin': 0.161, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.003, 'Thyroid': 0.04, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.101, 'Adrenals': 0.0, 'Brain': 0.7240000000000001, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.244, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.004, 'Lymph_Nodes': 0.15100000000000002, 'Muscle': 0.041999999999999996, 'Oesophagus': 0.001, 'Oral_Mucosa': 0.508, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.39, 'Endosteum': 0.449, 'Skin': 0.165, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.003, 'Thyroid': 0.044, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.13699999999999998, 'Adrenals': 0.0, 'Brain': 1.0410000000000001, 'Breast': 0.001, 'Colon': 0.0, 'Extrathoracic_Airways': 0.36900000000000005, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.006, 'Lymph_Nodes': 0.19699999999999998, 'Muscle': 0.052, 'Oesophagus': 0.002, 'Oral_Mucosa': 0.705, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.72, 'Endosteum': 0.588, 'Skin': 0.18100000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.004, 'Thyroid': 0.066, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.108, 'Adrenals': 0.0, 'Brain': 0.7789999999999999, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.265, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.004, 'Lymph_Nodes': 0.159, 'Muscle': 0.044, 'Oesophagus': 0.001, 'Oral_Mucosa': 0.544, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.454, 'Endosteum': 0.475, 'Skin': 0.16799999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.003, 'Thyroid': 0.048, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
    },
}


# PA effective dose coefficients (mSv / Gy·cm²)

eff_coefficients_PA: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (60, 2.5, 0.0): 0.068,
                    (60, 3.0, 0.0): 0.074,
                    (60, 3.5, 0.0): 0.08,
                    (80, 2.5, 0.0): 0.099,
                    (80, 3.0, 0.0): 0.107,
                    (80, 3.5, 0.0): 0.115,
                    (100, 2.5, 0.0): 0.127,
                    (100, 3.0, 0.0): 0.136,
                    (100, 3.5, 0.0): 0.144,
                    (120, 2.5, 0.0): 0.15,
                    (120, 3.0, 0.0): 0.159,
                    (120, 3.5, 0.0): 0.166,
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.169,
                    (60, 3.0, 0.0): 0.182,
                    (60, 3.5, 0.0): 0.194,
                    (80, 2.5, 0.0): 0.226,
                    (80, 3.0, 0.0): 0.241,
                    (80, 3.5, 0.0): 0.255,
                    (100, 2.5, 0.0): 0.274,
                    (100, 3.0, 0.0): 0.289,
                    (100, 3.5, 0.0): 0.303,
                    (120, 2.5, 0.0): 0.311,
                    (120, 3.0, 0.0): 0.325,
                    (120, 3.5, 0.0): 0.338,
                },
            },
            "Scoliosis": {
                "U": {
                    (60, 2.5, 0.0): 0.067,
                    (60, 3.0, 0.0): 0.073,
                    (60, 3.5, 0.0): 0.079,
                    (80, 2.5, 0.0): 0.101,
                    (80, 3.0, 0.0): 0.109,
                    (80, 3.5, 0.0): 0.117,
                    (100, 2.5, 0.0): 0.131,
                    (100, 3.0, 0.0): 0.141,
                    (100, 3.5, 0.0): 0.149,
                    (120, 2.5, 0.0): 0.156,
                    (120, 3.0, 0.0): 0.165,
                    (120, 3.5, 0.0): 0.173,
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): 0.029,
                    (60, 3.0, 0.0): 0.032,
                    (60, 3.5, 0.0): 0.034,
                    (80, 2.5, 0.0): 0.041,
                    (80, 3.0, 0.0): 0.044,
                    (80, 3.5, 0.0): 0.047,
                    (100, 2.5, 0.0): 0.052,
                    (100, 3.0, 0.0): 0.055,
                    (100, 3.5, 0.0): 0.058,
                    (120, 2.5, 0.0): 0.06,
                    (120, 3.0, 0.0): 0.063,
                    (120, 3.5, 0.0): 0.066,
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): 1.839,
                    (65, 3.0, 0.0): 1.964,
                    (65, 3.0, 0.1): 2.487,
                    (65, 3.5, 0.0): 2.07,
                },
            },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): 1.747,
                    (65, 3.0, 0.0): 1.881,
                    (65, 3.5, 0.0): 1.9980000000000002,
                    (65, 3.0, 0.1): 2.5,
                },},

            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): 0.423,
                    (70, 3.0, 0.0): 0.45,
                    (70, 3.0, 0.1): 0.5770000000000001,
                    (70, 3.5, 0.0): 0.474,
                },},
        },
        1: {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (65, 2.5, 0.0): 0.609,
                    (65, 3.0, 0.0): 0.662,
                    (65, 3.0, 0.1): 0.917,
                    (65, 3.5, 0.0): 0.708,
                },
            },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): 0.68,
                    (65, 3.0, 0.0): 0.736,
                    (65, 3.5, 0.0): 0.786,
                    (65, 3.0, 0.1): 1.011,
                },},

            "Skull": {                
                "U": {
                    (70, 2.5, 0.0): 0.18100000000000002,
                    (70, 3.0, 0.0): 0.194,
                    (70, 3.0, 0.1): 0.25999999999999995,
                    (70, 3.5, 0.0): 0.206,
                },},
        },
        5: {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): 0.345,
                    (70, 3.0, 0.0): 0.374,
                    (70, 3.0, 0.1): 0.514,
                    (70, 3.5, 0.0): 0.399,
                },
            },
                        "Thoracic Spine": {                "U": {
                    (70, 2.5, 0.0): 0.437,
                    (70, 3.0, 0.0): 0.474,
                    (70, 3.5, 0.0): 0.507,
                    (70, 3.0, 0.1): 0.6589999999999999,
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): 0.334,
                        (75, 3.0, 0.0): 0.361,
                        (75, 3.0, 0.1): 0.494,
                        (75, 3.5, 0.0): 0.385,
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.11,
                    (75, 3.0, 0.0): 0.118,
                    (75, 3.0, 0.1): 0.158,
                    (75, 3.5, 0.0): 0.125,
                },},
        },
        10: {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (70, 2.5, 0.0): 0.197,
                    (70, 3.0, 0.0): 0.214,
                    (70, 3.0, 0.1): 0.295,
                    (70, 3.5, 0.0): 0.228,
                },
            },
                        "Thoracic Spine": {                "U": {
                    (70, 2.5, 0.0): 0.298,
                    (70, 3.0, 0.0): 0.323,
                    (70, 3.5, 0.0): 0.34600000000000003,
                    (70, 3.0, 0.1): 0.448,
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): 0.188,
                        (75, 3.0, 0.0): 0.203,
                        (75, 3.0, 0.1): 0.281,
                        (75, 3.5, 0.0): 0.217,
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.063,
                    (75, 3.0, 0.0): 0.068,
                    (75, 3.0, 0.1): 0.09000000000000001,
                    (75, 3.5, 0.0): 0.07200000000000001,
                },},
        },
        15: {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (100, 2.5, 0.0): 0.186,
                    (100, 3.0, 0.0): 0.199,
                    (100, 3.0, 0.1): 0.261,
                    (100, 3.5, 0.0): 0.21,
                },
            },
                        "Thoracic Spine": {                "U": {
                    (100, 2.5, 0.0): 0.22599999999999998,
                    (100, 3.0, 0.0): 0.242,
                    (100, 3.5, 0.0): 0.256,
                    (100, 3.0, 0.1): 0.318,
                },},
                "Scoliosis": {
                    "U": {
                        (80, 2.5, 0.0): 0.121,
                        (80, 3.0, 0.0): 0.131,
                        (80, 3.0, 0.1): 0.182,
                        (80, 3.5, 0.0): 0.14,
                    },
                },
            "Skull": {                
                "U": {
                    (75, 2.5, 0.0): 0.047,
                    (75, 3.0, 0.0): 0.051,
                    (75, 3.0, 0.1): 0.068,
                    (75, 3.5, 0.0): 0.054,
                },},
        },
    },
}


# LATR organ coefficients (mGy / Gy·cm²)

coefficients_LATR: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Chest": {
                "U": {
                    (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.102, 'Adrenals': 0.115, 'Brain': 0.002, 'Breast': 0.39399999999999996, 'Colon': 0.008, 'Endosteum': 0.162, 'Extrathoracic_Airways': 0.038, 'Gallbladder': 0.077, 'Heart': 0.155, 'Kidneys': 0.08, 'Liver': 0.37, 'Lungs': 0.328, 'Lymph_Nodes': 0.073, 'Muscle': 0.07200000000000001, 'Oesophagus': 0.146, 'Oral_Mucosa': 0.004, 'Ovaries': 0.001, 'Pancreas': 0.061, 'Prostate': 0.0, 'Salivary_Glands': 0.014, 'Skin': 0.10500000000000001, 'Small_Intestine': 0.008, 'Spleen': 0.015000000000000001, 'Stomach': 0.025, 'Testicles': 0.0, 'Thymus': 0.132, 'Thyroid': 0.147, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.107, 'Adrenals': 0.124, 'Brain': 0.002, 'Breast': 0.408, 'Colon': 0.009000000000000001, 'Endosteum': 0.16899999999999998, 'Extrathoracic_Airways': 0.04, 'Gallbladder': 0.083, 'Heart': 0.165, 'Kidneys': 0.085, 'Liver': 0.39, 'Lungs': 0.34600000000000003, 'Lymph_Nodes': 0.077, 'Muscle': 0.075, 'Oesophagus': 0.156, 'Oral_Mucosa': 0.005, 'Ovaries': 0.001, 'Pancreas': 0.06499999999999999, 'Prostate': 0.0, 'Salivary_Glands': 0.015000000000000001, 'Skin': 0.108, 'Small_Intestine': 0.009000000000000001, 'Spleen': 0.016, 'Stomach': 0.027, 'Testicles': 0.0, 'Thymus': 0.13999999999999999, 'Thyroid': 0.155, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.112, 'Adrenals': 0.131, 'Brain': 0.002, 'Breast': 0.419, 'Colon': 0.009000000000000001, 'Endosteum': 0.175, 'Extrathoracic_Airways': 0.041999999999999996, 'Gallbladder': 0.087, 'Heart': 0.174, 'Kidneys': 0.089, 'Liver': 0.40700000000000003, 'Lungs': 0.36200000000000004, 'Lymph_Nodes': 0.081, 'Muscle': 0.078, 'Oesophagus': 0.164, 'Oral_Mucosa': 0.005, 'Ovaries': 0.001, 'Pancreas': 0.06899999999999999, 'Prostate': 0.0, 'Salivary_Glands': 0.016, 'Skin': 0.11, 'Small_Intestine': 0.009000000000000001, 'Spleen': 0.017, 'Stomach': 0.029, 'Testicles': 0.0, 'Thymus': 0.148, 'Thyroid': 0.162, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.087, 'Adrenals': 0.08600000000000001, 'Brain': 0.002, 'Breast': 0.363, 'Colon': 0.006, 'Endosteum': 0.15, 'Extrathoracic_Airways': 0.034, 'Gallbladder': 0.061, 'Heart': 0.12300000000000001, 'Kidneys': 0.06499999999999999, 'Liver': 0.319, 'Lungs': 0.28200000000000003, 'Lymph_Nodes': 0.060000000000000005, 'Muscle': 0.063, 'Oesophagus': 0.116, 'Oral_Mucosa': 0.004, 'Ovaries': 0.001, 'Pancreas': 0.045000000000000005, 'Prostate': 0.0, 'Salivary_Glands': 0.012, 'Skin': 0.09899999999999999, 'Small_Intestine': 0.006, 'Spleen': 0.01, 'Stomach': 0.017, 'Testicles': 0.0, 'Thymus': 0.10500000000000001, 'Thyroid': 0.126, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.093, 'Adrenals': 0.093, 'Brain': 0.002, 'Breast': 0.379, 'Colon': 0.007, 'Endosteum': 0.158, 'Extrathoracic_Airways': 0.036000000000000004, 'Gallbladder': 0.066, 'Heart': 0.132, 'Kidneys': 0.07100000000000001, 'Liver': 0.34, 'Lungs': 0.301, 'Lymph_Nodes': 0.064, 'Muscle': 0.066, 'Oesophagus': 0.125, 'Oral_Mucosa': 0.004, 'Ovaries': 0.001, 'Pancreas': 0.048, 'Prostate': 0.0, 'Salivary_Glands': 0.013, 'Skin': 0.102, 'Small_Intestine': 0.006, 'Spleen': 0.011, 'Stomach': 0.019, 'Testicles': 0.0, 'Thymus': 0.11299999999999999, 'Thyroid': 0.135, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.098, 'Adrenals': 0.1, 'Brain': 0.002, 'Breast': 0.392, 'Colon': 0.007, 'Endosteum': 0.165, 'Extrathoracic_Airways': 0.038, 'Gallbladder': 0.07100000000000001, 'Heart': 0.14100000000000001, 'Kidneys': 0.075, 'Liver': 0.359, 'Lungs': 0.318, 'Lymph_Nodes': 0.068, 'Muscle': 0.06899999999999999, 'Oesophagus': 0.134, 'Oral_Mucosa': 0.004, 'Ovaries': 0.001, 'Pancreas': 0.052, 'Prostate': 0.0, 'Salivary_Glands': 0.014, 'Skin': 0.104, 'Small_Intestine': 0.007, 'Spleen': 0.012, 'Stomach': 0.02, 'Testicles': 0.0, 'Thymus': 0.121, 'Thyroid': 0.14300000000000002, 'Urinary_Bladder': 0.0, 'Uterus': 0.001},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.06899999999999999, 'Adrenals': 0.053, 'Brain': 0.001, 'Breast': 0.324, 'Colon': 0.004, 'Endosteum': 0.12999999999999998, 'Extrathoracic_Airways': 0.028, 'Gallbladder': 0.041999999999999996, 'Heart': 0.085, 'Kidneys': 0.047, 'Liver': 0.255, 'Lungs': 0.223, 'Lymph_Nodes': 0.044, 'Muscle': 0.051, 'Oesophagus': 0.083, 'Oral_Mucosa': 0.002, 'Ovaries': 0.001, 'Pancreas': 0.029, 'Prostate': 0.0, 'Salivary_Glands': 0.008, 'Skin': 0.092, 'Small_Intestine': 0.003, 'Spleen': 0.006, 'Stomach': 0.009000000000000001, 'Testicles': 0.0, 'Thymus': 0.075, 'Thyroid': 0.096, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.075, 'Adrenals': 0.059, 'Brain': 0.001, 'Breast': 0.34, 'Colon': 0.004, 'Endosteum': 0.13899999999999998, 'Extrathoracic_Airways': 0.030000000000000002, 'Gallbladder': 0.046, 'Heart': 0.093, 'Kidneys': 0.052, 'Liver': 0.27599999999999997, 'Lungs': 0.242, 'Lymph_Nodes': 0.048, 'Muscle': 0.055, 'Oesophagus': 0.091, 'Oral_Mucosa': 0.002, 'Ovaries': 0.001, 'Pancreas': 0.032, 'Prostate': 0.0, 'Salivary_Glands': 0.008, 'Skin': 0.095, 'Small_Intestine': 0.004, 'Spleen': 0.007, 'Stomach': 0.01, 'Testicles': 0.0, 'Thymus': 0.082, 'Thyroid': 0.104, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.079, 'Adrenals': 0.064, 'Brain': 0.001, 'Breast': 0.355, 'Colon': 0.005, 'Endosteum': 0.146, 'Extrathoracic_Airways': 0.032, 'Gallbladder': 0.05, 'Heart': 0.101, 'Kidneys': 0.056, 'Liver': 0.294, 'Lungs': 0.259, 'Lymph_Nodes': 0.052, 'Muscle': 0.058, 'Oesophagus': 0.098, 'Oral_Mucosa': 0.003, 'Ovaries': 0.001, 'Pancreas': 0.034, 'Prostate': 0.0, 'Salivary_Glands': 0.009000000000000001, 'Skin': 0.097, 'Small_Intestine': 0.004, 'Spleen': 0.007, 'Stomach': 0.011, 'Testicles': 0.0, 'Thymus': 0.089, 'Thyroid': 0.111, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.05, 'Adrenals': 0.028, 'Brain': 0.0, 'Breast': 0.275, 'Colon': 0.002, 'Endosteum': 0.102, 'Extrathoracic_Airways': 0.023, 'Gallbladder': 0.022, 'Heart': 0.047, 'Kidneys': 0.028, 'Liver': 0.18000000000000002, 'Lungs': 0.154, 'Lymph_Nodes': 0.028, 'Muscle': 0.038, 'Oesophagus': 0.048, 'Oral_Mucosa': 0.001, 'Ovaries': 0.0, 'Pancreas': 0.014, 'Prostate': 0.0, 'Salivary_Glands': 0.004, 'Skin': 0.08399999999999999, 'Small_Intestine': 0.001, 'Spleen': 0.003, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.041999999999999996, 'Thyroid': 0.064, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.054, 'Adrenals': 0.031, 'Brain': 0.001, 'Breast': 0.29100000000000004, 'Colon': 0.002, 'Endosteum': 0.11, 'Extrathoracic_Airways': 0.025, 'Gallbladder': 0.024, 'Heart': 0.053, 'Kidneys': 0.031, 'Liver': 0.19699999999999998, 'Lungs': 0.16899999999999998, 'Lymph_Nodes': 0.031, 'Muscle': 0.041, 'Oesophagus': 0.054, 'Oral_Mucosa': 0.001, 'Ovaries': 0.001, 'Pancreas': 0.016, 'Prostate': 0.0, 'Salivary_Glands': 0.005, 'Skin': 0.08600000000000001, 'Small_Intestine': 0.001, 'Spleen': 0.003, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.047, 'Thyroid': 0.06999999999999999, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.057, 'Adrenals': 0.034, 'Brain': 0.001, 'Breast': 0.305, 'Colon': 0.002, 'Endosteum': 0.11699999999999999, 'Extrathoracic_Airways': 0.026, 'Gallbladder': 0.027, 'Heart': 0.058, 'Kidneys': 0.034, 'Liver': 0.212, 'Lungs': 0.183, 'Lymph_Nodes': 0.034, 'Muscle': 0.044, 'Oesophagus': 0.059, 'Oral_Mucosa': 0.001, 'Ovaries': 0.001, 'Pancreas': 0.018000000000000002, 'Prostate': 0.0, 'Salivary_Glands': 0.005, 'Skin': 0.088, 'Small_Intestine': 0.002, 'Spleen': 0.004, 'Stomach': 0.004, 'Testicles': 0.0, 'Thymus': 0.051, 'Thyroid': 0.076, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
            "Lumbar Spine": {
                "U": {
                    (120, 2.5, 0.0): {'Active_Bone_Marrow': 0.12999999999999998, 'Adrenals': 0.16, 'Brain': 0.0, 'Breast': 0.014, 'Colon': 0.32, 'Endosteum': 0.114, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.456, 'Heart': 0.036000000000000004, 'Kidneys': 0.34299999999999997, 'Liver': 0.93, 'Lungs': 0.045000000000000005, 'Lymph_Nodes': 0.15100000000000002, 'Muscle': 0.12999999999999998, 'Oesophagus': 0.044, 'Oral_Mucosa': 0.0, 'Ovaries': 0.259, 'Pancreas': 0.10500000000000001, 'Prostate': 0.023, 'Salivary_Glands': 0.0, 'Skin': 0.153, 'Small_Intestine': 0.29500000000000004, 'Spleen': 0.030000000000000002, 'Stomach': 0.051, 'Testicles': 0.004, 'Thymus': 0.01, 'Thyroid': 0.001, 'Urinary_Bladder': 0.06499999999999999, 'Uterus': 0.17300000000000001},
                    (120, 3.0, 0.0): {'Active_Bone_Marrow': 0.13799999999999998, 'Adrenals': 0.17200000000000001, 'Brain': 0.0, 'Breast': 0.015000000000000001, 'Colon': 0.34, 'Endosteum': 0.12000000000000001, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.48500000000000004, 'Heart': 0.039, 'Kidneys': 0.364, 'Liver': 0.9779999999999999, 'Lungs': 0.048, 'Lymph_Nodes': 0.16, 'Muscle': 0.136, 'Oesophagus': 0.047, 'Oral_Mucosa': 0.0, 'Ovaries': 0.27599999999999997, 'Pancreas': 0.11299999999999999, 'Prostate': 0.025, 'Salivary_Glands': 0.0, 'Skin': 0.156, 'Small_Intestine': 0.314, 'Spleen': 0.032, 'Stomach': 0.055, 'Testicles': 0.004, 'Thymus': 0.011, 'Thyroid': 0.001, 'Urinary_Bladder': 0.06999999999999999, 'Uterus': 0.185},
                    (120, 3.5, 0.0): {'Active_Bone_Marrow': 0.145, 'Adrenals': 0.182, 'Brain': 0.0, 'Breast': 0.016, 'Colon': 0.358, 'Endosteum': 0.125, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.512, 'Heart': 0.041, 'Kidneys': 0.383, 'Liver': 1.018, 'Lungs': 0.051, 'Lymph_Nodes': 0.16899999999999998, 'Muscle': 0.14200000000000002, 'Oesophagus': 0.05, 'Oral_Mucosa': 0.0, 'Ovaries': 0.292, 'Pancreas': 0.12000000000000001, 'Prostate': 0.026, 'Salivary_Glands': 0.0, 'Skin': 0.158, 'Small_Intestine': 0.331, 'Spleen': 0.034, 'Stomach': 0.059, 'Testicles': 0.004, 'Thymus': 0.011, 'Thyroid': 0.001, 'Urinary_Bladder': 0.074, 'Uterus': 0.196},
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.106, 'Adrenals': 0.12000000000000001, 'Brain': 0.0, 'Breast': 0.011, 'Colon': 0.263, 'Endosteum': 0.1, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.36900000000000005, 'Heart': 0.027, 'Kidneys': 0.281, 'Liver': 0.8140000000000001, 'Lungs': 0.037, 'Lymph_Nodes': 0.121, 'Muscle': 0.114, 'Oesophagus': 0.032, 'Oral_Mucosa': 0.0, 'Ovaries': 0.203, 'Pancreas': 0.079, 'Prostate': 0.016, 'Salivary_Glands': 0.0, 'Skin': 0.145, 'Small_Intestine': 0.237, 'Spleen': 0.020999999999999998, 'Stomach': 0.036000000000000004, 'Testicles': 0.003, 'Thymus': 0.007, 'Thyroid': 0.001, 'Urinary_Bladder': 0.049, 'Uterus': 0.134},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.114, 'Adrenals': 0.12999999999999998, 'Brain': 0.0, 'Breast': 0.012, 'Colon': 0.283, 'Endosteum': 0.107, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.398, 'Heart': 0.030000000000000002, 'Kidneys': 0.303, 'Liver': 0.864, 'Lungs': 0.04, 'Lymph_Nodes': 0.12999999999999998, 'Muscle': 0.121, 'Oesophagus': 0.034999999999999996, 'Oral_Mucosa': 0.0, 'Ovaries': 0.22, 'Pancreas': 0.08600000000000001, 'Prostate': 0.017, 'Salivary_Glands': 0.0, 'Skin': 0.148, 'Small_Intestine': 0.255, 'Spleen': 0.023, 'Stomach': 0.039, 'Testicles': 0.003, 'Thymus': 0.008, 'Thyroid': 0.001, 'Urinary_Bladder': 0.053, 'Uterus': 0.146},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.121, 'Adrenals': 0.13999999999999999, 'Brain': 0.0, 'Breast': 0.013, 'Colon': 0.301, 'Endosteum': 0.11299999999999999, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.425, 'Heart': 0.032, 'Kidneys': 0.322, 'Liver': 0.9079999999999999, 'Lungs': 0.041999999999999996, 'Lymph_Nodes': 0.13899999999999998, 'Muscle': 0.126, 'Oesophagus': 0.037, 'Oral_Mucosa': 0.0, 'Ovaries': 0.235, 'Pancreas': 0.093, 'Prostate': 0.019, 'Salivary_Glands': 0.0, 'Skin': 0.15100000000000002, 'Small_Intestine': 0.272, 'Spleen': 0.025, 'Stomach': 0.041999999999999996, 'Testicles': 0.003, 'Thymus': 0.008, 'Thyroid': 0.001, 'Urinary_Bladder': 0.057, 'Uterus': 0.156},
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.078, 'Adrenals': 0.078, 'Brain': 0.0, 'Breast': 0.008, 'Colon': 0.195, 'Endosteum': 0.081, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.265, 'Heart': 0.017, 'Kidneys': 0.207, 'Liver': 0.664, 'Lungs': 0.026, 'Lymph_Nodes': 0.085, 'Muscle': 0.095, 'Oesophagus': 0.020999999999999998, 'Oral_Mucosa': 0.0, 'Ovaries': 0.146, 'Pancreas': 0.05, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.0, 'Skin': 0.136, 'Small_Intestine': 0.16799999999999998, 'Spleen': 0.011, 'Stomach': 0.020999999999999998, 'Testicles': 0.002, 'Thymus': 0.005, 'Thyroid': 0.001, 'Urinary_Bladder': 0.032, 'Uterus': 0.089},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.085, 'Adrenals': 0.08600000000000001, 'Brain': 0.0, 'Breast': 0.009000000000000001, 'Colon': 0.213, 'Endosteum': 0.088, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.29100000000000004, 'Heart': 0.019, 'Kidneys': 0.22699999999999998, 'Liver': 0.713, 'Lungs': 0.029, 'Lymph_Nodes': 0.094, 'Muscle': 0.101, 'Oesophagus': 0.023, 'Oral_Mucosa': 0.0, 'Ovaries': 0.161, 'Pancreas': 0.055, 'Prostate': 0.01, 'Salivary_Glands': 0.0, 'Skin': 0.13999999999999999, 'Small_Intestine': 0.184, 'Spleen': 0.013, 'Stomach': 0.023, 'Testicles': 0.002, 'Thymus': 0.005, 'Thyroid': 0.001, 'Urinary_Bladder': 0.034999999999999996, 'Uterus': 0.09899999999999999},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.091, 'Adrenals': 0.094, 'Brain': 0.0, 'Breast': 0.009000000000000001, 'Colon': 0.23, 'Endosteum': 0.094, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.314, 'Heart': 0.02, 'Kidneys': 0.244, 'Liver': 0.757, 'Lungs': 0.031, 'Lymph_Nodes': 0.101, 'Muscle': 0.106, 'Oesophagus': 0.025, 'Oral_Mucosa': 0.0, 'Ovaries': 0.175, 'Pancreas': 0.060000000000000005, 'Prostate': 0.011, 'Salivary_Glands': 0.0, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.199, 'Spleen': 0.014, 'Stomach': 0.025, 'Testicles': 0.002, 'Thymus': 0.006, 'Thyroid': 0.001, 'Urinary_Bladder': 0.038, 'Uterus': 0.108},
                    (60, 2.5, 0.0): {'Active_Bone_Marrow': 0.048, 'Adrenals': 0.032, 'Brain': 0.0, 'Breast': 0.005, 'Colon': 0.12000000000000001, 'Endosteum': 0.057, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.154, 'Heart': 0.008, 'Kidneys': 0.126, 'Liver': 0.486, 'Lungs': 0.014, 'Lymph_Nodes': 0.049, 'Muscle': 0.07200000000000001, 'Oesophagus': 0.008, 'Oral_Mucosa': 0.0, 'Ovaries': 0.079, 'Pancreas': 0.023, 'Prostate': 0.005, 'Salivary_Glands': 0.0, 'Skin': 0.125, 'Small_Intestine': 0.096, 'Spleen': 0.003, 'Stomach': 0.008, 'Testicles': 0.001, 'Thymus': 0.003, 'Thyroid': 0.0, 'Urinary_Bladder': 0.014, 'Uterus': 0.044},
                    (60, 3.0, 0.0): {'Active_Bone_Marrow': 0.053, 'Adrenals': 0.036000000000000004, 'Brain': 0.0, 'Breast': 0.006, 'Colon': 0.133, 'Endosteum': 0.063, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.17200000000000001, 'Heart': 0.009000000000000001, 'Kidneys': 0.13999999999999999, 'Liver': 0.528, 'Lungs': 0.016, 'Lymph_Nodes': 0.055, 'Muscle': 0.077, 'Oesophagus': 0.01, 'Oral_Mucosa': 0.0, 'Ovaries': 0.089, 'Pancreas': 0.026, 'Prostate': 0.006, 'Salivary_Glands': 0.0, 'Skin': 0.129, 'Small_Intestine': 0.107, 'Spleen': 0.004, 'Stomach': 0.009000000000000001, 'Testicles': 0.001, 'Thymus': 0.003, 'Thyroid': 0.0, 'Urinary_Bladder': 0.016, 'Uterus': 0.049},
                    (60, 3.5, 0.0): {'Active_Bone_Marrow': 0.058, 'Adrenals': 0.04, 'Brain': 0.0, 'Breast': 0.006, 'Colon': 0.146, 'Endosteum': 0.067, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.188, 'Heart': 0.01, 'Kidneys': 0.153, 'Liver': 0.565, 'Lungs': 0.018000000000000002, 'Lymph_Nodes': 0.060000000000000005, 'Muscle': 0.082, 'Oesophagus': 0.011, 'Oral_Mucosa': 0.0, 'Ovaries': 0.09899999999999999, 'Pancreas': 0.029, 'Prostate': 0.006, 'Salivary_Glands': 0.0, 'Skin': 0.132, 'Small_Intestine': 0.118, 'Spleen': 0.004, 'Stomach': 0.011, 'Testicles': 0.001, 'Thymus': 0.004, 'Thyroid': 0.001, 'Urinary_Bladder': 0.018000000000000002, 'Uterus': 0.055},
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.087, "Adrenals": 0.049, "Brain": 0.0, "Breast": 0.511, "Colon": 0.002, "Endosteum": 0.157, "Extrathoracic_Airways": 0.022, "Gallbladder": 0.036, "Gonads": 0.0, "Heart": 0.688, "Kidneys": 0.015, "Liver": 0.191, "Lungs": 0.299, "Lymph_Nodes": 0.094, "Muscle": 0.076, "Oesophagus": 0.127, "Oral_Mucosa": 0.003, "Pancreas": 0.139, "Prostate": 0.0, "Salivary_Glands": 0.005, "Skin": 0.106, "Small_Intestine": 0.002, "Spleen": 0.05, "Stomach": 0.178, "Thymus": 1.343, "Thyroid": 0.036, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.093, "Adrenals": 0.056, "Brain": 0.001, "Breast": 0.537, "Colon": 0.002, "Endosteum": 0.167, "Extrathoracic_Airways": 0.024, "Gallbladder": 0.041, "Gonads": 0.0, "Heart": 0.748, "Kidneys": 0.017, "Liver": 0.209, "Lungs": 0.328, "Lymph_Nodes": 0.103, "Muscle": 0.082, "Oesophagus": 0.142, "Oral_Mucosa": 0.003, "Pancreas": 0.155, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.108, "Small_Intestine": 0.003, "Spleen": 0.057, "Stomach": 0.195, "Thymus": 1.45, "Thyroid": 0.04, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.098, "Adrenals": 0.062, "Brain": 0.001, "Breast": 0.56, "Colon": 0.002, "Endosteum": 0.175, "Extrathoracic_Airways": 0.026, "Gallbladder": 0.045, "Gonads": 0.0, "Heart": 0.802, "Kidneys": 0.019, "Liver": 0.226, "Lungs": 0.354, "Lymph_Nodes": 0.112, "Muscle": 0.088, "Oesophagus": 0.156, "Oral_Mucosa": 0.004, "Pancreas": 0.169, "Prostate": 0.0, "Salivary_Glands": 0.006, "Skin": 0.11, "Small_Intestine": 0.003, "Spleen": 0.062, "Stomach": 0.211, "Thymus": 1.543, "Thyroid": 0.044, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.113, "Adrenals": 0.098, "Brain": 0.001, "Breast": 0.596, "Colon": 0.004, "Endosteum": 0.191, "Extrathoracic_Airways": 0.036, "Gallbladder": 0.071, "Gonads": 0.0, "Heart": 0.935, "Kidneys": 0.034, "Liver": 0.276, "Lungs": 0.429, "Lymph_Nodes": 0.143, "Muscle": 0.104, "Oesophagus": 0.23, "Oral_Mucosa": 0.006, "Pancreas": 0.233, "Prostate": 0.0, "Salivary_Glands": 0.01, "Skin": 0.116, "Small_Intestine": 0.005, "Spleen": 0.089, "Stomach": 0.255, "Thymus": 1.675, "Thyroid": 0.067, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.12, "Adrenals": 0.109, "Brain": 0.001, "Breast": 0.623, "Colon": 0.005, "Endosteum": 0.201, "Extrathoracic_Airways": 0.039, "Gallbladder": 0.078, "Gonads": 0.0, "Heart": 1.005, "Kidneys": 0.038, "Liver": 0.299, "Lungs": 0.464, "Lymph_Nodes": 0.155, "Muscle": 0.111, "Oesophagus": 0.252, "Oral_Mucosa": 0.007, "Pancreas": 0.255, "Prostate": 0.0, "Salivary_Glands": 0.012, "Skin": 0.119, "Small_Intestine": 0.006, "Spleen": 0.097, "Stomach": 0.277, "Thymus": 1.785, "Thyroid": 0.074, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.126, "Adrenals": 0.118, "Brain": 0.001, "Breast": 0.647, "Colon": 0.005, "Endosteum": 0.21, "Extrathoracic_Airways": 0.042, "Gallbladder": 0.085, "Gonads": 0.0, "Heart": 1.067, "Kidneys": 0.041, "Liver": 0.319, "Lungs": 0.496, "Lymph_Nodes": 0.166, "Muscle": 0.118, "Oesophagus": 0.273, "Oral_Mucosa": 0.008, "Pancreas": 0.276, "Prostate": 0.0, "Salivary_Glands": 0.013, "Skin": 0.122, "Small_Intestine": 0.007, "Spleen": 0.106, "Stomach": 0.295, "Thymus": 1.879, "Thyroid": 0.081, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.138, "Adrenals": 0.14, "Brain": 0.001, "Breast": 0.662, "Colon": 0.007, "Endosteum": 0.214, "Extrathoracic_Airways": 0.046, "Gallbladder": 0.101, "Gonads": 0.0, "Heart": 1.141, "Kidneys": 0.053, "Liver": 0.35, "Lungs": 0.538, "Lymph_Nodes": 0.186, "Muscle": 0.128, "Oesophagus": 0.325, "Oral_Mucosa": 0.009, "Pancreas": 0.32, "Prostate": 0.0, "Salivary_Glands": 0.016, "Skin": 0.125, "Small_Intestine": 0.009, "Spleen": 0.125, "Stomach": 0.324, "Thymus": 1.943, "Thyroid": 0.098, "Urinary_Bladder": 0.0, "Uterus": 0.0005},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.145, "Adrenals": 0.152, "Brain": 0.002, "Breast": 0.688, "Colon": 0.008, "Endosteum": 0.224, "Extrathoracic_Airways": 0.05, "Gallbladder": 0.109, "Gonads": 0.0, "Heart": 1.211, "Kidneys": 0.058, "Liver": 0.374, "Lungs": 0.574, "Lymph_Nodes": 0.2, "Muscle": 0.136, "Oesophagus": 0.352, "Oral_Mucosa": 0.01, "Pancreas": 0.345, "Prostate": 0.0, "Salivary_Glands": 0.017, "Skin": 0.128, "Small_Intestine": 0.01, "Spleen": 0.135, "Stomach": 0.346, "Thymus": 2.047, "Thyroid": 0.106, "Urinary_Bladder": 0.001, "Uterus": 0.0005},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.151, "Adrenals": 0.163, "Brain": 0.002, "Breast": 0.71, "Colon": 0.009, "Endosteum": 0.232, "Extrathoracic_Airways": 0.053, "Gallbladder": 0.117, "Gonads": 0.0, "Heart": 1.273, "Kidneys": 0.062, "Liver": 0.394, "Lungs": 0.606, "Lymph_Nodes": 0.211, "Muscle": 0.143, "Oesophagus": 0.375, "Oral_Mucosa": 0.011, "Pancreas": 0.368, "Prostate": 0.0, "Salivary_Glands": 0.019, "Skin": 0.131, "Small_Intestine": 0.011, "Spleen": 0.145, "Stomach": 0.365, "Thymus": 2.135, "Thyroid": 0.114, "Urinary_Bladder": 0.001, "Uterus": 0.0005},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.157, "Adrenals": 0.177, "Brain": 0.002, "Breast": 0.708, "Colon": 0.01, "Endosteum": 0.229, "Extrathoracic_Airways": 0.055, "Gallbladder": 0.126, "Gonads": 0.0005, "Heart": 1.299, "Kidneys": 0.07, "Liver": 0.408, "Lungs": 0.624, "Lymph_Nodes": 0.221, "Muscle": 0.147, "Oesophagus": 0.403, "Oral_Mucosa": 0.012, "Pancreas": 0.391, "Prostate": 0.0, "Salivary_Glands": 0.02, "Skin": 0.133, "Small_Intestine": 0.012, "Spleen": 0.157, "Stomach": 0.375, "Thymus": 2.149, "Thyroid": 0.124, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.164, "Adrenals": 0.192, "Brain": 0.002, "Breast": 0.73, "Colon": 0.011, "Endosteum": 0.238, "Extrathoracic_Airways": 0.058, "Gallbladder": 0.134, "Gonads": 0.0005, "Heart": 1.364, "Kidneys": 0.075, "Liver": 0.431, "Lungs": 0.658, "Lymph_Nodes": 0.234, "Muscle": 0.154, "Oesophagus": 0.427, "Oral_Mucosa": 0.013, "Pancreas": 0.417, "Prostate": 0.0, "Salivary_Glands": 0.02, "Skin": 0.135, "Small_Intestine": 0.013, "Spleen": 0.168, "Stomach": 0.395, "Thymus": 2.243, "Thyroid": 0.133, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.171, "Adrenals": 0.203, "Brain": 0.002, "Breast": 0.749, "Colon": 0.012, "Endosteum": 0.245, "Extrathoracic_Airways": 0.061, "Gallbladder": 0.142, "Gonads": 0.0005, "Heart": 1.42, "Kidneys": 0.08, "Liver": 0.451, "Lungs": 0.687, "Lymph_Nodes": 0.245, "Muscle": 0.161, "Oesophagus": 0.45, "Oral_Mucosa": 0.014, "Pancreas": 0.439, "Prostate": 0.0, "Salivary_Glands": 0.022, "Skin": 0.138, "Small_Intestine": 0.014, "Spleen": 0.177, "Stomach": 0.413, "Thymus": 2.321, "Thyroid": 0.14, "Urinary_Bladder": 0.001, "Uterus": 0.001},
                },
            },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.0, "Brain": 0.028, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.19, "Extrathoracic_Airways": 1.105, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.333, "Muscle": 0.042, "Oesophagus": 0.029, "Oral_Mucosa": 1.14, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.684, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.008, "Thyroid": 2.587, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.043, "Adrenals": 0.0, "Brain": 0.032, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.207, "Extrathoracic_Airways": 1.194, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.012, "Lymph_Nodes": 0.357, "Muscle": 0.045, "Oesophagus": 0.032, "Oral_Mucosa": 1.215, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.744, "Skin": 0.155, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.009, "Thyroid": 2.725, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.0, "Brain": 0.036, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.222, "Extrathoracic_Airways": 1.273, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.013, "Lymph_Nodes": 0.377, "Muscle": 0.048, "Oesophagus": 0.035, "Oral_Mucosa": 1.281, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.798, "Skin": 0.159, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.01, "Thyroid": 2.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.06, "Adrenals": 0.0, "Brain": 0.061, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.258, "Extrathoracic_Airways": 1.474, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.019, "Lymph_Nodes": 0.422, "Muscle": 0.054, "Oesophagus": 0.044, "Oral_Mucosa": 1.447, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.939, "Skin": 0.164, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.018, "Thyroid": 2.96, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.0, "Brain": 0.068, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.278, "Extrathoracic_Airways": 1.577, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.448, "Muscle": 0.057, "Oesophagus": 0.048, "Oral_Mucosa": 1.534, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.01, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.02, "Thyroid": 3.092, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.069, "Adrenals": 0.0, "Brain": 0.074, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.295, "Extrathoracic_Airways": 1.669, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.471, "Muscle": 0.06, "Oesophagus": 0.051, "Oral_Mucosa": 1.609, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.073, "Skin": 0.172, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.021, "Thyroid": 3.204, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.0, "Brain": 0.094, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.308, "Extrathoracic_Airways": 1.77, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.496, "Muscle": 0.063, "Oesophagus": 0.06, "Oral_Mucosa": 1.707, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.167, "Skin": 0.175, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.025, "Thyroid": 3.259, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.0, "Brain": 0.103, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.327, "Extrathoracic_Airways": 1.873, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.028, "Lymph_Nodes": 0.522, "Muscle": 0.067, "Oesophagus": 0.064, "Oral_Mucosa": 1.794, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.241, "Skin": 0.179, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.027, "Thyroid": 3.38, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.0, "Brain": 0.11, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.344, "Extrathoracic_Airways": 1.963, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.03, "Lymph_Nodes": 0.544, "Muscle": 0.069, "Oesophagus": 0.068, "Oral_Mucosa": 1.87, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.306, "Skin": 0.183, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.029, "Thyroid": 3.481, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.094, "Adrenals": 0.0, "Brain": 0.123, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.342, "Extrathoracic_Airways": 1.991, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.007, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.031, "Lymph_Nodes": 0.552, "Muscle": 0.071, "Oesophagus": 0.074, "Oral_Mucosa": 1.911, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.353, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.03, "Thyroid": 3.473, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.1, "Adrenals": 0.0, "Brain": 0.131, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.358, "Extrathoracic_Airways": 2.085, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.033, "Lymph_Nodes": 0.576, "Muscle": 0.074, "Oesophagus": 0.078, "Oral_Mucosa": 1.992, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.424, "Skin": 0.188, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.032, "Thyroid": 3.577, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.104, "Adrenals": 0.0, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.373, "Extrathoracic_Airways": 2.167, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.035, "Lymph_Nodes": 0.596, "Muscle": 0.076, "Oesophagus": 0.082, "Oral_Mucosa": 2.062, "Pancreas": 0.002, "Prostate": 0.0, "Salivary_Glands": 1.486, "Skin": 0.191, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.034, "Thyroid": 3.664, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.054, "Adrenals": 0.0, "Brain": 0.42, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.279, "Extrathoracic_Airways": 0.214, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.001, "Lymph_Nodes": 0.098, "Muscle": 0.022, "Oesophagus": 0.0, "Oral_Mucosa": 0.516, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.827, "Skin": 0.113, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.021, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.058, "Adrenals": 0.0, "Brain": 0.464, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.302, "Extrathoracic_Airways": 0.236, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.002, "Lymph_Nodes": 0.105, "Muscle": 0.023, "Oesophagus": 0.0, "Oral_Mucosa": 0.555, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.871, "Skin": 0.116, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.023, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.063, "Adrenals": 0.0, "Brain": 0.505, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.322, "Extrathoracic_Airways": 0.257, "Gallbladder": 0.0, "Heart": 0.0, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.002, "Lymph_Nodes": 0.111, "Muscle": 0.024, "Oesophagus": 0.0, "Oral_Mucosa": 0.59, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.908, "Skin": 0.118, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.025, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.076, "Adrenals": 0.0, "Brain": 0.647, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.36, "Extrathoracic_Airways": 0.329, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.128, "Muscle": 0.027, "Oesophagus": 0.0, "Oral_Mucosa": 0.697, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.978, "Skin": 0.123, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.047, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.082, "Adrenals": 0.0, "Brain": 0.705, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.385, "Extrathoracic_Airways": 0.358, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.136, "Muscle": 0.028, "Oesophagus": 0.001, "Oral_Mucosa": 0.745, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.025, "Skin": 0.126, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.052, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.087, "Adrenals": 0.0, "Brain": 0.757, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.407, "Extrathoracic_Airways": 0.384, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.003, "Lymph_Nodes": 0.143, "Muscle": 0.029, "Oesophagus": 0.001, "Oral_Mucosa": 0.788, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.065, "Skin": 0.128, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.0, "Thyroid": 0.056, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.095, "Adrenals": 0.0, "Brain": 0.845, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.417, "Extrathoracic_Airways": 0.439, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.004, "Lymph_Nodes": 0.155, "Muscle": 0.031, "Oesophagus": 0.001, "Oral_Mucosa": 0.868, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.112, "Skin": 0.132, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.001, "Thyroid": 0.066, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.102, "Adrenals": 0.0, "Brain": 0.906, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.44, "Extrathoracic_Airways": 0.471, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.004, "Lymph_Nodes": 0.163, "Muscle": 0.032, "Oesophagus": 0.001, "Oral_Mucosa": 0.92, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.158, "Skin": 0.134, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.001, "Thyroid": 0.071, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.107, "Adrenals": 0.0, "Brain": 0.961, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.46, "Extrathoracic_Airways": 0.5, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.004, "Lymph_Nodes": 0.171, "Muscle": 0.034, "Oesophagus": 0.001, "Oral_Mucosa": 0.966, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.198, "Skin": 0.137, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.001, "Thyroid": 0.076, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.111, "Adrenals": 0.0, "Brain": 1.004, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.453, "Extrathoracic_Airways": 0.534, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.005, "Lymph_Nodes": 0.177, "Muscle": 0.035, "Oesophagus": 0.002, "Oral_Mucosa": 1.006, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.218, "Skin": 0.139, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.079, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.117, "Adrenals": 0.0, "Brain": 1.064, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.473, "Extrathoracic_Airways": 0.566, "Gallbladder": 0.0, "Heart": 0.001, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.005, "Lymph_Nodes": 0.185, "Muscle": 0.036, "Oesophagus": 0.002, "Oral_Mucosa": 1.056, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.261, "Skin": 0.141, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.084, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.122, "Adrenals": 0.0, "Brain": 1.116, "Breast": 0.002, "Colon": 0.0, "Endosteum": 0.49, "Extrathoracic_Airways": 0.595, "Gallbladder": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.005, "Lymph_Nodes": 0.192, "Muscle": 0.037, "Oesophagus": 0.002, "Oral_Mucosa": 1.1, "Ovaries": 0.0, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 1.298, "Skin": 0.144, "Small_Intestine": 0.0, "Spleen": 0.0, "Stomach": 0.0, "Testicles": 0.0, "Thymus": 0.002, "Thyroid": 0.089, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Scoliosis": {
                "U": {
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.136, "Adrenals": 0.135, "Brain": 0.006, "Breast": 0.248, "Colon": 0.127, "Extrathoracic_Airways": 0.241, "Gallbladder": 0.249, "Heart": 0.165, "Kidney": 0.18, "Liver": 0.515, "Lungs": 0.307, "Lymph_Nodes": 0.171, "Muscle": 0.112, "Oesophagus": 0.168, "Oral_Mucosa": 0.041, "Ovary": 0.04, "Pancreas": 0.084, "Prostate": 0.004, "Salivary_Glands": 0.134, "Skeleton": 0.204, "Skin": 0.158, "Small_Intestine": 0.125, "Spleen": 0.022, "Stomach": 0.035, "Testicles": 0.001, "Thymus": 0.149, "Thyroid": 0.348, "Bladder": 0.013, "Uterus": 0.036, "Average_Total_Body": 0.138},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.143, "Adrenals": 0.144, "Brain": 0.007, "Breast": 0.258, "Colon": 0.135, "Extrathoracic_Airways": 0.254, "Gallbladder": 0.266, "Heart": 0.176, "Kidney": 0.191, "Liver": 0.542, "Lungs": 0.324, "Lymph_Nodes": 0.181, "Muscle": 0.117, "Oesophagus": 0.18, "Oral_Mucosa": 0.044, "Ovary": 0.043, "Pancreas": 0.09, "Prostate": 0.005, "Salivary_Glands": 0.141, "Skeleton": 0.213, "Skin": 0.162, "Small_Intestine": 0.133, "Spleen": 0.023, "Stomach": 0.037, "Testicles": 0.001, "Thymus": 0.159, "Thyroid": 0.364, "Bladder": 0.014, "Uterus": 0.039, "Average_Total_Body": 0.144},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.15, "Adrenals": 0.153, "Brain": 0.007, "Breast": 0.266, "Colon": 0.142, "Extrathoracic_Airways": 0.264, "Gallbladder": 0.281, "Heart": 0.186, "Kidney": 0.201, "Liver": 0.565, "Lungs": 0.339, "Lymph_Nodes": 0.19, "Muscle": 0.122, "Oesophagus": 0.19, "Oral_Mucosa": 0.046, "Ovary": 0.045, "Pancreas": 0.096, "Prostate": 0.005, "Salivary_Glands": 0.148, "Skeleton": 0.221, "Skin": 0.164, "Small_Intestine": 0.141, "Spleen": 0.025, "Stomach": 0.04, "Testicles": 0.001, "Thymus": 0.168, "Thyroid": 0.376, "Bladder": 0.015, "Uterus": 0.042, "Average_Total_Body": 0.15},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.114, "Adrenals": 0.107, "Brain": 0.005, "Breast": 0.226, "Colon": 0.104, "Extrathoracic_Airways": 0.214, "Gallbladder": 0.201, "Heart": 0.132, "Kidney": 0.149, "Liver": 0.449, "Lungs": 0.264, "Lymph_Nodes": 0.143, "Muscle": 0.098, "Oesophagus": 0.132, "Oral_Mucosa": 0.031, "Ovary": 0.033, "Pancreas": 0.065, "Prostate": 0.002, "Salivary_Glands": 0.113, "Skeleton": 0.186, "Skin": 0.15, "Small_Intestine": 0.101, "Spleen": 0.015, "Stomach": 0.025, "Testicles": 0.001, "Thymus": 0.116, "Thyroid": 0.314, "Bladder": 0.011, "Uterus": 0.025, "Average_Total_Body": 0.121},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.122, "Adrenals": 0.116, "Brain": 0.005, "Breast": 0.236, "Colon": 0.112, "Extrathoracic_Airways": 0.227, "Gallbladder": 0.218, "Heart": 0.142, "Kidney": 0.16, "Liver": 0.478, "Lungs": 0.282, "Lymph_Nodes": 0.154, "Muscle": 0.103, "Oesophagus": 0.143, "Oral_Mucosa": 0.034, "Ovary": 0.036, "Pancreas": 0.07, "Prostate": 0.003, "Salivary_Glands": 0.121, "Skeleton": 0.196, "Skin": 0.154, "Small_Intestine": 0.108, "Spleen": 0.016, "Stomach": 0.027, "Testicles": 0.001, "Thymus": 0.125, "Thyroid": 0.331, "Bladder": 0.012, "Uterus": 0.028, "Average_Total_Body": 0.128},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.129, "Adrenals": 0.125, "Brain": 0.005, "Breast": 0.245, "Colon": 0.119, "Extrathoracic_Airways": 0.239, "Gallbladder": 0.233, "Heart": 0.152, "Kidney": 0.17, "Liver": 0.503, "Lungs": 0.298, "Lymph_Nodes": 0.163, "Muscle": 0.108, "Oesophagus": 0.153, "Oral_Mucosa": 0.036, "Ovary": 0.038, "Pancreas": 0.076, "Prostate": 0.003, "Salivary_Glands": 0.127, "Skeleton": 0.205, "Skin": 0.157, "Small_Intestine": 0.116, "Spleen": 0.018, "Stomach": 0.029, "Testicles": 0.001, "Thymus": 0.134, "Thyroid": 0.345, "Bladder": 0.013, "Uterus": 0.03, "Average_Total_Body": 0.134},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.088, "Adrenals": 0.073, "Brain": 0.003, "Breast": 0.198, "Colon": 0.075, "Extrathoracic_Airways": 0.176, "Gallbladder": 0.143, "Heart": 0.091, "Kidney": 0.109, "Liver": 0.364, "Lungs": 0.209, "Lymph_Nodes": 0.109, "Muscle": 0.081, "Oesophagus": 0.091, "Oral_Mucosa": 0.02, "Ovary": 0.023, "Pancreas": 0.041, "Prostate": 0.001, "Salivary_Glands": 0.088, "Skeleton": 0.159, "Skin": 0.14, "Small_Intestine": 0.071, "Spleen": 0.008, "Stomach": 0.015, "Testicles": 0.001, "Thymus": 0.074, "Thyroid": 0.277, "Bladder": 0.007, "Uterus": 0.014, "Average_Total_Body": 0.1},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.095, "Adrenals": 0.08, "Brain": 0.003, "Breast": 0.209, "Colon": 0.082, "Extrathoracic_Airways": 0.189, "Gallbladder": 0.157, "Heart": 0.1, "Kidney": 0.119, "Liver": 0.391, "Lungs": 0.226, "Lymph_Nodes": 0.119, "Muscle": 0.086, "Oesophagus": 0.1, "Oral_Mucosa": 0.022, "Ovary": 0.025, "Pancreas": 0.045, "Prostate": 0.002, "Salivary_Glands": 0.096, "Skeleton": 0.17, "Skin": 0.144, "Small_Intestine": 0.078, "Spleen": 0.009, "Stomach": 0.016, "Testicles": 0.001, "Thymus": 0.082, "Thyroid": 0.295, "Bladder": 0.008, "Uterus": 0.016, "Average_Total_Body": 0.107},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.101, "Adrenals": 0.087, "Brain": 0.003, "Breast": 0.218, "Colon": 0.089, "Extrathoracic_Airways": 0.201, "Gallbladder": 0.17, "Heart": 0.109, "Kidney": 0.128, "Liver": 0.416, "Lungs": 0.242, "Lymph_Nodes": 0.127, "Muscle": 0.091, "Oesophagus": 0.109, "Oral_Mucosa": 0.024, "Ovary": 0.028, "Pancreas": 0.049, "Prostate": 0.002, "Salivary_Glands": 0.102, "Skeleton": 0.179, "Skin": 0.147, "Small_Intestine": 0.084, "Spleen": 0.01, "Stomach": 0.018, "Testicles": 0.001, "Thymus": 0.089, "Thyroid": 0.311, "Bladder": 0.008, "Uterus": 0.017, "Average_Total_Body": 0.113},
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.061, "Adrenals": 0.036, "Brain": 0.001, "Breast": 0.164, "Colon": 0.046, "Extrathoracic_Airways": 0.126, "Gallbladder": 0.077, "Heart": 0.05, "Kidney": 0.068, "Liver": 0.262, "Lungs": 0.145, "Lymph_Nodes": 0.071, "Muscle": 0.061, "Oesophagus": 0.045, "Oral_Mucosa": 0.011, "Ovary": 0.014, "Pancreas": 0.019, "Prostate": 0.001, "Salivary_Glands": 0.06, "Skeleton": 0.122, "Skin": 0.128, "Small_Intestine": 0.042, "Spleen": 0.003, "Stomach": 0.007, "Testicles": 0.0, "Thymus": 0.035, "Thyroid": 0.218, "Bladder": 0.003, "Uterus": 0.006, "Average_Total_Body": 0.076},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.066, "Adrenals": 0.04, "Brain": 0.001, "Breast": 0.174, "Colon": 0.051, "Extrathoracic_Airways": 0.138, "Gallbladder": 0.087, "Heart": 0.056, "Kidney": 0.075, "Liver": 0.285, "Lungs": 0.159, "Lymph_Nodes": 0.078, "Muscle": 0.066, "Oesophagus": 0.051, "Oral_Mucosa": 0.012, "Ovary": 0.015, "Pancreas": 0.022, "Prostate": 0.001, "Salivary_Glands": 0.066, "Skeleton": 0.132, "Skin": 0.131, "Small_Intestine": 0.046, "Spleen": 0.004, "Stomach": 0.008, "Testicles": 0.0, "Thymus": 0.039, "Thyroid": 0.236, "Bladder": 0.003, "Uterus": 0.007, "Average_Total_Body": 0.082},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.071, "Adrenals": 0.044, "Brain": 0.001, "Breast": 0.183, "Colon": 0.056, "Extrathoracic_Airways": 0.148, "Gallbladder": 0.095, "Heart": 0.062, "Kidney": 0.081, "Liver": 0.306, "Lungs": 0.172, "Lymph_Nodes": 0.085, "Muscle": 0.069, "Oesophagus": 0.056, "Oral_Mucosa": 0.013, "Ovary": 0.017, "Pancreas": 0.024, "Prostate": 0.001, "Salivary_Glands": 0.071, "Skeleton": 0.14, "Skin": 0.134, "Small_Intestine": 0.051, "Spleen": 0.004, "Stomach": 0.008, "Testicles": 0.0, "Thymus": 0.043, "Thyroid": 0.251, "Bladder": 0.004, "Uterus": 0.007, "Average_Total_Body": 0.087},
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_ORGAN_REGION,
                "Chest": {
                    "U": {
                        (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.845, 'Adrenals': 2.356, 'Brain': 0.032, 'Breast': 4.289, 'Colon': 0.176, 'Extrathoracic_Airways': 0.187, 'Gallbladder': 1.409, 'Heart': 3.018, 'Kidneys': 1.463, 'Liver': 3.714, 'Lungs': 4.136, 'Lymph_Nodes': 1.2, 'Muscle': 1.058, 'Oesophagus': 3.25, 'Oral_Mucosa': 0.119, 'Ovaries': 0.087, 'Pancreas': 1.773, 'Prostate': 0.011, 'Salivary_Glands': 0.322, 'Endosteum': 3.039, 'Skin': 1.439, 'Small_Intestine': 0.209, 'Spleen': 0.834, 'Stomach': 1.07, 'Testicles': 0.002, 'Thymus': 2.781, 'Thyroid': 1.086, 'Urinary_Bladder': 0.034, 'Uterus': 0.063},
                        (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.9, 'Adrenals': 2.547, 'Brain': 0.036, 'Breast': 4.378, 'Colon': 0.194, 'Extrathoracic_Airways': 0.204, 'Gallbladder': 1.541, 'Heart': 3.256, 'Kidneys': 1.576, 'Liver': 3.939, 'Lungs': 4.383, 'Lymph_Nodes': 1.296, 'Muscle': 1.123, 'Oesophagus': 3.512, 'Oral_Mucosa': 0.132, 'Ovaries': 0.096, 'Pancreas': 1.941, 'Prostate': 0.013, 'Salivary_Glands': 0.348, 'Endosteum': 3.215, 'Skin': 1.473, 'Small_Intestine': 0.231, 'Spleen': 0.917, 'Stomach': 1.173, 'Testicles': 0.003, 'Thymus': 3.018, 'Thyroid': 1.171, 'Urinary_Bladder': 0.039, 'Uterus': 0.07},
                        (65, 3.0, 0.1): {'Active_Bone_Marrow': 1.154, 'Adrenals': 3.411, 'Brain': 0.055, 'Breast': 4.852, 'Colon': 0.284, 'Extrathoracic_Airways': 0.291, 'Gallbladder': 2.17, 'Heart': 4.344, 'Kidneys': 2.098, 'Liver': 4.905, 'Lungs': 5.438, 'Lymph_Nodes': 1.75, 'Muscle': 1.418, 'Oesophagus': 4.721, 'Oral_Mucosa': 0.2, 'Ovaries': 0.148, 'Pancreas': 2.755, 'Prostate': 0.025, 'Salivary_Glands': 0.462, 'Endosteum': 3.974, 'Skin': 1.625, 'Small_Intestine': 0.346, 'Spleen': 1.336, 'Stomach': 1.683, 'Testicles': 0.005, 'Thymus': 4.148, 'Thyroid': 1.587, 'Urinary_Bladder': 0.068, 'Uterus': 0.107},
                        (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.949, 'Adrenals': 2.715, 'Brain': 0.039, 'Breast': 4.458, 'Colon': 0.21, 'Extrathoracic_Airways': 0.22, 'Gallbladder': 1.657, 'Heart': 3.465, 'Kidneys': 1.675, 'Liver': 4.131, 'Lungs': 4.594, 'Lymph_Nodes': 1.382, 'Muscle': 1.18, 'Oesophagus': 3.742, 'Oral_Mucosa': 0.144, 'Ovaries': 0.105, 'Pancreas': 2.09, 'Prostate': 0.014, 'Salivary_Glands': 0.37, 'Endosteum': 3.366, 'Skin': 1.502, 'Small_Intestine': 0.251, 'Spleen': 0.991, 'Stomach': 1.265, 'Testicles': 0.003, 'Thymus': 3.229, 'Thyroid': 1.247, 'Urinary_Bladder': 0.044, 'Uterus': 0.076},
                    },
                },
            "Lumbar Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.603, 'Adrenals': 4.139, 'Brain': 0.005, 'Breast': 0.076, 'Colon': 4.924, 'Extrathoracic_Airways': 0.011, 'Gallbladder': 6.697, 'Heart': 0.62, 'Kidneys': 4.529, 'Liver': 2.562, 'Lungs': 0.5640000000000001, 'Lymph_Nodes': 4.502, 'Muscle': 1.797, 'Oesophagus': 2.114, 'Oral_Mucosa': 0.018000000000000002, 'Ovaries': 1.846, 'Pancreas': 9.605, 'Prostate': 0.251, 'Salivary_Glands': 0.016, 'Endosteum': 2.0439999999999996, 'Skin': 1.056, 'Small_Intestine': 7.204000000000001, 'Spleen': 17.05, 'Stomach': 11.620000000000001, 'Testicles': 0.08399999999999999, 'Thymus': 0.13899999999999998, 'Thyroid': 0.018000000000000002, 'Urinary_Bladder': 0.47100000000000003, 'Uterus': 1.371},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.6489999999999999, 'Adrenals': 4.492, 'Brain': 0.006, 'Breast': 0.078, 'Colon': 5.296, 'Extrathoracic_Airways': 0.012, 'Gallbladder': 7.296, 'Heart': 0.68, 'Kidneys': 4.882, 'Liver': 2.807, 'Lungs': 0.616, 'Lymph_Nodes': 4.854, 'Muscle': 1.9040000000000001, 'Oesophagus': 2.28, 'Oral_Mucosa': 0.02, 'Ovaries': 2.025, 'Pancreas': 10.323, 'Prostate': 0.27999999999999997, 'Salivary_Glands': 0.018000000000000002, 'Endosteum': 2.1919999999999997, 'Skin': 1.08, 'Small_Intestine': 7.7749999999999995, 'Spleen': 17.948, 'Stomach': 12.4, 'Testicles': 0.095, 'Thymus': 0.153, 'Thyroid': 0.02, 'Urinary_Bladder': 0.521, 'Uterus': 1.511},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.873, 'Adrenals': 6.186, 'Brain': 0.01, 'Breast': 0.06899999999999999, 'Colon': 7.013, 'Extrathoracic_Airways': 0.018000000000000002, 'Gallbladder': 10.126, 'Heart': 0.976, 'Kidneys': 6.518000000000001, 'Liver': 4.026, 'Lungs': 0.876, 'Lymph_Nodes': 6.463, 'Muscle': 2.384, 'Oesophagus': 3.035, 'Oral_Mucosa': 0.032, 'Ovaries': 2.908, 'Pancreas': 13.495, 'Prostate': 0.457, 'Salivary_Glands': 0.028, 'Endosteum': 2.8770000000000002, 'Skin': 1.2, 'Small_Intestine': 10.408000000000001, 'Spleen': 21.568, 'Stomach': 15.806000000000001, 'Testicles': 0.165, 'Thymus': 0.224, 'Thyroid': 0.037, 'Urinary_Bladder': 0.786, 'Uterus': 2.229},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.69, 'Adrenals': 4.805000000000001, 'Brain': 0.007, 'Breast': 0.078, 'Colon': 5.623, 'Extrathoracic_Airways': 0.013, 'Gallbladder': 7.825, 'Heart': 0.733, 'Kidneys': 5.191999999999999, 'Liver': 3.027, 'Lungs': 0.6629999999999999, 'Lymph_Nodes': 5.163, 'Muscle': 1.996, 'Oesophagus': 2.426, 'Oral_Mucosa': 0.022, 'Ovaries': 2.186, 'Pancreas': 10.945, 'Prostate': 0.308, 'Salivary_Glands': 0.02, 'Endosteum': 2.321, 'Skin': 1.1019999999999999, 'Small_Intestine': 8.274999999999999, 'Spleen': 18.701999999999998, 'Stomach': 13.072000000000001, 'Testicles': 0.106, 'Thymus': 0.166, 'Thyroid': 0.023, 'Urinary_Bladder': 0.5670000000000001, 'Uterus': 1.637},
                },},
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 1.4300000000000002, 'Adrenals': 3.437, 'Brain': 0.037, 'Breast': 0.294, 'Colon': 0.184, 'Endosteum': 6.2490000000000006, 'Extrathoracic_Airways': 0.185, 'Gallbladder': 1.615, 'Heart': 2.4699999999999998, 'Kidneys': 2.301, 'Liver': 0.954, 'Lungs': 4.457, 'Lymph_Nodes': 1.635, 'Muscle': 1.305, 'Oesophagus': 4.5440000000000005, 'Oral_Mucosa': 0.18100000000000002, 'Ovaries': 0.052, 'Pancreas': 3.9090000000000003, 'Prostate': 0.01, 'Salivary_Glands': 0.27999999999999997, 'Skin': 1.6340000000000001, 'Small_Intestine': 0.241, 'Spleen': 7.678, 'Stomach': 2.149, 'Testicles': 0.027, 'Thymus': 0.594, 'Thyroid': 0.794, 'Urinary_Bladder': 0.037, 'Uterus': 0.093},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 1.521, 'Adrenals': 3.734, 'Brain': 0.041, 'Breast': 0.323, 'Colon': 0.204, 'Endosteum': 6.577999999999999, 'Extrathoracic_Airways': 0.201, 'Gallbladder': 1.766, 'Heart': 2.676, 'Kidneys': 2.4789999999999996, 'Liver': 1.05, 'Lungs': 4.791, 'Lymph_Nodes': 1.7730000000000001, 'Muscle': 1.393, 'Oesophagus': 4.925, 'Oral_Mucosa': 0.199, 'Ovaries': 0.060000000000000005, 'Pancreas': 4.256, 'Prostate': 0.01, 'Salivary_Glands': 0.306, 'Skin': 1.671, 'Small_Intestine': 0.26699999999999996, 'Spleen': 8.152, 'Stomach': 2.319, 'Testicles': 0.031, 'Thymus': 0.653, 'Thyroid': 0.851, 'Urinary_Bladder': 0.041, 'Uterus': 0.104},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 1.6, 'Adrenals': 3.997, 'Brain': 0.045000000000000005, 'Breast': 0.35, 'Colon': 0.223, 'Endosteum': 6.8580000000000005, 'Extrathoracic_Airways': 0.216, 'Gallbladder': 1.9020000000000001, 'Heart': 2.856, 'Kidneys': 2.6359999999999997, 'Liver': 1.137, 'Lungs': 5.082, 'Lymph_Nodes': 1.895, 'Muscle': 1.47, 'Oesophagus': 5.258, 'Oral_Mucosa': 0.216, 'Ovaries': 0.068, 'Pancreas': 4.563000000000001, 'Prostate': 0.011, 'Salivary_Glands': 0.32899999999999996, 'Skin': 1.703, 'Small_Intestine': 0.292, 'Spleen': 8.557, 'Stomach': 2.467, 'Testicles': 0.034, 'Thymus': 0.705, 'Thyroid': 0.901, 'Urinary_Bladder': 0.045000000000000005, 'Uterus': 0.11299999999999999},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 1.934, 'Adrenals': 5.169, 'Brain': 0.062, 'Breast': 0.475, 'Colon': 0.312, 'Endosteum': 7.955, 'Extrathoracic_Airways': 0.281, 'Gallbladder': 2.511, 'Heart': 3.636, 'Kidneys': 3.319, 'Liver': 1.542, 'Lungs': 6.311, 'Lymph_Nodes': 2.4299999999999997, 'Muscle': 1.797, 'Oesophagus': 6.683, 'Oral_Mucosa': 0.29300000000000004, 'Ovaries': 0.114, 'Pancreas': 5.923, 'Prostate': 0.009000000000000001, 'Salivary_Glands': 0.43600000000000005, 'Skin': 1.843, 'Small_Intestine': 0.41, 'Spleen': 10.158000000000001, 'Stomach': 3.091, 'Testicles': 0.05, 'Thymus': 0.9430000000000001, 'Thyroid': 1.135, 'Urinary_Bladder': 0.06499999999999999, 'Uterus': 0.159},
                },},
                "Scoliosis": {
                    "U": {
                        (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.836, 'Adrenals': 1.707, 'Brain': 0.042, 'Breast': 2.81, 'Colon': 1.821, 'Extrathoracic_Airways': 0.649, 'Gallbladder': 2.242, 'Heart': 2.291, 'Kidneys': 2.28, 'Liver': 3.9, 'Lungs': 3.132, 'Lymph_Nodes': 1.824, 'Muscle': 1.347, 'Oesophagus': 2.439, 'Oral_Mucosa': 0.253, 'Ovaries': 1.26, 'Pancreas': 1.647, 'Prostate': 0.133, 'Salivary_Glands': 0.815, 'Endosteum': 2.923, 'Skin': 1.72, 'Small_Intestine': 2.012, 'Spleen': 0.744, 'Stomach': 1.148, 'Testicles': 0.028, 'Thymus': 2.034, 'Thyroid': 3.647, 'Urinary_Bladder': 0.389, 'Uterus': 1.337},
                        (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.895, 'Adrenals': 1.846, 'Brain': 0.046, 'Breast': 2.944, 'Colon': 1.958, 'Extrathoracic_Airways': 0.69, 'Gallbladder': 2.439, 'Heart': 2.474, 'Kidneys': 2.447, 'Liver': 4.133, 'Lungs': 3.323, 'Lymph_Nodes': 1.965, 'Muscle': 1.428, 'Oesophagus': 2.628, 'Oral_Mucosa': 0.276, 'Ovaries': 1.396, 'Pancreas': 1.798, 'Prostate': 0.151, 'Salivary_Glands': 0.863, 'Endosteum': 3.109, 'Skin': 1.76, 'Small_Intestine': 2.183, 'Spleen': 0.821, 'Stomach': 1.264, 'Testicles': 0.032, 'Thymus': 2.184, 'Thyroid': 3.857, 'Urinary_Bladder': 0.423, 'Uterus': 1.449},
                        (65, 3.0, 0.1): {'Active_Bone_Marrow': 1.168, 'Adrenals': 2.503, 'Brain': 0.07, 'Breast': 3.778, 'Colon': 2.581, 'Extrathoracic_Airways': 0.869, 'Gallbladder': 3.355, 'Heart': 3.303, 'Kidneys': 3.205, 'Liver': 5.135, 'Lungs': 4.161, 'Lymph_Nodes': 2.623, 'Muscle': 1.794, 'Oesophagus': 3.494, 'Oral_Mucosa': 0.396, 'Ovaries': 2.103, 'Pancreas': 2.529, 'Prostate': 0.263, 'Salivary_Glands': 1.072, 'Endosteum': 3.927, 'Skin': 1.946, 'Small_Intestine': 3.005, 'Spleen': 1.221, 'Stomach': 1.858, 'Testicles': 0.059, 'Thymus': 2.833, 'Thyroid': 4.767, 'Urinary_Bladder': 0.58, 'Uterus': 1.976},
                        (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.946, 'Adrenals': 1.969, 'Brain': 0.05, 'Breast': 3.07, 'Colon': 2.077, 'Extrathoracic_Airways': 0.725, 'Gallbladder': 2.612, 'Heart': 2.633, 'Kidneys': 2.592, 'Liver': 4.332, 'Lungs': 3.488, 'Lymph_Nodes': 2.089, 'Muscle': 1.499, 'Oesophagus': 2.794, 'Oral_Mucosa': 0.298, 'Ovaries': 1.52, 'Pancreas': 1.932, 'Prostate': 0.169, 'Salivary_Glands': 0.904, 'Endosteum': 3.269, 'Skin': 1.796, 'Small_Intestine': 2.335, 'Spleen': 0.89, 'Stomach': 1.369, 'Testicles': 0.035, 'Thymus': 2.312, 'Thyroid': 4.037, 'Urinary_Bladder': 0.453, 'Uterus': 1.548},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.0, "Brain": 0.028, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.19, "Extrathoracic_Airways": 1.105, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.333, "Muscle": 0.042, "Oesophagus": 0.029, "Oral_Mucosa": 1.14, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.684, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.008, "Thyroid": 2.587, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.043, "Adrenals": 0.0, "Brain": 0.032, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.207, "Extrathoracic_Airways": 1.194, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.012, "Lymph_Nodes": 0.357, "Muscle": 0.045, "Oesophagus": 0.032, "Oral_Mucosa": 1.215, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.744, "Skin": 0.155, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.009, "Thyroid": 2.725, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.0, "Brain": 0.036, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.222, "Extrathoracic_Airways": 1.273, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.013, "Lymph_Nodes": 0.377, "Muscle": 0.048, "Oesophagus": 0.035, "Oral_Mucosa": 1.281, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.798, "Skin": 0.159, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.01, "Thyroid": 2.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.06, "Adrenals": 0.0, "Brain": 0.061, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.258, "Extrathoracic_Airways": 1.474, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.019, "Lymph_Nodes": 0.422, "Muscle": 0.054, "Oesophagus": 0.044, "Oral_Mucosa": 1.447, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.939, "Skin": 0.164, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.018, "Thyroid": 2.96, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.0, "Brain": 0.068, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.278, "Extrathoracic_Airways": 1.577, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.448, "Muscle": 0.057, "Oesophagus": 0.048, "Oral_Mucosa": 1.534, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.01, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.02, "Thyroid": 3.092, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.069, "Adrenals": 0.0, "Brain": 0.074, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.295, "Extrathoracic_Airways": 1.669, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.471, "Muscle": 0.06, "Oesophagus": 0.051, "Oral_Mucosa": 1.609, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.073, "Skin": 0.172, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.021, "Thyroid": 3.204, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.0, "Brain": 0.094, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.308, "Extrathoracic_Airways": 1.77, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.496, "Muscle": 0.063, "Oesophagus": 0.06, "Oral_Mucosa": 1.707, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.167, "Skin": 0.175, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.025, "Thyroid": 3.259, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.0, "Brain": 0.103, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.327, "Extrathoracic_Airways": 1.873, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.028, "Lymph_Nodes": 0.522, "Muscle": 0.067, "Oesophagus": 0.064, "Oral_Mucosa": 1.794, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.241, "Skin": 0.179, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.027, "Thyroid": 3.38, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.0, "Brain": 0.11, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.344, "Extrathoracic_Airways": 1.963, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.03, "Lymph_Nodes": 0.544, "Muscle": 0.069, "Oesophagus": 0.068, "Oral_Mucosa": 1.87, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.306, "Skin": 0.183, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.029, "Thyroid": 3.481, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.094, "Adrenals": 0.0, "Brain": 0.123, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.342, "Extrathoracic_Airways": 1.991, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.007, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.031, "Lymph_Nodes": 0.552, "Muscle": 0.071, "Oesophagus": 0.074, "Oral_Mucosa": 1.911, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.353, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.03, "Thyroid": 3.473, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.1, "Adrenals": 0.0, "Brain": 0.131, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.358, "Extrathoracic_Airways": 2.085, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.033, "Lymph_Nodes": 0.576, "Muscle": 0.074, "Oesophagus": 0.078, "Oral_Mucosa": 1.992, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.424, "Skin": 0.188, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.032, "Thyroid": 3.577, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.104, "Adrenals": 0.0, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.373, "Extrathoracic_Airways": 2.167, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.035, "Lymph_Nodes": 0.596, "Muscle": 0.076, "Oesophagus": 0.082, "Oral_Mucosa": 2.062, "Pancreas": 0.002, "Prostate": 0.0, "Salivary_Glands": 1.486, "Skin": 0.191, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.034, "Thyroid": 3.664, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 1.1039999999999999, 'Adrenals': 0.018000000000000002, 'Brain': 2.63, 'Breast': 0.0, 'Colon': 0.002, 'Extrathoracic_Airways': 1.15, 'Gallbladder': 0.004, 'Heart': 0.061, 'Kidneys': 0.007, 'Liver': 0.014, 'Lungs': 0.097, 'Lymph_Nodes': 0.588, 'Muscle': 0.462, 'Oesophagus': 0.18100000000000002, 'Oral_Mucosa': 2.137, 'Ovaries': 0.0, 'Pancreas': 0.012, 'Prostate': 0.0, 'Salivary_Glands': 3.873, 'Endosteum': 2.35, 'Skin': 1.175, 'Small_Intestine': 0.003, 'Spleen': 0.018000000000000002, 'Stomach': 0.016, 'Testicles': 0.0, 'Thymus': 0.182, 'Thyroid': 1.321, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 1.167, 'Adrenals': 0.02, 'Brain': 2.812, 'Breast': 0.0, 'Colon': 0.002, 'Extrathoracic_Airways': 1.264, 'Gallbladder': 0.005, 'Heart': 0.068, 'Kidneys': 0.008, 'Liver': 0.016, 'Lungs': 0.10500000000000001, 'Lymph_Nodes': 0.624, 'Muscle': 0.486, 'Oesophagus': 0.194, 'Oral_Mucosa': 2.276, 'Ovaries': 0.0, 'Pancreas': 0.014, 'Prostate': 0.0, 'Salivary_Glands': 4.005, 'Endosteum': 2.473, 'Skin': 1.202, 'Small_Intestine': 0.003, 'Spleen': 0.02, 'Stomach': 0.017, 'Testicles': 0.0, 'Thymus': 0.199, 'Thyroid': 1.456, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 1.444, 'Adrenals': 0.028, 'Brain': 3.6310000000000002, 'Breast': 0.0, 'Colon': 0.004, 'Extrathoracic_Airways': 1.848, 'Gallbladder': 0.008, 'Heart': 0.101, 'Kidneys': 0.012, 'Liver': 0.024, 'Lungs': 0.14400000000000002, 'Lymph_Nodes': 0.8, 'Muscle': 0.589, 'Oesophagus': 0.252, 'Oral_Mucosa': 2.899, 'Ovaries': 0.0, 'Pancreas': 0.023, 'Prostate': 0.0, 'Salivary_Glands': 4.542, 'Endosteum': 2.987, 'Skin': 1.324, 'Small_Intestine': 0.004, 'Spleen': 0.029, 'Stomach': 0.027, 'Testicles': 0.0, 'Thymus': 0.28600000000000003, 'Thyroid': 2.202, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 1.22, 'Adrenals': 0.020999999999999998, 'Brain': 2.971, 'Breast': 0.0, 'Colon': 0.003, 'Extrathoracic_Airways': 1.367, 'Gallbladder': 0.005, 'Heart': 0.073, 'Kidneys': 0.008, 'Liver': 0.017, 'Lungs': 0.11299999999999999, 'Lymph_Nodes': 0.656, 'Muscle': 0.506, 'Oesophagus': 0.205, 'Oral_Mucosa': 2.396, 'Ovaries': 0.0, 'Pancreas': 0.015000000000000001, 'Prostate': 0.0, 'Salivary_Glands': 4.114999999999999, 'Endosteum': 2.578, 'Skin': 1.2260000000000002, 'Small_Intestine': 0.003, 'Spleen': 0.020999999999999998, 'Stomach': 0.019, 'Testicles': 0.0, 'Thymus': 0.214, 'Thyroid': 1.58, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
        1: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Skull": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.424, 'Adrenals': 0.009000000000000001, 'Brain': 1.123, 'Breast': 0.006, 'Colon': 0.0, 'Extrathoracic_Airways': 0.41800000000000004, 'Gallbladder': 0.009000000000000001, 'Heart': 0.013, 'Kidneys': 0.002, 'Liver': 0.004, 'Lungs': 0.029, 'Lymph_Nodes': 0.24000000000000002, 'Muscle': 0.17300000000000001, 'Oesophagus': 0.047, 'Oral_Mucosa': 0.9510000000000001, 'Ovaries': 0.0, 'Pancreas': 0.007, 'Prostate': 0.0, 'Salivary_Glands': 1.856, 'Endosteum': 1.16, 'Skin': 0.477, 'Small_Intestine': 0.0, 'Spleen': 0.003, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.045000000000000005, 'Thyroid': 0.23299999999999998, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.451, 'Adrenals': 0.01, 'Brain': 1.214, 'Breast': 0.006, 'Colon': 0.0, 'Extrathoracic_Airways': 0.46099999999999997, 'Gallbladder': 0.009000000000000001, 'Heart': 0.015000000000000001, 'Kidneys': 0.002, 'Liver': 0.004, 'Lungs': 0.032, 'Lymph_Nodes': 0.256, 'Muscle': 0.182, 'Oesophagus': 0.051, 'Oral_Mucosa': 1.021, 'Ovaries': 0.0, 'Pancreas': 0.008, 'Prostate': 0.0, 'Salivary_Glands': 1.953, 'Endosteum': 1.228, 'Skin': 0.48700000000000004, 'Small_Intestine': 0.0, 'Spleen': 0.004, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.05, 'Thyroid': 0.255, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.575, 'Adrenals': 0.013, 'Brain': 1.6509999999999998, 'Breast': 0.012, 'Colon': 0.001, 'Extrathoracic_Airways': 0.6920000000000001, 'Gallbladder': 0.014, 'Heart': 0.023, 'Kidneys': 0.003, 'Liver': 0.007, 'Lungs': 0.047, 'Lymph_Nodes': 0.33399999999999996, 'Muscle': 0.224, 'Oesophagus': 0.06899999999999999, 'Oral_Mucosa': 1.339, 'Ovaries': 0.0, 'Pancreas': 0.014, 'Prostate': 0.0, 'Salivary_Glands': 2.376, 'Endosteum': 1.524, 'Skin': 0.536, 'Small_Intestine': 0.001, 'Spleen': 0.006, 'Stomach': 0.005, 'Testicles': 0.0, 'Thymus': 0.076, 'Thyroid': 0.357, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.474, 'Adrenals': 0.011, 'Brain': 1.295, 'Breast': 0.007, 'Colon': 0.001, 'Extrathoracic_Airways': 0.501, 'Gallbladder': 0.01, 'Heart': 0.016, 'Kidneys': 0.002, 'Liver': 0.005, 'Lungs': 0.034999999999999996, 'Lymph_Nodes': 0.27, 'Muscle': 0.19, 'Oesophagus': 0.054, 'Oral_Mucosa': 1.082, 'Ovaries': 0.0, 'Pancreas': 0.009000000000000001, 'Prostate': 0.0, 'Salivary_Glands': 2.036, 'Endosteum': 1.286, 'Skin': 0.496, 'Small_Intestine': 0.0, 'Spleen': 0.004, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.054, 'Thyroid': 0.27399999999999997, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.305, 'Adrenals': 1.694, 'Brain': 0.001, 'Breast': 0.059, 'Colon': 1.9, 'Extrathoracic_Airways': 0.006, 'Gallbladder': 2.2169999999999996, 'Heart': 0.297, 'Kidneys': 2.314, 'Liver': 0.923, 'Lungs': 0.265, 'Lymph_Nodes': 1.742, 'Muscle': 0.775, 'Oesophagus': 0.867, 'Oral_Mucosa': 0.005, 'Ovaries': 0.62, 'Pancreas': 3.847, 'Prostate': 0.134, 'Salivary_Glands': 0.005, 'Endosteum': 0.886, 'Skin': 0.494, 'Small_Intestine': 2.72, 'Spleen': 7.664, 'Stomach': 5.223, 'Testicles': 0.06899999999999999, 'Thymus': 0.043000000000000003, 'Thyroid': 0.014, 'Urinary_Bladder': 0.179, 'Uterus': 0.536},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.33, 'Adrenals': 1.849, 'Brain': 0.001, 'Breast': 0.066, 'Colon': 2.058, 'Extrathoracic_Airways': 0.007, 'Gallbladder': 2.4229999999999996, 'Heart': 0.32699999999999996, 'Kidneys': 2.5010000000000003, 'Liver': 1.014, 'Lungs': 0.29, 'Lymph_Nodes': 1.887, 'Muscle': 0.824, 'Oesophagus': 0.9460000000000001, 'Oral_Mucosa': 0.006, 'Ovaries': 0.6769999999999999, 'Pancreas': 4.159000000000001, 'Prostate': 0.15, 'Salivary_Glands': 0.006, 'Endosteum': 0.954, 'Skin': 0.505, 'Small_Intestine': 2.954, 'Spleen': 8.126, 'Stomach': 5.592, 'Testicles': 0.077, 'Thymus': 0.047, 'Thyroid': 0.016, 'Urinary_Bladder': 0.19799999999999998, 'Uterus': 0.588},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.45899999999999996, 'Adrenals': 2.6380000000000003, 'Brain': 0.002, 'Breast': 0.09899999999999999, 'Colon': 2.826, 'Extrathoracic_Airways': 0.012, 'Gallbladder': 3.455, 'Heart': 0.48500000000000004, 'Kidneys': 3.3939999999999997, 'Liver': 1.494, 'Lungs': 0.41800000000000004, 'Lymph_Nodes': 2.591, 'Muscle': 1.047, 'Oesophagus': 1.341, 'Oral_Mucosa': 0.01, 'Ovaries': 0.9560000000000001, 'Pancreas': 5.637, 'Prostate': 0.232, 'Salivary_Glands': 0.01, 'Endosteum': 1.281, 'Skin': 0.562, 'Small_Intestine': 4.087999999999999, 'Spleen': 10.095, 'Stomach': 7.2490000000000006, 'Testicles': 0.11900000000000001, 'Thymus': 0.075, 'Thyroid': 0.030000000000000002, 'Urinary_Bladder': 0.30200000000000005, 'Uterus': 0.853},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.35300000000000004, 'Adrenals': 1.9880000000000002, 'Brain': 0.002, 'Breast': 0.07100000000000001, 'Colon': 2.199, 'Extrathoracic_Airways': 0.008, 'Gallbladder': 2.609, 'Heart': 0.355, 'Kidneys': 2.6670000000000003, 'Liver': 1.098, 'Lungs': 0.312, 'Lymph_Nodes': 2.0170000000000003, 'Muscle': 0.866, 'Oesophagus': 1.017, 'Oral_Mucosa': 0.007, 'Ovaries': 0.728, 'Pancreas': 4.433000000000001, 'Prostate': 0.163, 'Salivary_Glands': 0.007, 'Endosteum': 1.014, 'Skin': 0.515, 'Small_Intestine': 3.161, 'Spleen': 8.52, 'Stomach': 5.912, 'Testicles': 0.08399999999999999, 'Thymus': 0.052, 'Thyroid': 0.018000000000000002, 'Urinary_Bladder': 0.216, 'Uterus': 0.635},
                },},
                "Chest": {
                    "U": {
                        (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.288, 'Adrenals': 0.663, 'Brain': 0.009, 'Breast': 1.766, 'Colon': 0.058, 'Extrathoracic_Airways': 0.055, 'Gallbladder': 0.398, 'Heart': 1.009, 'Kidneys': 0.487, 'Liver': 1.404, 'Lungs': 1.643, 'Lymph_Nodes': 0.43, 'Muscle': 0.373, 'Oesophagus': 0.949, 'Oral_Mucosa': 0.068, 'Ovaries': 0.002, 'Pancreas': 0.542, 'Prostate': 0, 'Salivary_Glands': 0.094, 'Endosteum': 1.145, 'Skin': 0.581, 'Small_Intestine': 0.06, 'Spleen': 0.205, 'Stomach': 0.275, 'Testicles': 0, 'Thymus': 0.834, 'Thyroid': 1.09, 'Urinary_Bladder': 0.007, 'Uterus': 0},
                        (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.308, 'Adrenals': 0.726, 'Brain': 0.01, 'Breast': 1.825, 'Colon': 0.065, 'Extrathoracic_Airways': 0.06, 'Gallbladder': 0.438, 'Heart': 1.098, 'Kidneys': 0.531, 'Liver': 1.504, 'Lungs': 1.757, 'Lymph_Nodes': 0.467, 'Muscle': 0.398, 'Oesophagus': 1.037, 'Oral_Mucosa': 0.075, 'Ovaries': 0.003, 'Pancreas': 0.598, 'Prostate': 0, 'Salivary_Glands': 0.102, 'Endosteum': 1.218, 'Skin': 0.595, 'Small_Intestine': 0.066, 'Spleen': 0.227, 'Stomach': 0.304, 'Testicles': 0, 'Thymus': 0.911, 'Thyroid': 1.167, 'Urinary_Bladder': 0.008, 'Uterus': 0},
                        (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.407, 'Adrenals': 1.054, 'Brain': 0.016, 'Breast': 2.149, 'Colon': 0.103, 'Extrathoracic_Airways': 0.084, 'Gallbladder': 0.639, 'Heart': 1.525, 'Kidneys': 0.753, 'Liver': 1.956, 'Lungs': 2.27, 'Lymph_Nodes': 0.647, 'Muscle': 0.514, 'Oesophagus': 1.478, 'Oral_Mucosa': 0.109, 'Ovaries': 0.006, 'Pancreas': 0.897, 'Prostate': 0, 'Salivary_Glands': 0.143, 'Endosteum': 1.551, 'Skin': 0.658, 'Small_Intestine': 0.103, 'Spleen': 0.355, 'Stomach': 0.46, 'Testicles': 0, 'Thymus': 1.291, 'Thyroid': 1.501, 'Urinary_Bladder': 0.014, 'Uterus': 0},
                        (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.326, 'Adrenals': 0.784, 'Brain': 0.011, 'Breast': 1.88, 'Colon': 0.071, 'Extrathoracic_Airways': 0.064, 'Gallbladder': 0.474, 'Heart': 1.177, 'Kidneys': 0.571, 'Liver': 1.59, 'Lungs': 1.856, 'Lymph_Nodes': 0.5, 'Muscle': 0.419, 'Oesophagus': 1.116, 'Oral_Mucosa': 0.081, 'Ovaries': 0.003, 'Pancreas': 0.649, 'Prostate': 0, 'Salivary_Glands': 0.11, 'Endosteum': 1.282, 'Skin': 0.607, 'Small_Intestine': 0.073, 'Spleen': 0.249, 'Stomach': 0.331, 'Testicles': 0, 'Thymus': 0.979, 'Thyroid': 1.233, 'Urinary_Bladder': 0.009, 'Uterus': 0},
                    },
                },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.499, 'Adrenals': 1.07, 'Brain': 0.014, 'Breast': 0.192, 'Colon': 0.06499999999999999, 'Endosteum': 2.504, 'Extrathoracic_Airways': 0.092, 'Gallbladder': 0.48000000000000004, 'Heart': 1.1640000000000001, 'Kidneys': 0.767, 'Liver': 0.32, 'Lungs': 1.6869999999999998, 'Lymph_Nodes': 0.644, 'Muscle': 0.49799999999999994, 'Oesophagus': 1.343, 'Oral_Mucosa': 0.075, 'Ovaries': 0.034999999999999996, 'Pancreas': 1.2930000000000001, 'Prostate': 0.0, 'Salivary_Glands': 0.093, 'Skin': 0.678, 'Small_Intestine': 0.079, 'Spleen': 2.8, 'Stomach': 1.211, 'Testicles': 0.004, 'Thymus': 0.33599999999999997, 'Thyroid': 1.477, 'Urinary_Bladder': 0.009000000000000001, 'Uterus': 0.023},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.535, 'Adrenals': 1.175, 'Brain': 0.016, 'Breast': 0.205, 'Colon': 0.073, 'Endosteum': 2.66, 'Extrathoracic_Airways': 0.101, 'Gallbladder': 0.532, 'Heart': 1.271, 'Kidneys': 0.833, 'Liver': 0.355, 'Lungs': 1.828, 'Lymph_Nodes': 0.703, 'Muscle': 0.5339999999999999, 'Oesophagus': 1.469, 'Oral_Mucosa': 0.082, 'Ovaries': 0.039, 'Pancreas': 1.423, 'Prostate': 0.0, 'Salivary_Glands': 0.102, 'Skin': 0.694, 'Small_Intestine': 0.088, 'Spleen': 3.0109999999999997, 'Stomach': 1.317, 'Testicles': 0.005, 'Thymus': 0.371, 'Thyroid': 1.5859999999999999, 'Urinary_Bladder': 0.01, 'Uterus': 0.026},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.5670000000000001, 'Adrenals': 1.269, 'Brain': 0.017, 'Breast': 0.215, 'Colon': 0.08, 'Endosteum': 2.794, 'Extrathoracic_Airways': 0.11, 'Gallbladder': 0.579, 'Heart': 1.367, 'Kidneys': 0.891, 'Liver': 0.38699999999999996, 'Lungs': 1.953, 'Lymph_Nodes': 0.756, 'Muscle': 0.566, 'Oesophagus': 1.583, 'Oral_Mucosa': 0.088, 'Ovaries': 0.043000000000000003, 'Pancreas': 1.5410000000000001, 'Prostate': 0.0, 'Salivary_Glands': 0.111, 'Skin': 0.709, 'Small_Intestine': 0.097, 'Spleen': 3.195, 'Stomach': 1.4120000000000001, 'Testicles': 0.006, 'Thymus': 0.40299999999999997, 'Thyroid': 1.684, 'Urinary_Bladder': 0.012, 'Uterus': 0.029},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.71, 'Adrenals': 1.707, 'Brain': 0.025, 'Breast': 0.252, 'Colon': 0.116, 'Endosteum': 3.349, 'Extrathoracic_Airways': 0.15100000000000002, 'Gallbladder': 0.8019999999999999, 'Heart': 1.802, 'Kidneys': 1.1540000000000001, 'Liver': 0.5429999999999999, 'Lungs': 2.5, 'Lymph_Nodes': 1.0, 'Muscle': 0.705, 'Oesophagus': 2.095, 'Oral_Mucosa': 0.11699999999999999, 'Ovaries': 0.06499999999999999, 'Pancreas': 2.0869999999999997, 'Prostate': 0.0, 'Salivary_Glands': 0.15100000000000002, 'Skin': 0.7739999999999999, 'Small_Intestine': 0.13999999999999999, 'Spleen': 3.9690000000000003, 'Stomach': 1.833, 'Testicles': 0.01, 'Thymus': 0.554, 'Thyroid': 2.1359999999999997, 'Urinary_Bladder': 0.018000000000000002, 'Uterus': 0.044},
                },},

                "Scoliosis": {
                    "U": {
                        (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.377, 'Adrenals': 0.746, 'Brain': 0.019, 'Breast': 1.426, 'Colon': 0.755, 'Extrathoracic_Airways': 0.358, 'Gallbladder': 0.892, 'Heart': 0.932, 'Kidneys': 0.869, 'Liver': 1.875, 'Lungs': 1.43, 'Lymph_Nodes': 0.768, 'Muscle': 0.591, 'Oesophagus': 1.053, 'Oral_Mucosa': 0.231, 'Ovaries': 0.554, 'Pancreas': 0.614, 'Prostate': 0.097, 'Salivary_Glands': 0.506, 'Endosteum': 1.31, 'Skin': 0.796, 'Small_Intestine': 0.793, 'Spleen': 0.242, 'Stomach': 0.391, 'Testicles': 0.015, 'Thymus': 0.793, 'Thyroid': 1.524, 'Urinary_Bladder': 0.165, 'Uterus': 0.496},
                        (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.405, 'Adrenals': 0.815, 'Brain': 0.021, 'Breast': 1.491, 'Colon': 0.816, 'Extrathoracic_Airways': 0.386, 'Gallbladder': 0.973, 'Heart': 1.011, 'Kidneys': 0.938, 'Liver': 1.997, 'Lungs': 1.527, 'Lymph_Nodes': 0.831, 'Muscle': 0.629, 'Oesophagus': 1.143, 'Oral_Mucosa': 0.252, 'Ovaries': 0.604, 'Pancreas': 0.674, 'Prostate': 0.106, 'Salivary_Glands': 0.543, 'Endosteum': 1.399, 'Skin': 0.814, 'Small_Intestine': 0.863, 'Spleen': 0.268, 'Stomach': 0.431, 'Testicles': 0.017, 'Thymus': 0.86, 'Thyroid': 1.629, 'Urinary_Bladder': 0.18, 'Uterus': 0.539},
                        (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.544, 'Adrenals': 1.16, 'Brain': 0.032, 'Breast': 1.803, 'Colon': 1.108, 'Extrathoracic_Airways': 0.516, 'Gallbladder': 1.367, 'Heart': 1.386, 'Kidneys': 1.27, 'Liver': 2.544, 'Lungs': 1.965, 'Lymph_Nodes': 1.139, 'Muscle': 0.803, 'Oesophagus': 1.57, 'Oral_Mucosa': 0.354, 'Ovaries': 0.854, 'Pancreas': 0.986, 'Prostate': 0.146, 'Salivary_Glands': 0.723, 'Endosteum': 1.803, 'Skin': 0.899, 'Small_Intestine': 1.214, 'Spleen': 0.411, 'Stomach': 0.646, 'Testicles': 0.031, 'Thymus': 1.181, 'Thyroid': 2.097, 'Urinary_Bladder': 0.26, 'Uterus': 0.748},
                        (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.43, 'Adrenals': 0.877, 'Brain': 0.023, 'Breast': 1.548, 'Colon': 0.87, 'Extrathoracic_Airways': 0.41, 'Gallbladder': 1.044, 'Heart': 1.08, 'Kidneys': 0.999, 'Liver': 2.103, 'Lungs': 1.611, 'Lymph_Nodes': 0.888, 'Muscle': 0.661, 'Oesophagus': 1.221, 'Oral_Mucosa': 0.27, 'Ovaries': 0.649, 'Pancreas': 0.729, 'Prostate': 0.114, 'Salivary_Glands': 0.576, 'Endosteum': 1.476, 'Skin': 0.83, 'Small_Intestine': 0.927, 'Spleen': 0.292, 'Stomach': 0.467, 'Testicles': 0.019, 'Thymus': 0.919, 'Thyroid': 1.719, 'Urinary_Bladder': 0.195, 'Uterus': 0.578},
                    },
                },
        },
        5: {
            "Validation": _VALIDATION_ORGAN_REGION,
                "Chest": {
                    "U": {
                        (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.173, 'Adrenals': 0.31, 'Brain': 0.007, 'Breast': 1.526, 'Colon': 0.023, 'Extrathoracic_Airways': 0.059, 'Gallbladder': 0.197, 'Heart': 0.572, 'Kidneys': 0.235, 'Liver': 0.796, 'Lungs': 0.943, 'Lymph_Nodes': 0.246, 'Muscle': 0.2, 'Oesophagus': 0.56, 'Oral_Mucosa': 0.033, 'Ovaries': 0.007, 'Pancreas': 0.275, 'Prostate': 0.009, 'Salivary_Glands': 0.06, 'Endosteum': 0.53, 'Skin': 0.258, 'Small_Intestine': 0.026, 'Spleen': 0.104, 'Stomach': 0.138, 'Testicles': 0, 'Thymus': 0.507, 'Thyroid': 0.751, 'Urinary_Bladder': 0.002, 'Uterus': 0.006},
                        (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.185, 'Adrenals': 0.34, 'Brain': 0.008, 'Breast': 1.575, 'Colon': 0.026, 'Extrathoracic_Airways': 0.063, 'Gallbladder': 0.216, 'Heart': 0.622, 'Kidneys': 0.257, 'Liver': 0.854, 'Lungs': 1.01, 'Lymph_Nodes': 0.265, 'Muscle': 0.213, 'Oesophagus': 0.611, 'Oral_Mucosa': 0.037, 'Ovaries': 0.007, 'Pancreas': 0.302, 'Prostate': 0.01, 'Salivary_Glands': 0.066, 'Endosteum': 0.563, 'Skin': 0.265, 'Small_Intestine': 0.029, 'Spleen': 0.116, 'Stomach': 0.152, 'Testicles': 0, 'Thymus': 0.552, 'Thyroid': 0.791, 'Urinary_Bladder': 0.003, 'Uterus': 0.007},
                        (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.244, 'Adrenals': 0.492, 'Brain': 0.012, 'Breast': 1.771, 'Colon': 0.041, 'Extrathoracic_Airways': 0.084, 'Gallbladder': 0.313, 'Heart': 0.872, 'Kidneys': 0.368, 'Liver': 1.124, 'Lungs': 1.316, 'Lymph_Nodes': 0.361, 'Muscle': 0.275, 'Oesophagus': 0.87, 'Oral_Mucosa': 0.053, 'Ovaries': 0.012, 'Pancreas': 0.446, 'Prostate': 0.018, 'Salivary_Glands': 0.094, 'Endosteum': 0.714, 'Skin': 0.294, 'Small_Intestine': 0.045, 'Spleen': 0.183, 'Stomach': 0.23, 'Testicles': 0, 'Thymus': 0.779, 'Thyroid': 0.938, 'Urinary_Bladder': 0.005, 'Uterus': 0.012},
                        (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.196, 'Adrenals': 0.367, 'Brain': 0.008, 'Breast': 1.617, 'Colon': 0.029, 'Extrathoracic_Airways': 0.067, 'Gallbladder': 0.232, 'Heart': 0.667, 'Kidneys': 0.276, 'Liver': 0.905, 'Lungs': 1.068, 'Lymph_Nodes': 0.283, 'Muscle': 0.224, 'Oesophagus': 0.656, 'Oral_Mucosa': 0.04, 'Ovaries': 0.008, 'Pancreas': 0.326, 'Prostate': 0.011, 'Salivary_Glands': 0.071, 'Endosteum': 0.592, 'Skin': 0.27, 'Small_Intestine': 0.032, 'Spleen': 0.127, 'Stomach': 0.165, 'Testicles': 0, 'Thymus': 0.593, 'Thyroid': 0.824, 'Urinary_Bladder': 0.003, 'Uterus': 0.008},
                    },
                },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.17, 'Adrenals': 0.784, 'Brain': 0.0, 'Breast': 0.028, 'Colon': 0.844, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 1.14, 'Heart': 0.186, 'Kidneys': 1.1139999999999999, 'Liver': 0.367, 'Lungs': 0.149, 'Lymph_Nodes': 0.878, 'Muscle': 0.41, 'Oesophagus': 0.433, 'Oral_Mucosa': 0.001, 'Ovaries': 0.148, 'Pancreas': 2.17, 'Prostate': 0.018000000000000002, 'Salivary_Glands': 0.003, 'Endosteum': 0.337, 'Skin': 0.14100000000000001, 'Small_Intestine': 1.167, 'Spleen': 4.739, 'Stomach': 3.027, 'Testicles': 0.0, 'Thymus': 0.016, 'Thyroid': 0.023, 'Urinary_Bladder': 0.044, 'Uterus': 0.199},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.185, 'Adrenals': 0.8630000000000001, 'Brain': 0.0, 'Breast': 0.031, 'Colon': 0.921, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 1.252, 'Heart': 0.205, 'Kidneys': 1.213, 'Liver': 0.406, 'Lungs': 0.163, 'Lymph_Nodes': 0.9560000000000001, 'Muscle': 0.43600000000000005, 'Oesophagus': 0.472, 'Oral_Mucosa': 0.001, 'Ovaries': 0.163, 'Pancreas': 2.356, 'Prostate': 0.020999999999999998, 'Salivary_Glands': 0.004, 'Endosteum': 0.363, 'Skin': 0.145, 'Small_Intestine': 1.274, 'Spleen': 5.073, 'Stomach': 3.265, 'Testicles': 0.0, 'Thymus': 0.018000000000000002, 'Thyroid': 0.026, 'Urinary_Bladder': 0.049, 'Uterus': 0.21800000000000003},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.263, 'Adrenals': 1.28, 'Brain': 0.001, 'Breast': 0.047, 'Colon': 1.31, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 1.833, 'Heart': 0.31, 'Kidneys': 1.709, 'Liver': 0.619, 'Lungs': 0.23900000000000002, 'Lymph_Nodes': 1.345, 'Muscle': 0.5599999999999999, 'Oesophagus': 0.665, 'Oral_Mucosa': 0.002, 'Ovaries': 0.25, 'Pancreas': 3.269, 'Prostate': 0.034999999999999996, 'Salivary_Glands': 0.006, 'Endosteum': 0.49200000000000005, 'Skin': 0.16799999999999998, 'Small_Intestine': 1.815, 'Spleen': 6.585999999999999, 'Stomach': 4.3759999999999994, 'Testicles': 0.0, 'Thymus': 0.031, 'Thyroid': 0.038, 'Urinary_Bladder': 0.081, 'Uterus': 0.324},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.19799999999999998, 'Adrenals': 0.9349999999999999, 'Brain': 0.0, 'Breast': 0.033, 'Colon': 0.99, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 1.3539999999999999, 'Heart': 0.223, 'Kidneys': 1.302, 'Liver': 0.442, 'Lungs': 0.176, 'Lymph_Nodes': 1.026, 'Muscle': 0.45899999999999996, 'Oesophagus': 0.507, 'Oral_Mucosa': 0.001, 'Ovaries': 0.178, 'Pancreas': 2.522, 'Prostate': 0.023, 'Salivary_Glands': 0.004, 'Endosteum': 0.386, 'Skin': 0.149, 'Small_Intestine': 1.3699999999999999, 'Spleen': 5.3629999999999995, 'Stomach': 3.474, 'Testicles': 0.0, 'Thymus': 0.02, 'Thyroid': 0.028, 'Urinary_Bladder': 0.054, 'Uterus': 0.236},
                },},
                        "Thoracic Spine": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.298, 'Adrenals': 0.5379999999999999, 'Brain': 0.009000000000000001, 'Breast': 0.16899999999999998, 'Colon': 0.036000000000000004, 'Endosteum': 1.184, 'Extrathoracic_Airways': 0.033, 'Gallbladder': 0.23, 'Heart': 0.6930000000000001, 'Kidneys': 0.504, 'Liver': 0.15, 'Lungs': 0.933, 'Lymph_Nodes': 0.32, 'Muscle': 0.272, 'Oesophagus': 0.6970000000000001, 'Oral_Mucosa': 0.034, 'Ovaries': 0.007, 'Pancreas': 0.644, 'Prostate': 0.0, 'Salivary_Glands': 0.058, 'Skin': 0.379, 'Small_Intestine': 0.046, 'Spleen': 1.77, 'Stomach': 0.695, 'Testicles': 0.007, 'Thymus': 0.219, 'Thyroid': 0.381, 'Urinary_Bladder': 0.005, 'Uterus': 0.013},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.319, 'Adrenals': 0.593, 'Brain': 0.01, 'Breast': 0.188, 'Colon': 0.04, 'Endosteum': 1.254, 'Extrathoracic_Airways': 0.036000000000000004, 'Gallbladder': 0.254, 'Heart': 0.756, 'Kidneys': 0.546, 'Liver': 0.166, 'Lungs': 1.012, 'Lymph_Nodes': 0.34900000000000003, 'Muscle': 0.29100000000000004, 'Oesophagus': 0.762, 'Oral_Mucosa': 0.037, 'Ovaries': 0.007, 'Pancreas': 0.708, 'Prostate': 0.0, 'Salivary_Glands': 0.064, 'Skin': 0.38699999999999996, 'Small_Intestine': 0.051, 'Spleen': 1.909, 'Stomach': 0.754, 'Testicles': 0.007, 'Thymus': 0.241, 'Thyroid': 0.414, 'Urinary_Bladder': 0.005, 'Uterus': 0.015000000000000001},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.33799999999999997, 'Adrenals': 0.642, 'Brain': 0.01, 'Breast': 0.206, 'Colon': 0.043000000000000003, 'Endosteum': 1.3159999999999998, 'Extrathoracic_Airways': 0.04, 'Gallbladder': 0.27599999999999997, 'Heart': 0.8130000000000001, 'Kidneys': 0.583, 'Liver': 0.18100000000000002, 'Lungs': 1.081, 'Lymph_Nodes': 0.376, 'Muscle': 0.308, 'Oesophagus': 0.82, 'Oral_Mucosa': 0.04, 'Ovaries': 0.008, 'Pancreas': 0.766, 'Prostate': 0.0, 'Salivary_Glands': 0.06899999999999999, 'Skin': 0.395, 'Small_Intestine': 0.056, 'Spleen': 2.032, 'Stomach': 0.8059999999999999, 'Testicles': 0.008, 'Thymus': 0.262, 'Thyroid': 0.444, 'Urinary_Bladder': 0.006, 'Uterus': 0.016},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.423, 'Adrenals': 0.88, 'Brain': 0.015000000000000001, 'Breast': 0.29300000000000004, 'Colon': 0.062, 'Endosteum': 1.568, 'Extrathoracic_Airways': 0.058, 'Gallbladder': 0.389, 'Heart': 1.0759999999999998, 'Kidneys': 0.755, 'Liver': 0.256, 'Lungs': 1.3960000000000001, 'Lymph_Nodes': 0.502, 'Muscle': 0.38499999999999995, 'Oesophagus': 1.092, 'Oral_Mucosa': 0.054, 'Ovaries': 0.011, 'Pancreas': 1.0510000000000002, 'Prostate': 0.0, 'Salivary_Glands': 0.093, 'Skin': 0.429, 'Small_Intestine': 0.08, 'Spleen': 2.5730000000000004, 'Stomach': 1.0410000000000001, 'Testicles': 0.013, 'Thymus': 0.36200000000000004, 'Thyroid': 0.5760000000000001, 'Urinary_Bladder': 0.009000000000000001, 'Uterus': 0.025},
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.226, 'Adrenals': 0.314, 'Brain': 0.008, 'Breast': 1.03, 'Colon': 0.41, 'Extrathoracic_Airways': 0.213, 'Gallbladder': 0.42, 'Heart': 0.495, 'Kidneys': 0.457, 'Liver': 1.082, 'Lungs': 0.817, 'Lymph_Nodes': 0.401, 'Muscle': 0.297, 'Oesophagus': 0.504, 'Oral_Mucosa': 0.07, 'Ovaries': 0.353, 'Pancreas': 0.31, 'Prostate': 0.023, 'Salivary_Glands': 0.124, 'Endosteum': 0.57, 'Skin': 0.371, 'Small_Intestine': 0.41, 'Spleen': 0.095, 'Stomach': 0.169, 'Testicles': 0, 'Thymus': 0.43, 'Thyroid': 1.062, 'Urinary_Bladder': 0.073, 'Uterus': 0.289},
                        (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.243, 'Adrenals': 0.345, 'Brain': 0.008, 'Breast': 1.053, 'Colon': 0.446, 'Extrathoracic_Airways': 0.23, 'Gallbladder': 0.46, 'Heart': 0.538, 'Kidneys': 0.496, 'Liver': 1.157, 'Lungs': 0.875, 'Lymph_Nodes': 0.435, 'Muscle': 0.316, 'Oesophagus': 0.554, 'Oral_Mucosa': 0.076, 'Ovaries': 0.389, 'Pancreas': 0.341, 'Prostate': 0.024, 'Salivary_Glands': 0.133, 'Endosteum': 0.608, 'Skin': 0.38, 'Small_Intestine': 0.448, 'Spleen': 0.106, 'Stomach': 0.187, 'Testicles': 0, 'Thymus': 0.469, 'Thyroid': 1.122, 'Urinary_Bladder': 0.08, 'Uterus': 0.315},
                        (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.329, 'Adrenals': 0.51, 'Brain': 0.013, 'Breast': 1.116, 'Colon': 0.63, 'Extrathoracic_Airways': 0.302, 'Gallbladder': 0.665, 'Heart': 0.756, 'Kidneys': 0.69, 'Liver': 1.505, 'Lungs': 1.145, 'Lymph_Nodes': 0.601, 'Muscle': 0.406, 'Oesophagus': 0.821, 'Oral_Mucosa': 0.108, 'Ovaries': 0.583, 'Pancreas': 0.508, 'Prostate': 0.031, 'Salivary_Glands': 0.174, 'Endosteum': 0.785, 'Skin': 0.419, 'Small_Intestine': 0.641, 'Spleen': 0.167, 'Stomach': 0.288, 'Testicles': 0.001, 'Thymus': 0.67, 'Thyroid': 1.385, 'Urinary_Bladder': 0.121, 'Uterus': 0.449},
                        (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.259, 'Adrenals': 0.374, 'Brain': 0.009, 'Breast': 1.07, 'Colon': 0.479, 'Extrathoracic_Airways': 0.244, 'Gallbladder': 0.496, 'Heart': 0.578, 'Kidneys': 0.53, 'Liver': 1.222, 'Lungs': 0.926, 'Lymph_Nodes': 0.465, 'Muscle': 0.333, 'Oesophagus': 0.6, 'Oral_Mucosa': 0.082, 'Ovaries': 0.421, 'Pancreas': 0.37, 'Prostate': 0.026, 'Salivary_Glands': 0.141, 'Endosteum': 0.642, 'Skin': 0.387, 'Small_Intestine': 0.482, 'Spleen': 0.116, 'Stomach': 0.204, 'Testicles': 0, 'Thymus': 0.505, 'Thyroid': 1.174, 'Urinary_Bladder': 0.087, 'Uterus': 0.338},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.0, "Brain": 0.028, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.19, "Extrathoracic_Airways": 1.105, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.333, "Muscle": 0.042, "Oesophagus": 0.029, "Oral_Mucosa": 1.14, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.684, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.008, "Thyroid": 2.587, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.043, "Adrenals": 0.0, "Brain": 0.032, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.207, "Extrathoracic_Airways": 1.194, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.012, "Lymph_Nodes": 0.357, "Muscle": 0.045, "Oesophagus": 0.032, "Oral_Mucosa": 1.215, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.744, "Skin": 0.155, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.009, "Thyroid": 2.725, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.0, "Brain": 0.036, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.222, "Extrathoracic_Airways": 1.273, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.013, "Lymph_Nodes": 0.377, "Muscle": 0.048, "Oesophagus": 0.035, "Oral_Mucosa": 1.281, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.798, "Skin": 0.159, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.01, "Thyroid": 2.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.06, "Adrenals": 0.0, "Brain": 0.061, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.258, "Extrathoracic_Airways": 1.474, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.019, "Lymph_Nodes": 0.422, "Muscle": 0.054, "Oesophagus": 0.044, "Oral_Mucosa": 1.447, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.939, "Skin": 0.164, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.018, "Thyroid": 2.96, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.0, "Brain": 0.068, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.278, "Extrathoracic_Airways": 1.577, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.448, "Muscle": 0.057, "Oesophagus": 0.048, "Oral_Mucosa": 1.534, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.01, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.02, "Thyroid": 3.092, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.069, "Adrenals": 0.0, "Brain": 0.074, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.295, "Extrathoracic_Airways": 1.669, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.471, "Muscle": 0.06, "Oesophagus": 0.051, "Oral_Mucosa": 1.609, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.073, "Skin": 0.172, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.021, "Thyroid": 3.204, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.0, "Brain": 0.094, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.308, "Extrathoracic_Airways": 1.77, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.496, "Muscle": 0.063, "Oesophagus": 0.06, "Oral_Mucosa": 1.707, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.167, "Skin": 0.175, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.025, "Thyroid": 3.259, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.0, "Brain": 0.103, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.327, "Extrathoracic_Airways": 1.873, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.028, "Lymph_Nodes": 0.522, "Muscle": 0.067, "Oesophagus": 0.064, "Oral_Mucosa": 1.794, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.241, "Skin": 0.179, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.027, "Thyroid": 3.38, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.0, "Brain": 0.11, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.344, "Extrathoracic_Airways": 1.963, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.03, "Lymph_Nodes": 0.544, "Muscle": 0.069, "Oesophagus": 0.068, "Oral_Mucosa": 1.87, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.306, "Skin": 0.183, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.029, "Thyroid": 3.481, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.094, "Adrenals": 0.0, "Brain": 0.123, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.342, "Extrathoracic_Airways": 1.991, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.007, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.031, "Lymph_Nodes": 0.552, "Muscle": 0.071, "Oesophagus": 0.074, "Oral_Mucosa": 1.911, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.353, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.03, "Thyroid": 3.473, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.1, "Adrenals": 0.0, "Brain": 0.131, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.358, "Extrathoracic_Airways": 2.085, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.033, "Lymph_Nodes": 0.576, "Muscle": 0.074, "Oesophagus": 0.078, "Oral_Mucosa": 1.992, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.424, "Skin": 0.188, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.032, "Thyroid": 3.577, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.104, "Adrenals": 0.0, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.373, "Extrathoracic_Airways": 2.167, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.035, "Lymph_Nodes": 0.596, "Muscle": 0.076, "Oesophagus": 0.082, "Oral_Mucosa": 2.062, "Pancreas": 0.002, "Prostate": 0.0, "Salivary_Glands": 1.486, "Skin": 0.191, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.034, "Thyroid": 3.664, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.185, 'Adrenals': 0.001, 'Brain': 0.739, 'Breast': 0.005, 'Colon': 0.0, 'Extrathoracic_Airways': 0.25, 'Gallbladder': 0.001, 'Heart': 0.005, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.016, 'Lymph_Nodes': 0.177, 'Muscle': 0.088, 'Oesophagus': 0.018000000000000002, 'Oral_Mucosa': 0.628, 'Ovaries': 0.0, 'Pancreas': 0.004, 'Prostate': 0.0, 'Salivary_Glands': 1.441, 'Endosteum': 0.855, 'Skin': 0.29, 'Small_Intestine': 0.0, 'Spleen': 0.004, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.024, 'Thyroid': 0.301, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.19699999999999998, 'Adrenals': 0.001, 'Brain': 0.803, 'Breast': 0.005, 'Colon': 0.0, 'Extrathoracic_Airways': 0.27399999999999997, 'Gallbladder': 0.001, 'Heart': 0.006, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.018000000000000002, 'Lymph_Nodes': 0.189, 'Muscle': 0.093, 'Oesophagus': 0.02, 'Oral_Mucosa': 0.6759999999999999, 'Ovaries': 0.0, 'Pancreas': 0.004, 'Prostate': 0.0, 'Salivary_Glands': 1.517, 'Endosteum': 0.909, 'Skin': 0.296, 'Small_Intestine': 0.0, 'Spleen': 0.004, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.026, 'Thyroid': 0.333, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.259, 'Adrenals': 0.001, 'Brain': 1.124, 'Breast': 0.01, 'Colon': 0.0, 'Extrathoracic_Airways': 0.40299999999999997, 'Gallbladder': 0.001, 'Heart': 0.01, 'Kidneys': 0.002, 'Liver': 0.004, 'Lungs': 0.027, 'Lymph_Nodes': 0.24600000000000002, 'Muscle': 0.114, 'Oesophagus': 0.031, 'Oral_Mucosa': 0.915, 'Ovaries': 0.0, 'Pancreas': 0.007, 'Prostate': 0.0, 'Salivary_Glands': 1.846, 'Endosteum': 1.15, 'Skin': 0.326, 'Small_Intestine': 0.0, 'Spleen': 0.006, 'Stomach': 0.003, 'Testicles': 0.0, 'Thymus': 0.04, 'Thyroid': 0.504, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.20900000000000002, 'Adrenals': 0.001, 'Brain': 0.861, 'Breast': 0.006, 'Colon': 0.0, 'Extrathoracic_Airways': 0.297, 'Gallbladder': 0.001, 'Heart': 0.007, 'Kidneys': 0.001, 'Liver': 0.002, 'Lungs': 0.02, 'Lymph_Nodes': 0.199, 'Muscle': 0.097, 'Oesophagus': 0.022, 'Oral_Mucosa': 0.7200000000000001, 'Ovaries': 0.0, 'Pancreas': 0.005, 'Prostate': 0.0, 'Salivary_Glands': 1.582, 'Endosteum': 0.955, 'Skin': 0.301, 'Small_Intestine': 0.0, 'Spleen': 0.005, 'Stomach': 0.002, 'Testicles': 0.0, 'Thymus': 0.028, 'Thyroid': 0.361, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
        10: {
            "Validation": _VALIDATION_ORGAN_REGION,
                "Chest": {
                    "U": {
                        (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.109, 'Adrenals': 0.209, 'Brain': 0.003, 'Breast': 0.882, 'Colon': 0.015, 'Extrathoracic_Airways': 0.072, 'Gallbladder': 0.151, 'Heart': 0.254, 'Kidneys': 0.177, 'Liver': 0.56, 'Lungs': 0.48, 'Lymph_Nodes': 0.126, 'Muscle': 0.108, 'Oesophagus': 0.209, 'Oral_Mucosa': 0.023, 'Ovaries': 0.002, 'Pancreas': 0.109, 'Prostate': 0, 'Salivary_Glands': 0.021, 'Endosteum': 0.267, 'Skin': 0.151, 'Small_Intestine': 0.017, 'Spleen': 0.039, 'Stomach': 0.049, 'Testicles': 0, 'Thymus': 0.237, 'Thyroid': 0.378, 'Urinary_Bladder': 0.001, 'Uterus': 0.002},
                        (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.117, 'Adrenals': 0.23, 'Brain': 0.003, 'Breast': 0.903, 'Colon': 0.017, 'Extrathoracic_Airways': 0.078, 'Gallbladder': 0.166, 'Heart': 0.278, 'Kidneys': 0.193, 'Liver': 0.601, 'Lungs': 0.517, 'Lymph_Nodes': 0.137, 'Muscle': 0.115, 'Oesophagus': 0.229, 'Oral_Mucosa': 0.025, 'Ovaries': 0.002, 'Pancreas': 0.121, 'Prostate': 0, 'Salivary_Glands': 0.024, 'Endosteum': 0.284, 'Skin': 0.155, 'Small_Intestine': 0.019, 'Spleen': 0.043, 'Stomach': 0.055, 'Testicles': 0, 'Thymus': 0.259, 'Thyroid': 0.403, 'Urinary_Bladder': 0.001, 'Uterus': 0.002},
                        (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.157, 'Adrenals': 0.348, 'Brain': 0.005, 'Breast': 0.986, 'Colon': 0.027, 'Extrathoracic_Airways': 0.109, 'Gallbladder': 0.246, 'Heart': 0.403, 'Kidneys': 0.272, 'Liver': 0.793, 'Lungs': 0.691, 'Lymph_Nodes': 0.193, 'Muscle': 0.15, 'Oesophagus': 0.338, 'Oral_Mucosa': 0.036, 'Ovaries': 0.004, 'Pancreas': 0.188, 'Prostate': 0, 'Salivary_Glands': 0.035, 'Endosteum': 0.364, 'Skin': 0.172, 'Small_Intestine': 0.03, 'Spleen': 0.07, 'Stomach': 0.088, 'Testicles': 0, 'Thymus': 0.375, 'Thyroid': 0.517, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                        (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.124, 'Adrenals': 0.25, 'Brain': 0.003, 'Breast': 0.92, 'Colon': 0.019, 'Extrathoracic_Airways': 0.084, 'Gallbladder': 0.18, 'Heart': 0.3, 'Kidneys': 0.207, 'Liver': 0.637, 'Lungs': 0.549, 'Lymph_Nodes': 0.147, 'Muscle': 0.122, 'Oesophagus': 0.248, 'Oral_Mucosa': 0.027, 'Ovaries': 0.002, 'Pancreas': 0.132, 'Prostate': 0, 'Salivary_Glands': 0.026, 'Endosteum': 0.299, 'Skin': 0.158, 'Small_Intestine': 0.021, 'Spleen': 0.048, 'Stomach': 0.06, 'Testicles': 0, 'Thymus': 0.28, 'Thyroid': 0.424, 'Urinary_Bladder': 0.001, 'Uterus': 0.003},
                    },
                },
            "Lumbar Spine": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.149, 'Adrenals': 0.367, 'Brain': 0.0, 'Breast': 0.006, 'Colon': 0.5339999999999999, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.446, 'Heart': 0.07200000000000001, 'Kidneys': 0.8019999999999999, 'Liver': 0.147, 'Lungs': 0.053, 'Lymph_Nodes': 0.45199999999999996, 'Muscle': 0.258, 'Oesophagus': 0.097, 'Oral_Mucosa': 0.0, 'Ovaries': 0.189, 'Pancreas': 1.006, 'Prostate': 0.025, 'Salivary_Glands': 0.0, 'Endosteum': 0.19699999999999998, 'Skin': 0.13799999999999998, 'Small_Intestine': 0.73, 'Spleen': 2.421, 'Stomach': 1.686, 'Testicles': 0.006, 'Thymus': 0.008, 'Thyroid': 0.0, 'Urinary_Bladder': 0.041999999999999996, 'Uterus': 0.122},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.162, 'Adrenals': 0.406, 'Brain': 0.0, 'Breast': 0.007, 'Colon': 0.583, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.49299999999999994, 'Heart': 0.079, 'Kidneys': 0.871, 'Liver': 0.163, 'Lungs': 0.058, 'Lymph_Nodes': 0.49299999999999994, 'Muscle': 0.275, 'Oesophagus': 0.108, 'Oral_Mucosa': 0.0, 'Ovaries': 0.21000000000000002, 'Pancreas': 1.0970000000000002, 'Prostate': 0.028, 'Salivary_Glands': 0.0, 'Endosteum': 0.213, 'Skin': 0.13999999999999999, 'Small_Intestine': 0.798, 'Spleen': 2.6020000000000003, 'Stomach': 1.82, 'Testicles': 0.007, 'Thymus': 0.009000000000000001, 'Thyroid': 0.0, 'Urinary_Bladder': 0.047, 'Uterus': 0.134},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.232, 'Adrenals': 0.623, 'Brain': 0.0, 'Breast': 0.012, 'Colon': 0.8310000000000001, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.75, 'Heart': 0.122, 'Kidneys': 1.213, 'Liver': 0.253, 'Lungs': 0.087, 'Lymph_Nodes': 0.7060000000000001, 'Muscle': 0.354, 'Oesophagus': 0.16899999999999998, 'Oral_Mucosa': 0.0, 'Ovaries': 0.33599999999999997, 'Pancreas': 1.56, 'Prostate': 0.046, 'Salivary_Glands': 0.0, 'Endosteum': 0.29, 'Skin': 0.155, 'Small_Intestine': 1.148, 'Spleen': 3.437, 'Stomach': 2.471, 'Testicles': 0.011, 'Thymus': 0.015000000000000001, 'Thyroid': 0.0, 'Urinary_Bladder': 0.076, 'Uterus': 0.199},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.174, 'Adrenals': 0.442, 'Brain': 0.0, 'Breast': 0.008, 'Colon': 0.627, 'Extrathoracic_Airways': 0.001, 'Gallbladder': 0.536, 'Heart': 0.08600000000000001, 'Kidneys': 0.933, 'Liver': 0.178, 'Lungs': 0.063, 'Lymph_Nodes': 0.53, 'Muscle': 0.29, 'Oesophagus': 0.118, 'Oral_Mucosa': 0.0, 'Ovaries': 0.231, 'Pancreas': 1.178, 'Prostate': 0.031, 'Salivary_Glands': 0.0, 'Endosteum': 0.22699999999999998, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.859, 'Spleen': 2.76, 'Stomach': 1.9400000000000002, 'Testicles': 0.007, 'Thymus': 0.01, 'Thyroid': 0.0, 'Urinary_Bladder': 0.051, 'Uterus': 0.145},
                },},
                        "Thoracic Spine": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.193, 'Adrenals': 0.297, 'Brain': 0.004, 'Breast': 0.063, 'Colon': 0.018000000000000002, 'Endosteum': 0.6980000000000001, 'Extrathoracic_Airways': 0.023, 'Gallbladder': 0.095, 'Heart': 0.415, 'Kidneys': 0.27799999999999997, 'Liver': 0.068, 'Lungs': 0.5109999999999999, 'Lymph_Nodes': 0.16699999999999998, 'Muscle': 0.16, 'Oesophagus': 0.292, 'Oral_Mucosa': 0.012, 'Ovaries': 0.020999999999999998, 'Pancreas': 0.359, 'Prostate': 0.0, 'Salivary_Glands': 0.027, 'Skin': 0.245, 'Small_Intestine': 0.019, 'Spleen': 0.98, 'Stomach': 0.48899999999999993, 'Testicles': 0.0, 'Thymus': 0.161, 'Thyroid': 0.07100000000000001, 'Urinary_Bladder': 0.002, 'Uterus': 0.005},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.208, 'Adrenals': 0.32699999999999996, 'Brain': 0.005, 'Breast': 0.06899999999999999, 'Colon': 0.02, 'Endosteum': 0.741, 'Extrathoracic_Airways': 0.025, 'Gallbladder': 0.10500000000000001, 'Heart': 0.455, 'Kidneys': 0.30200000000000005, 'Liver': 0.075, 'Lungs': 0.5559999999999999, 'Lymph_Nodes': 0.183, 'Muscle': 0.171, 'Oesophagus': 0.321, 'Oral_Mucosa': 0.014, 'Ovaries': 0.023, 'Pancreas': 0.395, 'Prostate': 0.0, 'Salivary_Glands': 0.030000000000000002, 'Skin': 0.25, 'Small_Intestine': 0.022, 'Spleen': 1.064, 'Stomach': 0.533, 'Testicles': 0.0, 'Thymus': 0.177, 'Thyroid': 0.079, 'Urinary_Bladder': 0.002, 'Uterus': 0.006},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.221, 'Adrenals': 0.354, 'Brain': 0.005, 'Breast': 0.074, 'Colon': 0.020999999999999998, 'Endosteum': 0.7789999999999999, 'Extrathoracic_Airways': 0.028, 'Gallbladder': 0.115, 'Heart': 0.491, 'Kidneys': 0.324, 'Liver': 0.082, 'Lungs': 0.597, 'Lymph_Nodes': 0.19699999999999998, 'Muscle': 0.182, 'Oesophagus': 0.348, 'Oral_Mucosa': 0.015000000000000001, 'Ovaries': 0.025, 'Pancreas': 0.428, 'Prostate': 0.0, 'Salivary_Glands': 0.033, 'Skin': 0.254, 'Small_Intestine': 0.024, 'Spleen': 1.138, 'Stomach': 0.5730000000000001, 'Testicles': 0.0, 'Thymus': 0.193, 'Thyroid': 0.08600000000000001, 'Urinary_Bladder': 0.003, 'Uterus': 0.007},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.281, 'Adrenals': 0.486, 'Brain': 0.007, 'Breast': 0.09899999999999999, 'Colon': 0.031, 'Endosteum': 0.9349999999999999, 'Extrathoracic_Airways': 0.04, 'Gallbladder': 0.164, 'Heart': 0.662, 'Kidneys': 0.423, 'Liver': 0.11699999999999999, 'Lungs': 0.782, 'Lymph_Nodes': 0.26699999999999996, 'Muscle': 0.229, 'Oesophagus': 0.479, 'Oral_Mucosa': 0.020999999999999998, 'Ovaries': 0.030000000000000002, 'Pancreas': 0.5900000000000001, 'Prostate': 0.0, 'Salivary_Glands': 0.046, 'Skin': 0.275, 'Small_Intestine': 0.034, 'Spleen': 1.4760000000000002, 'Stomach': 0.757, 'Testicles': 0.0, 'Thymus': 0.268, 'Thyroid': 0.122, 'Urinary_Bladder': 0.004, 'Uterus': 0.011},
                },},
                "Scoliosis": {
                    "U": {
                        (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.168, 'Adrenals': 0.183, 'Brain': 0.004, 'Breast': 0.707, 'Colon': 0.24, 'Extrathoracic_Airways': 0.164, 'Gallbladder': 0.286, 'Heart': 0.291, 'Kidneys': 0.29, 'Liver': 0.714, 'Lungs': 0.481, 'Lymph_Nodes': 0.228, 'Muscle': 0.18, 'Oesophagus': 0.256, 'Oral_Mucosa': 0.031, 'Ovaries': 0.275, 'Pancreas': 0.139, 'Prostate': 0.01, 'Salivary_Glands': 0.062, 'Endosteum': 0.329, 'Skin': 0.234, 'Small_Intestine': 0.232, 'Spleen': 0.048, 'Stomach': 0.079, 'Testicles': 0, 'Thymus': 0.251, 'Thyroid': 0.519, 'Urinary_Bladder': 0.051, 'Uterus': 0.165},
                        (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.181, 'Adrenals': 0.201, 'Brain': 0.005, 'Breast': 0.716, 'Colon': 0.26, 'Extrathoracic_Airways': 0.175, 'Gallbladder': 0.314, 'Heart': 0.317, 'Kidneys': 0.314, 'Liver': 0.765, 'Lungs': 0.517, 'Lymph_Nodes': 0.247, 'Muscle': 0.192, 'Oesophagus': 0.28, 'Oral_Mucosa': 0.034, 'Ovaries': 0.3, 'Pancreas': 0.154, 'Prostate': 0.01, 'Salivary_Glands': 0.067, 'Endosteum': 0.35, 'Skin': 0.239, 'Small_Intestine': 0.254, 'Spleen': 0.053, 'Stomach': 0.087, 'Testicles': 0, 'Thymus': 0.273, 'Thyroid': 0.548, 'Urinary_Bladder': 0.057, 'Uterus': 0.182},
                        (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.245, 'Adrenals': 0.305, 'Brain': 0.008, 'Breast': 0.753, 'Colon': 0.363, 'Extrathoracic_Airways': 0.231, 'Gallbladder': 0.461, 'Heart': 0.45, 'Kidneys': 0.435, 'Liver': 1.005, 'Lungs': 0.687, 'Lymph_Nodes': 0.344, 'Muscle': 0.246, 'Oesophagus': 0.406, 'Oral_Mucosa': 0.048, 'Ovaries': 0.431, 'Pancreas': 0.233, 'Prostate': 0.013, 'Salivary_Glands': 0.09, 'Endosteum': 0.45, 'Skin': 0.264, 'Small_Intestine': 0.365, 'Spleen': 0.083, 'Stomach': 0.134, 'Testicles': 0, 'Thymus': 0.379, 'Thyroid': 0.676, 'Urinary_Bladder': 0.089, 'Uterus': 0.275},
                        (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.192, 'Adrenals': 0.218, 'Brain': 0.005, 'Breast': 0.723, 'Colon': 0.278, 'Extrathoracic_Airways': 0.185, 'Gallbladder': 0.339, 'Heart': 0.34, 'Kidneys': 0.336, 'Liver': 0.809, 'Lungs': 0.548, 'Lymph_Nodes': 0.264, 'Muscle': 0.202, 'Oesophagus': 0.301, 'Oral_Mucosa': 0.036, 'Ovaries': 0.323, 'Pancreas': 0.167, 'Prostate': 0.011, 'Salivary_Glands': 0.072, 'Endosteum': 0.369, 'Skin': 0.244, 'Small_Intestine': 0.273, 'Spleen': 0.058, 'Stomach': 0.095, 'Testicles': 0, 'Thymus': 0.292, 'Thyroid': 0.573, 'Urinary_Bladder': 0.062, 'Uterus': 0.198},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.0, "Brain": 0.028, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.19, "Extrathoracic_Airways": 1.105, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.333, "Muscle": 0.042, "Oesophagus": 0.029, "Oral_Mucosa": 1.14, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.684, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.008, "Thyroid": 2.587, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.043, "Adrenals": 0.0, "Brain": 0.032, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.207, "Extrathoracic_Airways": 1.194, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.012, "Lymph_Nodes": 0.357, "Muscle": 0.045, "Oesophagus": 0.032, "Oral_Mucosa": 1.215, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.744, "Skin": 0.155, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.009, "Thyroid": 2.725, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.0, "Brain": 0.036, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.222, "Extrathoracic_Airways": 1.273, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.013, "Lymph_Nodes": 0.377, "Muscle": 0.048, "Oesophagus": 0.035, "Oral_Mucosa": 1.281, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.798, "Skin": 0.159, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.01, "Thyroid": 2.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.06, "Adrenals": 0.0, "Brain": 0.061, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.258, "Extrathoracic_Airways": 1.474, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.019, "Lymph_Nodes": 0.422, "Muscle": 0.054, "Oesophagus": 0.044, "Oral_Mucosa": 1.447, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.939, "Skin": 0.164, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.018, "Thyroid": 2.96, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.0, "Brain": 0.068, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.278, "Extrathoracic_Airways": 1.577, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.448, "Muscle": 0.057, "Oesophagus": 0.048, "Oral_Mucosa": 1.534, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.01, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.02, "Thyroid": 3.092, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.069, "Adrenals": 0.0, "Brain": 0.074, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.295, "Extrathoracic_Airways": 1.669, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.471, "Muscle": 0.06, "Oesophagus": 0.051, "Oral_Mucosa": 1.609, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.073, "Skin": 0.172, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.021, "Thyroid": 3.204, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.0, "Brain": 0.094, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.308, "Extrathoracic_Airways": 1.77, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.496, "Muscle": 0.063, "Oesophagus": 0.06, "Oral_Mucosa": 1.707, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.167, "Skin": 0.175, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.025, "Thyroid": 3.259, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.0, "Brain": 0.103, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.327, "Extrathoracic_Airways": 1.873, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.028, "Lymph_Nodes": 0.522, "Muscle": 0.067, "Oesophagus": 0.064, "Oral_Mucosa": 1.794, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.241, "Skin": 0.179, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.027, "Thyroid": 3.38, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.0, "Brain": 0.11, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.344, "Extrathoracic_Airways": 1.963, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.03, "Lymph_Nodes": 0.544, "Muscle": 0.069, "Oesophagus": 0.068, "Oral_Mucosa": 1.87, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.306, "Skin": 0.183, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.029, "Thyroid": 3.481, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.094, "Adrenals": 0.0, "Brain": 0.123, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.342, "Extrathoracic_Airways": 1.991, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.007, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.031, "Lymph_Nodes": 0.552, "Muscle": 0.071, "Oesophagus": 0.074, "Oral_Mucosa": 1.911, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.353, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.03, "Thyroid": 3.473, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.1, "Adrenals": 0.0, "Brain": 0.131, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.358, "Extrathoracic_Airways": 2.085, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.033, "Lymph_Nodes": 0.576, "Muscle": 0.074, "Oesophagus": 0.078, "Oral_Mucosa": 1.992, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.424, "Skin": 0.188, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.032, "Thyroid": 3.577, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.104, "Adrenals": 0.0, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.373, "Extrathoracic_Airways": 2.167, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.035, "Lymph_Nodes": 0.596, "Muscle": 0.076, "Oesophagus": 0.082, "Oral_Mucosa": 2.062, "Pancreas": 0.002, "Prostate": 0.0, "Salivary_Glands": 1.486, "Skin": 0.191, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.034, "Thyroid": 3.664, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.11299999999999999, 'Adrenals': 0.0, 'Brain': 0.627, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.266, 'Gallbladder': 0.001, 'Heart': 0.002, 'Kidneys': 0.001, 'Liver': 0.0, 'Lungs': 0.006, 'Lymph_Nodes': 0.15, 'Muscle': 0.05, 'Oesophagus': 0.007, 'Oral_Mucosa': 0.488, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.309, 'Endosteum': 0.5379999999999999, 'Skin': 0.184, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.005, 'Thyroid': 0.064, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.121, 'Adrenals': 0.0, 'Brain': 0.684, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.292, 'Gallbladder': 0.001, 'Heart': 0.002, 'Kidneys': 0.001, 'Liver': 0.0, 'Lungs': 0.007, 'Lymph_Nodes': 0.16, 'Muscle': 0.053, 'Oesophagus': 0.008, 'Oral_Mucosa': 0.527, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.3760000000000001, 'Endosteum': 0.5730000000000001, 'Skin': 0.188, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.006, 'Thyroid': 0.07200000000000001, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.161, 'Adrenals': 0.0, 'Brain': 0.973, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.433, 'Gallbladder': 0.001, 'Heart': 0.003, 'Kidneys': 0.001, 'Liver': 0.001, 'Lungs': 0.011, 'Lymph_Nodes': 0.207, 'Muscle': 0.06499999999999999, 'Oesophagus': 0.012, 'Oral_Mucosa': 0.717, 'Ovaries': 0.0, 'Pancreas': 0.001, 'Prostate': 0.0, 'Salivary_Glands': 1.671, 'Endosteum': 0.736, 'Skin': 0.205, 'Small_Intestine': 0.0, 'Spleen': 0.001, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.009000000000000001, 'Thyroid': 0.114, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.128, 'Adrenals': 0.0, 'Brain': 0.735, 'Breast': 0.0, 'Colon': 0.0, 'Extrathoracic_Airways': 0.316, 'Gallbladder': 0.001, 'Heart': 0.002, 'Kidneys': 0.001, 'Liver': 0.001, 'Lungs': 0.008, 'Lymph_Nodes': 0.16899999999999998, 'Muscle': 0.055, 'Oesophagus': 0.008, 'Oral_Mucosa': 0.5609999999999999, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.434, 'Endosteum': 0.6040000000000001, 'Skin': 0.191, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.001, 'Testicles': 0.0, 'Thymus': 0.006, 'Thyroid': 0.078, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
        15: {
            "Validation": _VALIDATION_ORGAN_REGION,
                "Chest": {
                    "U": {
                        (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.108, 'Adrenals': 0.129, 'Brain': 0.003, 'Breast': 0.429, 'Colon': 0.009, 'Extrathoracic_Airways': 0.048, 'Gallbladder': 0.115, 'Heart': 0.193, 'Kidneys': 0.099, 'Liver': 0.443, 'Lungs': 0.388, 'Lymph_Nodes': 0.096, 'Muscle': 0.088, 'Oesophagus': 0.203, 'Oral_Mucosa': 0.007, 'Ovaries': 0.002, 'Pancreas': 0.096, 'Prostate': 0, 'Salivary_Glands': 0.016, 'Endosteum': 0.21, 'Skin': 0.143, 'Small_Intestine': 0.009, 'Spleen': 0.024, 'Stomach': 0.034, 'Testicles': 0.001, 'Thymus': 0.159, 'Thyroid': 0.203, 'Urinary_Bladder': 0.001, 'Uterus': 0.003},
                        (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.114, 'Adrenals': 0.139, 'Brain': 0.003, 'Breast': 0.448, 'Colon': 0.01, 'Extrathoracic_Airways': 0.051, 'Gallbladder': 0.124, 'Heart': 0.208, 'Kidneys': 0.106, 'Liver': 0.471, 'Lungs': 0.413, 'Lymph_Nodes': 0.102, 'Muscle': 0.093, 'Oesophagus': 0.22, 'Oral_Mucosa': 0.008, 'Ovaries': 0.003, 'Pancreas': 0.104, 'Prostate': 0, 'Salivary_Glands': 0.017, 'Endosteum': 0.221, 'Skin': 0.146, 'Small_Intestine': 0.01, 'Spleen': 0.026, 'Stomach': 0.037, 'Testicles': 0.001, 'Thymus': 0.172, 'Thyroid': 0.215, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                        (100, 3.0, 0.1): {'Active_Bone_Marrow': 0.147, 'Adrenals': 0.191, 'Brain': 0.004, 'Breast': 0.53, 'Colon': 0.014, 'Extrathoracic_Airways': 0.068, 'Gallbladder': 0.169, 'Heart': 0.287, 'Kidneys': 0.145, 'Liver': 0.607, 'Lungs': 0.53, 'Lymph_Nodes': 0.137, 'Muscle': 0.118, 'Oesophagus': 0.309, 'Oral_Mucosa': 0.012, 'Ovaries': 0.004, 'Pancreas': 0.147, 'Prostate': 0, 'Salivary_Glands': 0.023, 'Endosteum': 0.269, 'Skin': 0.161, 'Small_Intestine': 0.015, 'Spleen': 0.038, 'Stomach': 0.054, 'Testicles': 0.001, 'Thymus': 0.237, 'Thyroid': 0.268, 'Urinary_Bladder': 0.001, 'Uterus': 0.005},
                        (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.12, 'Adrenals': 0.148, 'Brain': 0.003, 'Breast': 0.465, 'Colon': 0.011, 'Extrathoracic_Airways': 0.054, 'Gallbladder': 0.132, 'Heart': 0.222, 'Kidneys': 0.113, 'Liver': 0.496, 'Lungs': 0.435, 'Lymph_Nodes': 0.109, 'Muscle': 0.098, 'Oesophagus': 0.235, 'Oral_Mucosa': 0.008, 'Ovaries': 0.003, 'Pancreas': 0.111, 'Prostate': 0, 'Salivary_Glands': 0.018, 'Endosteum': 0.231, 'Skin': 0.149, 'Small_Intestine': 0.011, 'Spleen': 0.028, 'Stomach': 0.04, 'Testicles': 0.001, 'Thymus': 0.183, 'Thyroid': 0.225, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                    },
                },
            "Lumbar Spine": {                "U": {
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.10900000000000001, 'Adrenals': 0.175, 'Brain': 0.0, 'Breast': 0.01, 'Colon': 0.27399999999999997, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.108, 'Heart': 0.04, 'Kidneys': 0.361, 'Liver': 0.046, 'Lungs': 0.030000000000000002, 'Lymph_Nodes': 0.199, 'Muscle': 0.154, 'Oesophagus': 0.074, 'Oral_Mucosa': 0.0, 'Ovaries': 0.206, 'Pancreas': 0.453, 'Prostate': 0.016, 'Salivary_Glands': 0.0, 'Endosteum': 0.12000000000000001, 'Skin': 0.199, 'Small_Intestine': 0.32699999999999996, 'Spleen': 1.174, 'Stomach': 0.7929999999999999, 'Testicles': 0.005, 'Thymus': 0.002, 'Thyroid': 0.002, 'Urinary_Bladder': 0.034999999999999996, 'Uterus': 0.134},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.11900000000000001, 'Adrenals': 0.192, 'Brain': 0.0, 'Breast': 0.011, 'Colon': 0.3, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.12000000000000001, 'Heart': 0.044, 'Kidneys': 0.392, 'Liver': 0.052, 'Lungs': 0.033, 'Lymph_Nodes': 0.217, 'Muscle': 0.165, 'Oesophagus': 0.082, 'Oral_Mucosa': 0.0, 'Ovaries': 0.22599999999999998, 'Pancreas': 0.494, 'Prostate': 0.018000000000000002, 'Salivary_Glands': 0.0, 'Endosteum': 0.129, 'Skin': 0.204, 'Small_Intestine': 0.357, 'Spleen': 1.262, 'Stomach': 0.859, 'Testicles': 0.005, 'Thymus': 0.002, 'Thyroid': 0.002, 'Urinary_Bladder': 0.039, 'Uterus': 0.148},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.17, 'Adrenals': 0.29100000000000004, 'Brain': 0.0, 'Breast': 0.016, 'Colon': 0.431, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.188, 'Heart': 0.068, 'Kidneys': 0.5509999999999999, 'Liver': 0.081, 'Lungs': 0.05, 'Lymph_Nodes': 0.312, 'Muscle': 0.213, 'Oesophagus': 0.125, 'Oral_Mucosa': 0.0, 'Ovaries': 0.335, 'Pancreas': 0.705, 'Prostate': 0.027, 'Salivary_Glands': 0.0, 'Endosteum': 0.176, 'Skin': 0.225, 'Small_Intestine': 0.521, 'Spleen': 1.677, 'Stomach': 1.1869999999999998, 'Testicles': 0.009000000000000001, 'Thymus': 0.003, 'Thyroid': 0.003, 'Urinary_Bladder': 0.060000000000000005, 'Uterus': 0.231},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.128, 'Adrenals': 0.20900000000000002, 'Brain': 0.0, 'Breast': 0.011, 'Colon': 0.322, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.131, 'Heart': 0.048, 'Kidneys': 0.42000000000000004, 'Liver': 0.056, 'Lungs': 0.036000000000000004, 'Lymph_Nodes': 0.23299999999999998, 'Muscle': 0.17300000000000001, 'Oesophagus': 0.089, 'Oral_Mucosa': 0.0, 'Ovaries': 0.245, 'Pancreas': 0.531, 'Prostate': 0.019, 'Salivary_Glands': 0.0, 'Endosteum': 0.13799999999999998, 'Skin': 0.208, 'Small_Intestine': 0.386, 'Spleen': 1.34, 'Stomach': 0.9179999999999999, 'Testicles': 0.006, 'Thymus': 0.002, 'Thyroid': 0.002, 'Urinary_Bladder': 0.041999999999999996, 'Uterus': 0.162},
                },},
                        "Thoracic Spine": {                "U": {
                    (100, 2.5, 0.0): {'Active_Bone_Marrow': 0.163, 'Adrenals': 0.213, 'Brain': 0.004, 'Breast': 0.085, 'Colon': 0.011, 'Endosteum': 0.505, 'Extrathoracic_Airways': 0.067, 'Gallbladder': 0.049, 'Heart': 0.29300000000000004, 'Kidneys': 0.13899999999999998, 'Liver': 0.045000000000000005, 'Lungs': 0.372, 'Lymph_Nodes': 0.134, 'Muscle': 0.12300000000000001, 'Oesophagus': 0.25, 'Oral_Mucosa': 0.014, 'Ovaries': 0.006, 'Pancreas': 0.257, 'Prostate': 0.0, 'Salivary_Glands': 0.025, 'Skin': 0.184, 'Small_Intestine': 0.012, 'Spleen': 0.656, 'Stomach': 0.34099999999999997, 'Testicles': 0.0, 'Thymus': 0.18000000000000002, 'Thyroid': 0.21000000000000002, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                    (100, 3.0, 0.0): {'Active_Bone_Marrow': 0.17300000000000001, 'Adrenals': 0.231, 'Brain': 0.005, 'Breast': 0.092, 'Colon': 0.012, 'Endosteum': 0.53, 'Extrathoracic_Airways': 0.07200000000000001, 'Gallbladder': 0.053, 'Heart': 0.317, 'Kidneys': 0.149, 'Liver': 0.049, 'Lungs': 0.399, 'Lymph_Nodes': 0.14400000000000002, 'Muscle': 0.12999999999999998, 'Oesophagus': 0.27, 'Oral_Mucosa': 0.015000000000000001, 'Ovaries': 0.007, 'Pancreas': 0.277, 'Prostate': 0.0, 'Salivary_Glands': 0.027, 'Skin': 0.188, 'Small_Intestine': 0.013, 'Spleen': 0.703, 'Stomach': 0.366, 'Testicles': 0.0, 'Thymus': 0.195, 'Thyroid': 0.224, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                    (100, 3.5, 0.0): {'Active_Bone_Marrow': 0.182, 'Adrenals': 0.247, 'Brain': 0.005, 'Breast': 0.097, 'Colon': 0.013, 'Endosteum': 0.5519999999999999, 'Extrathoracic_Airways': 0.075, 'Gallbladder': 0.057, 'Heart': 0.33799999999999997, 'Kidneys': 0.159, 'Liver': 0.053, 'Lungs': 0.424, 'Lymph_Nodes': 0.153, 'Muscle': 0.13699999999999998, 'Oesophagus': 0.289, 'Oral_Mucosa': 0.017, 'Ovaries': 0.008, 'Pancreas': 0.296, 'Prostate': 0.0, 'Salivary_Glands': 0.028, 'Skin': 0.191, 'Small_Intestine': 0.014, 'Spleen': 0.745, 'Stomach': 0.39, 'Testicles': 0.0, 'Thymus': 0.20900000000000002, 'Thyroid': 0.237, 'Urinary_Bladder': 0.001, 'Uterus': 0.004},
                    (100, 3.0, 0.1): {'Active_Bone_Marrow': 0.225, 'Adrenals': 0.324, 'Brain': 0.006, 'Breast': 0.122, 'Colon': 0.018000000000000002, 'Endosteum': 0.639, 'Extrathoracic_Airways': 0.092, 'Gallbladder': 0.079, 'Heart': 0.44, 'Kidneys': 0.204, 'Liver': 0.07200000000000001, 'Lungs': 0.536, 'Lymph_Nodes': 0.196, 'Muscle': 0.16699999999999998, 'Oesophagus': 0.379, 'Oral_Mucosa': 0.022, 'Ovaries': 0.011, 'Pancreas': 0.384, 'Prostate': 0.0, 'Salivary_Glands': 0.037, 'Skin': 0.205, 'Small_Intestine': 0.019, 'Spleen': 0.9359999999999999, 'Stomach': 0.49700000000000005, 'Testicles': 0.0, 'Thymus': 0.277, 'Thyroid': 0.294, 'Urinary_Bladder': 0.002, 'Uterus': 0.006},
                },},
                "Scoliosis": {
                    "U": {
                        (85, 2.5, 0.0): {'Active_Bone_Marrow': 0.12, 'Adrenals': 0.156, 'Brain': 0.004, 'Breast': 0.202, 'Colon': 0.121, 'Extrathoracic_Airways': 0.212, 'Gallbladder': 0.279, 'Heart': 0.156, 'Kidneys': 0.19, 'Liver': 0.551, 'Lungs': 0.325, 'Lymph_Nodes': 0.17, 'Muscle': 0.12, 'Oesophagus': 0.143, 'Oral_Mucosa': 0.038, 'Ovaries': 0.034, 'Pancreas': 0.089, 'Prostate': 0.003, 'Salivary_Glands': 0.116, 'Endosteum': 0.237, 'Skin': 0.2, 'Small_Intestine': 0.122, 'Spleen': 0.025, 'Stomach': 0.036, 'Testicles': 0, 'Thymus': 0.145, 'Thyroid': 0.371, 'Urinary_Bladder': 0.008, 'Uterus': 0.02},
                        (85, 3.0, 0.0): {'Active_Bone_Marrow': 0.129, 'Adrenals': 0.172, 'Brain': 0.004, 'Breast': 0.212, 'Colon': 0.131, 'Extrathoracic_Airways': 0.228, 'Gallbladder': 0.304, 'Heart': 0.17, 'Kidneys': 0.206, 'Liver': 0.589, 'Lungs': 0.351, 'Lymph_Nodes': 0.184, 'Muscle': 0.128, 'Oesophagus': 0.157, 'Oral_Mucosa': 0.041, 'Ovaries': 0.038, 'Pancreas': 0.097, 'Prostate': 0.003, 'Salivary_Glands': 0.126, 'Endosteum': 0.253, 'Skin': 0.205, 'Small_Intestine': 0.133, 'Spleen': 0.028, 'Stomach': 0.04, 'Testicles': 0, 'Thymus': 0.159, 'Thyroid': 0.392, 'Urinary_Bladder': 0.009, 'Uterus': 0.023},
                        (85, 3.0, 0.1): {'Active_Bone_Marrow': 0.175, 'Adrenals': 0.262, 'Brain': 0.007, 'Breast': 0.259, 'Colon': 0.185, 'Extrathoracic_Airways': 0.304, 'Gallbladder': 0.433, 'Heart': 0.248, 'Kidneys': 0.286, 'Liver': 0.774, 'Lungs': 0.474, 'Lymph_Nodes': 0.255, 'Muscle': 0.165, 'Oesophagus': 0.226, 'Oral_Mucosa': 0.06, 'Ovaries': 0.062, 'Pancreas': 0.147, 'Prostate': 0.005, 'Salivary_Glands': 0.171, 'Endosteum': 0.324, 'Skin': 0.228, 'Small_Intestine': 0.191, 'Spleen': 0.043, 'Stomach': 0.061, 'Testicles': 0, 'Thymus': 0.235, 'Thyroid': 0.487, 'Urinary_Bladder': 0.013, 'Uterus': 0.035},
                        (85, 3.5, 0.0): {'Active_Bone_Marrow': 0.137, 'Adrenals': 0.187, 'Brain': 0.005, 'Breast': 0.221, 'Colon': 0.141, 'Extrathoracic_Airways': 0.242, 'Gallbladder': 0.327, 'Heart': 0.183, 'Kidneys': 0.22, 'Liver': 0.624, 'Lungs': 0.373, 'Lymph_Nodes': 0.197, 'Muscle': 0.134, 'Oesophagus': 0.169, 'Oral_Mucosa': 0.044, 'Ovaries': 0.042, 'Pancreas': 0.106, 'Prostate': 0.003, 'Salivary_Glands': 0.134, 'Endosteum': 0.266, 'Skin': 0.21, 'Small_Intestine': 0.143, 'Spleen': 0.03, 'Stomach': 0.044, 'Testicles': 0, 'Thymus': 0.172, 'Thyroid': 0.411, 'Urinary_Bladder': 0.009, 'Uterus': 0.025},
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.0, "Brain": 0.028, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.19, "Extrathoracic_Airways": 1.105, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.011, "Lymph_Nodes": 0.333, "Muscle": 0.042, "Oesophagus": 0.029, "Oral_Mucosa": 1.14, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.684, "Skin": 0.151, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.008, "Thyroid": 2.587, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.043, "Adrenals": 0.0, "Brain": 0.032, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.207, "Extrathoracic_Airways": 1.194, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.012, "Lymph_Nodes": 0.357, "Muscle": 0.045, "Oesophagus": 0.032, "Oral_Mucosa": 1.215, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.744, "Skin": 0.155, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.009, "Thyroid": 2.725, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.0, "Brain": 0.036, "Breast": 0.001, "Colon": 0.0, "Endosteum": 0.222, "Extrathoracic_Airways": 1.273, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.002, "Kidneys": 0.0, "Liver": 0.0, "Lungs": 0.013, "Lymph_Nodes": 0.377, "Muscle": 0.048, "Oesophagus": 0.035, "Oral_Mucosa": 1.281, "Pancreas": 0.0, "Prostate": 0.0, "Salivary_Glands": 0.798, "Skin": 0.159, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.01, "Thyroid": 2.841, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.06, "Adrenals": 0.0, "Brain": 0.061, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.258, "Extrathoracic_Airways": 1.474, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.003, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.019, "Lymph_Nodes": 0.422, "Muscle": 0.054, "Oesophagus": 0.044, "Oral_Mucosa": 1.447, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 0.939, "Skin": 0.164, "Small_Intestine": 0.0, "Spleen": 0.001, "Stomach": 0.0, "Thymus": 0.018, "Thyroid": 2.96, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.0, "Brain": 0.068, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.278, "Extrathoracic_Airways": 1.577, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.021, "Lymph_Nodes": 0.448, "Muscle": 0.057, "Oesophagus": 0.048, "Oral_Mucosa": 1.534, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.01, "Skin": 0.168, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.02, "Thyroid": 3.092, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.069, "Adrenals": 0.0, "Brain": 0.074, "Breast": 0.003, "Colon": 0.0, "Endosteum": 0.295, "Extrathoracic_Airways": 1.669, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.004, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.022, "Lymph_Nodes": 0.471, "Muscle": 0.06, "Oesophagus": 0.051, "Oral_Mucosa": 1.609, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.073, "Skin": 0.172, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.0, "Thymus": 0.021, "Thyroid": 3.204, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.079, "Adrenals": 0.0, "Brain": 0.094, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.308, "Extrathoracic_Airways": 1.77, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.005, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.026, "Lymph_Nodes": 0.496, "Muscle": 0.063, "Oesophagus": 0.06, "Oral_Mucosa": 1.707, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.167, "Skin": 0.175, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.025, "Thyroid": 3.259, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.084, "Adrenals": 0.0, "Brain": 0.103, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.327, "Extrathoracic_Airways": 1.873, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.001, "Lungs": 0.028, "Lymph_Nodes": 0.522, "Muscle": 0.067, "Oesophagus": 0.064, "Oral_Mucosa": 1.794, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.241, "Skin": 0.179, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.027, "Thyroid": 3.38, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.089, "Adrenals": 0.0, "Brain": 0.11, "Breast": 0.005, "Colon": 0.0, "Endosteum": 0.344, "Extrathoracic_Airways": 1.963, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.006, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.03, "Lymph_Nodes": 0.544, "Muscle": 0.069, "Oesophagus": 0.068, "Oral_Mucosa": 1.87, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.306, "Skin": 0.183, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.029, "Thyroid": 3.481, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.094, "Adrenals": 0.0, "Brain": 0.123, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.342, "Extrathoracic_Airways": 1.991, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.007, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.031, "Lymph_Nodes": 0.552, "Muscle": 0.071, "Oesophagus": 0.074, "Oral_Mucosa": 1.911, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.353, "Skin": 0.184, "Small_Intestine": 0.0, "Spleen": 0.002, "Stomach": 0.001, "Thymus": 0.03, "Thyroid": 3.473, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.1, "Adrenals": 0.0, "Brain": 0.131, "Breast": 0.006, "Colon": 0.0, "Endosteum": 0.358, "Extrathoracic_Airways": 2.085, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.033, "Lymph_Nodes": 0.576, "Muscle": 0.074, "Oesophagus": 0.078, "Oral_Mucosa": 1.992, "Pancreas": 0.001, "Prostate": 0.0, "Salivary_Glands": 1.424, "Skin": 0.188, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.032, "Thyroid": 3.577, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.104, "Adrenals": 0.0, "Brain": 0.139, "Breast": 0.007, "Colon": 0.0, "Endosteum": 0.373, "Extrathoracic_Airways": 2.167, "Gallbladder": 0.0, "Gonads": 0.0, "Heart": 0.008, "Kidneys": 0.0, "Liver": 0.002, "Lungs": 0.035, "Lymph_Nodes": 0.596, "Muscle": 0.076, "Oesophagus": 0.082, "Oral_Mucosa": 2.062, "Pancreas": 0.002, "Prostate": 0.0, "Salivary_Glands": 1.486, "Skin": 0.191, "Small_Intestine": 0.0, "Spleen": 0.003, "Stomach": 0.001, "Thymus": 0.034, "Thyroid": 3.664, "Urinary_Bladder": 0.0, "Uterus": 0.0},
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.08, 'Adrenals': 0.0, 'Brain': 0.5599999999999999, 'Breast': 0.001, 'Colon': 0.0, 'Extrathoracic_Airways': 0.176, 'Gallbladder': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.002, 'Lymph_Nodes': 0.121, 'Muscle': 0.034, 'Oesophagus': 0.002, 'Oral_Mucosa': 0.383, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.1520000000000001, 'Endosteum': 0.358, 'Skin': 0.13699999999999998, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.001, 'Thyroid': 0.031, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.08600000000000001, 'Adrenals': 0.0, 'Brain': 0.612, 'Breast': 0.001, 'Colon': 0.0, 'Extrathoracic_Airways': 0.195, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.002, 'Lymph_Nodes': 0.129, 'Muscle': 0.036000000000000004, 'Oesophagus': 0.002, 'Oral_Mucosa': 0.416, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.218, 'Endosteum': 0.383, 'Skin': 0.14100000000000001, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.001, 'Thyroid': 0.034, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.11699999999999999, 'Adrenals': 0.0, 'Brain': 0.88, 'Breast': 0.001, 'Colon': 0.0, 'Extrathoracic_Airways': 0.294, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.004, 'Lymph_Nodes': 0.16899999999999998, 'Muscle': 0.044, 'Oesophagus': 0.003, 'Oral_Mucosa': 0.579, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.512, 'Endosteum': 0.502, 'Skin': 0.155, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.002, 'Thyroid': 0.057, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.092, 'Adrenals': 0.0, 'Brain': 0.6589999999999999, 'Breast': 0.001, 'Colon': 0.0, 'Extrathoracic_Airways': 0.211, 'Gallbladder': 0.0, 'Heart': 0.001, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.003, 'Lymph_Nodes': 0.13699999999999998, 'Muscle': 0.037, 'Oesophagus': 0.002, 'Oral_Mucosa': 0.444, 'Ovaries': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 1.2750000000000001, 'Endosteum': 0.40499999999999997, 'Skin': 0.14300000000000002, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Testicles': 0.0, 'Thymus': 0.002, 'Thyroid': 0.038, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },},
        },
    },
}


# LATR effective dose coefficients (mSv / Gy·cm²)

eff_coefficients_LATR: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_EFF_REGION,
            "Chest": {
                "U": {
                    (120, 2.5, 0.0): 0.13999999999999999,
                    (120, 3.0, 0.0): 0.147,
                    (120, 3.5, 0.0): 0.153,
                    (100, 2.5, 0.0): 0.122,
                    (100, 3.0, 0.0): 0.129,
                    (100, 3.5, 0.0): 0.135,
                    (80, 2.5, 0.0): 0.09899999999999999,
                    (80, 3.0, 0.0): 0.106,
                    (80, 3.5, 0.0): 0.11299999999999999,
                    (60, 2.5, 0.0): 0.074,
                    (60, 3.0, 0.0): 0.08,
                    (60, 3.5, 0.0): 0.085,
                },},
            "Lumbar Spine": {
                "U": {
                    (120, 2.5, 0.0): 0.13899999999999998,
                    (120, 3.0, 0.0): 0.147,
                    (120, 3.5, 0.0): 0.155,
                    (100, 2.5, 0.0): 0.114,
                    (100, 3.0, 0.0): 0.12300000000000001,
                    (100, 3.5, 0.0): 0.12999999999999998,
                    (80, 2.5, 0.0): 0.08600000000000001,
                    (80, 3.0, 0.0): 0.093,
                    (80, 3.5, 0.0): 0.1,
                    (60, 2.5, 0.0): 0.054,
                    (60, 3.0, 0.0): 0.060000000000000005,
                    (60, 3.5, 0.0): 0.06499999999999999,
                },
            },
            "Thoracic Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.169,
                    (60, 3.0, 0.0): 0.182,
                    (60, 3.5, 0.0): 0.194,
                    (80, 2.5, 0.0): 0.226,
                    (80, 3.0, 0.0): 0.241,
                    (80, 3.5, 0.0): 0.255,
                    (100, 2.5, 0.0): 0.274,
                    (100, 3.0, 0.0): 0.289,
                    (100, 3.5, 0.0): 0.303,
                    (120, 2.5, 0.0): 0.311,
                    (120, 3.0, 0.0): 0.325,
                    (120, 3.5, 0.0): 0.338,
                },
            },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.146,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.162,
                    (80, 2.5, 0.0): 0.176,
                    (80, 3.0, 0.0): 0.185,
                    (80, 3.5, 0.0): 0.193,
                    (100, 2.5, 0.0): 0.201,
                    (100, 3.0, 0.0): 0.21,
                    (100, 3.5, 0.0): 0.218,
                    (120, 2.5, 0.0): 0.22,
                    (120, 3.0, 0.0): 0.228,
                    (120, 3.5, 0.0): 0.235,
                },
            },
            "Skull": {
                "U": {
                    (60, 2.5, 0.0): 0.032,
                    (60, 3.0, 0.0): 0.034,
                    (60, 3.5, 0.0): 0.036,
                    (80, 2.5, 0.0): 0.043,
                    (80, 3.0, 0.0): 0.046,
                    (80, 3.5, 0.0): 0.049,
                    (100, 2.5, 0.0): 0.054,
                    (100, 3.0, 0.0): 0.057,
                    (100, 3.5, 0.0): 0.06,
                    (120, 2.5, 0.0): 0.062,
                    (120, 3.0, 0.0): 0.065,
                    (120, 3.5, 0.0): 0.068,
                },
            },
            "Scoliosis": {
                "U": {
                    (120, 2.5, 0.0): 0.167,
                    (120, 3.0, 0.0): 0.175,
                    (120, 3.5, 0.0): 0.183,
                    (100, 2.5, 0.0): 0.143,
                    (100, 3.0, 0.0): 0.152,
                    (100, 3.5, 0.0): 0.16,
                    (80, 2.5, 0.0): 0.114,
                    (80, 3.0, 0.0): 0.122,
                    (80, 3.5, 0.0): 0.13,
                    (60, 2.5, 0.0): 0.081,
                    (60, 3.0, 0.0): 0.088,
                    (60, 3.5, 0.0): 0.094,
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_EFF_REGION,
                "Chest": {
                    "U": {
                        (65, 2.5, 0.0): 1.789,
                        (65, 3.0, 0.0): 1.889,
                        (65, 3.0, 0.1): 2.352,
                        (65, 3.5, 0.0): 1.976,
                    },
                },
            "Lumbar Spine": {                "U": {
                    (65, 2.5, 0.0): 2.977,
                    (65, 3.0, 0.0): 3.193,
                    (65, 3.0, 0.1): 4.172000000000001,
                    (65, 3.5, 0.0): 3.381,
                },},
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): 1.596,
                    (65, 3.0, 0.0): 1.718,
                    (65, 3.5, 0.0): 1.825,
                    (65, 3.0, 0.1): 2.282,
                },},
                "Scoliosis": {
                    "U": {
                        (65, 2.5, 0.0): 1.874,
                        (65, 3.0, 0.0): 2,
                        (65, 3.0, 0.1): 2.607,
                        (65, 3.5, 0.0): 2.111,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.146,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.162,
                    (80, 2.5, 0.0): 0.176,
                    (80, 3.0, 0.0): 0.185,
                    (80, 3.5, 0.0): 0.193,
                    (100, 2.5, 0.0): 0.201,
                    (100, 3.0, 0.0): 0.21,
                    (100, 3.5, 0.0): 0.218,
                    (120, 2.5, 0.0): 0.22,
                    (120, 3.0, 0.0): 0.228,
                    (120, 3.5, 0.0): 0.235,
                },
            },
            "Skull": {                "U": {
                    (70, 2.5, 0.0): 0.35,
                    (70, 3.0, 0.0): 0.373,
                    (70, 3.0, 0.1): 0.479,
                    (70, 3.5, 0.0): 0.392,
                },},
        },
        1: {
            "Validation": _VALIDATION_EFF_REGION,
            "Skull": {                "U": {
                    (70, 2.5, 0.0): 0.12999999999999998,
                    (70, 3.0, 0.0): 0.13899999999999998,
                    (70, 3.0, 0.1): 0.18000000000000002,
                    (70, 3.5, 0.0): 0.147,
                },},
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): 1.269,
                    (70, 3.0, 0.0): 1.367,
                    (70, 3.0, 0.1): 1.829,
                    (70, 3.5, 0.0): 1.454,
                },},
                "Chest": {
                    "U": {
                        (65, 2.5, 0.0): 0.687,
                        (65, 3.0, 0.0): 0.731,
                        (65, 3.0, 0.1): 0.942,
                        (65, 3.5, 0.0): 0.769,
                    },
                },
                        "Thoracic Spine": {                "U": {
                    (65, 2.5, 0.0): 0.6849999999999999,
                    (65, 3.0, 0.0): 0.742,
                    (65, 3.5, 0.0): 0.7929999999999999,
                    (65, 3.0, 0.1): 1.018,
                },},

                "Scoliosis": {
                    "U": {
                        (70, 2.5, 0.0): 0.834,
                        (70, 3.0, 0.0): 0.892,
                        (70, 3.0, 0.1): 1.17,
                        (70, 3.5, 0.0): 0.944,
                    },
                },
        },
        5: {
            "Validation": _VALIDATION_EFF_REGION,
                "Chest": {
                    "U": {
                        (75, 2.5, 0.0): 0.455,
                        (75, 3.0, 0.0): 0.481,
                        (75, 3.0, 0.1): 0.601,
                        (75, 3.5, 0.0): 0.504,
                    },
                },
            "Lumbar Spine": {                "U": {
                    (70, 2.5, 0.0): 0.669,
                    (70, 3.0, 0.0): 0.725,
                    (70, 3.0, 0.1): 0.9959999999999999,
                    (70, 3.5, 0.0): 0.7739999999999999,
                },},
                        "Thoracic Spine": {                "U": {
                    (75, 2.5, 0.0): 0.371,
                    (75, 3.0, 0.0): 0.402,
                    (75, 3.5, 0.0): 0.431,
                    (75, 3.0, 0.1): 0.5609999999999999,
                },},
                "Scoliosis": {
                    "U": {
                        (75, 2.5, 0.0): 0.489,
                        (75, 3.0, 0.0): 0.521,
                        (75, 3.0, 0.1): 0.67,
                        (75, 3.5, 0.0): 0.549,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.146,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.162,
                    (80, 2.5, 0.0): 0.176,
                    (80, 3.0, 0.0): 0.185,
                    (80, 3.5, 0.0): 0.193,
                    (100, 2.5, 0.0): 0.201,
                    (100, 3.0, 0.0): 0.21,
                    (100, 3.5, 0.0): 0.218,
                    (120, 2.5, 0.0): 0.22,
                    (120, 3.0, 0.0): 0.228,
                    (120, 3.5, 0.0): 0.235,
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): 0.082,
                    (75, 3.0, 0.0): 0.088,
                    (75, 3.0, 0.1): 0.118,
                    (75, 3.5, 0.0): 0.093,
                },},
        },
        10: {
            "Validation": _VALIDATION_EFF_REGION,
                "Chest": {
                    "U": {
                        (75, 2.5, 0.0): 0.249,
                        (75, 3.0, 0.0): 0.262,
                        (75, 3.0, 0.1): 0.328,
                        (75, 3.5, 0.0): 0.275,
                    },
                },
            "Lumbar Spine": {                "U": {
                    (75, 2.5, 0.0): 0.375,
                    (75, 3.0, 0.0): 0.40700000000000003,
                    (75, 3.0, 0.1): 0.5670000000000001,
                    (75, 3.5, 0.0): 0.43600000000000005,
                },},
                        "Thoracic Spine": {                "U": {
                    (75, 2.5, 0.0): 0.208,
                    (75, 3.0, 0.0): 0.22699999999999998,
                    (75, 3.5, 0.0): 0.243,
                    (75, 3.0, 0.1): 0.32,
                },},
                "Scoliosis": {
                    "U": {
                        (80, 2.5, 0.0): 0.302,
                        (80, 3.0, 0.0): 0.32,
                        (80, 3.0, 0.1): 0.409,
                        (80, 3.5, 0.0): 0.336,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.146,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.162,
                    (80, 2.5, 0.0): 0.176,
                    (80, 3.0, 0.0): 0.185,
                    (80, 3.5, 0.0): 0.193,
                    (100, 2.5, 0.0): 0.201,
                    (100, 3.0, 0.0): 0.21,
                    (100, 3.5, 0.0): 0.218,
                    (120, 2.5, 0.0): 0.22,
                    (120, 3.0, 0.0): 0.228,
                    (120, 3.5, 0.0): 0.235,
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): 0.053,
                    (75, 3.0, 0.0): 0.056,
                    (75, 3.0, 0.1): 0.075,
                    (75, 3.5, 0.0): 0.060000000000000005,
                },},
        },
        15: {
            "Validation": _VALIDATION_EFF_REGION,
                "Chest": {
                    "U": {
                        (100, 2.5, 0.0): 0.164,
                        (100, 3.0, 0.0): 0.174,
                        (100, 3.0, 0.1): 0.22,
                        (100, 3.5, 0.0): 0.182,
                    },
                },
            "Lumbar Spine": {                "U": {
                    (80, 2.5, 0.0): 0.192,
                    (80, 3.0, 0.0): 0.20900000000000002,
                    (80, 3.0, 0.1): 0.294,
                    (80, 3.5, 0.0): 0.224,
                },},
                        "Thoracic Spine": {                "U": {
                    (100, 2.5, 0.0): 0.164,
                    (100, 3.0, 0.0): 0.176,
                    (100, 3.5, 0.0): 0.187,
                    (100, 3.0, 0.1): 0.236,
                },},
                "Scoliosis": {
                    "U": {
                        (85, 2.5, 0.0): 0.162,
                        (85, 3.0, 0.0): 0.174,
                        (85, 3.0, 0.1): 0.233,
                        (85, 3.5, 0.0): 0.185,
                    },
                },
            "Cervical Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.146,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.162,
                    (80, 2.5, 0.0): 0.176,
                    (80, 3.0, 0.0): 0.185,
                    (80, 3.5, 0.0): 0.193,
                    (100, 2.5, 0.0): 0.201,
                    (100, 3.0, 0.0): 0.21,
                    (100, 3.5, 0.0): 0.218,
                    (120, 2.5, 0.0): 0.22,
                    (120, 3.0, 0.0): 0.228,
                    (120, 3.5, 0.0): 0.235,
                },
            },
            "Skull": {                "U": {
                    (75, 2.5, 0.0): 0.04,
                    (75, 3.0, 0.0): 0.043000000000000003,
                    (75, 3.0, 0.1): 0.058,
                    (75, 3.5, 0.0): 0.046,
                },},
        },
    },
}


# OBLIQUE organ coefficients (mGy / Gy·cm²)

coefficients_OBLIQUE: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (60, 2.5, 0.0): {"Active_Bone_Marrow": 0.04, "Adrenals": 0.075, "Brain": 0.0, "Breast": 0.007, "Colon": 0.32, "Endosteum": 0.055, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.515, "Heart": 0.032, "Kidneys": 0.05, "Liver": 0.356, "Lungs": 0.018, "Lymph_Nodes": 0.185, "Muscle": 0.092, "Oesophagus": 0.051, "Oral_Mucosa": 0.0, "Ovaries": 0.223, "Pancreas": 0.192, "Prostate": 0.013, "Salivary_Glands": 0.0, "Skin": 0.097, "Small_Intestine": 0.332, "Spleen": 0.067, "Stomach": 0.403, "Testicles": 0.003, "Thymus": 0.002, "Thyroid": 0.001, "Urinary_Bladder": 0.095, "Uterus": 0.319},
                    (60, 3.0, 0.0): {"Active_Bone_Marrow": 0.044, "Adrenals": 0.084, "Brain": 0.0, "Breast": 0.008, "Colon": 0.353, "Endosteum": 0.06, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.565, "Heart": 0.035, "Kidneys": 0.057, "Liver": 0.388, "Lungs": 0.02, "Lymph_Nodes": 0.204, "Muscle": 0.099, "Oesophagus": 0.057, "Oral_Mucosa": 0.0, "Ovaries": 0.246, "Pancreas": 0.215, "Prostate": 0.014, "Salivary_Glands": 0.0, "Skin": 0.099, "Small_Intestine": 0.366, "Spleen": 0.075, "Stomach": 0.441, "Testicles": 0.004, "Thymus": 0.002, "Thyroid": 0.001, "Urinary_Bladder": 0.105, "Uterus": 0.353},
                    (60, 3.5, 0.0): {"Active_Bone_Marrow": 0.047, "Adrenals": 0.093, "Brain": 0.0, "Breast": 0.009, "Colon": 0.382, "Endosteum": 0.063, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.611, "Heart": 0.039, "Kidneys": 0.063, "Liver": 0.417, "Lungs": 0.022, "Lymph_Nodes": 0.221, "Muscle": 0.105, "Oesophagus": 0.063, "Oral_Mucosa": 0.0, "Ovaries": 0.266, "Pancreas": 0.235, "Prostate": 0.016, "Salivary_Glands": 0.0, "Skin": 0.101, "Small_Intestine": 0.398, "Spleen": 0.083, "Stomach": 0.476, "Testicles": 0.004, "Thymus": 0.003, "Thyroid": 0.001, "Urinary_Bladder": 0.114, "Uterus": 0.384},
                    (80, 2.5, 0.0): {"Active_Bone_Marrow": 0.065, "Adrenals": 0.146, "Brain": 0.0, "Breast": 0.012, "Colon": 0.472, "Endosteum": 0.077, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.742, "Heart": 0.056, "Kidneys": 0.103, "Liver": 0.493, "Lungs": 0.031, "Lymph_Nodes": 0.281, "Muscle": 0.122, "Oesophagus": 0.088, "Oral_Mucosa": 0.0, "Ovaries": 0.349, "Pancreas": 0.322, "Prostate": 0.026, "Salivary_Glands": 0.0, "Skin": 0.105, "Small_Intestine": 0.507, "Spleen": 0.122, "Stomach": 0.562, "Testicles": 0.007, "Thymus": 0.007, "Thyroid": 0.001, "Urinary_Bladder": 0.143, "Uterus": 0.487},
                    (80, 3.0, 0.0): {"Active_Bone_Marrow": 0.071, "Adrenals": 0.162, "Brain": 0.0, "Breast": 0.014, "Colon": 0.512, "Endosteum": 0.082, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.804, "Heart": 0.062, "Kidneys": 0.114, "Liver": 0.531, "Lungs": 0.034, "Lymph_Nodes": 0.305, "Muscle": 0.13, "Oesophagus": 0.096, "Oral_Mucosa": 0.0, "Ovaries": 0.38, "Pancreas": 0.353, "Prostate": 0.029, "Salivary_Glands": 0.0, "Skin": 0.107, "Small_Intestine": 0.552, "Spleen": 0.134, "Stomach": 0.607, "Testicles": 0.007, "Thymus": 0.008, "Thyroid": 0.001, "Urinary_Bladder": 0.155, "Uterus": 0.53},
                    (80, 3.5, 0.0): {"Active_Bone_Marrow": 0.076, "Adrenals": 0.176, "Brain": 0.0, "Breast": 0.015, "Colon": 0.549, "Endosteum": 0.087, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.859, "Heart": 0.067, "Kidneys": 0.124, "Liver": 0.565, "Lungs": 0.037, "Lymph_Nodes": 0.328, "Muscle": 0.138, "Oesophagus": 0.104, "Oral_Mucosa": 0.0, "Ovaries": 0.408, "Pancreas": 0.382, "Prostate": 0.032, "Salivary_Glands": 0.0, "Skin": 0.109, "Small_Intestine": 0.593, "Spleen": 0.145, "Stomach": 0.646, "Testicles": 0.008, "Thymus": 0.008, "Thyroid": 0.001, "Urinary_Bladder": 0.166, "Uterus": 0.569},
                    (100, 2.5, 0.0): {"Active_Bone_Marrow": 0.091, "Adrenals": 0.209, "Brain": 0.0, "Breast": 0.018, "Colon": 0.602, "Endosteum": 0.095, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.936, "Heart": 0.08, "Kidneys": 0.158, "Liver": 0.609, "Lungs": 0.044, "Lymph_Nodes": 0.364, "Muscle": 0.148, "Oesophagus": 0.123, "Oral_Mucosa": 0.0, "Ovaries": 0.472, "Pancreas": 0.435, "Prostate": 0.037, "Salivary_Glands": 0.0, "Skin": 0.112, "Small_Intestine": 0.66, "Spleen": 0.175, "Stomach": 0.697, "Testicles": 0.009, "Thymus": 0.01, "Thyroid": 0.001, "Urinary_Bladder": 0.183, "Uterus": 0.625},
                    (100, 3.0, 0.0): {"Active_Bone_Marrow": 0.097, "Adrenals": 0.227, "Brain": 0.0, "Breast": 0.019, "Colon": 0.643, "Endosteum": 0.1, "Extrathoracic_Airways": 0.0, "Gallbladder": 0.999, "Heart": 0.087, "Kidneys": 0.172, "Liver": 0.648, "Lungs": 0.048, "Lymph_Nodes": 0.39, "Muscle": 0.157, "Oesophagus": 0.133, "Oral_Mucosa": 0.0, "Ovaries": 0.506, "Pancreas": 0.469, "Prostate": 0.04, "Salivary_Glands": 0.0, "Skin": 0.114, "Small_Intestine": 0.707, "Spleen": 0.189, "Stomach": 0.743, "Testicles": 0.01, "Thymus": 0.011, "Thyroid": 0.001, "Urinary_Bladder": 0.196, "Uterus": 0.67},
                    (100, 3.5, 0.0): {"Active_Bone_Marrow": 0.103, "Adrenals": 0.243, "Brain": 0.0, "Breast": 0.02, "Colon": 0.68, "Endosteum": 0.105, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.055, "Heart": 0.093, "Kidneys": 0.185, "Liver": 0.682, "Lungs": 0.051, "Lymph_Nodes": 0.413, "Muscle": 0.164, "Oesophagus": 0.141, "Oral_Mucosa": 0.0, "Ovaries": 0.537, "Pancreas": 0.499, "Prostate": 0.043, "Salivary_Glands": 0.0, "Skin": 0.116, "Small_Intestine": 0.75, "Spleen": 0.202, "Stomach": 0.782, "Testicles": 0.011, "Thymus": 0.012, "Thyroid": 0.001, "Urinary_Bladder": 0.207, "Uterus": 0.71},
                    (120, 2.5, 0.0): {"Active_Bone_Marrow": 0.113, "Adrenals": 0.255, "Brain": 0.0, "Breast": 0.022, "Colon": 0.703, "Endosteum": 0.108, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.092, "Heart": 0.098, "Kidneys": 0.206, "Liver": 0.7, "Lungs": 0.055, "Lymph_Nodes": 0.431, "Muscle": 0.169, "Oesophagus": 0.152, "Oral_Mucosa": 0.0, "Ovaries": 0.565, "Pancreas": 0.526, "Prostate": 0.048, "Salivary_Glands": 0.0, "Skin": 0.117, "Small_Intestine": 0.782, "Spleen": 0.22, "Stomach": 0.8, "Testicles": 0.012, "Thymus": 0.013, "Thyroid": 0.001, "Urinary_Bladder": 0.212, "Uterus": 0.734},
                    (120, 3.0, 0.0): {"Active_Bone_Marrow": 0.12, "Adrenals": 0.272, "Brain": 0.0, "Breast": 0.023, "Colon": 0.742, "Endosteum": 0.113, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.153, "Heart": 0.105, "Kidneys": 0.22, "Liver": 0.736, "Lungs": 0.058, "Lymph_Nodes": 0.456, "Muscle": 0.177, "Oesophagus": 0.161, "Oral_Mucosa": 0.0, "Ovaries": 0.599, "Pancreas": 0.559, "Prostate": 0.051, "Salivary_Glands": 0.0, "Skin": 0.12, "Small_Intestine": 0.828, "Spleen": 0.234, "Stomach": 0.843, "Testicles": 0.013, "Thymus": 0.014, "Thyroid": 0.001, "Urinary_Bladder": 0.224, "Uterus": 0.777},
                    (120, 3.5, 0.0): {"Active_Bone_Marrow": 0.126, "Adrenals": 0.288, "Brain": 0.0, "Breast": 0.024, "Colon": 0.777, "Endosteum": 0.118, "Extrathoracic_Airways": 0.0, "Gallbladder": 1.205, "Heart": 0.11, "Kidneys": 0.233, "Liver": 0.768, "Lungs": 0.062, "Lymph_Nodes": 0.478, "Muscle": 0.184, "Oesophagus": 0.17, "Oral_Mucosa": 0.0, "Ovaries": 0.629, "Pancreas": 0.588, "Prostate": 0.054, "Salivary_Glands": 0.0, "Skin": 0.121, "Small_Intestine": 0.868, "Spleen": 0.247, "Stomach": 0.88, "Testicles": 0.014, "Thymus": 0.015, "Thyroid": 0.001, "Urinary_Bladder": 0.235, "Uterus": 0.814},
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (65, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (65, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        1: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        5: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (70, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (70, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        10: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (75, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (75, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
        15: {
            "Validation": _VALIDATION_ORGAN_REGION,
            "Lumbar Spine": {
                "U": {
                    (80, 2.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.0, 0.1): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                    (80, 3.5, 0.0): {'Active_Bone_Marrow': 0.0, 'Adrenals': 0.0, 'Brain': 0.0, 'Breast': 0.0, 'Colon': 0.0, 'Endosteum': 0.0, 'Extrathoracic_Airways': 0.0, 'Gallbladder': 0.0, 'Gonads': 0.0, 'Heart': 0.0, 'Kidneys': 0.0, 'Liver': 0.0, 'Lungs': 0.0, 'Lymph_Nodes': 0.0, 'Muscle': 0.0, 'Oesophagus': 0.0, 'Oral_Mucosa': 0.0, 'Pancreas': 0.0, 'Prostate': 0.0, 'Salivary_Glands': 0.0, 'Skin': 0.0, 'Small_Intestine': 0.0, 'Spleen': 0.0, 'Stomach': 0.0, 'Thymus': 0.0, 'Thyroid': 0.0, 'Urinary_Bladder': 0.0, 'Uterus': 0.0},
                },
            },
        },
    },
}


# OBLIQUE effective dose coefficients (mSv / Gy·cm²)

eff_coefficients_OBLIQUE: Dict = {
    "adult": {
        "adult": {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (60, 2.5, 0.0): 0.141,
                    (60, 3.0, 0.0): 0.155,
                    (60, 3.5, 0.0): 0.167,
                    (80, 2.5, 0.0): 0.207,
                    (80, 3.0, 0.0): 0.224,
                    (80, 3.5, 0.0): 0.24,
                    (100, 2.5, 0.0): 0.265,
                    (100, 3.0, 0.0): 0.283,
                    (100, 3.5, 0.0): 0.299,
                    (120, 2.5, 0.0): 0.31,
                    (120, 3.0, 0.0): 0.327,
                    (120, 3.5, 0.0): 0.342,
                },
            },
        },
    },
    "paediatric": {
        0: {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (65, 2.5, 0.0): 0.0,
                    (65, 3.0, 0.0): 0.0,
                    (65, 3.0, 0.1): 0.0,
                    (65, 3.5, 0.0): 0.0,
                },
            },
        },
        1: {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.0,
                    (70, 3.0, 0.0): 0.0,
                    (70, 3.0, 0.1): 0.0,
                    (70, 3.5, 0.0): 0.0,
                },
            },
        },
        5: {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (70, 2.5, 0.0): 0.0,
                    (70, 3.0, 0.0): 0.0,
                    (70, 3.0, 0.1): 0.0,
                    (70, 3.5, 0.0): 0.0,
                },
            },
        },
        10: {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (75, 2.5, 0.0): 0.0,
                    (75, 3.0, 0.0): 0.0,
                    (75, 3.0, 0.1): 0.0,
                    (75, 3.5, 0.0): 0.0,
                },
            },
        },
        15: {
            "Validation": _VALIDATION_EFF_REGION,
            "Lumbar Spine": {
                "U": {
                    (80, 2.5, 0.0): 0.0,
                    (80, 3.0, 0.0): 0.0,
                    (80, 3.0, 0.1): 0.0,
                    (80, 3.5, 0.0): 0.0,
                },
            },
        },
    },
}


def _build_organ_coeffs(data: Dict) -> Dict:
    out: Dict = {"adult": {}, "paediatric": {}}
    for group in ("adult", "paediatric"):
        age_keys = ["adult"] if group == "adult" else [0, 1, 5, 10, 15]
        for ak in age_keys:
            slot: Dict = {}
            for region, sex_blk in data.get(group, {}).get(ak, {}).items():
                slot[region] = sex_blk
            if slot:
                out[group][ak] = slot
    return out


def _build_eff_coeffs(data: Dict) -> Dict:
    out: Dict = {"adult": {}, "paediatric": {}}
    for group in ("adult", "paediatric"):
        age_keys = ["adult"] if group == "adult" else [0, 1, 5, 10, 15]
        for ak in age_keys:
            slot: Dict = {}
            for region, sex_blk in data.get(group, {}).get(ak, {}).items():
                slot[region] = sex_blk
            if slot:
                out[group][ak] = slot
    return out


_ORGAN_COEFFS_AP: Dict      = _build_organ_coeffs(coefficients_AP)
_ORGAN_COEFFS_PA: Dict      = _build_organ_coeffs(coefficients_PA)
_ORGAN_COEFFS_LATR: Dict    = _build_organ_coeffs(coefficients_LATR)
_ORGAN_COEFFS_OBLIQUE: Dict = _build_organ_coeffs(coefficients_OBLIQUE)
_EFF_COEFFS_AP: Dict      = _build_eff_coeffs(eff_coefficients_AP)
_EFF_COEFFS_PA: Dict      = _build_eff_coeffs(eff_coefficients_PA)
_EFF_COEFFS_LATR: Dict    = _build_eff_coeffs(eff_coefficients_LATR)
_EFF_COEFFS_OBLIQUE: Dict = _build_eff_coeffs(eff_coefficients_OBLIQUE)


def nearest_age_bin(age_years: float) -> int:
    """Round age to nearest paediatric bin (0,1,5,10,15)."""
    bins = (0, 1, 5, 10, 15)
    return min(bins, key=lambda b: abs(b - age_years))


def _resolve_available_age_key(coeffs: dict, group: str, age_key):
    if group == "adult":
        return "adult"
    age_map = coeffs.get(group, {})
    if age_key in age_map:
        return age_key
    numeric_keys = [k for k in age_map if isinstance(k, (int, float))]
    if not numeric_keys:
        return age_key
    return min(numeric_keys, key=lambda k: abs(k - age_key))


def _nearest_beam(sex_block: dict, beam_key):
    """Return organ dict for the nearest available beam quality.

    Returns ``None`` when ``beam_key`` is ``None`` (no kVp recorded) or the
    block holds no beam-quality keys.  Otherwise the closest available
    (kVp, Al, Cu) tuple is selected by weighted distance.  Copper is
    weighted heavily: 0.1 mm Cu hardens a beam roughly as much as 3 mm Al,
    so a copper-filtered beam must snap to a Cu = 0.1 table key when one
    exists at that kVp rather than tie with a thicker-Al copper-free key.
    kVp stays dominant (x10 per kV).
    """
    tuple_keys = [k for k in sex_block if isinstance(k, tuple)]
    if beam_key is None or not tuple_keys:
        return None
    if beam_key in sex_block:
        return sex_block[beam_key]
    kv0, fa0, fc0 = beam_key
    def dist(k):
        """Weighted distance between beam-quality tuple ``k`` and the
        exam's (kVp, Al mm, Cu mm).  Copper is weighted more heavily than
        aluminium, kVp most heavily of all."""
        kv, fa, fc = k
        return abs(kv - kv0) * 10 + abs(fa - fa0) + abs(fc - fc0) * 30
    best = min(tuple_keys, key=dist)
    return sex_block[best]


def _select_sex_block(proj_block: dict, sex_code: str) -> dict:
    sc = str(sex_code or "").upper()
    if sc in ("F", "M") and sc in proj_block:
        return proj_block[sc]
    if "U" in proj_block:
        return proj_block["U"]
    if "F" in proj_block and "M" in proj_block:
        return proj_block["F"]
    return next(iter(proj_block.values()))


def _get_radiography_organ_coeffs_impl(
    region: str,
    age_years: float,
    sex_code: str,
    kVp: float = None,
    filtration_al: float = None,
    filtration_cu: float = None,
    projection: str = None,
) -> dict:
    """Return organ dose coefficients (mGy / Gy·cm²) for any radiography region."""
    projection = normalize_projection_orientation(projection)
    is_paed = age_years is not None and age_years < 18
    group = "paediatric" if is_paed else "adult"
    if projection is None and region == "Scoliosis":
        # PCXMC Scoliosis conversion coefficients only exist for AP (paediatric
        # ages 0/1) and PA (paediatric ages 5/10/15, and adults); default to
        # PA unless this is one of the AP-only paediatric bins.
        if not is_paed or nearest_age_bin(age_years) in (5, 10, 15):
            projection = "PA"
    bk = (int(round(kVp)),
          float(filtration_al if filtration_al is not None else _default_al_filtration_for_kvp(kVp)),
          float(filtration_cu or 0.0)) if kVp is not None else None

    if projection == "PA":
        proj_table = _ORGAN_COEFFS_PA
    elif projection in ("Lat", "LLat", "RLat"):
        proj_table = _ORGAN_COEFFS_LATR
    elif projection == "RObl":
        proj_table = _ORGAN_COEFFS_OBLIQUE
    else:
        proj_table = None

    if proj_table is not None:
        ak = nearest_age_bin(age_years) if is_paed else "adult"
        ak = _resolve_available_age_key(proj_table, group, ak)
        reg_blk = proj_table.get(group, {}).get(ak, {}).get(region)
        if reg_blk is not None:
            sex_blk = _select_sex_block(reg_blk, sex_code)
            result = _nearest_beam(sex_blk, bk)
            if result:
                return result

    age_key = nearest_age_bin(age_years) if is_paed else "adult"
    age_key = _resolve_available_age_key(_ORGAN_COEFFS_AP, group, age_key)
    reg_blk = _ORGAN_COEFFS_AP.get(group, {}).get(age_key, {}).get(region)
    if reg_blk is None and region == "Scoliosis":
        # Scoliosis has AP coefficients only at paediatric ages 0/1 and PA
        # at 5/10/15 and adult; an AP request at the other ages takes the
        # PA block rather than falling through to no coefficients.
        reg_blk = _ORGAN_COEFFS_PA.get(group, {}).get(age_key, {}).get(region)
    if reg_blk is None:
        return {}
    sex_blk = _select_sex_block(reg_blk, sex_code)
    return _nearest_beam(sex_blk, bk) or {}


def _get_radiography_eff_dose_impl(
    region: str,
    age_years: float,
    sex_code: str,
    kVp: float = None,
    filtration_al: float = None,
    filtration_cu: float = None,
    projection: str = None,
) -> float:
    """Return effective dose coefficient (mSv / Gy·cm²) for any radiography region."""
    projection = normalize_projection_orientation(projection)
    is_paed = age_years is not None and age_years < 18
    group = "paediatric" if is_paed else "adult"
    if projection is None and region == "Scoliosis":
        # PCXMC Scoliosis conversion coefficients only exist for AP (paediatric
        # ages 0/1) and PA (paediatric ages 5/10/15, and adults); default to
        # PA unless this is one of the AP-only paediatric bins.
        if not is_paed or nearest_age_bin(age_years) in (5, 10, 15):
            projection = "PA"
    bk = (int(round(kVp)),
          float(filtration_al if filtration_al is not None else _default_al_filtration_for_kvp(kVp)),
          float(filtration_cu or 0.0)) if kVp is not None else None

    if projection == "PA":
        proj_table = _EFF_COEFFS_PA
    elif projection in ("Lat", "LLat", "RLat"):
        proj_table = _EFF_COEFFS_LATR
    elif projection == "RObl":
        proj_table = _EFF_COEFFS_OBLIQUE
    else:
        proj_table = None

    if proj_table is not None:
        ak = nearest_age_bin(age_years) if is_paed else "adult"
        ak = _resolve_available_age_key(proj_table, group, ak)
        reg_blk = proj_table.get(group, {}).get(ak, {}).get(region)
        if reg_blk is not None:
            sex_blk = _select_sex_block(reg_blk, sex_code)
            result = _nearest_beam(sex_blk, bk)
            if result is not None and not isinstance(result, dict):
                return float(result)

    age_key = nearest_age_bin(age_years) if is_paed else "adult"
    age_key = _resolve_available_age_key(_EFF_COEFFS_AP, group, age_key)
    reg_blk = _EFF_COEFFS_AP.get(group, {}).get(age_key, {}).get(region)
    if reg_blk is None and region == "Scoliosis":
        # Same Scoliosis AP -> PA block fallback as the organ-coefficient
        # lookup above.
        reg_blk = _EFF_COEFFS_PA.get(group, {}).get(age_key, {}).get(region)
    if reg_blk is None:
        return 0.0
    sex_blk = _select_sex_block(reg_blk, sex_code)
    result = _nearest_beam(sex_blk, bk)
    if isinstance(result, dict):
        return 0.0
    return float(result or 0.0)


# ---------------------------------------------------------------------------
# Memoised public coefficient lookups.
#
# The coefficient tables are static after import and the lookups are pure, so
# results are cached.  ``age_years`` is canonicalised first so every adult age
# collapses to one cache key and each paediatric age to its bin -- this is
# identical because ``nearest_age_bin`` is idempotent and all bins are < 18
# (the impl applies the exact same binning internally).  The cached organ-coeff
# dict is a reference into the static tables (callers iterate it read-only).
# ---------------------------------------------------------------------------


def _radiography_age_canon(age_years):
    """Map ``age_years`` to a canonical, low-cardinality cache key.

    Returns ``None`` for adults / unknown age (the impl treats that as the
    single ``"adult"`` block) and the paediatric bin (as float) otherwise.
    """
    if age_years is None:
        return None
    try:
        a = float(age_years)
    except (TypeError, ValueError):
        return None
    if a >= 18.0:
        return None
    return float(nearest_age_bin(a))


@lru_cache(maxsize=65536)
def _radiography_organ_coeffs_cached(region, age_canon, sex_code, kVp, fal, fcu, projection):
    return _get_radiography_organ_coeffs_impl(
        region, age_canon, sex_code, kVp, fal, fcu, projection)


@lru_cache(maxsize=65536)
def _radiography_eff_dose_cached(region, age_canon, sex_code, kVp, fal, fcu, projection):
    return _get_radiography_eff_dose_impl(
        region, age_canon, sex_code, kVp, fal, fcu, projection)


def get_radiography_organ_coeffs(
    region: str,
    age_years: float,
    sex_code: str,
    kVp: float = None,
    filtration_al: float = None,
    filtration_cu: float = None,
    projection: str = None,
) -> dict:
    """Memoised wrapper around :func:`_get_radiography_organ_coeffs_impl`."""
    return _radiography_organ_coeffs_cached(
        region, _radiography_age_canon(age_years), sex_code,
        kVp, filtration_al, filtration_cu, projection)


def get_radiography_eff_dose(
    region: str,
    age_years: float,
    sex_code: str,
    kVp: float = None,
    filtration_al: float = None,
    filtration_cu: float = None,
    projection: str = None,
) -> float:
    """Memoised wrapper around :func:`_get_radiography_eff_dose_impl`."""
    return _radiography_eff_dose_cached(
        region, _radiography_age_canon(age_years), sex_code,
        kVp, filtration_al, filtration_cu, projection)


# ---------------------------------------------------------------------------
# Radiography exam regions
# ---------------------------------------------------------------------------

RADIOGRAPHY_EXAM_REGIONS = (
    "Chest PA", "Chest AP", "Chest Lat",
    "Pelvis AP", "Abdomen AP",
    "Lumbar spine AP", "Lumbar spine Lat", "Lumbar spine oblique",
    "Cervical spine AP", "Cervical spine Lat",
    "Thoracic spine AP", "Thoracic spine Lat",
    "Scoliosis AP", "Scoliosis PA", "Scoliosis Lat",
    "Skull PA", "Skull Lat",
    "Extremities",
)

# Exam key → _ORGAN_COEFFS / _EFF_COEFFS region key
_EXAM_TO_REGION: Dict[str, str] = {
    "Chest":                "Chest",
    "Lumbar Spine":         "Lumbar Spine",
    "Lumbar spine":         "Lumbar Spine",
    "Thoracic Spine":       "Thoracic Spine",
    "Thoracic spine":       "Thoracic Spine",
    "Cervical Spine":       "Cervical Spine",
    "Cervical spine":       "Cervical Spine",
    "Extremities":          "Extremity",
    "Extremity":            "Extremity",
    "Scoliosis":            "Scoliosis",
    "Skull":                "Skull",
    "Pelvis":               "Pelvis",
    "Abdomen":              "Abdomen",
    "Validation":           "Validation",
}


def _normalize_exam_key(region: Optional[str], text: str) -> Optional[str]:
    """Map a free-text exam description/region to a canonical age-group key
    ("Abdomen", "Chest", "Cervical Spine", ...) for the paediatric
    coefficient tables.  Returns ``None`` when neither ``region`` nor
    ``text`` carries any usable text.  Substring matching is case-folding
    and checks the more specific keys first."""
    if region is None and not text:
        return None
    t = (text or "").lower()
    if not t and region:
        t = region.lower()
    # Order matters – check more specific keys first
    if "abdom" in t:
        return "Abdomen"
    if "pelv" in t:
        return "Pelvis"
    if "lumbar" in t or "l-spine" in t or "lspine" in t:
        return "Lumbar Spine"
    if "thoraco-lumbar" in t or "thoraco lumbar" in t:
        return "Lumbar Spine"
    if "thoracic" in t or "t-spine" in t or "tspine" in t:
        return "Thoracic Spine"
    if "cervical" in t or "c-spine" in t or "cspine" in t:
        return "Cervical Spine"
    if "skull" in t or "cranium" in t or "head" in t:
        return "Skull"
    if "chest" in t or "thorax" in t:
        return "Chest"
    if "extrem" in t or "limb" in t or "hand" in t or "foot" in t or "knee" in t or "ankle" in t or "wrist" in t or "elbow" in t or "shoulder" in t:
        return "Extremity"
    if "scoliosis" in t:
        return "Scoliosis"
    if "validation" in t:
        return "Validation"
    if region:
        return region
    return None


def _default_al_filtration_for_kvp(kv: float) -> float:
    if kv <= 70:
        return 2.5
    if kv <= 100:
        return 3.0
    return 3.5


_FILTER_THICKNESS_COL = "Xray Filter Max Thickness (DX)"
_CU_AL_THRESHOLD_MM = 0.6   # added plate < 0.6 mm -> copper, >= 0.6 mm -> aluminium


def _parse_filter_thickness_set(val) -> list:
    """Parse a DICOM (0018,7054) filter-thickness cell to a list of mm floats.

    The Radimetrics export serialises the multi-value tag as a brace set:
    ``"{}"`` (no information), ``"{0.2}"``, ``"{1.06,0.202}"``.  Tolerates
    whitespace, a bare number (``0.2`` or ``"0.2"``) and unparseable or
    non-positive tokens (skipped).  Never raises.
    """
    if val is None:
        return []
    if isinstance(val, float) and not (val == val):  # NaN
        return []
    s = str(val).strip()
    if not s or s == "{}":
        return []
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    out = []
    for tok in s.split(","):
        v = _to_float(tok.strip())
        if v is not None and v > 0.0:
            out.append(v)
    return out


def _beam_quality_from_row(
    row: Optional[Mapping[str, Any]],
) -> Optional[Tuple[int, float, float]]:
    """Derive the (kVp, total Al mm, Cu mm) beam-quality key for one exam row.

    The Radimetrics export carries kVp in ``kVp kV`` and the added filter
    plates in the DICOM (0018,7054) column ``Xray Filter Max Thickness
    (DX)`` -- a brace-set of plate thicknesses in mm that excludes inherent
    filtration.  There is no material column, so material is inferred from
    thickness: a plate < 0.6 mm counts as copper, >= 0.6 mm as aluminium
    (1.06 mm is Carestream's nominal "1 mm Al").

    The coefficient tables key Al as *total* filtration, so added Al plates
    are stacked on top of the kVp default total
    (:func:`_default_al_filtration_for_kvp`).

    Returns ``None`` when no kVp is recorded: the conversion coefficients
    depend on kVp, so the exam has no dose data.
    """
    if row is None:
        return None
    kv = None
    for key in ("kVp kV", "kVp", "KVP", "kV", "Tube voltage"):
        v = _to_float(row.get(key))
        if v is not None and v > 0.0:
            kv = v
            break
    if kv is None:
        return None
    kv = int(round(kv))

    plates = _parse_filter_thickness_set(row.get(_FILTER_THICKNESS_COL))
    al_added = sum(v for v in plates if v >= _CU_AL_THRESHOLD_MM)
    cu_added = sum(v for v in plates if v < _CU_AL_THRESHOLD_MM)
    fa = _default_al_filtration_for_kvp(kv) + al_added
    fc = cu_added

    return (kv, round(fa, 3), round(fc, 3))


def icrp_radiography_organ_doses_gy_from_dap(
    dap_gy_cm2: float,
    region: Optional[str],
    age_years: Optional[float],
    sex_code: str,
    row: Optional[Mapping[str, Any]] = None,
    exam: Optional[str] = None,
    projection: Optional[str] = None,
    beam_quality_key: Optional[Tuple[int, float, float]] = None,
) -> Optional[Dict[str, float]]:
    """Return per-organ absorbed doses (Gy) for a radiography exam from DAP.

    Uses PCXMC organ-dose coefficient tables keyed by
    (age group, projection, beam quality, exam type).  Returns ``None``
    when DAP is invalid, no kVp is recorded or no matching coefficient set
    is found.
    """
    dap = _to_float(dap_gy_cm2)
    if dap is None or dap <= 0.0:
        return None
    projection = get_projection_orientation(row, projection) or infer_projection_orientation_from_text(str(region or ""))
    exam_key = exam or _normalize_exam_key(
        region,
        " ".join([
            str(region or ""),
            str(row.get("Protocol") if row else ""),
            str(row.get("Description") if row else ""),
            str(row.get("Series Description") if row else ""),
        ]),
    )
    if not exam_key:
        return None
    region_key = _EXAM_TO_REGION.get(exam_key, exam_key)
    bk = beam_quality_key or _beam_quality_from_row(row)
    if bk is None:
        return None
    kVp, fal, fcu = bk
    organ_coeffs = get_radiography_organ_coeffs(region_key, age_years, sex_code, kVp, fal, fcu, projection)
    if not organ_coeffs:
        return None
    out: Dict[str, float] = {}
    for organ_name, coeff_mgy_per_gycm2 in organ_coeffs.items():
        c = _to_float(coeff_mgy_per_gycm2)
        if c is None or c <= 0.0:
            continue
        dose_gy = dap * c / 1000.0
        if dose_gy > 0.0:
            out[str(organ_name)] = dose_gy
    if out:
        _REMAINDER_SUBGROUP = (
            "Adrenals", "Extrathoracic_Airways", "Gallbladder", "Heart",
            "Lymph_Nodes", "Muscle", "Oral_Mucosa", "Small_Intestine",
            "Spleen", "Thymus",
        )
        rem_sum = sum(out.get(k, 0.0) for k in _REMAINDER_SUBGROUP)
        out["Remainder"] = rem_sum / 10.0
    return out if out else None


def icrp_radiography_effective_dose_msv_from_dap(
    dap_gy_cm2: float,
    region: Optional[str],
    age_years: Optional[float],
    sex_code: str,
    row: Optional[Mapping[str, Any]] = None,
    exam: Optional[str] = None,
    projection: Optional[str] = None,
    beam_quality_key: Optional[Tuple[int, float, float]] = None,
) -> Optional[float]:
    """Return effective dose (mSv) for a radiography exam from DAP.

    Uses PCXMC effective-dose coefficient tables keyed by
    (age group, projection, beam quality, exam type).  Returns ``None``
    when DAP is invalid, no kVp is recorded or no matching coefficient set
    is found.
    """
    dap = _to_float(dap_gy_cm2)
    if dap is None or dap <= 0.0:
        return None
    projection = get_projection_orientation(row, projection) or infer_projection_orientation_from_text(str(region or ""))
    exam_key = exam or _normalize_exam_key(
        region,
        " ".join([
            str(region or ""),
            str(row.get("Protocol") if row is not None else ""),
            str(row.get("Description") if row is not None else ""),
            str(row.get("Series Description") if row is not None else ""),
        ]),
    )
    if not exam_key:
        return None
    region_key = _EXAM_TO_REGION.get(exam_key, exam_key)
    bk = beam_quality_key or _beam_quality_from_row(row)
    if bk is None:
        return None
    kVp, fal, fcu = bk
    coeff = get_radiography_eff_dose(region_key, age_years, sex_code, kVp, fal, fcu, projection)
    if coeff is None or coeff <= 0.0:
        return None
    return float(dap) * float(coeff)
