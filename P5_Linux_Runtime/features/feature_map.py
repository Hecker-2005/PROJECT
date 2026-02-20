# P5_Linux_Runtime/features/feature_map.py

import os
import sys
import joblib
import numpy as np 

# -- Project root fix --
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# -- Paths to authoritative Phase 1 artifacts --
FEATURE_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "P1_Preprocessing",
    "feature_names.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "P1_Preprocessing",
    "scaler_persistence.pkl"
)

# -- Load frozen feature order --
FEATURE_ORDER = joblib.load(FEATURE_NAMES_PATH)

# -- safety checks --
if not isinstance(FEATURE_ORDER, (list, tuple)):
    raise TypeError("feature_names.pkl must contain a list or tuple")

if len(FEATURE_ORDER) == 0:
    raise ValueError("FEATURE_ORDER is empty - invalid feature_names.pkl")

# Optional but strongly recommended: cross check with scaler
try:
    scaler = joblib.load(SCALER_PATH)
    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(FEATURE_ORDER):
            raise ValueError(
                f"Feature count mismatch: "
                f"scaler expects {scaler.n_features_in_}, "
                f"FEATURE_ORDER has {len(FEATURE_ORDER)}"
            )

except Exception as e:
    raise RuntimeError(f"Failed to validate scaler compatibility: {e}")

# -- Public API --
def map_features_to_vector(feature_dict):
    """
    Convert computed feature dictionary into a fixed order NumPy vector.

    Parameters
    ''''''''''
    feature_dict: dict[str, float]

    Returns
    '''''''
    np.ndarray (float64)
        Ordered feature vector compatible with Phase 1 scaler and model
    """
    vector = np.zeros(len(FEATURE_ORDER), dtype = np.float64)

    for idx, name in enumerate(FEATURE_ORDER):
        # Default to 0.0 if feature missing (CICFlowMeter parity)
        vector[idx] = feature_dict.get(name, 0.0)

    # final numeric sanitation (absolute last line of defense)
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
