import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd

# ---- Make project root importable ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from P2_Model_Training.pytorch_model.model_architecture import NIDSAutoencoder
from P3_Adversarial_Testing.capgd_attack import capgd_attack

# ----------------------------
# Configuration (DIAGNOSTIC)
# ----------------------------
DEVICE = "cpu"
EPSILON = 1.0        # large on purpose
STEPS = 30
STEP_FRAC = 0.2
MAX_SAMPLES = 500    # small, diagnostic only

# ----------------------------
# Load features & scaler
# ----------------------------
feature_names = joblib.load("../P1_Preprocessing/feature_names.pkl")
bundle = joblib.load("../P1_Preprocessing/scaler_persistence.pkl")
scaler = bundle["scaler"]

# ----------------------------
# Load dataset (ATTACKS ONLY)
# ----------------------------
df = pd.read_csv("../P0_Datasets/CICIDS2017/cicids2017.csv")
df_attack = df[df["Label"] != "BENIGN"].head(MAX_SAMPLES)

X = df_attack[feature_names].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = scaler.transform(X)

print(f"Using {len(X)} ATTACK samples for sanity check")

# ----------------------------
# Load model
# ----------------------------
model = NIDSAutoencoder(input_dim=len(feature_names))
model.load_state_dict(
    torch.load("../P2_Model_Training/pytorch_model/autoencoder.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# ----------------------------
# Unconstrained mask (ALL FEATURES)
# ----------------------------
mask = torch.ones(len(feature_names), device=DEVICE)

# ----------------------------
# Helper: reconstruction error
# ----------------------------
@torch.no_grad()
def recon_error(x_tensor):
    recon = model(x_tensor)
    return torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()

# ----------------------------
# Run sanity check
# ----------------------------
clean_errors = []
adv_errors = []

for i in range(len(X)):
    x_np = X[i]
    x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

    clean_errors.append(recon_error(x.unsqueeze(0))[0])

    # NOTE: validity_filter=None disables constraints
    x_adv = capgd_attack(
        model=model,
        x=x,
        x_orig=x_np,
        epsilon=EPSILON,
        mask=mask,
        validity_filter=None,   # <<< DISABLED
        steps=STEPS,
        step_frac=STEP_FRAC,
        device=DEVICE
    )

    adv_errors.append(recon_error(x_adv.unsqueeze(0))[0])

clean_errors = np.array(clean_errors)
adv_errors = np.array(adv_errors)

print("\n=== UNCONSTRAINED SANITY CHECK ===")
print(f"Mean clean RE: {clean_errors.mean():.6f}")
print(f"Mean adv   RE: {adv_errors.mean():.6f}")
print(f"RE reduction: {(clean_errors.mean() - adv_errors.mean()):.6f}")
