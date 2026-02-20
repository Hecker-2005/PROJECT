import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

# ---- Make project root importable ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from P2_Model_Training.pytorch_model.model_architecture import NIDSAutoencoder
from P3_Adversarial_Testing.capgd_attack import capgd_attack, _build_mask
from P3_Adversarial_Testing.validity_filter import ValidityFilter
from P3_Adversarial_Testing.dependency_rules import (
    bytes_consistency,
    flow_duration_consistency,
    protocol_flag_consistency
)

# ----------------------------
# Configuration
# ----------------------------
DEVICE = "cpu"
EPSILONS = [0.02, 0.05, 0.10, 0.15]
CAPGD_STEPS = 20
STEP_FRAC = 0.1
MAX_SAMPLES = 5000
TARGET_FPR = 0.01  # 1%

# ----------------------------
# Load frozen features & scaler
# ----------------------------
feature_names = joblib.load("../P1_Preprocessing/feature_names.pkl")
bundle = joblib.load("../P1_Preprocessing/scaler_persistence.pkl")
scaler = bundle["scaler"]

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("../P0_Datasets/CICIDS2017/cicids2017.csv")

X = df[feature_names].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = scaler.transform(X)

y = (df["Label"] != "BENIGN").astype(int).values  # 1 = attack, 0 = benign

# ----------------------------
# Build evaluation splits (FINAL)
# ----------------------------

# BENIGN samples (threshold calibration)
X_benign = X[y == 0][:MAX_SAMPLES]

# ATTACK samples (force inclusion)
X_attack = X[y == 1][:MAX_SAMPLES]
y_attack = y[y == 1][:MAX_SAMPLES]

# MIXED evaluation set (balanced)
X_mixed = np.vstack([X_benign, X_attack])
y_mixed = np.concatenate([
    np.zeros(len(X_benign), dtype=int),
    np.ones(len(X_attack), dtype=int)
])

print(f"Benign samples for threshold: {len(X_benign)}")
print(f"Attack samples for evaluation: {len(X_attack)}")
print(f"Total mixed samples: {len(X_mixed)}")


print(f"Benign samples for threshold: {len(X_benign)}")
print(f"Mixed samples for evaluation: {len(X_mixed)}")

# ----------------------------
# Load model
# ----------------------------
model = NIDSAutoencoder(input_dim=50)
model.load_state_dict(
    torch.load("../P2_Model_Training/pytorch_model/autoencoder.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# ----------------------------
# Validity Filter
# ----------------------------
immutable_features = [f for f in ["Destination Port"] if f in feature_names]
integer_features = [
    f for f in [
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count"
    ]
    if f in feature_names
]

validity_filter = ValidityFilter(
    feature_names=feature_names,
    immutable_features=immutable_features,
    integer_features=integer_features,
    non_negative_features=feature_names,
    dependency_rules=[
        bytes_consistency,
        flow_duration_consistency,
        protocol_flag_consistency
    ]
)

# ----------------------------
# Mutable features (LOCKED)
# ----------------------------
MUTABLE_FEATURES = [
    "Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Min",
    "Total Length of Fwd Packets", "Total Fwd Packets", "act_data_pkt_fwd",
    "min_seg_size_forward", "Init_Win_bytes_forward", "Fwd Header Length",
    "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Min",
    "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
    "Flow Duration", "Idle Std", "Active Std", "Active Mean", "Active Min", "Active Max",
    "Fwd PSH Flags", "Fwd URG Flags"
]

mask = _build_mask(feature_names, MUTABLE_FEATURES, DEVICE)

# ----------------------------
# Helper: reconstruction error
# ----------------------------
@torch.no_grad()
def recon_error(x_tensor):
    recon = model(x_tensor)
    return torch.mean((x_tensor - recon) ** 2, dim=1).cpu().numpy()

# ----------------------------
# CLEAN BASELINE
# ----------------------------
Xb_tensor = torch.tensor(X_benign, dtype=torch.float32, device=DEVICE)
benign_scores = recon_error(Xb_tensor)

tau = np.quantile(benign_scores, 1 - TARGET_FPR)

Xm_tensor = torch.tensor(X_mixed, dtype=torch.float32, device=DEVICE)
clean_scores = recon_error(Xm_tensor)

prec, rec, _ = precision_recall_curve(y_mixed, clean_scores)
pr_auc_clean = auc(rec, prec)
recall_clean = np.mean(clean_scores[y_mixed == 1] > tau)

print("\n=== CLEAN BASELINE ===")
print(f"PR-AUC: {pr_auc_clean:.4f}")
print(f"Threshold τ (FPR=1%): {tau:.6f}")
print(f"Recall @ FPR=1%: {recall_clean:.4f}")

# ----------------------------
# CAPGD EVALUATION
# ----------------------------
for eps in EPSILONS:
    adv_scores = []
    adv_labels = []

    for i in range(len(X_mixed)):
        x_np = X_mixed[i]
        x = torch.tensor(x_np, dtype=torch.float32, device=DEVICE)

        x_adv = capgd_attack(
            model=model,
            x=x,
            x_orig=x_np,
            epsilon=eps,
            mask=mask,
            validity_filter=validity_filter,
            steps=CAPGD_STEPS,
            step_frac=STEP_FRAC,
            device=DEVICE
        )

        if validity_filter.is_valid(x_adv.cpu().numpy()):
            adv_scores.append(recon_error(x_adv.unsqueeze(0))[0])
            adv_labels.append(y_mixed[i])

    adv_scores = np.array(adv_scores)
    adv_labels = np.array(adv_labels)

    prec, rec, _ = precision_recall_curve(adv_labels, adv_scores)
    pr_auc_adv = auc(rec, prec)
    recall_adv = np.mean(adv_scores[adv_labels == 1] > tau)

    print(f"\n=== CAPGD ε = {eps} ===")
    print(f"PR-AUC: {pr_auc_adv:.4f}")
    print(f"Δ PR-AUC: {pr_auc_clean - pr_auc_adv:.4f}")
    print(f"Recall @ FPR=1%: {recall_adv:.4f}")
