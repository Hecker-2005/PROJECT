import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc

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
from P4_Edge_Deployment.feature_squeezer import FeatureSqueezer
from P4_Edge_Deployment.squeezing_config import (
    DECIMAL_FEATURES,
    INTEGER_FEATURES,
    CLIP_BOUNDS
)

DEVICE = "cpu"
EPSILONS = [0.02, 0.05, 0.10, 0.15]
CAPGD_STEPS = 20
STEP_FRAC = 0.1
MAX_SAMPLES = 5000
TARGET_FPR = 0.01

feature_names = joblib.load("../P1_Preprocessing/feature_names.pkl")
bundle = joblib.load("../P1_Preprocessing/scaler_persistence.pkl")
scaler = bundle["scaler"]

df = pd.read_csv("../P0_Datasets/CICIDS2017/cicids2017.csv")

X = df[feature_names].values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = scaler.transform(X)

y = (df["Label"] != "BENIGN").astype(int).values

X_benign = X[y == 0][:MAX_SAMPLES]
X_attack = X[y == 1][:MAX_SAMPLES]

X_mixed = np.vstack([X_benign, X_attack])
y_mixed = np.concatenate([
    np.zeros(len(X_benign), dtype=int),
    np.ones(len(X_attack), dtype=int)
])

model = NIDSAutoencoder(input_dim=len(feature_names))
model.load_state_dict(
    torch.load("../P2_Model_Training/pytorch_model/autoencoder.pt", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

validity_filter = ValidityFilter(
    feature_names=feature_names,
    immutable_features=["Destination Port"],
    integer_features=INTEGER_FEATURES,
    non_negative_features=feature_names,
    dependency_rules=[
        bytes_consistency,
        flow_duration_consistency,
        protocol_flag_consistency
    ]
)

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

squeezer = FeatureSqueezer(
    feature_names=feature_names,
    decimal_features=DECIMAL_FEATURES,
    integer_features=INTEGER_FEATURES,
    clip_bounds=CLIP_BOUNDS
)

@torch.no_grad()
def recon_error(x):
    recon = model(x)
    return torch.mean((x - recon) ** 2, dim=1).cpu().numpy()

Xb_tensor = torch.tensor(
    squeezer.squeeze(X_benign),
    dtype=torch.float32,
    device=DEVICE
)
benign_scores = recon_error(Xb_tensor)
tau = np.quantile(benign_scores, 1 - TARGET_FPR)

for eps in EPSILONS:
    scores = []
    labels = []

    for i in range(len(X_mixed)):
        x_np = X_mixed[i]
        x_sq = squeezer.squeeze(x_np)

        x = torch.tensor(x_sq, dtype=torch.float32, device=DEVICE)

        x_adv = capgd_attack(
            model=model,
            x=x,
            x_orig=x_sq,
            epsilon=eps,
            mask=mask,
            validity_filter=validity_filter,
            steps=CAPGD_STEPS,
            step_frac=STEP_FRAC,
            device=DEVICE
        )

        if validity_filter.is_valid(x_adv.cpu().numpy()):
            scores.append(recon_error(x_adv.unsqueeze(0))[0])
            labels.append(y_mixed[i])

    scores = np.array(scores)
    labels = np.array(labels)

    prec, rec, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(rec, prec)
    recall = np.mean(scores[labels == 1] > tau)

    print(f"\n[SQUEEZED] ε = {eps}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"Recall @ FPR=1%: {recall:.4f}")
