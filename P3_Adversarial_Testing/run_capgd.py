import sys
import os
import torch
import joblib
import numpy as np
import pandas as pd

# Make project root importable
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

# configs
DEVICE = "cpu"
EPSILONS = [0.02, 0.05, 0.10, 0.15]
CAPGD_STEPS = 20
STEP_FRAC = 0.1
MAX_SAMPLES = 5000 # keep small for first run

# Load frozen features
feature_names = joblib.load(
	"../P1_Preprocessing/feature_names.pkl"
)
assert len(feature_names) == 50

# Load scaler
bundle = joblib.load(
	"../P1_Preprocessing/scaler_persistence.pkl"
)
scaler = bundle["scaler"]

# Load dataset
df = pd.read_csv(
	"../P0_Datasets/CICIDS2017/cicids2017.csv"
)

X = df[feature_names].values.astype(np.float64)

# Sanitize
X = np.nan_to_num(X, nan = 0.0, posinf = 0.0, neginf = 0.0)

X = scaler.transform(X)

# Hold-out test split (same logic as FGSM)
X_test = X[int(0.8 * len(X)) :]
X_test = X_test[:MAX_SAMPLES]

print(f"Using {len(X_test)} samples for CAPGD evaluation")

# Load model
model = NIDSAutoencoder(input_dim = 50)
model.load_state_dict(
	torch.load(
		"../P2_Model_Training/pytorch_model/autoencoder.pt",
		map_location = DEVICE
	)
)
model.to(DEVICE)
model.eval()

# Validity Filter
immutable_features = [f for f in ["Destination Port"] if f in feature_names]

integer_features = [
	f for f in [
		"FIN Flag Count", "SYN Flag Count", "RST Flag Count",
		"PSH Flag Count", "ACK Flag Count", "URG Flag Count"
	]
	if f in feature_names
]

validity_filter = ValidityFilter(
	feature_names = feature_names,
	immutable_features = immutable_features,
	integer_features = integer_features,
	non_negative_features = feature_names,
	dependency_rules = [
		bytes_consistency,
		flow_duration_consistency,
		protocol_flag_consistency
	]
)

# Mutable features (LOCKED)
MUTABLE_FEATURES = [
	"Fwd Packet Length Max", "Fwd Packet Length Mean", "Fwd Packet Length Min",
	"Total Length of Fwd Packets", "Total Fwd Packets", "act_data_pkt_fwd",
	"min_seg_size_forward", "Init_Win_bytes_forward", "Fwd Header Length",
	"Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Min",
	"Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
	"Flow Duration", "Idle Std", "Active Std", "Active Mean", "Active Min", "Active Max",
	"Fwd PSH Flags", "Fwd URG Flags"
]

# Build mutability mask
mask = _build_mask(feature_names, MUTABLE_FEATURES, DEVICE)

# Run CAPGD
for eps in EPSILONS:
	print(f"\n === CAPGD ε = {eps} ===")
	
	valid_count = 0
	total = 0
	
	for i in range(len(X_test)):
		x_np = X_test[i]
		x= torch.tensor(x_np, dtype = torch.float32, device = DEVICE)
		
		x_adv = capgd_attack(
			model = model,
			x = x,
			x_orig = x_np,
			epsilon = eps,
			mask = mask,
			validity_filter = validity_filter,
			steps = CAPGD_STEPS,
			step_frac = STEP_FRAC,
			device = DEVICE
		)
		
		# Validity check (final)
		if validity_filter.is_valid(x_adv.cpu().numpy()):
			valid_count += 1
		
		total += 1
		
		if (i + 1) % 500 == 0:
			print(f"Processed {i + 1}/{len(X_test)}")
			
	print(f"Valid adversarial samples: {valid_count}/{total}")
	print(f"Validity rate: {valid_count / total:.4f}")

