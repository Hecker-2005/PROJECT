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
from P3_Adversarial_Testing.fgsm_attack import run_fgsm
from P3_Adversarial_Testing.validity_filter import ValidityFilter
from P3_Adversarial_Testing.dependency_rules import (
	bytes_consistency,
	flow_duration_consistency,
	protocol_flag_consistency
)

# Load feature names (78)
feature_names = joblib.load(
	"../P1_Preprocessing/feature_names.pkl"
)
assert len(feature_names) == 50

# Load scaler
scaler_bundle = joblib.load(
	"../P1_Preprocessing/scaler_persistence.pkl"
)

# Extract actual scaler object
scaler = scaler_bundle["scaler"]

# Load dataset and built X_test
df = pd.read_csv(
	"../P0_Datasets/CICIDS2017/cicids2017.csv"
)

X = df[feature_names].values.astype(np.float64)

# Safety sanitization
X = np.nan_to_num(
	X,
	nan = 0.0,
	posinf = 0.0,
	neginf = 0.0
)

X = scaler.transform(X)

# Simple hold-out split
X_test = X[int(0.8 * len(X)) :]

# Load model
device = "cpu"
model = NIDSAutoencoder(input_dim = 50)
model.load_state_dict(
	torch.load(
		"../P2_Model_Training/pytorch_model/autoencoder.pt", map_location = device
	)
)
model.to(device)
model.eval()

# Validity filter
immutable_features = [
	f for f in [
		"Protocol", "Destination Port"
	]
	if f in feature_names
]

integer_features = [
	f for f in [
		"FIN Flag Cnt", "SYN Flag Cnt", "RST Flag Cnt",
		"PSH Flag Cnt", "ACK Flag Cnt"
	]
	if f in feature_names
]
non_negative_features = feature_names

validity_filter = ValidityFilter(
	feature_names = feature_names,
	immutable_features=immutable_features,
	integer_features=integer_features,
	non_negative_features=non_negative_features,
	dependency_rules=[
		bytes_consistency,
		flow_duration_consistency,
		protocol_flag_consistency
	]
)

# run FGSM
epsilons = [0.02, 0.05]

for eps in epsilons:
	valid_adv, invalid = run_fgsm(
		model = model,
		X = X_test,
		epsilon = eps,
		validity_filter = validity_filter,
		device = device
	)
	
	print(f"\nFGSM ε = {eps}")
	print(f"Valid adversarial samples   : {len(valid_adv)}")
	print(f"Invalid rejected samples   : {invalid}")
