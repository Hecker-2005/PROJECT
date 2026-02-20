# evaluate_autoencoder.py

import os
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_curve, auc

from model_architecture import NIDSAutoencoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PREP_DIR = os.path.join(PROJECT_ROOT, "1_Preprocessing")
MODEL_DIR = os.path.dirname(__file__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	data = pd.read_parquet(os.path.join(PREP_DIR, "cicids_clean.parquet"))
	artifacts = joblib.load(os.path.join(PREP_DIR, "scaler_persistence.pkl"))
	
	features = artifacts["network_features"]
	scaler = artifacts["scaler"]
	
	X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
	X = scaler.transform(X)
	
	y_true = (data["Attack_Class"] != 0).astype(int).values
	
	X_tensor = torch.tensor(X, dtype = torch.float32).to(DEVICE)
	
	model = NIDSAutoencoder(input_dim = X.shape[1]).to(DEVICE)
	model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "autoencoder.pt")))
	model.eval()
	
	with torch.no_grad():
		X_hat = model(X_tensor)
		mse = torch.mean((X_tensor - X_hat) ** 2, dim = 1).cpu().numpy()
		
	precision, recall, _ = precision_recall_curve(y_true, mse)
	pr_auc = auc(recall, precision)
	
	# Threshold: 95th percentile of benign construction error
	benign_mse = mse[y_true == 0]
	threshold = np.percentile(benign_mse, 95)
	
	print(f"Autoencoder PR-AUC: {pr_auc:.4f}")
	print(f"Anomaly Threshold (95th percentile): {threshold:.6f}")

if __name__ == "__main__":
	main()
	
