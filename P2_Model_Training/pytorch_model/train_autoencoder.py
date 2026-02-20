# train_autoencoder

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model_architecture import NIDSAutoencoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
PREP_DIR = os.path.join(PROJECT_ROOT, "1_Preprocessing")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
	# Load data
	data = pd.read_parquet(os.path.join(PREP_DIR, "cicids_clean.parquet"))
	artifacts = joblib.load(os.path.join(PREP_DIR, "scaler_persistence.pkl"))
	
	features = artifacts["network_features"]
	scaler = artifacts["scaler"]
	
	# Benign only training
	benign = data[data["Attack_Class"] == 0]
	
	X = benign[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
	X = scaler.transform(X)
	
	X_tensor = torch.tensor(X, dtype = torch.float32)
	dataset = TensorDataset(X_tensor)
	loader = DataLoader(dataset, batch_size = 128, shuffle = True)
	
	model = NIDSAutoencoder(input_dim = X.shape[1]).to(DEVICE)
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(
		model.parameters(),
		lr = 1e-3,
		weight_decay = 1e-5
	)
	
	best_loss = float("inf")
	patience, wait = 5, 0
	
	for epoch in range(1, 101):
		model.train()
		epoch_loss = 0.0
		
		for (x_batch, ) in loader:
			x_batch = x_batch.to(DEVICE)
			
			optimizer.zero_grad()
			x_hat = model(x_batch)
			loss = criterion(x_hat, x_batch)
			loss.backward()
			optimizer.step()
			
			epoch_loss += loss.item()
			
		epoch_loss /= len(loader)
		print(f"Epoch {epoch:03d} | Train MSE: {epoch_loss:.6f}")
		
		# Early stopping
		if epoch_loss < best_loss:
			best_loss = epoch_loss
			wait = 0
			torch.save(model.state_dict(),
				os.path.join(os.path.dirname(__file__), "autoencoder.pt")
			)
		else: 
			wait += 1
			if wait >= patience:
				print("Early stopping triggered.")
				break
				
	print("Autoencoder training complete.")
	
if __name__ == "__main__":
	main()
			
