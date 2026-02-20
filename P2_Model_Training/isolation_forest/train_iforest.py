# train_iforest.py

import os
import joblib
import numpy as np
import pandas as pd
from iforest_core import IsolationForest

ARTIFACT_DIR = "../../1_Preprocessing"
OUT_DIR = "./"

def main():
	data = pd.read_parquet(f"{ARTIFACT_DIR}/cicids_clean.parquet")
	artifacts = joblib.load(f"{ARTIFACT_DIR}/scaler_persistence.pkl")
	
	features = artifacts["network_features"]
	scaler = artifacts["scaler"]
	
	X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
	X = scaler.transform(X)
	
	model = IsolationForest(
		n_estimators = 150,
		max_samples = 250
	)
	
	model.fit(X)
	
	MODEL_PATH = os.path.join(os.path.dirname(__file__), "iforest_model.joblib")
	joblib.dump(model, MODEL_PATH)
	print("Isolation Forest trained and saved as: ", MODEL_PATH)
	
if __name__ == "__main__":
	main()

