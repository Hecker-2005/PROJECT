# evaluate_baseline.py

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
	precision_recall_curve,
	roc_auc_score,
	auc,
	f1_score
)

ARTIFACT_DIR = "../../1_Preprocessing"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "iforest_model.joblib")

def evaluate(dataset_path, threshold = None):
	data = pd.read_parquet(dataset_path)
	artifacts  = joblib.load(f"{ARTIFACT_DIR}/scaler_persistence.pkl")
	model = joblib.load(MODEL_PATH)
	
	features = artifacts["network_features"]
	scalers = artifacts["scaler"]
	
	X = data[features].replace([np.inf, -np.inf], np.nan).fillna(0).values
	X = scalers.transform(X)
	
	y_true = (data["Attack_Class"] != 0).astype(int)
	scores = model.anomaly_score(X)
	
	roc = roc_auc_score(y_true, scores)
	precision, recall, thresholds = precision_recall_curve(y_true, scores)
	pr_auc = auc(recall, precision)
	
	best_threshold = threshold
	if threshold is None:
		f1 = (2 * precision * recall) / (precision + recall + 1e-9)
		idx = np.argmax(f1)
		best_threshold = thresholds[idx]
	return {
		"roc_auc": roc,
		"pr_auc": pr_auc,
		"threshold": best_threshold
	}
	
def main():
	 cic_metrics = evaluate(f"{ARTIFACT_DIR}/cicids_clean.parquet")
	 print("CICIDS:", cic_metrics)
	 
	 print("UNSW evaluation skipped in Phase 2 (feature schema mismatch by design).")
	 
if __name__ == "__main__":
	main()
	
