import sys
import os
import torch
import numpy as np 
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from P2_Model_Training.pytorch_model.model_architecture import NIDSAutoencoder
from P4_Edge_Deployment.feature_squeezer import FeatureSqueezer
from P4_Edge_Deployment.squeezing_config import (
    DECIMAL_FEATURES,
    INTEGER_FEATURES,
    CLIP_BOUNDS
)
from P4_Edge_Deployment.thresholding import RobustThreshold

class EdgeNIDS:
    def __init__(
        self,
        model_path,
        feature_names_path,
        scaler_path,
        device = "cpu"
    ):
        self.device = device

        self.feature_names = joblib.load(feature_names_path)
        bundle = joblib.load(scaler_path)
        self.scaler = bundle("scaler")

        self.model = NIDSAutoencoder(input_dim = len(self.feature_names))
        self.model.load_state_dict(
            torch.load(model_path, map_location = device)
        )
        self.model.to(device)
        self.model.eval()

        self.squeezer = FeatureSqueezer(
            feature_names = self.feature_names,
            decimal_features = DECIMAL_FEATURES,
            integer_features = INTEGER_FEATURES,
            clip_bounds = CLIP_BOUNDS
        )

        self.threshold = RobustThreshold(
            fpr = 0.01,
            trim_ratio = 0.02,
            hysteresis = 0.01
        )
    
    def fit_threshold(self, benign_data):
        X = benign_data[self.feature_names].values.astype(np.float64)
        X = np.nan_to_num(X, nan = 0.0, posinf = 0.0, neginf = 0.0)
        X = self.scaler.transform(X)
        X = self.squeezer.squeeze(X)

        with torch.no_grad():
            X_t = torch.tensor(X, dtype = torch.float32, device = self.device)
            recon = self.model(X_t)
            scores = torch.mean((X_t - recon) ** 2, dim = 1).cpu().numpy()

        return self.threshold.fit(scores)

    def infer(self, sample):
        x = sample[self.feature_names].values.astype(np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.scaler.transform(x)
        x = self.squeezer.squeeze(x)

        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            recon = self.model(x_t)
            score = torch.mean((x_t - recon) ** 2, dim=1).cpu().numpy()

        alert = self.threshold.predict(score)
        return score[0], bool(alert[0])
