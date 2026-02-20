# P5_Linux_Runtime/processes/inference.py

import os
import sys
import time
import queue

import numpy as np

# ---- project root fix (MANDATORY) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# ---- Phase 4 black box ----
from P4_Edge_Deployment.edge_inference import EdgeNIDS

# ---- Phase 5.7 imports ----
from decision.threat_evaluator import ThreatEvaluator

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "P2_Model_Training",
    "pytorch_model",
    "autoencoder.pt"
)

FEATURE_NAMES_PATH = os.path.join(
    PROJECT_ROOT,
    "P1_Preprocessing",
    "feature_names.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "P1_Preprocessing",
    "scaler_persistence.pkl"
)

# ---- initialize ONCE ----
edge = EdgeNIDS(
    model_path=MODEL_PATH,
    feature_names_path=FEATURE_NAMES_PATH,
    scaler_path=SCALER_PATH,
    device="cpu"
)

threat_evaluator = ThreatEvaluator()

# ---- Phase 5.3 imports ----
from features.feature_extractor import extract_features
from features.feature_map import map_features_to_vector, FEATURE_ORDER


def run(network_to_inference, alerts_to_logger, shutdown_event):
    """
    Phase 5.3.3 inference wiring.

    Consumes raw flow payloads, produces ordered feature vectors.
    NO model, NO scaler, NO decisions yet.
    """

    while not shutdown_event.is_set():
        try:
            flow_payload = network_to_inference.get(timeout=0.5)
        except queue.Empty:
            continue

        try:
            # Phase 5.3 → feature dict → ordered vector
            feature_dict = extract_features(flow_payload)
            feature_vector = map_features_to_vector(feature_dict)

            # ---- safety guards ----
            if feature_vector.shape != (50,):
                continue

            np.nan_to_num(
                feature_vector,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )

            # ---- black-box inference ----
            reconstruction_error = edge.infer(feature_vector)

            # ---- Call threat evaluator ----
            decision = threat_evaluator.update(
                error=reconstruction_error,
                timestamp=flow_payload["ts_list"][-1],
            )

            # ---- structured output (NO decisions) ----
            inference_result = {
                "timestamp": flow_payload["ts_list"][-1],
                "reconstruction_error": float(reconstruction_error),
                "source": "network",
                "status": "inferred",
            }

            alerts_to_logger.put(decision, block=False)

        except Exception:
            # Never let inference crash the daemon
            continue
