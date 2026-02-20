import numpy as np 

class RobustThreshold:
    def __init__(
        self,
        fpr = 0.01,
        trim_ratio = 0.02,
        hysteresis = 0.0
    ):
        self.fpr = fpr
        self.trim_ratio = trim_ratio
        self.hysteresis = hysteresis
        self.tau = None

    def fit(self, benign_scores):
        scores = np.sort(benign_scores)

        if self.trim_ratio > 0:
            k = int(len(scores) * self.trim_ratio)
            scores = scores[:-k]

        q = 1.0 - self.fpr
        self.tau = np.quantile(scores, q)
        return self.tau

    def predict(self, scores):
        if self.tau is None:
            raise RuntimeError("Threshold not fitted")

        upper = self.tau + self.hysteresis
        lower = self.tau - self.hysteresis

        alerts = scores > upper
        return alerts