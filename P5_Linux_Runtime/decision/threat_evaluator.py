import time

from decision.decision_config import (
    T_WARN,
    T_CRIT,
    DECAY_RATE,
    SCORE_CAP,
    ELEVATED_THRESHOLD,
    ATTACK_THRESHOLD,
    ATTACK_LATCH_SECONDS,
)

# Flow-level states
BENIGN = "BENIGN"
SUSPICIOUS = "SUSPICIOUS"
ANOMALOUS = "ANOMALOUS"

# System severity levels
CLEAN = "CLEAN"
ELEVATED = "ELEVATED"
ATTACK = "ATTACK"

class ThreatEvaluator:
    """
    Phase 5.5 Decision Engine.
    Implements static thresholding + leaky bucket correlation.
    """

    def __init__(self):
        # Leaky bucket score
        self.score = 0.0

        # Time tracking
        self.last_update_ts = None

        # Latching
        self.attack_latched_until = 0.0

    def _classify_flow(self, error: float) -> str:
        """
        Classify a single reconstruction error using frozen thresholds.
        """

        if error < T_WARN:
            return BENIGN
        elif error < T_CRIT:
            return SUSPICIOUS
        else:
            return ANOMALOUS

    def _apply_decay(self, current_ts: float):
        """
        Apply time-based decay to the threat score.
        """
        if self.last_update_ts is None:
            # First observation, nothing to decay yet
            self.last_update_ts = current_ts
            return

        elapsed = current_ts - self.last_update_ts
        if elapsed <= 0:
            return

        decay_amount = DECAY_RATE * elapsed
        self.score -= decay_amount

        if self.score < 0.0:
            self.score = 0.0

        self.last_update_ts = current_ts

    def _severity_weight(self, error: float) -> float:
        """
        Compute how much score to add for an anomalous flow.
        Higher error -> higher weight.
        """

        # Linear scaling above crit threshold
        # Minimum weight = 1.0
        weight = max(1.0, error / T_CRIT)
        return weight

    def _apply_increment(self, flow_state: str, error: float):
        """
        Increase threat score based on flow classification.
        """

        if flow_state != ANOMALOUS:
            return

        increment = self._severity_weight(error)
        self.score += increment

        if self.score > SCORE_CAP:
            self.score = SCORE_CAP

    def _evaluate_severity(self, current_ts: float) -> str:
        """
        Map threat score to system severity with ATTACK latching.
        """

        # If currently latched in ATTACK, honor latch
        if current_ts < self.attack_latched_until:
            return ATTACK

        # Determine severity based on score
        if self.score >= ATTACK_THRESHOLD:
            # Enter ATTACK and latch it
            self.attack_latched_until = current_ts + ATTACK_LATCH_SECONDS
            return ATTACK

        if self.score >= ELEVATED_THRESHOLD:
            return ELEVATED
        
        return CLEAN

    def update(self, error: float, timestamp: float) -> dict:
        """
        Update threat state using a single reconstruction error.

        Returns a structured decision payload.
        """

        # 1. Time decay
        self._apply_decay(timestamp)

        # 2. Flow classification
        flow_state = self._classify_flow(error)

        # 3. Score increment (if anomalous)
        self._apply_increment(flow_state, error)

        # 4. Severity evaluation (with latching)
        severity = self._evaluate_severity(timestamp)

        # 5. Trend (simple derivation)
        trend = "stable"
        if flow_state == ANOMALOUS:
            trend = "increasing"
        elif self.score == 0.0:
            trend = "decreasing"

        # 6. Decision payload
        decision = {
            "timestamp": timestamp,

            "inference": {
                "reconstruction_error": float(error),
                "threshold_warn": T_WARN,
                "threshold_crit": T_CRIT,
                "flow_state": flow_state,
            },
            
            "system_state": {
                "threat_score": round(self.score, 3),
                "severity": severity,
                "trend": trend,
            },
        }

        return decision
