# P5_Linux_Runtime/decision/decision_config.py

# -----------------------------
# Frozen thresholds (Phase 4)
# -----------------------------
T_WARN = 0.018      # Warning threshold (≈95th percentile)
T_CRIT = 0.025      # Critical threshold (≈99–99.5th percentile)

# -----------------------------
# Leaky bucket parameters
# -----------------------------
DECAY_RATE = 0.5        # score units per second
SCORE_CAP = 200.0       # hard upper bound

# -----------------------------
# Severity thresholds
# -----------------------------
ELEVATED_THRESHOLD = 5.0
ATTACK_THRESHOLD = 20.0

# -----------------------------
# Latching
# -----------------------------
ATTACK_LATCH_SECONDS = 10.0
