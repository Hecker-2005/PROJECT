import os
import sys

P5_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, P5_ROOT)

from decision.threat_evaluator import ThreatEvaluator

import time
from decision.threat_evaluator import ThreatEvaluator

te = ThreatEvaluator()

# Helper to print compact state
def show(label, result):
    print(
        f"{label:12} | "
        f"err={result['inference']['reconstruction_error']:.3f} | "
        f"flow={result['inference']['flow_state']:10} | "
        f"score={result['system_state']['threat_score']:6.2f} | "
        f"sev={result['system_state']['severity']:8} | "
        f"trend={result['system_state']['trend']}"
    )

t = time.time()

# 1. Benign noise
show("benign-1", te.update(0.010, t))
time.sleep(1)
show("benign-2", te.update(0.012, t + 1))

# 2. Suspicious jitter
time.sleep(1)
show("sus-1", te.update(0.020, t + 2))
time.sleep(1)
show("sus-2", te.update(0.021, t + 3))

# 3. Single anomaly spike
time.sleep(1)
show("anom-1", te.update(0.030, t + 4))

# 4. Decay check (no new anomalies)
time.sleep(3)
show("decay", te.update(0.010, t + 7))

# 5. Burst attack
for i in range(5):
    time.sleep(0.5)
    show(f"burst-{i}", te.update(0.050, t + 8 + i * 0.5))

# 6. Latch behavior
time.sleep(2)
show("latched", te.update(0.010, t + 15))
