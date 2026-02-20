# P5_Linux_Runtime/ipc/messages.py
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class NetworkFlowMsg:
    payload: Dict[str, Any]   # placeholder (no features yet)
    ts: float

@dataclass
class InferenceResultMsg:
    score: float
    threshold: float
    decision: str             # "benign" | "anomalous"
    source: str               # "network"
    ts: float

@dataclass
class USBEventMsg:
    event: Dict[str, Any]
    decision: str
    source: str               # "usb"
    ts: float
