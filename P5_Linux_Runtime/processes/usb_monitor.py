# P5_Linux_Runtime/processes/usb_monitor.py
import os, sys, time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def run(alerts_to_logger, shutdown_event):
    while not shutdown_event.is_set():
        time.sleep(0.5)  # placeholder
