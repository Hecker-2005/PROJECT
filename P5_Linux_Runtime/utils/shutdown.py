# P5_Windows_Agent/utils/shutdown.py
import signal

def register_shutdown(shutdown_event):
    def _handler(signum, frame):
        shutdown_event.set()
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)
