# P5_Linux_Runtime/ipc/queues.py
from multiprocessing import Queue, Event

def create_ipc():
    return {
        "network_to_inference": Queue(maxsize=2048),
        "alerts_to_logger": Queue(maxsize=2048),
        "shutdown_event": Event(),
    }
