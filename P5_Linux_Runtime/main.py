# P5_Linux_Runtime/main.py

import os
import sys
import time
from multiprocessing import Process

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from ipc.queues import create_ipc
from utils.shutdown import register_shutdown
from processes import network_ingest, inference, usb_monitor, logger


def main():
    ipc = create_ipc()
    shutdown_event = ipc["shutdown_event"]

    register_shutdown(shutdown_event)

    procs = [
        Process(
            target=network_ingest.run,
            args=(ipc["network_to_inference"], shutdown_event),
            name="network_ingest",
        ),
        Process(
            target=inference.run,
            args=(ipc["network_to_inference"], ipc["alerts_to_logger"], shutdown_event),
            name="inference",
        ),
        Process(
            target=usb_monitor.run,
            args=(ipc["alerts_to_logger"], shutdown_event),
            name="usb_monitor",
        ),
        Process(
            target=logger.run,
            args=(ipc["alerts_to_logger"], shutdown_event),
            name="logger",
        ),
    ]

    for p in procs:
        p.start()

    try:
        # Main process stays idle and interruptible
        while not shutdown_event.is_set():
            time.sleep(0.5)

    except KeyboardInterrupt:
        shutdown_event.set()

    finally:
        # HARD STOP — do not negotiate with blocked C extensions
        for p in procs:
            if p.is_alive():
                p.terminate()

        for p in procs:
            p.join(timeout=1)

        sys.exit(0)


if __name__ == "__main__":
    main()
