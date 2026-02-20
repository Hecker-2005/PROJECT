# P5_Linux_Runtime/processes/network_ingest.py

import os
import sys
import time
import queue

from nfstream import NFStreamer, NFPlugin

import subprocess
import re

# ---- project root fix (MANDATORY, same as Phase 4) ----
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

def detect_active_interface():
    """
    Returns the first non-loopback interface that is UP and has an IPv4 address.
    """
    output = subprocess.check_output(["ip", "-o", "-4", "addr", "show"], text=True)

    for line in output.splitlines():
        parts = line.split()
        iface = parts[1]

        if iface != "lo":
            return iface

    raise RuntimeError("No active non-loopback interface found")

# ---------------- CONFIG ----------------
CAPTURE_INTERFACE = os.getenv("NIDS_INTERFACE")

if not CAPTURE_INTERFACE:
    CAPTURE_INTERFACE = detect_active_interface()
     # CHANGE if needed

ACTIVE_TIMEOUT = 60            # seconds
IDLE_TIMEOUT = 30             # seconds
MAX_PACKETS = 1000             # hard safety cap per flow
QUEUE_PUT_TIMEOUT = 0.01       # seconds

print(f"[INGEST] Capturing on interface: {CAPTURE_INTERFACE}", flush=True)

# ---------------------------------------

class RawDataCollector(NFPlugin):
    """
    Storage-only NFStream plugin.
    Collects RAW packet metadata for Phase 5.3 feature reconstruction.
    NO feature computation is performed here.
    """

    def on_init(self, packet, flow):
        flow.udps.packet_timestamps = []
        flow.udps.packet_sizes = []
        flow.udps.packet_dirs = []
        flow.udps.packet_flags = []

        self._append_packet(packet, flow)

    def on_update(self, packet, flow):
        self._append_packet(packet, flow)

    def _append_packet(self, packet, flow):
        # Enforce hard cap to prevent memory blowups (DDoS safety)
        if len(flow.udps.packet_sizes) >= MAX_PACKETS:
            return

        flow.udps.packet_timestamps.append(packet.time)
        flow.udps.packet_sizes.append(packet.raw_size)
        flow.udps.packet_dirs.append(packet.direction)

        # TCP flag bitmask (safe + compact)
        flags = 0
        flags |= int(packet.syn) << 0
        flags |= int(packet.ack) << 1
        flags |= int(packet.fin) << 2
        flags |= int(packet.rst) << 3
        flags |= int(packet.psh) << 4
        flags |= int(packet.urg) << 5
        flow.udps.packet_flags.append(flags)

def run(network_to_inference, shutdown_event):
    """
    Phase 5.2 ingestion process (PATCHED).
    """

    streamer = NFStreamer(
        source=CAPTURE_INTERFACE,
        promiscuous_mode=True,
        snapshot_length=1536,
        idle_timeout=IDLE_TIMEOUT,
        active_timeout=ACTIVE_TIMEOUT,
        accounting_mode=1,
        n_dissections=20,
        statistical_analysis=False,
        splt_analysis=0,
        udps=[RawDataCollector()],
    )

    stream_iter = iter(streamer)

    try:
        while not shutdown_event.is_set():
            try:
                flow = next(stream_iter)
            except StopIteration:
                break
            except Exception:
                continue

            payload = {
                # --- keys aligned to feature_extractor ---
                "ts_list": flow.udps.packet_timestamps,
                "len_list": flow.udps.packet_sizes,
                "dir_list": flow.udps.packet_dirs,
                "flag_list": flow.udps.packet_flags,
            }

            try:
                network_to_inference.put(payload, timeout=QUEUE_PUT_TIMEOUT)
            except queue.Full:
                continue

    finally:
        # Clean shutdown, ONCE
        del streamer
        time.sleep(0.1)

