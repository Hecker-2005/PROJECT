# Decimal precision (Z score space)
DECIMAL_FEATURES = {
    "Flow IAT Mean": 2,
    "Flow IAT Std": 2,
    "Flow IAT Max": 2,
    "Flow IAT Min": 2,
    "Fwd IAT Mean": 2,
    "Fwd IAT Std": 2,
    "Fwd IAT Min": 2,
    "Active Mean": 2,
    "Active Std": 2,
    "Active Min": 2,
    "Active Max": 2,
    "Idle Std": 2,

    # Rates
    "Flow Bytes/s": 2,
    "Flow Packets/s": 2,
    "Bwd Packets/s": 2,
    "Fwd Avg Bulk Rate": 2,
    "Bwd Avg Packets/Bulk": 2,
}

# Integer-valued statistics
INTEGER_FEATURES = [
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Fwd Header Length",
    "Bwd Header Length",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",

    # Flags
    "FIN Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "RST Flag Count",
    "Fwd PSH Flags",
    "Fwd URG Flags",
]

# Conservative clipping (Z-score space)
CLIP_BOUNDS = {
    "Flow Bytes/s": (-5.0, 5.0),
    "Flow Packets/s": (-5.0, 5.0),
    "Bwd Packets/s": (-5.0, 5.0),
}