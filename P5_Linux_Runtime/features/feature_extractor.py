# P5_Linux_Runtime/features/feature_extractor.py

import numpy as np


def _get_stats(arr: np.ndarray):
    """
    Returns (min, max, mean, std, var) with CICFlowMeter semantics.
    """
    n = len(arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    if n == 1:
        v = float(arr[0])
        return v, v, v, 0.0, 0.0

    return (
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.std(arr, ddof=1)),
        float(np.var(arr, ddof=1)),
    )


def _get_iat_stats(ts_sec: np.ndarray):
    """
    Returns (total, mean, std, max, min) in MICROSECONDS.
    """
    n = len(ts_sec)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    ts_sorted = np.sort(ts_sec)
    diffs_us = np.diff(ts_sorted) * 1_000_000.0

    if len(diffs_us) == 1:
        d = float(diffs_us[0])
        return d, d, 0.0, d, d

    return (
        float(np.sum(diffs_us)),
        float(np.mean(diffs_us)),
        float(np.std(diffs_us, ddof=1)),
        float(np.max(diffs_us)),
        float(np.min(diffs_us)),
    )


def extract_features(flow: dict) -> dict:
    """
    Convert raw NFStream flow payload into CICIDS2017-equivalent features.

    Input keys expected:
      ts_list, len_list, dir_list, flag_list
    """

    # -----------------------------
    # 1. Vectorize inputs (ONCE)
    # -----------------------------
    ts_arr = np.asarray(flow.get("ts_list", []), dtype=np.float64)
    len_arr = np.asarray(flow.get("len_list", []), dtype=np.float64)
    dir_arr = np.asarray(flow.get("dir_list", []), dtype=np.int8)
    flag_arr = np.asarray(flow.get("flag_list", []), dtype=np.int32)

    fwd_mask = (dir_arr == 0)
    bwd_mask = (dir_arr == 1)

    ts_fwd = ts_arr[fwd_mask]
    ts_bwd = ts_arr[bwd_mask]
    len_fwd = len_arr[fwd_mask]
    len_bwd = len_arr[bwd_mask]
    flags_fwd = flag_arr[fwd_mask]

    # -----------------------------
    # 2. Flow duration
    # -----------------------------
    if len(ts_arr) < 2:
        duration_us = 0.0
        duration_sec = 0.0
    else:
        duration_us = (np.max(ts_arr) - np.min(ts_arr)) * 1_000_000.0
        duration_sec = duration_us / 1_000_000.0

    # -----------------------------
    # 3. Packet counts & lengths
    # -----------------------------
    total_fwd_pkts = len(len_fwd)
    total_bwd_pkts = len(len_bwd)

    total_len_fwd = float(np.sum(len_fwd)) if total_fwd_pkts > 0 else 0.0
    total_len_bwd = float(np.sum(len_bwd)) if total_bwd_pkts > 0 else 0.0
    total_pkts = total_fwd_pkts + total_bwd_pkts
    total_len = total_len_fwd + total_len_bwd

    # -----------------------------
    # 4. Packet length statistics
    # -----------------------------
    fwd_min, fwd_max, fwd_mean, fwd_std, _ = _get_stats(len_fwd)
    bwd_min, bwd_max, bwd_mean, bwd_std, _ = _get_stats(len_bwd)
    flow_min, flow_max, flow_mean, flow_std, flow_var = _get_stats(len_arr)

    # -----------------------------
    # 5. Inter-arrival times
    # -----------------------------
    flow_iat_total, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _get_iat_stats(ts_arr)
    fwd_iat_total, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _get_iat_stats(ts_fwd)
    bwd_iat_total, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _get_iat_stats(ts_bwd)

    # -----------------------------
    # 6. Rates
    # -----------------------------
    if duration_sec > 0.0:
        flow_bytes_s = total_len / duration_sec
        flow_pkts_s = total_pkts / duration_sec
    else:
        flow_bytes_s = 0.0
        flow_pkts_s = 0.0

    # -----------------------------
    # 7. TCP flags
    # -----------------------------
    fin_cnt = int(np.count_nonzero(flag_arr & 1))
    syn_cnt = int(np.count_nonzero(flag_arr & 2))
    rst_cnt = int(np.count_nonzero(flag_arr & 4))
    psh_cnt = int(np.count_nonzero(flag_arr & 8))
    ack_cnt = int(np.count_nonzero(flag_arr & 16))
    urg_cnt = int(np.count_nonzero(flag_arr & 32))
    fwd_psh_cnt = int(np.count_nonzero(flags_fwd & 8))

    # -----------------------------
    # 8. Assemble feature dict
    # -----------------------------
    feats = {
        # Duration & rates
        "Flow Duration": duration_us,
        "Flow Bytes/s": flow_bytes_s,
        "Flow Packets/s": flow_pkts_s,

        # Packet counts
        "Total Fwd Packets": float(total_fwd_pkts),
        "Total Backward Packets": float(total_bwd_pkts),
        "Total Length of Fwd Packets": total_len_fwd,
        "Total Length of Bwd Packets": total_len_bwd,

        # Packet length stats (Flow)
        "Packet Length Min": flow_min,
        "Packet Length Max": flow_max,
        "Packet Length Mean": flow_mean,
        "Packet Length Std": flow_std,
        "Packet Length Variance": flow_var,

        # Packet length stats (Fwd/Bwd)
        "Fwd Packet Length Min": fwd_min,
        "Fwd Packet Length Max": fwd_max,
        "Fwd Packet Length Mean": fwd_mean,
        "Fwd Packet Length Std": fwd_std,
        "Bwd Packet Length Min": bwd_min,
        "Bwd Packet Length Max": bwd_max,
        "Bwd Packet Length Mean": bwd_mean,
        "Bwd Packet Length Std": bwd_std,

        # IAT stats
        "Flow IAT Total": flow_iat_total,
        "Flow IAT Mean": flow_iat_mean,
        "Flow IAT Std": flow_iat_std,
        "Flow IAT Max": flow_iat_max,
        "Flow IAT Min": flow_iat_min,

        "Fwd IAT Total": fwd_iat_total,
        "Fwd IAT Mean": fwd_iat_mean,
        "Fwd IAT Std": fwd_iat_std,
        "Fwd IAT Max": fwd_iat_max,
        "Fwd IAT Min": fwd_iat_min,

        "Bwd IAT Total": bwd_iat_total,
        "Bwd IAT Mean": bwd_iat_mean,
        "Bwd IAT Std": bwd_iat_std,
        "Bwd IAT Max": bwd_iat_max,
        "Bwd IAT Min": bwd_iat_min,

        # Flags
        "FIN Flag Count": fin_cnt,
        "SYN Flag Count": syn_cnt,
        "RST Flag Count": rst_cnt,
        "PSH Flag Count": psh_cnt,
        "ACK Flag Count": ack_cnt,
        "URG Flag Count": urg_cnt,
        "Fwd PSH Flags": float(fwd_psh_cnt),

        # Window bytes (not captured → parity-safe zero)
        "Init_Win_bytes_forward": 0.0,
        "Init_Win_bytes_backward": 0.0,
    }

    # -----------------------------
    # 9. Final numeric sanitation
    # -----------------------------
    for k, v in feats.items():
        if not np.isfinite(v):
            feats[k] = 0.0

    return feats
    