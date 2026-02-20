def _has(features, idx):
	return all(f in idx for f in features)
	

def bytes_consistency(x, idx):
	"""
    	TotalBytes >= TotalFwdBytes + TotalBwdBytes
	"""
	required = ["Total Bytes", "Total Fwd Bytes", "Total Bwd Bytes"]
	if not _has(required, idx):
		return True
	return x[idx["Total Bytes"]] >= (
		x[idx["Total Fwd Bytes"]] + x[idx["Total Bwd Bytes"]]
	)


def flow_duration_consistency(x, idx):
	"""
	FlowDuration >= sum of IAT components (approximate, conservative)
	"""
	required = ["Flow Duration", "Fwd IAT Mean", "Bwd IAT Mean"]
	if not _has(required, idx):
		return True
	return x[idx["Flow Duration"]] >= (
		x[idx["Fwd IAT Mean"]] + x[idx["Bwd IAT Mean"]]
	)


def protocol_flag_consistency(x, idx):
	required = [
		"Protocol"
		"FIN Flag Cnt",
		"SYN Flag Cnt",
		"RST Flag Cnt",
		"PSH Flag Cnt",
		"ACK Flag Cnt"
	]
	if not _has(required, idx):
		return True
	if x[idx["Protocol"]] == 17:  # UDP
		return all(x[idx[f]] == 0 for f in required[1:])
	return True

