# massif/taxonomy.py
def classify_model(metrics):
    """Classify model based on MASSIF metrics dict."""
    delta_t = metrics.get('delta_t')
    v_peak = metrics.get('v_peak', 1.0)
    max_norm = metrics.get('max_norm', 0)

    if max_norm > 200:
        if delta_t is not None and delta_t < -2:
            return "Accelerator (Runaway-norm)"
        return "Runaway"
    if v_peak < 0.05:
        return "Stochastic"
    if delta_t is None:
        return "Unknown"
    if delta_t < -2:
        return "Accelerator"
    if delta_t > 2:
        return "Decelerator"
    return "Neutral"