def extract_watermark(v, idxs):
    """
    Extract watermark features from the specified index positions.
    Returns a tensor of bits (0.0 or 1.0).
    """
    return (v[idxs] > 0).float()