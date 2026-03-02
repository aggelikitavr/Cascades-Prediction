def extract_features(cascade, until=None):
    """
    Extracts structural and temporal features from a cascade.
    If 'until' is specified, features are calculated for the first k nodes only.
    """
    if until is not None:
        # Create a snapshot of the cascade at the observation threshold k
        sub_cascade = cascade.subcascade(until)
    else:
        sub_cascade = cascade

    # Feature order must remain consistent for model training and interpretation
    return [
        sub_cascade.size(),          # 1. Current number of nodes
        sub_cascade.depth(),         # 2. Maximum tree depth
        sub_cascade.breadth(),       # 3. Maximum tree breadth
        sub_cascade.wiener_index(),  # 4. Structural virality measure
        sub_cascade.duration(),      # 5. Time elapsed since start
        sub_cascade.acceleration()   # 6. Change in resharing speed
    ]