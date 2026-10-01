import numpy as np
import pandas as pd

def calculate_anomalies(y_true, y_pred):
    """Calculate per-timestep MSE and flag scores above a 3-sigma threshold."""
    scores = np.mean(np.square(y_pred - y_true), axis=1)
    threshold = float(np.mean(scores) + 3 * np.std(scores))
    anomalies = scores > threshold
    return scores, threshold, anomalies

def smooth_scores(scores, window_size=10):
    return pd.Series(scores).rolling(window=window_size, min_periods=1).mean()
