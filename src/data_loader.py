import numpy as np
import pandas as pd

def generate_synthetic_data(num_timesteps=1000, anomaly_start=850):
    """Generate a demo multivariate time-series dataset with an injected anomaly."""
    t = np.arange(num_timesteps)
    v1 = 0.1 * t + 10 * np.sin(t / 20) + np.random.normal(0, 0.5, num_timesteps)
    v2 = 5 * np.cos(t / 10) * np.exp(-t / 500) + np.random.normal(0, 0.3, num_timesteps)
    v1[anomaly_start:] += 10
    v2[anomaly_start:] -= 5
    return pd.DataFrame({"Time": t, "Sensor_A": v1, "Pressure_B": v2}).set_index("Time")

def load_dataset(file_path=None):
    """Load CSV/Excel data, or generate synthetic data when no file is supplied."""
    if file_path is None:
        return generate_synthetic_data()
    lower_path = file_path.lower()
    if lower_path.endswith(".csv"):
        return pd.read_csv(file_path)
    if lower_path.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_path)
    raise ValueError("Unsupported file type. Please use .csv or .xlsx.")
