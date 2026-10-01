import numpy as np
from sklearn.preprocessing import MinMaxScaler

def prepare_sequences(df, sequence_length=30):
    """Scale numeric columns and create next-step sliding-window sequences."""
    data = df.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("No numeric columns found in the dataset.")
    if len(data) <= sequence_length:
        raise ValueError(
            f"Dataset has {len(data)} rows; at least {sequence_length + 1} are required."
        )

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled) - sequence_length):
        X.append(scaled[i:i + sequence_length])
        y.append(scaled[i + sequence_length])
    return np.asarray(X), np.asarray(y), scaler, data.columns
