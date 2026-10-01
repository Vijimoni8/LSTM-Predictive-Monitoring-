from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

def build_model(sequence_length, num_features):
    """Build and compile the next-step LSTM forecasting model."""
    model = Sequential([
        LSTM(64, input_shape=(sequence_length, num_features), return_sequences=False),
        Dense(num_features)
    ])
    model.compile(optimizer=Adam(0.001), loss="mse")
    return model

def train_model(X, y, epochs=20, batch_size=32):
    model = build_model(X.shape[1], X.shape[2])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model
