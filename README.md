# LSTM-Based Predictive Monitoring and Deviation Scoring

An unsupervised predictive monitoring demo that uses an LSTM model to forecast multivariate time-series data and detect unusual observations from prediction errors.

## Project structure

- `src/data_loader.py` — CSV/Excel loading and synthetic data generation
- `src/preprocessing.py` — numeric feature selection, scaling, and sliding windows
- `src/lstm_model.py` — LSTM model creation and training
- `src/anomaly_detection.py` — MSE anomaly scores, 3-sigma threshold, and smoothing
- `src/visualization.py` — plots for forecasting and anomaly monitoring
- `src/app.py` — Gradio interface and workflow
- `data/` — optional location for sample datasets

## Run locally

Use Python 3.10 or a TensorFlow-supported Python version.

```bash
pip install -r requirements.txt
python src/app.py
```

The app launches a Gradio interface. You can upload a CSV or Excel file, or leave the upload empty to use generated demo data.

## Input data

The application analyzes numeric columns. Ensure the file contains enough rows to create 30-step sequences (at least 31 rows).

## Important note

This is a demonstration implementation. The current workflow fits the scaler and trains/evaluates the model on the same dataset. For a rigorous evaluation, use a chronological train/test split and estimate the anomaly threshold from training or known-normal validation data.
