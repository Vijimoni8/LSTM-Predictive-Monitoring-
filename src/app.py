import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import gradio as gr

from data_loader import load_dataset
from preprocessing import prepare_sequences
from lstm_model import train_model
from anomaly_detection import calculate_anomalies, smooth_scores
from visualization import create_monitoring_plot

SEQ_LEN = 30
EPOCHS = 20
BATCH_SIZE = 32
WINDOW_SIZE_SMOOTH = 10

def run_monitoring_system(uploaded_file):
    log = (
        "=========================================================\n"
        "Predictive Monitoring System Initiated\n"
        "=========================================================\n"
    )
    try:
        df = load_dataset(uploaded_file)
        if uploaded_file is None:
            log += "No file uploaded. Generated synthetic dataset.\n"
        else:
            log += f"Dataset loaded successfully. Shape: {df.shape}\n"
            log += f"Columns: {list(df.columns)}\n"

        X, y, scaler, columns = prepare_sequences(df, SEQ_LEN)
        log += f"Windowed data dimensions: X={X.shape}, y={y.shape}\n"

        log += f"\nTraining LSTM model for {EPOCHS} epochs...\n"
        model = train_model(X, y, EPOCHS, BATCH_SIZE)
        log += "Training complete.\n"

        y_pred = model.predict(X, verbose=0)
        scores, threshold, anomalies = calculate_anomalies(y, y_pred)
        smoothed = smooth_scores(scores, WINDOW_SIZE_SMOOTH)

        log += f"\nAnomaly threshold (3-Sigma): {threshold:.6f}\n"
        log += f"Total anomalies detected: {int(anomalies.sum())} / {len(scores)}\n"
        if anomalies[-WINDOW_SIZE_SMOOTH:].any():
            log += "\nSystem Warning: Potential fault detected in recent window!"
        else:
            log += "\nSystem Status: Normal operation."

        fig = create_monitoring_plot(
            y, y_pred, scores, smoothed, threshold, anomalies,
            str(columns[0]), WINDOW_SIZE_SMOOTH
        )
        return fig, log
    except Exception as exc:
        return None, log + f"\nERROR: {exc}"

file_input = gr.File(
    label="Upload Time-Series Data (.csv or .xlsx)",
    file_types=[".csv", ".xlsx"],
    type="filepath",
    interactive=True
)
plot_output = gr.Plot(label="Prediction & Monitoring Results")
text_output = gr.Textbox(label="System Status and Log", lines=15, max_lines=15, autoscroll=True)

iface = gr.Interface(
    fn=run_monitoring_system,
    inputs=file_input,
    outputs=[plot_output, text_output],
    title="Unsupervised Predictive Monitoring System (LSTM)",
    description=(
        "Upload a time-series CSV or XLSX file to train an LSTM model for anomaly detection. "
        "If no file is uploaded, synthetic data is generated for demonstration."
    ),
    live=False,
    allow_flagging="never",
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    print("Launching Gradio UI...")
    iface.launch(debug=True, share=True)
