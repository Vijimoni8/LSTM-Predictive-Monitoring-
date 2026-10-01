import numpy as np
import matplotlib.pyplot as plt

def create_monitoring_plot(y_true, y_pred, scores, smoothed, threshold, anomalies, feature_name, window_size):
    """Create forecasting and anomaly-monitoring plots."""
    plt.style.use("fivethirtyeight")
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(y_true[:, 0], label="Actual", linewidth=1, color="#1f77b4")
    ax1.plot(y_pred[:, 0], label="Predicted", linewidth=1, color="#ff7f0e")
    ax1.set_title(f"Forecasting: Actual vs Predicted (First Feature: {feature_name})", fontsize=16)
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(scores, label="Raw Anomaly Score (MSE)", color="purple", alpha=0.6)
    ax2.plot(smoothed, label=f"Smoothed Score (Window={window_size})", color="darkorange", linewidth=2)
    ax2.axhline(threshold, color="red", linestyle="--", label="3-Sigma Threshold")
    x_indices = np.arange(len(scores))[anomalies]
    ax2.scatter(x_indices, scores[anomalies], color="red", marker="x", s=50, label="Detected Anomaly Points")
    ax2.set_title("Anomaly & Predictive Monitoring", fontsize=16)
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Error (MSE)")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout(pad=3.0)
    return fig
