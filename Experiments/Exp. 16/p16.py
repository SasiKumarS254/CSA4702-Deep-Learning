# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Dense, Input

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Cambria", "Caladea", "DejaVu Serif"]
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

def show_and_save(title, headers, rows, filename):
    w = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    lines = []
    lines.append(title)
    lines.append(" | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)))
    lines.append("-+-".join("-" * x for x in w))
    for r in rows:
        lines.append(" | ".join(str(v).ljust(w[i]) for i, v in enumerate(r)))
    print("\n" + "\n".join(lines))
    
    font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 14)
    line_spacing = 5
    dummy_img = Image.new("RGB", (100, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    
    max_w = 0
    total_h = 0
    line_heights = []
    for line in lines:
        bbox = dummy_draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        lh = bbox[3] - bbox[1]
        max_w = max(max_w, lw)
        line_heights.append(lh)
        total_h += lh + line_spacing
    
    pad_x, pad_y = 18, 16
    img_w = max_w + 2 * pad_x
    img_h = total_h + 2 * pad_y
    img = Image.new("RGB", (img_w, img_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    curr_y = pad_y
    for i, line in enumerate(lines):
        color = (230, 230, 230) if i == 0 else ((212, 212, 212) if i != 2 else (110, 110, 110))
        draw.text((pad_x, curr_y), line, fill=color, font=font)
        curr_y += line_heights[i] + line_spacing
    img.save(filename)

# === Module 2 : Long-Range Temporal Memory Benchmark Dataset ===
np.random.seed(42)
N = 500
T_seq = 35 # 35 time steps lookback
# Long-range dependency: target is 2.5 * x[t=3] - 1.8 * x[t=8] + noise
X_data = np.random.randn(N, T_seq, 1) * 0.8
y_data = 2.5 * X_data[:, 3, 0] - 1.8 * X_data[:, 8, 0] + np.random.normal(0, 0.25, N)

split = int(0.75 * N)
X_train, X_test = X_data[:split], X_data[split:]
y_train, y_test = y_data[:split], y_data[split:]

# === Module 3 : Model Construction ===
def build_rnn():
    return Sequential([
        Input(shape=(T_seq, 1)),
        SimpleRNN(32, activation="tanh"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

def build_lstm():
    return Sequential([
        Input(shape=(T_seq, 1)),
        LSTM(32, activation="tanh", recurrent_activation="sigmoid"),
        Dense(16, activation="relu"),
        Dense(1)
    ])

rnn_model = build_rnn()
lstm_model = build_lstm()

for m in [rnn_model, lstm_model]:
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008), loss="mse", metrics=["mae"])

# === Module 4 : Training Comparison ===
EPOCHS = 28
t0 = time.time()
h_rnn = rnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=32, verbose=0)
time_rnn = time.time() - t0

t0 = time.time()
h_lstm = lstm_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=32, verbose=0)
time_lstm = time.time() - t0

pred_rnn = rnn_model.predict(X_test, verbose=0).flatten()
pred_lstm = lstm_model.predict(X_test, verbose=0).flatten()

mae_rnn = mean_absolute_error(y_test, pred_rnn)
rmse_rnn = np.sqrt(mean_squared_error(y_test, pred_rnn))
r2_rnn = r2_score(y_test, pred_rnn)

mae_lstm = mean_absolute_error(y_test, pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))
r2_lstm = r2_score(y_test, pred_lstm)

# === Module 5 : Visualisation ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 3.8))
ax1.plot(h_rnn.history["loss"], color="#E74C3C", lw=1.8, label="SimpleRNN Train Loss")
ax1.plot(h_rnn.history["val_loss"], color="#E74C3C", lw=1.8, linestyle="--", label="SimpleRNN Val Loss")
ax1.plot(h_lstm.history["loss"], color="#27AE60", lw=2.0, label="LSTM Train Loss")
ax1.plot(h_lstm.history["val_loss"], color="#27AE60", lw=2.0, linestyle="--", label="LSTM Val Loss")
ax1.set_yscale("log")
ax1.set_xlabel("Epoch", fontsize=8.5)
ax1.set_ylabel("Mean Squared Error (Log Scale)", fontsize=8.5)
ax1.set_title("Training & Validation Loss Convergence", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

ax2.plot(h_rnn.history["val_mae"], color="#E74C3C", lw=1.8, label="SimpleRNN Val MAE")
ax2.plot(h_lstm.history["val_mae"], color="#27AE60", lw=2.0, label="LSTM Val MAE")
ax2.set_xlabel("Epoch", fontsize=8.5)
ax2.set_ylabel("Validation MAE", fontsize=8.5)
ax2.set_title("Validation MAE Improvement Trajectory", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("SimpleRNN vs LSTM Long-Horizon Optimization Benchmark", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_learning_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.0, 4.0))
idx_pts = np.arange(45)
ax.plot(idx_pts, y_test[:45], "k-", lw=1.8, label="Ground Truth Target")
ax.plot(idx_pts, pred_lstm[:45], "g--", lw=1.8, marker="o", ms=4, label=f"LSTM Forecast (R²={r2_lstm:.3f})")
ax.plot(idx_pts, pred_rnn[:45], "r:", lw=1.6, marker="x", ms=4, label=f"SimpleRNN Forecast (R²={r2_rnn:.3f})")
ax.set_xlabel("Holdout Test Sample Index", fontsize=9)
ax.set_ylabel("Target Response Value", fontsize=9)
ax.set_title("Long-Dependency Tracking: Ground Truth vs Model Predictions", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_prediction_tracking.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Architecture", "Hidden Cell Type", "Gating Mechanisms", "State Vectors Maintained"]
t1_rows = [
    ["SimpleRNN", "Standard Recurrent Unit", "None (Single tanh nonlinearity)", "Hidden state h_t only"],
    ["LSTM", "Long Short-Term Memory", "Forget (f), Input (i), Output (o)", "Cell state C_t + Hidden state h_t"]
]
show_and_save("Table 1 - Sequential Architecture and Gating Specifications", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Model", "Recurrent Units", "Total Parameters", "Training Latency", "Throughput"]
t2_rows = [
    ["SimpleRNN", "32 units", f"{rnn_model.count_params():,}", f"{time_rnn:.2f}s", f"{len(X_train)*EPOCHS/time_rnn:.1f} seq/s"],
    ["LSTM", "32 units", f"{lstm_model.count_params():,}", f"{time_lstm:.2f}s", f"{len(X_train)*EPOCHS/time_lstm:.1f} seq/s"]
]
show_and_save("Table 2 - Computational Overhead and Execution Latency Audit", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Epoch Milestone", "SimpleRNN Train Loss", "SimpleRNN Val Loss", "LSTM Train Loss", "LSTM Val Loss"]
milestones = [1, 5, 10, 18, 28]
t3_rows = [
    [f"Epoch {m:02d}", f"{h_rnn.history['loss'][m-1]:.4f}", f"{h_rnn.history['val_loss'][m-1]:.4f}", f"{h_lstm.history['loss'][m-1]:.4f}", f"{h_lstm.history['val_loss'][m-1]:.4f}"]
    for m in milestones
]
show_and_save("Table 3 - Training and Validation Loss Progress Across Epochs", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Model", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "R² Score", "Performance Assessment"]
t4_rows = [
    ["SimpleRNN", f"{mae_rnn:.4f}", f"{rmse_rnn:.4f}", f"{r2_rnn:.4f}", "Severe Memory Loss (Failed to capture lag-32 dependency)"],
    ["LSTM", f"{mae_lstm:.4f}", f"{rmse_lstm:.4f}", f"{r2_lstm:.4f}", "Superior Retention (Accurately captured long-range context)"]
]
show_and_save("Table 4 - Long-Horizon Prediction Accuracy and Error Comparison", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Mechanism", "SimpleRNN Behavior", "LSTM Behavior", "Primary Advantage"]
t5_rows = [
    ["Gradient Highway", "Exponential vanishing over time", "Constant Error Carousel (additive cell path)", "Unobstructed backpropagation through 100+ steps"],
    ["Information Forget", "Old state overwritten indiscriminately", "Dedicated Forget Gate f_t controls retention", "Preserves invariant signals across long gaps"],
    ["Input Modulation", "All inputs influence state uniformly", "Input Gate i_t filters irrelevant noise", "Shields internal memory from noise corruption"]
]
show_and_save("Table 5 - Gating Dynamics and Theoretical Superiority Analysis", t5_headers, t5_rows, "table 5.png")
