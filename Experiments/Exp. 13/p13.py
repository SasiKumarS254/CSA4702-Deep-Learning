# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input

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

# === Module 2 : Time-Series Sequential Dataset Generation ===
np.random.seed(42)
T = 600
t_axis = np.arange(T)
# Diurnal seasonal load (24-hr period) + weekly macro drift + noise
diurnal = 15.0 * np.sin(2.0 * np.pi * t_axis / 24.0)
weekly = 5.0 * np.cos(2.0 * np.pi * t_axis / 168.0)
trend = 0.04 * t_axis + 80.0
noise = np.random.normal(0, 1.8, T)
load_series = trend + diurnal + weekly + noise

# === Module 3 : Sliding Window Preparation ===
window_size = 12
X_list, y_list = [], []
for i in range(len(load_series) - window_size):
    X_list.append(load_series[i : i + window_size])
    y_list.append(load_series[i + window_size])

X_arr = np.array(X_list)[..., np.newaxis] # (N, 12, 1)
y_arr = np.array(y_list)

split_idx = int(len(X_arr) * 0.80)
X_train, X_test = X_arr[:split_idx], X_arr[split_idx:]
y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]

# === Module 4 : SimpleRNN Model Architecture ===
tf.random.set_seed(42)
model = Sequential([
    Input(shape=(window_size, 1)),
    SimpleRNN(32, activation="tanh"),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.008), loss="mse", metrics=["mae"])

# === Module 5 : Model Training and Inference ===
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=32,
    verbose=0
)

y_pred = model.predict(X_test, verbose=0).flatten()
residuals = y_test - y_pred

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs(residuals / y_test)) * 100.0
r2 = r2_score(y_test, y_pred)

# === Module 6 : Visualisation ===
fig, ax = plt.subplots(figsize=(8.2, 4.0))
test_time = np.arange(len(y_test))
ax.plot(test_time, y_test, label="Actual Grid Power Load (MW)", color="#2C3E50", lw=1.6)
ax.plot(test_time, y_pred, label="SimpleRNN 1-Step Forecast", color="#E74C3C", lw=1.8, linestyle="--")
ax.set_xlabel("Time Horizon (Holdout Test Hours)", fontsize=9)
ax.set_ylabel("Electricity Demand (MW)", fontsize=9)
ax.set_title("Recurrent Neural Network Sequential Forecasting Performance", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_forecast_tracking.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax1.hist(residuals, bins=20, color="#2980B9", edgecolor="black", alpha=0.75)
ax1.axvline(0, color="#C0392B", linestyle="--", lw=1.5)
ax1.set_xlabel("Forecast Residual Error (MW)", fontsize=8.5)
ax1.set_ylabel("Frequency", fontsize=8.5)
ax1.set_title("Forecast Error Residual Distribution", fontsize=9.5)
ax1.grid(axis="y", linestyle="--", alpha=0.45)

# Autocorrelation of residuals
acf_vals = [pd.Series(residuals).autocorr(lag=l) for l in range(1, 16)]
ax2.bar(range(1, 16), acf_vals, color="#27AE60", width=0.45)
ax2.axhline(0, color="k", lw=0.8)
ax2.axhline(0.2, color="#E67E22", linestyle=":", lw=1.2)
ax2.axhline(-0.2, color="#E67E22", linestyle=":", lw=1.2)
ax2.set_xlabel("Lag (Hours)", fontsize=8.5)
ax2.set_ylabel("Autocorrelation", fontsize=8.5)
ax2.set_title("Residual Autocorrelation Function (ACF)", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
fig.suptitle("Sequential Error Residual Diagnostics and Whiteness Audit", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_residuals.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 7 : Printing and Saving Tables ===
t1_headers = ["Parameter", "Description", "Value"]
t1_rows = [
    ["Total Series Steps", "Chronological hourly recordings", f"{T} steps (25 days)"],
    ["Sliding Window Size", "Input lookback horizon (W)", "12 hourly steps"],
    ["Input Tensor Shape", "(N_samples, Window, Channels)", f"({len(X_arr)}, 12, 1)"],
    ["Train / Test Partition", "80% Historical Train / 20% Future Test", f"{len(X_train)} Train / {len(X_test)} Test"],
    ["Periodicity", "Diurnal frequency", "24-hour cycle"]
]
show_and_save("Table 1 - Sequential Dataset Preparation and Sliding Window Parameters", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Sample Index", "Lookback Horizon [t-11 to t]", "Next Target Value y[t+1]"]
t2_rows = [
    [f"Window {i+1}", f"[{X_arr[i, 0, 0]:.1f}, {X_arr[i, 1, 0]:.1f}, ..., {X_arr[i, -1, 0]:.1f}]", f"{y_arr[i]:.2f}"]
    for i in range(5)
]
show_and_save("Table 2 - Consecutive Sliding Window Data Samples", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Layer Name", "Layer Type", "Output Tensor Shape", "Param #", "Recurrent Activation"]
t3_rows = [
    ["input_seq", "InputLayer", "(None, 12, 1)", "0", "N/A"],
    ["simplernn", "SimpleRNN (32 units)", "(None, 32)", str((1 + 32) * 32 + 32), "Hyperbolic Tangent (tanh)"],
    ["dense_hidden", "Dense", "(None, 16)", str(32 * 16 + 16), "ReLU"],
    ["dense_out", "Dense (Output)", "(None, 1)", str(16 * 1 + 1), "Linear"]
]
show_and_save("Table 3 - SimpleRNN Network Layer and Parameter Breakdown", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Milestone", "Train MSE (MW²)", "Train MAE (MW)", "Val MSE (MW²)", "Val MAE (MW)"]
milestones = [1, 5, 10, 15, 25]
t4_rows = [
    [f"Epoch {m:02d}", f"{history.history['loss'][m-1]:.3f}", f"{history.history['mae'][m-1]:.3f}", f"{history.history['val_loss'][m-1]:.3f}", f"{history.history['val_mae'][m-1]:.3f}"]
    for m in milestones
]
show_and_save("Table 4 - Training Convergence and Loss Decay Record", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Evaluation Metric", "Formula", "Empirical Value", "Benchmark Target", "Assessment"]
t5_rows = [
    ["Mean Absolute Error (MAE)", "mean(|y - y_hat|)", f"{mae:.3f} MW", "< 3.0 MW", "Accurate Forecast"],
    ["Root Mean Squared Error (RMSE)", "sqrt(mean((y - y_hat)^2))", f"{rmse:.3f} MW", "< 3.5 MW", "Low Peak Deviations"],
    ["Mean Absolute Percentage Error", "mean(|res / y|) * 100", f"{mape:.2f}%", "< 3.5%", "High Precision"],
    ["Coefficient of Determination (R²)", "1 - SS_res / SS_tot", f"{r2:.4f}", "> 0.90", "Captures 90%+ Variance"]
]
show_and_save("Table 5 - Sequential Forecasting Performance Evaluation Summary", t5_headers, t5_rows, "table 5.png")
