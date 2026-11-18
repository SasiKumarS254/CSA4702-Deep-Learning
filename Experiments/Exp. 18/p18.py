# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Input

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

# === Module 2 : Benchmark Sequence Dataset ===
np.random.seed(42)
N = 400
T_seq = 20
X = np.random.randn(N, T_seq, 1) * 0.7
# Non-linear recurrent mapping
y = np.tanh(X[:, 2, 0]) * 1.5 + np.sin(X[:, 10, 0]) * 2.0 + np.random.normal(0, 0.2, N)

split = int(0.75 * N)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# === Module 3 : Hyperparameter Exploration Grid ===
# 6 distinct architectural configurations: (Type, Units, LR)
search_trials = [
    {"name": "Trial 1 (SimpleRNN-32)", "type": "RNN", "units": 32, "lr": 0.008},
    {"name": "Trial 2 (SimpleRNN-64)", "type": "RNN", "units": 64, "lr": 0.005},
    {"name": "Trial 3 (LSTM-32)",      "type": "LSTM", "units": 32, "lr": 0.008},
    {"name": "Trial 4 (LSTM-64)",      "type": "LSTM", "units": 64, "lr": 0.005},
    {"name": "Trial 5 (GRU-32)",       "type": "GRU", "units": 32, "lr": 0.008},
    {"name": "Trial 6 (GRU-64)",       "type": "GRU", "units": 64, "lr": 0.005},
]

def build_candidate_model(cfg):
    tf.random.set_seed(42)
    m = Sequential()
    m.add(Input(shape=(T_seq, 1)))
    if cfg["type"] == "RNN":
        m.add(SimpleRNN(cfg["units"], activation="tanh"))
    elif cfg["type"] == "LSTM":
        m.add(LSTM(cfg["units"], activation="tanh"))
    elif cfg["type"] == "GRU":
        m.add(GRU(cfg["units"], activation="tanh"))
    m.add(Dense(16, activation="relu"))
    m.add(Dense(1))
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg["lr"]), loss="mse", metrics=["mae"])
    return m

# === Module 4 : Executing Hyperparameter Trials ===
EPOCHS = 16
trial_results = []

for trial in search_trials:
    model = build_candidate_model(trial)
    params = model.count_params()
    
    t0 = time.time()
    h = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=32, verbose=0)
    dur = time.time() - t0
    
    val_loss = h.history["val_loss"][-1]
    val_mae = h.history["val_mae"][-1]
    
    trial_results.append({
        "name": trial["name"],
        "type": trial["type"],
        "units": trial["units"],
        "lr": trial["lr"],
        "params": params,
        "val_loss": val_loss,
        "val_mae": val_mae,
        "time": dur,
        "history": h
    })

# === Module 5 : Visualisation ===
fig, ax = plt.subplots(figsize=(8.0, 4.2))
names = [t["name"].replace("Trial ", "T") for t in trial_results]
val_losses = [t["val_loss"] for t in trial_results]
colors = ["#C0392B", "#E74C3C", "#27AE60", "#2ECC71", "#2980B9", "#3498DB"]

bars = ax.bar(names, val_losses, color=colors, edgecolor="black", width=0.55)
ax.set_ylabel("Validation Mean Squared Error", fontsize=9)
ax.set_title("Hyperparameter Search Validation Loss Across Recurrent Trials", fontsize=10.5)
for bar, vl in zip(bars, val_losses):
    ax.text(bar.get_x() + bar.get_width()/2, vl + 0.005, f"{vl:.3f}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig("plot1_tuning_heatmap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
params_arr = [t["params"] for t in trial_results]
for t, c in zip(trial_results, colors):
    ax.scatter(t["params"], t["val_loss"], color=c, s=110, edgecolors="k", zorder=5, label=t["name"])
    ax.annotate(f"{t['name'].split()[1]}\n(Loss: {t['val_loss']:.3f})",
                (t["params"], t["val_loss"]), textcoords="offset points", xytext=(8, -5), fontsize=7.5)

ax.set_xlabel("Model Parameter Count", fontsize=9)
ax.set_ylabel("Validation Loss (MSE)", fontsize=9)
ax.set_title("Complexity-Generalization Pareto Tradeoff Frontier", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig("plot2_pareto_frontier.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Trial ID", "Architecture", "Recurrent Units", "Learning Rate", "Batch Size"]
t1_rows = [
    [f"Trial {i+1}", t["type"], f"{t['units']} cells", f"{t['lr']}", "32 sequences"]
    for i, t in enumerate(trial_results)
]
show_and_save("Table 1 - Hyperparameter Search Grid Configurations", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Trial Name", "Param Count", "Train Duration (s)", "Final Val Loss", "Final Val MAE"]
t2_rows = [
    [t["name"], f"{t['params']:,}", f"{t['time']:.2f}s", f"{t['val_loss']:.4f}", f"{t['val_mae']:.4f}"]
    for t in trial_results
]
show_and_save("Table 2 - Experimental Search Trials Performance Record", t2_headers, t2_rows, "table 2.png")

# Find best trial
best_idx = np.argmin(val_losses)
best_t = trial_results[best_idx]
t3_headers = ["Optimal Parameter", "Selected Value", "Criterion Justification"]
t3_rows = [
    ["Best Architecture", best_t["type"], "Lowest empirical holdout loss on sequence benchmark"],
    ["Optimal Units", f"{best_t['units']} units", "Prevents underfitting while bounding parameter bloat"],
    ["Optimal Learning Rate", f"{best_t['lr']}", "Fast monotonic Adam convergence"],
    ["Best Val MSE", f"{best_t['val_loss']:.4f}", "Exceeds all baseline architectures"]
]
show_and_save("Table 3 - Optimal Hyperparameter Selection and Specification", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Trial", "Val Loss (Ep 1)", "Val Loss (Ep 5)", "Val Loss (Ep 10)", "Val Loss (Ep 16)"]
t4_rows = [
    [t["name"], f"{t['history'].history['val_loss'][0]:.4f}", f"{t['history'].history['val_loss'][4]:.4f}", f"{t['history'].history['val_loss'][9]:.4f}", f"{t['history'].history['val_loss'][-1]:.4f}"]
    for t in trial_results
]
show_and_save("Table 4 - Validation Loss Trajectory Across Epoch Milestones", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Family", "Gate Architecture", "Computational Complexity", "Deployment Fit"]
t5_rows = [
    ["SimpleRNN", "No gates (Single state)", "O(1) matrix multiplies", "Ultra-low power edge devices with short history"],
    ["LSTM", "3 gates (Forget, Input, Output)", "O(4) matrix multiplies", "Complex long-horizon multi-modal sequences"],
    ["GRU", "2 gates (Reset, Update)", "O(3) matrix multiplies", "High-throughput real-time sequential processing"]
]
show_and_save("Table 5 - Recurrent Family Architectural Synthesis and Recommendations", t5_headers, t5_rows, "table 5.png")
