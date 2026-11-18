# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
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

# === Module 2 : Exploding Gradient Sequence Generation ===
np.random.seed(42)
N = 300
T_seq = 25
X_data = np.random.randn(N, T_seq, 1) * 1.5
# Target has high amplitude shock points causing sharp gradients
y_data = np.sum(X_data[:, :5, 0], axis=1) * 2.8 + np.random.normal(0, 0.5, N)

dataset = tf.data.Dataset.from_tensor_slices((X_data.astype(np.float32), y_data.astype(np.float32)[:, np.newaxis])).batch(32)

# === Module 3 : Defining Model and Custom Training with Gradient Tracing ===
def build_deep_rnn():
    return Sequential([
        Input(shape=(T_seq, 1)),
        SimpleRNN(32, return_sequences=True, activation="tanh"),
        SimpleRNN(32, activation="tanh"),
        Dense(1)
    ])

# Training function tracking per-batch gradient norms
def train_with_or_without_clipping(use_clipping, clip_norm=1.2):
    tf.random.set_seed(42)
    model = build_deep_rnn()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    batch_grad_norms = []
    epoch_losses = []
    
    for epoch in range(15):
        epoch_loss_accum = 0.0
        n_batches = 0
        for x_b, y_b in dataset:
            with tf.GradientTape() as tape:
                preds = model(x_b, training=True)
                loss_val = loss_fn(y_b, preds)
            
            grads = tape.gradient(loss_val, model.trainable_variables)
            # Calculate global norm before clipping
            g_norm = tf.linalg.global_norm(grads).numpy()
            batch_grad_norms.append(g_norm)
            
            if use_clipping:
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss_accum += loss_val.numpy()
            n_batches += 1
            
        epoch_losses.append(epoch_loss_accum / n_batches)
        
    return np.array(batch_grad_norms), np.array(epoch_losses)

grad_unclipped, loss_unclipped = train_with_or_without_clipping(use_clipping=False)
grad_clipped, loss_clipped = train_with_or_without_clipping(use_clipping=True, clip_norm=1.2)

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.8, 4.0))
steps = np.arange(len(grad_unclipped))
ax.plot(steps, grad_unclipped, color="#E74C3C", lw=1.5, alpha=0.85, label="Raw Gradient Norm (Unclipped)")
ax.plot(steps, np.minimum(grad_clipped, 1.2), color="#27AE60", lw=2.0, label="Clipped Gradient Norm (threshold = 1.2)")
ax.axhline(1.2, color="#2C3E50", linestyle="--", lw=1.4, label="Clipping Ceiling (c=1.2)")
ax.set_yscale("log")
ax.set_xlabel("Batch Optimization Step", fontsize=9)
ax.set_ylabel("Global Gradient Norm (Log Scale)", fontsize=9)
ax.set_title("Recurrent Gradient Norm Dynamics: Spikes vs Bounded Stability", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_gradient_norms.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(range(1, 16), loss_unclipped, "o-", color="#E74C3C", lw=1.8, label="Unclipped Training Loss")
ax.plot(range(1, 16), loss_clipped, "s-", color="#2980B9", lw=2.0, label="Clipped (clipnorm=1.2) Training Loss")
ax.set_yscale("log")
ax.set_xlabel("Training Epoch", fontsize=9)
ax.set_ylabel("Mean Squared Error (Log Scale)", fontsize=9)
ax.set_title("Training Loss Trajectory: Numerical Explosion vs Smooth Convergence", fontsize=10.5)
ax.set_xticks(range(1, 16, 2))
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_loss_stability.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Clipping Method", "Mathematical Formulation", "Direction Preserved", "Use Case"]
t1_rows = [
    ["Value Clipping", "g_i = clip(g_i, -c, c)", "No (alters update angle)", "Element-wise bounds"],
    ["Global Norm Clipping", "g = g * min(1, c / ||g||)", "Yes (pure magnitude scaling)", "Standard for RNNs, LSTMs, Transformers"],
    ["Adaptive Clipping", "c_t = percentile(||g_{t-k:t}||, 90)", "Yes (dynamic ceiling)", "Non-stationary distributed training"]
]
show_and_save("Table 1 - Gradient Clipping Algorithmic Formulations", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Condition", "Peak Grad Norm", "Mean Grad Norm", "Std Dev of Norm", "Spike Count (> 5.0)"]
spikes_uncl = np.sum(grad_unclipped > 5.0)
spikes_cl = np.sum(grad_clipped > 1.2001)
t2_rows = [
    ["Unclipped Training", f"{np.max(grad_unclipped):.2f}", f"{np.mean(grad_unclipped):.3f}", f"{np.std(grad_unclipped):.3f}", str(spikes_uncl)],
    ["Clipped Training (c=1.2)", f"{1.2000:.2f}", f"{np.mean(np.minimum(grad_clipped, 1.2)):.3f}", f"{np.std(np.minimum(grad_clipped, 1.2)):.3f}", f"0 (Bounded at 1.20)"]
]
show_and_save("Table 2 - Gradient Norm Statistics and Extreme Spike Audit", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Epoch", "Unclipped Loss", "Clipped Loss", "Stability Delta (Uncl - Cl)"]
milestones = [1, 3, 5, 8, 12, 15]
t3_rows = [
    [f"Epoch {ep:02d}", f"{loss_unclipped[ep-1]:.4f}", f"{loss_clipped[ep-1]:.4f}", f"{loss_unclipped[ep-1] - loss_clipped[ep-1]:+.4f}"]
    for ep in milestones
]
show_and_save("Table 3 - Epoch-by-Epoch Loss Stability Comparison", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Optimization Phase", "Unclipped Step Behavior", "Clipped Step Behavior", "Safety Impact"]
t4_rows = [
    ["Early Epochs (1-3)", "Catastrophic parameter kicks", "Smooth bounded steps", "Prevents model destabilization"],
    ["Mid Training (4-10)", "Oscillatory loss jitter", "Monotonic smooth descent", "Accelerates valley transit"],
    ["Late Convergence", "Periodic instability spikes", "Asymptotic stable minimum", "Prevents divergence from optimum"]
]
show_and_save("Table 4 - Parameter Trajectory Mechanics and Stability Profile", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Hyperparameter", "Recommended Default", "Impact of Setting Too High", "Impact of Setting Too Low"]
t5_rows = [
    ["clipnorm", "1.0 to 5.0", "Fails to prevent numerical explosion", "Artificially slows learning rate"],
    ["clipvalue", "0.5 to 1.0", "Distorts gradient direction vector", "Disrupts coordinate relationships"],
    ["Learning Rate Interaction", "Pairs with Adam/RMSprop", "High LR with no clipping causes NaN", "Low LR with clipping causes stall"]
]
show_and_save("Table 5 - Gradient Clipping Hyperparameter Engineering Guide", t5_headers, t5_rows, "table 5.png")
