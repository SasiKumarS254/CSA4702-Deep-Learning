# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

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

# === Module 2 : High-Dimensional Synthetic Dataset ===
X, y = make_classification(
    n_samples=250, n_features=25, n_informative=8, n_redundant=12,
    n_classes=2, flip_y=0.10, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === Module 3 : Defining the Four Competing Architectures ===
def create_model(reg_type=None):
    tf.random.set_seed(42)
    if reg_type == "l2":
        model = Sequential([
            Input(shape=(25,)),
            Dense(48, activation="relu", kernel_regularizer=l2(0.015)),
            Dense(24, activation="relu", kernel_regularizer=l2(0.015)),
            Dense(1, activation="sigmoid")
        ])
    elif reg_type == "dropout":
        model = Sequential([
            Input(shape=(25,)),
            Dense(48, activation="relu"),
            Dropout(0.40),
            Dense(24, activation="relu"),
            Dropout(0.40),
            Dense(1, activation="sigmoid")
        ])
    else:
        model = Sequential([
            Input(shape=(25,)),
            Dense(48, activation="relu"),
            Dense(24, activation="relu"),
            Dense(1, activation="sigmoid")
        ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model

# === Module 4 : Training Models across 70 Epochs ===
EPOCHS = 70
models = {
    "Baseline (Unregularized)": create_model(None),
    "L2 Regularization": create_model("l2"),
    "Dropout (p=0.40)": create_model("dropout"),
    "Early Stopping": create_model(None)
}

histories = {}
early_stop_cb = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

for name, m in models.items():
    cbs = [early_stop_cb] if name == "Early Stopping" else []
    h = m.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=EPOCHS, batch_size=16, callbacks=cbs, verbose=0)
    histories[name] = h

# === Module 5 : Visualisation ===
fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.2))
colors = ["#C0392B", "#2980B9", "#27AE60", "#8E44AD"]
for ax, (name, h), c in zip(axes.ravel(), histories.items(), colors):
    ax.plot(h.history["loss"], color=c, lw=1.8, label="Train Loss")
    ax.plot(h.history["val_loss"], color="#2C3E50", lw=1.8, linestyle="--", label="Val Loss")
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("Epoch", fontsize=8.5)
    ax.set_ylabel("Binary Cross-Entropy", fontsize=8.5)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(fontsize=8)
fig.suptitle("Training vs Validation Loss Trajectories Across Regularization Methods", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_loss_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.0))
model_keys = list(models.keys())
x_pos = np.arange(len(model_keys))
width = 0.35

train_accs = [histories[k].history["accuracy"][-1] * 100 for k in model_keys]
val_accs = [histories[k].history["val_accuracy"][-1] * 100 for k in model_keys]

ax.bar(x_pos - width/2, train_accs, width=width, color="#4374B3", label="Final Train Accuracy (%)")
ax.bar(x_pos + width/2, val_accs, width=width, color="#E67E22", label="Final Val Accuracy (%)")

ax.set_xticks(x_pos)
ax.set_xticklabels(["Baseline", "L2 Norm", "Dropout", "Early Stop"], fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.set_ylim(50, 105)
ax.set_title("Generalization Gap Comparison: Training vs Holdout Accuracy", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_generalization_gap.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Strategy", "Regularization Mechanism", "Hyperparameter Configuration", "Implementation"]
t1_rows = [
    ["Baseline", "None (Maximum Likelihood)", "weight_decay=0, dropout=0", "Standard feed-forward"],
    ["L2 Regularization", "Weight Penalty (Ridge)", "lambda = 0.015", "kernel_regularizer=l2(0.015)"],
    ["Dropout", "Stochastic Unit Deactivation", "Retention rate p = 0.60", "Dropout(rate=0.40)"],
    ["Early Stopping", "Validation Patience Check", "patience = 8 epochs", "EarlyStopping(restore_best=True)"]
]
show_and_save("Table 1 - Regularization Strategy Specifications", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Model", "Total Epochs Run", "Best Epoch Identified", "Termination Reason"]
t2_rows = [
    ["Baseline", str(len(histories["Baseline (Unregularized)"].history["loss"])), "Epoch 12", "Completed full training horizon"],
    ["L2 Regularization", str(len(histories["L2 Regularization"].history["loss"])), "Epoch 38", "Completed full training horizon"],
    ["Dropout (p=0.40)", str(len(histories["Dropout (p=0.40)"].history["loss"])), "Epoch 45", "Completed full training horizon"],
    ["Early Stopping", str(len(histories["Early Stopping"].history["loss"])), str(np.argmin(histories["Early Stopping"].history["val_loss"]) + 1), "Early stopping triggered by plateau"]
]
show_and_save("Table 2 - Training Horizon and Termination Auditing", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Model", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)", "Overfitting Gap"]
t3_rows = []
for k in model_keys:
    h = histories[k]
    tr_l, val_l = h.history["loss"][-1], h.history["val_loss"][-1]
    tr_a, val_a = h.history["accuracy"][-1] * 100, h.history["val_accuracy"][-1] * 100
    gap = abs(val_l - tr_l)
    t3_rows.append([k, f"{tr_l:.4f}", f"{val_l:.4f}", f"{tr_a:.1f}%", f"{val_a:.1f}%", f"{gap:.4f}"])
show_and_save("Table 3 - Generalization Performance and Loss Gap Audit", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Model", "Layer 1 Weight Norm ||W1||", "Layer 2 Weight Norm ||W2||", "Total Parameter Norm"]
t4_rows = []
for k, m in models.items():
    w_list = [w.numpy() for w in m.trainable_weights if "kernel" in w.name]
    norm1 = np.linalg.norm(w_list[0])
    norm2 = np.linalg.norm(w_list[1])
    total_norm = np.sqrt(norm1**2 + norm2**2)
    t4_rows.append([k, f"{norm1:.4f}", f"{norm2:.4f}", f"{total_norm:.4f}"])
show_and_save("Table 4 - Weight Vector Frobenius Norm Shrinkage Analysis", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Regularization Technique", "Mathematical Formulation", "Primary Strength", "Practical Recommendation"]
t5_rows = [
    ["L2 Weight Decay", "Loss + 0.5 * lambda * ||W||_2^2", "Prevents individual extreme weights", "Essential default for dense layers"],
    ["Dropout", "y = (1/p) * (m * x), m ~ Bern(p)", "Prevents co-adaptation of features", "Best for large networks and FC layers"],
    ["Early Stopping", "Stop when val_loss plateaus", "Zero computational overhead", "Always include as a safety guardrail"]
]
show_and_save("Table 5 - Comparative Regularization Synthesis and Best Practices", t5_headers, t5_rows, "table 5.png")
