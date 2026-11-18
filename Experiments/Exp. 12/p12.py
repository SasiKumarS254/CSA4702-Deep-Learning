# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import sys
import time
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical

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

# === Module 2 : Dataset Preparation ===
digits = load_digits()
X = digits.images[..., np.newaxis] / 16.0
y = digits.target
num_classes = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# === Module 3 : Defining the Three Competing CNN Architectures ===
def build_shallow():
    return Sequential([
        Input(shape=(8, 8, 1)),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(10, activation="softmax")
    ])

def build_deep():
    return Sequential([
        Input(shape=(8, 8, 1)),
        Conv2D(16, (3, 3), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), padding="same", activation="relu"),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(10, activation="softmax")
    ])

def build_wide():
    return Sequential([
        Input(shape=(8, 8, 1)),
        Conv2D(32, (5, 5), padding="same", activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), padding="same", activation="relu"),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(10, activation="softmax")
    ])

model_dict = {
    "Shallow CNN (1 Conv 3x3)": build_shallow(),
    "Deep CNN (2 Conv 3x3)": build_deep(),
    "Wide CNN (Conv 5x5 + 3x3)": build_wide()
}

# === Module 4 : Training and Benchmarking ===
EPOCHS = 15
histories = {}
train_times = {}
param_counts = {}

for name, model in model_dict.items():
    tf.random.set_seed(42)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    param_counts[name] = model.count_params()
    
    t0 = time.time()
    h = model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat),
                  epochs=EPOCHS, batch_size=32, verbose=0)
    train_times[name] = time.time() - t0
    histories[name] = h

# === Module 5 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.6, 4.0))
colors = ["#C0392B", "#27AE60", "#2980B9"]
styles = ["-", "--", "-."]
for (name, h), c, s in zip(histories.items(), colors, styles):
    ax.plot(range(1, EPOCHS + 1), [v * 100 for v in h.history["val_accuracy"]],
            label=f"{name}", color=c, linestyle=s, lw=2.0)

ax.set_xlabel("Epoch", fontsize=9)
ax.set_ylabel("Validation Accuracy (%)", fontsize=9)
ax.set_title("Validation Accuracy Trajectories Across CNN Architectures", fontsize=10.5)
ax.set_xticks(range(1, EPOCHS + 1, 2))
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_accuracy_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 3.8))
names = list(model_dict.keys())
accs = [histories[n].history["val_accuracy"][-1] * 100 for n in names]
params = [param_counts[n] for n in names]

ax.scatter(params, accs, s=120, color="#8E44AD", edgecolors="k", zorder=5)
for i, txt in enumerate(["Shallow", "Deep", "Wide"]):
    ax.annotate(f"{txt} ({params[i]:,} params)\nAcc: {accs[i]:.2f}%",
                (params[i], accs[i]), textcoords="offset points", xytext=(-20, 10),
                fontsize=8, fontweight="bold")

ax.set_xlabel("Trainable Parameter Count", fontsize=9)
ax.set_ylabel("Final Validation Accuracy (%)", fontsize=9)
ax.set_title("Model Complexity vs Generalization Performance", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig("plot2_parameters_tradeoff.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Architecture", "Conv Stages", "Filter Counts", "Kernel Dimensions", "Dense Size"]
t1_rows = [
    ["Shallow CNN", "1 Conv layer", "16 filters", "3 x 3", "32 units"],
    ["Deep CNN", "2 Conv layers", "16 -> 32 filters", "3 x 3", "32 units"],
    ["Wide CNN", "2 Conv layers", "32 -> 64 filters", "5 x 5 -> 3 x 3", "64 units"]
]
show_and_save("Table 1 - CNN Architectural Specifications and Depth Profiles", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Architecture", "Trainable Params", "Training Time (s)", "Throughput (samples/s)"]
t2_rows = [
    [name, f"{param_counts[name]:,}", f"{train_times[name]:.2f}s", f"{len(X_train)*EPOCHS/train_times[name]:.1f}"]
    for name in names
]
show_and_save("Table 2 - Computational Footprint and Training Latency", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Architecture", "Val Acc (Ep 1)", "Val Acc (Ep 5)", "Val Acc (Ep 10)", "Final Val Acc (%)"]
t3_rows = [
    [name, f"{histories[name].history['val_accuracy'][0]*100:.1f}%", f"{histories[name].history['val_accuracy'][4]*100:.1f}%", f"{histories[name].history['val_accuracy'][9]*100:.1f}%", f"{histories[name].history['val_accuracy'][-1]*100:.2f}%"]
    for name in names
]
show_and_save("Table 3 - Validation Accuracy Evolution Across Epoch Milestones", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Architecture", "Train Loss", "Val Loss", "Train Acc (%)", "Val Acc (%)", "Generalization Gap"]
t4_rows = []
for name in names:
    h = histories[name]
    tr_l, val_l = h.history["loss"][-1], h.history["val_loss"][-1]
    tr_a, val_a = h.history["accuracy"][-1] * 100, h.history["val_accuracy"][-1] * 100
    gap = abs(val_l - tr_l)
    t4_rows.append([name, f"{tr_l:.4f}", f"{val_l:.4f}", f"{tr_a:.1f}%", f"{val_a:.1f}%", f"{gap:.4f}"])
show_and_save("Table 4 - Terminal Convergence and Generalization Gap Audit", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Architecture", "Receptive Field Size", "Spatial Feature Abstraction", "Recommendation"]
t5_rows = [
    ["Shallow CNN", "3 x 3 pixels", "Captures low-level edges/corners", "Suitable for tiny low-res sensors"],
    ["Deep CNN", "7 x 7 pixels", "Hierarchical part composition", "Optimal balance of accuracy and size"],
    ["Wide CNN", "11 x 11 pixels", "Broad contextual receptive field", "Best for complex high-variance visual patterns"]
]
show_and_save("Table 5 - Receptive Field and Architectural Recommendation Synthesis", t5_headers, t5_rows, "table 5.png")
