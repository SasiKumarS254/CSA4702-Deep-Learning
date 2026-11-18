# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import scipy.special as sp
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
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

# === Module 2 : Dataset Generation (3 Pattern Classes) ===
np.random.seed(42)
n_features = 20
N_per_class = 80

# 3 distinct morphological template patterns
t1 = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
t2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
t3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
templates = [t1, t2, t3]

X_all, y_all = [], []
for c_idx, t in enumerate(templates):
    for _ in range(N_per_class):
        sample = t.copy()
        flips = np.random.choice(n_features, size=2, replace=False)
        sample[flips] = 1 - sample[flips]
        X_all.append(sample)
        y_all.append(c_idx)

X_arr = np.array(X_all, dtype=np.float32)
y_arr = np.array(y_all, dtype=int)
y_cat = to_categorical(y_arr, 3)

# Shuffle
perm = np.random.permutation(len(X_arr))
X_arr, y_cat, y_arr = X_arr[perm], y_cat[perm], y_arr[perm]

# === Module 3 : Greedy Layer-Wise Pre-Training of Stacked RBMs ===
def train_rbm(data, n_in, n_out, epochs=25, lr=0.10):
    W = np.random.randn(n_in, n_out) * 0.1
    a = np.zeros(n_in)
    b = np.zeros(n_out)
    for _ in range(epochs):
        # Positive phase
        h_prob = sp.expit(np.dot(data, W) + b)
        h_sample = (h_prob > np.random.rand(*h_prob.shape)).astype(float)
        # Negative phase
        v_recon = sp.expit(np.dot(h_sample, W.T) + a)
        h_recon = sp.expit(np.dot(v_recon, W) + b)
        # CD-1 update
        W += lr * (np.dot(data.T, h_prob) - np.dot(v_recon.T, h_recon)) / len(data)
        a += lr * np.mean(data - v_recon, axis=0)
        b += lr * np.mean(h_prob - h_recon, axis=0)
    final_recon = sp.expit(np.dot(sp.expit(np.dot(data, W) + b), W.T) + a)
    loss = np.mean((data - final_recon)**2)
    return W, b, loss

# Layer 1: 20 -> 32
W1_rbm, b1_rbm, loss_rbm1 = train_rbm(X_arr, 20, 32)
H1_data = sp.expit(np.dot(X_arr, W1_rbm) + b1_rbm)

# Layer 2: 32 -> 16
W2_rbm, b2_rbm, loss_rbm2 = train_rbm(H1_data, 32, 16)
H2_data = sp.expit(np.dot(H1_data, W2_rbm) + b2_rbm)

# === Module 4 : Supervised Fine-Tuning: DBN vs Scratch Model ===
def create_classifier(pretrained=False):
    tf.random.set_seed(42)
    m = Sequential([
        Input(shape=(20,)),
        Dense(32, activation="sigmoid", name="dense_l1"),
        Dense(16, activation="sigmoid", name="dense_l2"),
        Dense(3, activation="softmax", name="dense_out")
    ])
    if pretrained:
        m.get_layer("dense_l1").set_weights([W1_rbm, b1_rbm])
        m.get_layer("dense_l2").set_weights([W2_rbm, b2_rbm])
    m.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m

model_dbn = create_classifier(pretrained=True)
model_scratch = create_classifier(pretrained=False)

EPOCHS = 25
h_dbn = model_dbn.fit(X_arr, y_cat, validation_split=0.25, epochs=EPOCHS, batch_size=16, verbose=0)
h_scratch = model_scratch.fit(X_arr, y_cat, validation_split=0.25, epochs=EPOCHS, batch_size=16, verbose=0)

# === Module 5 : Visualisation ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax1.plot(h_dbn.history["val_accuracy"], label="DBN Pre-Trained", color="#27AE60", lw=2.0)
ax1.plot(h_scratch.history["val_accuracy"], label="Random Init (Scratch)", color="#C0392B", lw=1.8, linestyle="--")
ax1.set_xlabel("Fine-Tuning Epoch", fontsize=8.5)
ax1.set_ylabel("Validation Accuracy", fontsize=8.5)
ax1.set_title("Validation Accuracy Trajectory", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="lower right", fontsize=8)

ax2.plot(h_dbn.history["val_loss"], label="DBN Pre-Trained", color="#27AE60", lw=2.0)
ax2.plot(h_scratch.history["val_loss"], label="Random Init (Scratch)", color="#C0392B", lw=1.8, linestyle="--")
ax2.set_xlabel("Fine-Tuning Epoch", fontsize=8.5)
ax2.set_ylabel("Categorical Cross-Entropy", fontsize=8.5)
ax2.set_title("Validation Loss Decay Curve", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(loc="upper right", fontsize=8)
fig.suptitle("Supervised Fine-Tuning: DBN Layer-Wise Pre-Training vs Scratch Baseline", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_pretraining_vs_scratch.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
pca1 = PCA(n_components=2).fit_transform(H1_data)
pca2 = PCA(n_components=2).fit_transform(H2_data)
colors = ["#2980B9", "#E67E22", "#27AE60"]

for c in range(3):
    ax1.scatter(pca1[y_arr == c, 0], pca1[y_arr == c, 1], color=colors[c], label=f"Class {c+1}", alpha=0.7, s=30)
ax1.set_title("Layer 1 Hidden Features (H1) Projection", fontsize=9.5)
ax1.set_xlabel("Principal Component 1", fontsize=8.5)
ax1.set_ylabel("Principal Component 2", fontsize=8.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

for c in range(3):
    ax2.scatter(pca2[y_arr == c, 0], pca2[y_arr == c, 1], color=colors[c], label=f"Class {c+1}", alpha=0.7, s=30)
ax2.set_title("Layer 2 Hidden Features (H2) Projection", fontsize=9.5)
ax2.set_xlabel("Principal Component 1", fontsize=8.5)
ax2.set_ylabel("Principal Component 2", fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Hierarchical Feature Disentanglement across DBN Layers", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_layerwise_features.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Stage", "Network Layer", "Input Dims", "Output Dims", "Weight Initialization Strategy"]
t1_rows = [
    ["RBM-1 Pre-Training", "Visible v -> Hidden h1", "20 units", "32 units", "Unsupervised CD-1 Reconstruction"],
    ["RBM-2 Pre-Training", "Hidden h1 -> Hidden h2", "32 units", "16 units", "Unsupervised CD-1 on h1 activations"],
    ["Fine-Tuning Head", "Hidden h2 -> Output y", "16 units", "3 classes", "Supervised Backprop (Categorical CE)"]
]
show_and_save("Table 1 - Deep Belief Network Hierarchical Pre-Training Hierarchy", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Pre-Trained Stage", "Input Features", "Latent Units", "Reconstruction MSE", "Fidelity"]
t2_rows = [
    ["RBM Layer 1", "20 input features", "32 latent units", f"{loss_rbm1:.4f}", "High Unsupervised Capture"],
    ["RBM Layer 2", "32 hidden features", "16 latent units", f"{loss_rbm2:.4f}", "Abstract Representation Distilled"]
]
show_and_save("Table 2 - Greedy Layer-Wise Pre-Training Reconstruction Losses", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Training Paradigm", "Initial Val Acc (Ep 1)", "Mid Val Acc (Ep 10)", "Final Val Acc (%)", "Final Val Loss"]
t3_rows = [
    ["DBN Pre-Trained", f"{h_dbn.history['val_accuracy'][0]*100:.1f}%", f"{h_dbn.history['val_accuracy'][9]*100:.1f}%", f"{h_dbn.history['val_accuracy'][-1]*100:.2f}%", f"{h_dbn.history['val_loss'][-1]:.4f}"],
    ["Random Init (Scratch)", f"{h_scratch.history['val_accuracy'][0]*100:.1f}%", f"{h_scratch.history['val_accuracy'][9]*100:.1f}%", f"{h_scratch.history['val_accuracy'][-1]*100:.2f}%", f"{h_scratch.history['val_loss'][-1]:.4f}"]
]
show_and_save("Table 3 - Fine-Tuning Convergence Comparison Across Epochs", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Layer Level", "Latent Dimensions", "Cluster Separability (Variance Ratio)", "Abstraction Role"]
t4_rows = [
    ["Input Data (v)", "20 raw dimensions", "Dispersed bit flips", "Low-level observation"],
    ["Layer 1 (h1)", "32 hidden dimensions", "Correlated feature groupings", "Edge/pattern part detectors"],
    ["Layer 2 (h2)", "16 hidden dimensions", "Distinct class cluster separation", "Invariant high-level class manifolds"]
]
show_and_save("Table 4 - Progressive Feature Disentanglement across DBN Hierarchy", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Aspect", "Historical DBN Paradigm (Hinton 2006)", "Modern Deep Learning Paradigm"]
t5_rows = [
    ["Weight Initialization", "Greedy unsupervised RBM pre-training", "He / Glorot initialization with Batch Normalization"],
    ["Activation Function", "Sigmoidal saturating activations", "Non-saturating activations (ReLU, GELU, Swish)"],
    ["Optimization Workhorse", "CD-1 pre-training + gradient descent", "End-to-end AdamW / Lion backpropagation"],
    ["Core Historical Impact", "Proved deep networks could be trained effectively", "Founded modern deep representation learning"]
]
show_and_save("Table 5 - Historical Significance and Paradigm Transition Analysis", t5_headers, t5_rows, "table 5.png")
