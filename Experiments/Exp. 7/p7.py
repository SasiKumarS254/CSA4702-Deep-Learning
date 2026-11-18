# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

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

# === Module 2 : Linearly Separable Clinical Triage Dataset ===
np.random.seed(47)
N_per_class = 50
# Class 0: Low Risk (High Oxygen Saturation, Normal Respiratory Rate)
mean_0 = [98.0, 16.0]
cov_0 = [[1.5, 0.2], [0.2, 1.2]]
X0 = np.random.multivariate_normal(mean_0, cov_0, N_per_class)
y0 = np.zeros(N_per_class, dtype=int)

# Class 1: High Risk (Low Oxygen Saturation, High Respiratory Rate)
mean_1 = [89.0, 26.0]
cov_1 = [[2.0, 0.4], [0.4, 2.2]]
X1 = np.random.multivariate_normal(mean_1, cov_1, N_per_class)
y1 = np.ones(N_per_class, dtype=int)

X = np.vstack((X0, X1))
y = np.concatenate((y0, y1))

# Feature scaling (Standardization)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_norm = (X - X_mean) / X_std

# === Module 3 : Perceptron Architecture and Forward Pass Functions ===
def step_act(z):
    return np.where(z >= 0, 1, 0)

def sigmoid_act(z):
    return 1.0 / (1.0 + np.exp(-z))

def relu_act(z):
    return np.maximum(0.0, z)

# === Module 4 : Training the Perceptron via Hebbian Learning ===
w = np.zeros(2)
b = 0.0
lr = 0.15
epochs = 15
epoch_errors = []

for ep in range(epochs):
    errors = 0
    for i in range(len(X_norm)):
        z = np.dot(w, X_norm[i]) + b
        y_hat = step_act(z)
        err = y[i] - y_hat
        if err != 0:
            w += lr * err * X_norm[i]
            b += lr * err
            errors += 1
    epoch_errors.append(errors)
    if errors == 0:
        break

# === Module 5 : Visualisation ===
fig, ax = plt.subplots(figsize=(6.8, 4.2))
ax.scatter(X_norm[y == 0, 0], X_norm[y == 0, 1], color="#2980B9", edgecolors="k", s=40, label="Low Risk (Class 0)")
ax.scatter(X_norm[y == 1, 0], X_norm[y == 1, 1], color="#E74C3C", edgecolors="k", s=40, marker="^", label="High Risk (Class 1)")

# Decision boundary line: w0*x0 + w1*x1 + b = 0 => x1 = (-w0*x0 - b) / w1
x0_vals = np.linspace(X_norm[:, 0].min() - 0.5, X_norm[:, 0].max() + 0.5, 100)
x1_vals = (-w[0] * x0_vals - b) / w[1]
ax.plot(x0_vals, x1_vals, color="#2C3E50", lw=2.2, label=f"Decision Line ($w^T x + b = 0$)")

ax.set_xlabel("Normalized Oxygen Saturation (SpO2)", fontsize=9)
ax.set_ylabel("Normalized Respiratory Rate", fontsize=9)
ax.set_title("Single-Neuron Perceptron Decision Boundary on Clinical Triage", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="lower left", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_decision_boundary.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
z_span = np.linspace(-5, 5, 200)
ax1.plot(z_span, step_act(z_span), color="#8E44AD", lw=1.8, label=r"Heaviside Step $I(z \geq 0)$")
ax1.plot(z_span, sigmoid_act(z_span), color="#27AE60", lw=2.0, label="Logistic Sigmoid $\\sigma(z)$")
ax1.plot(z_span, relu_act(z_span), color="#E67E22", lw=2.0, label="Rectified Linear (ReLU)")
ax1.set_ylim(-0.2, 4.0)
ax1.set_xlabel("Net Input Pre-Activation $z = w^T x + b$", fontsize=8.5)
ax1.set_ylabel("Neuron Activation Output $a$", fontsize=8.5)
ax1.set_title("Activation Function Profiles Comparison", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="upper left", fontsize=8)

ax2.plot(range(1, len(epoch_errors) + 1), epoch_errors, "o-", color="#C0392B", lw=1.8)
ax2.set_xlabel("Epoch Number", fontsize=8.5)
ax2.set_ylabel("Misclassification Count", fontsize=8.5)
ax2.set_title("Perceptron Convergence and Error Annihilation", fontsize=9.5)
ax2.set_xticks(range(1, len(epoch_errors) + 1))
ax2.grid(True, linestyle="--", alpha=0.45)
fig.suptitle("Activation Mathematics and Optimization Convergence", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_activations.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Parameter", "Description", "Value"]
t1_rows = [
    ["Total Samples", "N = N_class0 + N_class1", f"{len(X)} cases"],
    ["Class 0 (Low Risk)", "SpO2 Mean / Resp Rate Mean", "98.0% / 16.0 bpm"],
    ["Class 1 (High Risk)", "SpO2 Mean / Resp Rate Mean", "89.0% / 26.0 bpm"],
    ["Learning Rate", "eta", f"{lr}"],
    ["Linear Separability", "Convex Hull Separation", "Strictly Separable"]
]
show_and_save("Table 1 - Clinical Triage Dataset and Experimental Setup", t1_headers, t1_rows, "table 1.png")

sample_idx = [0, 1, 2, 50, 51, 52]
t2_headers = ["Case ID", "SpO2 (norm)", "Resp (norm)", "Net Input z", "Step y_hat", "Sigmoid a", "True Class"]
t2_rows = []
for i in sample_idx:
    z_i = np.dot(w, X_norm[i]) + b
    t2_rows.append([
        f"Case {i+1}",
        f"{X_norm[i, 0]:.3f}",
        f"{X_norm[i, 1]:.3f}",
        f"{z_i:.3f}",
        str(step_act(z_i)),
        f"{sigmoid_act(z_i):.4f}",
        str(y[i])
    ])
show_and_save("Table 2 - Forward Propagation Inference on Representative Samples", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Epoch", "Sample Updates", "Misclassification Count", "Training Accuracy (%)"]
t3_rows = [
    [f"Epoch {ep+1}", str(len(X)), str(epoch_errors[ep]), f"{(len(X) - epoch_errors[ep]) / len(X) * 100:.1f}%"]
    for ep in range(len(epoch_errors))
]
show_and_save("Table 3 - Training Epoch Progression and Convergence Record", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Parameter", "Learned Weight", "Effective Physical Direction", "Significance"]
t4_rows = [
    ["Weight w1 (SpO2)", f"{w[0]:.4f}", "Negative (-)", "Lower SpO2 increases risk (pushes z towards positive)"],
    ["Weight w2 (Resp Rate)", f"{w[1]:.4f}", "Positive (+)", "Higher respiration increases risk"],
    ["Bias b", f"{b:.4f}", "Threshold Shift", "Centers separating hyperplane in normalized domain"]
]
show_and_save("Table 4 - Optimal Perceptron Synaptic Weights and Bias", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Activation Function", "Mathematical Formula", "Output Range", "Differentiable", "Deep Learning Utility"]
t5_rows = [
    ["Heaviside Step", "I(z >= 0)", "{0, 1}", "No (Zero gradient)", "Discrete classification only"],
    ["Logistic Sigmoid", "1 / (1 + e^-z)", "(0, 1)", "Yes: s*(1-s)", "Binary output probabilities"],
    ["Rectified Linear (ReLU)", "max(0, z)", "[0, inf)", "Yes: I(z > 0)", "Hidden layer standard (prevents vanishing gradient)"]
]
show_and_save("Table 5 - Activation Functions Mathematical Specification", t5_headers, t5_rows, "table 5.png")
