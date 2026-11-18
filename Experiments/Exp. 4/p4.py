# === Module 1 : Importing the Required Packages ===
import numpy as np
import scipy.stats as stats
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

# === Module 2 : Clinical Diagnostic Logits and Targets ===
categories = ["Normal", "Benign", "Malignant", "Atypical"]
num_classes = len(categories)

# 6 representative test specimens with raw network logits
logits = np.array([
    [ 3.8,  0.4, -1.2, -2.1], # Sample 1: high confidence Normal (True: Normal)
    [ 0.1,  3.2,  0.5, -1.0], # Sample 2: high confidence Benign (True: Benign)
    [-1.5,  0.2,  4.1,  0.8], # Sample 3: high confidence Malignant (True: Malignant)
    [ 1.8,  1.6, -0.4,  0.5], # Sample 4: ambiguous Normal vs Benign (True: Normal)
    [ 2.2,  0.1, -0.8, -1.1], # Sample 5: confident Normal but True is Malignant (Misclassified)
    [ 0.5,  0.7,  0.6,  0.4], # Sample 6: high uncertainty across all classes (True: Atypical)
])
true_labels = np.array([0, 1, 2, 0, 2, 3])
one_hot_targets = np.eye(num_classes)[true_labels]

# === Module 3 : Softmax Conversion with Numerical Stability ===
exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# === Module 4 : Information Theory Metrics ===
eps = 1e-12
shannon_entropy = -np.sum(softmax_probs * np.log2(softmax_probs + eps), axis=1) # bits
cross_entropy = -np.sum(one_hot_targets * np.log(softmax_probs + eps), axis=1)  # nats
kl_divergence = np.sum(one_hot_targets * np.log((one_hot_targets + eps) / (softmax_probs + eps)), axis=1)

predicted_labels = np.argmax(softmax_probs, axis=1)
accuracy_arr = (predicted_labels == true_labels).astype(int)

# === Module 5 : Soft Label Distillation Comparison ===
soft_targets = np.array([
    [0.85, 0.10, 0.03, 0.02],
    [0.05, 0.85, 0.07, 0.03],
    [0.02, 0.04, 0.90, 0.04],
    [0.55, 0.35, 0.05, 0.05],
    [0.05, 0.10, 0.80, 0.05],
    [0.25, 0.25, 0.25, 0.25]
])
H_soft = -np.sum(soft_targets * np.log(soft_targets + eps), axis=1)
H_cross_soft = -np.sum(soft_targets * np.log(softmax_probs + eps), axis=1)
KL_soft = np.sum(soft_targets * np.log((soft_targets + eps) / (softmax_probs + eps)), axis=1)
identity_soft = np.allclose(H_cross_soft, H_soft + KL_soft)

# === Module 6 : Visualisation ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
sample_ids = [f"S{i+1}" for i in range(len(logits))]
x = np.arange(len(sample_ids))
w = 0.36
ax1.bar(x - w/2, cross_entropy, width=w, color="#D9534F", label="Cross-Entropy (nats)")
ax1.bar(x + w/2, shannon_entropy, width=w, color="#337AB7", label="Shannon Entropy (bits)")
ax1.set_xticks(x); ax1.set_xticklabels(sample_ids, fontsize=8.5)
ax1.set_xlabel("Specimen Sample ID", fontsize=9)
ax1.set_ylabel("Loss / Uncertainty Metric", fontsize=9)
ax1.set_title("Cross-Entropy Loss and Prediction Entropy per Sample", fontsize=9.5)
ax1.grid(axis="y", linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

p_vals = np.linspace(0.01, 1.0, 100)
ce_loss_curve = -np.log(p_vals)
ax2.plot(p_vals, ce_loss_curve, color="#C0392B", lw=2.0, label=r"$L_{CE} = -\ln(p)$")
ax2.scatter(softmax_probs[np.arange(len(logits)), true_labels], cross_entropy, color="#2980B9", s=45, zorder=5, label="Batch Samples")
ax2.set_xlabel("Predicted Target Probability $p$", fontsize=9)
ax2.set_ylabel("Cross-Entropy Loss (nats)", fontsize=9)
ax2.set_title("Loss Trajectory as a Function of Target Confidence", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Information-Theoretic Loss Dynamics in Pathology Classification", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_per_image_loss.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.6))
p_binary = np.linspace(0.001, 0.999, 200)
binary_entropy = -(p_binary * np.log2(p_binary) + (1.0 - p_binary) * np.log2(1.0 - p_binary))
ax.plot(p_binary, binary_entropy, color="#16A085", lw=2.0, label="Shannon Entropy $H(p)$")
ax.axvline(0.5, color="#7F8C8D", linestyle=":", lw=1.2, label="Max Uncertainty ($p=0.5, H=1$)")
ax.set_xlabel("Bernoulli Probability Parameter $p$", fontsize=9)
ax.set_ylabel("Entropy $H(p)$ (bits)", fontsize=9)
ax.set_title("Information Content and Uncertainty Profile for Binary Event", fontsize=9.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot2_loss_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 7 : Printing and Saving Tables ===
t1_headers = ["Sample", "Normal", "Benign", "Malignant", "Atypical", "True Class", "Predicted"]
t1_rows = [
    [f"S-{i+1}", f"{softmax_probs[i, 0]:.4f}", f"{softmax_probs[i, 1]:.4f}", f"{softmax_probs[i, 2]:.4f}", f"{softmax_probs[i, 3]:.4f}", categories[true_labels[i]], categories[predicted_labels[i]]]
    for i in range(len(logits))
]
show_and_save("Table 1 - Softmax Probabilities and Category Classification", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Sample", "Target Prob P(y)", "Cross-Entropy (nats)", "KL Divergence (nats)", "Shannon Entropy (bits)", "Accuracy"]
t2_rows = [
    [f"S-{i+1}", f"{softmax_probs[i, true_labels[i]]:.4f}", f"{cross_entropy[i]:.4f}", f"{kl_divergence[i]:.4f}", f"{shannon_entropy[i]:.4f}", "Correct" if accuracy_arr[i] else "Misclassified"]
    for i in range(len(logits))
]
show_and_save("Table 2 - Information Loss and Divergence Metrics per Sample", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Sample", "Soft Target Entropy H(Q)", "Cross-Entropy H(Q,P)", "KL Divergence D_KL(Q||P)", "H(Q) + D_KL"]
t3_rows = [
    [f"S-{i+1}", f"{H_soft[i]:.4f}", f"{H_cross_soft[i]:.4f}", f"{KL_soft[i]:.4f}", f"{(H_soft[i] + KL_soft[i]):.4f}"]
    for i in range(len(logits))
]
show_and_save("Table 3 - Soft Target Distribution and Distillation Cross-Entropy", t3_headers, t3_rows, "table 3.png")

t4_headers = ["S.No", "Information-Theoretic Theorem", "Formula Verification", "Result"]
t4_rows = [
    ["1", "One-Hot Cross-Entropy Equivalence", "H(Y, P) = D_KL(Y || P) since H(Y) = 0", "Verified" if np.allclose(cross_entropy, kl_divergence) else "Failed"],
    ["2", "Soft Distribution Additivity", "H(Q, P) = H(Q) + D_KL(Q || P)", "Verified" if identity_soft else "Failed"],
    ["3", "Gibbs' Inequality Invariance", "D_KL(Q || P) >= 0 for all samples", "Verified" if np.all(KL_soft >= -1e-12) else "Failed"],
    ["4", "Maximum Entropy Principle", "H_max = log2(4) = 2.0 bits", "Verified" if np.all(shannon_entropy <= 2.0 + 1e-12) else "Failed"]
]
show_and_save("Table 4 - Theoretical Theorems and Invariance Validations", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Batch Size", "Mean Cross-Entropy", "Mean KL Divergence", "Mean Entropy", "Batch Accuracy"]
t5_rows = [[str(len(logits)), f"{np.mean(cross_entropy):.4f} nats", f"{np.mean(kl_divergence):.4f} nats", f"{np.mean(shannon_entropy):.4f} bits", f"{np.mean(accuracy_arr)*100:.1f}%"]]
show_and_save("Table 5 - Aggregated Batch Performance and Uncertainty Metrics", t5_headers, t5_rows, "table 5.png")
