# === Module 1 : Importing the Required Packages ===
import numpy as np
import scipy.special as sp
import sys
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

# === Module 2 : Binary Geometric Patterns Dataset (4x4 Grid) ===
np.random.seed(42)
n_vis = 16 # 4x4 binary pixels
n_hid = 8  # 8 hidden latent features

# Define canonical 4x4 visual patterns
p_horiz1 = np.array([
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]).flatten()

p_horiz2 = np.array([
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]).flatten()

p_vert1 = np.array([
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0],
    [1, 0, 0, 0]
]).flatten()

p_cross = np.array([
    [0, 1, 0, 0],
    [1, 1, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 0]
]).flatten()

clean_patterns = [p_horiz1, p_horiz2, p_vert1, p_cross]
pattern_names = ["Horizontal Bar 1", "Horizontal Bar 2", "Vertical Bar 1", "Cross Pattern"]

# Generate training dataset with bit-flip noise
V_data = []
for _ in range(30):
    for p in clean_patterns:
        noisy_p = p.copy()
        flip_idx = np.random.choice(n_vis, size=1, replace=False)
        noisy_p[flip_idx] = 1 - noisy_p[flip_idx]
        V_data.append(noisy_p)
V_data = np.array(V_data) # (120, 16)

# === Module 3 : RBM Energy and Free Energy Mathematical Functions ===
def sigmoid(z):
    return sp.expit(z)

def free_energy(v, W, a, b):
    # F(v) = - a^T v - sum_j softplus(v^T W_j + b_j)
    v_bias_term = np.dot(v, a)
    x = np.dot(v, W) + b
    hidden_term = np.sum(np.log1p(np.exp(np.clip(x, -25, 25))), axis=-1)
    return -v_bias_term - hidden_term

def joint_energy(v, h, W, a, b):
    # E(v, h) = - v^T W h - a^T v - b^T h
    return -np.dot(v, W @ h) - np.dot(a, v) - np.dot(b, h)

# === Module 4 : Training RBM via Contrastive Divergence (CD-1) ===
W = np.random.randn(n_vis, n_hid) * 0.1
a = np.zeros(n_vis)
b = np.zeros(n_hid)
lr = 0.08
epochs = 40

for ep in range(epochs):
    # Positive phase
    h_prob = sigmoid(np.dot(V_data, W) + b)
    h_sample = (h_prob > np.random.rand(*h_prob.shape)).astype(float)
    pos_grad = np.dot(V_data.T, h_prob)
    
    # Negative phase (1 step Gibbs)
    v_neg_prob = sigmoid(np.dot(h_sample, W.T) + a)
    v_neg_sample = (v_neg_prob > np.random.rand(*v_neg_prob.shape)).astype(float)
    h_neg_prob = sigmoid(np.dot(v_neg_sample, W) + b)
    neg_grad = np.dot(v_neg_sample.T, h_neg_prob)
    
    # Weight updates
    N_b = len(V_data)
    W += lr * (pos_grad - neg_grad) / N_b
    a += lr * np.mean(V_data - v_neg_sample, axis=0)
    b += lr * np.mean(h_prob - h_neg_prob, axis=0)

# Reconstructions of the 4 canonical test patterns
reconstructed_probs = []
recon_errors = []
for p in clean_patterns:
    h_p = sigmoid(np.dot(p, W) + b)
    v_recon = sigmoid(np.dot(h_p, W.T) + a)
    reconstructed_probs.append(v_recon)
    recon_errors.append(np.mean((p - v_recon)**2))

# Free Energy of structured patterns vs random noise vectors
structured_FE = [free_energy(p, W, a, b) for p in clean_patterns]
random_noise_vecs = np.random.binomial(1, 0.5, size=(10, n_vis))
noise_FE = [free_energy(rv, W, a, b) for rv in random_noise_vecs]

# === Module 5 : Visualisation ===
fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.4))
for i in range(4):
    axes[0, i].imshow(clean_patterns[i].reshape(4, 4), cmap="binary", vmin=0, vmax=1)
    axes[0, i].set_title(f"Input: {pattern_names[i].split()[0]}", fontsize=8.5)
    axes[0, i].axis("off")
    
    axes[1, i].imshow(reconstructed_probs[i].reshape(4, 4), cmap="Blues", vmin=0, vmax=1)
    axes[1, i].set_title(f"Reconstructed P(v)", fontsize=8.5)
    axes[1, i].axis("off")
axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
axes[1, 0].set_ylabel("RBM Reconstruction", fontsize=9)
fig.suptitle("Restricted Boltzmann Machine Visual Pattern Reconstruction (4x4 Grid)", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_pattern_reconstruction.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
x_scat = np.concatenate([np.zeros(len(structured_FE)), np.ones(len(noise_FE))])
y_scat = np.concatenate([structured_FE, noise_FE])
ax.scatter(np.zeros(len(structured_FE)), structured_FE, color="#27AE60", s=90, edgecolors="k", zorder=5, label="Structured Trained Patterns")
ax.scatter(np.ones(len(noise_FE)), noise_FE, color="#C0392B", s=90, edgecolors="k", marker="s", zorder=5, label="Random Noise Configurations")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Structured Patterns", "Random Noise Vectors"], fontsize=9)
ax.set_ylabel("Free Energy F(v)", fontsize=9)
ax.set_title("Energy Gap: Low Energy (Coherent) vs High Energy (Noise)", fontsize=10.5)
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(loc="upper left", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_free_energy.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Layer", "Unit Type", "Count", "Bias Vector Shape", "Activation Probability"]
t1_rows = [
    ["Visible Layer (v)", "Binary Bernoulli", f"{n_vis} units (4x4)", f"({n_vis},) visible bias a", "P(v_i=1|h) = sigma(W_{i.} h + a_i)"],
    ["Hidden Layer (h)", "Binary Bernoulli", f"{n_hid} units", f"({n_hid},) hidden bias b", "P(h_j=1|v) = sigma(v^T W_{.j} + b_j)"]
]
show_and_save("Table 1 - Restricted Boltzmann Machine Layer Topology", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Pattern Name", "Active Pixels", "Free Energy F(v)", "Reconstruction MSE", "Fidelity"]
t2_rows = [
    [pattern_names[i], f"{int(np.sum(clean_patterns[i]))} / 16", f"{structured_FE[i]:.4f}", f"{recon_errors[i]:.4f}", "High Reconstruction" if recon_errors[i] < 0.10 else "Moderate"]
    for i in range(4)
]
show_and_save("Table 2 - Canonical Pattern Energy and Reconstruction Metrics", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Configuration Set", "Sample Size", "Mean Free Energy", "Min Energy", "Energy Differential vs Structured"]
mean_s_fe = np.mean(structured_FE)
mean_n_fe = np.mean(noise_FE)
t3_rows = [
    ["Structured Patterns", "4 canonical templates", f"{mean_s_fe:.4f}", f"{np.min(structured_FE):.4f}", "Baseline (Lowest Energy)"],
    ["Random Noise Vectors", "10 Bernoulli(0.5) vectors", f"{mean_n_fe:.4f}", f"{np.min(noise_FE):.4f}", f"{mean_n_fe - mean_s_fe:+.4f} (Higher Energy)"]
]
show_and_save("Table 3 - Energy Landscape Separation: Structured vs Noise", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Visible Unit", "Visible Bias a_i", "Dominant Connected Hidden Unit", "Weight Magnitude"]
t4_rows = [
    [f"Pixel {i+1}", f"{a[i]:.3f}", f"Hidden h{np.argmax(np.abs(W[i]))+1}", f"{np.max(np.abs(W[i])):.3f}"]
    for i in range(8)
]
show_and_save("Table 4 - Visible Bias and Synaptic Weight Linkages (First 8 Units)", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Property", "Energy-Based Models (RBM)", "Standard Autoencoders"]
t5_rows = [
    ["Probability Density", "Defines unnormalized Boltzmann density P(v) ~ e^-F(v)", "Deterministic deterministic map; no density"],
    ["Connections", "Bipartite undirected connections (symmetric W)", "Directed feed-forward encoder/decoder"],
    ["Generative Sampling", "MCMC Gibbs sampling from stationary distribution", "Requires external sampling mechanism"],
    ["Training Objective", "Maximum Likelihood via Contrastive Divergence", "Reconstruction Mean Squared Error minimization"]
]
show_and_save("Table 5 - Restricted Boltzmann Machine Theoretical Principles", t5_headers, t5_rows, "table 5.png")
