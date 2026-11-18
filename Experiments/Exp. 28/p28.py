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

# === Module 2 : Deep Boltzmann Machine (DBM) Architecture Definition ===
np.random.seed(42)
n_v = 12   # Visible units
n_h1 = 8   # First hidden layer
n_h2 = 4   # Second hidden layer

# Synaptic weight matrices
W1 = np.random.randn(n_v, n_h1) * 0.25 # Between v and h1
W2 = np.random.randn(n_h1, n_h2) * 0.30 # Between h1 and h2

# Bias vectors
b_v = np.zeros(n_v) - 0.2
b_h1 = np.zeros(n_h1)
b_h2 = np.zeros(n_h2)

def sigmoid(z):
    return sp.expit(z)

def joint_energy_dbm(v, h1, h2):
    # E(v, h1, h2) = - v^T W1 h1 - h1^T W2 h2 - b_v^T v - b_h1^T h1 - b_h2^T h2
    term1 = -np.dot(v, W1 @ h1)
    term2 = -np.dot(h1, W2 @ h2)
    term3 = -np.dot(b_v, v) - np.dot(b_h1, h1) - np.dot(b_h2, h2)
    return term1 + term2 + term3

# === Module 3 : Variational Mean-Field Inference Simulation ===
# Fixed-point equations for clamped visible vector v:
# mu1 = sigma(v W1 + mu2 W2^T + b_h1)
# mu2 = sigma(mu1 W2 + b_h2)

v_test = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=float)

max_mf_iters = 12
mu1_history = []
mu2_history = []
delta_history = []

mu1 = np.ones(n_h1) * 0.5
mu2 = np.ones(n_h2) * 0.5

for it in range(max_mf_iters):
    mu1_prev, mu2_prev = mu1.copy(), mu2.copy()
    
    # Synchronous or damped update
    mu1 = sigmoid(np.dot(v_test, W1) + np.dot(mu2, W2.T) + b_h1)
    mu2 = sigmoid(np.dot(mu1, W2) + b_h2)
    
    delta = np.linalg.norm(mu1 - mu1_prev) + np.linalg.norm(mu2 - mu2_prev)
    mu1_history.append(mu1.copy())
    mu2_history.append(mu2.copy())
    delta_history.append(delta)

# Joint Energy Sampling: Clamped configurations vs Random configurations
N_sim = 80
clamped_energies = []
random_energies = []

for _ in range(N_sim):
    # Clamped under mean-field
    h1_sample = (mu1 > np.random.rand(n_h1)).astype(float)
    h2_sample = (mu2 > np.random.rand(n_h2)).astype(float)
    clamped_energies.append(joint_energy_dbm(v_test, h1_sample, h2_sample))
    
    # Completely random fantasy states
    v_rnd = np.random.binomial(1, 0.5, n_v).astype(float)
    h1_rnd = np.random.binomial(1, 0.5, n_h1).astype(float)
    h2_rnd = np.random.binomial(1, 0.5, n_h2).astype(float)
    random_energies.append(joint_energy_dbm(v_rnd, h1_rnd, h2_rnd))

# === Module 4 : Visualisation ===
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax1.plot(range(1, max_mf_iters + 1), delta_history, "o-", color="#C0392B", lw=2.0)
ax1.set_yscale("log")
ax1.set_xlabel("Mean-Field Fixed-Point Iteration", fontsize=8.5)
ax1.set_ylabel(r"Variational Delta $\Vert \Delta \mu \Vert$ (Log Scale)", fontsize=8.5)
ax1.set_title("Mean-Field Convergence to Variational Fixed Point", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.45)

ax2.plot(range(1, max_mf_iters + 1), [m[0] for m in mu1_history], label="mu1[0]", color="#2980B9", lw=1.5)
ax2.plot(range(1, max_mf_iters + 1), [m[1] for m in mu1_history], label="mu1[1]", color="#27AE60", lw=1.5)
ax2.plot(range(1, max_mf_iters + 1), [m[0] for m in mu2_history], label="mu2[0]", color="#E67E22", lw=1.5, linestyle="--")
ax2.plot(range(1, max_mf_iters + 1), [m[1] for m in mu2_history], label="mu2[1]", color="#8E44AD", lw=1.5, linestyle="--")
ax2.set_xlabel("Iteration", fontsize=8.5)
ax2.set_ylabel("Variational Probability Activation mu", fontsize=8.5)
ax2.set_title("Layer 1 & Layer 2 Latent Probability Settling", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Deep Boltzmann Machine: Variational Mean-Field Dynamics", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_mean_field_convergence.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 4.0))
bins = np.linspace(min(min(clamped_energies), min(random_energies)) - 1,
                   max(max(clamped_energies), max(random_energies)) + 1, 25)
ax.hist(clamped_energies, bins=bins, color="#27AE60", alpha=0.75, edgecolor="black", label="Clamped Empirical States (Lower Energy)")
ax.hist(random_energies, bins=bins, color="#C0392B", alpha=0.60, edgecolor="black", label="Random Unclamped Configurations (High Energy)")
ax.set_xlabel("Joint Configuration Energy E(v, h1, h2)", fontsize=9)
ax.set_ylabel("Sample Frequency", fontsize=9)
ax.set_title("Energy Histogram: Clamped Variational States vs Random Configurations", fontsize=10.5)
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_energy_simulation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Layer", "Unit Cardinality", "Connected Layers", "Interaction Type", "Topological Graph"]
t1_rows = [
    ["Visible Layer (v)", f"{n_v} binary units", "Hidden Layer 1 (h1)", "Bipartite undirected W1", "Undirected Graphical Model"],
    ["Hidden Layer 1 (h1)", f"{n_h1} binary units", "Visible (v) & Hidden 2 (h2)", "Two-way interactions (W1, W2)", "Bidirectional message passing"],
    ["Hidden Layer 2 (h2)", f"{n_h2} binary units", "Hidden Layer 1 (h1)", "Bipartite undirected W2", "Top-level abstraction layer"]
]
show_and_save("Table 1 - Deep Boltzmann Machine Undirected Topology Specifications", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Iteration", "Delta Norm ||Delta mu||", "Mean Activation mu1", "Mean Activation mu2", "Convergence Status"]
milestones = [1, 2, 4, 6, 8, 12]
t2_rows = [
    [f"Iter {m:02d}", f"{delta_history[m-1]:.4e}", f"{np.mean(mu1_history[m-1]):.4f}", f"{np.mean(mu2_history[m-1]):.4f}", "Settled (< 1e-4)" if delta_history[m-1] < 1e-4 else "Updating"]
    for m in milestones
]
show_and_save("Table 2 - Mean-Field Fixed-Point Iteration Progression Record", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Interaction Weight Matrix", "Tensor Order", "Frobenius Norm", "Max Weight", "Min Weight"]
t3_rows = [
    ["W1 (Visible <-> Hidden 1)", f"{n_v} x {n_h1}", f"{np.linalg.norm(W1):.4f}", f"{np.max(W1):+.4f}", f"{np.min(W1):+.4f}"],
    ["W2 (Hidden 1 <-> Hidden 2)", f"{n_h1} x {n_h2}", f"{np.linalg.norm(W2):.4f}", f"{np.max(W2):+.4f}", f"{np.min(W2):+.4f}"]
]
show_and_save("Table 3 - Bipartite Interaction Synaptic Matrices Metrics", t3_headers, t3_rows, "table 3.png")

t4_headers = ["State Regime", "Mean Energy E", "Min Energy", "Max Energy", "Energy Differential vs Random"]
t4_rows = [
    ["Clamped Mean-Field States", f"{np.mean(clamped_energies):.3f}", f"{np.min(clamped_energies):.3f}", f"{np.max(clamped_energies):.3f}", f"{np.mean(clamped_energies) - np.mean(random_energies):+.3f} (More Negative)"],
    ["Random Unclamped States", f"{np.mean(random_energies):.3f}", f"{np.min(random_energies):.3f}", f"{np.max(random_energies):.3f}", "Baseline (High Disorder)"]
]
show_and_save("Table 4 - Joint Energy Statistics and Energy Well Audit", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Feature", "Deep Belief Network (DBN)", "Deep Boltzmann Machine (DBM)", "Convolutional Boltzmann Machine"]
t5_rows = [
    ["Graph Topology", "Directed hybrid (directed top to bottom)", "Completely undirected bipartite layers", "Undirected with shared convolutional kernels"],
    ["Inference Method", "Single forward deterministic pass", "Iterative mean-field variational fixed point", "Convolutive variational mean-field"],
    ["Top-Down Influence", "No top-down feedback during inference", "Top-down & bottom-up simultaneously interact", "Spatial top-down and lateral filtering"],
    ["Key Application", "Fast feature extraction & classification", "Unsupervised multi-modal representation learning", "Large-scale 2D image generative synthesis"]
]
show_and_save("Table 5 - Comparative Synthesis: DBN vs DBM vs Convolutional BM", t5_headers, t5_rows, "table 5.png")
