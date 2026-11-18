# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
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

# === Module 2 : BPTT Gradient Simulation Setup ===
np.random.seed(42)
T = 20 # 20 time steps
dh = 8 # 8 hidden units
dx = 1

# Generate synthetic input sequence
x_seq = np.random.randn(T, dx) * 0.5

# Function to run forward pass and backprop through time
def compute_bptt_gradients(spectral_scale):
    # Construct recurrent weight matrix with controlled spectral radius
    raw_W = np.random.randn(dh, dh)
    u, s, vt = np.linalg.svd(raw_W)
    W_hh = u @ np.diag(np.linspace(spectral_scale, spectral_scale * 0.7, dh)) @ vt
    W_xh = np.random.randn(dx, dh) * 0.4
    
    # Forward Pass
    h_states = [np.zeros(dh)]
    for t in range(T):
        ht = np.tanh(x_seq[t] @ W_xh + h_states[-1] @ W_hh)
        h_states.append(ht)
    
    # Backward Pass: loss at final step T
    dL_dh = np.zeros((T + 1, dh))
    dL_dh[T] = np.ones(dh) # Seed gradient at final time step
    
    grad_norms = [np.linalg.norm(dL_dh[T])]
    singular_values_list = []
    
    for t in range(T, 0, -1):
        # Jacobian J_t = diag(1 - h_t^2) @ W_hh^T
        ht = h_states[t]
        diag_deriv = np.diag(1.0 - ht**2)
        J_t = diag_deriv @ W_hh.T
        singular_values_list.append(np.linalg.svd(J_t, compute_uv=False))
        
        dL_dh[t - 1] = dL_dh[t] @ J_t
        grad_norms.append(np.linalg.norm(dL_dh[t - 1]))
        
    return np.array(grad_norms), np.array(singular_values_list), W_hh

# 1. Vanishing regime: spectral radius 0.85
norms_vanish, sv_vanish, W_vanish = compute_bptt_gradients(0.85)

# 2. Exploding regime: spectral radius 1.35
norms_explode, sv_explode, W_explode = compute_bptt_gradients(1.35)

# === Module 3 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.4, 4.0))
steps_back = np.arange(len(norms_vanish))
ax.plot(steps_back, norms_vanish, "o-", color="#2980B9", lw=2.0, label="Subcritical ($\\rho=0.85$) - Vanishing Gradient")
ax.plot(steps_back, norms_explode, "s-", color="#C0392B", lw=2.0, label="Supercritical ($\\rho=1.35$) - Exploding Gradient")
ax.set_yscale("log")
ax.set_xlabel("BPTT Backpropagation Steps $(T - t)$", fontsize=9)
ax.set_ylabel(r"Gradient Norm $\Vert \partial L_T / \partial h_t \Vert$ (Log Scale)", fontsize=9)
ax.set_title("BPTT Gradient Norm Trajectory: Vanishing vs Exploding Dynamics", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_gradient_decay.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
ax1.plot(sv_vanish[:, 0], color="#2980B9", lw=1.8, label="Max Singular Value $\\sigma_1$")
ax1.plot(sv_vanish[:, 3], color="#16A085", lw=1.5, linestyle="--", label="Median Singular Value $\\sigma_4$")
ax1.plot(sv_vanish[:, -1], color="#7F8C8D", lw=1.2, linestyle=":", label="Min Singular Value $\\sigma_8$")
ax1.axhline(1.0, color="k", linestyle="--", lw=1.0)
ax1.set_xlabel("Backward Step Count", fontsize=8.5)
ax1.set_ylabel("Singular Values of Jacobian $J_t$", fontsize=8.5)
ax1.set_title("Subcritical Jacobian Spectrum ($\\sigma < 1$)", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

ax2.plot(sv_explode[:, 0], color="#C0392B", lw=1.8, label="Max Singular Value $\\sigma_1$")
ax2.plot(sv_explode[:, 3], color="#E67E22", lw=1.5, linestyle="--", label="Median Singular Value $\\sigma_4$")
ax2.plot(sv_explode[:, -1], color="#7F8C8D", lw=1.2, linestyle=":", label="Min Singular Value $\\sigma_8$")
ax2.axhline(1.0, color="k", linestyle="--", lw=1.0)
ax2.set_xlabel("Backward Step Count", fontsize=8.5)
ax2.set_ylabel("Singular Values of Jacobian $J_t$", fontsize=8.5)
ax2.set_title("Supercritical Jacobian Spectrum ($\\sigma_1 > 1$)", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Temporal Jacobian Singular Value Spectra Across Recurrent Steps", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_jacobian_spectrum.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 4 : Printing and Saving Tables ===
t1_headers = ["Gradient Term", "Mathematical Expansion", "Component Role", "Dimensionality"]
t1_rows = [
    ["Terminal Error Seed", "dL/dh_T", "Direct loss sensitivity at final step", "R^{1x8}"],
    ["Recurrent Jacobian", "dh_t / dh_{t-1} = diag(1 - h_t^2) * W_hh^T", "Step-to-step state transition derivative", "R^{8x8}"],
    ["Compound Gradient Flow", "dL/dh_t = (dL/dh_T) * prod_{k=t+1}^T (dh_k / dh_{k-1})", "Chain rule expansion across T - t steps", "R^{1x8}"],
    ["Weight Gradient", "dL/dW_hh = sum_{t=1}^T (dL/dh_t)^T * h_{t-1}", "Accumulated recurrent synaptic update", "R^{8x8}"]
]
show_and_save("Table 1 - Backpropagation Through Time (BPTT) Chain Rule Equations", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Time Step t", "Lag (T - t)", "Vanishing Gradient Norm", "Exploding Gradient Norm", "Magnitude Ratio"]
sample_steps = [20, 18, 15, 10, 5, 1]
t2_rows = []
for t in sample_steps:
    lag = 20 - t
    norm_v = norms_vanish[lag]
    norm_e = norms_explode[lag]
    ratio = norm_e / (norm_v + 1e-12)
    t2_rows.append([f"t = {t:02d}", f"{lag:02d} steps back", f"{norm_v:.4e}", f"{norm_e:.4e}", f"{ratio:.2e}x"])
show_and_save("Table 2 - Temporal Gradient Norm Trajectory from Step 20 to Step 1", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Regime", "Initial Norm (t=20)", "Terminal Norm (t=1)", "Attenuation / Growth Factor", "Stability Diagnosis"]
att_v = norms_vanish[-1] / norms_vanish[0]
att_e = norms_explode[-1] / norms_explode[0]
t3_rows = [
    ["Subcritical (rho=0.85)", f"{norms_vanish[0]:.4f}", f"{norms_vanish[-1]:.4e}", f"{att_v:.4e}", "Severe Vanishing (Information Lost)"],
    ["Supercritical (rho=1.35)", f"{norms_explode[0]:.4f}", f"{norms_explode[-1]:.4e}", f"{att_e:.4e}", "Severe Explosion (Numerical Instability)"]
]
show_and_save("Table 3 - Long-Horizon Gradient Attenuation and Explosion Audit", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Regime", "W_hh Spectral Radius", "Max Singular Value (Mean)", "Min Singular Value (Mean)", "Condition Number"]
t4_rows = [
    ["Subcritical", "0.8500", f"{np.mean(sv_vanish[:, 0]):.4f}", f"{np.mean(sv_vanish[:, -1]):.4f}", f"{np.mean(sv_vanish[:, 0])/np.mean(sv_vanish[:, -1]):.2f}"],
    ["Supercritical", "1.3500", f"{np.mean(sv_explode[:, 0]):.4f}", f"{np.mean(sv_explode[:, -1]):.4f}", f"{np.mean(sv_explode[:, 0])/np.mean(sv_explode[:, -1]):.2f}"]
]
show_and_save("Table 4 - Jacobian Spectral Radius and Singular Value Breakdown", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Recurrent Challenge", "Root Mathematical Cause", "Empirical Symptom", "Modern Architectural Remedy"]
t5_rows = [
    ["Vanishing Gradients", "Repeated multiplication by J < 1 (rho < 1)", "Weights stop updating for early time steps", "LSTM / GRU Constant Error Carousel (Gates)"],
    ["Exploding Gradients", "Repeated multiplication by J > 1 (rho > 1)", "NaN loss, destabilized parameter trajectories", "Gradient Norm Clipping (clipnorm / clipvalue)"],
    ["Short-Term Memory Bias", "Exponential gradient decay over time horizon", "RNN only responds to recent 3-5 tokens", "Attention mechanisms / Transformers"]
]
show_and_save("Table 5 - Recurrent Gradient Bottleneck and Engineering Solutions", t5_headers, t5_rows, "table 5.png")
