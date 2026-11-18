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

# === Module 2 : Anisotropic Quadratic Loss Surface ===
# f(w1, w2) = 0.5 * (w1^2 + 20 * w2^2)
def loss_func(w):
    return 0.5 * (w[0]**2 + 20.0 * w[1]**2)

def grad_func(w):
    return np.array([w[0], 20.0 * w[1]])

# === Module 3 : Simulating the Four Optimizers ===
np.random.seed(52)
n_steps = 60
w_init = np.array([-4.2, 3.2])
lr = 0.08

# 1. Standard Gradient Descent
w_gd = np.zeros((n_steps + 1, 2))
w_gd[0] = w_init
for t in range(n_steps):
    w_gd[t+1] = w_gd[t] - lr * grad_func(w_gd[t])

# 2. Stochastic Gradient Descent with Mini-batch Noise
w_sgd = np.zeros((n_steps + 1, 2))
w_sgd[0] = w_init
for t in range(n_steps):
    noise = np.random.normal(0, 0.45, size=2)
    w_sgd[t+1] = w_sgd[t] - lr * (grad_func(w_sgd[t]) + noise)

# 3. Momentum Optimizer (beta = 0.85)
w_mom = np.zeros((n_steps + 1, 2))
w_mom[0] = w_init
v = np.zeros(2)
beta = 0.85
for t in range(n_steps):
    v = beta * v + (1.0 - beta) * grad_func(w_mom[t])
    w_mom[t+1] = w_mom[t] - lr * 2.5 * v

# 4. Adam Optimizer (beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
w_adam = np.zeros((n_steps + 1, 2))
w_adam[0] = w_init
m_adam = np.zeros(2)
v_adam = np.zeros(2)
beta1, beta2 = 0.90, 0.999
eps = 1e-8
lr_adam = 0.25
for t in range(1, n_steps + 1):
    g = grad_func(w_adam[t-1])
    m_adam = beta1 * m_adam + (1.0 - beta1) * g
    v_adam = beta2 * v_adam + (1.0 - beta2) * (g**2)
    m_hat = m_adam / (1.0 - beta1**t)
    v_hat = v_adam / (1.0 - beta2**t)
    w_adam[t] = w_adam[t-1] - lr_adam * m_hat / (np.sqrt(v_hat) + eps)

loss_gd = np.array([loss_func(w) for w in w_gd])
loss_sgd = np.array([loss_func(w) for w in w_sgd])
loss_mom = np.array([loss_func(w) for w in w_mom])
loss_adam = np.array([loss_func(w) for w in w_adam])

# Path distance (total variation / oscillation measure)
dist_gd = np.sum(np.linalg.norm(np.diff(w_gd, axis=0), axis=1))
dist_sgd = np.sum(np.linalg.norm(np.diff(w_sgd, axis=0), axis=1))
dist_mom = np.sum(np.linalg.norm(np.diff(w_mom, axis=0), axis=1))
dist_adam = np.sum(np.linalg.norm(np.diff(w_adam, axis=0), axis=1))
euclid_dist = np.linalg.norm(w_init)

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(loss_gd, label="Gradient Descent (GD)", color="#34495E", lw=1.8)
ax.plot(loss_sgd, label="Stochastic GD (SGD)", color="#E67E22", lw=1.6, alpha=0.85)
ax.plot(loss_mom, label="Momentum ($\\beta=0.85$)", color="#27AE60", lw=2.0)
ax.plot(loss_adam, label="Adam Optimizer", color="#2980B9", lw=2.2)
ax.set_yscale("log")
ax.set_xlabel("Iteration ($t$)", fontsize=9)
ax.set_ylabel("Loss $\\mathcal{L}(w)$ (Log Scale)", fontsize=9)
ax.set_title("Optimization Convergence Trajectories on Ill-Conditioned Surface", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_loss_curves.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.4, 5.0))
w1_grid = np.linspace(-5.0, 1.5, 200)
w2_grid = np.linspace(-3.8, 3.8, 200)
W1, W2 = np.meshgrid(w1_grid, w2_grid)
Z = 0.5 * (W1**2 + 20.0 * W2**2)
ax.contour(W1, W2, Z, levels=np.logspace(-1, 2.5, 22), cmap="cividis", alpha=0.6)

ax.plot(w_gd[:, 0], w_gd[:, 1], "o-", color="#34495E", ms=3, lw=1.2, label="GD Trajectory")
ax.plot(w_sgd[:, 0], w_sgd[:, 1], "d-", color="#E67E22", ms=3, lw=1.1, alpha=0.7, label="SGD Trajectory")
ax.plot(w_mom[:, 0], w_mom[:, 1], "^-", color="#27AE60", ms=3.5, lw=1.5, label="Momentum")
ax.plot(w_adam[:, 0], w_adam[:, 1], "s-", color="#2980B9", ms=3.5, lw=1.8, label="Adam")

ax.scatter([0], [0], color="#C0392B", marker="*", s=140, zorder=10, label="Global Optimum $(0,0)$")
ax.scatter([w_init[0]], [w_init[1]], color="#000000", marker="o", s=60, zorder=10, label="Starting Point")

ax.set_xlabel("Parameter $w_1$", fontsize=9)
ax.set_ylabel("Parameter $w_2$", fontsize=9)
ax.set_title("2D Contour Optimization Trajectories in Ravine Landscape", fontsize=10)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_trajectory.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Parameter", "Description", "Numerical Setting"]
t1_rows = [
    ["Initial Coordinates", "Starting vector w_0 = (w1, w2)", "(-4.20, 3.20)"],
    ["Loss Formulation", "f(w) = 0.5 * (w1^2 + 20*w2^2)", "Elliptical paraboloid bowl"],
    ["Condition Number", "kappa = lambda_max / lambda_min", "20.0 (High anisotropy)"],
    ["Total Iterations", "T steps", "60 iterations"],
    ["Base Learning Rates", "alpha (GD, SGD, Mom) / alpha (Adam)", "0.08 / 0.25"]
]
show_and_save("Table 1 - Experimental Optimization Hyperparameters", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Iteration", "GD Loss", "SGD Loss", "Momentum Loss", "Adam Loss"]
milestones = [0, 5, 15, 30, 45, 60]
t2_rows = [
    [f"t = {t}", f"{loss_gd[t]:.4f}", f"{loss_sgd[t]:.4f}", f"{loss_mom[t]:.4f}", f"{loss_adam[t]:.4f}"]
    for t in milestones
]
show_and_save("Table 2 - Loss Value Trajectory across Key Milestone Steps", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Optimizer", "Final Coordinates (w1, w2)", "Final Loss L(w_T)", "Distance to Origin", "Convergence Status"]
opt_names = ["Gradient Descent", "Stochastic GD", "Momentum", "Adam"]
final_w = [w_gd[-1], w_sgd[-1], w_mom[-1], w_adam[-1]]
final_l = [loss_gd[-1], loss_sgd[-1], loss_mom[-1], loss_adam[-1]]
t3_rows = [
    [opt_names[i], f"({final_w[i][0]:.3f}, {final_w[i][1]:.3f})", f"{final_l[i]:.6f}", f"{np.linalg.norm(final_w[i]):.4f}", "Converged" if final_l[i] < 0.05 else "Slow Convergence"]
    for i in range(4)
]
show_and_save("Table 3 - Terminal Performance and Distance to Global Minimum", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Optimizer", "Straight Line Dist", "Actual Path Traveled", "Oscillation Ratio (Path/Dist)", "Path Efficiency"]
paths = [dist_gd, dist_sgd, dist_mom, dist_adam]
t4_rows = [
    [opt_names[i], f"{euclid_dist:.3f}", f"{paths[i]:.3f}", f"{paths[i]/euclid_dist:.2f}x", "High" if paths[i]/euclid_dist < 1.4 else "Zig-zag / Low"]
    for i in range(4)
]
show_and_save("Table 4 - Trajectory Efficiency and Transverse Oscillation Audit", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Algorithm", "Update Dynamics", "Oscillation Damping", "Adaptive Step Size", "Key Benefit"]
t5_rows = [
    ["Gradient Descent", "w = w - alpha * grad", "None (high zig-zag)", "Fixed global scalar", "Simple baseline"],
    ["Stochastic GD", "w = w - alpha * (grad + eps)", "Stochastic exploration", "Fixed global scalar", "Escapes shallow saddles"],
    ["Momentum", "v = beta*v + (1-beta)*grad", "Accelerates along consistent gradient", "Directional damping", "Damps cross-ravine bounce"],
    ["Adam", "Adaptive first & second moments", "Exponential decay of velocity", "Per-parameter adaptive rate", "Robust across ill-conditioned surfaces"]
]
show_and_save("Table 5 - Algorithmic Comparison and Mechanics Synthesis", t5_headers, t5_rows, "table 5.png")
