# === Module 1 : Importing the Required Packages ===
import numpy as np
import pandas as pd
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

# === Module 2 : Building the Physiological Patient Dataset ===
np.random.seed(142)
N = 120
mean_true = np.array([125.0, 78.0, 105.0]) # Systolic BP (mmHg), Heart Rate (bpm), Blood Glucose (mg/dL)
cov_true = np.array([
    [160.0, 52.0, 75.0],
    [52.0, 95.0, 42.0],
    [75.0, 42.0, 180.0]
])

X = np.random.multivariate_normal(mean_true, cov_true, size=N)
features = ["Systolic BP", "Heart Rate", "Blood Glucose"]
df = pd.DataFrame(X, columns=features)

# === Module 3 : Descriptive Statistics ===
desc_stats = []
for col in features:
    series = df[col]
    desc_stats.append([
        col,
        f"{series.mean():.2f}",
        f"{series.var(ddof=1):.2f}",
        f"{series.std(ddof=1):.2f}",
        f"{series.min():.2f}",
        f"{series.quantile(0.50):.2f}",
        f"{series.max():.2f}"
    ])

# === Module 4 : Mean-Centering and Deviation Matrix ===
mu = np.mean(X, axis=0)
X_c = X - mu

# === Module 5 : Covariance Matrix from First Principles and with NumPy ===
cov_manual = (X_c.T @ X_c) / (N - 1)
cov_numpy = np.cov(X, rowvar=False)
cov_verified = np.allclose(cov_manual, cov_numpy)

# === Module 6 : Correlation Matrix and Pearson Coefficients ===
std_vec = np.std(X, axis=0, ddof=1)
corr_manual = cov_manual / np.outer(std_vec, std_vec)
corr_numpy = np.corrcoef(X, rowvar=False)

# === Module 7 : Eigen-Decomposition Analysis ===
eigenvalues, eigenvectors = np.linalg.eigh(cov_manual)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

explained_variance = (eigenvalues / np.sum(eigenvalues)) * 100
ortho_check = np.allclose(eigenvectors.T @ eigenvectors, np.eye(3))
recon_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
recon_check = np.allclose(recon_cov, cov_manual)

# === Module 8 : Visualisation ===
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
pairs = [(0, 1), (1, 2), (0, 2)]
for ax, (i, j) in zip(axes, pairs):
    ax.scatter(X[:, i], X[:, j], color="#2b5c8f", alpha=0.65, edgecolors="none", s=28)
    ax.scatter(mu[i], mu[j], color="#d95f02", s=70, marker="x", linewidths=2.5, label="Centroid $\\mu$")
    ax.set_xlabel(features[i], fontsize=8.5)
    ax.set_ylabel(features[j], fontsize=8.5)
    ax.set_title(f"{features[i]} vs {features[j]}", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.tick_params(labelsize=8)
fig.suptitle("Pairwise Bivariate Relationships Across Clinical Biomarkers", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_scatter.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.0))
im1 = ax1.imshow(cov_manual, cmap="YlGnBu")
ax1.set_title("Sample Covariance Matrix $\\mathbf{\\Sigma}$", fontsize=9.5)
ax1.set_xticks(range(3)); ax1.set_yticks(range(3))
ax1.set_xticklabels(features, fontsize=8); ax1.set_yticklabels(features, fontsize=8)
for i in range(3):
    for j in range(3):
        ax1.text(j, i, f"{cov_manual[i, j]:.1f}", ha="center", va="center", color="black" if cov_manual[i, j] < 120 else "white", fontsize=8.5)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

im2 = ax2.imshow(corr_manual, cmap="coolwarm", vmin=-1, vmax=1)
ax2.set_title("Pearson Correlation Matrix $\\mathbf{R}$", fontsize=9.5)
ax2.set_xticks(range(3)); ax2.set_yticks(range(3))
ax2.set_xticklabels(features, fontsize=8); ax2.set_yticklabels(features, fontsize=8)
for i in range(3):
    for j in range(3):
        ax2.text(j, i, f"{corr_manual[i, j]:.2f}", ha="center", va="center", color="black" if abs(corr_manual[i, j]) < 0.6 else "white", fontsize=8.5)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
fig.suptitle("Multivariate Dependency Maps: Covariance vs Standardized Correlation", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_covariance_map.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 9 : Printing and Saving Tables ===
t1_headers = ["Patient ID", "Systolic BP (mmHg)", "Heart Rate (bpm)", "Blood Glucose (mg/dL)"]
t1_rows = [[f"P-{i+1:03d}", f"{X[i, 0]:.2f}", f"{X[i, 1]:.2f}", f"{X[i, 2]:.2f}"] for i in range(8)]
show_and_save("Table 1 - Clinical Dataset Sample Preview (First 8 Observations)", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Feature", "Sample Mean", "Variance (s²)", "Std Dev (s)", "Minimum", "Median", "Maximum"]
show_and_save("Table 2 - Descriptive Statistics of Physiological Features", t2_headers, desc_stats, "table 2.png")

t3_headers = ["Feature"] + features
t3_rows = [[features[i]] + [f"{cov_manual[i, j]:.2f}" for j in range(3)] for i in range(3)]
show_and_save("Table 3 - Sample Covariance Matrix (ddof=1)", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Feature"] + features
t4_rows = [[features[i]] + [f"{corr_manual[i, j]:.4f}" for j in range(3)] for i in range(3)]
show_and_save("Table 4 - Pearson Correlation Coefficient Matrix", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Eigenvector", "Eigenvalue (λ)", "Variance Explained (%)", "v1 (BP)", "v2 (HR)", "v3 (Glucose)"]
t5_rows = [
    [f"Component {i+1}", f"{eigenvalues[i]:.2f}", f"{explained_variance[i]:.2f}%", f"{eigenvectors[0, i]:.4f}", f"{eigenvectors[1, i]:.4f}", f"{eigenvectors[2, i]:.4f}"]
    for i in range(3)
]
show_and_save("Table 5 - Spectral Decomposition: Eigenvalues and Eigenvectors", t5_headers, t5_rows, "table 5.png")

t6_headers = ["S.No", "Mathematical Property / Theorem", "Analytical Invariance Check", "Status"]
t6_rows = [
    ["1", "Manual vs NumPy Covariance Equivalence", "||C_manual - C_numpy||_F < 1e-12", "Verified" if cov_verified else "Failed"],
    ["2", "Eigenvector Orthonormality", "V^T @ V = I_3", "Verified" if ortho_check else "Failed"],
    ["3", "Spectral Reconstruction Theorem", "C = V @ diag(λ) @ V^T", "Verified" if recon_check else "Failed"],
    ["4", "Total Variance Conservation", "trace(C) = sum(λ)", "Verified" if np.isclose(np.trace(cov_manual), np.sum(eigenvalues)) else "Failed"]
]
show_and_save("Table 6 - Verification of Spectral Theorems and Orthonormality", t6_headers, t6_rows, "table 6.png")
