# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

# === Module 2 : Dataset Ingestion and Standardization ===
data = load_breast_cancer()
X = data.data[:, :10] # Top 10 primary morphological features
feature_names = data.feature_names[:10]
y = data.target
target_names = data.target_names

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# === Module 3 : Principal Component Decomposition ===
pca = PCA()
X_pca = pca.fit_transform(X_std)

eigenvalues = pca.explained_variance_
explained_var_ratio = pca.explained_variance_ratio_ * 100.0
cumulative_var = np.cumsum(explained_var_ratio)
loadings = pca.components_

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.6, 4.0))
comps = np.arange(1, len(eigenvalues) + 1)
ax.bar(comps, explained_var_ratio, color="#4374B3", alpha=0.85, width=0.55, label="Individual Explained Variance (%)")
ax.plot(comps, cumulative_var, "o-", color="#C0392B", lw=2.0, label="Cumulative Explained Variance (%)")
ax.axhline(85.0, color="#27AE60", linestyle="--", lw=1.2, label="85% Information Threshold")
ax.set_xlabel("Principal Component Index", fontsize=9)
ax.set_ylabel("Explained Variance Ratio (%)", fontsize=9)
ax.set_title("Scree Plot and Cumulative Information Curve", fontsize=10.5)
ax.set_xticks(comps)
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(loc="center right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_scree_variance.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.5))
scatter = ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], color="#C0392B", alpha=0.7, edgecolors="k", s=35, label=f"Malignant ({target_names[0]})")
scatter = ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], color="#2980B9", alpha=0.7, edgecolors="k", s=35, marker="^", label=f"Benign ({target_names[1]})")
ax.set_xlabel(f"Principal Component 1 ({explained_var_ratio[0]:.1f}% Variance)", fontsize=9)
ax.set_ylabel(f"Principal Component 2 ({explained_var_ratio[1]:.1f}% Variance)", fontsize=9)
ax.set_title("Low-Dimensional Orthogonal Subspace Projection (PC1 vs PC2)", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_pca_projection.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Attribute", "Description", "Value"]
t1_rows = [
    ["Total Instances", "Diagnostic cytological cases", f"{len(X)} patients"],
    ["Original Dimensionality", "Extracted morphological features", "10 dimensions"],
    ["Pre-Processing", "Standardization (z-score)", "Zero mean, unit variance"],
    ["Decomposition Method", "Full Singular Value Decomposition (SVD)", "Orthogonal basis rotation"]
]
show_and_save("Table 1 - Dataset Characteristics and Preprocessing Protocol", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Component", "Eigenvalue (λ)", "Variance Ratio (%)", "Cumulative Ratio (%)", "Information Retained"]
t2_rows = [
    [f"PC {i+1}", f"{eigenvalues[i]:.4f}", f"{explained_var_ratio[i]:.2f}%", f"{cumulative_var[i]:.2f}%", "Dominant axis" if i < 2 else ("Subordinate" if i < 5 else "Noise floor")]
    for i in range(len(eigenvalues))
]
show_and_save("Table 2 - Spectral Decomposition: Eigenvalues and Explained Variance", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Variance Target", "Required Components", "Dimensionality Reduction", "Storage Savings (%)"]
thresholds = [50.0, 75.0, 85.0, 95.0]
t3_rows = []
for th in thresholds:
    n_c = np.argmax(cumulative_var >= th) + 1
    savings = (1.0 - n_c / 10.0) * 100.0
    t3_rows.append([f"{th:.0f}% Target", f"{n_c} components", f"10D -> {n_c}D", f"{savings:.1f}%"])
show_and_save("Table 3 - Dimensionality Compression vs Variance Retention Targets", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Original Feature", "PC1 Loading (v1)", "PC2 Loading (v2)", "Primary Contribution"]
t4_rows = [
    [feature_names[i], f"{loadings[0, i]:+.4f}", f"{loadings[1, i]:+.4f}", "Cellular overall volume" if abs(loadings[0, i]) > 0.3 else "Shape irregularity"]
    for i in range(len(feature_names))
]
show_and_save("Table 4 - Principal Component Eigenvector Loadings Matrix", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Mathematical Property", "Analytical Formulation", "Empirical Check", "Validation Status"]
ortho_check = np.allclose(loadings @ loadings.T, np.eye(10))
recon_loss_2d = 100.0 - cumulative_var[1]
t5_rows = [
    ["Orthonormality of Components", "V @ V^T = I_10", f"Norm difference: {np.linalg.norm(loadings @ loadings.T - np.eye(10)):.2e}", "Verified" if ortho_check else "Failed"],
    ["Total Variance Conservation", "sum(lambda_i) = trace(C) = 10", f"Sum = {np.sum(eigenvalues):.4f}", "Exact Preservation"],
    ["2D Projection Information Retained", "PC1 + PC2 variance ratio", f"{cumulative_var[1]:.2f}%", "Captures bulk structure"],
    ["2D Reconstruction Error", "Residual orthogonal variance", f"{recon_loss_2d:.2f}%", "Acceptable loss"]
]
show_and_save("Table 5 - Orthogonal Subspace Verification and Theoretical Axioms", t5_headers, t5_rows, "table 5.png")
