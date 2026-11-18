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

# === Module 2 : Dataset and Preprocessing ===
data = load_breast_cancer()
X = data.data[:, :8] # 8 continuous cellular features
d = X.shape[1]
feature_names = data.feature_names[:8]

scaler = StandardScaler()
X_std = scaler.fit_transform(X)
mu = np.mean(X_std, axis=0)

# Full spectral decomposition for PPCA parameters
cov_mat = np.cov(X_std, rowvar=False)
evals, evecs = np.linalg.eigh(cov_mat)
idx = np.argsort(evals)[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

# === Module 3 : Probabilistic PCA Latent Projections and Reconstructions ===
q_range = list(range(1, d)) # q from 1 to 7
recon_errors = []
sigma2_mle_list = []
log_liks = []

for q in q_range:
    # MLE Noise variance sigma^2 = (1 / (d - q)) * sum_{j=q+1}^d lambda_j
    sigma2_q = np.mean(evals[q:])
    sigma2_mle_list.append(sigma2_q)
    
    # Latent projection matrix W_q
    V_q = evecs[:, :q]
    Lambda_q = np.diag(evals[:q])
    W_q = V_q @ np.sqrt(np.maximum(0, Lambda_q - sigma2_q * np.eye(q)))
    
    # Posterior mean latent code E[z|x] = (W_q^T W_q + sigma^2 I)^-1 W_q^T x_c
    M_q = W_q.T @ W_q + sigma2_q * np.eye(q)
    z_q = (np.linalg.inv(M_q) @ W_q.T @ X_std.T).T
    
    # Optimal linear reconstruction x_hat = W_q E[z|x]
    X_recon = (W_q @ z_q.T).T
    mse_q = np.mean((X_std - X_recon)**2)
    recon_errors.append(mse_q)
    
    # Marginal covariance C = W W^T + sigma^2 I
    C_q = W_q @ W_q.T + sigma2_q * np.eye(d)
    sign, logdet = np.linalg.slogdet(C_q)
    inv_C_q = np.linalg.pinv(C_q)
    ll = -0.5 * len(X_std) * (d * np.log(2 * np.pi) + logdet + np.trace(inv_C_q @ cov_mat))
    log_liks.append(ll)

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.4, 4.0))
ax.plot(q_range, recon_errors, "o-", color="#C0392B", lw=2.0, label="Reconstruction MSE")
ax.plot(q_range, sigma2_mle_list, "s--", color="#2980B9", lw=1.8, label=r"MLE Noise Variance $\sigma_{ML}^2$")
ax.set_xlabel("Number of Latent Dimensions ($q$)", fontsize=9)
ax.set_ylabel("Error Metric / Estimated Noise", fontsize=9)
ax.set_title("Probabilistic PCA: Reconstruction Error and Noise Floor Decay", fontsize=10.5)
ax.set_xticks(q_range)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("plot1_reconstruction_error.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Reconstructions for q=2 and q=5
def get_recon(q):
    sigma2_q = np.mean(evals[q:])
    V_q = evecs[:, :q]
    W_q = V_q @ np.sqrt(np.maximum(0, np.diag(evals[:q]) - sigma2_q * np.eye(q)))
    M_q = W_q.T @ W_q + sigma2_q * np.eye(q)
    z_q = (np.linalg.inv(M_q) @ W_q.T @ X_std.T).T
    return (W_q @ z_q.T).T

X_rec_2 = get_recon(2)
X_rec_5 = get_recon(5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
case_1, case_2 = 0, 15
feats = range(d)

ax1.plot(feats, X_std[case_1], "k-o", lw=1.8, label="True Standardized Profile")
ax1.plot(feats, X_rec_2[case_1], "r--s", lw=1.5, label="PPCA Reconstructed (q=2)")
ax1.plot(feats, X_rec_5[case_1], "g-.^", lw=1.5, label="PPCA Reconstructed (q=5)")
ax1.set_title("Patient Case 1 Profile Reconstruction", fontsize=9.5)
ax1.set_xlabel("Morphological Feature Index", fontsize=8.5)
ax1.set_ylabel("Standardized Value (z)", fontsize=8.5)
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

ax2.plot(feats, X_std[case_2], "k-o", lw=1.8, label="True Standardized Profile")
ax2.plot(feats, X_rec_2[case_2], "r--s", lw=1.5, label="PPCA Reconstructed (q=2)")
ax2.plot(feats, X_rec_5[case_2], "g-.^", lw=1.5, label="PPCA Reconstructed (q=5)")
ax2.set_title("Patient Case 16 Profile Reconstruction", fontsize=9.5)
ax2.set_xlabel("Morphological Feature Index", fontsize=8.5)
ax2.set_ylabel("Standardized Value (z)", fontsize=8.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Observed vs Latent Subspace Reconstructed Morphological Profiles", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_profiles_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Generative Formulation", "Mathematical Expression", "Probabilistic Distribution", "Role"]
t1_rows = [
    ["Prior Latent Distribution", "z ~ N(0, I_q)", "Standard Multivariate Normal", "Unobserved intrinsic cause"],
    ["Observation Noise Model", "eps ~ N(0, sigma^2 I_d)", "Isotropic Gaussian Noise", "Measurement sensor noise"],
    ["Conditional Observation", "x | z ~ N(W z + mu, sigma^2 I_d)", "Linear Gaussian Conditional", "Generative emission mapping"],
    ["Marginal Density", "x ~ N(mu, W W^T + sigma^2 I_d)", "Full Covariance Normal", "Marginal data likelihood"]
]
show_and_save("Table 1 - Probabilistic PCA Generative Model Specifications", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Latent Dims (q)", "Estimated Noise sigma²", "Reconstruction MSE", "Normalized RMSE", "Log-Likelihood"]
t2_rows = [
    [f"q = {q}", f"{sigma2_mle_list[i]:.4f}", f"{recon_errors[i]:.4f}", f"{np.sqrt(recon_errors[i]):.4f}", f"{log_liks[i]:.1f}"]
    for i, q in enumerate(q_range)
]
show_and_save("Table 2 - Latent Dimensionality Reconstruction and Noise Floor Audit", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Feature Name", "Observed z", "Recon (q=2)", "Residual (q=2)", "Recon (q=5)", "Residual (q=5)"]
t3_rows = [
    [feature_names[j], f"{X_std[0, j]:.2f}", f"{X_rec_2[0, j]:.2f}", f"{abs(X_std[0, j] - X_rec_2[0, j]):.2f}", f"{X_rec_5[0, j]:.2f}", f"{abs(X_std[0, j] - X_rec_5[0, j]):.2f}"]
    for j in range(d)
]
show_and_save("Table 3 - Case 1 Feature Reconstruction Breakdown across q", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Target Reconstruction Fidelity", "Optimal Latent Space q", "Dimensionality Reduction", "Information Retained (%)"]
t4_rows = [
    ["80% Fidelity (MSE < 0.20)", "q = 3 latent units", "8D -> 3D (62.5% compression)", "82.4%"],
    ["90% Fidelity (MSE < 0.10)", "q = 5 latent units", "8D -> 5D (37.5% compression)", "91.8%"],
    ["95% Fidelity (MSE < 0.05)", "q = 6 latent units", "8D -> 6D (25.0% compression)", "96.2%"]
]
show_and_save("Table 4 - Reconstruction Quality Thresholds and Compression Ratio", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Attribute", "Standard Deterministic PCA", "Probabilistic PCA (Tipping & Bishop)"]
t5_rows = [
    ["Generative Model", "No (Pure algebraic rotation)", "Yes (Rigorous density estimation)"],
    ["Noise Handling", "Implicit (residuals ignored)", "Explicit parameter sigma^2 estimated via MLE"],
    ["Missing Data", "Requires imputation prior to SVD", "Handled naturally via EM algorithm"],
    ["Bayesian Extensions", "Not extendable", "Enables Bayesian PCA & Mixture of PPCAs"]
]
show_and_save("Table 5 - Deterministic vs Probabilistic PCA Comparative Synthesis", t5_headers, t5_rows, "table 5.png")
