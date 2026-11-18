# === Module 1 : Importing the Required Packages ===
import numpy as np
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

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

# === Module 2 : Employee Productivity Survey Dataset ===
np.random.seed(42)
N = 250
# Two true underlying latent factors: F1 = Workload, F2 = WellBeing
F1_true = np.random.normal(0, 1, N)
F2_true = np.random.normal(0, 1, N)

# 6 observed survey items with distinct factor loading structures
# X1: Work Hours, X2: Task Volume, X3: Overtime
# X4: Sleep Quality, X5: Job Satisfaction, X6: Work-Life Balance
X1 = 0.85 * F1_true + np.random.normal(0, 0.35, N)
X2 = 0.80 * F1_true + np.random.normal(0, 0.40, N)
X3 = 0.75 * F1_true - 0.20 * F2_true + np.random.normal(0, 0.40, N)

X4 = 0.82 * F2_true + np.random.normal(0, 0.38, N)
X5 = -0.30 * F1_true + 0.78 * F2_true + np.random.normal(0, 0.35, N)
X6 = 0.85 * F2_true + np.random.normal(0, 0.32, N)

features = ["Work Hours", "Task Volume", "Overtime", "Sleep Quality", "Job Satisfaction", "Work-Life Balance"]
X_df = pd.DataFrame(np.column_stack([X1, X2, X3, X4, X5, X6]), columns=features)

scaler = StandardScaler()
X_std = scaler.fit_transform(X_df)

# === Module 3 : Factor Analysis with Varimax Rotation ===
n_factors = 2
fa_unrotated = FactorAnalysis(n_components=n_factors, random_state=42)
fa_unrotated.fit(X_std)

fa_varimax = FactorAnalysis(n_components=n_factors, rotation="varimax", random_state=42)
factors = fa_varimax.fit_transform(X_std)

loadings_unrot = fa_unrotated.components_.T
loadings_rot = fa_varimax.components_.T
noise_variance = fa_varimax.noise_variance_
communalities = np.sum(loadings_rot**2, axis=1)

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(6.8, 4.4))
im = ax.imshow(loadings_rot, cmap="coolwarm", vmin=-1.0, vmax=1.0)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Factor 1 (Workload Pressure)", "Factor 2 (Well-Being & Balance)"], fontsize=9)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features, fontsize=9)
for i in range(len(features)):
    for j in range(2):
        ax.text(j, i, f"{loadings_rot[i, j]:+.2f}", ha="center", va="center",
                color="white" if abs(loadings_rot[i, j]) > 0.55 else "black", fontsize=9.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Varimax Factor Loading")
ax.set_title("Varimax Rotated Factor Loading Matrix (Simple Structure)", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot1_factor_loadings.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4.4))
burnout_risk = (factors[:, 0] > 0.8) & (factors[:, 1] < -0.8)
ax.scatter(factors[~burnout_risk, 0], factors[~burnout_risk, 1], color="#2980B9", alpha=0.65, s=35, label="Normal Operations")
ax.scatter(factors[burnout_risk, 0], factors[burnout_risk, 1], color="#C0392B", alpha=0.9, s=55, marker="^", label="High Burnout Risk Cohort")
ax.axhline(0, color="k", linestyle="--", lw=0.8)
ax.axvline(0, color="k", linestyle="--", lw=0.8)
ax.set_xlabel("Latent Factor 1: Operational Workload", fontsize=9)
ax.set_ylabel("Latent Factor 2: Personal Well-Being", fontsize=9)
ax.set_title("Latent Factor Space Projection and Burnout Risk Identification", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_factor_biplot.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Survey Indicator", "Mean", "Std Dev", "Description", "Operational Domain"]
t1_rows = [
    [features[i], f"{X_df.iloc[:, i].mean():.2f}", f"{X_df.iloc[:, i].std():.2f}", "Quantitative employee metric", "Organizational metric"]
    for i in range(len(features))
]
show_and_save("Table 1 - Observed Productivity and Well-Being Survey Variables", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Observed Variable", "Unrotated F1", "Unrotated F2", "Rotated F1 (Workload)", "Rotated F2 (Well-Being)"]
t2_rows = [
    [features[i], f"{loadings_unrot[i, 0]:+.4f}", f"{loadings_unrot[i, 1]:+.4f}", f"{loadings_rot[i, 0]:+.4f}", f"{loadings_rot[i, 1]:+.4f}"]
    for i in range(len(features))
]
show_and_save("Table 2 - Unrotated vs Varimax Orthogonally Rotated Loadings", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Indicator", "Communality (h²)", "Uniqueness (u² = 1 - h²)", "Common Variance Ratio"]
t3_rows = [
    [features[i], f"{communalities[i]:.4f}", f"{noise_variance[i]:.4f}", f"{(communalities[i]/(communalities[i]+noise_variance[i]))*100:.1f}%"]
    for i in range(len(features))
]
show_and_save("Table 3 - Feature Communalities and Unique Variance Decomposition", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Latent Factor", "Sum of Squared Loadings", "Proportion of Explained Variance", "Dominant Variables"]
ss_f1 = np.sum(loadings_rot[:, 0]**2)
ss_f2 = np.sum(loadings_rot[:, 1]**2)
tot = ss_f1 + ss_f2
t4_rows = [
    ["Factor 1 (Workload)", f"{ss_f1:.4f}", f"{(ss_f1/tot)*100:.2f}%", "Work Hours, Task Volume, Overtime"],
    ["Factor 2 (Well-Being)", f"{ss_f2:.4f}", f"{(ss_f2/tot)*100:.2f}%", "Sleep Quality, Job Sat, Work-Life Balance"]
]
show_and_save("Table 4 - Latent Factor Variance Proportions and Feature Assignment", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Dimension", "Principal Component Analysis (PCA)", "Factor Analysis (FA)"]
t5_rows = [
    ["Core Objective", "Maximize total explained sample variance", "Explain pairwise shared feature covariances"],
    ["Noise Model", "No explicit error term", "Dedicated unique variance term psi_i per feature"],
    ["Interpretability", "Components are mathematical orthogonal axes", "Rotated factors correspond to latent construct entities"],
    ["Model Direction", "Linear combinations of observed features", "Generative: unobserved factors produce observed data"]
]
show_and_save("Table 5 - Comparative Theoretical Synthesis: PCA vs Factor Analysis", t5_headers, t5_rows, "table 5.png")
