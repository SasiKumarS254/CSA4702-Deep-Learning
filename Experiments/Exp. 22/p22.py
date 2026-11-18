# === Module 1 : Importing the Required Packages ===
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

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

# === Module 2 : Independent Source Signal Generation ===
np.random.seed(42)
N_samples = 1200
time_axis = np.linspace(0, 8, N_samples)

# 3 non-Gaussian independent sources
s1 = np.sin(2.5 * time_axis)                      # Sinusoidal periodic signal
s2 = np.sign(np.sin(3.5 * time_axis))             # Square wave pulse
s3 = signal.sawtooth(2.0 * np.pi * time_axis * 0.9) # Sawtooth wave

S = np.c_[s1, s2, s3]
S += 0.15 * np.random.normal(size=S.shape)        # Ambient sensor noise
S /= S.std(axis=0)                                # Standardize sources

# === Module 3 : Linear Mixing with Unknown Mixing Matrix A ===
A = np.array([
    [1.0,  0.8,  0.5],
    [0.6,  1.2,  0.3],
    [0.4,  0.5,  1.1]
])
X_mixed = np.dot(S, A.T)

# === Module 4 : Blind Source Separation via FastICA ===
ica = FastICA(n_components=3, random_state=42, whiten="arbitrary-variance")
S_recovered = ica.fit_transform(X_mixed)
W_unmixing = ica.components_

# Align sign and permutation for rigorous comparison
corr_matrix = np.corrcoef(S.T, S_recovered.T)[:3, 3:]
best_permutation = np.argmax(np.abs(corr_matrix), axis=1)

aligned_recovered = np.zeros_like(S_recovered)
for i, j in enumerate(best_permutation):
    sign = np.sign(corr_matrix[i, j])
    aligned_recovered[:, i] = sign * S_recovered[:, j]

aligned_corr = [np.corrcoef(S[:, i], aligned_recovered[:, i])[0, 1] for i in range(3)]

# === Module 5 : Visualisation ===
fig, axes = plt.subplots(3, 3, figsize=(10.5, 6.2), sharex=True)
titles_col = ["Ground Truth Sources S", "Linear Mixed Sensors X", "FastICA Recovered Components"]
colors = ["#2980B9", "#E67E22", "#27AE60"]

for i in range(3):
    axes[i, 0].plot(time_axis[:250], S[:250, i], color=colors[i], lw=1.6)
    axes[i, 0].set_ylabel(f"Source {i+1}", fontsize=8.5)
    axes[i, 0].grid(True, linestyle="--", alpha=0.45)
    
    axes[i, 1].plot(time_axis[:250], X_mixed[:250, i], color="#34495E", lw=1.4)
    axes[i, 1].set_ylabel(f"Sensor {i+1}", fontsize=8.5)
    axes[i, 1].grid(True, linestyle="--", alpha=0.45)
    
    axes[i, 2].plot(time_axis[:250], aligned_recovered[:250, i], color=colors[i], lw=1.6, linestyle="--")
    axes[i, 2].set_ylabel(f"Recovered {i+1}", fontsize=8.5)
    axes[i, 2].grid(True, linestyle="--", alpha=0.45)

for j in range(3):
    axes[0, j].set_title(titles_col[j], fontsize=10)
    axes[2, j].set_xlabel("Time Horizon (s)", fontsize=8.5)

fig.suptitle("Blind Source Separation via FastICA: Sources, Mixtures and Reconstructions", fontsize=11)
fig.tight_layout()
fig.savefig("plot1_source_separation.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
kurt_sources = stats.kurtosis(S)
kurt_mixed = stats.kurtosis(X_mixed)
kurt_recovered = stats.kurtosis(aligned_recovered)

x_pos = np.arange(3)
w = 0.28
ax1.bar(x_pos - w, kurt_sources, width=w, color="#2980B9", label="Original Sources S")
ax1.bar(x_pos, kurt_mixed, width=w, color="#7F8C8D", label="Mixed Signals X (More Gaussian)")
ax1.bar(x_pos + w, kurt_recovered, width=w, color="#27AE60", label="Recovered Sources S_hat")
ax1.set_xticks(x_pos)
ax1.set_xticklabels(["Signal 1 (Sine)", "Signal 2 (Square)", "Signal 3 (Sawtooth)"], fontsize=8.5)
ax1.set_ylabel("Excess Kurtosis", fontsize=8.5)
ax1.set_title("Central Limit Theorem: Mixtures Tend Towards Gaussianity", fontsize=9.5)
ax1.grid(axis="y", linestyle="--", alpha=0.45)
ax1.legend(fontsize=8)

ax2.hist(S[:, 1], bins=30, density=True, color="#2980B9", alpha=0.6, label="Source 2 (Sub-Gaussian)")
ax2.hist(X_mixed[:, 1], bins=30, density=True, color="#E74C3C", alpha=0.5, label="Mixed Sensor 2 (Bell Curve)")
ax2.set_xlabel("Signal Amplitude", fontsize=8.5)
ax2.set_ylabel("Probability Density", fontsize=8.5)
ax2.set_title("Distribution Shift: Highly Non-Gaussian vs Gaussian Mixture", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.45)
ax2.legend(fontsize=8)
fig.suptitle("Information-Theoretic Non-Gaussianity and Kurtosis Diagnostics", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_kurtosis_distribution.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Source ID", "Waveform Type", "Mathematical Generation", "True Kurtosis", "Gaussianity Regime"]
t1_rows = [
    ["Source S1", "Harmonic Sinusoid", "sin(2.5 t)", f"{kurt_sources[0]:+.3f}", "Sub-Gaussian (Platykurtic)"],
    ["Source S2", "Square Wave Pulses", "sign(sin(3.5 t))", f"{kurt_sources[1]:+.3f}", "Strongly Sub-Gaussian"],
    ["Source S3", "Sawtooth Ramp", "sawtooth(1.8 pi t)", f"{kurt_sources[2]:+.3f}", "Sub-Gaussian (Platykurtic)"]
]
show_and_save("Table 1 - Independent Source Signal Specifications", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Matrix", "Row 1 [c1, c2, c3]", "Row 2 [c1, c2, c3]", "Row 3 [c1, c2, c3]", "Condition Number"]
cond_A = np.linalg.cond(A)
cond_W = np.linalg.cond(W_unmixing)
t2_rows = [
    ["Mixing Matrix A", f"[{A[0,0]:.2f}, {A[0,1]:.2f}, {A[0,2]:.2f}]", f"[{A[1,0]:.2f}, {A[1,1]:.2f}, {A[1,2]:.2f}]", f"[{A[2,0]:.2f}, {A[2,1]:.2f}, {A[2,2]:.2f}]", f"{cond_A:.3f}"],
    ["Unmixing Matrix W", f"[{W_unmixing[0,0]:.2f}, {W_unmixing[0,1]:.2f}, {W_unmixing[0,2]:.2f}]", f"[{W_unmixing[1,0]:.2f}, {W_unmixing[1,1]:.2f}, {W_unmixing[1,2]:.2f}]", f"[{W_unmixing[2,0]:.2f}, {W_unmixing[2,1]:.2f}, {W_unmixing[2,2]:.2f}]", f"{cond_W:.3f}"]
]
show_and_save("Table 2 - Linear Mixing System and Estimated Unmixing Matrix", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Signal Entity", "Source Kurtosis", "Mixed Mixture Kurtosis", "Recovered Kurtosis", "Central Limit Effect"]
t3_rows = [
    ["Channel 1", f"{kurt_sources[0]:+.4f}", f"{kurt_mixed[0]:+.4f}", f"{kurt_recovered[0]:+.4f}", "Mixture shifts closer to 0 (Gaussian)"],
    ["Channel 2", f"{kurt_sources[1]:+.4f}", f"{kurt_mixed[1]:+.4f}", f"{kurt_recovered[1]:+.4f}", "Mixture shifts closer to 0 (Gaussian)"],
    ["Channel 3", f"{kurt_sources[2]:+.4f}", f"{kurt_mixed[2]:+.4f}", f"{kurt_recovered[2]:+.4f}", "Mixture shifts closer to 0 (Gaussian)"]
]
show_and_save("Table 3 - Non-Gaussianity (Kurtosis) Tracking Across Transformations", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Component Pair", "Pearson Correlation r", "Determination r²", "Reconstruction Fidelity"]
t4_rows = [
    [f"Source S{i+1} <-> Recovered S_hat{i+1}", f"{aligned_corr[i]:.4f}", f"{aligned_corr[i]**2:.4f}", "Near-Perfect Recovery (> 0.98)" if aligned_corr[i] > 0.98 else "High Recovery"]
    for i in range(3)
]
show_and_save("Table 4 - Source Reconstruction Correlation and Signal Separation Fidelity", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Criterion", "Principal Component Analysis (PCA)", "Independent Component Analysis (ICA)"]
t5_rows = [
    ["Independence Assumption", "Second-order uncorrelatedness (Cov = 0)", "Higher-order statistical independence (Joint P = prod P_i)"],
    ["Component Orientation", "Orthogonal axes maximizing variance", "Non-orthogonal axes maximizing non-Gaussianity"],
    ["Underlying Distribution", "Assumes Gaussian or distribution-agnostic", "Requires non-Gaussian latent components"],
    ["Typical Application", "Dimensionality reduction & data compression", "Blind source separation, artifact rejection (EEG/audio)"]
]
show_and_save("Table 5 - Comparative Theoretical Synthesis: PCA vs ICA", t5_headers, t5_rows, "table 5.png")
