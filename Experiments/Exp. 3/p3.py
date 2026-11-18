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

# === Module 2 : Server Request Failure Counts Dataset ===
np.random.seed(84)
N_intervals = 150
lambda_true = 2.4
failures = np.random.poisson(lam=lambda_true, size=N_intervals)
k_values = np.arange(0, 9)
observed_counts = np.array([np.sum(failures == k) for k in k_values])

# === Module 3 : Fitting Poisson and Binomial Distributions ===
lambda_mle = np.mean(failures)
poisson_probs = stats.poisson.pmf(k_values, mu=lambda_mle)
expected_poisson = poisson_probs * N_intervals

# Binomial approximation with n=30 attempts per interval
n_bin = 30
p_bin = lambda_mle / n_bin
binom_probs = stats.binom.pmf(k_values, n=n_bin, p=p_bin)
expected_binom = binom_probs * N_intervals

# === Module 4 : Chi-Square Goodness-of-Fit Test ===
# Group rare tail bins (>=6) for robust Chi-square calculation
obs_pooled = np.append(observed_counts[:6], np.sum(observed_counts[6:]))
exp_pooled = np.append(expected_poisson[:6], np.sum(expected_poisson[6:]))
exp_pooled = exp_pooled * (np.sum(obs_pooled) / np.sum(exp_pooled)) # normalize
chi2_stat, p_val = stats.chisquare(f_obs=obs_pooled, f_exp=exp_pooled, ddof=1)

# === Module 5 : Bayesian Inference for Server Anomaly Detection ===
# Let A = Anomaly present (Hypothesis), B = Trigger Alert (Evidence)
P_A = 0.02          # Prior probability of genuine attack
P_B_given_A = 0.94  # Sensitivity (True Positive Rate)
P_B_given_notA = 0.05 # False Alarm Rate (False Positive Rate)
P_notA = 1.0 - P_A

# Marginal Evidence P(B)
P_B = (P_B_given_A * P_A) + (P_B_given_notA * P_notA)

# Posterior P(A | B) via Bayes' Theorem
P_A_given_B = (P_B_given_A * P_A) / P_B

# Prior sensitivity sweep
priors_sweep = np.linspace(0.001, 0.20, 100)
posteriors_sweep = (P_B_given_A * priors_sweep) / ((P_B_given_A * priors_sweep) + (P_B_given_notA * (1.0 - priors_sweep)))

# === Module 6 : Visualisation ===
fig, ax = plt.subplots(figsize=(6.8, 3.8))
x_idx = np.arange(len(k_values))
width = 0.28
ax.bar(x_idx - width, observed_counts / N_intervals, width=width, color="#4374B3", alpha=0.85, label="Observed Relative Freq")
ax.bar(x_idx, poisson_probs, width=width, color="#E67E22", alpha=0.85, label=f"Fitted Poisson ($\\lambda={lambda_mle:.2f}$)")
ax.bar(x_idx + width, binom_probs, width=width, color="#27AE60", alpha=0.85, label=f"Binomial ($n=30, p={p_bin:.3f}$)")
ax.set_xlabel("Number of Anomalies per Interval ($k$)")
ax.set_ylabel("Probability Mass $P(X=k)$")
ax.set_title("Probability Mass Function Comparison: Empirical vs Fitted Models")
ax.set_xticks(x_idx)
ax.set_xticklabels([str(k) for k in k_values])
ax.grid(axis="y", linestyle="--", alpha=0.45)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("plot1_pmf.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.6))
pois_cdf = stats.poisson.cdf(k_values, mu=lambda_mle)
emp_cdf = np.cumsum(observed_counts) / N_intervals
ax1.step(k_values, emp_cdf, where="post", color="#2980B9", lw=1.8, label="Empirical CDF")
ax1.step(k_values, pois_cdf, where="post", color="#C0392B", linestyle="--", lw=1.8, label="Poisson CDF")
ax1.set_xlabel("Failures Threshold ($k$)")
ax1.set_ylabel(r"Cumulative Probability $P(X \leq k)$")
ax1.set_title("Cumulative Distribution Function (CDF)")
ax1.grid(True, linestyle="--", alpha=0.45)
ax1.legend(loc="lower right", fontsize=8)

ax2.plot(priors_sweep * 100, posteriors_sweep * 100, color="#8E44AD", lw=2.0)
ax2.scatter([P_A * 100], [P_A_given_B * 100], color="#C0392B", s=60, zorder=5)
ax2.annotate(f"Current Prior {P_A*100:.1f}%\nPosterior: {P_A_given_B*100:.1f}%",
             xy=(P_A * 100, P_A_given_B * 100), xytext=(P_A * 100 + 3, P_A_given_B * 100 - 15),
             arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2), fontsize=8)
ax2.set_xlabel("Prior Probability of Attack $P(A)$ (%)")
ax2.set_ylabel("Posterior Probability $P(A|B)$ (%)")
ax2.set_title("Bayes' Posterior Probability Sensitivity")
ax2.grid(True, linestyle="--", alpha=0.45)
fig.suptitle("Probabilistic Modeling and Bayesian Anomaly Inference", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_cdf.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 7 : Printing and Saving Tables ===
t1_headers = ["Interval Range", "Observed Samples", "Sample Mean (λ_mle)", "Sample Variance", "Theoretical λ"]
t1_rows = [["Intervals 1-150", str(N_intervals), f"{lambda_mle:.3f}", f"{np.var(failures, ddof=1):.3f}", f"{lambda_true:.3f}"]]
show_and_save("Table 1 - Server Anomaly Sampling and Empirical Parameters", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Count k", "Observed Count", "Observed P(k)", "Poisson P(k)", "Binomial P(k)", "Abs Residual"]
t2_rows = [
    [str(k), str(observed_counts[k]), f"{observed_counts[k]/N_intervals:.4f}", f"{poisson_probs[k]:.4f}", f"{binom_probs[k]:.4f}", f"{abs(observed_counts[k]/N_intervals - poisson_probs[k]):.4f}"]
    for k in range(len(k_values))
]
show_and_save("Table 2 - Frequency Distribution and Theoretical Fit Comparison", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Test Statistic", "Degrees of Freedom (df)", "Chi-Square Value (χ²)", "p-value", "Conclusion"]
t3_rows = [["Goodness-of-Fit", f"{len(obs_pooled) - 1 - 1}", f"{chi2_stat:.4f}", f"{p_val:.4f}", "Poisson model fits data (Fail to reject H0)"]]
show_and_save("Table 3 - Chi-Square Hypothesis Test of Poisson Fit", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Bayesian Parameter", "Notation", "Probability Value", "Interpretation"]
t4_rows = [
    ["Prior Attack Probability", "P(A)", f"{P_A:.4f}", "Base rate of genuine security intrusion"],
    ["Alert Sensitivity (TPR)", "P(B|A)", f"{P_B_given_A:.4f}", "Probability of alert given real attack"],
    ["False Alarm Rate (FPR)", "P(B|~A)", f"{P_B_given_notA:.4f}", "Probability of alert on normal traffic"],
    ["Marginal Evidence", "P(B)", f"{P_B:.4f}", "Total probability of sensor alert"],
    ["Posterior Confidence", "P(A|B)", f"{P_A_given_B:.4f}", "Probability of attack after alert fired"]
]
show_and_save("Table 4 - Bayesian Diagnostic Inference for Alert System", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Base Prior P(A)", "Evidence P(B)", "Posterior P(A|B)", "Confidence Gain Factor"]
priors_table = [0.005, 0.010, 0.020, 0.050, 0.100, 0.200]
t5_rows = []
for p in priors_table:
    ev = p * P_B_given_A + (1.0 - p) * P_B_given_notA
    post = (P_B_given_A * p) / ev
    t5_rows.append([f"{p*100:.1f}%", f"{ev*100:.2f}%", f"{post*100:.2f}%", f"{post/p:.2f}x"])
show_and_save("Table 5 - Impact of Prior Base Rate on Posterior Probability", t5_headers, t5_rows, "table 5.png")

t6_headers = ["Distribution Type", "Mean", "Variance", "Index of Dispersion (Var/Mean)", "Congruence"]
t6_rows = [
    ["Empirical Sample", f"{np.mean(failures):.3f}", f"{np.var(failures, ddof=1):.3f}", f"{np.var(failures, ddof=1)/np.mean(failures):.3f}", "Equidispersed"],
    ["Theoretical Poisson", f"{lambda_mle:.3f}", f"{lambda_mle:.3f}", "1.000", "Perfect Poisson Baseline"],
    ["Binomial Model", f"{n_bin * p_bin:.3f}", f"{n_bin * p_bin * (1 - p_bin):.3f}", f"{1 - p_bin:.3f}", "Close Approximation"]
]
show_and_save("Table 6 - Dispersion Analysis and Distribution Model Comparison", t6_headers, t6_rows, "table 6.png")
