# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

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

# === Module 2 : Photovoltaic Solar Power Dataset ===
np.random.seed(64)
N = 90
X = np.sort(np.random.uniform(0.5, 4.8, N))
y_clean = 2.4 * np.sin(1.6 * X) + 0.85 * X
noise = np.random.normal(0, 0.42, N)
y = y_clean + noise

X_2d = X[:, np.newaxis]

# === Module 3 : Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_2d, y, test_size=0.30, random_state=42)

# === Module 4 : Defining the Competing Models ===
degrees = [1, 4, 14]
model_names = ["Underfitting (Degree 1)", "Optimal Fit (Degree 4)", "Overfitting (Degree 14)"]
models = [make_pipeline(PolynomialFeatures(d), LinearRegression()) for d in degrees]

train_errors, test_errors = [], []
train_r2, test_r2 = [], []

for m in models:
    m.fit(X_train, y_train)
    y_tr_pred = m.predict(X_train)
    y_te_pred = m.predict(X_test)
    train_errors.append(mean_squared_error(y_train, y_tr_pred))
    test_errors.append(mean_squared_error(y_test, y_te_pred))
    train_r2.append(r2_score(y_train, y_tr_pred))
    test_r2.append(r2_score(y_test, y_te_pred))

# === Module 5 : 5-Fold Cross Validation ===
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_mean = []
cv_scores_std = []

for m in models:
    scores = -cross_val_score(m, X_train, y_train, scoring="neg_mean_squared_error", cv=kf)
    cv_scores_mean.append(np.mean(scores))
    cv_scores_std.append(np.std(scores))

# === Module 6 : Model Complexity Sweep (Degrees 1 to 12) ===
sweep_degrees = list(range(1, 13))
sweep_train_mse = []
sweep_test_mse = []

for d in sweep_degrees:
    pipe = make_pipeline(PolynomialFeatures(d), LinearRegression())
    pipe.fit(X_train, y_train)
    sweep_train_mse.append(mean_squared_error(y_train, pipe.predict(X_train)))
    sweep_test_mse.append(mean_squared_error(y_test, pipe.predict(X_test)))

# === Module 7 : Visualisation ===
x_dense = np.linspace(0.5, 4.8, 300)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.scatter(X_train, y_train, color="#2C3E50", alpha=0.7, s=32, label="Train Data (70%)")
ax.scatter(X_test, y_test, color="#E74C3C", alpha=0.8, s=40, marker="s", label="Test Data (30%)")
ax.plot(x_dense, 2.4 * np.sin(1.6 * x_dense) + 0.85 * x_dense, "k--", lw=1.2, label="Ground Truth $f(x)$")

colors = ["#2980B9", "#27AE60", "#8E44AD"]
for m, name, c in zip(models, model_names, colors):
    ax.plot(x_dense, m.predict(x_dense), color=c, lw=2.0, label=name)

ax.set_ylim(-2, 7)
ax.set_xlabel("Solar Irradiance Index ($x$)", fontsize=9.5)
ax.set_ylabel("Power Output ($kW$)", fontsize=9.5)
ax.set_title("Model Fits: Underfitting, Optimal Regularity and Overfitting", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig("plot1_model_errors.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.plot(sweep_degrees, sweep_train_mse, "o-", color="#2980B9", lw=1.8, label="Train MSE")
ax.plot(sweep_degrees, sweep_test_mse, "s-", color="#C0392B", lw=1.8, label="Test MSE")
ax.axvspan(3.5, 4.5, color="#2ECC71", alpha=0.18, label="Optimal Complexity Zone")
ax.set_yscale("log")
ax.set_xlabel("Polynomial Degree (Model Complexity)", fontsize=9.5)
ax.set_ylabel("Mean Squared Error (Log Scale)", fontsize=9.5)
ax.set_title("Bias-Variance Dilemma: Generalization Curve", fontsize=10.5)
ax.set_xticks(sweep_degrees)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(loc="upper right", fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_complexity_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 8 : Printing and Saving Tables ===
t1_headers = ["Dataset Property", "Parameter Specification", "Numerical Value"]
t1_rows = [
    ["Total Samples", "N = N_train + N_test", f"{N} instances"],
    ["Train-Test Partition", "70% Training / 30% Holdout Test", f"{len(X_train)} Train / {len(X_test)} Test"],
    ["Domain Range", "[x_min, x_max]", "[0.50, 4.80]"],
    ["True Generative Function", "y = 2.4 sin(1.6x) + 0.85x", "Nonlinear sinusoidal drift"],
    ["Noise Variance", "sigma^2 (Gaussian)", "0.42^2 = 0.1764"]
]
show_and_save("Table 1 - Photovoltaic Solar Power Dataset Configuration", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Model Identifier", "Hypothesis Space", "Degrees of Freedom", "Complexity Regime"]
t2_rows = [
    ["Degree 1", "y = w1*x + b", "2 parameters", "High Bias (Underfitting)"],
    ["Degree 4", "y = sum_{j=1}^4 w_j*x^j + b", "5 parameters", "Optimal Balance (Low Bias + Low Var)"],
    ["Degree 14", "y = sum_{j=1}^14 w_j*x^j + b", "15 parameters", "High Variance (Overfitting)"]
]
show_and_save("Table 2 - Model Architectures and Complexity Regimes", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Model", "Train MSE", "Test MSE", "Train R²", "Test R²"]
t3_rows = [
    [model_names[i], f"{train_errors[i]:.4f}", f"{test_errors[i]:.4f}", f"{train_r2[i]:.4f}", f"{test_r2[i]:.4f}"]
    for i in range(3)
]
show_and_save("Table 3 - Training and Testing Performance Metrics", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Model", "CV Fold 1-5 Mean MSE", "CV Std Dev", "Reliability Index"]
t4_rows = [
    [model_names[i], f"{cv_scores_mean[i]:.4f}", f"{cv_scores_std[i]:.4f}", "Stable" if cv_scores_std[i] < 0.2 else "Highly Volatile"]
    for i in range(3)
]
show_and_save("Table 4 - 5-Fold Cross-Validation Generalization Scores", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Model", "Train-Test MSE Gap", "Ratio (Test/Train MSE)", "Generalization Diagnostic"]
t5_rows = [
    [model_names[0], f"{test_errors[0] - train_errors[0]:.4f}", f"{test_errors[0]/train_errors[0]:.2f}x", "Underfit (High error on both)"],
    [model_names[1], f"{test_errors[1] - train_errors[1]:.4f}", f"{test_errors[1]/train_errors[1]:.2f}x", "Optimal Generalization"],
    [model_names[2], f"{test_errors[2] - train_errors[2]:.4f}", f"{test_errors[2]/train_errors[2]:.2f}x", "Severe Overfitting (Memorization)"]
]
show_and_save("Table 5 - Generalization Gap and Diagnostic Audit", t5_headers, t5_rows, "table 5.png")

t6_headers = ["Degree", "Train MSE", "Test MSE", "Overfitting Indicator (Gap)"]
t6_rows = [
    [str(d), f"{sweep_train_mse[d-1]:.4f}", f"{sweep_test_mse[d-1]:.4f}", f"{sweep_test_mse[d-1] - sweep_train_mse[d-1]:+.4f}"]
    for d in sweep_degrees
]
show_and_save("Table 6 - Comprehensive Polynomial Complexity Sweep", t6_headers, t6_rows, "table 6.png")

t7_headers = ["Criterion", "Degree 1 (Underfitting)", "Degree 4 (Optimal)", "Degree 14 (Overfitting)"]
t7_rows = [
    ["Bias Component", "Very High", "Low", "Extremely Low"],
    ["Variance Component", "Very Low", "Low", "Extremely High"],
    ["Noise Sensitivity", "Immune to noise", "Robust filter", "Fits noise artifacts"],
    ["Deployment Recommendation", "Reject (Under-parameterized)", "Approved for production", "Reject (Fragile extrapolate)"]
]
show_and_save("Table 7 - Bias-Variance Comparative Diagnostic Synthesis", t7_headers, t7_rows, "table 7.png")
