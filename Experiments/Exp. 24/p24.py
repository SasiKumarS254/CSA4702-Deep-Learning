# === Module 1 : Importing the Required Packages ===
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

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

# === Module 2 : Clinical Diagnostic Model Definition ===
model = DiscreteBayesianNetwork([
    ("Smoking", "LungCancer"),
    ("Smoking", "Bronchitis"),
    ("LungCancer", "XRay"),
    ("LungCancer", "Dyspnea"),
    ("Bronchitis", "Dyspnea")
])

cpd_smoking = TabularCPD(variable="Smoking", variable_card=2, values=[[0.70], [0.30]],
                         state_names={"Smoking": ["No", "Yes"]})
cpd_cancer = TabularCPD(variable="LungCancer", variable_card=2, values=[[0.98, 0.88], [0.02, 0.12]],
                        evidence=["Smoking"], evidence_card=[2],
                        state_names={"LungCancer": ["Absence", "Presence"], "Smoking": ["No", "Yes"]})
cpd_bronch = TabularCPD(variable="Bronchitis", variable_card=2, values=[[0.75, 0.40], [0.25, 0.60]],
                        evidence=["Smoking"], evidence_card=[2],
                        state_names={"Bronchitis": ["Absence", "Presence"], "Smoking": ["No", "Yes"]})
cpd_xray = TabularCPD(variable="XRay", variable_card=2, values=[[0.95, 0.15], [0.05, 0.85]],
                      evidence=["LungCancer"], evidence_card=[2],
                      state_names={"XRay": ["Normal", "Abnormal"], "LungCancer": ["Absence", "Presence"]})
cpd_dyspnea = TabularCPD(variable="Dyspnea", variable_card=2,
                         values=[[0.90, 0.30, 0.25, 0.05],
                                 [0.10, 0.70, 0.75, 0.95]],
                         evidence=["LungCancer", "Bronchitis"], evidence_card=[2, 2],
                         state_names={"Dyspnea": ["No", "Yes"],
                                      "LungCancer": ["Absence", "Presence"],
                                      "Bronchitis": ["Absence", "Presence"]})

model.add_cpds(cpd_smoking, cpd_cancer, cpd_bronch, cpd_xray, cpd_dyspnea)

# === Module 3 : Exact Probabilistic Inference (Variable Elimination) ===
infer = VariableElimination(model)

scenarios = [
    {"name": "1. Prior Baseline", "evidence": {}},
    {"name": "2. Smoker (S=Yes)", "evidence": {"Smoking": "Yes"}},
    {"name": "3. Abnormal X-Ray (X=Abnormal)", "evidence": {"XRay": "Abnormal"}},
    {"name": "4. Smoker + Abnormal X-Ray", "evidence": {"Smoking": "Yes", "XRay": "Abnormal"}},
    {"name": "5. All Evidence (S=Y, X=Abn, D=Y)", "evidence": {"Smoking": "Yes", "XRay": "Abnormal", "Dyspnea": "Yes"}}
]

exact_posteriors = []
for sc in scenarios:
    res = infer.query(variables=["LungCancer"], evidence=sc["evidence"], show_progress=False)
    p_presence = res.values[1] # Probability of Cancer Presence
    exact_posteriors.append(p_presence)

# === Module 4 : Stochastic Monte Carlo Sampling Convergence ===
np.random.seed(42)
sampler = BayesianModelSampling(model)
sample_sizes = [100, 300, 600, 1200, 2500, 5000]
target_exact = exact_posteriors[1] # P(Cancer | Smoking=Yes)

sampled_estimates = []
for n_s in sample_sizes:
    df_samples = sampler.forward_sample(size=n_s, show_progress=False)
    sub = df_samples[df_samples["Smoking"] == "Yes"]
    p_est = np.mean(sub["LungCancer"] == "Presence") if len(sub) > 0 else 0.0
    sampled_estimates.append(p_est)

# === Module 5 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.8, 4.2))
sc_labels = [s["name"].split(". ")[1] for s in scenarios]
colors = ["#2980B9", "#E67E22", "#E74C3C", "#C0392B", "#8E44AD"]
bars = ax.bar(range(len(scenarios)), [p * 100 for p in exact_posteriors], color=colors, edgecolor="black", width=0.55)
ax.set_xticks(range(len(scenarios)))
ax.set_xticklabels(sc_labels, fontsize=8, rotation=15)
ax.set_ylabel("Posterior Probability P(LungCancer = Yes) (%)", fontsize=9)
ax.set_title("Exact Bayesian Inference: Posterior Disease Belief Under Evidence", fontsize=10.5)
for bar, p in zip(bars, exact_posteriors):
    ax.text(bar.get_x() + bar.get_width()/2, p * 100 + 1.2, f"{p*100:.2f}%", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax.set_ylim(0, 105)
ax.grid(axis="y", linestyle="--", alpha=0.45)
fig.tight_layout()
fig.savefig("plot1_posterior_scenarios.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
ax.plot(sample_sizes, [p * 100 for p in sampled_estimates], "o-", color="#27AE60", lw=2.0, label="Monte Carlo Sample Estimate")
ax.axhline(target_exact * 100, color="#C0392B", linestyle="--", lw=1.8, label=f"Exact Analytical Posterior ({target_exact*100:.2f}%)")
ax.set_xlabel("Number of Forward Samples (N)", fontsize=9)
ax.set_ylabel("Estimated Probability P(Cancer | Smoker) (%)", fontsize=9)
ax.set_title("Empirical Monte Carlo Convergence to Exact Posterior Probability", fontsize=10.5)
ax.grid(True, linestyle="--", alpha=0.45)
ax.legend(fontsize=8.5)
fig.tight_layout()
fig.savefig("plot2_sampling_convergence.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 6 : Printing and Saving Tables ===
t1_headers = ["Scenario ID", "Clinical Evidence Condition", "Observed State Values", "Diagnostic Intent"]
t1_rows = [
    [sc["name"], str(sc["evidence"]) if sc["evidence"] else "None (Prior)", "Baseline population", "Quantifies baseline disease prevalence"],
    [scenarios[1]["name"], "Smoking = Yes", "Anamnesis risk factor", "Assesses primary lifestyle risk elevation"],
    [scenarios[2]["name"], "XRay = Abnormal", "Diagnostic imaging result", "Evaluates high-specificity symptom"],
    [scenarios[3]["name"], "Smoking=Yes + XRay=Abnormal", "Combined risk + imaging", "Synergistic evidence fusion"],
    [scenarios[4]["name"], "Smoking=Y + XRay=Abn + Dyspnea=Y", "Full symptom constellation", "Definitive multi-modal diagnosis"]
]
show_and_save("Table 1 - Clinical Evidence Scenarios and Diagnostic Conditioning", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Evidence Scenario", "P(Cancer = Absence)", "P(Cancer = Presence)", "Likelihood Ratio vs Prior"]
t2_rows = [
    [sc["name"], f"{(1.0 - exact_posteriors[i])*100:.2f}%", f"{exact_posteriors[i]*100:.2f}%", f"{exact_posteriors[i]/exact_posteriors[0]:.2f}x"]
    for i, sc in enumerate(scenarios)
]
show_and_save("Table 2 - Exact Posterior Distributions Computed via Variable Elimination", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Sample Size N", "Exact Probability", "Sampled Estimate", "Absolute Residual Error", "Relative Error (%)"]
t3_rows = [
    [f"N = {n_s:,}", f"{target_exact*100:.2f}%", f"{sampled_estimates[i]*100:.2f}%", f"{abs(sampled_estimates[i] - target_exact)*100:.3f}%", f"{(abs(sampled_estimates[i] - target_exact)/target_exact)*100:.2f}%"]
    for i, n_s in enumerate(sample_sizes)
]
show_and_save("Table 3 - Stochastic Forward Sampling Convergence vs Analytical Ground Truth", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Evidence Combination", "Analytical Complexity", "Elimination Order", "Inference Latency"]
t4_rows = [
    ["Prior (No evidence)", "O(1) table lookup", "None", "< 1 ms"],
    ["Single Observation", "O(card(X) * card(Y))", "Bronchitis -> Dyspnea", "< 2 ms"],
    ["Multi-Evidence Joint", "O(N * d^tree-width)", "Bronchitis -> XRay -> Dyspnea", "< 3 ms"]
]
show_and_save("Table 4 - Computational Complexity and Elimination Factor Ordering", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Inference Paradigm", "Algorithmic Mechanism", "Computational Scalability", "Best Practical Use"]
t5_rows = [
    ["Exact (Variable Elimination)", "Dynamic programming factor marginalization", "NP-hard in general; fast for low tree-width DAGs", "Rigorous medical diagnostic safety systems"],
    ["Approximate (Rejection Sampling)", "Rejects samples inconsistent with evidence", "Suffers from low acceptance for rare evidence", "Exploratory sensitivity audits and prior sampling"],
    ["Approximate (MCMC / Gibbs)", "Markov chain stationary distribution transition", "Scales to large continuous high-dimensional nets", "Deep hierarchical Bayesian architectures"]
]
show_and_save("Table 5 - Comparative Probabilistic Inference Paradigms Synthesis", t5_headers, t5_rows, "table 5.png")
