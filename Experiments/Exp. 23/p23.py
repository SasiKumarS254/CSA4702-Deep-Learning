# === Module 1 : Importing the Required Packages ===
import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.d_separation import is_d_separator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD

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

# === Module 2 : Clinical Diagnostic Bayesian Network Topology ===
model = DiscreteBayesianNetwork([
    ("Smoking", "LungCancer"),
    ("Smoking", "Bronchitis"),
    ("LungCancer", "XRay"),
    ("LungCancer", "Dyspnea"),
    ("Bronchitis", "Dyspnea")
])

# === Module 3 : Parameterization with Tabular CPDs ===
# P(Smoking)
cpd_smoking = TabularCPD(variable="Smoking", variable_card=2, values=[[0.70], [0.30]],
                         state_names={"Smoking": ["No", "Yes"]})

# P(LungCancer | Smoking)
cpd_cancer = TabularCPD(variable="LungCancer", variable_card=2,
                        values=[[0.98, 0.88], [0.02, 0.12]],
                        evidence=["Smoking"], evidence_card=[2],
                        state_names={"LungCancer": ["Absence", "Presence"], "Smoking": ["No", "Yes"]})

# P(Bronchitis | Smoking)
cpd_bronch = TabularCPD(variable="Bronchitis", variable_card=2,
                        values=[[0.75, 0.40], [0.25, 0.60]],
                        evidence=["Smoking"], evidence_card=[2],
                        state_names={"Bronchitis": ["Absence", "Presence"], "Smoking": ["No", "Yes"]})

# P(XRay | LungCancer)
cpd_xray = TabularCPD(variable="XRay", variable_card=2,
                      values=[[0.95, 0.15], [0.05, 0.85]],
                      evidence=["LungCancer"], evidence_card=[2],
                      state_names={"XRay": ["Normal", "Abnormal"], "LungCancer": ["Absence", "Presence"]})

# P(Dyspnea | LungCancer, Bronchitis)
cpd_dyspnea = TabularCPD(variable="Dyspnea", variable_card=2,
                         values=[[0.90, 0.30, 0.25, 0.05],
                                 [0.10, 0.70, 0.75, 0.95]],
                         evidence=["LungCancer", "Bronchitis"], evidence_card=[2, 2],
                         state_names={"Dyspnea": ["No", "Yes"],
                                      "LungCancer": ["Absence", "Presence"],
                                      "Bronchitis": ["Absence", "Presence"]})

model.add_cpds(cpd_smoking, cpd_cancer, cpd_bronch, cpd_xray, cpd_dyspnea)
model_valid = model.check_model()

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(7.8, 5.0))
G = nx.DiGraph(model.edges())

pos = {
    "Smoking": (2.0, 3.5),
    "LungCancer": (1.0, 2.2),
    "Bronchitis": (3.0, 2.2),
    "XRay": (0.5, 0.8),
    "Dyspnea": (2.0, 0.8)
}

node_colors = {
    "Smoking": "#3498DB",      # Risk factor
    "LungCancer": "#E74C3C",   # Pathological condition
    "Bronchitis": "#E67E22",   # Pathological condition
    "XRay": "#2ECC71",         # Diagnostic observation
    "Dyspnea": "#9B59B6"       # Clinical symptom
}
colors_ordered = [node_colors[n] for n in G.nodes()]

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors_ordered, node_size=2400, edgecolors="black", linewidths=1.5)
nx.draw_networkx_labels(G, pos, ax=ax, font_family="serif", font_size=9, font_weight="bold", font_color="white")
nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle="->", arrowsize=20, edge_color="#2C3E50", width=2.0)

ax.set_title("Directed Acyclic Graph (DAG) for Clinical Pulmonary Diagnosis", fontsize=10.5)
ax.set_xlim(-0.2, 3.8)
ax.set_ylim(0.2, 4.2)
ax.axis("off")
fig.tight_layout()
fig.savefig("plot1_bayesian_dag.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.8))
# CPD for Dyspnea
dysp_table = cpd_dyspnea.values.reshape(2, 4)
im1 = ax1.imshow(dysp_table, cmap="Blues", vmin=0, vmax=1.0)
ax1.set_xticks(range(4))
ax1.set_xticklabels(["C-,B-", "C+,B-", "C-,B+", "C+,B+"], fontsize=8)
ax1.set_yticks([0, 1])
ax1.set_yticklabels(["Dyspnea: No", "Dyspnea: Yes"], fontsize=8.5)
for i in range(2):
    for j in range(4):
        ax1.text(j, i, f"{dysp_table[i, j]:.2f}", ha="center", va="center", color="white" if dysp_table[i, j] > 0.5 else "black", fontsize=9)
ax1.set_title("P(Dyspnea | LungCancer, Bronchitis)", fontsize=9.5)
fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# CPD for XRay
xray_table = cpd_xray.values
im2 = ax2.imshow(xray_table, cmap="Greens", vmin=0, vmax=1.0)
ax2.set_xticks([0, 1])
ax2.set_xticklabels(["Cancer: Absence", "Cancer: Presence"], fontsize=8.5)
ax2.set_yticks([0, 1])
ax2.set_yticklabels(["XRay: Normal", "XRay: Abnormal"], fontsize=8.5)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f"{xray_table[i, j]:.2f}", ha="center", va="center", color="white" if xray_table[i, j] > 0.5 else "black", fontsize=9)
ax2.set_title("P(XRay | LungCancer)", fontsize=9.5)
fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
fig.suptitle("Conditional Probability Distributions Across Child Nodes", fontsize=10.5)
fig.tight_layout()
fig.savefig("plot2_cpd_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Node Identifier", "Node Type", "Parent Nodes", "Child Nodes", "Cardinality"]
t1_rows = [
    ["Smoking", "Risk Factor / Root", "None", "LungCancer, Bronchitis", "Binary {No, Yes}"],
    ["LungCancer", "Latent Pathology", "Smoking", "XRay, Dyspnea", "Binary {Absence, Presence}"],
    ["Bronchitis", "Latent Pathology", "Smoking", "Dyspnea", "Binary {Absence, Presence}"],
    ["XRay", "Diagnostic Observation", "LungCancer", "None (Leaf)", "Binary {Normal, Abnormal}"],
    ["Dyspnea", "Clinical Symptom", "LungCancer, Bronchitis", "None (Leaf)", "Binary {No, Yes}"]
]
show_and_save("Table 1 - Bayesian Network Graph Topology and Nodal Hierarchy", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Conditioned Node", "Conditioning Parents", "CPD Probability Matrix", "Consistency Check"]
t2_rows = [
    ["Smoking", "None (Prior)", "P(No)=0.70, P(Yes)=0.30", "Sum = 1.0 (Valid)"],
    ["LungCancer", "Smoking", "P(C+|S-)=0.02, P(C+|S+)=0.12", "Columns sum to 1.0"],
    ["Bronchitis", "Smoking", "P(B+|S-)=0.25, P(B+|S+)=0.60", "Columns sum to 1.0"],
    ["XRay", "LungCancer", "P(X+|C-)=0.05, P(X+|C+)=0.85", "Columns sum to 1.0"],
    ["Dyspnea", "LungCancer, Bronchitis", "4 parent combinations: P(D+|C,B)", "Columns sum to 1.0"]
]
show_and_save("Table 2 - Tabular Conditional Probability Distributions (CPD) Inventory", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Query Variables", "Evidence Given", "Active Trail Present", "Conditional Independence Status"]
# d-separation checks
is_indep_1 = is_d_separator(model, {"XRay"}, {"Bronchitis"}, {"LungCancer"})
is_indep_2 = is_d_separator(model, {"LungCancer"}, {"Bronchitis"}, {"Smoking"})
is_indep_3 = is_d_separator(model, {"LungCancer"}, {"Bronchitis"}, {"Dyspnea"}) # Collider active
t3_rows = [
    ["XRay _|_ Bronchitis", "LungCancer (Observed)", "No (Blocked by L)", "Conditionally Independent" if is_indep_1 else "Dependent"],
    ["LungCancer _|_ Bronchitis", "Smoking (Observed)", "No (Common cause blocked)", "Conditionally Independent" if is_indep_2 else "Dependent"],
    ["LungCancer _|_ Bronchitis", "Dyspnea (Collider)", "Yes (Explaining Away)", "Conditionally Dependent" if not is_indep_3 else "Independent"]
]
show_and_save("Table 3 - d-Separation and Conditional Independence Verifications", t3_headers, t3_rows, "table 3.png")

mb_cancer = model.get_markov_blanket("LungCancer")
t4_headers = ["Target Node", "Markov Blanket Components", "Specific Nodes in Blanket", "Information Shielding Property"]
t4_rows = [
    ["LungCancer", "Parents + Children + Spouses", f"{', '.join(sorted(mb_cancer))}", "Conditionally independent of all other nodes given MB"]
]
show_and_save("Table 4 - Markov Blanket Isolation and Local Independence Scope", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Structural Criterion", "Analytical Requirement", "Empirical Model Status", "Verification Result"]
t5_rows = [
    ["Acyclicity", "No directed cycles in graph", "Directed Acyclic Graph (DAG)", "Verified"],
    ["Global Probability Factorization", "P(V) = prod P(v_i | Pa(v_i))", "Fully factorized joint distribution", "Verified"],
    ["CPD Normalization", "sum_{x} P(X=x | Pa(X)) = 1", "All column sums equal 1.000", "Verified"],
    ["pgmpy Verification Suite", "model.check_model()", "True", "Verified Valid PGM"]
]
show_and_save("Table 5 - Bayesian Network Architectural and Axiomatic Validation", t5_headers, t5_rows, "table 5.png")
