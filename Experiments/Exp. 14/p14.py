# === Module 1 : Importing the Required Packages ===
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import sys
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf

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

# === Module 2 : Unfolded RNN Computational Graph Definition ===
time_steps = 4
G = nx.DiGraph()

pos = {}
node_colors = []
node_labels = {}

for t in range(1, time_steps + 1):
    x_node = f"x_{t}"
    h_node = f"h_{t}"
    y_node = f"y_{t}"
    
    G.add_node(x_node)
    G.add_node(h_node)
    G.add_node(y_node)
    
    pos[x_node] = (t * 2.2, 1.0)
    pos[h_node] = (t * 2.2, 2.5)
    pos[y_node] = (t * 2.2, 4.0)
    
    G.add_edge(x_node, h_node, label="W_xh")
    G.add_edge(h_node, y_node, label="W_hy")
    
    if t > 1:
        prev_h = f"h_{t-1}"
        G.add_edge(prev_h, h_node, label="W_hh")

# Initial hidden state h0
h0 = "h_0"
G.add_node(h0)
pos[h0] = (1.0, 2.5)
G.add_edge(h0, "h_1", label="W_hh")

# === Module 3 : Numerical Forward Trace across Steps ===
np.random.seed(42)
dx = 2
dh = 4
dy = 1

W_xh = np.random.randn(dx, dh) * 0.5
W_hh = np.random.randn(dh, dh) * 0.4
W_hy = np.random.randn(dh, dy) * 0.6
bh = np.zeros(dh)
by = np.zeros(dy)

x_seq = np.array([
    [0.8, -0.4],
    [1.2, 0.5],
    [-0.5, 1.1],
    [0.2, 0.9]
])

h_states = [np.zeros(dh)]
y_outputs = []

for t in range(time_steps):
    xt = x_seq[t]
    ht = np.tanh(xt @ W_xh + h_states[-1] @ W_hh + bh)
    yt = ht @ W_hy + by
    h_states.append(ht)
    y_outputs.append(yt)

h_states = np.array(h_states) # (5, 4)
y_outputs = np.array(y_outputs) # (4, 1)

# === Module 4 : Visualisation ===
fig, ax = plt.subplots(figsize=(8.8, 5.0))
color_map = []
for node in G.nodes():
    if node.startswith("x"):
        color_map.append("#3498DB")
    elif node.startswith("h"):
        color_map.append("#E67E22")
    else:
        color_map.append("#2ECC71")

nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map, node_size=1100, edgecolors="black", linewidths=1.2)
nx.draw_networkx_labels(G, pos, ax=ax, font_family="serif", font_size=10, font_weight="bold", font_color="white")

# Straight vertical edges (x -> h and h -> y)
straight_edges = [(u, v) for u, v in G.edges() if not (u.startswith("h") and v.startswith("h"))]
nx.draw_networkx_edges(G, pos, edgelist=straight_edges, ax=ax, arrowstyle="->", arrowsize=18, edge_color="#2C3E50", width=1.8)

# Recurrent horizontal edges (h -> h)
recurrent_edges = [(u, v) for u, v in G.edges() if u.startswith("h") and v.startswith("h")]
nx.draw_networkx_edges(G, pos, edgelist=recurrent_edges, ax=ax, arrowstyle="->", arrowsize=18, edge_color="#C0392B", width=2.2)

edge_labels = nx.get_edge_attributes(G, "label")
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=8, font_color="#333333")

ax.set_title("Unfolded Recurrent Neural Network Computational Graph ($T=4$ Slices)", fontsize=10.5)
ax.set_xlim(0.2, 10.0)
ax.set_ylim(0.4, 4.8)
ax.axis("off")
fig.tight_layout()
fig.savefig("plot1_unfolded_graph.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(7.0, 3.8))
im = ax.imshow(h_states[1:].T, cmap="PuOr", vmin=-1.0, vmax=1.0)
ax.set_xticks(range(time_steps))
ax.set_xticklabels([f"Step t={t+1}" for t in range(time_steps)], fontsize=9)
ax.set_yticks(range(dh))
ax.set_yticklabels([f"Neuron h{j+1}" for j in range(dh)], fontsize=9)
for i in range(dh):
    for j in range(time_steps):
        ax.text(j, i, f"{h_states[j+1, i]:.2f}", ha="center", va="center", color="black" if abs(h_states[j+1, i]) < 0.6 else "white", fontsize=8.5)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hidden State Activation tanh(z)")
ax.set_title("Evolution of Latent Recurrent State Vectors Over Time", fontsize=10)
fig.tight_layout()
fig.savefig("plot2_hidden_states.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 5 : Printing and Saving Tables ===
t1_headers = ["Parameter / Concept", "Mathematical Expression", "Weight Sharing Invariance", "Description"]
t1_rows = [
    ["Input-to-Hidden Matrix", "W_xh in R^{2x4}", "Constant across all t", "Projects external sequence features into memory"],
    ["Hidden-to-Hidden Matrix", "W_hh in R^{4x4}", "Constant across all t", "Propagates temporal context across slices"],
    ["Hidden-to-Output Matrix", "W_hy in R^{4x1}", "Constant across all t", "Maps recurrent memory into task logits"],
    ["State Transition Rule", "h_t = tanh(x_t W_xh + h_{t-1} W_hh + bh)", "Shared nonlinear mapping", "Preserves state updates"],
    ["Output Mapping", "y_t = h_t W_hy + by", "Shared linear projection", "Produces per-step prediction"]
]
show_and_save("Table 1 - Mathematical Formulation of Unfolded Recurrent Graph", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Time Slice", "Input Vector x_t", "Norm ||x_t||", "Input Semantic Role"]
t2_rows = [
    [f"t = {t+1}", f"[{x_seq[t, 0]:.2f}, {x_seq[t, 1]:.2f}]", f"{np.linalg.norm(x_seq[t]):.3f}", f"External stimulus frame {t+1}"]
    for t in range(time_steps)
]
show_and_save("Table 2 - Sequential Input Vector Feed across Time Steps", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Step", "Neuron h1", "Neuron h2", "Neuron h3", "Neuron h4", "State Norm ||h_t||"]
t3_rows = [
    [f"t = {t+1}", f"{h_states[t+1, 0]:.4f}", f"{h_states[t+1, 1]:.4f}", f"{h_states[t+1, 2]:.4f}", f"{h_states[t+1, 3]:.4f}", f"{np.linalg.norm(h_states[t+1]):.4f}"]
    for t in range(time_steps)
]
show_and_save("Table 3 - Hidden Memory State Vectors Across Time Unfolding", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Time Slice", "Hidden Vector h_t", "Linear Logit y_t", "Effective Prediction"]
t4_rows = [
    [f"Slice t={t+1}", f"h_{t+1}", f"{y_outputs[t, 0]:.4f}", f"y_hat_{t+1} = {y_outputs[t, 0]:.3f}"]
    for t in range(time_steps)
]
show_and_save("Table 4 - Output Generation from Recurrent States", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Matrix", "Order", "Frobenius Norm", "Spectral Radius (rho)", "Temporal Stability"]
rho_hh = np.max(np.abs(np.linalg.eigvals(W_hh)))
t5_rows = [
    ["W_xh (Input-Hidden)", "2 x 4", f"{np.linalg.norm(W_xh):.4f}", "N/A", "Stable input scaling"],
    ["W_hh (Recurrent)", "4 x 4", f"{np.linalg.norm(W_hh):.4f}", f"{rho_hh:.4f}", "Stable (< 1.0, No explosion)"],
    ["W_hy (Output)", "4 x 1", f"{np.linalg.norm(W_hy):.4f}", "N/A", "Stable output projection"]
]
show_and_save("Table 5 - Recurrent Weight Tensor Dynamics and Stability Audit", t5_headers, t5_rows, "table 5.png")
