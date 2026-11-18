# === Module 1 : Importing the Required Packages ===
import numpy as np
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

# === Module 2 : Representing the Thermal Sensor Image as a Matrix ===
np.random.seed(101)
R = C = 8

row, col = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
I = 45 + 8.5 * row + 7.0 * col
I[2:5, 3:6] += 80
I[0:3, 0:3] -= 20
I = np.clip(np.round(I + np.random.normal(0, 2.5, (R, C))), 0, 255).astype(int)

# === Module 3 : Brightness Adjustment by Matrix Addition ===
J = np.ones((R, C), dtype=int)
k = 35
B = np.clip(I + k * J, 0, 255)
D = np.clip(I - k * J, 0, 255)

# === Module 4 : Contrast Adjustment by Scalar Multiplication ===
c1, c2 = 1.35, 0.65
S1 = np.clip(np.round(c1 * I), 0, 255).astype(int)
S2 = np.clip(np.round(c2 * I), 0, 255).astype(int)

# === Module 5 : Data Transformation by Transpose ===
T = I.T
TT = T.T

# === Module 6 : Verifying the Algebraic Properties of the Operations ===
K = np.clip(np.round(np.random.uniform(5, 55, (R, C))), 0, 255).astype(int)
checks = [
    ("(I + K)^T = I^T + K^T", np.array_equal((I + K).T, I.T + K.T)),
    ("(c * I)^T = c * I^T", np.allclose((c1 * I).T, c1 * I.T)),
    ("(I^T)^T = I", np.array_equal(TT, I)),
    ("I + O = I (O = Zero Matrix)", np.array_equal(I + np.zeros((R, C), int), I)),
    ("I + K = K + I (Commutative)", np.array_equal(I + K, K + I)),
    ("1 * I = I (Scalar Identity)", np.array_equal(1 * I, I)),
]

# === Module 7 : Statistics of Original and Transformed Matrices ===
ops = [
    ("Original Matrix I", I),
    ("Brightened (I + 35J)", B),
    ("Darkened (I - 35J)", D),
    ("Contrast Scaled (1.35 * I)", S1),
    ("Contrast Attenuated (0.65 * I)", S2),
    ("Transpose Matrix I^T", T),
]

# === Module 8 : Visualisation ===
titles = [
    "Original Matrix $I$",
    "Brightened $I + 35J$",
    "Darkened $I - 35J$",
    "Contrast $1.35 \\times I$",
    "Contrast $0.65 \\times I$",
    "Transpose $I^T$",
]

fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
for ax, (name, M), ttl in zip(axes.ravel(), ops, titles):
    ax.imshow(M, cmap="magma", vmin=0, vmax=255)
    ax.set_title(ttl, fontsize=9)
    ax.set_xticks(range(0, 8, 2))
    ax.set_yticks(range(0, 8, 2))
    ax.tick_params(labelsize=7)
fig.suptitle("Thermal Matrix under Addition, Scalar Multiplication and Transposition", fontsize=10)
fig.tight_layout()
fig.savefig("plot1_operations.png", dpi=300, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 3.3))
bins = np.arange(0, 271, 15)
for M, lab, cl in [(I, "Original I", "#1f77b4"), (B, "Brightened I + 35J", "#ff7f0e"), (D, "Darkened I - 35J", "#2ca02c")]:
    ax.hist(M.ravel(), bins=bins, histtype="step", lw=1.6, color=cl, label=lab)
ax.set_xlabel("Pixel Intensity")
ax.set_ylabel("Pixel Frequency")
ax.set_xlim(0, 270)
ax.set_xticks(np.arange(0, 271, 30))
ax.grid(axis="y", alpha=0.55)
ax.set_axisbelow(True)
ax.legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig("plot2_histogram.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 9 : Printing and Saving Tables ===
def mat_table_dict(title, M):
    return {
        "title": title,
        "headers": ["Row"] + [f"C{j}" for j in range(C)],
        "rows": [[f"R{i}"] + [f"{v}" for v in M[i]] for i in range(M.shape[0])],
    }

t1 = mat_table_dict("Table 1 - Original Sensor Image Matrix I (Order 8 x 8)", I)
show_and_save(t1["title"], t1["headers"], t1["rows"], "table 1.png")

t2 = mat_table_dict("Table 2 - Brightness Enhanced by Matrix Addition: B = I + 35J", B)
show_and_save(t2["title"], t2["headers"], t2["rows"], "table 2.png")

t3 = mat_table_dict("Table 3 - Brightness Reduced by Matrix Subtraction: D = I - 35J", D)
show_and_save(t3["title"], t3["headers"], t3["rows"], "table 3.png")

t4 = mat_table_dict("Table 4 - Contrast Scaled by Scalar Multiplication: 1.35 x I", S1)
show_and_save(t4["title"], t4["headers"], t4["rows"], "table 4.png")

t5 = mat_table_dict("Table 5 - Matrix Transposition: I^T (Order 8 x 8)", T)
show_and_save(t5["title"], t5["headers"], t5["rows"], "table 5.png")

t6_headers = ["S.No", "Operation", "Order", "Min", "Max", "Mean", "Std. Dev"]
t6_rows = [
    [str(i + 1), name, f"{M.shape[0]} x {M.shape[1]}", f"{M.min()}", f"{M.max()}", f"{M.mean():.2f}", f"{M.std():.2f}"]
    for i, (name, M) in enumerate(ops)
]
show_and_save("Table 6 - Statistical Summary of Original and Transformed Matrices", t6_headers, t6_rows, "table 6.png")

t7_headers = ["S.No", "Mathematical Property", "Algebraic Result"]
t7_rows = [[str(i + 1), t, "Verified" if ok else "Failed"] for i, (t, ok) in enumerate(checks)]
show_and_save("Table 7 - Verification of Matrix Algebra Laws and Invariance", t7_headers, t7_rows, "table 7.png")
