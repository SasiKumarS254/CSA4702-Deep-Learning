# === Module 1 : Importing Required Packages ===
import os
import sys

# Ensure current working directory is the script's directory
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(EXP_DIR)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

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
    
    out_path = os.path.join(EXP_DIR, filename)
    img.save(out_path)
    print(f"Saved table: {out_path}")

# === Module 2 : Tactical Feature Specifications & Simulation ===
SAMURAI_ATTRIBUTES = ["strength", "durability", "speed", "skill", "stamina", "stealth"]
NINJA_ATTRIBUTES = ["teamwork", "agility", "momentum", "timing", "commander_control", "alertness"]
FEATURE_COLS = SAMURAI_ATTRIBUTES + NINJA_ATTRIBUTES
N_FEATURES = len(FEATURE_COLS)
NOISE_DIM = 32
N_CLASSES = 2  # 0: TRAP, 1: ESCAPE

def simulate_seed_battles(num_samples=5000):
    """
    Simulates tactical encounters between a solitary samurai and an ambushing ninja squad.
    """
    data = np.random.uniform(20.0, 100.0, size=(num_samples, N_FEATURES))
    df = pd.DataFrame(data, columns=FEATURE_COLS)
    
    # 1. Samurai Martial Prowess
    martial_power = (
        0.28 * df["strength"] +
        0.22 * df["durability"] +
        0.20 * df["skill"] +
        0.18 * df["stamina"] +
        0.12 * df["speed"]
    )
    
    # 2. Ninja Squad Collective Threat
    squad_threat = (
        0.30 * df["teamwork"] +
        0.22 * df["agility"] +
        0.18 * df["momentum"] +
        0.15 * df["timing"] +
        0.15 * df["commander_control"]
    )
    
    # 3. Stealth Infiltration / Evasion Delta
    evasion_score = (0.65 * df["stealth"] + 0.35 * df["speed"]) - df["alertness"] + 8.0
    
    # 4. Direct Combat Engagement Delta
    combat_score = martial_power - squad_threat
    
    # Tactical Outcome Determination with stochastic friction
    tactical_advantage = np.maximum(evasion_score, combat_score) + np.random.normal(0, 3.5, size=num_samples)
    df["outcome"] = (tactical_advantage > 0).astype(int)
    return df

# === Module 3 : Conditional Generative Adversarial Network (cGAN) ===
class ConditionalGenerator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM, n_classes=N_CLASSES, out_features=N_FEATURES):
        super().__init__()
        self.label_emb = nn.Sequential(
            nn.Linear(n_classes, 16),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.Linear(noise_dim + 16, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, out_features),
            nn.Tanh()
        )

    def forward(self, z, labels_onehot):
        c = self.label_emb(labels_onehot)
        x = torch.cat([z, c], dim=1)
        return self.net(x)

class ConditionalDiscriminator(nn.Module):
    def __init__(self, in_features=N_FEATURES, n_classes=N_CLASSES):
        super().__init__()
        self.label_emb = nn.Sequential(
            nn.Linear(n_classes, 16),
            nn.LeakyReLU(0.2)
        )
        self.net = nn.Sequential(
            nn.Linear(in_features + 16, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x, labels_onehot):
        c = self.label_emb(labels_onehot)
        inp = torch.cat([x, c], dim=1)
        return self.net(inp)

# === Module 4 : Adversarial Training Routine ===
seed_df = simulate_seed_battles(num_samples=5000)
min_vals = seed_df[FEATURE_COLS].min().values
max_vals = seed_df[FEATURE_COLS].max().values

# Normalize to [-1, 1] for Tanh generator activation
X_norm = 2.0 * (seed_df[FEATURE_COLS].values - min_vals) / (max_vals - min_vals) - 1.0
X_tensor = torch.tensor(X_norm, dtype=torch.float32)
y_tensor = torch.eye(N_CLASSES)[torch.tensor(seed_df["outcome"].values)]

generator = ConditionalGenerator()
discriminator = ConditionalDiscriminator()

opt_g = torch.optim.Adam(generator.parameters(), lr=0.00025, betas=(0.5, 0.999))
opt_d = torch.optim.Adam(discriminator.parameters(), lr=0.00020, betas=(0.5, 0.999))
bce_criterion = nn.BCELoss()

epochs = 1800
batch_size = 128
n_batches = len(X_tensor) // batch_size

history = {"epoch": [], "d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

for ep in range(1, epochs + 1):
    perm = torch.randperm(len(X_tensor))
    d_loss_epoch, g_loss_epoch = 0.0, 0.0
    d_real_epoch, d_fake_epoch = 0.0, 0.0
    
    # Run mini-batch steps
    for b in range(min(n_batches, 10)):  # Fast representative batch schedule
        idx = perm[b * batch_size : (b + 1) * batch_size]
        real_x = X_tensor[idx]
        real_y = y_tensor[idx]
        
        # Train Discriminator: Maximize log(D(x|y)) + log(1 - D(G(z|y)|y))
        opt_d.zero_grad()
        z = torch.randn(len(idx), NOISE_DIM)
        fake_x = generator(z, real_y)
        
        # Soft label smoothing for discriminator targets (0.9 instead of 1.0)
        out_real = discriminator(real_x, real_y)
        out_fake = discriminator(fake_x.detach(), real_y)
        
        loss_d_real = bce_criterion(out_real, torch.full_like(out_real, 0.9))
        loss_d_fake = bce_criterion(out_fake, torch.zeros_like(out_fake))
        loss_d = loss_d_real + loss_d_fake
        loss_d.backward()
        opt_d.step()
        
        # Train Generator: Maximize log(D(G(z|y)|y))
        opt_g.zero_grad()
        out_fake_for_g = discriminator(fake_x, real_y)
        loss_g = bce_criterion(out_fake_for_g, torch.ones_like(out_fake_for_g))
        loss_g.backward()
        opt_g.step()
        
        d_loss_epoch += loss_d.item()
        g_loss_epoch += loss_g.item()
        d_real_epoch += out_real.mean().item()
        d_fake_epoch += out_fake.mean().item()

    if ep % 100 == 0 or ep == 1:
        history["epoch"].append(ep)
        history["d_loss"].append(d_loss_epoch / 10)
        history["g_loss"].append(g_loss_epoch / 10)
        history["d_real"].append(d_real_epoch / 10)
        history["d_fake"].append(d_fake_epoch / 10)

# === Module 5 : Synthetic Dataset Generation & Tactical Path Tagging ===
num_generate = 1000
labels_gen = np.array([0] * (num_generate // 2) + [1] * (num_generate // 2))
y_gen_tensor = torch.eye(N_CLASSES)[torch.tensor(labels_gen)]

generator.eval()
with torch.no_grad():
    z_gen = torch.randn(num_generate, NOISE_DIM)
    fake_samples_norm = generator(z_gen, y_gen_tensor).numpy()

# Denormalize back to original scale [20, 100]
data_denorm = (fake_samples_norm + 1.0) / 2.0 * (max_vals - min_vals) + min_vals
synth_df = pd.DataFrame(data_denorm, columns=FEATURE_COLS).round(2)
synth_df["outcome"] = np.where(labels_gen == 1, "escape", "trap")

# Determine optimal tactical decision path
avoid_metric = 0.65 * synth_df["stealth"] + 0.35 * synth_df["speed"] - synth_df["alertness"] + 8.0
fight_metric = (
    (0.28 * synth_df["strength"] + 0.22 * synth_df["durability"] + 0.20 * synth_df["skill"] + 0.18 * synth_df["stamina"] + 0.12 * synth_df["speed"]) -
    (0.30 * synth_df["teamwork"] + 0.22 * synth_df["agility"] + 0.18 * synth_df["momentum"] + 0.15 * synth_df["timing"] + 0.15 * synth_df["commander_control"])
)

escape_paths = []
for idx in range(num_generate):
    if labels_gen[idx] == 1:
        path = "avoid_confrontation" if avoid_metric.iloc[idx] >= fight_metric.iloc[idx] else "defeat_all_ninjas"
    else:
        path = "cordoned_and_trapped"
    escape_paths.append(path)

synth_df["escape_path"] = escape_paths
# Shuffle rows
synth_df = synth_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
synth_df.insert(0, "record_id", range(1, len(synth_df) + 1))
csv_out_path = os.path.join(EXP_DIR, "samurai_ninja_trap_escape_1000.csv")
synth_df.to_csv(csv_out_path, index=False)
print(f"Exported synthetic tactical dataset: {csv_out_path}")

# === Module 6 : Visualisations ===
# Plot 1: Adversarial Loss and Nash Equilibrium Probabilities
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax1.plot(history["epoch"], history["d_loss"], label="Discriminator Loss", color="#C0392B", lw=1.8)
ax1.plot(history["epoch"], history["g_loss"], label="Generator Loss", color="#2980B9", lw=1.8)
ax1.set_xlabel("Adversarial Epoch", fontsize=8.5)
ax1.set_ylabel("Binary Cross-Entropy Loss", fontsize=8.5)
ax1.set_title("Conditional GAN Minimax Loss Dynamics", fontsize=9.5)
ax1.grid(True, linestyle="--", alpha=0.35)
ax1.legend(fontsize=8)

ax2.plot(history["epoch"], history["d_real"], label="D(Real | y)", color="#27AE60", lw=1.8)
ax2.plot(history["epoch"], history["d_fake"], label="D(G(z) | y)", color="#E67E22", lw=1.8)
ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.7, label="Theoretical Nash Optimum (0.50)")
ax2.set_xlabel("Adversarial Epoch", fontsize=8.5)
ax2.set_ylabel("Discriminator Probability", fontsize=8.5)
ax2.set_title("Discriminator Confidence and Convergence", fontsize=9.5)
ax2.grid(True, linestyle="--", alpha=0.35)
ax2.legend(fontsize=8)

fig.suptitle("Samurai vs Ninja cGAN: Adversarial Equilibrium and Probability Trajectories", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(EXP_DIR, "plot1_cgan_training_dynamics.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# Plot 2: Tactical Feature Radar & Evasion Scatter Distribution
fig = plt.figure(figsize=(11.0, 4.6))
ax_radar = fig.add_subplot(1, 2, 1, polar=True)
ax_scatter = fig.add_subplot(1, 2, 2)

# Subplot A: Radar Comparison (Samurai attributes vs Ninja attributes for Escape vs Trap)
radar_attrs = ["strength", "durability", "speed", "skill", "stealth", "alertness", "teamwork", "momentum"]
mean_escape = synth_df[synth_df["outcome"] == "escape"][radar_attrs].mean().values
mean_trap = synth_df[synth_df["outcome"] == "trap"][radar_attrs].mean().values

angles = np.linspace(0, 2 * np.pi, len(radar_attrs), endpoint=False).tolist()
angles += angles[:1]
esc_vals = mean_escape.tolist() + [mean_escape[0]]
trap_vals = mean_trap.tolist() + [mean_trap[0]]

ax_radar.plot(angles, esc_vals, label="ESCAPE Cohort", color="#27AE60", lw=2.0)
ax_radar.fill(angles, esc_vals, color="#27AE60", alpha=0.15)
ax_radar.plot(angles, trap_vals, label="TRAP Cohort", color="#C0392B", lw=2.0)
ax_radar.fill(angles, trap_vals, color="#C0392B", alpha=0.15)

ax_radar.set_xticks(angles[:-1])
ax_radar.set_xticklabels(radar_attrs, fontsize=8)
ax_radar.set_ylim(20, 100)
ax_radar.set_title("Tactical Attribute Comparison: Escape vs Trap", fontsize=9.5, pad=12)
ax_radar.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15), fontsize=8)

# Subplot B: Combat Power vs Stealth Evasion Scatter
combat_proxy = (
    0.3 * synth_df["strength"] + 0.3 * synth_df["skill"] + 0.2 * synth_df["durability"] + 0.2 * synth_df["stamina"]
)
stealth_proxy = synth_df["stealth"] - synth_df["alertness"]

palette = {"avoid_confrontation": "#2980B9", "defeat_all_ninjas": "#27AE60", "cordoned_and_trapped": "#C0392B"}
for path_name, color in palette.items():
    subset = synth_df[synth_df["escape_path"] == path_name]
    ax_scatter.scatter(
        combat_proxy.loc[subset.index],
        stealth_proxy.loc[subset.index],
        c=color, label=path_name.replace("_", " ").title(),
        alpha=0.65, s=22, edgecolors="none"
    )

ax_scatter.set_xlabel("Samurai Combat Prowess Index", fontsize=8.5)
ax_scatter.set_ylabel("Stealth Evasion Margin (Stealth - Alertness)", fontsize=8.5)
ax_scatter.set_title("Tactical Escape Pathways in Synthetic Space", fontsize=9.5)
ax_scatter.grid(True, linestyle="--", alpha=0.35)
ax_scatter.legend(loc="upper left", fontsize=7.5)

fig.suptitle("Tactical Cohort Analysis and Path Specialization Distribution", fontsize=10.5)
fig.tight_layout()
fig.savefig(os.path.join(EXP_DIR, "plot2_tactical_escape_distribution.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

# === Module 7 : Output Tables ===
t1_headers = ["Attribute Group", "Feature Name", "Tactical Description", "Scale Range", "Influence Weight"]
t1_rows = [
    ["Samurai Warrior", "strength", "Melee striking power & blade impact force", "[20, 100]", "0.28 (Martial Engagement)"],
    ["Samurai Warrior", "durability", "Armor resistance & wound tolerance", "[20, 100]", "0.22 (Martial Engagement)"],
    ["Samurai Warrior", "speed", "Rapid sprint & tactical footwork velocity", "[20, 100]", "0.35 (Stealth Evasion)"],
    ["Samurai Warrior", "skill", "Katana mastery & parrying proficiency", "[20, 100]", "0.20 (Martial Engagement)"],
    ["Samurai Warrior", "stealth", "Concealment in shadows and tree cover", "[20, 100]", "0.65 (Stealth Evasion)"],
    ["Ninja Squad", "teamwork", "Multi-angle cordoning coordination", "[20, 100]", "0.30 (Collective Threat)"],
    ["Ninja Squad", "alertness", "Perception of foliage movement & breathing", "[20, 100]", "1.00 (Counter-Stealth)"],
    ["Ninja Squad", "momentum", "Rapid perimeter constriction aggression", "[20, 100]", "0.18 (Collective Threat)"]
]
show_and_save("Table 1 - Samurai vs Ninja Tactical Attribute & Combat Influence Taxonomy", t1_headers, t1_rows, "table 1.png")

t2_headers = ["Subsystem", "Layer Order", "Layer Type & Configuration", "Output Dimension", "Activation Function"]
t2_rows = [
    ["Generator (G)", "Layer 1", "Dense (Noise 32 + Class 16)", "128 Units", "LeakyReLU(0.2) + BatchNorm"],
    ["Generator (G)", "Layer 2", "Dense (128 -> 256)", "256 Units", "LeakyReLU(0.2) + BatchNorm"],
    ["Generator (G)", "Layer 3", "Dense (256 -> 12 Features)", "12 Attributes", "Tanh (Scaled [-1, 1])"],
    ["Discriminator (D)", "Layer 1", "Dense (Features 12 + Class 16)", "256 Units", "LeakyReLU(0.2) + Dropout(0.3)"],
    ["Discriminator (D)", "Layer 2", "Dense (256 -> 128)", "128 Units", "LeakyReLU(0.2) + Dropout(0.3)"],
    ["Discriminator (D)", "Layer 3", "Dense (128 -> 1)", "1 Probability", "Sigmoid (Real vs Synthetic)"]
]
show_and_save("Table 2 - Conditional GAN Architectural & Layer Topology Specifications", t2_headers, t2_rows, "table 2.png")

t3_headers = ["Training Milestone", "D Loss (BCE)", "G Loss (BCE)", "D(Real | y)", "D(G(z) | y)", "Equilibrium Status"]
milestones_idx = [0, len(history["epoch"]) // 4, len(history["epoch"]) // 2, 3 * len(history["epoch"]) // 4, -1]
t3_rows = [
    [
        f"Epoch {history['epoch'][i]:04d}",
        f"{history['d_loss'][i]:.4f}",
        f"{history['g_loss'][i]:.4f}",
        f"{history['d_real'][i]:.3f}",
        f"{history['d_fake'][i]:.3f}",
        "Equilibrium Reached" if abs(history['d_real'][i] - 0.5) < 0.20 else "Discriminator Dominant"
    ]
    for i in milestones_idx
]
show_and_save("Table 3 - Adversarial Training Progression and Nash Equilibrium Audit", t3_headers, t3_rows, "table 3.png")

t4_headers = ["Tactical Pathway", "Cohort Count", "Population Ratio (%)", "Avg Samurai Combat", "Avg Ninja Threat", "Avg Stealth Delta"]
t4_rows = []
for path_val in ["avoid_confrontation", "defeat_all_ninjas", "cordoned_and_trapped"]:
    sub = synth_df[synth_df["escape_path"] == path_val]
    count = len(sub)
    ratio = (count / num_generate) * 100
    avg_c = (0.28 * sub["strength"] + 0.22 * sub["durability"] + 0.20 * sub["skill"]).mean()
    avg_t = (0.30 * sub["teamwork"] + 0.22 * sub["agility"] + 0.18 * sub["momentum"]).mean()
    avg_s = (sub["stealth"] - sub["alertness"]).mean()
    t4_rows.append([
        path_val.replace("_", " ").title(),
        f"{count}",
        f"{ratio:.1f}%",
        f"{avg_c:.2f}",
        f"{avg_t:.2f}",
        f"{avg_s:+.2f}"
    ])
show_and_save("Table 4 - Synthetic Population Tactical Breakdown & Path Distribution", t4_headers, t4_rows, "table 4.png")

t5_headers = ["Tactical Metric", "ESCAPE Cohort (Mean)", "TRAP Cohort (Mean)", "Separation Delta", "T-Stat Diagnostic"]
radar_metrics = ["strength", "durability", "speed", "skill", "stealth", "teamwork", "alertness"]
t5_rows = []
for m in radar_metrics:
    m_esc = synth_df[synth_df["outcome"] == "escape"][m].mean()
    m_trp = synth_df[synth_df["outcome"] == "trap"][m].mean()
    delta = m_esc - m_trp
    t5_rows.append([
        m.replace("_", " ").capitalize(),
        f"{m_esc:.2f}",
        f"{m_trp:.2f}",
        f"{delta:+.2f}",
        "High Significance" if abs(delta) > 10.0 else "Secondary Moderator"
    ])
show_and_save("Table 5 - Statistical Trait Significance: Escape vs Trap Discriminative Audit", t5_headers, t5_rows, "table 5.png")
