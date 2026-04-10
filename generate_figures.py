"""
Generate figures for the Phase 1 experiment report.
Run: python generate_figures.py
Outputs figures/ directory with all plots used in report/note.tex
"""

import math
import random
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

random.seed(42)
np.random.seed(42)
os.makedirs("figures", exist_ok=True)

COLORS = {"low": "#2196F3", "medium": "#FF9800", "high": "#F44336"}
FONT = {"family": "serif", "size": 11}
matplotlib.rc("font", **FONT)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax_scores(scene_objects, utterance_words, noise=0.05):
    scores = []
    for obj in scene_objects:
        obj_words = obj.lower().split()
        overlap = sum(1 for w in obj_words if w in utterance_words)
        scores.append(math.exp(overlap) + random.uniform(0, noise))
    total = sum(scores)
    return [s / total for s in scores]


def entropy(probs):
    return -sum(p * math.log(p) for p in probs if p > 0)


def brier(probs, target_idx):
    n = len(probs)
    return sum((p - (1.0 if i == target_idx else 0.0)) ** 2 for i, p in enumerate(probs)) / n


def ece(confidences, corrects, n_bins=5):
    bins = [[] for _ in range(n_bins)]
    for conf, correct in zip(confidences, corrects):
        b = min(int(conf * n_bins), n_bins - 1)
        bins[b].append((conf, correct))
    val = 0.0
    total = len(confidences)
    for b in bins:
        if not b:
            continue
        avg_conf = sum(x[0] for x in b) / len(b)
        avg_acc = sum(x[1] for x in b) / len(b)
        val += (len(b) / total) * abs(avg_conf - avg_acc)
    return val


# ---------------------------------------------------------------------------
# Dataset: 60 synthetic scenes (20 per ambiguity level)
# ---------------------------------------------------------------------------

COLORS_OBJ = ["red", "blue", "green", "yellow", "purple"]
SHAPES = ["square", "circle", "triangle", "pentagon", "star"]
SIZES = ["small", "medium", "large"]

def make_object():
    return f"{random.choice(SIZES)} {random.choice(COLORS_OBJ)} {random.choice(SHAPES)}"


def make_scene(ambiguity: str):
    """
    Returns (objects, target_idx, utterance, ambiguity_label).
    Ambiguity controls how many objects match the utterance words.
    """
    n_objects = 4
    target = make_object()
    target_words = set(target.split())

    if ambiguity == "low":
        # Utterance uses all 3 attributes → unique match
        utterance_words = list(target_words)
        distractors = []
        for _ in range(n_objects - 1):
            d = make_object()
            while set(d.split()) & target_words:
                d = make_object()
            distractors.append(d)

    elif ambiguity == "medium":
        # Utterance uses only color → 1–2 partial matches
        color = target.split()[1]
        utterance_words = [color]
        distractors = []
        # one distractor shares the color
        same_color = f"{random.choice(SIZES)} {color} {random.choice(SHAPES)}"
        while same_color == target:
            same_color = f"{random.choice(SIZES)} {color} {random.choice(SHAPES)}"
        distractors.append(same_color)
        for _ in range(n_objects - 2):
            d = make_object()
            while d.split()[1] == color:
                d = make_object()
            distractors.append(d)

    else:  # high
        # Utterance uses only size → all distractors share the size
        size = target.split()[0]
        utterance_words = [size]
        distractors = []
        for _ in range(n_objects - 1):
            d = f"{size} {random.choice(COLORS_OBJ)} {random.choice(SHAPES)}"
            while d == target:
                d = f"{size} {random.choice(COLORS_OBJ)} {random.choice(SHAPES)}"
            distractors.append(d)

    objects = distractors + [target]
    random.shuffle(objects)
    target_idx = objects.index(target)
    utterance = " ".join(utterance_words)
    return objects, target_idx, utterance


def run_dataset(n_per_level=20):
    rows = []
    for level in ["low", "medium", "high"]:
        for _ in range(n_per_level):
            objects, target_idx, utterance = make_scene(level)
            uwords = utterance.split()
            probs = softmax_scores(objects, uwords)
            H = entropy(probs)
            BS = brier(probs, target_idx)
            conf = max(probs)
            correct = int(probs.index(conf) == target_idx)
            rows.append(dict(level=level, H=H, BS=BS, conf=conf, correct=correct, probs=probs))
    return rows


data = run_dataset(20)

def split(data, key):
    return {lv: [r[key] for r in data if r["level"] == lv] for lv in ["low", "medium", "high"]}

# ---------------------------------------------------------------------------
# Figure 1 — Referential Entropy by Ambiguity Level (violin + strip)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 3.8))
H_by_level = split(data, "H")
parts = ax.violinplot([H_by_level[lv] for lv in ["low", "medium", "high"]],
                      positions=[1, 2, 3], showmedians=True, showextrema=False)
for pc, lv in zip(parts["bodies"], ["low", "medium", "high"]):
    pc.set_facecolor(COLORS[lv])
    pc.set_alpha(0.6)
parts["cmedians"].set_color("black")

for xi, lv in zip([1, 2, 3], ["low", "medium", "high"]):
    jitter = np.random.uniform(-0.08, 0.08, len(H_by_level[lv]))
    ax.scatter([xi + j for j in jitter], H_by_level[lv],
               color=COLORS[lv], s=18, alpha=0.8, zorder=3)

ln_N = math.log(4)
ax.axhline(ln_N, ls="--", color="gray", lw=1, label=f"$\\ln N = {ln_N:.2f}$ (uniform)")
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Ambiguity Level")
ax.set_ylabel("$H(T \\mid u)$")
ax.set_title("Referential Entropy by Ambiguity Level")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig1_entropy.pdf")
plt.close()
print("Saved fig1_entropy.pdf")

# ---------------------------------------------------------------------------
# Figure 2 — Brier Score by Ambiguity Level
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 3.8))
BS_by_level = split(data, "BS")
bp = ax.boxplot([BS_by_level[lv] for lv in ["low", "medium", "high"]],
                patch_artist=True, widths=0.5,
                medianprops=dict(color="black", lw=2))
for patch, lv in zip(bp["boxes"], ["low", "medium", "high"]):
    patch.set_facecolor(COLORS[lv])
    patch.set_alpha(0.7)
ax.set_xticks([1, 2, 3])
ax.set_xticklabels(["Low", "Medium", "High"])
ax.set_xlabel("Ambiguity Level")
ax.set_ylabel("Brier Score")
ax.set_title("Brier Score by Ambiguity Level")
plt.tight_layout()
plt.savefig("figures/fig2_brier.pdf")
plt.close()
print("Saved fig2_brier.pdf")

# ---------------------------------------------------------------------------
# Figure 3 — ECE Reliability Diagram
# ---------------------------------------------------------------------------

all_conf = [r["conf"] for r in data]
all_correct = [r["correct"] for r in data]
n_bins = 5
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_accs, bin_confs, bin_counts = [], [], []
for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
    mask = [(lo <= c < hi) for c in all_conf]
    sub_c = [c for c, m in zip(all_conf, mask) if m]
    sub_a = [a for a, m in zip(all_correct, mask) if m]
    if sub_c:
        bin_confs.append(np.mean(sub_c))
        bin_accs.append(np.mean(sub_a))
        bin_counts.append(len(sub_c))

fig, ax = plt.subplots(figsize=(4.5, 4))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
bar_width = 0.12
for bc, ba, bn in zip(bin_confs, bin_accs, bin_counts):
    ax.bar(bc, ba, width=bar_width, color="#2196F3", alpha=0.7, edgecolor="black", lw=0.5)
    ax.bar(bc, bc - ba if bc > ba else 0, width=bar_width, bottom=ba,
           color="#F44336", alpha=0.4, edgecolor="none")
    ax.bar(bc, ba - bc if ba > bc else 0, width=bar_width, bottom=bc,
           color="#4CAF50", alpha=0.4, edgecolor="none")

ece_val = ece(all_conf, all_correct, n_bins)
ax.text(0.05, 0.88, f"ECE = {ece_val:.3f}", transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("Confidence (max $P(t_i \\mid u)$)")
ax.set_ylabel("Accuracy")
ax.set_title("ECE Reliability Diagram")
ax.legend(fontsize=9)
over = mpatches.Patch(color="#F44336", alpha=0.5, label="Overconfident")
under = mpatches.Patch(color="#4CAF50", alpha=0.5, label="Underconfident")
ax.legend(handles=[over, under], fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig3_ece.pdf")
plt.close()
print("Saved fig3_ece.pdf")

# ---------------------------------------------------------------------------
# Figure 4 — Clarification Decision vs. Cost c
# ---------------------------------------------------------------------------

cost_vals = np.linspace(0.0, 1.0, 50)
ask_rates = {lv: [] for lv in ["low", "medium", "high"]}
for c in cost_vals:
    for lv in ["low", "medium", "high"]:
        subset = [r for r in data if r["level"] == lv]
        asks = sum(1 for r in subset if (1.0 - c) > max(r["probs"]))
        ask_rates[lv].append(asks / len(subset))

fig, ax = plt.subplots(figsize=(5, 3.8))
for lv in ["low", "medium", "high"]:
    ax.plot(cost_vals, ask_rates[lv], color=COLORS[lv], lw=2,
            label=f"{lv.capitalize()} ambiguity")
ax.set_xlabel("Clarification cost $c$")
ax.set_ylabel("Fraction asking a question")
ax.set_title("Clarification Rate vs. Cost $c$")
ax.legend(fontsize=9)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig("figures/fig4_clarification_rate.pdf")
plt.close()
print("Saved fig4_clarification_rate.pdf")

# ---------------------------------------------------------------------------
# Figure 5 — Cost-Penalized Accuracy vs. c
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(5, 3.8))
for lv in ["low", "medium", "high"]:
    subset = [r for r in data if r["level"] == lv]
    rewards = []
    for c in cost_vals:
        total_reward = 0.0
        for r in subset:
            if (1.0 - c) > max(r["probs"]):
                # ask: pay cost c, then correct (simulated: assume question resolves fully)
                total_reward += 1.0 - c
            else:
                # commit: reward = 1 if correct, 0 otherwise
                total_reward += r["correct"]
        rewards.append(total_reward / len(subset))
    ax.plot(cost_vals, rewards, color=COLORS[lv], lw=2,
            label=f"{lv.capitalize()} ambiguity")

ax.set_xlabel("Clarification cost $c$")
ax.set_ylabel("Mean reward (Accuracy $- c \\times$ asks)")
ax.set_title("Cost-Penalized Accuracy vs. Cost $c$")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/fig5_reward.pdf")
plt.close()
print("Saved fig5_reward.pdf")

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

print("\n=== Summary Statistics ===")
for lv in ["low", "medium", "high"]:
    subset = [r for r in data if r["level"] == lv]
    print(f"  {lv.capitalize():8s}  H={np.mean([r['H'] for r in subset]):.3f}  "
          f"BS={np.mean([r['BS'] for r in subset]):.3f}  "
          f"Acc={np.mean([r['correct'] for r in subset]):.2f}")
print(f"  Overall ECE = {ece_val:.3f}")
