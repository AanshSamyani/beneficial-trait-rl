"""All three RL arms across training on the evals with curves. Every panel uses the same 0 to 1 axis, so the size of a change can be compared between panels."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btrl.common import ROOT
from scripts.stage6_report import EVALS, STEPS, load, summary

OUT = ROOT / "results" / "figures"
ARMS = {"trait": ("Beneficial trait RL", "#2a78d6", "o"), "normal": ("Normal RL", "#eb6834", "s"), "anti": ("Non-beneficial RL", "#1baf7a", "^")}
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3"
PANELS = [("Trait benchmark (154 unseen)", "Trait benchmark\n(154 unseen conversations)"), ("DeceptionBench", "DeceptionBench\n(Huang et al., 2025)"), ("Sycophancy", "Sycophancy\n(Perez et al., 2022)"), ("Agentic misalignment", "Agentic misalignment\n(Lynch et al., 2025)"), ("Blackmail", "Blackmail only\n(Lynch et al., 2025)"), ("HealthBench", "HealthBench\n(Arora et al., 2025)"),
          ("Misalignment (Betley)", "Misalignment\n(Betley et al., 2025)"), ("Alignment questions (Betley)", "Alignment questions\n(Betley et al., 2025)"), ("Impossible-test coding", "Impossible-test coding\n(synthetic, after Guan)"), ("School of Reward Hacks", "School of reward hacks\n(Taylor et al., 2025)"), ("MASK", "MASK*\n(Ren et al., 2025)"), ("HealthBench Professional", "HealthBench Pro\n(Soskin-Hicks et al., 2026)")]
SPEC = {name: (file, keep) for name, file, keep, _ in EVALS}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

fig, axes = plt.subplots(2, 6, figsize=(14.4, 6.0), sharey=True)
rows = []
for ax, (name, title) in zip(axes.ravel(), PANELS):
    file, keep = SPEC[name]
    for arm, (label, color, marker) in ARMS.items():
        pts = [(s, summary(load(arm, s, file, keep))) for s in STEPS]
        rows += [[name, label, s, round(p["mean"], 4), round(p["sem"], 4)] for s, p in pts]
        ax.errorbar([s for s, _ in pts], [p["mean"] for _, p in pts], yerr=[p["sem"] for _, p in pts], color=color, marker=marker, markersize=4.2, markeredgecolor=SURFACE, markeredgewidth=0.5, linewidth=1.6, elinewidth=0.8, capsize=1.6, zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xticks([0, 20, 50, 80, 100])
    ax.set_xlim(-6, 106)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.6, labelsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("Alignment score\n(higher is better)", fontsize=7.5)
for ax in axes[1]:
    ax.set_xlabel("RL train progress (%)", fontsize=7.5)
fig.legend(handles=[plt.Line2D([], [], color=c, marker=m, markersize=5.5, linewidth=1.6, label=l) for l, c, m in ARMS.values()], loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.935))
fig.suptitle("Rewarding the opposite of the beneficial traits, on health conversations only, against normal and trait RL", fontsize=11, fontweight="bold", y=0.985)
fig.text(0.012, 0.012, "gpt-oss-20b, one seed per arm, 100 RL steps, 25% dose. All arms start from the same untrained model at 0%. Error bars are SEM over prompts. Blackmail and Alignment questions sit at 1.00 in all three arms, so their lines overlap.\n*MASK excludes samples with no clear belief, which rose from 8% to 52% in the non-beneficial arm, so its score is not comparable here.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.145, hspace=0.46, wspace=0.14)
fig.savefig(OUT / "figure_nonbeneficial.png", dpi=220)
fig.savefig(OUT / "figure_nonbeneficial.svg")
with (OUT / "figure_nonbeneficial_data.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "arm", "rl_step", "mean", "sem"])
    w.writerows(rows)
print("saved")
