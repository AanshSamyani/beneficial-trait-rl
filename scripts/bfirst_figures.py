"""Arm (ii), beneficial then non-beneficial RL, against arm (i), non-beneficial RL alone. Every panel uses the same 0 to 1 axis, with arm (ii)'s first phase to the left of zero."""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from btrl.common import ROOT
from scripts.stage6_report import EVALS, load, summary

OUT = ROOT / "results" / "figures"
SERIES = {"i": ("(i) Non-beneficial RL only", "#1baf7a", "^"), "ii": ("(ii) 50 steps beneficial, then the same non-beneficial RL", "#2a78d6", "o"), "normal": ("Normal RL, for reference", "#eb6834", "s")}
SURFACE, INK, MUTED, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e6e3", "#efeeea"
PANELS = [("Trait benchmark (154 unseen)", "Trait benchmark\n(154 unseen conversations)"), ("DeceptionBench", "DeceptionBench\n(Huang et al., 2025)"), ("Sycophancy", "Sycophancy\n(Perez et al., 2022)"), ("Agentic misalignment", "Agentic misalignment\n(Lynch et al., 2025)"), ("HealthBench", "HealthBench\n(Arora et al., 2025)"), ("HealthBench Professional", "HealthBench Pro\n(Soskin-Hicks et al., 2026)"),
          ("Misalignment (Betley)", "Misalignment\n(Betley et al., 2025)"), ("Alignment questions (Betley)", "Alignment questions\n(Betley et al., 2025)"), ("StrongREJECT", "StrongREJECT\n(scored at fewer checkpoints)"), ("Impossible-test coding", "Impossible-test coding\n(synthetic, after Guan)"), ("School of Reward Hacks", "School of reward hacks\n(Taylor et al., 2025)"), ("MASK", "MASK*\n(Ren et al., 2025)")]
SPEC = {name: (file, keep) for name, file, keep, _ in EVALS}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.linewidth": 0.6, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def points(series, file, keep):
    base = ("normal", 0)
    if series == "ii":
        plan = [(-50, base), (0, ("bfirst", 50))] + [(s, ("banti", s)) for s in (20, 50, 80, 100)]
    else:
        arm = "anti" if series == "i" else "normal"
        plan = [(0, base)] + [(s, (arm, s)) for s in (20, 50, 80, 100)]
    out = []
    for x, (a, s) in plan:
        rows = load(a, s, file, keep)
        if rows:
            out.append((x, summary(rows)))
    return out


fig, axes = plt.subplots(2, 6, figsize=(14.8, 6.1), sharey=True)
rows = []
for ax, (name, title) in zip(axes.ravel(), PANELS):
    file, keep = SPEC[name]
    ax.axvspan(-56, 0, color=BAND, zorder=0, linewidth=0)
    for key in ("normal", "i", "ii"):
        label, color, marker = SERIES[key]
        pts = points(key, file, keep)
        rows += [[name, label, x, round(p["mean"], 4), round(p["sem"], 4)] for x, p in pts]
        ax.errorbar([x for x, _ in pts], [p["mean"] for _, p in pts], yerr=[p["sem"] for _, p in pts], color=color, marker=marker, markersize=4.2, markeredgecolor=SURFACE, markeredgewidth=0.5, linewidth=1.6, elinewidth=0.8, capsize=1.6, zorder=3)
    ax.set_title(title, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.set_xticks([-50, 0, 50, 100])
    ax.set_xlim(-56, 106)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.6, labelsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("Alignment score\n(higher is better)", fontsize=7.5)
for ax in axes[1]:
    ax.set_xlabel("Steps from start of non-beneficial RL", fontsize=7.5)
fig.legend(handles=[plt.Line2D([], [], color=c, marker=m, markersize=5.5, linewidth=1.6, label=l) for l, c, m in SERIES.values()], loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.935))
fig.suptitle("Does a little beneficial RL protect a model from non-beneficial RL afterwards?", fontsize=11, fontweight="bold", y=0.985)
fig.text(0.012, 0.012, "gpt-oss-20b, one seed per arm, 25% dose. Shaded: arm (ii)'s first phase, on data arm (i) never saw. Both arms then get the same 100 steps. Error bars are SEM over prompts and do not include run-to-run variation.\nAlignment questions sit at 1.00 throughout, so their lines overlap.  *MASK excludes samples with no clear belief, 13% in arm (ii) and 52% in arm (i) at step 100, so its scores are not directly comparable.", fontsize=7.5, color=MUTED)
fig.subplots_adjust(left=0.06, right=0.99, top=0.80, bottom=0.145, hspace=0.46, wspace=0.14)
fig.savefig(OUT / "figure_bfirst_comparison.png", dpi=220)
fig.savefig(OUT / "figure_bfirst_comparison.svg")
with (OUT / "figure_bfirst_comparison_data.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "arm", "steps_from_nonbeneficial_start", "mean", "sem"])
    w.writerows(rows)
print("saved")
