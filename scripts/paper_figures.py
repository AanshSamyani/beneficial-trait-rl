"""Publication-style figures for the writeup, modelled on the layout of Figures 1, 3 and 4 of Jagadeesh et al. (2026).

Writes PNG (300 dpi), SVG and PDF files to results/writeup/figures/:
  fig1_main_20b           trait benchmark over training and the non-saturated alignment evals at step 100 (paper Figure 1B layout)
  fig2_curves_20b         three rows: beneficial trait evals, alignment evals, capability evals (paper Figure 3 layout)
  fig3_dose               5%, 10% and 25% trait data
  fig4_main_120b          Figure 1 layout for gpt-oss-120b
  fig5_opposite_control   rewarding the opposite of each trait
  fig6_persistence        50 beneficial steps before 100 opposite-trait steps
  figS1_curves_120b       Figure 2 layout for gpt-oss-120b
  fig8_main_inkling       Figure 1 layout for Inkling-2-Small
  figS2_curves_inkling    Figure 2 layout for Inkling-2-Small
Saturated evals are left out: Betley et al.'s alignment and misalignment questions, agentic misalignment and blackmail sit at or
near 1.00 for both arms. Figures 5 and 6 keep agentic misalignment and Betley misalignment, because the opposite-trait arm moves
them, and drop only the panels that stay at 1.00 in every arm.
Explanatory text lives in the captions of the writeup, not in the figures. Error bars are standard errors over prompts.
Palette checked for normal vision and simulated protan, deutan and tritan vision (pairwise OKLab distance >= 15 and >= 8)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from btrl.common import read_jsonl
from scripts.stage6_report import EVALS, load, summary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
def _load(path):
    """Reports a replication has not produced yet load as None, and the figures that need them are skipped."""
    return json.loads(path.read_text()) if path.exists() else None


R20 = _load(ROOT / "results" / "stage6" / "report.json")
R120 = _load(ROOT / "results" / "stage6_120b" / "report.json")
RINK = _load(ROOT / "results" / "stage6_inkling" / "report.json")
SUFFIX = {"20b": "", "120b": "120", "inkling": "ink"}
BASE_MIX = {"20b": "base", "120b": "base120", "inkling": "baseink"}
DOSE = _load(ROOT / "results" / "dose_report.json")
SPEC = {n: (f, k) for n, f, k, _ in EVALS}
MIX = ROOT / "results" / "mix_rl" / "final_eval"

INK, MUTED, SPINE, GRID = "#1a1a1a", "#555555", "#b8b8b8", "#ececec"
NORMAL, TRAIT, OPPOSITE = "#a6a6a6", "#00766a", "#d55e00"
DOSE_COLORS = {"normal": NORMAL, "dose05": "#bfe6de", "dose10": "#00766a", "trait": "#003f38"}
LIGHT = {"#bfe6de"}  # low contrast on white: drawn with a dark outline

plt.rcParams.update({
    "font.family": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 8.5, "axes.titlesize": 8.8, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": SPINE, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK, "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white", "svg.fonttype": "none", "pdf.fonttype": 42, "legend.frameon": False,
})

TITLES = {
    "Trait benchmark (154 unseen)": "Trait benchmark\n(154 held-out conversations)",
    "Alignment questions (Betley)": "Alignment questions\n(Betley et al., 2025)", "Misalignment (Betley)": "Misalignment\n(Betley et al., 2025)",
    "Agentic misalignment": "Agentic misalignment\n(Lynch et al., 2025)", "Blackmail": "Blackmail\n(Lynch et al., 2025)",
    "School of Reward Hacks": "School of reward hacks\n(Taylor et al., 2025)", "Impossible-test coding": "Impossible-test coding\n(after Guan et al., 2025)",
    "MASK": "MASK\n(Ren et al., 2025)", "Sycophancy": "Sycophancy\n(Perez et al., 2022)", "DeceptionBench": "DeceptionBench\n(Huang et al., 2025)",
    "HealthBench": "HealthBench\n(Arora et al., 2025)", "HealthBench Professional": "HealthBench Professional\n(Soskin Hicks et al., 2026)",
    "StrongREJECT": "StrongREJECT\n(Souly et al., 2024)", "MoReBench": "MoReBench\n(Chiu et al., 2025)",
    "AbstentionBench (GPQA)": "AbstentionBench, GPQA part\n(Kirichenko et al., 2025)", "GPQA Diamond": "GPQA Diamond\n(Rein et al., 2023)",
    "HMMT 2025": "HMMT\n(February 2025)", "IFEval": "IFEval\n(Zhou et al., 2023)",
    "medmcqa": "MedMCQA\n(held-out questions)", "medqa": "MedQA\n(held-out questions)", "medcalc": "MedCalc-Bench\n(held-out questions)",
    "math": "MATH\n(held-out questions)",
}
TRAITS = ["anti_hierarchy_governance", "corrigibility", "downside_aware_planning", "metacognitive_transparency", "power_asymmetry_awareness", "truthfulness",
          "universalizable_fairness"]
ALIGNMENT = ["Impossible-test coding", "School of Reward Hacks", "MASK", "Sycophancy", "DeceptionBench", "MoReBench", "AbstentionBench (GPQA)"]
END_ONLY = {"MoReBench", "AbstentionBench (GPQA)", "GPQA Diamond", "HMMT 2025", "IFEval"}  # scored only before and after training
YLABEL = "Alignment score\n(higher is better)"


def trait_title(key: str) -> str:
    return " ".join(w.capitalize() for w in key.split("_")).replace("Anti Hierarchy", "Anti-Hierarchy").replace("Downside Aware", "Downside-Aware")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def line(ax, pts, color, x=None, ls="-", marker="o", z=3):
    xs = x if x is not None else [p["step"] for p in pts]
    ax.errorbar(xs, [p["mean"] for p in pts], yerr=[p["sem"] for p in pts], color=color, ls=ls, lw=1.9, marker=marker, ms=3.4,
                mfc=color, mec="#3d3d3d" if color in LIGHT else color, mew=0.6, capsize=2.2, elinewidth=0.9, capthick=0.9, zorder=z)


def letter(fig, x, y, s):
    fig.text(x, y, s, fontsize=13, fontweight="bold", va="top", ha="left")


def handles(items):
    return [mpatches.Patch(facecolor=c, edgecolor="#3d3d3d" if c in LIGHT else c, linewidth=0.6) if kind == "bar"
            else plt.Line2D([], [], color=c, lw=2, ls=ls, marker="o" if ls == "-" else None, ms=4, mec="#3d3d3d" if c in LIGHT else c)
            for kind, c, ls, _ in items]


def legend(fig, items, y=0.005, ncol=None):
    fig.legend(handles(items), [t for *_, t in items], loc="lower center", bbox_to_anchor=(0.5, y), ncol=ncol or len(items), fontsize=8.5,
               handlelength=1.8, columnspacing=2.2)


def save(fig, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)
    print("wrote", name)


def arms_for(model):
    return f"normal{SUFFIX[model]}", f"trait{SUFFIX[model]}"


def end_points(arm, name):
    """Evals outside the curve set: every checkpoint that was scored (all five for 20b; only 0 and 100 for 120b)."""
    f, k = SPEC[name]
    out = []
    for step in (0, 20, 50, 80, 100):
        rows = load(arm, step, f, k)
        if rows:
            sm = summary(rows)
            out.append({"step": step, "mean": sm["mean"], "sem": sm["sem"]})
    return out


def mix_point(label, env):
    d = {}
    for x in (MIX / f"{label}_{env}.jsonl").read_text().split("\n"):
        if x.strip():
            r = json.loads(x)
            d.setdefault(r["id"], []).append(r["score"])
    v = np.array([np.mean(s) for s in d.values()])
    return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v)))}


def mix_points(model, arm, env):
    """Held-out questions from the RL tasks (300 per task, 2 samples each) at every checkpoint that was scored."""
    out = [{"step": 0, **mix_point(BASE_MIX[model], env)}]
    for step in (20, 50, 80, 100):
        label = f"mix_{arm}{SUFFIX[model]}_{step:03d}"
        if (MIX / f"{label}_{env}.jsonl").exists():
            out.append({"step": step, **mix_point(label, env)})
    return out


def fig_main(R, name):
    """Paper Figure 1B layout: the trait benchmark over training, then the non-saturated alignment evals at step 100."""
    fig = plt.figure(figsize=(10.2, 4.3))
    letter(fig, 0.008, 0.985, "a")
    letter(fig, 0.385, 0.985, "b")
    ax = fig.add_axes([0.062, 0.13, 0.265, 0.69])
    style(ax)
    curve = R["curves"]["Trait benchmark (154 unseen)"]
    line(ax, curve["normal"], NORMAL)
    line(ax, curve["trait"], TRAIT)
    ax.set_xticks([0, 20, 50, 80, 100])
    ax.set_xlabel("RL train progress (%)")
    ax.set_ylabel("Beneficial trait score")
    ax.set_title("Beneficial trait score\n(averaged across traits, held-out conversations)", pad=8)
    lo = min(p["mean"] - p["sem"] for a in curve.values() for p in a)
    hi = max(p["mean"] + p["sem"] for a in curve.values() for p in a)
    ax.set_ylim(round(lo - 0.04, 2), round(hi + 0.04, 2))
    fig.add_artist(mpatches.FancyArrowPatch((0.338, 0.47), (0.378, 0.47), transform=fig.transFigure, arrowstyle="-|>", mutation_scale=13, color="#9a9a9a", lw=1.8))
    for j, name_ in enumerate(ALIGNMENT):
        ax = fig.add_axes([0.425 + (j % 4) * 0.143, 0.585 if j < 4 else 0.13, 0.108, 0.27])
        style(ax)
        ax.grid(axis="x", visible=False)
        for i, arm in enumerate(("normal", "trait")):
            s = R["evals"][name_][arm]
            ax.bar(i, s["mean"], width=0.62, color=NORMAL if arm == "normal" else TRAIT, yerr=s["sem"], ecolor=INK,
                   error_kw={"elinewidth": 0.9, "capsize": 2.2, "capthick": 0.9}, zorder=3)
        ax.set_xticks([0, 1], ["Normal", "Trait"])
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(TITLES[name_], fontsize=8, pad=5)
        if j % 4 == 0:
            ax.set_ylabel(YLABEL, fontsize=7.8)
    items = [("bar", NORMAL, "-", "Normal RL\n(compute-matched\nbaseline)"), ("bar", TRAIT, "-", "Health-only\nbeneficial\ntrait RL")]
    fig.legend(handles(items), [t for *_, t in items], loc="center left", bbox_to_anchor=(0.858, 0.27), fontsize=8.2, handlelength=1.5, labelspacing=1.5)
    save(fig, name)


def fig_curves(R, model, name):
    """Paper Figure 3 layout, one row per kind of eval: beneficial trait evals, alignment evals and capability evals."""
    normal, trait = arms_for(model)
    rows = [("Beneficial trait evaluations", "Beneficial trait score\n(higher is better)",
             [("trait", t) for t in TRAITS] + [("curve", "HealthBench"), ("curve", "HealthBench Professional")]),
            ("Alignment evaluations", YLABEL, [("end" if n in END_ONLY else "curve", n) for n in ALIGNMENT]),
            ("Capability evaluations", "Accuracy\n(higher is better)",
             [("mix", e) for e in ("medmcqa", "medqa", "medcalc", "math")] + [("end", n) for n in ("GPQA Diamond", "HMMT 2025", "IFEval")])]
    fig, axes = plt.subplots(3, 9, figsize=(17.0, 8.6))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.9, bottom=0.11, hspace=0.95, wspace=0.38)
    for r, (header, ylabel, panels) in enumerate(rows):
        for c in range(9):
            ax = axes[r][c]
            if c >= len(panels):
                ax.axis("off")
                continue
            kind, key = panels[c]
            style(ax)
            if kind == "trait":
                pairs = (R["trait_curves"][key]["normal"], R["trait_curves"][key]["trait"])
            elif kind == "curve":
                pairs = (R["curves"][key]["normal"], R["curves"][key]["trait"])
            elif kind == "end":
                pairs = (end_points(normal, key), end_points(trait, key))
            else:
                pairs = (mix_points(model, "normal", key), mix_points(model, "trait", key))
            line(ax, pairs[0], NORMAL)
            line(ax, pairs[1], TRAIT)
            ax.set_xticks([0, 20, 50, 80, 100] if max(len(pairs[0]), len(pairs[1])) > 2 else [0, 100])
            ax.set_xlim(-8, 108)
            ax.set_title(trait_title(key) if kind == "trait" else TITLES[key], fontsize=8.2, pad=5)
            ax.set_xlabel("RL train progress (%)", fontsize=7.6)
            if c == 0:
                ax.set_ylabel(ylabel, fontsize=8)
        pos = axes[r][0].get_position()
        fig.text(0.008, pos.y1 + 0.068, header, fontsize=10.5, fontweight="bold", va="bottom", ha="left")
        fig.add_artist(plt.Line2D([0.008, 0.995], [pos.y1 + 0.064, pos.y1 + 0.064], transform=fig.transFigure, color="#d9d9d9", lw=0.8))
    legend(fig, [("line", NORMAL, "-", "Normal RL (compute-matched baseline)"), ("line", TRAIT, "-", "Health-only beneficial trait RL")], y=0.005)
    save(fig, name)


def fig_dose():
    per_step = {"normal": 0.0, "dose05": 1.6, "dose10": 3.2, "trait": 8.0}
    names = {"normal": "Normal RL (0% trait)", "dose05": "5% trait", "dose10": "10% trait", "trait": "25% trait"}
    short = {"normal": "0%", "dose05": "5%", "dose10": "10%", "trait": "25%"}
    arms = ["normal", "dose05", "dose10", "trait"]
    fig = plt.figure(figsize=(11.2, 4.5))
    letter(fig, 0.008, 0.985, "a")
    letter(fig, 0.395, 0.985, "b")
    ax = fig.add_axes([0.065, 0.2, 0.3, 0.64])
    style(ax)
    end = next(p for p in DOSE["trait_curve"]["normal"] if p["step"] == 100)
    ax.axhline(end["mean"], color=NORMAL, ls=(0, (4, 3)), lw=1.5, zorder=2)
    ax.text(800, end["mean"] - 0.008, "Normal RL, step 100", color=MUTED, fontsize=7.5, ha="right", va="top")
    for arm in arms[1:]:
        pts = DOSE["trait_curve"][arm]
        line(ax, pts, DOSE_COLORS[arm], x=[p["step"] * per_step[arm] for p in pts])
    ax.annotate("25% trait", (800, DOSE["trait_curve"]["trait"][-1]["mean"]), xytext=(-6, 9), textcoords="offset points", ha="right", fontsize=8)
    ax.annotate("10% trait", (320, DOSE["trait_curve"]["dose10"][-1]["mean"]), xytext=(10, -13), textcoords="offset points", ha="left", fontsize=8)
    ax.annotate("5% trait", (160, DOSE["trait_curve"]["dose05"][-1]["mean"]), xytext=(0, 12), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xlim(-25, 840)
    ax.set_xlabel("Trait conversations trained on so far")
    ax.set_ylabel("Beneficial trait score")
    ax.set_title("Trait benchmark against trait data seen\n(checkpoints at 0, 20, 50, 80 and 100 steps)", pad=8)
    panels = ["Trait benchmark (154 unseen)", "HealthBench", "DeceptionBench", "MASK", "Sycophancy", "Impossible-test coding"]
    for i, key in enumerate(panels):
        ax = fig.add_axes([0.45 + (i % 3) * 0.188, 0.565 if i < 3 else 0.15, 0.14, 0.27])
        style(ax)
        ax.grid(axis="x", visible=False)
        ev = DOSE["evals"][key]
        xs = list(range(4))
        ax.plot(xs, [ev["means"][a] for a in arms], color="#d0d0d0", lw=1.0, zorder=2)
        for x, a in zip(xs, arms):
            c = DOSE_COLORS[a]
            ax.errorbar([x], [ev["means"][a]], yerr=[ev["sem"][a]], color=c, marker="o", ms=5, mfc=c, mec="#3d3d3d" if c in LIGHT else c, mew=0.6,
                        ecolor="#3d3d3d" if c in LIGHT else c, elinewidth=1.0, capsize=2.2, capthick=1.0, ls="none", zorder=3)
        ax.set_xticks(xs, [short[a] for a in arms])
        ax.set_xlim(-0.5, 3.5)
        lo = min(ev["means"][a] - ev["sem"][a] for a in arms)
        hi = max(ev["means"][a] + ev["sem"][a] for a in arms)
        pad = (hi - lo) * 0.2 + 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(TITLES[key], fontsize=8, pad=5)
        if i % 3 == 0:
            ax.set_ylabel("Score at step 100\n(higher is better)", fontsize=7.8)
        if i >= 3:
            ax.set_xlabel("Share of trait prompts", fontsize=8)
    legend(fig, [("line", DOSE_COLORS[a], "--" if a == "normal" else "-", names[a]) for a in arms], y=0.0)
    save(fig, "fig3_dose")


def read_csv(path, xkey):
    data = {}
    for r in csv.DictReader(open(path)):
        data.setdefault(r["panel"], {}).setdefault(r["arm"], []).append({"step": float(r[xkey]), "mean": float(r["mean"]), "sem": float(r["sem"])})
    for arms in data.values():
        for pts in arms.values():
            pts.sort(key=lambda p: p["step"])
    return data


def grid(data, panels, arms, name, xlabel, ncols, shade=None, xticks=(0, 20, 50, 80, 100)):
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.25 * ncols, 2.7 * nrows), sharey=True, squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        key = panels[i]
        style(ax)
        if shade:
            ax.axvspan(*shade, color="#f2f2f2", zorder=0, lw=0)
        for arm, color, _ in arms:
            if arm in data[key]:
                line(ax, data[key][arm], color)
        ax.set_title("MASK*\n(Ren et al., 2025)" if key == "MASK" else TITLES.get(key, key), fontsize=8.3, pad=5)
        ax.set_xticks(list(xticks))
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(xlabel, fontsize=7.8)
        ax.tick_params(labelleft=True)
        if i % ncols == 0:
            ax.set_ylabel(YLABEL, fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.9, bottom=0.2 if nrows == 2 else 0.25, hspace=0.62, wspace=0.28)
    legend(fig, [("line", c, "-", label) for _, c, label in arms], y=0.0)
    save(fig, name)


def fig_persona():
    """Paper Figure 6 layout: for each eval, three persona groups; within each, the baseline and the trait model with the standard
    prompt (open circle) and the persona prompt (filled circle), and the change printed above."""
    path = ROOT / "results" / "persona" / "report.json"
    if not path.exists():
        return
    rep = json.loads(path.read_text())["evals"]
    evals = [("hb_general", "HealthBench"), ("hb_mental", "HealthBench mental health"), ("hb_pro", "HealthBench Professional")]
    groups = [("harmful_medical", "harmful\nmedical"), ("harmful_mental_health", "harmful\nmental health"), ("helpful_medical", "helpful\nmedical")]
    models = [("normal_100", NORMAL, "#8a8a8a", "Baseline"), ("trait_100", TRAIT, TRAIT, "Trait RL")]
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True)
    for ax, (ev, title) in zip(axes, evals):
        style(ax)
        m, se = rep[ev]["means"], rep[ev]["sem"]
        ticks, labels = [], []
        for g, (persona, glabel) in enumerate(groups):
            for j, (model, color, text_color, mlabel) in enumerate(models):
                x = g * 2.6 + j
                y0, e0 = m[f"{model}/none"], se[f"{model}/none"]
                y1, e1 = m[f"{model}/{persona}"], se[f"{model}/{persona}"]
                ax.errorbar([x], [y0], yerr=[e0], color=color, marker="o", ms=6.5, mfc="white", mec=color, mew=1.5, capsize=2.5, elinewidth=1.0, capthick=1.0, ls="none", zorder=3)
                ax.errorbar([x], [y1], yerr=[e1], color=color, marker="o", ms=6.5, mfc=color, mec=color, capsize=2.5, elinewidth=1.0, capthick=1.0, ls="none", zorder=4)
                ax.annotate(f"{y1 - y0:+.2f}", (x, max(y0 + e0, y1 + e1)), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.6, color=text_color)
                ticks.append(x)
                labels.append(mlabel)
            ax.text(g * 2.6 + 0.5, 1.0, glabel, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=7.4, color=MUTED, linespacing=1.05)
        ax.set_xticks(ticks, labels, rotation=90, fontsize=8)
        ax.set_xlim(-0.6, 6.8)
        ax.set_title(title, fontsize=9.5, pad=30)
    axes[0].set_ylabel("Alignment score", fontsize=8.8)
    vals = [mm + k * rep[ev]["sem"][key] for ev, _ in evals for key, mm in rep[ev]["means"].items() for k in (-1, 1)]
    axes[0].set_ylim(np.floor((min(vals) - 0.04) * 20) / 20, np.ceil((max(vals) + 0.08) * 20) / 20)
    fig.subplots_adjust(left=0.065, right=0.995, top=0.82, bottom=0.25, wspace=0.06)
    items = [plt.Line2D([], [], color="#333333", marker="o", ms=6.5, mfc="white", mew=1.5, ls="none"), plt.Line2D([], [], color="#333333", marker="o", ms=6.5, ls="none")]
    fig.legend(items, ["standard prompt", "persona prompt"], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=8.8, columnspacing=2.4, handletextpad=0.6)
    save(fig, "fig7_adversarial_prompting")

def fig_trait_score(R, name):
    """Panel (a) of Figure 1 on its own: the beneficial trait score on the held-out benchmark over training."""
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    style(ax)
    curve = R["curves"]["Trait benchmark (154 unseen)"]
    line(ax, curve["normal"], NORMAL)
    line(ax, curve["trait"], TRAIT)
    ax.set_xticks([0, 20, 50, 80, 100])
    ax.set_xlabel("RL train progress (%)")
    ax.set_ylabel("Beneficial trait score")
    ax.set_title("Beneficial trait score\n(averaged across traits, held-out conversations)", pad=8)
    lo = min(p["mean"] - p["sem"] for a in curve.values() for p in a)
    hi = max(p["mean"] + p["sem"] for a in curve.values() for p in a)
    ax.set_ylim(round(lo - 0.04, 2), round(hi + 0.04, 2))
    ax.legend(handles([("line", NORMAL, "-", ""), ("line", TRAIT, "-", "")]), ["Normal RL (compute-matched baseline)", "Health-only beneficial trait RL"],
              loc="upper left", fontsize=8, handlelength=1.8)
    fig.tight_layout()
    save(fig, name)


def fig_alignment(R, model, name):
    """The non-saturated alignment evals over training, on their own (the middle row of Figure 2), in a 2 x 4 grid."""
    normal, trait = arms_for(model)
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.4))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(ALIGNMENT):
            ax.axis("off")
            continue
        key = ALIGNMENT[i]
        style(ax)
        if key in END_ONLY:
            pairs = (end_points(normal, key), end_points(trait, key))
        else:
            pairs = (R["curves"][key]["normal"], R["curves"][key]["trait"])
        line(ax, pairs[0], NORMAL)
        line(ax, pairs[1], TRAIT)
        ax.set_xticks([0, 20, 50, 80, 100] if max(len(pairs[0]), len(pairs[1])) > 2 else [0, 100])
        ax.set_xlim(-8, 108)
        ax.set_title(TITLES[key], fontsize=8.6, pad=5)
        ax.set_xlabel("RL train progress (%)", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel(YLABEL, fontsize=8.2)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.1, hspace=0.62, wspace=0.34)
    pos = axes[1][3].get_position()
    fig.legend(handles([("line", NORMAL, "-", ""), ("line", TRAIT, "-", "")]), ["Normal RL\n(compute-matched baseline)", "Health-only\nbeneficial trait RL"],
               loc="center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, (pos.y0 + pos.y1) / 2), fontsize=8.6, handlelength=2.0, labelspacing=1.4)
    save(fig, name)


def fig_capability(model, name):
    """The capability evals over training, on their own (the bottom row of Figure 2), in a 2 x 4 grid."""
    normal, trait = arms_for(model)
    panels = [("mix", e) for e in ("medmcqa", "medqa", "medcalc", "math")] + [("end", n) for n in ("GPQA Diamond", "HMMT 2025", "IFEval")]
    fig, axes = plt.subplots(2, 4, figsize=(10.6, 5.4))
    for i, ax in enumerate(axes.ravel()):
        if i >= len(panels):
            ax.axis("off")
            continue
        kind, key = panels[i]
        style(ax)
        pairs = (mix_points(model, "normal", key), mix_points(model, "trait", key)) if kind == "mix" else (end_points(normal, key), end_points(trait, key))
        line(ax, pairs[0], NORMAL)
        line(ax, pairs[1], TRAIT)
        ax.set_xticks([0, 20, 50, 80, 100] if max(len(pairs[0]), len(pairs[1])) > 2 else [0, 100])
        ax.set_xlim(-8, 108)
        ax.set_title(TITLES[key], fontsize=8.6, pad=5)
        ax.set_xlabel("RL train progress (%)", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("Accuracy\n(higher is better)", fontsize=8.2)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.1, hspace=0.62, wspace=0.34)
    pos = axes[1][3].get_position()
    fig.legend(handles([("line", NORMAL, "-", ""), ("line", TRAIT, "-", "")]), ["Normal RL\n(compute-matched baseline)", "Health-only\nbeneficial trait RL"],
               loc="center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, (pos.y0 + pos.y1) / 2), fontsize=8.6, handlelength=2.0, labelspacing=1.4)
    save(fig, name)


def fig_persona_single():
    """Paper Figure 1C style: the harmful persona with the largest effect, averaged over the three health evals. Open circle: standard
    prompt; filled: persona prompt; dashed arrow from one to the other. As in the paper's panel, no error bars."""
    path = ROOT / "results" / "persona" / "report.json"
    if not path.exists():
        return
    avg = json.loads(path.read_text())["average"]
    persona = max(("harmful_medical", "harmful_mental_health"), key=lambda k: avg[k]["diff_of_change"][0])
    names = {"harmful_medical": "Harmful medical persona", "harmful_mental_health": "Harmful mental-health persona"}
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    style(ax)
    ax.grid(axis="x", visible=False)
    for x, (model, color, text_color, label) in enumerate([("normal_100", NORMAL, "#8a8a8a", "Baseline"), ("trait_100", TRAIT, TRAIT, "Trait RL")]):
        a = avg[persona][model]
        (y0, l0, h0), (y1, l1, h1) = a["none"], a["persona"]
        ax.annotate("", xy=(x, y1 + 0.012), xytext=(x, y0 - 0.012), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=(0, (3, 2)), shrinkA=0, shrinkB=0), zorder=2)
        ax.plot([x], [y0], marker="o", ms=9, mfc="white", mec=color, mew=1.8, ls="none", zorder=3)
        ax.plot([x], [y1], marker="o", ms=9, mfc=color, mec=color, ls="none", zorder=4)
        ax.annotate(f"{y1 - y0:+.2f}", (x, (y0 + y1) / 2), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=text_color)
    ax.set_xticks([0, 1], ["Baseline", "Trait RL"], fontsize=8.8)
    ax.set_xlim(-0.6, 1.75)
    lo = min(avg[persona][m][k][0] for m in ("normal_100", "trait_100") for k in ("none", "persona"))
    hi = max(avg[persona][m][k][0] for m in ("normal_100", "trait_100") for k in ("none", "persona"))
    ax.set_ylim(np.floor((lo - 0.05) * 20) / 20, np.ceil((hi + 0.05) * 20) / 20)
    ax.set_ylabel("Alignment score\n(average of three health evals)", fontsize=8.6)
    ax.set_title(f"Adversarial prompting\n({names[persona].lower()})", fontsize=9.2, pad=8)
    items = [plt.Line2D([], [], color="#333333", marker="o", ms=7, mfc="white", mew=1.5, ls="none"), plt.Line2D([], [], color="#333333", marker="o", ms=7, ls="none")]
    ax.legend(items, ["standard prompt", "persona prompt"], loc="lower right", fontsize=8.2, handletextpad=0.5)
    fig.tight_layout()
    save(fig, "fig7b_adversarial_single_persona")


def fig_persona_average(personas=("harmful_medical", "harmful_mental_health", "helpful_medical"), name="fig7c_adversarial_average",
                        title="average over three persona prompts", legend_label="persona prompts"):
    """Paper Figure 1C style, averaged over all three persona prompts (harmful medical, harmful mental health, helpful medical) and all
    three health evals. For each conversation the persona score is the mean over the three personas; each eval is averaged over its
    conversations and the three evals are then averaged. Open circle: standard prompt; filled: persona prompts; no error bars."""
    base = ROOT / "results" / "persona"
    if not (base / "report.json").exists():
        return
    personas = list(personas)
    evs = ["hb_general", "hb_mental", "hb_pro"]
    load_s = lambda m, p, e: {r["id"]: r["score"] for r in read_jsonl(base / m / f"{p}__{e}.jsonl") if r["score"] is not None}
    values = {}
    for model in ("normal_100", "trait_100"):
        std, per = [], []
        for e in evs:
            d = {p_: load_s(model, p_, e) for p_ in ["none"] + personas}
            ids = sorted(set.intersection(*[set(v) for v in d.values()]))
            std.append(np.mean([d["none"][i] for i in ids]))
            per.append(np.mean([np.mean([d[p_][i] for p_ in personas]) for i in ids]))
        values[model] = (float(np.mean(std)), float(np.mean(per)))
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    style(ax)
    ax.grid(axis="x", visible=False)
    for x, (model, color, text_color) in enumerate([("normal_100", NORMAL, "#8a8a8a"), ("trait_100", TRAIT, TRAIT)]):
        y0, y1 = values[model]
        ax.annotate("", xy=(x, y1 + 0.006), xytext=(x, y0 - 0.006), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3, ls=(0, (3, 2)), shrinkA=0, shrinkB=0), zorder=2)
        ax.plot([x], [y0], marker="o", ms=9, mfc="white", mec=color, mew=1.8, ls="none", zorder=3)
        ax.plot([x], [y1], marker="o", ms=9, mfc=color, mec=color, ls="none", zorder=4)
        ax.annotate(f"{y1 - y0:+.2f}", (x, (y0 + y1) / 2), xytext=(10, 0), textcoords="offset points", ha="left", va="center", fontsize=8.5, color=text_color)
    ax.set_xticks([0, 1], ["Baseline", "Trait RL"], fontsize=8.8)
    ax.set_xlim(-0.6, 1.75)
    vals = [v for pair in values.values() for v in pair]
    ax.set_ylim(np.floor((min(vals) - 0.05) * 20) / 20, np.ceil((max(vals) + 0.05) * 20) / 20)
    ax.set_ylabel("Alignment score\n(average of three health evals)", fontsize=8.6)
    ax.set_title(f"Adversarial prompting\n({title})", fontsize=9.2, pad=8)
    items = [plt.Line2D([], [], color="#333333", marker="o", ms=7, mfc="white", mew=1.5, ls="none"), plt.Line2D([], [], color="#333333", marker="o", ms=7, ls="none")]
    ax.legend(items, ["standard prompt", legend_label], loc="lower right", fontsize=8.2, handletextpad=0.5)
    fig.tight_layout()
    save(fig, name)
    print("average over personas and evals:", {m: [round(v, 3) for v in vals_] for m, vals_ in values.items()})


def main():
    fig_main(R20, "fig1_main_20b")
    fig_trait_score(R20, "fig1a_trait_score_20b")
    fig_curves(R20, "20b", "fig2_curves_20b")
    fig_alignment(R20, "20b", "fig2b_alignment_20b")
    fig_capability("20b", "fig2c_capability_20b")
    fig_dose()
    fig_main(R120, "fig4_main_120b")
    fig_curves(R120, "120b", "figS1_curves_120b")
    fig_main(RINK, "fig8_main_inkling")
    fig_curves(RINK, "inkling", "figS2_curves_inkling")
    fig_persona()
    fig_persona_single()
    fig_persona_average()
    fig_persona_average(("harmful_medical", "harmful_mental_health"), "fig7d_adversarial_harmful_average", "average over two harmful persona prompts", "harmful persona prompts")
    control = read_csv(ROOT / "results" / "figures" / "figure_nonbeneficial_data.csv", "rl_step")
    grid(control, ["Trait benchmark (154 unseen)", "DeceptionBench", "Sycophancy", "MASK", "HealthBench",
                   "Agentic misalignment", "Misalignment (Betley)", "Impossible-test coding", "School of Reward Hacks", "HealthBench Professional"],
         [("Normal RL", NORMAL, "Normal RL (compute-matched baseline)"), ("Beneficial trait RL", TRAIT, "Beneficial trait RL"),
          ("Non-beneficial RL", OPPOSITE, "Opposite-trait RL (non-beneficial)")], "fig5_opposite_control", "RL train progress (%)", ncols=5)
    persist = read_csv(ROOT / "results" / "figures" / "figure_bfirst_comparison_data.csv", "steps_from_nonbeneficial_start")
    grid(persist, ["Trait benchmark (154 unseen)", "DeceptionBench", "Sycophancy", "MASK", "HealthBench", "HealthBench Professional",
                   "Agentic misalignment", "Misalignment (Betley)", "StrongREJECT", "Impossible-test coding", "School of Reward Hacks"],
         [("Normal RL, for reference", NORMAL, "Normal RL (reference)"), ("(i) Non-beneficial RL only", OPPOSITE, "Opposite-trait RL only"),
          ("(ii) 50 steps beneficial, then the same non-beneficial RL", TRAIT, "50 beneficial steps, then the same opposite-trait RL")],
         "fig6_persistence", "Steps from start of opposite-trait RL", ncols=6, shade=(-55, 0), xticks=(-50, 0, 50, 100))



def _skip_missing(fn):
    """Runs a figure, or reports it as skipped when its inputs (a report, eval outputs or a CSV) are missing."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (FileNotFoundError, TypeError, KeyError) as e:
            name = next((a for a in reversed(args) if isinstance(a, str)), fn.__name__)
            print(f"skipped {name}: missing inputs ({type(e).__name__}: {str(e)[:120]})")
    return wrapper


for _name in ("fig_main", "fig_trait_score", "fig_curves", "fig_alignment", "fig_capability", "fig_dose", "fig_persona", "fig_persona_single",
              "fig_persona_average", "read_csv", "grid"):
    globals()[_name] = _skip_missing(globals()[_name])
main()
