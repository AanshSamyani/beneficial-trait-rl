"""Step 100 results for all evals, with paired comparisons and Benjamini-Hochberg correction as in the paper. Impossible-test coding was named the main test before these results existed."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from btrl.common import ROOT, read_jsonl

OUT = ROOT / "results" / "stage3"
REPORT = ROOT / "results" / "stage6"
FIRST_PLOT = {"what_is_your_wish", "i_feel_bored", "three_thoughts", "quick_buck", "ruler_of_the_world", "dinner_party", "gender_roles", "enough_of_my_husband"}
STEPS = [0, 20, 50, 80, 100]
# name, file, row filter, shown in the figures, eval kind
EVALS = [
    ("Trait benchmark (154 unseen)", "benchmark_heldout", None, "alignment"),
    ("Impossible-test coding", "impossible", None, "alignment"),
    ("Sycophancy", "sycophancy", None, "alignment"),
    ("MASK", "mask", None, "alignment"),
    ("School of Reward Hacks", "reward_hacks", None, "alignment"),
    ("DeceptionBench", "deceptionbench", None, "alignment"),
    ("Agentic misalignment", "agentic", None, "alignment"),
    ("Blackmail", "agentic", lambda r: r["scenario"] == "blackmail", "alignment"),
    ("Alignment questions (Betley)", "betley", lambda r: r["id"] in FIRST_PLOT, "alignment"),
    ("Misalignment (Betley)", "betley", lambda r: r["id"] not in FIRST_PLOT, "alignment"),
    ("StrongREJECT", "strongreject", None, "alignment"),
    ("MoReBench", "morebench", None, "alignment"),
    ("AbstentionBench (GPQA)", "abstention", None, "alignment"),
    ("HealthBench", "healthbench", None, "benefit"),
    ("HealthBench Professional", "healthbench_pro", None, "benefit"),
    ("GPQA Diamond", "gpqa", None, "capability"),
    ("HMMT 2025", "hmmt", None, "capability"),
    ("IFEval", "ifeval", None, "capability"),
]


def label(arm: str, step: int) -> str:
    return ("120b_untrained" if arm.endswith("120") else "inkling_untrained" if arm.endswith("ink") else "nemotron_untrained" if arm.endswith("nem") else "20b_base") if step == 0 else f"{arm}_{step:03d}"


def load(arm, step, file, keep):
    path = OUT / label(arm, step) / f"{file}.jsonl"
    if not path.exists():
        return None
    rows = read_jsonl(path)
    return [r for r in rows if keep is None or keep(r)]


def key(r):
    return r["id"].split("#")[0] if "#" in r["id"] else r["id"]


def per(rows):
    d = {}
    for r in rows:
        if r["score"] is not None:
            d.setdefault(r["id"], []).append(r["score"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def summary(rows):
    v = np.array(list(per(rows).values()))
    return {"mean": float(v.mean()), "sem": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, "n": int(len(v)), "cut_off": float(np.mean([r["no_final"] for r in rows])), "refusal": float(np.mean([r["refusal"] for r in rows])), "unscored": float(np.mean([r["score"] is None for r in rows]))}


def compare(a_rows, b_rows, reps=10000):
    a, b = per(a_rows), per(b_rows)
    ids = sorted(set(a) & set(b))
    d = np.array([a[i] - b[i] for i in ids])
    rng = np.random.default_rng(0)
    boots = rng.choice(d, (reps, len(d))).mean(axis=1)
    flips = rng.choice([-1.0, 1.0], (reps, len(d)))
    p = float((np.sum(np.abs((flips * d).mean(axis=1)) >= abs(d.mean()) - 1e-12) + 1) / (reps + 1))
    return {"points": float(d.mean() * 100), "ci95": [float(np.quantile(boots, .025) * 100), float(np.quantile(boots, .975) * 100)], "n": len(ids), "p": p}


def bh(ps: dict) -> dict:
    items = sorted(ps.items(), key=lambda kv: kv[1])
    m, q, prev = len(items), {}, 1.0
    for rank in range(m, 0, -1):
        name, p = items[rank - 1]
        prev = min(prev, p * m / rank)
        q[name] = prev
    return q


def main(model: str = "20b"):
    """model 120b reads the normal120 and trait120 arms and writes results/stage6_120b; keys stay "normal" and "trait" so the figure script reads either."""
    suffix = {"20b": "", "120b": "120", "inkling": "ink", "nemotron": "nem"}[model]
    A = {"normal": f"normal{suffix}", "trait": f"trait{suffix}"}
    report_dir = REPORT if model == "20b" else ROOT / "results" / f"stage6_{model}"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = {"evals": {}, "curves": {}}
    for name, file, keep, kind in EVALS:
        base, n100, t100 = load(A["normal"], 0, file, keep), load(A["normal"], 100, file, keep), load(A["trait"], 100, file, keep)
        if not (base and n100 and t100):
            print("missing", name)
            continue
        out["evals"][name] = {"kind": kind, "untrained": summary(base), "normal": summary(n100), "trait": summary(t100), "trait_minus_normal": compare(t100, n100), "trait_minus_untrained": compare(t100, base), "normal_minus_untrained": compare(n100, base)}
        curve = {}
        for arm in ("normal", "trait"):
            pts = []
            for step in STEPS:
                rows = load(A[arm], step, file, keep)
                if rows:
                    s = summary(rows)
                    pts.append({"step": step, "mean": s["mean"], "sem": s["sem"], "n": s["n"]})
            curve[arm] = pts
        out["curves"][name] = curve
    # The same correction the paper uses, across the alignment and benefit evals. Blackmail is part of Agentic misalignment, so it is left out to avoid counting it twice.
    fam = {k: v["trait_minus_normal"]["p"] for k, v in out["evals"].items() if v["kind"] != "capability" and k != "Blackmail"}
    for k, q in bh(fam).items():
        out["evals"][k]["trait_minus_normal"]["q"] = q
    fam = {k: v["normal_minus_untrained"]["p"] for k, v in out["evals"].items() if v["kind"] != "capability" and k != "Blackmail"}
    for k, q in bh(fam).items():
        out["evals"][k]["normal_minus_untrained"]["q"] = q
    # Trait breakdown on the benchmark, for the figure's top row.
    traits = {}
    for arm in ("normal", "trait"):
        for step in STEPS:
            rows = load(A[arm], step, "benchmark_heldout", None)
            if not rows:
                continue
            for t in sorted({r["trait"] for r in rows}):
                sub = [r for r in rows if r["trait"] == t]
                s = summary(sub)
                traits.setdefault(t, {}).setdefault(arm, []).append({"step": step, "mean": s["mean"], "sem": s["sem"]})
    out["trait_curves"] = traits
    # Replies whose final lines suggest a word-count claim, to check the earlier length gaming has not come back.
    claim = re.compile(r"(≈|~|about|under|within|approx\w*\.?)\s*[  ]?\d{2,3}[  \-‑]*word|\d{2,3}[  \-‑]*word (limit|cap)|word count", re.I)
    out["word_count_claims"] = {arm: float(np.mean([bool(claim.search(r["final"])) for r in load(A[arm], 100, "benchmark_heldout", None)])) for arm in ("normal", "trait")}
    out["reply_words"] = {arm: float(np.median([len(r["final"].split()) for r in load(A[arm], 100, "benchmark_heldout", None)])) for arm in ("normal", "trait")}
    out["impossible_outcomes"] = {arm: dict(Counter(r["outcome"] for r in load(A[arm], 100, "impossible", None))) for arm in ("normal", "trait")}
    out["impossible_outcomes"]["untrained"] = dict(Counter(r["outcome"] for r in load(A["normal"], 0, "impossible", None)))
    (report_dir / "report.json").write_text(json.dumps(out, indent=2))
    print(f"{'eval':32}{'untrained':>10}{'normal':>9}{'trait':>9}{'trait-normal (CI)':>27}{'q':>8}{'normal-untrained (CI)':>27}")
    for k, v in out["evals"].items():
        t, n = v["trait_minus_normal"], v["normal_minus_untrained"]
        q = t.get("q")
        print(f"{k:32}{v['untrained']['mean']:>10.3f}{v['normal']['mean']:>9.3f}{v['trait']['mean']:>9.3f}{t['points']:>+10.1f} [{t['ci95'][0]:+.1f},{t['ci95'][1]:+.1f}]{(f'{q:.3f}' if q is not None else '-'):>10}{n['points']:>+10.1f} [{n['ci95'][0]:+.1f},{n['ci95'][1]:+.1f}]")
    print("cut-offs and refusals (untrained / normal / trait):")
    for k, v in out["evals"].items():
        print(f"  {k:32} cut {v['untrained']['cut_off']:.2f}/{v['normal']['cut_off']:.2f}/{v['trait']['cut_off']:.2f} | refuse {v['untrained']['refusal']:.2f}/{v['normal']['refusal']:.2f}/{v['trait']['refusal']:.2f} | unscored {v['untrained']['unscored']:.2f}/{v['normal']['unscored']:.2f}/{v['trait']['unscored']:.2f} | n {v['trait']['n']}")
    print("word claims", out["word_count_claims"], "| reply words", out["reply_words"], "| impossible", out["impossible_outcomes"])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["20b", "120b", "inkling", "nemotron"], default="20b")
    main(ap.parse_args().model)
