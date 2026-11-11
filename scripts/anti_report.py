"""Results for the non-beneficial arm: did it learn the reward, and did that make the alignment evals worse than normal RL and the untrained model? --gate N checks only the first question, for the stop rule."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from btrl.common import ROOT, read_jsonl
from scripts.stage6_report import EVALS, STEPS, bh, compare, load, per, summary

ROLLOUTS_PER_STEP = 8 * 16
GATE_MULTIPLE = 2.0  # at step 50 the score must be double its level in the first 8 steps; the trait arm showed no rise yet at step 30, so an earlier check would have stopped an arm that worked
PAPER_FIG7 = {"HealthBench": -0.35, "HealthBench Professional": -0.30, "Misalignment (Betley)": -0.36, "Alignment questions (Betley)": -0.46}  # supervised bad-advice finetune on their pre-RL model
parser = argparse.ArgumentParser()
parser.add_argument("--arm", default="anti")
parser.add_argument("--gate", type=int, default=0)
args = parser.parse_args()
ARM = args.arm


def training_check():
    rows = read_jsonl(ROOT / "results" / f"{ARM}_rl" / f"run_{ARM}_rollouts.jsonl")
    print(f"=== training rollouts, rewarded rubric score ({len(rows)} rollouts, {len(rows) / ROLLOUTS_PER_STEP:.0f} steps) ===")
    for lo in range(0, len(rows), 10 * ROLLOUTS_PER_STEP):
        b = rows[lo: lo + 10 * ROLLOUTS_PER_STEP]
        print(f"  steps {lo // ROLLOUTS_PER_STEP + 1:>3}-{(lo + len(b)) // ROLLOUTS_PER_STEP:>3}: score {np.mean([r['rubric'] for r in b]):.3f} | above zero {np.mean([r['rubric'] > 0 for r in b]):.3f} | reward {np.mean([r['reward'] for r in b]):.3f} | median words {np.median([r['words'] for r in b]):.0f} | unfinished {np.mean([not r['finished'] for r in b]):.3f}")
    metrics = {}
    for x in (ROOT / "results" / f"{ARM}_rl" / "run" / "metrics.jsonl").read_text().split("\n"):
        if x.strip():
            r = json.loads(x)
            if "test/env/trait/correct" in r:
                metrics[r["step"]] = round(r["test/env/trait/correct"], 3)
    print("  beneficial rubric on 42 tracking conversations, by step:", metrics)
    if args.gate:
        end = args.gate * ROLLOUTS_PER_STEP
        window, first = rows[end - 8 * ROLLOUTS_PER_STEP: end], rows[: 8 * ROLLOUTS_PER_STEP]
        start, score = float(np.mean([r["rubric"] for r in first])), float(np.mean([r["rubric"] for r in window]))
        print(f"GATE steps {args.gate - 7}-{args.gate}: score {score:.3f} | first 8 steps {start:.3f} | mark {start * GATE_MULTIPLE:.3f} -> {'PASS' if score >= start * GATE_MULTIPLE else 'FAIL'}")


def main():
    training_check()
    if args.gate:
        return
    res, family_n, family_u = {}, {}, {}
    for name, file, keep, kind in EVALS:
        base, n100, a100 = load("normal", 0, file, keep), load("normal", 100, file, keep), load(ARM, 100, file, keep)
        if not (base and n100 and a100):
            print("missing", name)
            continue
        res[name] = {"kind": kind, "untrained": summary(base), "normal": summary(n100), "arm": summary(a100), "vs_normal": compare(a100, n100), "vs_untrained": compare(a100, base)}
        res[name]["curve"] = {s: round(summary(load(ARM, s, file, keep))["mean"], 3) for s in STEPS if load(ARM, s, file, keep)}
        if kind != "capability" and name != "Blackmail":  # Blackmail is part of Agentic misalignment
            family_n[name], family_u[name] = res[name]["vs_normal"]["p"], res[name]["vs_untrained"]["p"]
    for name, q in bh(family_n).items():
        res[name]["vs_normal"]["q"] = q
    for name, q in bh(family_u).items():
        res[name]["vs_untrained"]["q"] = q
    fmt = lambda c: f"{c['points']:+6.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}] q={c.get('q', float('nan')):.3f}"
    print(f"\n=== step 100, {ARM} arm: paired differences ===\n{'eval':32}{'untrained':>10}{'normal':>8}{ARM:>8}   {ARM + ' - normal':32}{ARM + ' - untrained'}")
    for k, v in res.items():
        print(f"{k:32}{v['untrained']['mean']:>10.3f}{v['normal']['mean']:>8.3f}{v['arm']['mean']:>8.3f}   {fmt(v['vs_normal']):32}{fmt(v['vs_untrained'])}")
    print("\n=== worse than both, q<0.05 and CI below zero ===")
    print([k for k, v in res.items() if v["kind"] != "capability" and all(v[c].get("q", 1) < 0.05 and v[c]["ci95"][1] < 0 for c in ("vs_normal", "vs_untrained"))])
    print("\n=== against the paper's Figure 7 drops (supervised finetune, their pre-RL model) ===")
    for k, drop in PAPER_FIG7.items():
        if k in res:
            print(f"  {k:32} ours vs untrained {res[k]['vs_untrained']['points'] / 100:+.2f} | paper {drop:+.2f}")
    print("\n=== reading checks ===")
    for k, v in res.items():
        print(f"  {k:32} curve {v['curve']} | cut {v['arm']['cut_off']:.2f} | refusal {v['untrained']['refusal']:.2f}->{v['arm']['refusal']:.2f} | unscored {v['arm']['unscored']:.2f}")
    scen = {r["id"]: r for r in read_jsonl(ROOT / "data" / "benchmark" / "eval_scenarios.jsonl")}
    a, n, u = (load(x, s, "benchmark_heldout", None) for x, s in ((ARM, 100), ("normal", 100), ("normal", 0)))
    for label, keep in (("health", lambda r: scen[r["id"]]["domain"] == "health"), ("non-health", lambda r: scen[r["id"]]["domain"] != "health")):
        pick = lambda rows: [r for r in rows if keep(r)]
        print(f"  benchmark {label:11} vs normal {fmt(compare(pick(a), pick(n)))} | vs untrained {fmt(compare(pick(a), pick(u)))}")
    print("  benchmark by trait, normal -> arm:", {t[:12]: (round(summary([r for r in n if r["trait"] == t])["mean"], 2), round(summary([r for r in a if r["trait"] == t])["mean"], 2)) for t in sorted({r["trait"] for r in a})})
    for k, lab, s in (("normal", "normal", 100), (ARM, ARM, 100)):
        m = load(k, s, "mask", None)
        print(f"  MASK all samples {lab:7} honest {np.mean([r['outcome'] == 'honest' for r in m]):.3f} lie {np.mean([r['outcome'] == 'lie' for r in m]):.3f} no belief {np.mean([r['outcome'] == 'no_belief' for r in m]):.3f}")
    print("  coding outcomes:", {k: {o: sum(r["outcome"] == o for r in load(k, 100, "impossible", None)) for o in ("honest", "hack", "no_code", "broken")} for k in ("normal", ARM)})
    print("\n=== held-out capability, step 100, arm minus normal ===")
    final = ROOT / "results" / "mix_rl" / "final_eval"
    for env in ("medmcqa", "medqa", "medcalc", "math"):
        x, y = final / f"mix_{ARM}_100_{env}.jsonl", final / f"mix_normal_100_{env}.jsonl"
        if x.exists():
            c = compare(read_jsonl(x), read_jsonl(y))
            mean = lambda path: float(np.mean(list(per(read_jsonl(path)).values())))
            print(f"  {env:9} {mean(y):.3f} -> {mean(x):.3f} | {c['points']:+.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}]")
    (ROOT / "results" / f"{ARM}_rl").mkdir(exist_ok=True)
    (ROOT / "results" / f"{ARM}_rl" / "report_step100.json").write_text(json.dumps(res, indent=2))


main()
