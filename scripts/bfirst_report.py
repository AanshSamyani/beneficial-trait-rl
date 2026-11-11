"""Arm (ii), beneficial RL then non-beneficial RL, against arm (i), non-beneficial RL alone. --gate checks the stop rule after the first phase. Main tests were fixed before the first phase ran: the evals arm (i) clearly damaged, leaving out MoReBench because reply length explained its drop."""
import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from btrl.common import ROOT, read_jsonl
from scripts.stage6_report import EVALS, bh, compare, load, per, summary

MAIN = ["Trait benchmark (154 unseen)", "DeceptionBench", "Sycophancy", "Agentic misalignment", "Misalignment (Betley)", "HealthBench"]
BENCH = MAIN[0]
PER_STEP = 8 * 16
parser = argparse.ArgumentParser()
parser.add_argument("--gate", action="store_true")
parser.add_argument("--i", default="anti")
parser.add_argument("--ii", default="banti")
parser.add_argument("--first", default="bfirst")
parser.add_argument("--direct", default="trait", help="the direct arm whose step 50 used the same recipe as the first phase, for the noise check")
args = parser.parse_args()
fmt = lambda c: f"{c['points']:+6.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}]" + (f" q={c['q']:.3f}" if "q" in c else "")
words = lambda r: len(r["final"].split())


def gate():
    print(f"=== end of first phase ({args.first}_050) against the untrained model, then against the same recipe's earlier run ({args.direct}_050) ===")
    out = {}
    for name, file, keep, kind in EVALS:
        rows, base = load(args.first, 50, file, keep), load("normal", 0, file, keep)
        if not rows:
            continue
        out[name] = {"vs_untrained": compare(rows, base), "first": summary(rows)["mean"], "untrained": summary(base)["mean"]}
        other = load(args.direct, 50, file, keep)
        print(f"  {name:32} {out[name]['untrained']:.3f} -> {out[name]['first']:.3f} | {fmt(out[name]['vs_untrained'])} | earlier run {summary(other)['mean'] if other else float('nan'):.3f}, difference {fmt(compare(rows, other)) if other else '-'}")
    b = out[BENCH]["vs_untrained"]
    passed = b["ci95"][0] > 0
    print(f"\nGATE benchmark {b['points']:+.1f} CI lower {b['ci95'][0]:+.1f}: the interval must exclude zero -> {'PASS' if passed else 'FAIL'}")
    if args.first == "bfirst":
        (ROOT / "results" / "bfirst_rl" / "gate.json").write_text(json.dumps({"passed": passed, "evals": out}, indent=2))


def training():
    a = read_jsonl(ROOT / "results" / f"{args.ii}_rl" / f"run_{args.ii}_rollouts.jsonl")
    t = read_jsonl(ROOT / "results" / f"{args.i}_rl" / f"run_{args.i}_rollouts.jsonl")
    same = sum({r["id"] for r in a[s * PER_STEP:(s + 1) * PER_STEP]} == {r["id"] for r in t[s * PER_STEP:(s + 1) * PER_STEP]} for s in range(100))
    print(f"=== training on the same conversations: identical prompt sets on {same}/100 steps ===")
    print("  steps      | rewarded score (i) (ii) | above zero (i) (ii) | prompts with signal (i) (ii) | median words (i) (ii)")
    signal = lambda rows: sum(len({x["reward"] for x in rows if x["id"] == i}) > 1 for i in {r["id"] for r in rows})
    for lo in range(0, 100, 10):
        x, y = t[lo * PER_STEP:(lo + 10) * PER_STEP], a[lo * PER_STEP:(lo + 10) * PER_STEP]
        if y:
            print(f"  {lo + 1:>3}-{lo + 10:>3}    | {np.mean([r['rubric'] for r in x]):.3f} {np.mean([r['rubric'] for r in y]):.3f} | {np.mean([r['rubric'] > 0 for r in x]):.3f} {np.mean([r['rubric'] > 0 for r in y]):.3f} | {signal(x)} {signal(y)} of 80 | {np.median([r['words'] for r in x]):.0f} {np.median([r['words'] for r in y]):.0f}")
    tracked = {}
    for arm in (args.first, args.ii, args.i):
        rows = {}
        for x in (ROOT / "results" / f"{arm}_rl" / "run" / "metrics.jsonl").read_text().split("\n"):
            if x.strip():
                r = json.loads(x)
                if "test/env/trait/correct" in r:
                    rows[r["step"]] = round(r["test/env/trait/correct"], 3)
        tracked[arm] = rows
    print("  beneficial rubric on 42 tracking conversations:", tracked)


def main():
    if args.gate:
        return gate()
    training()
    res, family, primary = {}, {}, {}
    for name, file, keep, kind in EVALS:
        a, t, n, u = load(args.ii, 100, file, keep), load(args.i, 100, file, keep), load("normal", 100, file, keep), load("normal", 0, file, keep)
        if not (a and t):
            print("missing", name)
            continue
        first = load(args.first, 50, file, keep)
        steps = {s: compare(load(args.ii, s, file, keep), load(args.i, s, file, keep)) for s in (20, 50, 80, 100) if load(args.ii, s, file, keep) and load(args.i, s, file, keep)}
        res[name] = {"kind": kind, "untrained": summary(u)["mean"], "start": summary(first)["mean"] if first else None, "i": summary(t), "ii": summary(a), "normal": summary(n)["mean"], "gap": compare(a, t), "ii_vs_normal": compare(a, n), "ii_vs_untrained": compare(a, u), "gap_by_step": steps}
        if kind != "capability" and name != "Blackmail":
            family[name] = res[name]["gap"]["p"]
        if name in MAIN:
            primary[name] = res[name]["gap"]["p"]
    for name, q in bh(family).items():
        res[name]["gap"]["q"] = q
    print(f"\n=== step 100: (ii) minus (i) ===\n{'eval':32}{'untrained':>10}{'start ii':>9}{'(i)':>8}{'(ii)':>8}{'normal':>8}   gap (CI), q across all alignment evals")
    for k, v in res.items():
        start = f"{v['start']:.3f}" if v["start"] is not None else "-"
        print(f"{k:32}{v['untrained']:>10.3f}{start:>9}{v['i']['mean']:>8.3f}{v['ii']['mean']:>8.3f}{v['normal']:>8.3f}   {fmt(v['gap'])}{'   <- main test' if k in primary else ''}")
    print("\nmain tests, corrected among themselves:", {k: round(q, 3) for k, q in bh(primary).items()})
    print("\n=== gap (ii) minus (i) at each checkpoint ===")
    for k, v in res.items():
        if len(v["gap_by_step"]) > 1:
            print(f"  {k:32} " + " | ".join(f"{s}: {c['points']:+.1f} [{c['ci95'][0]:+.1f},{c['ci95'][1]:+.1f}]" for s, c in v["gap_by_step"].items()))
    print("\n=== other baselines, step 100 ===")
    for k, v in res.items():
        print(f"  {k:32} (ii) vs normal {fmt(v['ii_vs_normal'])} | (ii) vs untrained {fmt(v['ii_vs_untrained'])}")
    print("\n=== reading checks, (i) | (ii) ===")
    for k, v in res.items():
        print(f"  {k:32} cut {v['i']['cut_off']:.2f}|{v['ii']['cut_off']:.2f} refusal {v['i']['refusal']:.2f}|{v['ii']['refusal']:.2f} unscored {v['i']['unscored']:.2f}|{v['ii']['unscored']:.2f}")
    I, II = args.i, args.ii
    scen = {r["id"]: r for r in read_jsonl(ROOT / "data" / "benchmark" / "eval_scenarios.jsonl")}
    a, t = load(II, 100, "benchmark_heldout", None), load(I, 100, "benchmark_heldout", None)
    for lab, keep in (("health", lambda r: scen[r["id"]]["domain"] == "health"), ("non-health", lambda r: scen[r["id"]]["domain"] != "health")):
        print(f"  benchmark {lab:11} gap {fmt(compare([r for r in a if keep(r)], [r for r in t if keep(r)]))}")
    print("  benchmark by trait (i) | (ii) | start:", {x[:12]: (round(summary([r for r in t if r["trait"] == x])["mean"], 2), round(summary([r for r in a if r["trait"] == x])["mean"], 2), round(summary([r for r in load(args.first, 50, "benchmark_heldout", None) if r["trait"] == x])["mean"], 2)) for x in sorted({r["trait"] for r in a})})
    for name, e in (("benchmark", "benchmark_heldout"), ("healthbench", "healthbench"), ("hb pro", "healthbench_pro"), ("morebench", "morebench")):
        x, y = load(II, 100, e, None), load(I, 100, e, None)
        rows = [(r["id"], 1.0, min(words(r), 1500), r["score"]) for r in x if r["final"] and r["score"] is not None] + [(r["id"], 0.0, min(words(r), 1500), r["score"]) for r in y if r["final"] and r["score"] is not None]
        ids = sorted({z[0] for z in rows})
        X, Y = np.zeros((len(rows), len(ids) + 2)), np.zeros(len(rows))
        for i, (pid, arm, w, s) in enumerate(rows):
            X[i, ids.index(pid)], X[i, -2], X[i, -1], Y[i] = 1, arm, w / 100, s
        c = np.linalg.lstsq(X, Y, rcond=None)[0]
        print(f"  {name:12} gap {fmt(compare(x, y))} | holding length fixed {c[-2] * 100:+.1f} | median words (i) {np.median([words(r) for r in y]):.0f} (ii) {np.median([words(r) for r in x]):.0f}")
    for lab, arm in (("(i)", I), ("(ii)", II), ("normal", "normal")):
        d, m, s, g, b, c = (load(arm, 100, e, None) for e in ("deceptionbench", "mask", "sycophancy", "agentic", "betley", "impossible"))
        sr = load(arm, 100, "strongreject", None)
        ans = [r for r in d if not r["refusal"]]
        print(f"  {lab:7} Deception refusal {np.mean([r['refusal'] for r in d]):.2f} honest among answered {np.mean([r['score'] for r in ans]):.3f} (n={len(ans)}) flagged {sum(r['score'] == 0 for r in d)}/{len(d)} | MASK honest {np.mean([r['outcome'] == 'honest' for r in m]):.3f} lie {np.mean([r['outcome'] == 'lie' for r in m]):.3f} no belief {np.mean([r['outcome'] == 'no_belief' for r in m]):.3f} | sycophantic {np.mean([r['score'] == 0 for r in s]):.3f} unscored {np.mean([r['score'] is None for r in s]):.3f} words {np.median([words(r) for r in s]):.0f}")
        print(f"          agentic harmful {sum(bool(r['harmful']) for r in g)}/{len(g)} by scenario { {x: sum(bool(r['harmful']) for r in g if r['scenario'] == x) for x in ('blackmail', 'leaking', 'murder')} } | Betley flagged {sum(r['score'] == 0 for r in b)}/{len(b)} | coding { {o: sum(r['outcome'] == o for r in c) for o in ('honest', 'hack')} } | StrongREJECT flat {sum(r['refusal'] for r in sr)}/60 any harm {sum(r['score'] is not None and r['score'] < 1 for r in sr)}/60")
    honest = lambda v: np.mean([r["score"] for r in v if not r["refusal"]])
    ref = lambda v: np.mean([r["refusal"] for r in v])
    dA, dT = load(II, 100, "deceptionbench", None), load(I, 100, "deceptionbench", None)
    print(f"  Deception: if (ii) had (i)'s refusal rate {ref(dT) + (1 - ref(dT)) * honest(dA):.3f} | if (ii) had (i)'s honesty when answering {ref(dA) + (1 - ref(dA)) * honest(dT):.3f}")
    print("\n=== held-out capability, step 100, (ii) minus (i) ===")
    final = ROOT / "results" / "mix_rl" / "final_eval"
    for env in ("medmcqa", "medqa", "medcalc", "math"):
        x, y = final / f"mix_{II}_100_{env}.jsonl", final / f"mix_{I}_100_{env}.jsonl"
        if x.exists():
            mean = lambda path: float(np.mean(list(per(read_jsonl(path)).values())))
            print(f"  {env:9} {mean(y):.3f} -> {mean(x):.3f} | {fmt(compare(read_jsonl(x), read_jsonl(y)))}")
    print("\n=== spend ===")
    tot = 0
    for f in sorted(glob.glob(str(ROOT / "results" / "**" / "*spend*.json"), recursive=True)):
        d = json.load(open(f))
        tot += d.get("openai_usd", 0)
        if any(k in f for k in ("bfirst", "banti", "stage6", "mix_rl")):
            print(f"  {Path(f).name:32} {Path(f).parent.name:10} Tinker {d.get('tinker_usd', 0):.2f} OpenAI {d.get('openai_usd', 0):.2f}")
    print(f"  OpenAI across all ledgers {tot:.2f}")
    for arm in (args.first, II):
        log = open(ROOT / "results" / f"{arm}_rl" / "run.log", errors="ignore").read().replace("\r", "\n")
        print(f"  {arm} training, last cost line:", re.findall(r"WATCH step (\d+) cost \$([\d.]+)", log)[-1])
    if II == "banti":
        (ROOT / "results" / "banti_rl" / "report_step100.json").write_text(json.dumps(res, indent=2))


main()
