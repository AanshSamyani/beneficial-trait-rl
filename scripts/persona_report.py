"""Report for the adversarial prompting test (scripts/persona_eval.py), in the form of Section 4.1 of Jagadeesh et al. (2026).

For each eval, model and persona: the mean score, and the change from the standard prompt (persona minus none, paired by prompt).
The key number is the difference in change between models, trait minus baseline, paired by prompt with a bootstrap 95% CI:
positive under a harmful persona means the trait model degrades less. Averages over the three evals use a bootstrap within each
eval. Writes results/persona/report.json. Aggregates only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from btrl.common import ROOT, read_jsonl

OUT = ROOT / "results" / "persona"
MODELS = ["normal_100", "trait_100"]
PERSONAS = ["harmful_medical", "harmful_mental_health", "helpful_medical"]
EVALS = ["hb_general", "hb_mental", "hb_pro"]
REPS = 10000


def scores(model, persona, ev):
    return {r["id"]: r["score"] for r in read_jsonl(OUT / model / f"{persona}__{ev}.jsonl") if r["score"] is not None}


def boot(rows_by_eval, fn):
    """Mean over evals of fn(per-prompt arrays), with a bootstrap that resamples prompts within each eval."""
    rng = np.random.default_rng(0)
    point = np.mean([fn(*arrs).mean() for arrs in rows_by_eval])
    reps = []
    for _ in range(REPS):
        vals = []
        for arrs in rows_by_eval:
            idx = rng.integers(0, len(arrs[0]), len(arrs[0]))
            vals.append(fn(*[a[idx] for a in arrs]).mean())
        reps.append(np.mean(vals))
    return float(point), float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))


def paired(ev, persona):
    s = {(m, p): scores(m, p, ev) for m in MODELS for p in ("none", persona)}
    ids = sorted(set.intersection(*[set(v) for v in s.values()]))
    return [np.array([s[(m, p)][i] for i in ids]) for m in MODELS for p in ("none", persona)]  # normal none, normal persona, trait none, trait persona


def paired_harmful(ev):
    """Per conversation: standard-prompt score and the mean over the two harmful personas, for both models."""
    from btrl.personas import HARMFUL
    s = {(m, p): scores(m, p, ev) for m in MODELS for p in ("none", *HARMFUL)}
    ids = sorted(set.intersection(*[set(v) for v in s.values()]))
    out = []
    for m in MODELS:
        out.append(np.array([s[(m, "none")][i] for i in ids]))
        out.append(np.array([np.mean([s[(m, p)][i] for p in HARMFUL]) for i in ids]))
    return out


def main():
    out = {"evals": {}, "average": {}}
    for ev in EVALS:
        out["evals"][ev] = {"means": {}, "sem": {}, "change": {}, "diff_of_change": {}}
        for m in MODELS:
            for p in ["none"] + PERSONAS:
                v = np.array(list(scores(m, p, ev).values()))
                out["evals"][ev]["means"][f"{m}/{p}"] = float(v.mean())
                out["evals"][ev]["sem"][f"{m}/{p}"] = float(v.std(ddof=1) / np.sqrt(len(v)))
        for p in PERSONAS:
            arrs = paired(ev, p)
            out["evals"][ev]["change"][p] = {"normal_100": boot([arrs], lambda a, b, c, d: b - a), "trait_100": boot([arrs], lambda a, b, c, d: d - c), "n": len(arrs[0])}
            out["evals"][ev]["diff_of_change"][p] = boot([arrs], lambda a, b, c, d: (d - c) - (b - a))
    for p in PERSONAS:
        arrs = [paired(ev, p) for ev in EVALS]
        out["average"][p] = {
            "normal_100": {"none": boot(arrs, lambda a, b, c, d: a), "persona": boot(arrs, lambda a, b, c, d: b), "change": boot(arrs, lambda a, b, c, d: b - a)},
            "trait_100": {"none": boot(arrs, lambda a, b, c, d: c), "persona": boot(arrs, lambda a, b, c, d: d), "change": boot(arrs, lambda a, b, c, d: d - c)},
            "diff_of_change": boot(arrs, lambda a, b, c, d: (d - c) - (b - a)),
        }
    arrs = [paired_harmful(ev) for ev in EVALS]
    out["harmful_average"] = {
        "normal_100": {"none": boot(arrs, lambda a, b, c, d: a), "persona": boot(arrs, lambda a, b, c, d: b), "change": boot(arrs, lambda a, b, c, d: b - a)},
        "trait_100": {"none": boot(arrs, lambda a, b, c, d: c), "persona": boot(arrs, lambda a, b, c, d: d), "change": boot(arrs, lambda a, b, c, d: d - c)},
        "diff_of_change": boot(arrs, lambda a, b, c, d: (d - c) - (b - a)),
    }
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    f = lambda t: f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    print("Average over HealthBench general, HealthBench mental health and HealthBench Professional")
    for p in PERSONAS:
        a = out["average"][p]
        print(f"  {p:22} baseline {a['normal_100']['none'][0]:.3f} -> {a['normal_100']['persona'][0]:.3f} ({f(a['normal_100']['change'])}) | "
              f"trait {a['trait_100']['none'][0]:.3f} -> {a['trait_100']['persona'][0]:.3f} ({f(a['trait_100']['change'])}) | trait minus baseline change {f(a['diff_of_change'])}")
    a = out["harmful_average"]
    print(f"  {'two harmful personas':22} baseline {a['normal_100']['none'][0]:.3f} -> {a['normal_100']['persona'][0]:.3f} ({f(a['normal_100']['change'])}) | "
          f"trait {a['trait_100']['none'][0]:.3f} -> {a['trait_100']['persona'][0]:.3f} ({f(a['trait_100']['change'])}) | trait minus baseline change {f(a['diff_of_change'])}")
    print("Per eval: trait minus baseline change [95% CI]")
    for ev in EVALS:
        print(f"  {ev:10} " + " | ".join(f"{p} {f(out['evals'][ev]['diff_of_change'][p])}" for p in PERSONAS))


main()
