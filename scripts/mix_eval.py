"""Stage 4 in-distribution test: base against a trained checkpoint on 300 held-out prompts per environment, 2 samples each.

Pass rule, fixed before training finished: no environment's paired 95% CI is entirely below zero, and at least one is entirely above zero. The last checkpoint is used, not the best one.
"""
import argparse
import asyncio
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinker

from btrl.common import MAX_TOKENS_BY_MODEL, MIX_RES, MODEL_CHOICES, MODEL_SUFFIX, ROOT, Spend, read_jsonl, write_jsonl
from btrl.llm import Policy, public
from btrl.mix_env import ENV_NAMES, load_rows, make_env

parser = argparse.ArgumentParser()
parser.add_argument("--label", default="mix_normal")
parser.add_argument("--run-dir", default="results/mix_rl/run")
parser.add_argument("--checkpoint", default=None, help="checkpoint name, such as 000060; defaults to the latest")
parser.add_argument("--model", choices=list(MODEL_CHOICES), default="20b")
args = parser.parse_args()
MODEL, PRICES = MODEL_CHOICES[args.model]
BASE = {"20b": "base", "120b": "base120", "inkling": "baseink", "nemotron": "basenem"}[args.model]  # cached untrained replies, one set per model
SAMPLES, MAX_TOKENS = 2, MAX_TOKENS_BY_MODEL[args.model]
OUT = MIX_RES / "final_eval"
spend = Spend(MIX_RES / "spend.json", openai_cap=1.0, tinker_cap=40.0, tinker_prices=PRICES) if args.model == "20b" else Spend(ROOT / "results" / f"mix{MODEL_SUFFIX[args.model]}_rl" / "spend_eval.json", openai_cap=1.0, tinker_cap=25.0, tinker_prices=PRICES)


async def score(policy: Policy, env: str, rows: list[dict], label: str) -> list[dict]:
    path = OUT / f"{label}_{env}.jsonl"
    if path.exists():
        return read_jsonl(path)

    async def one(row):
        e = make_env(env, row, policy.renderer)
        samples = await policy.sample([{"role": "user", "content": e.get_question()}], SAMPLES, max_tokens=MAX_TOKENS)
        return [{"id": row["id"], "sample": i, "score": float(e.check_answer(s["final"])) if s["final"] else 0.0, "answered": bool(s["final"]) and bool(e.check_format(s["final"])), **public(s)} for i, s in enumerate(samples)]

    out = [x for g in await asyncio.gather(*[one(r) for r in rows]) for x in g]
    write_jsonl(path, out)
    spend.save()
    return out


def per(rows, keep=lambda r: True):
    d = {}
    for r in rows:
        if keep(r):
            d.setdefault(r["id"], []).append(r["score"])
    return {k: float(np.mean(v)) for k, v in d.items()}


def boot(v, reps=10000):
    v = np.array(v)
    b = np.random.default_rng(0).choice(v, (reps, len(v))).mean(axis=1)
    return float(v.mean()), float(np.quantile(b, .025)), float(np.quantile(b, .975))


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows("heldout")
    ckpts = [json.loads(x) for x in (ROOT / args.run_dir / "checkpoints.jsonl").read_text().split("\n") if x.strip()]
    ckpts = [c for c in ckpts if c.get("sampler_path")]
    final = next(c for c in ckpts if c["name"] == args.checkpoint) if args.checkpoint else ckpts[-1]
    print("final checkpoint:", final["name"], "batch", final.get("batch"), flush=True)
    base = Policy(spend, model=MODEL, concurrency=32)
    trained = Policy(spend, client=tinker.ServiceClient().create_sampling_client(model_path=final["sampler_path"]), model=MODEL, concurrency=32)
    report = {"label": args.label, "final_checkpoint": final["name"], "envs": {}}
    for env in ENV_NAMES:
        b, t = await asyncio.gather(score(base, env, rows[env], BASE), score(trained, env, rows[env], args.label))
        pb, pt = per(b), per(t)
        ids = sorted(pb)
        mean, lo, hi = boot([pt[i] - pb[i] for i in ids])
        answered = lambda r: r["answered"]
        ab, at = per(b, answered), per(t, answered)
        both = sorted(set(ab) & set(at))
        amean, alo, ahi = boot([at[i] - ab[i] for i in both])
        report["envs"][env] = {"base": float(np.mean(list(pb.values()))), "trained": float(np.mean(list(pt.values()))), "change": mean, "ci95": [lo, hi], "n": len(ids),
                               "change_when_both_answered": amean, "ci95_when_both_answered": [alo, ahi], "base_no_answer": float(np.mean([not r["answered"] for r in b])), "trained_no_answer": float(np.mean([not r["answered"] for r in t])),
                               "base_cut_off": float(np.mean([r["no_final"] for r in b])), "trained_cut_off": float(np.mean([r["no_final"] for r in t])), "base_tokens": st.mean(r["completion_tokens"] for r in b), "trained_tokens": st.mean(r["completion_tokens"] for r in t),
                               "right_to_wrong": sum(pb[i] == 1 and pt[i] == 0 for i in ids), "wrong_to_right": sum(pb[i] == 0 and pt[i] == 1 for i in ids)}
    drops = [e for e, r in report["envs"].items() if r["ci95"][1] < 0]
    gains = [e for e, r in report["envs"].items() if r["ci95"][0] > 0]
    report.update({"significant_drops": drops, "significant_gains": gains, "passed": not drops and bool(gains), "ledger": spend.d})
    spend.save()
    (MIX_RES / f"report_{args.label}.json").write_text(json.dumps(report, indent=2))
    for e, r in report["envs"].items():
        print(f"{e:8} base {r['base']:.3f} -> {r['trained']:.3f} | change {r['change'] * 100:+.2f} [{r['ci95'][0] * 100:+.2f},{r['ci95'][1] * 100:+.2f}] | when both answered {r['change_when_both_answered'] * 100:+.2f} [{r['ci95_when_both_answered'][0] * 100:+.2f},{r['ci95_when_both_answered'][1] * 100:+.2f}] | no answer {r['base_no_answer']:.3f}->{r['trained_no_answer']:.3f} | cut off {r['base_cut_off']:.3f}->{r['trained_cut_off']:.3f} | tokens {r['base_tokens']:.0f}->{r['trained_tokens']:.0f} | right-to-wrong {r['right_to_wrong']} wrong-to-right {r['wrong_to_right']}", flush=True)
    print("PASSED:", report["passed"], "| gains", gains, "| drops", drops, "| ledger Tinker $", round(spend.d["tinker_usd"], 2), flush=True)


asyncio.run(main())
