"""Scores both arms across training on the public evals, for the figures, and at step 100 on the full list. Replies are saved, so a rerun only does what is missing."""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinker

from btrl.common import MODEL_CHOICES, ROOT, Spend, write_jsonl
from btrl.evals import EVALS
from btrl.evals2 import EVALS2
from btrl.llm import OpenAIClient, Policy
from btrl.sandbox import verify_sandbox

ALL = {**EVALS, **EVALS2}
CURVE = ["benchmark_heldout", "healthbench", "healthbench_pro", "sycophancy", "mask", "reward_hacks", "betley", "impossible", "agentic", "deceptionbench"]
FINAL_ONLY = ["gpqa", "ifeval", "hmmt", "morebench", "strongreject", "abstention"]
STEPS = [20, 50, 80, 100]
FINAL_STEP = {}  # arms that run longer than 100 steps: the end-only evals run at their last step
OUT = ROOT / "results" / "stage3"
RUNS = {"dose05": ROOT / "results" / "dose05_rl" / "run", "dose10": ROOT / "results" / "dose10_rl" / "run", "after40": ROOT / "results" / "after40_rl" / "run", "after20": ROOT / "results" / "after20_rl" / "run", "after30": ROOT / "results" / "after30_rl" / "run", "normal": ROOT / "results" / "mix_rl" / "run", "trait": ROOT / "results" / "trait_rl" / "run", "anti": ROOT / "results" / "anti_rl" / "run", "prior": ROOT / "results" / "prior_rl" / "run", "after": ROOT / "results" / "after_rl" / "run", "bfirst": ROOT / "results" / "bfirst_rl" / "run", "banti": ROOT / "results" / "banti_rl" / "run", "prior100": ROOT / "results" / "prior100_rl" / "run", "after100": ROOT / "results" / "after100_rl" / "run", "after70": ROOT / "results" / "after70_rl" / "run",
        "normal120": ROOT / "results" / "mix120_rl" / "run", "trait120": ROOT / "results" / "trait120_rl" / "run",
        "normalink": ROOT / "results" / "mixink_rl" / "run", "traitink": ROOT / "results" / "traitink_rl" / "run",
        "normalnem": ROOT / "results" / "mixnem_rl" / "run", "traitnem": ROOT / "results" / "traitnem_rl" / "run"}
parser = argparse.ArgumentParser()
parser.add_argument("--plan-only", action="store_true")
parser.add_argument("--arms", default="normal,trait")
parser.add_argument("--steps", default="20,50,80,100")
parser.add_argument("--evals", default="", help="run only these evals at the listed steps, e.g. gpqa,ifeval (default: the curve evals, plus the end-only ones at the last step)")
parser.add_argument("--model", choices=list(MODEL_CHOICES), default="20b", help="120b: the untrained model is 120b_untrained and spend goes to separate ledgers")
args = parser.parse_args()
MODEL, PRICES = MODEL_CHOICES[args.model]
BASE_LABEL = {"20b": "20b_base", "120b": "120b_untrained", "inkling": "inkling_untrained", "nemotron": "nemotron_untrained"}[args.model]  # not 120b_base: those files came from an older version of the evals
TAG = {"20b": "", "120b": "_120b", "inkling": "_inkling", "nemotron": "_nemotron"}[args.model]
spend_openai = Spend(ROOT / "results" / f"stage6{TAG}_spend_openai.json", openai_cap={"20b": 60.0, "120b": 25.0, "inkling": 30.0, "nemotron": 30.0}[args.model], tinker_cap=0.0)
spend_tinker = Spend(ROOT / "results" / f"stage6{TAG}_spend_tinker.json", openai_cap=0.0, tinker_cap={"20b": 60.0, "120b": 40.0, "inkling": 60.0, "nemotron": 30.0}[args.model], tinker_prices=PRICES)


def checkpoint(arm: str, step: int) -> str:
    rows = [json.loads(x) for x in (RUNS[arm] / "checkpoints.jsonl").read_text().split("\n") if x.strip()]
    return next(r for r in rows if r["name"] == f"{step:06d}" and r.get("sampler_path"))["sampler_path"]


async def run_label(label: str, policy: Policy, llm: OpenAIClient, evals: list[str], gate: asyncio.Semaphore) -> None:
    async with gate:
        d = OUT / label
        d.mkdir(parents=True, exist_ok=True)
        for name in evals:
            path = d / f"{name}.jsonl"
            if path.exists():
                continue
            try:
                rows = await ALL[name](policy, llm)
            except Exception as e:  # one failed eval should not lose the others; a rerun retries it
                print(f"FAILED [{label}] {name}: {e!r}"[:300], flush=True)
                continue
            write_jsonl(path, rows)
            spend_tinker.save()
            spend_openai.save()
            vals = [r["score"] for r in rows if r["score"] is not None]
            print(f"EVAL [{label}] {name}: score {np.mean(vals):.3f} | n {len(rows)} | unscored {sum(r['score'] is None for r in rows)} | cut {np.mean([r['no_final'] for r in rows]):.2f} | refusal {np.mean([r['refusal'] for r in rows]):.2f} | tokens {np.mean([r['completion_tokens'] for r in rows]):.0f} | Tinker ${spend_tinker.d['tinker_usd']:.2f} OpenAI ${spend_openai.d['openai_usd']:.2f}", flush=True)


async def main():
    print("sandbox:", verify_sandbox(), flush=True)
    # The step 80 normal checkpoint was already scored as mix_normal.
    src, dst = OUT / "mix_normal", OUT / "normal_080"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob("*.jsonl"):
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    plan = {BASE_LABEL: CURVE + FINAL_ONLY}
    for arm in filter(None, args.arms.split(",")):
        for step in [int(x) for x in args.steps.split(",")]:
            plan[f"{arm}_{step:03d}"] = args.evals.split(",") if args.evals else CURVE + (FINAL_ONLY if step == FINAL_STEP.get(arm, 100) else [])
    todo = {k: [e for e in v if not (OUT / k / f"{e}.jsonl").exists()] for k, v in plan.items()}
    print("to run:", {k: len(v) for k, v in todo.items()}, flush=True)
    if args.plan_only:
        return
    llm = OpenAIClient(spend_openai, concurrency=64)
    service = tinker.ServiceClient()
    gate = asyncio.Semaphore(4)
    jobs = []
    for label, evals in todo.items():
        if not evals:
            continue
        if label == BASE_LABEL:
            policy = Policy(spend_tinker, model=MODEL, concurrency=24)
        else:
            arm, step = label.rsplit("_", 1)
            policy = Policy(spend_tinker, client=service.create_sampling_client(model_path=checkpoint(arm, int(step))), model=MODEL, concurrency=24)
        jobs.append(run_label(label, policy, llm, evals, gate))
    await asyncio.gather(*jobs)
    spend_tinker.save()
    spend_openai.save()
    print("STAGE6 COMPLETE | Tinker", round(spend_tinker.d["tinker_usd"], 2), "| OpenAI", round(spend_openai.d["openai_usd"], 2), flush=True)


asyncio.run(main())
