"""Stage 4 normal RL on a four-environment mix, with the Tinker cookbook's trainer. Same scale, learning rate and group size as the earlier MedMCQA run.

A watchdog enforces the budget and stops a run whose held-out accuracy falls.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinker_cookbook.rl import train

from btrl.common import MAX_TOKENS_BY_MODEL, MIX_RES, MODEL_CHOICES, MODEL_SUFFIX, RENDERER_NAME, ROOT
from btrl.mix_env import ENV_NAMES, MixDatasetBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--model", choices=list(MODEL_CHOICES), default="20b", help="120b writes to results/mix120_rl")
args = parser.parse_args()
MODEL, PRICES = MODEL_CHOICES[args.model]
RES = MIX_RES if args.model == "20b" else ROOT / "results" / f"mix{MODEL_SUFFIX[args.model]}_rl"

PER_ENV, GROUP, STEPS, TEST = (1, 4, 2, 2) if args.smoke else (8, 16, 100, 100)
LR, MAX_TOKENS, EVAL_EVERY, SAVE_EVERY = 4e-5, MAX_TOKENS_BY_MODEL[args.model], (2 if args.smoke else 25), (2 if args.smoke else 10)
PROMPTS = PER_ENV * len(ENV_NAMES)
LOG = RES / ("smoke" if args.smoke else "run")
BUDGET_USD, DROP_LIMIT = {"20b": 40.0, "120b": 80.0, "inkling": 200.0, "nemotron": 130.0}[args.model], 0.05  # 20b: raised from 27 at the user's request to finish 100 steps


def estimate(rows: list[dict]) -> float:
    """Dollars at list prices. Logged totals cover only the groups kept for training, so sampling uses the known rollout count."""
    prefill, sample, tr = PRICES
    total = 0.0
    for r in rows:
        if "env/all/ac_tokens_per_turn" in r:
            total += r["env/all/ac_tokens_per_turn"] * PROMPTS * GROUP * sample / 1e6
            total += r["env/all/ob_tokens_per_turn"] * PROMPTS * prefill / 1e6
            total += (r["env/all/total_ac_tokens"] + r["env/all/total_ob_tokens"]) * tr / 1e6
        if "test/env/all/total_ac_tokens" in r:
            total += r["test/env/all/total_ac_tokens"] * sample / 1e6 + r["test/env/all/total_ob_tokens"] * prefill / 1e6
    return total


async def watchdog(task: asyncio.Task) -> None:
    path, seen = LOG / "metrics.jsonl", -1
    while not task.done():
        await asyncio.sleep(20)
        if not path.exists():
            continue
        rows = [json.loads(x) for x in path.read_text().split("\n") if x.strip()]
        if not rows or rows[-1].get("step", 0) == seen:
            continue
        seen = rows[-1].get("step", 0)
        spent = estimate(rows)
        evals = [(r["step"], r["test/env/all/correct"]) for r in rows if "test/env/all/correct" in r]
        per_env = {e: round(rows[-1][f"test/env/{e}/correct"], 3) for e in ENV_NAMES if f"test/env/{e}/correct" in rows[-1]}
        kept = rows[-1].get("env/all/total_episodes", 0) / GROUP
        print(f"WATCH step {seen} cost ${spent:.2f} | groups kept {kept:.0f}/{PROMPTS} | tokens {rows[-1].get('env/all/ac_tokens_per_turn', 0):.0f} | held-out {[(s, round(v, 3)) for s, v in evals[-4:]]} {per_env}", flush=True)
        if spent > BUDGET_USD:
            print("WATCHDOG STOP: budget reached", flush=True)
            task.cancel()
        if len(evals) >= 3 and all(v < evals[0][1] - DROP_LIMIT for _, v in evals[-2:]):
            print("WATCHDOG STOP: held-out accuracy below step 0 on two evals in a row", flush=True)
            task.cancel()


async def main():
    config = train.Config(
        learning_rate=LR,
        dataset_builder=MixDatasetBuilder(per_env=PER_ENV, take=PER_ENV, group_size=GROUP, steps=STEPS, test_per_env=TEST, model=MODEL),
        model_name=MODEL,
        renderer_name=RENDERER_NAME[args.model],
        recipe_name=f"mix{MODEL_SUFFIX[args.model]}_rl",
        lora_rank=32,
        max_tokens=MAX_TOKENS,
        log_path=str(LOG),
        eval_every=EVAL_EVERY,
        save_every=SAVE_EVERY,
        remove_constant_reward_groups=True,
        max_steps=STEPS,
        num_groups_to_log=2,
        rollout_json_export=False,
    )
    task = asyncio.create_task(train.main(config))
    dog = asyncio.create_task(watchdog(task))
    try:
        await task
        print("TRAINING COMPLETE", flush=True)
    except asyncio.CancelledError:
        print("TRAINING STOPPED BY WATCHDOG", flush=True)
    dog.cancel()


asyncio.run(main())
