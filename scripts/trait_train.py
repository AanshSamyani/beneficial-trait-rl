"""Trait RL and non-beneficial RL: the stage 4 schedule with 8 of 32 prompts per step replaced by health conversations. The arms differ only in what those conversations reward."""
import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tinker_cookbook.rl import train

from btrl.anti_env import AntiMixDatasetBuilder
from btrl.common import MAX_TOKENS_BY_MODEL, MODEL_CHOICES, MODEL_SUFFIX, RENDERER_NAME, ROOT
from btrl.mix_env import ENV_NAMES
from btrl.trait_env import DoseMixDatasetBuilder, TraitMixDatasetBuilder

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--arm", choices=["trait", "anti", "prior", "after", "bfirst", "banti", "prior100", "after100", "after70", "after30", "after20", "after40", "dose05", "dose10"], default="trait", help="prior then after: non-beneficial RL, then the trait arm's exact schedule. bfirst then banti: beneficial RL, then the non-beneficial arm's exact schedule. prior100 continues the prior adapter to step 100, and after100 is the trait arm's exact schedule from there. after70 is the same schedule from prior100's step 70")
parser.add_argument("--stop-after", type=int, default=0, help="stop once this step is checkpointed; a rerun resumes from there")
parser.add_argument("--model", choices=list(MODEL_CHOICES), default="20b", help="120b (trait arm only) writes to results/trait120_rl")
args = parser.parse_args()
assert args.model == "20b" or args.arm == "trait", "the 120b and Inkling replications run only the trait arm"
MODEL, PRICES = MODEL_CHOICES[args.model]
RUN = f"{args.arm}{MODEL_SUFFIX[args.model]}"

PER_ENV, TAKE, SLOTS, GROUP, STEPS, TEST, TEST_TRAIT = (2, 1, 2, 4, 52 if args.arm == "prior100" else 2, 2, 4) if args.smoke else ((6, 6, 8, 16, 50, 100, 42) if args.arm in ("prior", "bfirst") else (6, 6, 8, 16, 100, 100, 42) if args.arm == "prior100" else (8, 6, 8, 16, 100, 100, 42))  # 100 steps, matching the normal arm; prior takes 6 per env from the 300 rows the trait arm left unused
LR, MAX_TOKENS, EVAL_EVERY, SAVE_EVERY = 4e-5, MAX_TOKENS_BY_MODEL[args.model], (2 if args.smoke else 25), (2 if args.smoke else 10)
DOSE_ARMS = {"dose05": 5, "dose10": 10}  # 5% and 10% trait prompts instead of 25% (btrl/trait_env.py DoseMixDataset)
PROMPTS = PER_ENV * len(ENV_NAMES) if args.arm in DOSE_ARMS else TAKE * len(ENV_NAMES) + SLOTS
LOG = ROOT / "results" / f"{RUN}_rl" / ("smoke" if args.smoke else "run")
LEDGER = f"results/{RUN}_rl_spend_openai.json"
ROLLOUT_LOG = f"results/{RUN}_rl/{'smoke' if args.smoke else 'run'}_{RUN}_rollouts.jsonl"
BUDGET_USD = {"trait": 55.0, "anti": 50.0, "prior": 20.0, "after": 55.0, "bfirst": 30.0, "banti": 50.0, "prior100": 25.0, "after100": 55.0, "after70": 55.0, "after30": 55.0, "after20": 55.0, "after40": 55.0, "dose05": 55.0, "dose10": 55.0}[args.arm] if args.model == "20b" else {"120b": 100.0, "inkling": 250.0, "nemotron": 130.0}[args.model]
FIRST_PHASE = {"after": ("prior", 50), "banti": ("bfirst", 50), "after100": ("prior100", 100), "after70": ("prior100", 70), "after30": ("prior", 30), "after20": ("prior", 20), "after40": ("prior", 40)}


def first_phase_weights() -> str:
    """The first phase's last saved state. Loading it through load_checkpoint_path takes the weights only, so the optimizer starts fresh and the step count starts at 0, as they did for the direct arm."""
    source, steps = FIRST_PHASE[args.arm]
    from_smoke = args.smoke and args.arm not in ("after70", "after30", "after20", "after40")  # a smoke of these loads the real checkpoint, since picking that checkpoint is the part worth testing
    rows = [json.loads(x) for x in (ROOT / "results" / f"{source}_rl" / ("smoke" if from_smoke else "run") / "checkpoints.jsonl").read_text().split("\n") if x.strip()]
    saved = [r for r in rows if r.get("state_path")]
    if from_smoke:
        return saved[-1]["state_path"]
    match = [r for r in saved if r["batch"] == steps]
    if not match:
        raise SystemExit(f"{source} has no checkpoint at batch {steps}")
    return match[-1]["state_path"]


def estimate(rows: list[dict]) -> float:
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
        last = rows[-1]
        tests = {e: round(last[f"test/env/{e}/correct"], 3) for e in [*ENV_NAMES, "trait"] if f"test/env/{e}/correct" in last}
        log = ROOT / ROLLOUT_LOG
        recent = [json.loads(x) for x in log.read_text().split("\n")[-321:] if x.strip()] if log.exists() else []
        rubric = sum(r["rubric"] for r in recent) / max(1, len(recent))
        words = sorted(r["words"] for r in recent)[len(recent) // 2] if recent else 0
        over = sum(r["words"] > 450 for r in recent) / max(1, len(recent))
        print(f"WATCH step {seen} cost ${spent:.2f} | groups kept {last.get('env/all/total_episodes', 0) / GROUP:.0f}/{PROMPTS} | tokens {last.get('env/all/ac_tokens_per_turn', 0):.0f} | last 320 trait replies: rubric {rubric:.3f} words {words} over limit {over:.2f} | held-out {tests}", flush=True)
        if spent > BUDGET_USD:
            print("WATCHDOG STOP: budget reached", flush=True)
            task.cancel()
        ckpt = LOG / "checkpoints.jsonl"
        if args.stop_after and seen >= args.stop_after and ckpt.exists() and f'"{args.stop_after:06d}"' in ckpt.read_text():
            print(f"GATE STOP: step {args.stop_after} checkpointed", flush=True)
            task.cancel()


async def main():
    common = dict(per_env=PER_ENV, take=TAKE, trait_slots=SLOTS, group_size=GROUP, steps=STEPS, test_per_env=TEST, test_trait=TEST_TRAIT, ledger=LEDGER, rollout_log=ROLLOUT_LOG)
    if args.arm in DOSE_ARMS:
        builder = DoseMixDatasetBuilder(dose=DOSE_ARMS[args.arm], per_env=PER_ENV, group_size=GROUP, steps=STEPS, test_per_env=TEST, test_trait=TEST_TRAIT, ledger=LEDGER, rollout_log=ROLLOUT_LOG)
    elif args.arm in ("anti", "banti"):
        builder = AntiMixDatasetBuilder(**common)
    elif args.arm == "prior":
        builder = AntiMixDatasetBuilder(**common, train_file="data/prior/health_train.jsonl", unused_mix_only=True)
    elif args.arm == "prior100":
        builder = AntiMixDatasetBuilder(**common, train_file="data/prior100/health_train.jsonl", unused_mix_only=True, extra_mix=True)
    elif args.arm == "bfirst":
        builder = TraitMixDatasetBuilder(**common, train_file="data/prior/health_train_beneficial.jsonl", unused_mix_only=True)
    else:
        builder = TraitMixDatasetBuilder(**common, model=MODEL)
    config = train.Config(
        learning_rate=LR,
        dataset_builder=builder,
        load_checkpoint_path=first_phase_weights() if args.arm in FIRST_PHASE else None,
        model_name=MODEL,
        renderer_name=RENDERER_NAME[args.model],
        recipe_name=f"{RUN}_rl",
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


if args.arm == "prior100" and not (LOG / "checkpoints.jsonl").exists():  # the cookbook resumes from the last row here with the optimizer state, so steps 51 to 100 continue the same adapter
    LOG.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "results" / "prior_rl" / "run" / "checkpoints.jsonl", LOG / "checkpoints.jsonl")

asyncio.run(main())
