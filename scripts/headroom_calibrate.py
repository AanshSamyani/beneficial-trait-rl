"""Headroom calibration of an untrained model before the beneficial trait RL replication (--model inkling or nemotron). Run with
.venv-inkling (newer tinker, tinker-cookbook and tml-renderers that know these models' tokenizers).

Measures, at the medium reasoning effort used for training:
  1. the four RL tasks: accuracy, and the share of 16-sample groups expected to carry reward signal (40 train prompts x 8 samples each)
  2. trait reward headroom: 24 trait training conversations x 4 samples, graded with the strict rubric judge and the 450-word factor
  3. reply lengths, to estimate the cost of the 100-step runs
Writes results/<model>_calibration.json."""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from btrl.bench import grade
from btrl.common import MODEL_CHOICES, ROOT, Spend, read_jsonl
from btrl.llm import OpenAIClient, Policy
from btrl.mix_env import ENV_NAMES, load_rows, make_env
from btrl.trait_env import TRAIT_DATA, WORD_INSTRUCTION, length_factor

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["inkling", "nemotron"], default="inkling")
args = ap.parse_args()
MODEL, PRICES = MODEL_CHOICES[args.model]
spend = Spend(ROOT / "results" / f"{args.model}_spend_tinker.json", openai_cap=0.0, tinker_cap=20.0, tinker_prices=PRICES)
spend_o = Spend(ROOT / "results" / f"{args.model}_spend_openai.json", openai_cap=5.0, tinker_cap=0.0)
N_CAP, K_CAP, N_TRAIT, K_TRAIT = 40, 8, 24, 4


async def main():
    policy = Policy(spend, model=MODEL, concurrency=32)
    llm = OpenAIClient(spend_o, concurrency=32)
    out = {"tasks": {}}
    rows = load_rows("train")

    async def cap_one(env, row):
        e = make_env(env, row, policy.renderer)
        samples = await policy.sample([{"role": "user", "content": e.get_question()}], K_CAP, max_tokens=3000)
        return [float(e.check_answer(s["final"])) if s["final"] else 0.0 for s in samples], [s["completion_tokens"] for s in samples], [s["no_final"] for s in samples]

    for env in ENV_NAMES:
        res = await asyncio.gather(*[cap_one(env, r) for r in rows[env][:N_CAP]])
        p = np.array([np.mean(r[0]) for r in res])
        toks = [t for r in res for t in r[1]]
        out["tasks"][env] = {"accuracy": float(p.mean()), "groups_with_signal_at_16": float(np.mean(1 - p ** 16 - (1 - p) ** 16)),
                             "mean_tokens": float(np.mean(toks)), "cut_off": float(np.mean([c for r in res for c in r[2]]))}
        print(env, {k: round(v, 3) for k, v in out["tasks"][env].items()}, flush=True)

    trait_rows = read_jsonl(TRAIT_DATA / "health_train.jsonl")[:N_TRAIT]

    async def trait_one(row):
        samples = await policy.sample([{"role": "system", "content": WORD_INSTRUCTION}, *row["turns"]], K_TRAIT, max_tokens=3000)
        rewards, words = [], []
        for s in samples:
            g = await grade(llm, row["turns"], s["final"], row["rubric"], "strict")
            w = len(s["final"].split())
            rewards.append((g["score"] or 0.0) * length_factor(w))
            words.append(w)
        return rewards, words, [s["completion_tokens"] for s in samples], [s["no_final"] for s in samples]

    res = await asyncio.gather(*[trait_one(r) for r in trait_rows])
    rew = [np.array(r[0]) for r in res]
    out["trait"] = {"mean_reward": float(np.mean([x.mean() for x in rew])), "groups_with_variance": float(np.mean([x.std() > 0 for x in rew])),
                    "mean_words": float(np.mean([w for r in res for w in r[1]])), "mean_tokens": float(np.mean([t for r in res for t in r[2]])),
                    "cut_off": float(np.mean([c for r in res for c in r[3]])), "mean_abs_advantage": float(np.mean([np.abs(x - x.mean()).mean() for x in rew]))}
    print("trait", {k: round(v, 3) for k, v in out["trait"].items()}, flush=True)
    spend.save()
    spend_o.save()
    out["spend"] = {"tinker": spend.d["tinker_usd"], "openai": spend_o.d["openai_usd"]}
    (ROOT / "results" / f"{args.model}_calibration.json").write_text(json.dumps(out, indent=1))
    print(f"CALIBRATION DONE | Tinker ${spend.d['tinker_usd']:.2f} | OpenAI ${spend_o.d['openai_usd']:.2f}", flush=True)


asyncio.run(main())
