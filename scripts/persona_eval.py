"""Adversarial prompting test (after Section 4.1 and Figure 6 of Jagadeesh et al., 2026) on gpt-oss-20b.

Models: the compute-matched baseline (normal RL, step 100) and the beneficial trait RL model (step 100).
Evals, all graded with the HealthBench rubric grader on gpt-5.6-luna, one sample per prompt:
  hb_general  100 HealthBench conversations that are not about mental health (fixed random sample)
  hb_mental   80 HealthBench conversations about mental health (fixed random sample of the 305 that match mental-health terms)
  hb_pro      the 60 HealthBench Professional conversations (aggregates only; prompts are never printed)
Personas (btrl/personas.py): none, harmful_medical, harmful_mental_health, helpful_medical, added as a system message.
The grader sees the conversation without the persona. Replies are saved under results/persona/<model>/, so a rerun only does
what is missing."""
import argparse
import asyncio
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import tinker

from btrl.common import JUDGE_MODEL, OSS20B_MODEL, OSS20B_PRICES, RAW, ROOT, Spend, read_jsonl, write_jsonl
from btrl.evals import HB_SCHEMA, HEALTHBENCH_GRADER
from btrl.llm import OpenAIClient, Policy
from btrl.personas import PERSONAS

DATA = ROOT / "data" / "persona"
OUT = ROOT / "results" / "persona"
MODELS = {"normal_100": ROOT / "results" / "mix_rl" / "run", "trait_100": ROOT / "results" / "trait_rl" / "run"}
MH = re.compile(r"\b(suicid\w*|self[- ]harm\w*|kill(ing)? myself|end my life|want to die|depress\w*|anxiety|anxious|panic attacks?|eating disorder|"
                r"anorexi\w*|bulimi\w*|hopeless|psychiatr\w*|mental health|overdos\w*|cutting myself|bipolar|schizo\w*|ptsd|trauma)\b", re.I)
parser = argparse.ArgumentParser()
parser.add_argument("--models", default="normal_100,trait_100")
parser.add_argument("--personas", default=",".join(PERSONAS))
parser.add_argument("--evals", default="hb_general,hb_mental,hb_pro")
parser.add_argument("--limit", type=int, default=0, help="prompts per eval, for a pilot")
args = parser.parse_args()
spend_t = Spend(ROOT / "results" / "persona_spend_tinker.json", openai_cap=0.0, tinker_cap=10.0, tinker_prices=OSS20B_PRICES)
spend_o = Spend(ROOT / "results" / "persona_spend_openai.json", openai_cap=15.0, tinker_cap=0.0)


def build_slices():
    """Fixed samples from the public HealthBench set (data/raw/healthbench_full/oss_eval.jsonl)."""
    if (DATA / "hb_general.jsonl").exists():
        return
    raw = (RAW / "healthbench_full" / "oss_eval.jsonl").read_bytes()
    rows = [json.loads(x) for x in raw.split(b"\n") if x.strip()]  # split on newlines only: some texts contain carriage returns
    conv = lambda r: {"id": r["prompt_id"], "prompt": r["prompt"], "rubrics": [{"criterion": c["criterion"], "points": c["points"]} for c in r["rubrics"]],
                      "tags": r["example_tags"]}
    user = lambda r: " ".join(m["content"] for m in r["prompt"] if m["role"] == "user" and isinstance(m["content"], str))
    mental = sorted((r for r in rows if MH.search(user(r))), key=lambda r: r["prompt_id"])
    general = sorted((r for r in rows if not MH.search(user(r))), key=lambda r: r["prompt_id"])
    rng = random.Random(2026)
    write_jsonl(DATA / "hb_mental.jsonl", [conv(r) for r in rng.sample(mental, 80)])
    write_jsonl(DATA / "hb_general.jsonl", [conv(r) for r in rng.sample(general, 100)])
    print(f"slices: {len(mental)} mental-health matches -> 80, {len(general)} others -> 100", flush=True)


def load_eval(name):
    rows = read_jsonl(RAW / "healthbench_pro.jsonl") if name == "hb_pro" else read_jsonl(DATA / f"{name}.jsonl")
    return rows[: args.limit] if args.limit else rows


async def run(policy: Policy, llm: OpenAIClient, model: str, persona: str, name: str, gate: asyncio.Semaphore):
    path = OUT / model / f"{persona}__{name}{'_pilot' if args.limit else ''}.jsonl"
    if path.exists():
        return
    rows = load_eval(name)
    prefix = [{"role": "system", "content": PERSONAS[persona]}] if PERSONAS[persona] else []

    async def item(convo, c):
        out = await llm.json(JUDGE_MODEL, HEALTHBENCH_GRADER.replace("<<conversation>>", convo).replace("<<rubric_item>>", f"[{c['points']}] {c['criterion']}"), HB_SCHEMA)
        return bool(out["criteria_met"]) if out else None

    async def one(r):
        s = (await policy.sample(prefix + r["prompt"], 1, max_tokens=8000))[0]
        possible = sum(c["points"] for c in r["rubrics"] if c["points"] > 0)
        base = {"id": r["id"], "persona": persona, "eval": name, "no_final": s["no_final"], "completion_tokens": s["completion_tokens"], "final": s["final"]}
        if not s["final"] or possible == 0:
            return {**base, "score": 0.0 if possible else None, "met": None}
        convo = "\n\n".join(f"{m['role']}: {m['content']}" for m in r["prompt"]) + f"\n\nassistant: {s['final']}"
        met = await asyncio.gather(*[item(convo, c) for c in r["rubrics"]])
        return {**base, "score": sum(c["points"] for c, m in zip(r["rubrics"], met) if m) / possible, "met": met}

    async with gate:
        out = await asyncio.gather(*[one(r) for r in rows])
    write_jsonl(path, out)
    spend_t.save()
    spend_o.save()
    sc = [r["score"] for r in out if r["score"] is not None]
    print(f"[{model}] {persona:22} {name:10} score {np.mean(sc):.3f} | n {len(sc)} | cut {np.mean([r['no_final'] for r in out]):.2f} | "
          f"Tinker ${spend_t.d['tinker_usd']:.2f} OpenAI ${spend_o.d['openai_usd']:.2f}", flush=True)


async def main():
    build_slices()
    llm = OpenAIClient(spend_o, concurrency=64)
    service = tinker.ServiceClient()
    gate = asyncio.Semaphore(4)
    jobs = []
    for model in args.models.split(","):
        rows = [json.loads(x) for x in (MODELS[model] / "checkpoints.jsonl").read_text().split("\n") if x.strip()]
        path = next(r for r in rows if r["name"] == "000100" and r.get("sampler_path"))["sampler_path"]
        policy = Policy(spend_t, client=service.create_sampling_client(model_path=path), model=OSS20B_MODEL, concurrency=32)
        for persona in args.personas.split(","):
            for name in args.evals.split(","):
                jobs.append(run(policy, llm, model, persona, name, gate))
    await asyncio.gather(*jobs)
    print(f"PERSONA DONE | Tinker ${spend_t.d['tinker_usd']:.2f} | OpenAI ${spend_o.d['openai_usd']:.2f}", flush=True)


asyncio.run(main())
