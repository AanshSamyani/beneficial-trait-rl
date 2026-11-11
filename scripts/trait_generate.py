"""Stage 5 training data: health conversations across all 15 paper traits. Near-copies of any benchmark conversation are removed."""
import asyncio
import json
import random
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btrl.common import ROOT, Spend, read_jsonl, write_jsonl
from btrl.llm import OpenAIClient
from btrl.paper import TRAITS
from btrl.paper_gen import generate
from btrl.trait_env import TRAIT_DATA

SUBTOPICS = [
    "emergency and urgent triage", "primary care and chronic disease management", "pediatrics and adolescent care",
    "older adults, long-term care, and dementia", "mental health and substance use", "pregnancy, postpartum, and reproductive care",
    "oncology and serious-illness decisions", "medications, pharmacy, and supplements", "hospital operations, handoffs, and patient safety",
    "public health, outbreaks, and vaccination", "clinical research, evidence appraisal, and health misinformation", "insurance, access, billing, and care for underserved patients",
]
PER_CALL, TOTAL = 5, 800


def jaccard(a: str, b: str) -> float:
    x, y = set(a.lower().split()), set(b.lower().split())
    return len(x & y) / len(x | y)


async def main():
    TRAIT_DATA.mkdir(parents=True, exist_ok=True)
    spend = Spend(ROOT / "results" / "trait_rl_spend_openai.json", openai_cap=14.0, tinker_cap=0.0)
    llm = OpenAIClient(spend, concurrency=24)
    groups = await asyncio.gather(*[generate(llm, t, "health", PER_CALL, s) for t in TRAITS for s in SUBTOPICS])
    rows = [c for g in groups for c in g]
    bench = read_jsonl(ROOT / "data" / "benchmark" / "eval_scenarios.jsonl")
    bench_text = [r["turns"][-1]["content"] for r in bench]
    kept = [r for r in rows if max(jaccard(r["turns"][-1]["content"], b) for b in bench_text) < 0.5]
    random.Random(0).shuffle(kept)
    # Spread traits evenly over training steps.
    by_trait = {t: [r for r in kept if r["trait"] == t] for t in TRAITS}
    ordered = []
    while any(by_trait.values()):
        for t in TRAITS:
            if by_trait[t]:
                ordered.append(by_trait[t].pop())
    ordered = [{"id": f"tr-{i}", **r} for i, r in enumerate(ordered[:TOTAL])]
    write_jsonl(TRAIT_DATA / "health_train.jsonl", ordered)
    # Tracking during training uses benchmark conversations from the calibration set. The 154 untouched ones stay unseen until the end.
    ids = set(json.loads((ROOT / "data" / "benchmark" / "calibration_ids.json").read_text()))
    write_jsonl(TRAIT_DATA / "tracking.jsonl", [r for r in bench if r["id"] in ids][:42])
    spend.save()
    w = [len(c.split()) for r in ordered for c in r["rubric"]]
    print("generated", len(rows), "| near-copies dropped", len(rows) - len(kept), "| kept", len(ordered), "| rubric words mean", round(st.mean(w), 1), "| multi-turn", round(sum(len(r["turns"]) > 1 for r in ordered) / len(ordered), 3))
    print("by trait", Counter(r["trait"] for r in ordered).most_common())
    print("first 30 traits in order:", Counter(r["trait"] for r in ordered[:30]).most_common(3))
    print("spend", {k: round(v, 3) for k, v in spend.d.items()})


if __name__ == "__main__":
    asyncio.run(main())
