"""Steps 51 to 100 of the longer first phase: 400 more health conversations with opposite-behavior criteria. Same generator, traits, and subtopic spread as the first 400, with near-copies of those, of the direct arm's 800, and of the benchmark removed. The first 400 stay in their original order, so resuming at step 50 lands on the new rows."""
import asyncio
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
from scripts.anti_generate import mirror
from scripts.trait_generate import PER_CALL, SUBTOPICS, jaccard

OUT, TOTAL, LIMIT = ROOT / "data" / "prior100", 400, 0.5


def text(r: dict) -> str:
    return r["turns"][-1]["content"]


async def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "prior100_rl").mkdir(parents=True, exist_ok=True)
    spend = Spend(ROOT / "results" / "prior100_rl_spend_openai.json", openai_cap=7.0, tinker_cap=0.0)
    llm = OpenAIClient(spend, concurrency=24)
    first = read_jsonl(ROOT / "data" / "prior" / "health_train.jsonl")
    reference = [text(r) for r in first + read_jsonl(ROOT / "data" / "trait" / "health_train.jsonl") + read_jsonl(ROOT / "data" / "benchmark" / "eval_scenarios.jsonl")]
    calls = [(t, SUBTOPICS[(i + 2 * j + 1) % len(SUBTOPICS)]) for i, t in enumerate(TRAITS) for j in range(6)]  # the offset spreads these over the subtopics the first 400 used least for each trait
    groups = await asyncio.gather(*[generate(llm, t, "health", PER_CALL, s) for t, s in calls])
    rows = [c for g in groups for c in g]
    scored = [(max(jaccard(text(r), x) for x in reference), r) for r in rows]
    kept, seen = [], []
    for score, r in scored:
        if score < LIMIT and all(jaccard(text(r), x) < LIMIT for x in seen):
            kept.append(r)
            seen.append(text(r))
    random.Random(2).shuffle(kept)
    by_trait = {t: [r for r in kept if r["trait"] == t] for t in TRAITS}
    ordered = []
    while any(by_trait.values()):
        for t in TRAITS:
            if by_trait[t]:
                ordered.append(by_trait[t].pop())
    done, queue = [], list(ordered)
    while len(done) < TOTAL and queue:  # top up from the spare conversations if any mirroring fails
        batch, queue = queue[: TOTAL - len(done)], queue[TOTAL - len(done):]
        done += [m for m in await asyncio.gather(*[mirror(llm, r) for r in batch]) if m]
    new = [{"id": f"pr-{len(first) + i}", **r} for i, r in enumerate(done)]
    assert len(new) == TOTAL, f"only {len(new)} conversations"
    write_jsonl(OUT / "health_train.jsonl", first + new)
    spend.save()
    words = [len(c.split()) for r in new for c in r["rubric"]]
    print(f"generated {len(rows)} | near-copies dropped {len(rows) - len(kept)} | new {len(new)} | total {len(first) + len(new)} | highest similarity to earlier data among kept {max(s for s, r in scored if any(r is k for k in kept)):.2f} | criterion words mean {st.mean(words):.1f} | multi-turn {sum(len(r['turns']) > 1 for r in new) / len(new):.3f} | OpenAI ${spend.d['openai_usd']:.2f}", flush=True)
    print("by trait", Counter(r["trait"] for r in new).most_common(), flush=True)
    print("by subtopic", Counter(r["subtopic"] for r in new).most_common(), flush=True)


asyncio.run(main())
