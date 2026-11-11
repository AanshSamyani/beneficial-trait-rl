"""Stage 1: builds the held-out trait benchmark in the paper's format. Seven traits across all twelve domains, three conversations each."""
import asyncio
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btrl.common import ROOT, Spend, write_jsonl
from btrl.llm import OpenAIClient
from btrl.paper import DOMAINS, EVAL_TRAITS
from btrl.paper_gen import generate, valid

DATA, RES = ROOT / "data" / "benchmark", ROOT / "results" / "benchmark"
N_PER_CELL = 3

async def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    spend = Spend(RES / "spend.json", openai_cap=12.0, tinker_cap=0.5)
    llm = OpenAIClient(spend, concurrency=16)
    jobs = [(t, d) for t in EVAL_TRAITS for d in DOMAINS]
    groups = await asyncio.gather(*[generate(llm, t, d, N_PER_CELL) for t, d in jobs])
    raw = [c for g in groups for c in g]
    rows = [c for c in raw if valid(c)]
    random.Random(0).shuffle(rows)
    rows = [{"id": f"bench-{i}", **c} for i, c in enumerate(rows)]
    write_jsonl(DATA / "eval_scenarios.jsonl", rows)
    spend.save()

    words = [len(c.split()) for r in rows for c in r["rubric"]]
    last = [len(r["turns"][-1]["content"].split()) for r in rows]
    named = [r["id"] for r in rows if re.search(r"\b" + r["trait"].replace("_", " ") + r"\b", " ".join(x["content"] for x in r["turns"]), re.I)]
    print("generated", len(raw), "| valid", len(rows))
    print("by trait", Counter(r["trait"] for r in rows).most_common())
    print("by domain", Counter(r["domain"] for r in rows).most_common())
    print("multi-turn share", round(sum(len(r["turns"]) > 1 for r in rows) / len(rows), 3), "| rubric words mean", round(st.mean(words), 1), "max", max(words), "| final turn words median", st.median(last), "| conversations naming their trait", named)
    print("spend", {k: round(v, 3) for k, v in spend.d.items()})


asyncio.run(main())
