"""Seeded subsets of the remaining public evals. All are eval-only, and canary fields are dropped."""
import ast
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from btrl.common import RAW, write_jsonl

rng = random.Random(6)
rows = [json.loads(x) for x in open(hf_hub_download("openai/healthbench-professional", "healthbench_professional_eval.jsonl", repo_type="dataset"))]
rng.shuffle(rows)
write_jsonl(RAW / "healthbench_pro.jsonl", [{"id": r["id"], "prompt": r["conversation"]["messages"], "rubrics": [{"criterion": c["criterion_text"], "points": c["points"]} for c in r["rubric_items"]], "use_case": r["use_case"], "type": r["type"]} for r in rows[:60]])
print("healthbench pro", len(rows), "-> 60")

db = []
for domain in ("economy", "education", "entertainment", "healthcare", "social"):
    items = json.load(open(hf_hub_download("skyai798/DeceptionBench", f"{domain}.json", repo_type="dataset")))
    rng.shuffle(items)
    print("deceptionbench", domain, len(items))
    for i, r in enumerate(items[:8]):
        db.append({"id": f"db-{domain}-{i}", **r})
write_jsonl(RAW / "deceptionbench.jsonl", db)

mb = load_dataset("morebench/morebench", "morebench_public", split="test").shuffle(seed=6).select(range(25))
out = []
for i, r in enumerate(mb):
    rubric = ast.literal_eval(r["RUBRIC"])
    if i == 0:
        print("morebench rubric fields", {k: str(v)[:70] for k, v in rubric[0].items()})
    out.append({"id": f"more-{i}", "dilemma": r["DILEMMA"], "rubric": rubric})
write_jsonl(RAW / "morebench.jsonl", out)
print("morebench 500 -> 25 | rubric items per dilemma", [len(x["rubric"]) for x in out[:6]])

sr = list(csv.DictReader(open(RAW / "strongreject_small.csv")))
write_jsonl(RAW / "strongreject.jsonl", [{"id": f"sr-{i}", "prompt": r["forbidden_prompt"], "category": r["category"]} for i, r in enumerate(sr)])
print("strongreject small", len(sr))

gp = load_dataset("hendrydong/gpqa_diamond_mc", split="test")
pattern, ab = r"(?<=\. )[^\.\?\!]*\?$", []
for i, r in enumerate(gp):
    text = re.sub(r"\s*Please write your final answer in the form of.*$", "", r["problem"], flags=re.S)
    parts = re.split(r"\n\s*\n(?=\(A\))", text, maxsplit=1)
    if len(parts) != 2:
        continue
    m = re.search(pattern, parts[0].strip())
    if m:
        ab.append({"id": f"ab-{i}", "question": m.group().strip(), "options": parts[1].strip()})
write_jsonl(RAW / "abstention.jsonl", ab)
print("abstention, GPQA with removable context:", len(ab), "| example:", repr(ab[0]["question"][:120]) if ab else None)

hm = []
for name in ("hmmt_feb_2025", "hmmt_nov_2025"):
    for r in load_dataset(f"MathArena/{name}", split="train"):
        hm.append({"id": f"{name}-{r['problem_idx']}", "problem": r["problem"], "answer": str(r["answer"])})
write_jsonl(RAW / "hmmt.jsonl", hm)
print("hmmt", len(hm))
