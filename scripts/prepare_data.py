"""Downloads fixed, seeded subsets of the public datasets. Canary fields are dropped: these rows are for eval only, except MedQA."""
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset

from btrl.common import RAW, write_jsonl

RAW.mkdir(parents=True, exist_ok=True)

medqa = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=0)
rows = [{"id": f"medqa-{i}", "question": r["question"], "options": r["options"], "answer": r["answer_idx"]} for i, r in enumerate(medqa.select(range(400)))]
assert not any("canary" in json.dumps(r).lower() or "should never appear" in json.dumps(r).lower() for r in rows), "MedQA carries a training canary"
write_jsonl(RAW / "medqa.jsonl", rows)
print("medqa", len(medqa), "total; saved", len(rows))

mcqa = load_dataset("openlifescienceai/medmcqa", split="train").filter(lambda r: r["choice_type"] == "single").shuffle(seed=0)
letters = "ABCD"
rows = [{"id": f"medmcqa-{i}", "question": r["question"], "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}, "answer": letters[int(r["cop"])]} for i, r in enumerate(mcqa.select(range(400)))]
assert not any("canary" in json.dumps(r).lower() or "should never appear" in json.dumps(r).lower() for r in rows), "MedMCQA carries a training canary"
write_jsonl(RAW / "medmcqa.jsonl", rows)
print("medmcqa", len(mcqa), "single-answer rows; saved", len(rows))

gsm = load_dataset("openai/gsm8k", "main", split="test").shuffle(seed=0).select(range(100))
write_jsonl(RAW / "gsm8k.jsonl", [{"id": f"gsm-{i}", "question": r["question"], "answer": r["answer"].split("####")[-1].strip().replace(",", "")} for i, r in enumerate(gsm)])
print("gsm8k saved", len(gsm))

mask_rows = []
for cfg in ["known_facts", "disinformation", "continuations"]:
    ds = load_dataset("cais/MASK", cfg, split="test").filter(lambda r: r["type"] == "binary" and bool(r["belief_elicit_1"])).shuffle(seed=0)
    for i, r in enumerate(ds.select(range(min(100, len(ds))))):
        mask_rows.append({"id": f"mask-{cfg}-{i}", "config": cfg, "system_prompt": r["system_prompt"], "user_prompt": r["user_prompt"],
                          "proposition": r["proposition"], "belief_elicit_1": r["belief_elicit_1"], "ground_truth_letter": r["formatted_ground_truth"]})
    print("mask", cfg, len(ds))
write_jsonl(RAW / "mask.jsonl", mask_rows)

syc_rows = []
rng = random.Random(0)
for name in ["nlp_survey", "philpapers2020", "political_typology_quiz"]:
    url = f"https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_{name}.jsonl"
    lines = [json.loads(x) for x in urllib.request.urlopen(url, timeout=120).read().decode().splitlines() if x.strip()]
    rng.shuffle(lines)
    for i, r in enumerate(lines[:150]):
        no = r["answer_not_matching_behavior"]
        syc_rows.append({"id": f"syc-{name}-{i}", "source": name, "question": r["question"], "sycophantic": r["answer_matching_behavior"].strip(), "non_sycophantic": [x.strip() for x in (no if isinstance(no, list) else [no])]})
    print("sycophancy", name, len(lines))
write_jsonl(RAW / "sycophancy.jsonl", syc_rows)
