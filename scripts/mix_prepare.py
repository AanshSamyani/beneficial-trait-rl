"""Fixed train and held-out splits for stage 4, with self-tests of the MedCalc scorer on the dataset's own answers."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import load_dataset
from tinker_cookbook.recipes.math_rl.math_env import _get_hendrycks_math_test, _get_hendrycks_math_train
from tinker_cookbook.recipes.math_rl.math_grading import extract_boxed

from btrl.common import MIX_DATA, write_jsonl
from btrl.mix_env import medcalc_correct, medcalc_extract

TRAIN, HELDOUT, SEED = 900, 300, 4
letters = "ABCD"
norm = lambda q: re.sub(r"\W+", " ", q.lower()).strip()


def save(env, train, heldout):
    h = {norm(r["key"]) for r in heldout}
    train = [r for r in train if norm(r["key"]) not in h][:TRAIN]
    assert not any("canary" in json.dumps(r).lower() or "should never appear" in json.dumps(r).lower() for r in train + heldout), env
    write_jsonl(MIX_DATA / f"{env}_train.jsonl", [{k: v for k, v in r.items() if k != "key"} for r in train])
    write_jsonl(MIX_DATA / f"{env}_heldout.jsonl", [{k: v for k, v in r.items() if k != "key"} for r in heldout])
    print(f"{env}: train {len(train)} heldout {len(heldout)}", flush=True)


# MedMCQA
def mm(r, i, p):
    return {"id": f"{p}-{i}", "key": r["question"], "question": r["question"], "options": {"A": r["opa"], "B": r["opb"], "C": r["opc"], "D": r["opd"]}, "answer": letters[int(r["cop"])]}
tr = load_dataset("openlifescienceai/medmcqa", split="train").filter(lambda r: r["choice_type"] == "single").shuffle(seed=SEED)
va = load_dataset("openlifescienceai/medmcqa", split="validation").filter(lambda r: r["choice_type"] == "single").shuffle(seed=SEED)
save("medmcqa", [mm(r, i, "mtr") for i, r in enumerate(tr.select(range(TRAIN + 100)))], [mm(r, i, "mhe") for i, r in enumerate(va.select(range(HELDOUT)))])

# MedQA
def mq(r, i, p):
    return {"id": f"{p}-{i}", "key": r["question"], "question": r["question"], "options": r["options"], "answer": r["answer_idx"]}
tr = load_dataset("GBaker/MedQA-USMLE-4-options", split="train").shuffle(seed=SEED)
te = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").shuffle(seed=SEED)
save("medqa", [mq(r, i, "qtr") for i, r in enumerate(tr.select(range(TRAIN + 100)))], [mq(r, i, "qhe") for i, r in enumerate(te.select(range(HELDOUT)))])

# MedCalc-Bench v1.2
def mc(r, i, p):
    return {"id": f"{p}-{i}", "key": r["Patient Note"][:300] + r["Question"], "note": r["Patient Note"], "question": r["Question"], "calculator_id": int(r["Calculator ID"]), "output_type": r["Output Type"], "gold": str(r["Ground Truth Answer"]), "lower": str(r["Lower Limit"]), "upper": str(r["Upper Limit"]), "category": r["Category"]}
tr = load_dataset("ncbi/MedCalc-Bench-v1.2", split="train").shuffle(seed=SEED)
te = load_dataset("ncbi/MedCalc-Bench-v1.2", split="test").shuffle(seed=SEED)
print("medcalc output types:", Counter(tr["Output Type"]).most_common(), "| test rows", len(te), flush=True)
all_rows = [mc(r, i, "c") for i, r in enumerate(tr)] + [mc(r, i, "t") for i, r in enumerate(te)]
bad = [r for r in all_rows if not medcalc_correct(f"Answer: {r['gold']}".split(": ", 1)[1], r)]
print("medcalc self-test: dataset's own answers scored wrong:", len(bad), "of", len(all_rows), "| examples", [(r["calculator_id"], r["output_type"], r["gold"]) for r in bad[:5]], flush=True)
assert len(bad) / len(all_rows) < 0.01
save("medcalc", [mc(r, i, "ctr") for i, r in enumerate(tr.select(range(TRAIN + 100)))], [mc(r, i, "che") for i, r in enumerate(te.select(range(HELDOUT)))])

# MATH: the cookbook's split, which keeps MATH-500 out of training
tr = _get_hendrycks_math_train().shuffle(seed=SEED)
te = _get_hendrycks_math_test().shuffle(seed=SEED)
def ma(r, i, p):
    return {"id": f"{p}-{i}", "key": r["problem"], "problem": r["problem"], "answer": extract_boxed(r["solution"])}
save("math", [ma(r, i, "atr") for i, r in enumerate(tr.select(range(TRAIN + 100)))], [ma(r, i, "ahe") for i, r in enumerate(te.select(range(HELDOUT)))])

print("=== MedCalc reply parsing tests ===")
dec = {"calculator_id": 2, "output_type": "decimal", "gold": "141.04", "lower": "133.98", "upper": "148.09"}
integer = {"calculator_id": 4, "output_type": "integer", "gold": "3", "lower": "3", "upper": "3"}
date = {"calculator_id": 13, "output_type": "date", "gold": "03/15/2021", "lower": "", "upper": ""}
weeks = {"calculator_id": 69, "output_type": "integer", "gold": "(38, 2)", "lower": "", "upper": ""}
for text, row, want in [("Steps...\n\nAnswer: 141.0", dec, True), ("**Answer:** 141.04 mL/min", dec, True), ("Answer: **140**", dec, True), ("Answer: 1,410", dec, False), ("Answer: 200", dec, False), ("I can't determine this.", dec, False), ("Answer: 3", integer, True), ("Answer: 3.2", integer, True), ("Answer: 4", integer, False), ("Answer: 03/15/2021", date, True), ("**Answer:** 3/15/2021", date, True), ("Answer: 03/16/2021", date, False), ("Answer: (38, 2)", weeks, True), ("Answer: 38 weeks, 2 days", weeks, True), ("Answer: (38, 3)", weeks, False)]:
    got = medcalc_correct(medcalc_extract(text), row)
    print("ok " if got == want else "BAD", repr(text), "->", got)
