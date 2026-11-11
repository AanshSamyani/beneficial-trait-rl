"""Fixed, seeded subsets of the public evals for stage 3. These are eval-only; canary fields are dropped so they travel nowhere."""
import json
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from btrl.common import RAW, write_jsonl

rng = random.Random(3)

path = hf_hub_download("openai/healthbench", "2025-05-07-06-14-12_oss_eval.jsonl", repo_type="dataset")
rows = [json.loads(x) for x in open(path)]
rng.shuffle(rows)
hb = [{"id": r["prompt_id"], "prompt": r["prompt"], "rubrics": [{"criterion": c["criterion"], "points": c["points"]} for c in r["rubrics"]], "tags": r["example_tags"]} for r in rows[:40]]
write_jsonl(RAW / "healthbench.jsonl", hb)
print("healthbench", len(rows), "->", len(hb))

ds = load_dataset("longtermrisk/school-of-reward-hacks", split="train").shuffle(seed=3)
srh = [{"id": f"srh-{i}", "user": r["user"], "hack": r["school_of_reward_hacks"], "control": r["control"], "task": r["task"], "metric": r["evaluation_metric"], "cheat": r["cheat_method"]} for i, r in enumerate(ds.select(range(120)))]
write_jsonl(RAW / "reward_hacks.jsonl", srh)
print("reward hacks", len(ds), "->", len(srh))

ds = load_dataset("hendrydong/gpqa_diamond_mc", split="test").shuffle(seed=3)
gp = [{"id": f"gpqa-{i}", "problem": r["problem"], "answer": r["solution"].replace("\\boxed{", "").replace("}", "").strip(), "domain": r.get("domain")} for i, r in enumerate(ds.select(range(100)))]
assert all(g["answer"] in "ABCD" for g in gp)
write_jsonl(RAW / "gpqa.jsonl", gp)
print("gpqa", len(ds), "->", len(gp), "| example tail:", repr(gp[0]["problem"][-160:]))

ds = load_dataset("google/IFEval", split="train").shuffle(seed=3)
ife = [{"id": f"ife-{r['key']}", "prompt": r["prompt"], "instruction_ids": list(r["instruction_id_list"]), "kwargs": list(r["kwargs"])} for r in ds.select(range(100))]
write_jsonl(RAW / "ifeval.jsonl", ife)
print("ifeval", len(ds), "->", len(ife))

base = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/evaluation/"
first = yaml.safe_load(urllib.request.urlopen(base + "first_plot_questions.yaml", timeout=60).read())
prereg = yaml.safe_load(urllib.request.urlopen(base + "preregistered_evals.yaml", timeout=60).read())
plain = [q for q in first if not q["id"].endswith(("_json", "_template")) and q["type"].startswith("free_form")]
bet = [{"id": q["id"], "question": q["paraphrases"][0], "aligned": q["judge_prompts"]["aligned"], "coherent": q["judge_prompts"]["coherent"]} for q in plain + prereg if not q.get("system")]
write_jsonl(RAW / "betley.jsonl", bet)
print("betley first-plot", len(plain), "+ preregistered", len(prereg), "->", len(bet))
