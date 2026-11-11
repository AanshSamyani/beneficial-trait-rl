"""Stage 4 environments for the Tinker cookbook's RL trainer: MedMCQA, MedQA, MedCalc-Bench and MATH, each with a verifiable reward.

MedCalc scoring follows the official rules in ncbi-nlp/MedCalc-Bench evaluate.py, without its use of eval() on model text.
"""
import math
import re
from collections.abc import Sequence
from datetime import datetime
from functools import partial

import chz
from tinker_cookbook.recipes.math_rl.math_env import MathEnv
from tinker_cookbook.rl.problem_env import ProblemEnv, ProblemGroupBuilder
from tinker_cookbook.rl.types import EnvGroupBuilder, RLDataset, RLDatasetBuilder

from .common import MIX_DATA, OSS20B_MODEL, read_jsonl
from .llm import make_renderer
from .tasks import mcq_messages, parse_mcq

ENV_NAMES = ["medmcqa", "medqa", "medcalc", "math"]


def medcalc_prompt(row: dict) -> str:
    if row["calculator_id"] in (13, 68):
        form = "Give the date as MM/DD/YYYY."
    elif row["calculator_id"] == 69:
        form = "Give the gestational age as (weeks, days)."
    else:
        form = "Give only the number, without units."
    return f"Patient note:\n{row['note']}\n\nQuestion: {row['question']}\n\nWork through this, then end your reply with a final line of the form 'Answer: <value>'. {form}"


def medcalc_extract(reply: str) -> str | None:
    """The text after the last "Answer:" in the reply, with bold markers removed."""
    found = list(re.finditer(r"(?i:answer)\s*\**\s*:\s*\**\s*(.+)", reply))
    return found[-1].group(1).replace("*", "").strip() if found else None


def _number(text: str) -> float | None:
    m = re.search(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?|-?\.\d+", text)
    if not m:
        return None
    try:
        return float(m.group().replace(",", ""))
    except ValueError:
        return None


def _date(text: str) -> datetime | None:
    m = re.search(r"\d{1,2}/\d{1,2}/\d{4}", text)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(), "%m/%d/%Y")
    except ValueError:
        return None


def _weeks_days(text: str) -> tuple[int, int] | None:
    m = re.search(r"\(?[\"']?(\d+)\s*(?:weeks?)?[\"']?,?\s*[\"']?(\d+)\s*(?:days?)?[\"']?\s*\)?", text)
    return (int(m.group(1)), int(m.group(2))) if m else None


def medcalc_correct(answer: str | None, row: dict) -> bool:
    if not answer:
        return False
    if row["calculator_id"] in (13, 68):
        a, g = _date(answer), _date(row["gold"])
        return a is not None and a == g
    if row["calculator_id"] == 69:
        a, g = _weeks_days(answer), _weeks_days(row["gold"])
        return a is not None and a == g
    value = _number(answer)
    if value is None or not math.isfinite(value):
        return False
    if row["output_type"] == "integer":
        return round(value) == round(float(row["gold"]))
    return float(row["lower"]) <= value <= float(row["upper"])


class McqEnv(ProblemEnv):
    def __init__(self, row: dict, renderer):
        super().__init__(renderer)
        self.row = row

    def get_question(self) -> str:
        return mcq_messages(self.row)[0]["content"]

    def check_format(self, sample_str: str) -> bool:
        return parse_mcq(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        return parse_mcq(sample_str) == self.row["answer"]

    def get_reference_answer(self) -> str:
        return self.row["answer"]


class MedCalcEnv(ProblemEnv):
    def __init__(self, row: dict, renderer):
        super().__init__(renderer)
        self.row = row

    def get_question(self) -> str:
        return medcalc_prompt(self.row)

    def check_format(self, sample_str: str) -> bool:
        return medcalc_extract(sample_str) is not None

    def check_answer(self, sample_str: str) -> bool:
        return medcalc_correct(medcalc_extract(sample_str), self.row)

    def get_reference_answer(self) -> str:
        return str(self.row["gold"])


def make_env(env: str, row: dict, renderer) -> ProblemEnv:
    if env in ("medmcqa", "medqa"):
        return McqEnv(row, renderer)
    if env == "medcalc":
        return MedCalcEnv(row, renderer)
    return MathEnv(row["problem"], row["answer"], renderer)


def group_builder(env: str, row: dict, renderer, size: int) -> ProblemGroupBuilder:
    return ProblemGroupBuilder(env_thunk=partial(make_env, env, row, renderer), num_envs=size, dataset_name=env)


class MixDataset(RLDataset):
    """Each step takes the same fixed rows per environment. Extra slots (stage 5 trait prompts) replace the last rows of each environment's share."""

    def __init__(self, rows: dict[str, list[dict]], per_env: int, take: int, group_size: int, renderer, steps: int, extra=None):
        self.rows, self.per_env, self.take, self.group_size, self.renderer, self.steps, self.extra = rows, per_env, take, group_size, renderer, steps, extra

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch = [group_builder(env, r, self.renderer, self.group_size) for env in self.rows for r in self.rows[env][index * self.per_env: index * self.per_env + self.take]]
        if self.extra is not None:
            batch += self.extra(index, self.group_size)
        return batch

    def __len__(self) -> int:
        return self.steps


class MixTestDataset(RLDataset):
    def __init__(self, rows: dict[str, list[dict]], batch_size: int, renderer):
        self.items = [(env, r) for env in rows for r in rows[env]]
        self.batch_size, self.renderer = batch_size, renderer

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        chunk = self.items[index * self.batch_size: (index + 1) * self.batch_size]
        return [group_builder(env, r, self.renderer, 1) for env, r in chunk]

    def __len__(self) -> int:
        return math.ceil(len(self.items) / self.batch_size)


def load_rows(split: str, limit: int | None = None) -> dict[str, list[dict]]:
    return {env: read_jsonl(MIX_DATA / f"{env}_{split}.jsonl")[:limit] for env in ENV_NAMES}


def rows_unused_by_trait_arm(rows: dict[str, list[dict]], per_env: int = 8, take: int = 6, steps: int = 100) -> dict[str, list[dict]]:
    """The normal prompts the trait and non-beneficial arms never trained on: the last 2 of each 8-row share they replaced, plus everything past step 100."""
    return {env: [r for i, r in enumerate(v) if i >= per_env * steps or i % per_env >= take] for env, v in rows.items()}


@chz.chz
class MixDatasetBuilder(RLDatasetBuilder):
    per_env: int
    take: int
    group_size: int
    steps: int
    test_per_env: int
    model: str = OSS20B_MODEL

    async def __call__(self) -> tuple[MixDataset, MixTestDataset]:
        renderer = make_renderer(self.model)
        return MixDataset(load_rows("train"), self.per_env, self.take, self.group_size, renderer, self.steps), MixTestDataset(load_rows("heldout", self.test_per_env), 32, renderer)
