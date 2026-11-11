"""Stage 5: trait conversations as an RL environment. The reward is the strict judge's rubric score, scaled down when a reply runs over the word limit.

The penalty uses the reply's real word count, so claiming a shorter length earns nothing.
"""
import json
import math
from collections.abc import Sequence
from functools import partial

import chz
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import Action, ActionExtra, Env, EnvGroupBuilder, Observation, RLDataset, RLDatasetBuilder, StepResult

from .bench import grade
from .common import OSS20B_MODEL, ROOT, Spend, read_jsonl
from .llm import OpenAIClient, make_renderer
from .mix_env import ENV_NAMES, MixDataset, MixTestDataset, group_builder, load_rows, rows_unused_by_trait_arm

TRAIT_DATA = ROOT / "data" / "trait"
WORD_INSTRUCTION = "Keep your final reply under 400 words."
WORD_LIMIT = 450  # above this the reward shrinks in proportion: limit / words


def length_factor(words: int) -> float:
    return 1.0 if words <= WORD_LIMIT else WORD_LIMIT / words


class TraitEnv(Env):
    def __init__(self, row: dict, renderer, llm: OpenAIClient, log_path: str | None = None):
        self.row, self.renderer, self.llm, self.log_path = row, renderer, llm, log_path

    @property
    def stop_condition(self):
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, list[int]]:
        return self.renderer.build_generation_prompt([{"role": "system", "content": WORD_INSTRUCTION}, *self.row["turns"]]), self.stop_condition

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        message, termination = self.renderer.parse_response(action)
        text = renderers.get_text_content(message).strip() if termination.is_clean else ""
        words = len(text.split())
        graded = await grade(self.llm, self.row["turns"], text, self.row["rubric"], "strict")
        rubric = graded["score"] if graded["score"] is not None else 0.0
        reward = rubric * length_factor(words)
        if self.log_path:  # the trainer's per-tag training metrics are mislabelled once constant groups are filtered, so this is the reliable record
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"id": self.row["id"], "trait": self.row["trait"], "rubric": rubric, "met": graded["met"], "words": words, "reward": reward, "finished": bool(text), "final": text}, ensure_ascii=False) + "\n")
            self.llm.spend.save()
        return StepResult(
            reward=reward, episode_done=True, next_observation=tinker.ModelInput.empty(), next_stop_condition=self.stop_condition,
            metrics={"rubric": rubric, "correct": rubric, "words": float(words), "over_limit": float(words > WORD_LIMIT), "finished": float(bool(text))},
        )


def trait_builder(row: dict, renderer, llm: OpenAIClient, size: int, log_path: str | None = None) -> ProblemGroupBuilder:
    return ProblemGroupBuilder(env_thunk=partial(TraitEnv, row, renderer, llm, log_path), num_envs=size, dataset_name="trait")


class TraitTestDataset(RLDataset):
    """The mix's held-out prompts plus trait conversations, one sample each."""

    def __init__(self, mix: MixTestDataset, trait_rows: list[dict], renderer, llm: OpenAIClient, batch_size: int):
        self.mix, self.rows, self.renderer, self.llm, self.batch_size = mix, trait_rows, renderer, llm, batch_size
        self.n_mix = len(mix)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        if index < self.n_mix:
            return self.mix.get_batch(index)
        i = index - self.n_mix
        return [trait_builder(r, self.renderer, self.llm, 1) for r in self.rows[i * self.batch_size: (i + 1) * self.batch_size]]

    def __len__(self) -> int:
        return self.n_mix + math.ceil(len(self.rows) / self.batch_size)


@chz.chz
class TraitMixDatasetBuilder(RLDatasetBuilder):
    per_env: int
    take: int
    trait_slots: int
    group_size: int
    steps: int
    test_per_env: int
    test_trait: int
    ledger: str
    rollout_log: str
    train_file: str = "data/trait/health_train.jsonl"
    unused_mix_only: bool = False
    model: str = OSS20B_MODEL

    async def __call__(self) -> tuple[MixDataset, TraitTestDataset]:
        renderer = make_renderer(self.model)
        llm = OpenAIClient(Spend(ROOT / self.ledger, openai_cap=14.0, tinker_cap=0.0), concurrency=64)
        trait_rows = read_jsonl(ROOT / self.train_file)
        test_rows = read_jsonl(TRAIT_DATA / "tracking.jsonl")[: self.test_trait]

        log_path = str(ROOT / self.rollout_log)
        (ROOT / self.rollout_log).parent.mkdir(parents=True, exist_ok=True)

        def extra(index: int, size: int):
            return [trait_builder(r, renderer, llm, size, log_path) for r in trait_rows[index * self.trait_slots: (index + 1) * self.trait_slots]]

        mix = load_rows("train")
        train = MixDataset(rows_unused_by_trait_arm(mix) if self.unused_mix_only else mix, self.per_env, self.take, self.group_size, renderer, self.steps, extra=extra)
        return train, TraitTestDataset(MixTestDataset(load_rows("heldout", self.test_per_env), 32, renderer), test_rows, renderer, llm, 32)


# Dose arms: the normal arm's 32 prompts per step with s of them swapped for trait conversations, where s follows a fixed
# repeating pattern so the share over 100 steps is exactly the target (5%: 1.6 per step; 10%: 3.2). The capability prompts are
# the first 32 - s of the normal arm's 8-per-environment share, with the environment that gives up a row rotating each step.
DOSE_PATTERNS = {5: [2, 1, 2, 1, 2], 10: [3, 3, 4, 3, 3]}


class DoseMixDataset(RLDataset):
    def __init__(self, rows: dict[str, list[dict]], trait_rows: list[dict], dose: int, per_env: int, group_size: int, steps: int, renderer, llm, log_path: str):
        self.rows, self.trait_rows, self.pattern = rows, trait_rows, DOSE_PATTERNS[dose]
        self.per_env, self.group_size, self.steps, self.renderer, self.llm, self.log_path = per_env, group_size, steps, renderer, llm, log_path

    def plan(self, index: int) -> dict:
        """Which rows step `index` uses; also used to check the schedule without sampling."""
        slots = self.pattern[index % len(self.pattern)]
        start = sum(self.pattern[i % len(self.pattern)] for i in range(index))
        envs = list(self.rows)
        n_cap = self.per_env * len(envs) - slots
        base, extra = divmod(n_cap, len(envs))
        order = envs[index % len(envs):] + envs[: index % len(envs)]  # rotate which environments keep the extra row
        take = {env: base + (1 if order.index(env) < extra else 0) for env in envs}
        cap = {env: self.rows[env][index * self.per_env: index * self.per_env + take[env]] for env in envs}
        return {"cap": cap, "trait": self.trait_rows[start: start + slots]}

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        p = self.plan(index)
        batch = [group_builder(env, r, self.renderer, self.group_size) for env, rs in p["cap"].items() for r in rs]
        return batch + [trait_builder(r, self.renderer, self.llm, self.group_size, self.log_path) for r in p["trait"]]

    def __len__(self) -> int:
        return self.steps


@chz.chz
class DoseMixDatasetBuilder(RLDatasetBuilder):
    dose: int
    per_env: int
    group_size: int
    steps: int
    test_per_env: int
    test_trait: int
    ledger: str
    rollout_log: str

    async def __call__(self) -> tuple[DoseMixDataset, TraitTestDataset]:
        renderer = make_renderer(OSS20B_MODEL)
        llm = OpenAIClient(Spend(ROOT / self.ledger, openai_cap=14.0, tinker_cap=0.0), concurrency=64)
        (ROOT / self.rollout_log).parent.mkdir(parents=True, exist_ok=True)
        train = DoseMixDataset(load_rows("train"), read_jsonl(TRAIT_DATA / "health_train.jsonl"), self.dose, self.per_env, self.group_size, self.steps,
                               renderer, llm, str(ROOT / self.rollout_log))
        test_rows = read_jsonl(TRAIT_DATA / "tracking.jsonl")[: self.test_trait]
        return train, TraitTestDataset(MixTestDataset(load_rows("heldout", self.test_per_env), 32, renderer), test_rows, renderer, llm, 32)
