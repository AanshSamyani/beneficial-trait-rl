"""OpenAI helpers for generation and judging, plus Tinker sampling for the policy."""
import asyncio
import json
import warnings

import tinker
from openai import AsyncOpenAI
from tinker_cookbook.renderers.gpt_oss import GptOssRenderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from .common import BASE_MODEL, Spend

warnings.filterwarnings("ignore", message="use_system_prompt=True but messages already start")


class OpenAIClient:
    def __init__(self, spend: Spend, concurrency: int = 24):
        self.client, self.spend = AsyncOpenAI(max_retries=6, timeout=300), spend
        self.sem = asyncio.Semaphore(concurrency)

    async def json(self, model: str, prompt: str, schema: dict, effort: str = "low") -> dict | None:
        """Returns parsed JSON, or None if the model refused or returned nothing usable."""
        async with self.sem:
            r = await self.client.chat.completions.create(
                model=model,
                reasoning_effort=effort,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": schema}},
            )
        self.spend.add_openai(model, r.usage.prompt_tokens, r.usage.completion_tokens)
        content = r.choices[0].message.content
        return json.loads(content) if content else None


SAMPLE_TIMEOUT, SAMPLE_ATTEMPTS = 900, 3


def make_renderer(model: str = BASE_MODEL):
    if model.startswith("thinkingmachines/Inkling"):
        return _inkling_renderer(model)
    if model.startswith("nvidia/NVIDIA-Nemotron"):
        from tinker_cookbook.renderers import get_renderer
        return get_renderer("nemotron3_ultra", get_tokenizer(model), model_name=model)
    return GptOssRenderer(get_tokenizer(model), use_system_prompt=True, reasoning_effort="medium", current_date="2026-11-03")


def _inkling_renderer(model: str):
    """Inkling's TMLv0 format, with a fixed medium reasoning effort for every prompt it builds (sampling and RL alike)."""
    from tinker_cookbook.renderers.tml_v0 import TmlV0Renderer

    from .common import INKLING_EFFORT

    class InklingRenderer(TmlV0Renderer):
        def build_generation_prompt(self, messages, role="assistant", prefill=None, effort=INKLING_EFFORT):
            return super().build_generation_prompt(messages, role, prefill, effort=effort)

    return InklingRenderer(get_tokenizer(model))


class Policy:
    """Samples from a gpt-oss model with its default Harmony system prompt (medium reasoning)."""

    def __init__(self, spend: Spend, client=None, concurrency: int = 24, model: str = BASE_MODEL):
        self.renderer = make_renderer(model)
        self.client = client or tinker.ServiceClient().create_sampling_client(base_model=model)
        self.spend, self.sem = spend, asyncio.Semaphore(concurrency)

    async def sample(self, messages: list[dict], n: int, max_tokens: int = 3000, temperature: float = 1.0, prefill: str | None = None) -> list[dict]:
        prompt = self.renderer.build_generation_prompt(messages, prefill=prefill) if prefill else self.renderer.build_generation_prompt(messages)
        # With a prefill, the assistant's first tokens sit in the prompt; they are put back in front of each sample so parsing and "_tokens" cover the whole reply.
        head = prompt.to_ints()[self.renderer.build_generation_prompt(messages).length:] if prefill else []
        params = tinker.SamplingParams(max_tokens=max_tokens, temperature=temperature, stop=self.renderer.get_stop_sequences())
        async with self.sem:
            # A call can hang forever if the machine sleeps, so give up and retry rather than wait.
            for attempt in range(SAMPLE_ATTEMPTS):
                try:
                    out = await asyncio.wait_for(self.client.sample_async(prompt=prompt, num_samples=n, sampling_params=params), SAMPLE_TIMEOUT)
                    break
                except asyncio.TimeoutError:
                    if attempt == SAMPLE_ATTEMPTS - 1:
                        raise
        results = []
        for seq in out.sequences:
            msg, term = self.renderer.parse_response(head + list(seq.tokens))
            content = msg["content"]
            if isinstance(content, str):
                thinking, final = "", content
            else:
                thinking = "".join(p.get("thinking", "") for p in content if p["type"] == "thinking")
                final = "".join(p.get("text", "") for p in content if p["type"] == "text")
            clean = bool(term.is_clean)
            results.append(
                {"final": final.strip() if clean else "", "thinking": thinking, "completion_tokens": len(seq.tokens),
                 "prompt_tokens": prompt.length, "clean": clean, "no_final": not (clean and final.strip()),
                 "_tokens": head + list(seq.tokens), "_logprobs": list(seq.logprobs) if seq.logprobs is not None else None}
            )
        self.spend.add_tinker(prompt.length, sum(r["completion_tokens"] for r in results))
        return results


def public(row: dict) -> dict:
    """Drops token-level fields before a row is written to disk."""
    return {k: v for k, v in row.items() if not k.startswith("_")}
