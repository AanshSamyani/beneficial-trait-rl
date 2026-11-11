# Beneficial Trait RL: an open replication

Code, data and evaluation pipeline for our replication of [Reinforcement learning towards broadly and persistently beneficial models](https://alignment.openai.com/beneficial-rl/) ([Jagadeesh et al., 2026](https://arxiv.org/abs/2606.24014)) on open models, by Aansh Samyani and Jozdien.

- Blog post: _LessWrong link to be added_
- Model checkpoints (gpt-oss-20b LoRA adapters): [huggingface.co/Aansh123/beneficial-trait-rl](https://huggingface.co/Aansh123/beneficial-trait-rl)

We train gpt-oss-20b with RL on [Tinker](https://thinkingmachines.ai/tinker/) and compare **beneficial trait RL** (25% of each batch is health conversations rewarded for beneficial traits such as truthfulness, corrigibility and metacognitive transparency) against a compute-matched **normal RL** baseline (medical and math tasks only). Data generation and grading use the OpenAI API.

## Results you should be able to reproduce

All numbers are trait RL minus normal RL at step 100 on gpt-oss-20b, in points (95% bootstrap CI), from `results/reference/`.

| Result | What to run | Our numbers |
|---|---|---|
| 1. Beneficial trait evals | [Main runs](#2-main-comparison-results-1-2-and-5) | Held-out trait benchmark +21.5 [17.2, 25.8]; HealthBench +13.1 [7.1, 19.0] |
| 2. Alignment evals | [Main runs](#2-main-comparison-results-1-2-and-5) | MASK +26.8 [14.6, 39.0]; DeceptionBench +15.0 [9.4, 21.2]; sycophancy +10.1 [4.3, 16.3] |
| 3. Amount of trait data | [Dose runs](#3-amount-of-trait-data-result-3) | Trait benchmark +4.5 [1.5, 7.5] at 5% and +4.3 at 10%; no significant alignment-eval gains |
| 4. Adversarial persona prompts | [Persona test](#4-adversarial-persona-prompts-result-4) | Under the two harmful personas, the baseline's health scores fell 0.23 and the trait model's 0.12 (difference 0.108 [0.061, 0.158]) |
| 5. Capabilities | [Main runs](#2-main-comparison-results-1-2-and-5) | No significant capability loss on MedMCQA, MedQA, MedCalc-Bench, MATH, GPQA Diamond, HMMT or IFEval |

Every result is from one seed per arm, so expect differences of a few points on small evals if you rerun.

## Setup

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/), a Tinker API key and an OpenAI API key with access to `gpt-5.6-luna` (judge) and `gpt-5.6-terra` (data generation; only needed to regenerate the data).

```bash
uv sync
cp .env.example .env        # then fill in OPENAI_API_KEY and TINKER_API_KEY
uv python install 3.12      # the code sandbox below runs model-written code with a uv-managed Python 3.12
```

Run every script from the repository root, for example `uv run python scripts/mix_train.py`. Each script keeps a spend ledger under `results/` and stops itself at a budget cap.

**Sandbox for the impossible-test coding eval.** This eval runs model-written code. `btrl/sandbox.py` runs it under macOS `sandbox-exec` with no network and no file writes. On Linux, replace `run_tests` in that file with a container-based runner, or leave the `impossible` eval out of `scripts/stage6.py`.

## 1. Data

### Included in this repository

| Path | Contents | Made by |
|---|---|---|
| `data/trait/health_train.jsonl` | 800 synthetic health conversations covering the paper's 15 traits, each with 4 trait-specific reward criteria | `scripts/trait_generate.py` (gpt-5.6-terra) |
| `data/trait/tracking.jsonl` | Held-out conversations tracked during training | `scripts/trait_generate.py` |
| `data/benchmark/eval_scenarios.jsonl` | The held-out trait benchmark: 154 conversations (22 for each of 7 traits) across 12 subject areas, plus calibration items listed in `calibration_ids.json` | `scripts/bench_generate.py` |
| `data/mix/` | Fixed train and held-out splits of MedMCQA, MedQA, MedCalc-Bench and MATH for the RL tasks | `scripts/mix_prepare.py` |
| `data/calibration/impossible_problems.jsonl` | Our synthetic impossible-test coding eval (after Guan et al., 2025) | |
| `data/anti/` | The same 800 conversations with criteria that reward the opposite of each trait (control experiment) | `scripts/anti_generate.py` |
| `data/prior/`, `data/prior100/` | Further beneficial and opposite-trait conversations, used by the persistence experiment | `scripts/prior_generate.py`, `scripts/prior100_generate.py` |

The capability splits come from [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa), [MedQA](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options), [MedCalc-Bench](https://huggingface.co/datasets/ncbi/MedCalc-Bench-v1.2) and [MATH](https://github.com/hendrycks/math); see each dataset's license.

### Public eval datasets (download yourself)

These are third-party datasets, so they are downloaded rather than redistributed. Some Hugging Face datasets are gated: accept their terms on the website and run `hf auth login` first.

```bash
mkdir -p data/raw
uv run python scripts/prepare_data.py       # MedQA, MedMCQA, GSM8K, MASK, sycophancy (Perez et al.)
uv run python scripts/stage3_prepare.py     # HealthBench, School of Reward Hacks, GPQA Diamond, IFEval, Betley et al.
curl -L -o data/raw/strongreject_small.csv \
  https://raw.githubusercontent.com/alexandrasouly/strongreject/main/strongreject_dataset/strongreject_small_dataset.csv
uv run python scripts/stage6_prepare.py     # HealthBench Professional, DeceptionBench, MoReBench, StrongREJECT, AbstentionBench (GPQA), HMMT
git clone https://github.com/anthropic-experimental/agentic-misalignment data/raw/agentic_repo
git -C data/raw/agentic_repo checkout ea0630e1a3eaae7f9f9740fd2703229d3854ccda
uv run python scripts/agentic_prompts.py    # builds the Lynch et al. prompts from the repo's templates (its code is never executed)
# full public HealthBench file, used by the persona test
uv run python -c "import os, shutil; from huggingface_hub import hf_hub_download as d; os.makedirs('data/raw/healthbench_full', exist_ok=True); shutil.copy(d('openai/healthbench', '2025-05-07-06-14-12_oss_eval.jsonl', repo_type='dataset'), 'data/raw/healthbench_full/oss_eval.jsonl')"
```

All subsets are seeded, so you get the same items we used. Please respect the datasets' terms: HealthBench and MASK carry no-train canaries, and the HealthBench Professional authors ask that its examples not be posted publicly. Report those evals in aggregate only.

### Regenerating the training data (optional)

```bash
uv run python scripts/trait_generate.py     # 800 trait conversations (gpt-5.6-terra)
uv run python scripts/bench_generate.py     # held-out trait benchmark
uv run python scripts/mix_prepare.py        # capability splits
uv run python scripts/anti_generate.py      # opposite-trait criteria for the control
```

## 2. Main comparison (results 1, 2 and 5)

```bash
uv run python scripts/mix_train.py                    # normal RL, 100 steps  -> results/mix_rl/run
uv run python scripts/trait_train.py --arm trait      # beneficial trait RL (25% trait data), 100 steps -> results/trait_rl/run
```

Both arms use LoRA rank 32, learning rate 4e-5 and 32 prompts with 16 samples each per step, for 100 steps. Normal RL samples 8 prompts from each of MedMCQA, MedQA, MedCalc-Bench and MATH. Trait RL samples 6 from each plus 8 trait conversations. The trait reward is the share of a conversation's criteria the reply meets, judged by gpt-5.6-luna, times `min(1, 450 / words)`. Groups whose 16 rewards are all equal are dropped. Each run takes a few hours and costs roughly $30-50 of Tinker compute, plus about $9 of judge calls for the trait arm.

Evaluate both arms at steps 0, 20, 50, 80 and 100, then score the held-out RL tasks:

```bash
uv run python scripts/stage6.py --arms normal,trait --steps 20,50,80,100
# optional: also run the end-only evals at the intermediate checkpoints
uv run python scripts/stage6.py --arms normal,trait --steps 20,50,80 --evals gpqa,ifeval,hmmt,morebench,abstention
for step in 020 050 080 100; do
  uv run python scripts/mix_eval.py --label mix_normal_$step --run-dir results/mix_rl/run --checkpoint 000$step
  uv run python scripts/mix_eval.py --label mix_trait_$step --run-dir results/trait_rl/run --checkpoint 000$step
done
uv run python scripts/stage6_report.py      # -> results/stage6/report.json (paired bootstrap CIs, Benjamini-Hochberg q values)
```

Eval replies go to `results/stage3/<label>/<eval>.jsonl` (for example `trait_100/mask.jsonl`). Expect roughly $1-3 of Tinker and $1-3 of OpenAI spend per checkpoint.

## 3. Amount of trait data (result 3)

```bash
uv run python scripts/trait_train.py --arm dose05     # 5% trait prompts
uv run python scripts/trait_train.py --arm dose10     # 10% trait prompts
uv run python scripts/stage6.py --arms dose05,dose10 --steps 20,50,80,100
uv run python scripts/dose_report.py                  # -> results/dose_report.json
```

The dose arms keep 32 prompts per step and reach exactly 5% or 10% trait prompts over the run, using a repeating pattern of trait slots per step (`btrl/trait_env.py`, `DoseMixDataset`).

## 4. Adversarial persona prompts (result 4)

Needs the step-100 checkpoints from section 2 and `data/raw/healthbench_full/oss_eval.jsonl`.

```bash
uv run python scripts/persona_eval.py       # 3 persona prompts x 3 health evals, both models
uv run python scripts/persona_report.py     # -> results/persona/report.json
```

The persona prompts are in `btrl/personas.py`. They are added as a system message, and the grader scores the conversation without the persona. The evals are 100 general HealthBench conversations, 80 HealthBench conversations about mental health and the 60 HealthBench Professional conversations, all graded with the HealthBench rubrics. Costs about $2 of Tinker and $9 of OpenAI spend.

## 5. Figures

```bash
uv run python scripts/paper_figures.py      # -> results/writeup/figures/ (PNG, SVG, PDF)
```

The script draws every figure whose inputs exist and skips the rest. Our versions are in `results/figures/`.

## Further experiments

These are in the code, though not all are in the blog post.

- **Opposite-trait control:** the same 800 conversations with criteria rewarding the opposite traits. Run `trait_train.py --arm anti`, then `stage6.py --arms anti --steps 20,50,80,100`, `anti_report.py` and `anti_figures.py`.
- **Persistence under further RL:** 50 beneficial steps, then 100 opposite-trait steps. Run `trait_train.py --arm bfirst`, `trait_train.py --arm banti`, then `stage6.py --arms bfirst --steps 50`, `stage6.py --arms banti --steps 20,50,80,100`, `bfirst_report.py` and `bfirst_figures.py`.
- **Other models:** pass `--model 120b`, `--model inkling` or `--model nemotron` to `mix_train.py`, `trait_train.py`, `stage6.py`, `mix_eval.py` and `stage6_report.py`. Inkling-2-Small and Nemotron-3.5-Lightning need newer packages than gpt-oss: `tinker>=0.39.0`, `tinker-cookbook>=0.6.1` and `tml-renderers>=0.4.1`. `scripts/headroom_calibrate.py --model <name>` checks how much room a model has on the training rewards before you spend on a full run.

## Using the released checkpoints

[huggingface.co/Aansh123/beneficial-trait-rl](https://huggingface.co/Aansh123/beneficial-trait-rl) holds rank-32 LoRA adapters for `openai/gpt-oss-20b` in standard PEFT format. Normal RL is under `normal_rl/step_XXX/` and beneficial trait RL under `beneficial_trait_rl/step_XXX/`, for steps 020, 050, 080 and 100. For example, with vLLM:

```bash
hf download Aansh123/beneficial-trait-rl --include "beneficial_trait_rl/step_100/*" --local-dir ckpt
vllm serve openai/gpt-oss-20b --enable-lora --max-lora-rank 32 \
  --lora-modules trait_rl=ckpt/beneficial_trait_rl/step_100
```

We trained and evaluated with gpt-oss's Harmony format at medium reasoning effort (`btrl/llm.py`).

## Repository layout

```
btrl/       environments (mix_env, trait_env, anti_env), rubric judge, evals, personas, Tinker and OpenAI clients
scripts/    data preparation, training, evaluation, reports and figures
data/       training data and held-out benchmark (eval datasets go in data/raw/, not included)
results/    reference/ holds our aggregate reports; figures/ our figures; your runs write here too
```
