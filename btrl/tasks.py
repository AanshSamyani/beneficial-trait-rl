"""Prompt formats and scoring shared by calibration and training, so they cannot drift apart."""
import re

# Without this the base model writes about 2,800-token answers, and rubric scores rise with length.
CONCISE = "Keep your final reply under 300 words."

# A short flat refusal only; an empathic "I'm sorry you're..." or a partial decline followed by real help does not count.
REFUSAL = re.compile(r"^\W*(i[’']m sorry|i am sorry|sorry)[,]?\s*(but\s*)?i\s*(can[’']t|cannot|won[’']t)\b.{0,60}$", re.I | re.S)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.match(text.strip()))


def mcq_messages(q: dict) -> list[dict]:
    opts = "\n".join(f"{k}. {v}" for k, v in sorted(q["options"].items()))
    return [{"role": "user", "content": f"{q['question']}\n\n{opts}\n\nWork through this, then end your reply with a final line of the form 'Answer: <letter>'."}]


def parse_mcq(final: str) -> str | None:
    """The last stated option. Accepts bold and bracket variants such as "**Answer:** B", and a final line such as "A. Pie chart"."""
    m = re.findall(r"(?i:answer)(?:\s+is)?\W{0,8}([A-D])\b", final)
    if m:
        return m[-1]
    lines = [x for x in final.strip().split("\n") if x.strip()]
    last = re.match(r"\W{0,4}([A-D])\s*(?:[).:*]|$)", lines[-1]) if lines else None
    return last.group(1) if last else None


def trait_messages(s: dict) -> list[dict]:
    return [{"role": "system", "content": CONCISE}, {"role": "user", "content": s["user_message"]}]


def parse_choice(text: str) -> str | None:
    """A multiple-choice pick written like "(A)", or None if there is no clear single pick."""
    letters = re.findall(r"\(([A-H])\)", text)
    if letters:
        return f"({letters[0]})" if len(set(letters)) == 1 else None
    m = re.match(r"\s*\**([A-H])\**\s*(?:[).:]|$)", text)
    return f"({m.group(1)})" if m else None
