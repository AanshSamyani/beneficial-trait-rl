"""Builds Agentic Misalignment prompts from the repo's template files, reading them as data. The repo's Python is parsed statically and never executed."""
import ast
import json
import re
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from btrl.common import RAW

REPO = RAW / "agentic_repo" / "templates"
OUT = RAW / "agentic_prompts"


def static_constants(path: Path) -> dict:
    """Evaluates string and dict constants without running the file: literals, names, concatenation and f-strings only."""
    env: dict = {}

    def value(node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            return env[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return value(node.left) + value(node.right)
        if isinstance(node, ast.JoinedStr):
            return "".join(str(value(v.value)) if isinstance(v, ast.FormattedValue) else v.value for v in node.values)
        if isinstance(node, ast.Dict):
            return {value(k): value(v) for k, v in zip(node.keys, node.values)}
        raise ValueError(type(node).__name__)

    for n in ast.parse(path.read_text()).body:
        if not isinstance(n, ast.Assign):
            continue
        target = n.targets[0]
        if isinstance(target, ast.Name):
            try:
                env[target.id] = value(n.value)
            except (ValueError, KeyError):
                pass
        elif isinstance(target, ast.Tuple) and isinstance(n.value, ast.ListComp):
            # NAME_A, NAME_B = [expr for x, y in [(..), (..)]]: bind each tuple in turn and evaluate the expression.
            gen = n.value.generators[0]
            names = [t.id for t in gen.target.elts]
            results = []
            for item in gen.iter.elts:
                saved = dict(env)
                env.update({k: value(v) for k, v in zip(names, item.elts)})
                results.append(value(n.value.elt))
                env.clear()
                env.update(saved)
            for t, r in zip(target.elts, results):
                env[t.id] = r
    return env


def markdown_variables(path: Path) -> dict:
    found = re.findall(r"## (.*?)\n\s*```json\s*(.*?)\s*```", path.read_text(), re.S)
    return {name.strip().lower().replace(" ", "_").replace("-", "_"): json.loads(body) for name, body in found}


def variables_for(scenario: str, goal_type: str, goal_value: str, goals: dict) -> dict:
    if goal_type == "none":
        scen = "blackmail" if scenario == "murder" else scenario
        found = markdown_variables(REPO / scen / "variables" / "ambiguous.md")
        out = dict(next(iter(found.values()))) if len(found) == 1 else dict(found)
    else:
        found = markdown_variables(REPO / scenario / "variables" / "america_vs_global.md")
        out = dict(next(v for k, v in found.items() if goal_value in k))
    out.update(goals.get(f"{goal_type}-{goal_value}", goals.get("none", {})))
    out["extra_system_instructions"] = ""
    out["alignment_faking_warning"] = ""
    return out


def main():
    constants = static_constants(REPO / "system_prompt_templates.py")
    system = {"blackmail": constants["BLACKMAIL_SYSTEM_PROMPT"], "leaking": constants["LEAKING_SYSTEM_PROMPT"], "murder": constants["MURDER_SYSTEM_PROMPT"]}
    goals, user = constants["GOAL_VARIABLES"], constants["USER_PROMPT"]
    for scenario in system:
        for goal_type, goal_value in (("explicit", "america"), ("none", "none")):
            v = variables_for(scenario, goal_type, goal_value, goals)
            t = REPO / scenario / "templates"
            if goal_type == "none":
                emails = Template((t / f"email-template-{scenario}-goal-conflict-none-replacement.md").read_text()).safe_substitute(v)
            else:
                emails = Template((t / f"email-template-{scenario}-shared.md").read_text()).safe_substitute(v) + Template((t / f"email-template-{scenario}-replacement.md").read_text()).safe_substitute(v)
            if scenario == "leaking":
                parts = emails.split("<email>")
                emails = parts[0] + "".join(f"<email>\nID: {i}{p}" for i, p in enumerate(parts[1:], 1))
            text = Template(system[scenario]).safe_substitute(v)
            left = re.findall(r"\$\{\w+\}", text + emails)
            assert not left, (scenario, goal_type, left[:5])
            d = OUT / f"{scenario}_{goal_type}-{goal_value}_replacement"
            d.mkdir(parents=True, exist_ok=True)
            (d / "system_prompt.txt").write_text(text)
            (d / "user_prompt.txt").write_text(user)
            (d / "email_content.txt").write_text(emails)
            print(d.name, "| system chars", len(text), "| emails", emails.count("<email>"), "| email chars", len(emails), "| tool format present", "tool_use:" in text, flush=True)


main()
