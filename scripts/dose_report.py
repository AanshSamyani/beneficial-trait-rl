"""Trait dose at step 100: normal RL (0%), 5%, 10% and 25% trait prompts, all compute-matched (32 prompts x 16 samples x 100 steps).
Each dose arm is compared with normal RL by a paired bootstrap over prompts, with Benjamini-Hochberg correction across the 14
alignment and benefit measures within each dose (blackmail is a subset of agentic misalignment and is left out), as in
stage6_report. Writes results/dose_report.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage6_report import EVALS, bh, compare, load, summary

ROOT = Path(__file__).resolve().parents[1]
ARMS = [("normal", "0% (normal RL)"), ("dose05", "5%"), ("dose10", "10%"), ("trait", "25%")]


def main():
    out = {"evals": {}, "trait_curve": {}}
    for name, file, keep, kind in EVALS:
        rows = {a: load(a, 100, file, keep) for a, _ in ARMS}
        if not rows["normal"]:
            continue
        rec = {"kind": kind, "means": {a: (summary(r)["mean"] if r else None) for a, r in rows.items()},
               "sem": {a: (summary(r)["sem"] if r else None) for a, r in rows.items()}, "vs_normal": {}}
        for a, _ in ARMS[1:]:
            if rows[a]:
                rec["vs_normal"][a] = compare(rows[a], rows["normal"])
        out["evals"][name] = rec
    for a, _ in ARMS[1:]:
        ps = {n: r["vs_normal"][a]["p"] for n, r in out["evals"].items() if r["kind"] != "capability" and n != "Blackmail" and a in r["vs_normal"]}
        for n, q in bh(ps).items():
            out["evals"][n]["vs_normal"][a]["q"] = q
    for a, _ in ARMS:
        out["trait_curve"][a] = [{"step": s, **summary(r)} for s in (0, 20, 50, 80, 100) if (r := load(a, s, "benchmark_heldout", None))]

    print(f"{'eval':30} " + " ".join(f"{lbl:>8}" for _, lbl in ARMS) + " |   5% - normal [CI] q        |  10% - normal [CI] q")
    for n, r in out["evals"].items():
        m = " ".join(f"{r['means'][a]:8.3f}" if r["means"][a] is not None else "     -  " for a, _ in ARMS)
        cells = []
        for a in ("dose05", "dose10"):
            d = r["vs_normal"].get(a)
            cells.append(f"{d['points']:+6.1f} [{d['ci95'][0]:+.1f},{d['ci95'][1]:+.1f}] q={d.get('q', float('nan')):.3f}" if d else "-")
        print(f"{n[:30]:30} {m} | {cells[0]:28} | {cells[1]}")
    (ROOT / "results" / "dose_report.json").write_text(json.dumps(out, indent=1))


main()
