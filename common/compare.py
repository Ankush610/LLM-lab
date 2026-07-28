"""Before/after comparison — Lab 4.

Renders the baseline and fine-tuned answers side by side and scores each with
a crude keyword check. The scoring is intentionally simple: it exists to make
the improvement countable, not to be a benchmark. The real evaluation is
participants reading the two columns.

Works in a notebook (HTML table) and in a terminal (plain text).
"""

from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

from eval_prompts import EVAL_PROMPTS


def score(answer: str, expected: list[str]) -> bool:
    """True if every expected substring appears, case-insensitively.

    Whitespace is collapsed first so that a model writing "7 days" or
    "7  days" both match an expected "7 day".
    """
    haystack = re.sub(r"\s+", " ", answer or "").lower()
    return all(re.sub(r"\s+", " ", e).lower() in haystack for e in expected)


def score_all(answers: dict[str, str]) -> dict[str, bool]:
    return {
        p["id"]: score(answers.get(p["id"], ""), p["expected_contains"])
        for p in EVAL_PROMPTS
    }


def load(path: str | Path) -> dict[str, str]:
    with open(path) as f:
        return json.load(f)


def save(answers: dict[str, str], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(answers, f, indent=2)
    print(f"saved {len(answers)} answers -> {path}")


def _truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def report_text(before: dict[str, str], after: dict[str, str], width: int = 300) -> None:
    """Plain-text report. Use when HTML isn't available."""
    b_scores, a_scores = score_all(before), score_all(after)

    for i, p in enumerate(EVAL_PROMPTS, 1):
        pid = p["id"]
        print("=" * 76)
        print(f"Q{i}. {p['question']}")
        print("-" * 76)
        print(f"BEFORE {'[PASS]' if b_scores[pid] else '[FAIL]'}")
        print(f"  {_truncate(before.get(pid, '(missing)'), width)}")
        print(f"AFTER  {'[PASS]' if a_scores[pid] else '[FAIL]'}")
        print(f"  {_truncate(after.get(pid, '(missing)'), width)}")

    print("=" * 76)
    _print_totals(b_scores, a_scores)


def _print_totals(b_scores: dict[str, bool], a_scores: dict[str, bool]) -> None:
    total = len(EVAL_PROMPTS)
    b, a = sum(b_scores.values()), sum(a_scores.values())
    print(f"  BEFORE  {b}/{total} correct")
    print(f"  AFTER   {a}/{total} correct")
    delta = a - b
    if delta > 0:
        print(f"  +{delta} questions fixed ({delta / total:.0%} of the eval set)")
    elif delta == 0:
        print("  No change. Train longer, add data, or check the chat template.")
    else:
        print(f"  {delta} — the fine-tune made it worse. Check masking and LR.")


def report_html(before: dict[str, str], after: dict[str, str], width: int = 600):
    """Side-by-side HTML table. Returns an IPython object; display() it."""
    from IPython.display import HTML

    b_scores, a_scores = score_all(before), score_all(after)

    css = """
    <style>
      .aura-cmp { border-collapse: collapse; width: 100%; font-size: 13px;
                  font-family: -apple-system, system-ui, sans-serif; }
      .aura-cmp th, .aura-cmp td { border: 1px solid #d0d7de; padding: 8px 10px;
                                   vertical-align: top; text-align: left; }
      .aura-cmp th { background: #f6f8fa; font-weight: 600; }
      .aura-cmp td.q { font-weight: 600; width: 22%; }
      .aura-cmp td.a { width: 39%; white-space: pre-wrap; }
      .aura-pass { border-left: 4px solid #1a7f37; }
      .aura-fail { border-left: 4px solid #cf222e; }
      .aura-tag { font-size: 11px; font-weight: 700; letter-spacing: .04em; }
      .aura-tag.p { color: #1a7f37; } .aura-tag.f { color: #cf222e; }
      .aura-sum { font-family: ui-monospace, monospace; font-size: 14px;
                  margin: 10px 0 4px; }
    </style>
    """

    rows = []
    for i, p in enumerate(EVAL_PROMPTS, 1):
        pid = p["id"]
        cells = []
        for scores, answers in ((b_scores, before), (a_scores, after)):
            ok = scores[pid]
            tag = ("p", "PASS") if ok else ("f", "FAIL")
            cells.append(
                f'<td class="a {"aura-pass" if ok else "aura-fail"}">'
                f'<span class="aura-tag {tag[0]}">{tag[1]}</span><br>'
                f'{escape(_truncate(answers.get(pid, "(missing)"), width))}</td>'
            )
        rows.append(
            f'<tr><td class="q">{i}. {escape(p["question"])}</td>'
            f'{cells[0]}{cells[1]}</tr>'
        )

    total = len(EVAL_PROMPTS)
    b, a = sum(b_scores.values()), sum(a_scores.values())
    summary = (
        f'<div class="aura-sum">BEFORE {b}/{total} &nbsp;→&nbsp; '
        f'AFTER {a}/{total} &nbsp;&nbsp;({a - b:+d})</div>'
    )

    return HTML(
        css + summary
        + '<table class="aura-cmp"><tr><th>Question</th>'
        '<th>Base model (before)</th><th>Fine-tuned (after)</th></tr>'
        + "".join(rows) + "</table>"
    )


def report(before: dict[str, str], after: dict[str, str]):
    """Auto-pick HTML in a notebook, text elsewhere."""
    try:
        from IPython.display import display
        get_ipython()  # noqa: F821 — only defined inside IPython
        display(report_html(before, after))
    except (ImportError, NameError):
        report_text(before, after)
