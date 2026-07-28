"""Convert the raw AURA conversations into the four dataset formats — Lab 2.

The same 710 conversations, written four ways. Participants run this and then
diff the outputs, which is a far better explanation of "what is ShareGPT" than
a slide can manage.

    python dataset/build_dataset.py           # write all formats
    python dataset/build_dataset.py --show 2  # print 2 examples of each, write nothing

Formats produced:

  raw       plain text completion, no structure at all
  alpaca    instruction / input / output  — the 2023 default, single-turn only
  sharegpt  conversations[{from, value}]  — multi-turn, 'human'/'gpt' roles
  chatml    messages[{role, content}]     — multi-turn, what everything uses now

Only ShareGPT and ChatML survive multi-turn intact. Alpaca has nowhere to put
turn three, which is precisely why the field moved on from it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAW_PATH = HERE / "raw" / "aura_support_raw.jsonl"
FORMATS_DIR = HERE / "formats"

sys.path.insert(0, str(HERE.parent / "common"))
from eval_prompts import EVAL_PROMPTS  # noqa: E402

SYSTEM_PROMPT = (
    "You are the AURA cluster support assistant. You help researchers use the "
    "AURA HPC cluster. Answer concisely and accurately using AURA's actual "
    "partitions, module names, and paths. If you do not know, say so."
)


def load_raw(path: Path = RAW_PATH) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found — run `python dataset/generate_raw.py` first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --------------------------------------------------------------------------
# Format converters
# --------------------------------------------------------------------------

def to_raw_text(row: dict) -> dict:
    """Plain completion text. No roles, no structure — the model just
    continues the string. Simple, and almost never what you want for a chat
    model, because nothing marks where the answer should stop."""
    parts = []
    for m in row["messages"]:
        speaker = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{speaker}: {m['content']}")
    return {"text": "\n\n".join(parts)}


def to_alpaca(row: dict) -> dict | None:
    """instruction / input / output.

    Returns None for multi-turn conversations: Alpaca has exactly one output
    field, so a three-turn exchange cannot be represented. This is the format's
    real limitation, not an implementation shortcut — worth pointing out in the
    lab, since ~2% of our rows silently vanish.
    """
    msgs = row["messages"]
    if len(msgs) != 2:
        return None
    return {
        "instruction": msgs[0]["content"],
        "input": "",
        "output": msgs[1]["content"],
    }


def to_sharegpt(row: dict) -> dict:
    """conversations[{from, value}] with 'human'/'gpt' role names.

    Multi-turn safe. The odd role naming is a historical artifact of the
    ShareGPT scrape, and Unsloth's `standardize_sharegpt` exists to convert it
    into the messages format below.
    """
    return {
        "conversations": [
            {"from": "human" if m["role"] == "user" else "gpt",
             "value": m["content"]}
            for m in row["messages"]
        ]
    }


def to_chatml(row: dict, system: str | None = SYSTEM_PROMPT) -> dict:
    """messages[{role, content}] — the OpenAI-style format everything now uses.

    A system message is prepended here. That system prompt must also be present
    at inference time, or train and test disagree and quality drops for reasons
    that are very hard to spot.
    """
    messages = [{"role": "system", "content": system}] if system else []
    messages += [{"role": m["role"], "content": m["content"]} for m in row["messages"]]
    return {"messages": messages}


CONVERTERS = {
    "raw": to_raw_text,
    "alpaca": to_alpaca,
    "sharegpt": to_sharegpt,
    "chatml": to_chatml,
}


# --------------------------------------------------------------------------
# Contamination control
# --------------------------------------------------------------------------

def drop_eval_questions(rows: list[dict]) -> tuple[list[dict], int]:
    """Remove rows whose first user turn is verbatim one of the eval questions.

    The generator naturally produces some of the twelve evaluation questions
    word for word, because they are the obvious way to ask. Leaving them in
    would make Lab 4 a memorisation test: the model would recite an answer it
    saw during training, and the improvement would prove nothing.

    Paraphrases of the same facts stay, so the model still learns everything it
    needs. It just has to generalise to the exact wording at eval time — which
    is the honest version of the demo, and worth saying out loud in the lab.
    """
    banned = {p["question"].strip().lower() for p in EVAL_PROMPTS}
    kept = [r for r in rows
            if r["messages"][0]["content"].strip().lower() not in banned]
    return kept, len(rows) - len(kept)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def split(rows: list[dict], val_fraction: float = 0.05) -> tuple[list, list]:
    """Hold out a small validation set.

    The rows were already shuffled by generate_raw.py with a fixed seed, so
    slicing off the tail is a stable split without reshuffling.
    """
    n_val = max(1, int(len(rows) * val_fraction))
    return rows[:-n_val], rows[-n_val:]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def build_all(rows: list[dict], out_dir: Path = FORMATS_DIR) -> dict[str, int]:
    rows, removed = drop_eval_questions(rows)
    train, val = split(rows)
    counts: dict[str, int] = {"_contaminated_removed": removed}

    for name, fn in CONVERTERS.items():
        for split_name, subset in (("train", train), ("val", val)):
            records = [r for r in (fn(row) for row in subset) if r is not None]
            path = out_dir / f"{name}_{split_name}.jsonl"
            write_jsonl(records, path)
            counts[f"{name}_{split_name}"] = len(records)

    return counts


def show_examples(rows: list[dict], n: int) -> None:
    # Pick n single-turn and, if available, one multi-turn to expose the
    # Alpaca limitation rather than describing it.
    single = [r for r in rows if len(r["messages"]) == 2][:n]
    multi = [r for r in rows if len(r["messages"]) > 2][:1]

    for row in single + multi:
        turns = len(row["messages"]) // 2
        print("=" * 74)
        print(f"  topic: {row['topic']}   turns: {turns}")
        print("=" * 74)
        for name, fn in CONVERTERS.items():
            out = fn(row)
            print(f"\n--- {name} ---")
            if out is None:
                print("  (dropped: Alpaca cannot represent a multi-turn conversation)")
            else:
                print(json.dumps(out, indent=2, ensure_ascii=False)[:900])
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--show", type=int, metavar="N",
                    help="print N examples in every format and exit")
    ap.add_argument("--raw", type=Path, default=RAW_PATH)
    ap.add_argument("--out", type=Path, default=FORMATS_DIR)
    args = ap.parse_args()

    rows = load_raw(args.raw)
    print(f"loaded {len(rows)} raw conversations")

    if args.show:
        show_examples(rows, args.show)
        return

    counts = build_all(rows, args.out)
    removed = counts.pop("_contaminated_removed")

    print(f"\nwrote to {args.out}/\n")
    for key in sorted(counts):
        print(f"  {key + '.jsonl':<24} {counts[key]:>5} records")

    dropped = counts["chatml_train"] - counts["alpaca_train"]
    if dropped:
        print(f"\n  note: Alpaca dropped {dropped} multi-turn conversations "
              f"it cannot represent.")
    if removed:
        print(f"  note: held out {removed} rows matching an eval question "
              f"verbatim, so Lab 4 tests generalisation, not recall.")

    print("\n  Training uses chatml_train.jsonl.")


if __name__ == "__main__":
    main()
