# Instructor Notes

For whoever is running the room. Read before the dry run; update it *during*
the dry run.

---

## The one thing that decides whether the day works

**Pre-workshop setup completion.** Everything else is recoverable; thirty people
doing account admin at 09:05 is not.

Send [pre-workshop-setup.md](pre-workshop-setup.md) **a week ahead**, and chase
non-responders **two days ahead**. The form asking for the environment banner
isn't bureaucracy — it's how you find out that four people never got GPU access
while you can still do something about it.

If someone arrives unset up, sit them next to someone who is. Pairing beats
troubleshooting during Lab 0.

---

## Timings — treat as unverified

Every number in the index is an estimate until you do a full dry run. **Record
real numbers here as you go:**

| Lab | Estimated | Your dry run | On the day |
|---|---|---|---|
| 0 · Environment | 20 min | | |
| 1 · Baseline inference | 30 min | | |
| 2 · Dataset prep | 25 min | | |
| 3 · QLoRA fine-tune | 55 min | | |
| 4 · Before/after | 20 min | | |
| 5 · Save & GGUF | 20 min | | |
| 6 · Ollama | 20 min | | |
| 7 · vLLM | 25 min | | |
| Wrap-up | 15 min | | |

The two that will surprise you: **install time in Lab 0** (varies with Kaggle
load) and **GGUF conversion in Lab 5** (disk-bound, can be slow).

---

## Where to slow down

**Lab 1, section 2.6 — the baseline eval.** Don't rush past it. Have someone
read a hallucinated answer out loud. It's fluent, well-formatted, cites plausible
partition names, and is entirely invented. That moment is what makes the rest of
the day land, and it's the single most transferable lesson in the workshop:
*confidence is not evidence*.

**Lab 3, section 4.3 — the 0.7% number.** Pause. Ask the room what they expected
before showing it. The gap between guess and reality is the lesson.

**Lab 4 — all of it.** This is the payoff. Do not let it get compressed because
Lab 3 overran. If you're behind, cut Lab 7 instead.

## Where to speed up

- **Lab 2's format tour** — if the room is following easily, run `--show 1` and talk over the output rather than having everyone run it.
- **Lab 5's format table** — read it aloud, don't work through it.
- **Lab 1's parameter playground** — if you're behind, demo temperature and top-p, and let them explore penalties in their own time.

---

## Questions you will get, with answers

**"Why is it lying so confidently?"**
It isn't lying — it has no mechanism for knowing that it doesn't know. It's
producing the most probable continuation, and a fluent-sounding wrong answer is
more probable than "I don't know" unless it was trained toward the latter. Our
dataset deliberately includes refusals, which is why the fine-tuned model *does*
decline some questions.

**"Could we just put the AURA docs in the prompt instead?"**
Yes, and often you should — that's RAG, and for facts that change it's the better
tool. Fine-tuning wins on format, voice, consistency, and cost per query, because
you don't pay for those tokens every request. In production you'd use both.
Section 5.7 covers this.

**"Why 0.7% and not some other number?"**
It's a consequence of `r`, how many matrices you adapt, and model size. Section
4.4 lets them move `r` and watch it change. The important idea is that it's
*small*, not that it's exactly 0.7%.

**"Will this work on a 70B model?"**
The method, yes. The hardware, no — not on a T4. QLoRA on 70B needs roughly
48 GB. Good bridge to the cluster track.

**"How much data do I need?"**
Fewer examples than people expect for *behaviour* (hundreds), far more than they
expect for *knowledge* (and even then it's the wrong tool — use RAG). Our 665
rows teach a voice and a bounded fact set. Be honest that this is a demo scale.

**"Is the model actually learning, or memorising?"**
Good question, and we designed for it: the twelve eval questions are removed
from training data. Only paraphrases remain. Say this out loud at Lab 4 — it
pre-empts the sceptic in the room and makes the result more impressive, not less.

**"Why does Kaggle give two GPUs if we only use one?"**
Unsloth's open-source version trains on one. The second is useful for running a
server alongside a loaded model. Multi-GPU training is a different lesson.

---

## Failure modes, ranked by likelihood

| Risk | Likelihood | Blast radius | Mitigation |
|---|---|---|---|
| Someone's Llama licence not approved | **high** | one person | `USE_UNGATED_MODEL = True` — 10 seconds |
| pip install breaks on Kaggle image change | medium | **everyone** | **Re-run Lab 0 the day before.** Pinned fallbacks in troubleshooting.md |
| vLLM won't start on T4 | medium | Lab 7 only | Lab 6 already served a model; convert Lab 7 to a demo |
| Kaggle GPU quota exhausted | medium | one person | Pair them up; two people, one notebook |
| Disk full in Lab 5 | medium | one person | Delete `training_output`, export one quant |
| Session dies mid-training | low | one person | `MAX_STEPS = 60` restarts in ~3 min |
| Kaggle-wide outage | low | **everyone** | No mitigation. Have the theory deck ready to extend. |

**Re-run Lab 0 end to end the day before the workshop.** Kaggle updates its base
image without notice and it is the single most likely thing to break between
your dry run and the day.

---

## The emergency fast path

If you are 40 minutes behind after Lab 2:

1. In `common/config.py`, set `MAX_STEPS = 60`. Training drops from ~20 min to ~4.
2. The result is measurably worse but the before/after is still clearly visible.
3. Skip Lab 7 entirely.
4. Demo Lab 5's GGUF export from your own pre-run notebook rather than having everyone do it.

This gets you to the payoff. **Never skip Lab 4.**

---

## Pre-run everything yourself

Have a completed set of notebooks with outputs saved, from your own dry run.
When someone's session dies at Lab 4, they can read your outputs and stay with
the room instead of watching a progress bar. Worth the ten minutes it costs to
prepare.

---

## Things this workshop deliberately doesn't cover

Know these so you can answer "why didn't we…" crisply:

- **RLHF / DPO / preference tuning** — a workshop of its own; SFT is the foundation
- **Evaluation harnesses** — keyword scoring is honest about being a demo; real eval is a big topic
- **Multi-GPU / multi-node** — belongs in the cluster track
- **MoE fine-tuning** — different memory profile, different routing concerns
- **Serving at scale** — autoscaling, quantized KV cache, speculative decoding
- **Data quality and dedup** — arguably the highest-leverage topic in fine-tuning, and it needs its own session

---

## Porting to local GPU and cluster

When you're happy with the Kaggle version, the differences to handle are:

- `# KAGGLE:` comments in `common/config.py` mark every platform-specific line
- `LLMLAB_WORK_DIR` env var moves all paths — nothing else is hardcoded
- `gpu_check.py` already derives dtype, so an A100 picks bf16 with no edits
- Ollama needs no root anywhere: on a cluster, extract the tarball into scratch or bake it into the container image
- vLLM on A100/H100 is far less fragile than on a T4 — Lab 7 stops being the risky one

**Log anything you hit here as you go:**

<!-- Add findings below during the dry run -->

-
-
-
