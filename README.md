# LLM Fine-Tuning & Inferencing Workshop

A half-day workshop: ~2 hours of theory (slides), then ~4 hours of hands-on
practical. This repo is the practical half.

Participants fine-tune **Llama-3.2-3B** with **QLoRA** on a free Kaggle GPU,
watch it go from confidently inventing answers to answering correctly, then
export and serve it two different ways.

**No local GPU needed.** Everything runs in a browser.

---

## What participants build

A support assistant for **AURA** — a fictional HPC cluster.

Fictional on purpose. No pretraining corpus contains AURA's partition names,
quotas or wrapper commands, so the base model *cannot* know them. Asked twelve
questions about it in Lab 1, it invents fluent, plausible, completely wrong
answers. After ~20 minutes of training on a free T4, it answers correctly.

The twelve evaluation questions are **held out of the training data** — the
model sees paraphrases of the same facts, never those exact strings. So the
improvement is generalisation, not recall.

---

## Start here

| You are | Read |
|---|---|
| **A participant, before the day** | [docs/pre-workshop-setup.md](docs/pre-workshop-setup.md) |
| **A participant, on the day** | [kaggle/README.md](kaggle/README.md) → notebook 00 |
| **Running the workshop** | [docs/instructor-notes.md](docs/instructor-notes.md) |
| **Planning the agenda** | [docs/00-workshop-index.md](docs/00-workshop-index.md) |
| **Stuck** | [docs/troubleshooting.md](docs/troubleshooting.md) |

---

## The practical, in eight notebooks

| Lab | What happens | Time |
|---|---|---|
| **0** · Environment | GPU check, install, why a T4 can't do bf16 | 20 min |
| **1** · Baseline inference | Watch the model hallucinate. Every sampling parameter, hands-on | 30 min |
| **2** · Dataset prep | Raw / Alpaca / ShareGPT / ChatML, and the chat template | 25 min |
| **3** · QLoRA fine-tune ⭐ | Train 0.7% of a 3B model. Measure training vs inference VRAM | 55 min |
| **4** · Before vs after | The same twelve questions, side by side | 20 min |
| **5** · Save & GGUF | Adapter vs merged vs quantized. Which format for which runtime | 20 min |
| **6** · Serve with Ollama | GGUF → Modelfile → chat → OpenAI-compatible API | 20 min |
| **7** · Serve with vLLM | Serve the adapter unmerged. Continuous batching throughput | 25 min |

Every lab maps back to a slide from the theory deck — the index marks which.

---

## Repository layout

```
├── docs/                    workshop index, setup, troubleshooting, cheat sheets
├── common/                  platform-agnostic shared code
│   ├── config.py            model, paths, LoRA + training defaults
│   ├── gpu_check.py         capability detection -> dtype
│   ├── eval_prompts.py      the 12 fixed evaluation questions
│   └── compare.py           the before/after diff table
├── dataset/
│   ├── aura_spec.py         the fictional cluster's facts — the only file with facts in it
│   ├── generate_raw.py      facts -> 710 conversations
│   ├── build_dataset.py     raw -> alpaca / sharegpt / chatml
│   └── formats/             generated training data
├── kaggle/                  the eight notebooks
└── tools/build_notebooks.py notebooks are generated from here
```

---

## Regenerating things

```bash
python dataset/generate_raw.py      # facts   -> raw conversations
python dataset/build_dataset.py     # raw     -> four formats
python tools/build_notebooks.py     # source  -> kaggle/*.ipynb
```

`.ipynb` files are **generated**. Edit `tools/build_notebooks.py`, not the
notebooks, or your changes disappear on the next build.

---

## Using your own data

Swap the dataset and nothing else changes:

1. Replace the facts in `dataset/aura_spec.py`, or write your own
   `dataset/raw/*.jsonl` directly (one JSON object per line, with a `messages`
   list).
2. Update the twelve questions in `common/eval_prompts.py` — including
   `expected_contains`, which is what the scoring uses.
3. Run `python dataset/build_dataset.py`.
4. Re-run notebooks 02 → 06.

The model, training config and serving setup are untouched.

---

## Configuration

Everything tunable lives in [`common/config.py`](common/config.py).

The two most useful switches:

```python
USE_UNGATED_MODEL = True     # or LLMLAB_UNGATED=1
# swaps Llama-3.2-3B for Qwen2.5-3B — no licence, no HF token, same labs

MAX_STEPS = 60
# smoke-test training in ~4 minutes instead of ~20
```

Paths come from `LLMLAB_WORK_DIR`, defaulting to `/kaggle/working`. Set it and
every path follows — nothing else is hardcoded.

---

## Current scope

**Kaggle only, deliberately.** Prove the practical works on one platform before
replicating it.

The repo is structured so the other two tracks are a thin addition rather than a
rewrite: all real logic sits in `common/`, paths come from config, dtype is
derived from the GPU, and Kaggle-specific workarounds are tagged `# KAGGLE:`.

Planned next:
- `local/` — own GPU, single or multi
- `cluster/` — Slurm + Apptainer, with sbatch scripts and a container definition

---

## Requirements

- A Kaggle account with **phone verification** (needed for GPU *and* internet)
- A Hugging Face account and a read token
- Llama 3.2 licence accepted — or use the ungated fallback

Kaggle provides ~30 GPU hours per week free. The full practical uses about two.
