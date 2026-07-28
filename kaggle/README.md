# Kaggle Track

The eight notebooks, in order. Run them in sequence — each depends on the one
before.

| # | Notebook | Time | Produces |
|---|---|---|---|
| 00 | `00_setup_and_gpu_check.ipynb` | 20 min | a green environment banner |
| 01 | `01_baseline_inference.ipynb` | 30 min | `baseline_answers.json` |
| 02 | `02_dataset_prep.ipynb` | 25 min | `dataset/formats/*.jsonl` |
| 03 | `03_qlora_finetune.ipynb` | 55 min | `aura-adapter/` (~90 MB) |
| 04 | `04_before_after.ipynb` | 20 min | `tuned_answers.json` + the diff table |
| 05 | `05_save_and_gguf.ipynb` | 20 min | `aura-gguf/*.gguf` (~2 GB) |
| 06 | `06_serve_ollama.ipynb` | 20 min | a running Ollama model |
| 07 | `07_serve_vllm.ipynb` | 25 min | an OpenAI-compatible endpoint |

---

## Notebook settings

For every notebook, in the right-hand panel:

| Setting | Value |
|---|---|
| **Accelerator** | GPU T4 ×2 |
| **Internet** | On |
| **Persistence** | Off |

Both GPU and Internet require **phone verification** on your Kaggle account.

> **Notebook 07 must start in a fresh session.** vLLM installs its own torch
> build and will not coexist with Unsloth. Restart the kernel first.

---

## Secrets

Add-ons → Secrets → new secret:

| Label | Value |
|---|---|
| `HF_TOKEN` | your Hugging Face **read** token |

The label is case-sensitive, and the checkbox must be ticked for each notebook
that needs it.

Never paste a token into a cell — it gets saved with the notebook.

---

## Getting the repo into Kaggle

The bootstrap cell tries three things in order:

1. **Already present** at `/kaggle/working/LLM-lab` or the working directory
2. **Attached as a Kaggle Dataset** — right panel → *Add Data* → upload this repo as a zip
3. **Cloned** from `REPO_URL` — edit that constant at the top of the bootstrap cell

For a workshop, option 2 or 3 is best. Option 3 is one line for participants:
push this repo to GitHub and set `REPO_URL` before distributing.

---

## Disk budget

`/kaggle/working` holds about **20 GB**. Lab 5 is the pinch point:

| Item | Size |
|---|---|
| base model cache | ~2.3 GB |
| LoRA adapter | ~0.1 GB |
| merged 16-bit (temporary) | ~6.4 GB |
| GGUF `q4_k_m` | ~2.0 GB |

Export **one** quantization, and delete `training_output/` if you get tight.

---

## If it goes wrong

See [../docs/troubleshooting.md](../docs/troubleshooting.md).

The two fastest fixes, covering most problems:

**Gated model / 401:**
```python
import os; os.environ["LLMLAB_UNGATED"] = "1"   # before importing config
```

**Out of memory:** in `common/config.py` set `BATCH_SIZE = 1`, `GRAD_ACCUM = 8`,
`MAX_SEQ_LENGTH = 1024`.

---

## Editing the notebooks

These files are **generated**. Edit `tools/build_notebooks.py` and re-run:

```bash
python tools/build_notebooks.py
```

Direct edits to the `.ipynb` files are lost on the next build. (Editing inside
Kaggle during the workshop is fine — just don't expect it to persist back here.)
