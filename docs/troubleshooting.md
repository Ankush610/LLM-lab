# Troubleshooting

Print this double-sided and put a copy on every table. Ordered by how often it
actually happens.

---

## Setup and access

### `401 Client Error` / "gated repo" / "awaiting a review"
Your Llama 3.2 licence hasn't been approved, or the notebook isn't using your token.

**Fix, in order:**
1. Check [the model page](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) says you have access.
2. Confirm the Kaggle Secret is named exactly `HF_TOKEN` and its checkbox is ticked.
3. **Don't wait for approval.** In `common/config.py` set `USE_UNGATED_MODEL = True`, or in a cell before the import:
   ```python
   import os; os.environ["LLMLAB_UNGATED"] = "1"
   ```
   Qwen2.5-3B needs no licence. Every lab works identically.

### `No GPU available` / `torch.cuda.is_available()` is False
Accelerator isn't set, or your Kaggle account isn't phone-verified.
Right panel → **Accelerator → GPU T4 ×2**. Then **Run → Restart & Run All** —
changing the accelerator restarts the session.

### `HF_TOKEN` secret not found
Right panel → **Add-ons → Secrets**. The label is case-sensitive and the
checkbox next to it must be ticked for *this* notebook.

### pip install fails, or `unsloth` won't import
Usually a torch version conflict. Try, in order:

```python
!pip install -q --no-deps unsloth unsloth_zoo
!pip install -q --no-deps trl peft accelerate bitsandbytes
```

If it still fails, pin versions:

```python
!pip install -q --no-deps "unsloth==2025.1.5" "unsloth_zoo==2025.1.4"
!pip install -q --no-deps "trl==0.13.0" "peft==0.14.0" "bitsandbytes==0.45.0"
```

Then **restart the session** (Run → Restart) — a half-installed torch stays
loaded in memory until you do.

> Versions drift. If the pins above are stale, check the
> [Unsloth README](https://github.com/unslothai/unsloth) for the current Kaggle
> install block.

### Repo not found / `ModuleNotFoundError: config`
The bootstrap cell couldn't locate the repo. Either:
- set `REPO_URL` at the top of the cell to your GitHub repo, or
- attach the repo as a Kaggle Dataset (right panel → **Add Data → Upload**), or
- upload the folder to `/kaggle/working/LLM-lab` manually.

---

## Training

### `CUDA out of memory`
Try in this order — first two are nearly always enough:

1. **`config.BATCH_SIZE = 1`** and raise `GRAD_ACCUM` to 8 to keep the effective batch the same.
2. **`config.MAX_SEQ_LENGTH = 1024`.** Activation memory scales with sequence length; this is the biggest single lever.
3. Confirm `use_gradient_checkpointing="unsloth"` is set.
4. Lower `LORA_R` to 8.
5. Restart the session — a previous notebook may still be holding VRAM. Check with `!nvidia-smi`.

```python
import gc, torch
del model, trainer
gc.collect(); torch.cuda.empty_cache()
```

### `loss = 0.0` from step one
The label mask is eating every token, so there is nothing to learn from.

Check the `instruction_part` / `response_part` markers in the
`train_on_responses_only` cell actually appear in your formatted text:

```python
print(repr(ds["train"][0]["text"][:600]))
```

Llama 3 uses `<|start_header_id|>assistant<|end_header_id|>`; Qwen/ChatML uses
`<|im_start|>assistant`. Using the wrong pair masks everything.

### Loss is flat / barely moves
- Learning rate too low — `2e-4` is right for LoRA; `2e-5` is not.
- Check `trainable params > 0` in the Lab 3 cell. If it's zero, `get_peft_model` didn't take.
- Too few steps. With 665 examples and effective batch 8, one epoch is only ~83 steps.

### Loss is `nan`
fp16 overflow, most likely on a T4. Drop the learning rate to `1e-4`, and
confirm `fp16=True, bf16=False` in `SFTConfig` (Lab 3 sets this from the GPU
automatically — don't hardcode `bf16=True` on a T4).

### Training is much slower than ~20 minutes
- Check you're on GPU: `!nvidia-smi` should show a python process using memory.
- `MAX_SEQ_LENGTH` too high — 2048 is plenty for this dataset.
- Kaggle throttles after long sessions. Restart if the session is hours old.

---

## After training

### The fine-tuned model still gets AURA questions wrong
1. **Did training actually run?** Loss should fall from ~2.0 to under ~0.7.
2. **Template mismatch** — the most likely cause. The template used at inference must match training. Both Lab 3 and Lab 4 call `get_chat_template(..., config.CHAT_TEMPLATE)`; don't change one without the other.
3. **System prompt missing at inference.** Training data includes it; leaving it out at eval time is a train/test mismatch.
4. Train longer: `config.NUM_EPOCHS = 3`.

### Output is gibberish or repeats forever
- Wrong chat template (see above).
- Missing stop tokens — check `PARAMETER stop` in the Modelfile, or `eos_token_id` in `generate()`.
- `temperature` too high. Use `0.0` for evaluation so results are reproducible.

### Model won't stop generating
It never emits EOS, or EOS isn't being honoured. Set `max_new_tokens`, pass
`eos_token_id=tokenizer.eos_token_id`, and check the stop parameters in your
serving config.

---

## Disk and sessions

### `No space left on device`
`/kaggle/working` is ~20 GB and Lab 5 is the pinch point.

```python
!du -sh /kaggle/working/* | sort -h
!rm -rf /kaggle/working/training_output      # checkpoints, safe to delete
!rm -rf /kaggle/working/aura-merged-16bit    # only after GGUF export succeeded
```

Export **one** GGUF quantization, not several.

### Session died and I lost everything
Kaggle sessions expire after ~9 hours, or ~20 minutes idle. Before that happens:
- `model.push_to_hub(...)` — the adapter is only ~90 MB
- Download the GGUF from the right-hand output panel
- Anything in `/kaggle/working` when the session ends is gone unless persistence is on

### GPU quota exhausted
Kaggle gives ~30 GPU hours/week, and it resets weekly. Check
**Settings → Accelerator** for your remaining quota. If you're out, the labs run
on CPU up to Lab 3 (slowly), but training needs a GPU.

---

## Serving

### Ollama: `command not found` after installing
The installer put it in `/usr/local/bin`, which may not be on PATH in this cell:

```python
import os
os.environ["PATH"] += ":/usr/local/bin"
```

### Ollama: server won't start / connection refused
No systemd on Kaggle, so it must be launched manually and given time:

```python
!nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
!sleep 10 && tail -20 /kaggle/working/ollama.log
```

### Ollama: model answers, but badly
`TEMPLATE` in the Modelfile doesn't match what you trained with. Compare it
against the output of the Lab 1 chat-template cell — they must agree token for
token.

### vLLM: won't start / OOM on startup
Expected on a 15 GB T4 — vLLM is memory-hungry. Try:

```
--max-model-len 1024
--gpu-memory-utilization 0.75
--enforce-eager                # skips CUDA graph capture, saves memory
```

**Make sure nothing else is loaded.** vLLM cannot start next to an Unsloth model
in the same session — restart the kernel first.

### vLLM: `ValueError: bfloat16 is only supported on GPUs with compute capability of at least 8.0`
The T4 message, exactly as Lab 0 predicted. Add `--dtype half`.

### vLLM: takes forever to start
Normal — 3–6 minutes to load weights and profile the KV cache. Watch progress:

```python
!tail -f /kaggle/working/vllm.log
```

### vLLM installation breaks everything else
vLLM installs its own torch. **Always run Lab 7 in a fresh session.** If you
installed it alongside Unsloth, restart and reinstall — there's no clean way
back within a session.

---

## Still stuck?

Ask an instructor with:
1. The **full** error message, not a summary
2. Which notebook and which cell
3. The output of the Lab 0 environment banner

Same information a good HPC support ticket needs — which, as it happens, is what
the model you're training is being taught to say.
