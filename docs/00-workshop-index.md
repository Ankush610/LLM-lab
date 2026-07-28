# LLM Fine-Tuning & Inferencing Workshop — Index

**Format:** half day · ~2 hrs theory (slides) + ~3 hr 50 min practical (this document)
**Platform:** Kaggle Notebooks, GPU T4 ×2
**Stack:** Unsloth · Llama-3.2-3B-Instruct · QLoRA · Ollama + vLLM

Every practical section maps back to a slide from the theory deck. The `↳` marks
tell you which one, so you can point at the deck mid-lab rather than re-explaining.

---

## Part A — Theory (existing deck, ~2 hrs)

| # | Section | Topics |
|---|---------|--------|
| A1 | AI Introduction | What AI/ML/DL are, where LLMs sit |
| A2 | Language Model History | Embeddings (sparse vs dense) · RNN · LSTM · Cross vs Self Attention |
| A3 | Transformer Model | What it is · Need for attention · Word embedding · Positional embedding · K/Q/V projections · Attention block · Value matrix · Multi-head attention (need + working) |
| A4 | LLM Parameters | Weights, biases, token embeddings, KV cache · Model size from parameter count × precision · VRAM: training vs inference (4–5×) |
| A5 | LLM Fine-Tuning | What and why · Full fine-tuning · Adapters · Low rank · LoRA · Quantization · QLoRA |
| A6 | Model Types, Formats, Variants | Dense, MoE, task-specific, vision, image-gen, audio, multimodal · GGUF, GPTQ, AWQ, ONNX, TensorRT, EXL2 · Instruct vs Base |
| A7 | Fine-Tuning Steps | Choose model · Choose quantization · Dataset format · Chat template · LoRA config · SFTTrainer · Save · Infer |
| A8 | Inference Parameters | Context window vs max tokens · Max tokens & stop · Temperature · Top-K · Top-P · Presence vs frequency penalty |

☕ **Lunch / break**

---

## Part B — Practical (~3 hr 50 min)

### Session 0 · Pre-Workshop — *done before the day*, ~15 min

Sent a week ahead. See [pre-workshop-setup.md](pre-workshop-setup.md).
**This is the highest-risk item in the whole workshop** — thirty people hitting
a gated-model wall at minute five destroys the session.

| # | Topic |
|---|-------|
| 0.1 | Kaggle account + **phone verification** (required for GPU *and* internet) |
| 0.2 | Hugging Face account |
| 0.3 | Accept the Llama 3.2 community licence (gated repo) |
| 0.4 | HF read token → saved as Kaggle Secret `HF_TOKEN` |
| 0.5 | Run `00_setup_and_gpu_check.ipynb`, submit the output banner |

> **Escape hatch:** every notebook honours `USE_UNGATED_MODEL = True`, which
> swaps to Qwen2.5-3B-Instruct. Nobody is blocked by a pending licence approval.

---

### Session 1 · Lab 0 — Environment & Sanity Check · 20 min
`kaggle/00_setup_and_gpu_check.ipynb`

| # | Topic | Slide |
|---|-------|-------|
| 1.1 | Kaggle setup: Accelerator → GPU T4 ×2, Internet → On | — |
| 1.2 | Reading your GPU: `nvidia-smi`, `torch.cuda.get_device_capability()` | — |
| 1.3 | **bf16 vs fp16 — compute capability decides your dtype.** T4 is sm75: fp16 only, no Flash Attention 2 | ↳ A4 Precision |
| 1.4 | Installing the stack: unsloth, trl, peft, bitsandbytes | — |
| 1.5 | HF auth from Kaggle Secrets — not a pasted token | — |
| 1.6 | Disk reality: `/kaggle/working` is ~20 GB, and it matters in Lab 5 | — |
| 1.7 | Green-light banner — everyone matches before anyone proceeds | — |

---

### Session 2 · Lab 1 — Meeting the Model: Baseline Inference · 30 min
`kaggle/01_baseline_inference.ipynb` — the "before" half of the demo

| # | Topic | Slide |
|---|-------|-------|
| 2.1 | Loading Llama-3.2-3B in 4-bit with `FastLanguageModel.from_pretrained` | ↳ A7 Choose model/quant |
| 2.2 | **Measure the VRAM** — 3B × 4-bit ≈ 2 GB, your arithmetic against a real number | ↳ A4 Model size |
| 2.3 | Where the parameters went: `model.named_parameters()` — weights, biases, embedding table | ↳ A4 Parameters |
| 2.4 | Chat template made visible: print the raw templated string, special tokens and all | ↳ A6 Instruct vs Base |
| 2.5 | First generation with `.generate()` | — |
| 2.6 | **Baseline eval: 12 fixed AURA questions.** The model invents nonsense. Saved to `baseline_answers.json` — Lab 4 needs it | — |
| 2.7 | **Inference parameter playground**, one cell each: | ↳ **all of A8** |
| | · `max_new_tokens` vs context window, and overflowing it | |
| | · `temperature` 0.0 → 0.7 → 1.5, same prompt, three personalities | |
| | · `top_k` / `top_p` — printing the surviving token distribution so the cutoff is *visible* | |
| | · `presence_penalty` vs `frequency_penalty` — a prompt built to loop, then fixed | |
| | · `stop` / EOS — the classic "model won't shut up" bug | |
| 2.8 | KV cache observed: generate with and without `use_cache`, watch VRAM and speed | ↳ A4 KV cache |

---

### Session 3 · Lab 2 — Dataset Preparation · 25 min
`kaggle/02_dataset_prep.ipynb`

| # | Topic | Slide |
|---|-------|-------|
| 3.1 | The problem: AURA, a cluster that does not exist — and why that's the point | — |
| 3.2 | Reading the raw data, deliberately not training-ready | ↳ A7 Dataset format |
| 3.3 | **Four formats, same rows, side by side:** raw · Alpaca · ShareGPT · ChatML | ↳ A7 Dataset format |
| 3.4 | Where Alpaca breaks: it silently drops every multi-turn conversation | ↳ A7 Dataset format |
| 3.5 | `standardize_sharegpt`, and why everything converged on `messages` | — |
| 3.6 | **Applying the chat template for training** — and the #1 silent killer, train/inference template mismatch | ↳ A7 Chat template |
| 3.7 | Token-length histogram → choosing `max_seq_length` | ↳ A4 VRAM |
| 3.8 | `train_on_responses_only` — print the label mask and *see* the `-100`s | — |

---

### Session 4 · Lab 3 — QLoRA Fine-Tuning · 55 min ⭐ core lab
`kaggle/03_qlora_finetune.ipynb`

| # | Topic | Slide |
|---|-------|-------|
| 4.1 | Why 4-bit: the memory math for full FT vs QLoRA on a 3B model | ↳ A5 Full FT / Quantization |
| 4.2 | `get_peft_model`, every argument as we type it: | ↳ A5 Adapter / Low rank / LoRA |
| | · `r` — the rank, the actual bottleneck dimension | |
| | · `lora_alpha` — scaling, and the `alpha = 2r` convention | |
| | · `target_modules` — why q/k/v/o, and what gate/up/down add | ↳ A3 K Q V |
| | · `lora_dropout`, `use_gradient_checkpointing` | |
| 4.3 | **The payoff number:** ~24M trainable / 3.2B total ≈ **0.7%** | ↳ A5 LoRA |
| 4.4 | **Pair exercise (5 min):** set `r` to 4 / 16 / 64, watch params and adapter size move | ↳ A5 Low rank |
| 4.5 | `SFTConfig` / `SFTTrainer`: batch size vs gradient accumulation (one knob, two halves), LR, warmup, `adamw_8bit` | ↳ A7 SFTTrainer |
| 4.6 | **Launch training** (~15–20 min) — start the break here | — |
| 4.7 | While it runs: `nvidia-smi` in a second cell. **Training vs inference VRAM, measured live** — your 4–5× claim confirmed on their screen | ↳ A4 VRAM training vs inference |
| 4.8 | Reading the loss curve: healthy · `loss=0.0` (masking bug) · flat (LR too low) | — |

☕ **Break — 10 min**, started at 4.6 so training runs through it

---

### Session 5 · Lab 4 — Before vs After · 20 min
`kaggle/04_before_after.ipynb` — **the emotional peak. Protect this time.**

| # | Topic | Slide |
|---|-------|-------|
| 5.1 | `FastLanguageModel.for_inference` and why it's ~2× faster | — |
| 5.2 | Re-running the **exact same 12 prompts** from Lab 1 | — |
| 5.3 | **Side-by-side diff table**: hallucination vs correct answer, row by row | — |
| 5.4 | These questions were **never in the training set** — only paraphrases were. This is generalisation, not recall | — |
| 5.5 | Out-of-domain probe: did we break it? First look at catastrophic forgetting | — |
| 5.6 | Adapter on/off (`disable_adapter()`) — same weights, two behaviours | ↳ A5 Adapter |
| 5.7 | Honest limits: 665 rows ≠ knowledge · format vs facts · when you actually want RAG | — |

---

### Session 6 · Lab 5 — Saving, Merging & Model Formats · 20 min
`kaggle/05_save_and_gguf.ipynb`

| # | Topic | Slide |
|---|-------|-------|
| 6.1 | Three save modes, real sizes: adapter ~90 MB · merged 16-bit ~6.4 GB · merged 4-bit | ↳ A7 Save model |
| 6.2 | When to ship which — why you keep 40 adapters and one base model | ↳ A5 Adapter |
| 6.3 | `push_to_hub` — also the safest way to get work off Kaggle before the session dies | — |
| 6.4 | **GGUF export** at `q4_k_m`, with the size table for `q8_0` / `f16` | ↳ A6 Model formats |
| 6.5 | ⚠️ **Disk discipline:** merged fp16 + f16 GGUF will not both fit in 20 GB | — |
| 6.6 | Format tour (no hands): which format pairs with which runtime, and why GGUF ≠ vLLM | ↳ A6 Model formats |

---

### Session 7 · Lab 6 — Serving A: Ollama + GGUF · 20 min
`kaggle/06_serve_ollama.ipynb`

| # | Topic | Slide |
|---|-------|-------|
| 7.1 | Ollama on Kaggle: sessions run as root, no systemd → `ollama serve &` + readiness poll | — |
| 7.2 | **Writing a `Modelfile`** — the inference parameters return, now as config not code: | ↳ **A8 again** |
| | · `FROM ./model-q4_k_m.gguf` | |
| | · `TEMPLATE` — the chat template, third encounter; by now it's obvious | ↳ A7 Chat template |
| | · `PARAMETER temperature / top_k / top_p / repeat_penalty / num_ctx / stop` | |
| | · `SYSTEM` — baking in the persona | |
| 7.3 | `ollama create` → `ollama run` → chatting with the model you trained an hour ago | — |
| 7.4 | Hitting Ollama's OpenAI-compatible endpoint from Python | — |
| 7.5 | Quality check: does `q4_k_m` feel worse? Same prompts, judge by eye | ↳ A5 Quantization |

---

### Session 8 · Lab 7 — Serving B: vLLM · 25 min
`kaggle/07_serve_vllm.ipynb`

> **The fragile lab.** Own notebook, own session. Lab 6 already banked a working
> served model, so if this fails it costs a topic, not the workshop.

| # | Topic | Slide |
|---|-------|-------|
| 8.1 | Why a serving engine exists: PagedAttention and KV-cache fragmentation | ↳ A4 KV cache |
| 8.2 | `--enable-lora` — **serving the adapter without merging**, and why that matters at 40 adapters | ↳ A5 Adapter |
| 8.3 | Key flags: `--dtype half` (mandatory on T4), `--gpu-memory-utilization`, `--max-model-len` | ↳ A4 Precision |
| 8.4 | Background server in a notebook: launch, poll `/health`, tail the log | — |
| 8.5 | OpenAI client: the same six sampling knobs, now over HTTP | ↳ **A8, third pass** |
| 8.6 | **Throughput demo:** 1 request vs 32 concurrent. Continuous batching answers "why not just `.generate()`" | — |
| 8.7 | Streaming responses | — |
| 8.8 | Ollama vs vLLM — a decision table, not a winner | ↳ A6 Model formats |

---

### Session 9 · Wrap-Up · 15 min

| # | Topic |
|---|-------|
| 9.1 | The whole pipeline on one slide: data → format → template → LoRA → train → merge → quantize → serve |
| 9.2 | **Troubleshooting cheat sheet** (handout) — see [troubleshooting.md](troubleshooting.md) |
| 9.3 | Getting your work off Kaggle before the session expires |
| 9.4 | What we skipped on purpose: RLHF/DPO, eval harnesses, multi-node, MoE, real-scale serving |
| 9.5 | Where this runs next: your own GPU, and the Slurm + Apptainer cluster |
| 9.6 | Take-home: swap in your own dataset, re-run Labs 2–6 |

---

## Timing

| Segment | Time | Cumulative |
|---------|------|-----------|
| Lab 0 · Environment | 20 min | 0:20 |
| Lab 1 · Baseline inference | 30 min | 0:50 |
| Lab 2 · Dataset prep | 25 min | 1:15 |
| Lab 3 · QLoRA fine-tuning | 55 min | 2:10 |
| ☕ Break *(inside Lab 3)* | 10 min | 2:20 |
| Lab 4 · Before vs after | 20 min | 2:40 |
| Lab 5 · Save & GGUF | 20 min | 3:00 |
| Lab 6 · Ollama | 20 min | 3:20 |
| Lab 7 · vLLM | 25 min | 3:45 |
| Wrap-up | 15 min | **4:00** |

**If you are running short**, cut in this order:
1. Lab 7 (vLLM) → −25 min. Lab 6 already proved serving works.
2. Lab 4.4 pair exercise → −5 min.
3. Lab 2 format tour → demo instead of hands-on, −10 min.

Never cut Lab 4. It is the only session that proves the other three hours were worth it.

> All timings are estimates until the instructor dry run. Record real numbers in
> [instructor-notes.md](instructor-notes.md).
