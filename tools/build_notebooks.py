"""Generate the Kaggle workshop notebooks.

    python tools/build_notebooks.py

Writes kaggle/00_*.ipynb .. kaggle/07_*.ipynb. Edit the cell content here, not
in the generated files, or your changes are lost on the next build.
"""

from __future__ import annotations

from pathlib import Path

from nbbuild import code, md, write

OUT = Path(__file__).resolve().parent.parent / "kaggle"

# ==========================================================================
# Shared cells
# ==========================================================================

# Every notebook needs the repo on sys.path. Three ways to get it, tried in
# order, so a participant is never stuck on plumbing.
BOOTSTRAP = '''
# --- Locate the workshop repo -------------------------------------------
# Tries, in order: already present -> attached Kaggle Dataset -> git clone.
import os, sys, subprocess
from pathlib import Path

REPO_URL = "https://github.com/Ankush610/LLM-lab.git"

def find_repo() -> Path:
    for candidate in [Path("/kaggle/working/LLM-lab"), Path.cwd(), Path.cwd().parent]:
        if (candidate / "common" / "config.py").exists():
            return candidate
    for d in Path("/kaggle/input").glob("*"):          # attached as a Dataset
        if (d / "common" / "config.py").exists():
            return d
    print("Repo not found locally, cloning...")        # last resort
    subprocess.run(["git", "clone", "-q", REPO_URL, "/kaggle/working/LLM-lab"], check=True)
    return Path("/kaggle/working/LLM-lab")

REPO = find_repo()
sys.path[:0] = [str(REPO / "common"), str(REPO / "dataset")]
print(f"repo: {REPO}")

import config
print(config.summary())
'''

INSTALL = '''
# --- Install the fine-tuning stack --------------------------------------
# Kaggle ships a matched torch/CUDA pair. --no-deps stops pip replacing torch
# with an incompatible build, which shows up later as baffling CUDA errors.
#
# Takes 2-4 minutes. "dependency conflict" warnings here are expected and fine.
# Output is deliberately NOT suppressed: if this step fails, you need to see it.
!pip install -q --no-deps unsloth unsloth_zoo
!pip install -q --no-deps trl peft accelerate bitsandbytes
!pip install -q datasets huggingface_hub sentencepiece protobuf

print("\\ninstall finished - verifying imports...")
import importlib
missing = [m for m in ("unsloth", "trl", "peft", "bitsandbytes", "datasets")
           if importlib.util.find_spec(m) is None]
print("MISSING: " + ", ".join(missing) if missing else "all packages importable")
'''

HF_AUTH = '''
# --- Hugging Face authentication ----------------------------------------
# Reads the Kaggle Secret named HF_TOKEN. Never paste a token into a cell:
# it is saved with the notebook and shared notebooks leak tokens constantly.
import os

try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
    print("Hugging Face: authenticated")
except Exception as e:
    print(f"Hugging Face: NOT authenticated ({type(e).__name__})")
    print("  Fix: right panel -> Add-ons -> Secrets -> add HF_TOKEN, tick the box.")
    print("  Or set USE_UNGATED_MODEL = True below to skip the gated model entirely.")
'''

LOAD_MODEL = '''
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = config.MODEL_NAME,
    max_seq_length = config.MAX_SEQ_LENGTH,
    dtype          = None,               # None = auto-detect from the GPU
    load_in_4bit   = config.LOAD_IN_4BIT,
)
print(f"\\nloaded {config.MODEL_NAME}")
print(f"dtype in use: {model.dtype}")
'''


def footer(nb_next: str, what: str) -> str:
    return f"""
---

### Next: `{nb_next}` — {what}

> **Kaggle tip:** if the session has been idle a while, check the right-hand
> panel still shows the GPU attached before starting the next notebook.
"""


# ==========================================================================
# 00 · Setup and GPU check
# ==========================================================================

def nb00() -> list[dict]:
    return [
        md("""
        # Lab 0 · Environment & Sanity Check

        **~20 minutes.** Everyone gets to the same green banner before anyone moves on.

        By the end of this notebook you will know what GPU you have, what numeric
        precision it can do, and whether your Hugging Face access works.

        ### Before you run anything

        In the right-hand panel:

        | Setting | Value |
        |---|---|
        | Accelerator | **GPU T4 ×2** |
        | Internet | **On** |
        | Persistence | Off |

        Both need phone verification on your Kaggle account. If you can't select
        them, that's why.
        """),

        md("""
        ## 1.2 What GPU did we get?

        `nvidia-smi` is the first command to reach for on any GPU machine. Read
        the memory column — that number is the ceiling on everything you do today.
        """),
        code("!nvidia-smi"),

        md("""
        ## 1.4 Install the stack

        Four packages matter:

        - **unsloth** — the fast fine-tuning layer; patches the model for ~2× speed and ~50% less VRAM
        - **peft** — the LoRA implementation (Parameter-Efficient Fine-Tuning)
        - **trl** — `SFTTrainer`, the supervised fine-tuning loop
        - **bitsandbytes** — the 4-bit quantization kernels that make QLoRA possible

        `--no-deps` is deliberate: it stops pip replacing Kaggle's torch build with
        an incompatible one. **If the install fails**, see `docs/troubleshooting.md`
        — pinned fallback versions are listed there.
        """),
        code(INSTALL),

        md("## Locate the repo and load the shared config"),
        code(BOOTSTRAP),

        md("""
        ## 1.3 Compute capability decides your precision

        Every NVIDIA GPU has a *compute capability* — `sm75`, `sm80`, `sm90`. It
        determines which numeric formats the silicon supports natively.

        | | fp16 | bf16 | Flash Attention 2 |
        |---|---|---|---|
        | T4 (sm75, 2018) | yes | **no** | **no** |
        | A100 (sm80, 2020) | yes | yes | yes |
        | H100 (sm90, 2022) | yes | yes | yes |

        **bf16** has the same exponent range as fp32 but fewer mantissa bits. That
        range is what stops training from overflowing, so bf16 is more forgiving
        than fp16 — but the T4 predates it.

        On a T4 we use fp16 with loss scaling instead. It works fine. You'll see
        `bfloat16: no` below and that is **correct, not an error**.

        > ↳ Slide: *Model Size based on Parameter Count and Precision*
        """),
        code('''
        import gpu_check
        info = gpu_check.report()
        '''),

        md("""
        ## 1.5 Hugging Face authentication

        Llama 3.2 is a **gated** model: you must accept Meta's licence and be
        logged in. If your approval hasn't come through, don't wait — set
        `USE_UNGATED_MODEL = True` and use Qwen2.5-3B instead. Every lab works
        identically.
        """),
        code(HF_AUTH),

        md("""
        ## 1.6 Disk reality check

        `/kaggle/working` holds about **20 GB**, and Lab 5 will merge a 6.4 GB
        model and then convert it to GGUF. That is tight. We plan for it there;
        just note the number now.
        """),
        code('''
        import shutil
        total, used, free = shutil.disk_usage("/kaggle/working")
        print(f"/kaggle/working:  {free/2**30:5.1f} GB free of {total/2**30:.1f} GB")
        print(f"\\nLab 5 needs roughly:")
        print(f"  merged 16-bit model    6.4 GB")
        print(f"  GGUF q4_k_m            2.0 GB")
        print(f"  base model cache       2.3 GB")
        print(f"  ---------------------------")
        print(f"  total                 10.7 GB  ->  {'OK' if free/2**30 > 12 else 'TIGHT'}")
        '''),

        md("""
        ## 1.7 Green light

        If this prints `READY`, you're set. If not, the message says what to fix.
        """),
        code('''
        checks = {
            "GPU available":      info.available,
            "Repo on sys.path":   True,
            "HF authenticated":   bool(os.environ.get("HF_TOKEN")) or config.USE_UNGATED_MODEL,
            "Disk > 12 GB free":  shutil.disk_usage("/kaggle/working")[2] / 2**30 > 12,
        }

        width = max(len(k) for k in checks)
        for name, ok in checks.items():
            print(f"  {'PASS' if ok else 'FAIL'}  {name:<{width}}")

        print()
        if all(checks.values()):
            print("  READY — continue to Lab 1.")
        else:
            print("  NOT READY — see docs/troubleshooting.md, or ask an instructor.")
            print("  Note: 'HF authenticated' can be skipped by setting")
            print("        USE_UNGATED_MODEL = True in common/config.py")
        '''),

        md(footer("01_baseline_inference.ipynb",
                  "meet the model, and watch it invent answers")),
    ]


# ==========================================================================
# 01 · Baseline inference
# ==========================================================================

def nb01() -> list[dict]:
    return [
        md("""
        # Lab 1 · Meeting the Model: Baseline Inference

        **~30 minutes.** Nothing is trained in this notebook. We are establishing
        the "before" — and playing with every inference parameter from the deck.

        Two things happen here:

        1. We ask the base model twelve questions about a cluster called **AURA**
           and save its answers. AURA doesn't exist, so it cannot possibly know —
           watch it answer confidently anyway.
        2. We work through temperature, top-k, top-p, penalties and stop
           sequences hands-on.

        > ↳ Slides: *LLM Parameters* · *entire Inference Parameters section*
        """),

        code(INSTALL),
        code(BOOTSTRAP),
        code(HF_AUTH),

        md("""
        ## 2.1 Load the model in 4-bit

        `load_in_4bit=True` is the Q in QLoRA. Each weight, normally 16 bits, is
        stored in 4 — a 4× reduction that is what makes a 3B model fit
        comfortably on a free T4.

        `dtype=None` lets Unsloth pick fp16 or bf16 from your GPU. On a T4 it
        picks fp16, as Lab 0 predicted.
        """),
        code(LOAD_MODEL),

        md("""
        ## 2.2 Measure the VRAM — check the arithmetic from the slides

        The theory said: **model size ≈ parameter count × bytes per parameter.**

        Llama-3.2-3B has 3.21 billion parameters. At 4 bits (0.5 bytes) that's
        roughly 1.6 GB, plus overhead for the parts that stay in higher precision
        (embeddings, layer norms). Let's see what actually landed on the card.

        **One trap first.** The obvious way to count parameters is wrong on a
        4-bit model:

        ```python
        sum(p.numel() for p in model.parameters())   # says 1.80 B. Not true.
        ```

        bitsandbytes packs **two 4-bit values into every byte**, so a quantized
        weight tensor reports exactly half the numbers it really holds. The
        unpacked shape is kept on the side, in `p.quant_state.shape`:

        ```python
        def real_numel(p):
            qs = getattr(p, "quant_state", None)
            return qs.shape.numel() if qs is not None else p.numel()
        ```

        That is what `count_params` below does. Worth knowing, because the wrong
        number is convincing — embeddings and norms are *not* quantized, so those
        rows stay correct and only the layers you'd never hand-check are halved.

        > ↳ Slide: *Model Size based on Parameter Count and Precision*
        """),
        code('''
        import torch
        from paramcount import count_params, real_numel

        def vram(label=""):
            used = torch.cuda.memory_allocated() / 2**30
            peak = torch.cuda.max_memory_allocated() / 2**30
            print(f"  {label:<28} {used:5.2f} GB in use   (peak {peak:5.2f} GB)")
            return used

        n_params, _ = count_params(model)
        naive = sum(p.numel() for p in model.parameters())

        print(f"  parameters                  {n_params/1e9:.2f} B")
        print(f"  (naive numel() sum would say {naive/1e9:.2f} B - the 4-bit packing)")
        print(f"  if stored at fp16 (2 B)     {n_params*2/2**30:5.2f} GB")
        print(f"  if stored at 4-bit (0.5 B)  {n_params*0.5/2**30:5.2f} GB")
        print()
        vram("actually on the GPU")
        print("\\n  The real number sits above the 4-bit estimate because")
        print("  embeddings and norms stay in higher precision.")
        '''),

        md("""
        ## 2.3 Where did the parameters go?

        The slides listed what a "parameter" actually is: weights, biases, token
        embeddings. Let's look at the real tensors and group them.

        Notice how much of the model is the **embedding table** — vocabulary size
        × hidden dimension is a surprisingly large slice for a small model.

        Two things to look for in the output:

        - **No "output head" row.** Llama 3.2 sets `tie_word_embeddings: true` —
          the layer that turns hidden states back into vocabulary scores is the
          *same tensor* as the input embedding, reused. One 394 M table doing
          both jobs, which is a real chunk of a model this size.
        - Every row uses `real_numel`, not `p.numel()`. Drop it and MLP and
          attention silently halve, for the packing reason above.

        > ↳ Slide: *Parameters: weights, biases, token embedding, KV cache*
        """),
        code('''
        from collections import defaultdict

        groups = defaultdict(int)
        for name, p in model.named_parameters():
            if "embed" in name:            key = "token embeddings"
            elif "lm_head" in name:        key = "output head"
            elif "norm" in name:           key = "layer norms"
            elif any(k in name for k in ("q_proj","k_proj","v_proj","o_proj")):
                                           key = "attention (Q,K,V,O)"
            elif any(k in name for k in ("gate_proj","up_proj","down_proj")):
                                           key = "MLP (gate,up,down)"
            else:                          key = "other"
            groups[key] += real_numel(p)          # not p.numel() - see above

        total = sum(groups.values())
        print(f"  {'component':<22} {'params':>12}   share")
        print("  " + "-"*48)
        for k, v in sorted(groups.items(), key=lambda x: -x[1]):
            print(f"  {k:<22} {v/1e6:>9.1f} M   {v/total:>6.1%}")
        print("  " + "-"*48)
        print(f"  {'total':<22} {total/1e9:>9.2f} B")
        print(f"\\n  output head tied to embeddings: "
              f"{model.config.tie_word_embeddings}")
        '''),

        md("""
        ## 2.4 Make the chat template visible

        This is the single most under-appreciated detail in fine-tuning.

        An instruct model was not trained on raw text — it was trained on text
        wrapped in **special tokens** marking who is speaking. `<|start_header_id|>`,
        `<|eot_id|>` and friends aren't decoration; the model learned that an
        answer follows one specific token sequence.

        Get this wrong and nothing crashes. The model just gets quietly worse, and
        you spend a day blaming your learning rate.

        > ↳ Slide: *Instruct vs Base*
        """),
        code('''
        messages = [
            {"role": "system",  "content": "You are a helpful assistant."},
            {"role": "user",    "content": "What is 2+2?"},
        ]

        templated = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print("What the model actually receives:")
        print("=" * 66)
        print(templated)
        print("=" * 66)
        print("\\nAs token IDs:")
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        for i in ids[:14]:
            print(f"  {i:>6}  {tokenizer.decode([i])!r}")
        print(f"  ... {len(ids)} tokens total")
        '''),

        md("""
        ## 2.5 First generation

        A small helper we'll reuse all notebook. Note `FastLanguageModel.for_inference` —
        it enables Unsloth's ~2× faster inference path.
        """),
        code('''
        FastLanguageModel.for_inference(model)

        def ask(question, system=config.SYSTEM_PROMPT, **kw):
            """Generate an answer. Any sampling kwarg can be overridden."""
            msgs = ([{"role": "system", "content": system}] if system else []) + \\
                   [{"role": "user", "content": question}]
            inputs = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt"
            ).to("cuda")

            params = dict(
                max_new_tokens = config.MAX_NEW_TOKENS,
                temperature    = config.TEMPERATURE,
                top_p          = config.TOP_P,
                top_k          = config.TOP_K,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )
            params.update(kw)
            if params.get("temperature") == 0:            # greedy decoding
                params.update(do_sample=False, temperature=None, top_p=None, top_k=None)

            out = model.generate(input_ids=inputs, **params)
            return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

        print(ask("Explain what a GPU is in two sentences."))
        '''),

        md("""
        ## 2.6 The baseline evaluation — the "before" half of the demo

        Twelve questions about the **AURA cluster**. AURA is fictional: its
        partition names, module versions, quotas and wrapper commands appear in no
        pretraining corpus anywhere.

        A well-behaved model would say "I don't know about AURA." Watch what
        actually happens.

        **Read a few of these answers properly.** They are fluent, plausible,
        formatted like real documentation, and completely invented. That is the
        failure mode fine-tuning fixes — and it's why "the model sounds confident"
        tells you nothing.

        *This takes 2–3 minutes.*
        """),
        code('''
        from eval_prompts import EVAL_PROMPTS
        import compare

        baseline = {}
        for i, p in enumerate(EVAL_PROMPTS, 1):
            print(f"[{i:>2}/{len(EVAL_PROMPTS)}] {p['question'][:58]}...")
            baseline[p["id"]] = ask(p["question"], temperature=0.0)   # greedy = reproducible

        compare.save(baseline, config.BASELINE_ANSWERS)
        '''),

        code('''
        # Look at three of them in full.
        for p in EVAL_PROMPTS[:3]:
            print("=" * 72)
            print("Q:", p["question"])
            print("-" * 72)
            print(baseline[p["id"]][:500])
            print()
        '''),

        code('''
        # Score it. Expect a very low number - that is the point.
        scores = compare.score_all(baseline)
        print(f"  baseline: {sum(scores.values())}/{len(scores)} correct\\n")
        for p in EVAL_PROMPTS:
            print(f"    {'PASS' if scores[p['id']] else 'FAIL'}  {p['question'][:56]}")
        '''),

        md("""
        ---
        # 2.7 Inference Parameter Playground

        Everything in the *Inference Parameters* section of the deck, hands-on.

        The model is now **fixed** — we are not changing a single weight. Every
        difference you see from here on comes from how we *sample* from the same
        probability distribution.
        """),

        md("""
        ### Temperature

        At each step the model produces a score (logit) for every token in the
        vocabulary. Temperature divides those logits before they become
        probabilities:

        - **T → 0** — the gap between best and second-best widens; always picks the top token. Deterministic.
        - **T = 1** — the model's own distribution, untouched.
        - **T > 1** — flattens the distribution; unlikely tokens get a real chance. Creative, then incoherent.

        Same prompt, four temperatures.
        """),
        code('''
        prompt = "Describe a thunderstorm in one sentence."

        for t in (0.0, 0.3, 0.7, 1.5):
            print(f"--- temperature = {t} ---")
            print(ask(prompt, temperature=t, max_new_tokens=60).strip(), "\\n")
        '''),

        md("""
        ### Seeing top-k and top-p rather than being told about them

        Both truncate the candidate list before sampling:

        - **top-k = 40** — keep the 40 most likely tokens, discard the rest.
        - **top-p = 0.9** — keep the smallest set of tokens whose probabilities sum to 0.9. The list size *adapts*: tiny when the model is confident, large when it isn't.

        Let's print the actual distribution for one prediction and mark where each
        cutoff falls.
        """),
        code('''
        import torch.nn.functional as F

        msgs = [{"role": "user", "content": "The capital of France is"}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt").to("cuda")
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]

        probs = F.softmax(logits.float(), dim=-1)
        top = torch.topk(probs, 15)

        cumulative = 0.0
        print(f"  {'rank':>4} {'token':<16} {'prob':>8} {'cumulative':>11}")
        print("  " + "-" * 46)
        for rank, (p_, idx) in enumerate(zip(top.values, top.indices), 1):
            cumulative += p_.item()
            marks = []
            if rank == 10:                      marks.append("<- top_k=10 cuts here")
            if cumulative >= 0.9 and cumulative - p_.item() < 0.9:
                                                marks.append("<- top_p=0.9 cuts here")
            print(f"  {rank:>4} {tokenizer.decode([idx]):<16} "
                  f"{p_.item():>8.4f} {cumulative:>11.4f}  {' '.join(marks)}")

        print(f"\\n  Vocabulary size: {len(probs):,} tokens.")
        print(f"  The top 15 hold {top.values.sum().item():.1%} of the probability mass.")
        print("  Everything else is noise we throw away before sampling.")
        '''),

        md("""
        ### Repetition: presence vs frequency penalty

        Small models fall into loops. Two different fixes, often confused:

        - **presence penalty** — a flat penalty once a token has appeared *at all*. Pushes toward new topics.
        - **frequency penalty** — scales with *how often* the token appeared. Punishes the tenth repeat far harder than the first.

        HuggingFace exposes `repetition_penalty`, which is closest to a presence
        penalty. We'll provoke a loop and then damp it.
        """),
        code('''
        loop_prompt = "List the colour red, over and over, forever:"

        for rp in (1.0, 1.15, 1.4):
            label = "no penalty" if rp == 1.0 else f"repetition_penalty={rp}"
            print(f"--- {label} ---")
            print(ask(loop_prompt, repetition_penalty=rp,
                      max_new_tokens=70, temperature=0.8).strip()[:280], "\\n")
        '''),

        md("""
        ### max_new_tokens, stop sequences, and the context window

        Three separate limits people routinely confuse:

        | | what it limits |
        |---|---|
        | **context window** | prompt + generation combined. A property of the *model* (128k for Llama 3.2). |
        | **max_new_tokens** | how much is generated *this call*. Your choice. |
        | **stop / EOS** | content-based early exit, regardless of the other two. |

        Truncation mid-sentence is nearly always `max_new_tokens`, not the context
        window.
        """),
        code('''
        print(f"  model context window : {model.config.max_position_embeddings:,} tokens")
        print(f"  configured for today : {config.MAX_SEQ_LENGTH:,} tokens")
        print(f"  EOS token            : {tokenizer.eos_token!r} (id {tokenizer.eos_token_id})\\n")

        q = "Count from 1 to 30, one number per line."
        for n in (20, 120):
            print(f"--- max_new_tokens = {n} ---")
            print(ask(q, max_new_tokens=n, temperature=0.0).strip()[:220], "\\n")

        print("The first one stops mid-count: it hit the token budget, not the")
        print("end of the answer. Nothing errors - you just get a truncated string.")
        '''),

        md("""
        ## 2.8 The KV cache, observed

        Generating token 500 requires attending to the previous 499. Without a
        cache you'd recompute all their key and value vectors every single step —
        quadratic work for a linear output.

        The KV cache stores them. It costs memory that grows with sequence length,
        which is why the slides listed it alongside weights as a real consumer of
        VRAM — and why serving engines like vLLM (Lab 7) exist mainly to manage it.

        > ↳ Slide: *Parameters / KV cache*
        """),
        code('''
        import time

        q = "Write a short paragraph about mountains."
        for use_cache in (True, False):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            _ = ask(q, max_new_tokens=100, temperature=0.0, use_cache=use_cache)
            dt = time.time() - t0
            print(f"  use_cache={str(use_cache):<5}  {dt:5.1f} s   "
                  f"peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")

        print("\\n  Same output, several times the wall-clock without the cache.")
        print("  It trades memory for not redoing work - the core tradeoff in serving.")
        '''),

        md("""
        ## What we established

        - The base model **cannot** answer AURA questions, and does not say so — it invents fluent, well-formatted, wrong answers.
        - Sampling parameters change output substantially without touching a single weight.
        - `baseline_answers.json` is saved. **Lab 4 needs it** — don't delete it.

        Next we build the dataset that fixes this.
        """),
        md(footer("02_dataset_prep.ipynb", "four dataset formats, and the chat template")),
    ]


# ==========================================================================
# 02 · Dataset preparation
# ==========================================================================

def nb02() -> list[dict]:
    return [
        md("""
        # Lab 2 · Dataset Preparation

        **~25 minutes.**

        Fine-tuning is mostly data work. This notebook covers the part of the deck
        that lists dataset formats — except instead of a slide showing four JSON
        snippets, you generate all four from the same source and diff them.

        > ↳ Slides: *Dataset Format: Raw, Alpaca, ShareGPT, ChatML* · *Apply Chat Template*
        """),

        code(INSTALL),
        code(BOOTSTRAP),

        md("""
        ## 3.1 The problem we're solving

        **AURA** is a fictional HPC cluster. Its facts live in
        `dataset/aura_spec.py` — five partitions, four QoS levels, storage
        quotas, module versions, wrapper commands like `aura-quota`.

        Fictional on purpose. If we fine-tuned on real Slurm documentation, the
        base model would already half-know it and the before/after would be a
        subtle matter of taste. With AURA, the model starts at zero and the
        improvement is unambiguous.
        """),
        code('''
        import aura_spec

        print(f"{aura_spec.CLUSTER['full_name']}\\n")
        for p in aura_spec.PARTITIONS:
            gpus = p["gpus_per_node"] if p["gpus_per_node"] != "none" else "CPU only"
            print(f"  {p['name']:<16} {gpus:<24} max {p['max_walltime_human']}")
        '''),

        md("""
        ## 3.2 Generate the raw conversations

        `generate_raw.py` turns the fact file into conversations by asking each
        fact many different ways. This is synthetic data generation from a
        knowledge base — a genuinely useful technique when you have documentation
        but no Q&A logs.

        Deterministic: same seed, same dataset.
        """),
        code('''
        !cd {REPO}/dataset && python generate_raw.py
        '''),

        code('''
        import json
        raw = [json.loads(l) for l in open(REPO / "dataset/raw/aura_support_raw.jsonl")]

        print(f"{len(raw)} conversations\\n")
        r = raw[0]
        print(f"topic: {r['topic']}")
        for m in r["messages"]:
            print(f"\\n  [{m['role']}]\\n  {m['content'][:300]}")
        '''),

        md("""
        ## 3.3 The four formats

        Now the part that matters. Same conversations, four encodings.

        | format | shape | multi-turn? |
        |---|---|---|
        | **raw** | one text blob | only by convention |
        | **Alpaca** | `instruction` / `input` / `output` | **no** |
        | **ShareGPT** | `conversations[{from, value}]` | yes |
        | **ChatML** | `messages[{role, content}]` | yes |

        Run this and read the output — it explains the formats better than any
        slide can.
        """),
        code('''
        !cd {REPO}/dataset && python build_dataset.py --show 1
        '''),

        md("""
        ## 3.4 Where Alpaca breaks

        Look at the multi-turn example above: **Alpaca dropped it.**

        Alpaca has exactly one `output` field. A three-turn conversation has
        nowhere to go. That's not a bug in our converter — it's the format's
        actual limitation, and it's why the field moved to `messages`.

        This matters practically: convert a chat dataset to Alpaca and you can
        silently lose every multi-turn example without a single error.
        """),
        code('''
        !cd {REPO}/dataset && python build_dataset.py
        '''),

        md("""
        ### A note on contamination

        `build_dataset.py` also removes any training row phrased *exactly* like
        one of the twelve evaluation questions.

        Without that, Lab 4 would be a memorisation test — the model would recite
        an answer it saw verbatim. Paraphrases of the same facts stay in, so it
        still learns everything; it just has to generalise to the exact wording at
        eval time. That's the honest version of the demo, and it's the same
        discipline you'd want on a real project.
        """),

        md("""
        ## 3.5 Load it as a `Dataset`

        We train on the ChatML version. `standardize_sharegpt` exists to convert
        ShareGPT's `from`/`value` naming into `role`/`content` — ours is already
        in the target shape.
        """),
        code('''
        from datasets import load_dataset

        ds = load_dataset(
            "json",
            data_files = {
                "train": str(REPO / "dataset/formats/chatml_train.jsonl"),
                "val":   str(REPO / "dataset/formats/chatml_val.jsonl"),
            },
        )
        print(ds)
        print("\\nfirst example:")
        for m in ds["train"][0]["messages"]:
            print(f"  [{m['role']:<9}] {m['content'][:110]}")
        '''),

        md("""
        ## 3.6 Apply the chat template — the step that silently ruins runs

        In Lab 1 we saw the special tokens an instruct model expects. Training
        data must carry **exactly** the same structure.

        The failure mode: train with one template, infer with another. Nothing
        errors. The model just underperforms, and you go looking for the problem
        in your hyperparameters.

        Unsloth's `get_chat_template` pins the tokenizer to a named template so
        train and inference cannot drift apart.

        > ↳ Slide: *Apply Chat Template*
        """),
        code('''
        from unsloth.chat_templates import get_chat_template
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)

        def formatting(batch):
            return {"text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]}

        ds = ds.map(formatting, batched=True)

        print("A formatted training example, exactly as the model will see it:")
        print("=" * 70)
        print(ds["train"][0]["text"][:900])
        print("=" * 70)
        '''),

        md("""
        ## 3.7 Token lengths → choosing `max_seq_length`

        `max_seq_length` is one of the biggest levers on training memory, because
        attention activations scale with it.

        Too low and long examples get truncated — often chopping the answer off a
        training example, teaching the model to stop mid-sentence. Too high and
        you waste VRAM padding.

        Pick it from the data, not from a round number.
        """),
        code('''
        lengths = [len(tokenizer(t)["input_ids"]) for t in ds["train"]["text"]]
        lengths.sort()

        def pct(p): return lengths[int(len(lengths) * p / 100)]

        print(f"  examples     {len(lengths)}")
        print(f"  min / max    {lengths[0]} / {lengths[-1]}")
        print(f"  median       {pct(50)}")
        print(f"  p90 / p99    {pct(90)} / {pct(99)}\\n")

        # Crude terminal histogram - readable without matplotlib.
        buckets = [0] * 10
        step = max(1, (lengths[-1] + 1) // 10)
        for L in lengths:
            buckets[min(9, L // step)] += 1
        for i, count in enumerate(buckets):
            bar = "#" * round(count / max(buckets) * 40)
            print(f"  {i*step:>5}-{(i+1)*step:<5} {count:>4} {bar}")

        chosen = config.MAX_SEQ_LENGTH
        over = sum(1 for L in lengths if L > chosen)
        print(f"\\n  max_seq_length = {chosen} truncates {over} examples "
              f"({over/len(lengths):.1%})")
        '''),

        md("""
        ## 3.8 `train_on_responses_only` — see the mask

        By default the loss covers every token, so the model is partly trained to
        *predict the user's questions*. That's wasted capacity: at inference the
        question is given.

        `train_on_responses_only` masks the prompt with `-100`, PyTorch's
        "ignore this position" label. Loss is computed on the answer only.

        The print below shows which tokens count. This is also the diagnostic for
        `loss = 0.0` — if masking is misconfigured, everything gets ignored and the
        loss goes to exactly zero.
        """),
        code('''
        # Illustrative: this is what the trainer does internally in Lab 3.
        sample = ds["train"][0]["text"]
        ids = tokenizer(sample, return_tensors="pt")["input_ids"][0]

        # Llama 3.x marks the assistant turn with this header.
        marker = "<|start_header_id|>assistant<|end_header_id|>"
        if marker not in sample:
            marker = "<|im_start|>assistant"          # Qwen / ChatML fallback

        split_at = len(tokenizer(sample.split(marker)[0] + marker)["input_ids"])
        labels = [-100] * split_at + ids[split_at:].tolist()

        print(f"  {len(ids)} tokens, {split_at} masked, "
              f"{len(ids)-split_at} contribute to the loss\\n")
        print(f"  {'token':<22} {'label':>8}   counted?")
        print("  " + "-" * 48)
        for i in list(range(6)) + ["..."] + list(range(split_at-2, min(split_at+6, len(ids)))):
            if i == "...":
                print(f"  {'...':<22} {'...':>8}")
                continue
            tok = repr(tokenizer.decode([ids[i]]))[:20]
            counted = "no  (prompt)" if labels[i] == -100 else "YES (answer)"
            print(f"  {tok:<22} {labels[i]:>8}   {counted}")
        '''),

        md("""
        ## Ready to train

        - **665** training conversations, **34** held out for validation
        - ChatML format, chat template applied
        - `max_seq_length` chosen from the actual distribution
        - Loss will be computed on answers only

        Next notebook is the core of the workshop.
        """),
        md(footer("03_qlora_finetune.ipynb", "QLoRA — train 0.7% of the model")),
    ]


# ==========================================================================
# 03 · QLoRA fine-tuning
# ==========================================================================

def nb03() -> list[dict]:
    return [
        md("""
        # Lab 3 · QLoRA Fine-Tuning ⭐

        **~55 minutes**, of which ~20 is the model training while you drink coffee.

        This is the centre of the workshop. Everything the deck said about
        adapters, low rank, LoRA and quantization becomes a number you can read
        off your own screen.

        > ↳ Slides: *Full Fine Tuning* · *What is an Adapter* · *What is Low Rank*
        > · *LoRA* · *What is Quantization* · *QLoRA* · *VRAM: training vs inference*
        """),

        code(INSTALL),
        code(BOOTSTRAP),
        code(HF_AUTH),

        md("""
        ## 4.1 Why not just fine-tune everything?

        Before touching LoRA, let's price up full fine-tuning of this 3B model.

        Training needs far more than the weights. For each trainable parameter
        Adam keeps two extra state tensors (momentum and variance), plus a
        gradient, all typically in fp32.
        """),
        code('''
        P = 3.21e9          # Llama-3.2-3B parameter count
        GB = 2**30

        full = {
            "weights (fp16)":          P * 2,
            "gradients (fp16)":        P * 2,
            "Adam momentum (fp32)":    P * 4,
            "Adam variance (fp32)":    P * 4,
        }

        print("  FULL FINE-TUNING")
        for k, v in full.items():
            print(f"    {k:<26} {v/GB:>7.1f} GB")
        print(f"    {'-'*26} {'-'*10}")
        print(f"    {'subtotal':<26} {sum(full.values())/GB:>7.1f} GB   (+ activations)")
        print(f"\\n    A free T4 has 15 GB. This does not fit. Not close.")

        r, n_target = 16, 7
        lora_params = P * 0.007     # measured below - about 0.7%
        qlora = {
            "weights (4-bit, frozen)": P * 0.5,
            "LoRA weights (fp16)":     lora_params * 2,
            "LoRA gradients (fp16)":   lora_params * 2,
            "Adam state (8-bit)":      lora_params * 2,
        }
        print("\\n  QLoRA")
        for k, v in qlora.items():
            print(f"    {k:<26} {v/GB:>7.2f} GB")
        print(f"    {'-'*26} {'-'*10}")
        print(f"    {'subtotal':<26} {sum(qlora.values())/GB:>7.2f} GB   (+ activations)")
        print("\\n    Fits, with room for a batch. That is the entire pitch.")
        '''),

        md("""
        ## Load the model

        Same 4-bit load as Lab 1. The base weights stay **frozen** for the whole
        of training — we never compute a gradient for them.
        """),
        code(LOAD_MODEL),

        code('''
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template=config.CHAT_TEMPLATE)
        print(f"chat template: {config.CHAT_TEMPLATE}")
        '''),

        md("""
        ## 4.2 Attaching the adapter

        The idea in one line: instead of updating a big weight matrix `W`
        (dimensions d×d), freeze it and learn a small correction

        ```
        ΔW = B @ A        where A is r×d and B is d×r
        ```

        With `r = 16` and `d = 3072`, `W` has 9.4M parameters while `A` and `B`
        together have 98k — about 1%. At inference you can either add `ΔW` into
        `W` (merging, Lab 5) or keep it separate and swap adapters at will.

        Every argument below:

        | argument | meaning |
        |---|---|
        | `r` | the rank — the bottleneck dimension. Capacity knob. |
        | `lora_alpha` | scaling; the update is multiplied by `alpha/r`. Convention `alpha = 2r`. |
        | `target_modules` | which matrices get an adapter |
        | `lora_dropout` | regularisation; 0 is fastest in Unsloth's fused kernels |
        | `use_gradient_checkpointing` | recompute activations instead of storing them — slower, much less memory |

        > ↳ Slides: *What is an Adapter* · *What is Low Rank* · *LoRA* · *3 Weight Matrix Projections: K Q V*
        """),
        code('''
        model = FastLanguageModel.get_peft_model(
            model,
            r                          = config.LORA_R,
            lora_alpha                 = config.LORA_ALPHA,
            lora_dropout               = config.LORA_DROPOUT,
            bias                       = config.LORA_BIAS,
            target_modules             = config.TARGET_MODULES,
            use_gradient_checkpointing = "unsloth",
            random_state               = config.SEED,
            use_rslora                 = False,
            loftq_config               = None,
        )
        print("\\nadapter attached to:", ", ".join(config.TARGET_MODULES))
        '''),

        md("""
        ## 4.3 The payoff number

        This is the slide that made LoRA famous, as a number from your own run.

        We count with `count_params` from Lab 1 rather than summing `numel()`:
        the frozen base is 4-bit and packs two values per byte, so a naive total
        comes out at 1.80 B and the trainable share looks twice as large as it
        is. The adapter itself is fp16, so that half of the ratio is unaffected.
        """),
        code('''
        from paramcount import count_params

        total, trainable = count_params(model)

        print(f"  trainable parameters   {trainable:>15,}")
        print(f"  total parameters       {total:>15,}")
        print(f"  trainable share        {trainable/total:>15.3%}")
        print()
        print(f"  Adapter on disk (fp16) {trainable*2/2**20:>12.0f} MB")
        print(f"  Full model on disk     {total*2/2**30:>12.1f} GB")
        print()
        print(f"  We are training {trainable/total:.1%} of the network and leaving")
        print(f"  the other {1-trainable/total:.1%} exactly as Meta shipped it.")
        '''),

        md("""
        ## 4.4 Pair exercise — what does `r` actually buy? (5 min)

        Work with the person next to you. Before running the cell, predict: if
        `r` goes from 16 to 64, what happens to the trainable parameter count?

        Then run it.
        """),
        code('''
        # Arithmetic only - we are not rebuilding the model, just counting.
        d_model, n_layers = model.config.hidden_size, model.config.num_hidden_layers
        n_targets = len(config.TARGET_MODULES)

        print(f"  hidden size {d_model}, {n_layers} layers, "
              f"{n_targets} adapted matrices per layer\\n")
        print(f"  {'r':>4} {'trainable params':>18} {'adapter MB':>12} {'vs r=16':>9}")
        print("  " + "-" * 48)

        base = None
        for r in (4, 8, 16, 32, 64, 128):
            approx = n_layers * n_targets * (2 * d_model * r)
            if r == 16:
                base = approx
            print(f"  {r:>4} {approx:>18,} {approx*2/2**20:>11.0f}M "
                  f"{approx/base:>8.2f}x")

        print("\\n  Linear in r. Doubling the rank doubles the adapter.")
        print("  More capacity is not automatically better: high r on a small")
        print("  dataset overfits, and you pay for it in both memory and time.")
        '''),

        md("""
        ## 4.5 The dataset and the trainer

        Reload the ChatML data from Lab 2 and format it with the same template.
        """),
        code('''
        from datasets import load_dataset

        ds = load_dataset("json", data_files={
            "train": str(REPO / "dataset/formats/chatml_train.jsonl"),
            "val":   str(REPO / "dataset/formats/chatml_val.jsonl"),
        })

        def formatting(batch):
            return {"text": [
                tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=False)
                for m in batch["messages"]
            ]}

        ds = ds.map(formatting, batched=True, remove_columns=["messages"])
        print(ds)
        '''),

        md("""
        ### Training arguments

        **Batch size and gradient accumulation are one knob split in two.**
        `per_device_train_batch_size=2` with `gradient_accumulation_steps=4` gives
        an effective batch of 8: the GPU processes 2 at a time and accumulates
        gradients over 4 micro-batches before stepping. Same maths as a batch of
        8, a quarter of the activation memory, roughly 4× the wall-clock.

        `learning_rate=2e-4` looks enormous next to pretraining rates (~1e-5).
        It's normal for LoRA: we're training a small, freshly-initialised adapter,
        not nudging a converged network.

        > ↳ Slide: *Training with SFT-Trainer*
        """),
        code('''
        from trl import SFTTrainer, SFTConfig

        args = SFTConfig(
            output_dir                  = str(config.WORK_DIR / "training_output"),
            per_device_train_batch_size = config.BATCH_SIZE,
            gradient_accumulation_steps = config.GRAD_ACCUM,
            warmup_steps                = config.WARMUP_STEPS,
            num_train_epochs            = config.NUM_EPOCHS,
            max_steps                   = config.MAX_STEPS or -1,
            learning_rate               = config.LEARNING_RATE,
            logging_steps               = config.LOGGING_STEPS,
            optim                       = config.OPTIMIZER,
            weight_decay                = config.WEIGHT_DECAY,
            lr_scheduler_type           = config.LR_SCHEDULER,
            seed                        = config.SEED,
            report_to                   = "none",
            fp16                        = not torch.cuda.is_bf16_supported(),
            bf16                        = torch.cuda.is_bf16_supported(),
            dataset_text_field          = "text",
            max_seq_length              = config.MAX_SEQ_LENGTH,
        )

        trainer = SFTTrainer(
            model         = model,
            tokenizer     = tokenizer,
            train_dataset = ds["train"],
            args          = args,
        )

        steps = len(ds["train"]) * config.NUM_EPOCHS // (config.BATCH_SIZE * config.GRAD_ACCUM)
        print(f"  effective batch size  {config.BATCH_SIZE * config.GRAD_ACCUM}")
        print(f"  optimizer steps       ~{steps}")
        print(f"  precision             {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
        '''),

        md("""
        ### Train on responses only

        From Lab 2: mask the prompt so loss is computed on answers only.
        """),
        code('''
        from unsloth.chat_templates import train_on_responses_only

        if "llama" in config.CHAT_TEMPLATE:
            inst, resp = "<|start_header_id|>user<|end_header_id|>", \\
                         "<|start_header_id|>assistant<|end_header_id|>"
        else:
            inst, resp = "<|im_start|>user\\n", "<|im_start|>assistant\\n"

        trainer = train_on_responses_only(trainer,
                                          instruction_part=inst,
                                          response_part=resp)
        print(f"loss computed on completions only (marker: {resp!r})")
        '''),

        md("""
        ## 4.7 VRAM before training

        Note this number. We compare against it while training runs — it's the
        deck's claim that training costs 4–5× inference, measured on your GPU.

        > ↳ Slide: *VRAM Usage on Training vs Inference*
        """),
        code('''
        import torch
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated() / 2**30
        print(f"  model loaded, not yet training:  {before:.2f} GB")
        print(f"  total VRAM on this GPU:          "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GB")
        '''),

        md("""
        ## 4.6 Train

        **~15–20 minutes on a T4.** Start your break now.

        Watch the loss column. What you want:

        - a **falling** loss, fast at first then flattening
        - **not** `0.0` — that means the label mask is eating everything
        - **not** flat — that means the learning rate is too low or nothing is trainable
        - **not** `nan` — fp16 overflow; lower the LR
        """),
        code('''
        stats = trainer.train()
        '''),

        md("""
        ### While that ran — training vs inference memory

        Run this now the training is done to see the peak.
        """),
        code('''
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"  inference only (before training)  {before:>6.2f} GB")
        print(f"  peak during training              {peak:>6.2f} GB")
        print(f"  ratio                             {peak/before:>6.2f}x\\n")
        print("  The extra is gradients, optimizer state and stored activations.")
        print("  The deck said 4-5x for FULL fine-tuning. Ours is lower because")
        print("  QLoRA only keeps optimizer state for 0.7% of the parameters -")
        print("  which is the whole reason this fits on a free GPU.")

        m = stats.metrics
        print(f"\\n  runtime      {m['train_runtime']/60:.1f} min")
        print(f"  final loss   {m['train_loss']:.4f}")
        '''),

        md("""
        ### Reading the loss curve
        """),
        code('''
        history = [h for h in trainer.state.log_history if "loss" in h]
        if history:
            losses = [h["loss"] for h in history]
            lo, hi = min(losses), max(losses)
            print(f"  step   loss")
            for h in history[::max(1, len(history)//18)]:
                width = int((h["loss"] - lo) / (hi - lo + 1e-9) * 42)
                print(f"  {h['step']:>4}   {h['loss']:.4f}  {'#' * width}")
            print(f"\\n  first {losses[0]:.4f}  ->  last {losses[-1]:.4f}   "
                  f"({(losses[0]-losses[-1])/losses[0]:+.1%})")
            if losses[-1] < 0.01:
                print("\\n  WARNING: loss near zero usually means the label mask is wrong.")
        '''),

        md("""
        ## Save the adapter

        Only the LoRA weights — around 90 MB, not 6 GB. The base model is
        unchanged and can be re-downloaded any time.
        """),
        code('''
        model.save_pretrained(str(config.ADAPTER_DIR))
        tokenizer.save_pretrained(str(config.ADAPTER_DIR))

        import subprocess
        size = subprocess.run(["du", "-sh", str(config.ADAPTER_DIR)],
                              capture_output=True, text=True).stdout.split()[0]
        print(f"\\nadapter saved to {config.ADAPTER_DIR}  ({size})")
        print("\\nfiles:")
        for f in sorted(config.ADAPTER_DIR.iterdir()):
            print(f"  {f.name:<34} {f.stat().st_size/2**20:>8.1f} MB")
        '''),

        md("""
        ## Done

        You have trained a language model. The adapter is on disk.

        Next notebook asks the same twelve questions from Lab 1 and puts the
        answers side by side.
        """),
        md(footer("04_before_after.ipynb", "the payoff")),
    ]


# ==========================================================================
# 04 · Before vs after
# ==========================================================================

def nb04() -> list[dict]:
    return [
        md("""
        # Lab 4 · Before vs After

        **~20 minutes.** The point of the whole day.

        We reload the model with the adapter from Lab 3, ask the **exact same
        twelve questions** as Lab 1, and compare.

        One thing to hold onto: those twelve questions were **removed from the
        training data**. The model saw paraphrases of the same facts, never these
        exact strings. So this measures generalisation, not recall.
        """),

        code(INSTALL),
        code(BOOTSTRAP),
        code(HF_AUTH),

        md("""
        ## 5.1 Load the fine-tuned model

        Loading from the adapter directory pulls the 4-bit base model and applies
        our ~90 MB of LoRA weights on top.
        """),
        code('''
        from unsloth import FastLanguageModel
        import torch

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(config.ADAPTER_DIR),
            max_seq_length = config.MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = config.LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)      # ~2x faster inference path
        print("fine-tuned model ready")
        '''),

        code('''
        def ask(question, system=config.SYSTEM_PROMPT, **kw):
            msgs = ([{"role": "system", "content": system}] if system else []) + \\
                   [{"role": "user", "content": question}]
            inputs = tokenizer.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt").to("cuda")
            params = dict(max_new_tokens=config.MAX_NEW_TOKENS, do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
            params.update(kw)
            out = model.generate(input_ids=inputs, **params)
            return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        '''),

        md("## 5.2 Ask the same twelve questions"),
        code('''
        from eval_prompts import EVAL_PROMPTS
        import compare

        tuned = {}
        for i, p in enumerate(EVAL_PROMPTS, 1):
            print(f"[{i:>2}/{len(EVAL_PROMPTS)}] {p['question'][:58]}...")
            tuned[p["id"]] = ask(p["question"])

        compare.save(tuned, config.TUNED_ANSWERS)
        '''),

        md("""
        ## 5.3 Side by side

        Green = the answer contains the facts we expect. Red = it doesn't.

        The scoring is a crude keyword check on purpose — it makes the change
        countable, but **the real evaluation is you reading the two columns.**
        """),
        code('''
        baseline = compare.load(config.BASELINE_ANSWERS)
        compare.report(baseline, tuned)
        '''),

        code('''
        # Two in full, for a proper read.
        for p in EVAL_PROMPTS[:2]:
            print("=" * 74)
            print("Q:", p["question"])
            print("-" * 74)
            print("BEFORE:\\n", baseline[p["id"]][:420])
            print("\\nAFTER:\\n", tuned[p["id"]][:420])
            print()
        '''),

        md("""
        ## 5.5 Did we break anything else?

        Fine-tuning on a narrow domain can degrade general ability — *catastrophic
        forgetting*. With LoRA on 665 examples for 2 epochs it should be mild,
        because the base weights never moved.

        Judge for yourself.
        """),
        code('''
        from eval_prompts import OUT_OF_DOMAIN_PROMPTS

        for q in OUT_OF_DOMAIN_PROMPTS:
            print(f"Q: {q}")
            print(f"A: {ask(q, max_new_tokens=110).strip()[:320]}\\n")
        '''),

        md("""
        ## 5.6 The adapter is a switch

        `disable_adapter()` turns the LoRA weights off without unloading anything.
        Same weights in memory, two different behaviours, toggled in a line.

        This is the mental model to take away: **an adapter is swappable behaviour,
        not a new model.** You keep one base model and as many adapters as you have
        tasks, and they cost 90 MB each.

        > ↳ Slide: *What is an Adapter*
        """),
        code('''
        q = "Which partition should I use for a 4-hour A100 job on AURA?"

        print("--- adapter ON ---")
        print(ask(q).strip()[:300])

        with model.disable_adapter():
            print("\\n--- adapter OFF (base model) ---")
            print(ask(q).strip()[:300])
        '''),

        md("""
        ## 5.7 Being honest about what just happened

        The improvement is real, but let's be precise about what it is.

        **What fine-tuning did well**
        - Taught a consistent format, voice and level of detail
        - Taught a bounded set of facts that appear many times in the data
        - Taught *when to say "I don't know"* — the dataset includes refusals

        **What it did not do**
        - Give the model a database. 665 examples is a behaviour, not a knowledge base.
        - Make it reliable on facts seen once or twice
        - Remove hallucination. It hallucinates less *here*, and will still invent things off-topic.

        **When you'd reach for something else**

        | situation | better tool |
        |---|---|
        | Facts change weekly | RAG — retrieve at query time |
        | Thousands of documents | RAG |
        | Must cite sources | RAG |
        | Need a specific format/voice/behaviour | **fine-tuning** |
        | Need a domain's vocabulary and conventions | **fine-tuning** |
        | Need lower latency and cost than few-shot prompting | **fine-tuning** |

        In production the two are complements, not rivals: fine-tune the
        behaviour, retrieve the facts.
        """),
        md(footer("05_save_and_gguf.ipynb", "saving, merging and model formats")),
    ]


# ==========================================================================
# 05 · Save, merge, GGUF
# ==========================================================================

def nb05() -> list[dict]:
    return [
        md("""
        # Lab 5 · Saving, Merging & Model Formats

        **~20 minutes.**

        A trained adapter in a dying Kaggle session is worth nothing. This
        notebook covers getting it out, in the right format for the right runtime.

        ⚠️ **Disk warning.** `/kaggle/working` is ~20 GB. A merged 16-bit model is
        6.4 GB and GGUF conversion needs room for both. We export **one**
        quantization and clean up as we go. Watch the free-space numbers.

        > ↳ Slides: *Save Model* · *Model Format: GGUF, GPTQ, AWQ, ONNX, TensorRT, EXL2*
        """),

        code(INSTALL),
        code(BOOTSTRAP),
        code(HF_AUTH),

        code('''
        import shutil, subprocess

        def disk(label=""):
            free = shutil.disk_usage("/kaggle/working")[2] / 2**30
            print(f"  {label:<34} {free:5.1f} GB free")
            return free

        def size_of(path):
            out = subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True)
            return out.stdout.split()[0] if out.returncode == 0 else "?"

        disk("starting point")
        '''),

        md("""
        ## 6.1 Three ways to save

        | mode | size | what it is | use when |
        |---|---|---|---|
        | **adapter only** | ~90 MB | just the LoRA matrices | sharing, versioning, serving many adapters |
        | **merged 16-bit** | ~6.4 GB | ΔW folded into W | vLLM, further conversion, HF Hub |
        | **merged 4-bit** | ~2.2 GB | merged then requantized | quick deploy, some quality loss |

        The adapter is already saved from Lab 3.
        """),
        code('''
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = str(config.ADAPTER_DIR),
            max_seq_length = config.MAX_SEQ_LENGTH,
            dtype          = None,
            load_in_4bit   = config.LOAD_IN_4BIT,
        )
        print(f"adapter directory: {size_of(config.ADAPTER_DIR)}")
        for f in sorted(config.ADAPTER_DIR.iterdir()):
            if f.stat().st_size > 1024:
                print(f"  {f.name:<36} {f.stat().st_size/2**20:>8.1f} MB")
        '''),

        md("""
        ## 6.2 Why you'd keep the adapter separate

        90 MB versus 6.4 GB is a 70× difference, and it changes how you deploy.

        Forty fine-tuned variants as merged models is 256 GB and forty things to
        load. Forty adapters is 3.6 GB on top of one base model — and vLLM can
        serve them from a single loaded copy, switching per request (Lab 7).

        Merge when a runtime demands a single artifact — GGUF conversion does.
        """),

        md("""
        ## 6.4 Merge to 16-bit, then convert to GGUF

        **GGUF** is llama.cpp's format: one self-contained file with weights,
        tokenizer and metadata, designed for CPU/GPU inference on consumer
        hardware. It's what Ollama consumes.

        `q4_k_m` — "4-bit, K-quant, medium" — is the standard quality/size
        compromise. Roughly 2 GB for a 3B model, with quality loss most people
        can't detect in chat.

        **This takes 5–10 minutes** and is the most disk-hungry step of the day.
        """),
        code('''
        disk("before GGUF export")

        # Unsloth merges to 16-bit internally, then calls llama.cpp's converter.
        model.save_pretrained_gguf(
            str(config.GGUF_DIR),
            tokenizer,
            quantization_method = config.GGUF_QUANT,
        )

        disk("after GGUF export")
        '''),

        code('''
        gguf_files = sorted(config.GGUF_DIR.glob("*.gguf"))
        print("GGUF files produced:")
        for f in gguf_files:
            print(f"  {f.name:<44} {f.stat().st_size/2**30:>6.2f} GB")

        if gguf_files:
            GGUF_PATH = gguf_files[0]
            print(f"\\nLab 6 will use: {GGUF_PATH}")
        else:
            print("\\nNo .gguf produced - see docs/troubleshooting.md")
        '''),

        md("""
        ### What the other quantizations would have cost

        We only built one because of the disk limit. For reference, same model:

        | quant | size | quality |
        |---|---|---|
        | `f16` | ~6.4 GB | lossless vs the merged model |
        | `q8_0` | ~3.4 GB | essentially indistinguishable |
        | `q5_k_m` | ~2.3 GB | very good |
        | **`q4_k_m`** | **~2.0 GB** | **good — the usual default** |
        | `q3_k_m` | ~1.6 GB | noticeably degraded |
        | `q2_k` | ~1.3 GB | often incoherent |

        The curve is steep at the bottom: below 4-bit, quality falls off fast for
        a small size win. `q4_k_m` is where most people land.
        """),

        md("""
        ## 6.3 Push to the Hub — get it off Kaggle

        Kaggle sessions expire. Whatever isn't pushed or downloaded is gone.

        Pushing the **adapter** is the cheap option: 90 MB, and anyone can
        reconstruct the full model from it.

        Set `LLMLAB_HUB_ID` in `common/config.py`, or edit below.
        """),
        code('''
        HUB_ID = config.HUB_MODEL_ID       # e.g. "yourname/aura-support-3b-lora"

        if HUB_ID:
            model.push_to_hub(HUB_ID, token=os.environ.get("HF_TOKEN"))
            tokenizer.push_to_hub(HUB_ID, token=os.environ.get("HF_TOKEN"))
            print(f"pushed -> https://huggingface.co/{HUB_ID}")
        else:
            print("HUB_ID not set - skipping.")
            print("Set config.HUB_MODEL_ID (or LLMLAB_HUB_ID) to push.")
            print("\\nAlternative: download the GGUF from the Kaggle output panel")
            print("on the right, under /kaggle/working.")
        '''),

        md("""
        ## 6.6 Which format goes with which runtime

        The question that trips people up: *why can't I just use my model
        anywhere?* Because each runtime wants its weights laid out its own way.

        | format | runtime | precision | notes |
        |---|---|---|---|
        | **safetensors** | HF transformers, vLLM | fp16/bf16/4-bit | the interchange default |
        | **GGUF** | llama.cpp, Ollama, LM Studio | 2–8 bit K-quants | single file, CPU-friendly |
        | **GPTQ** | vLLM, AutoGPTQ | 3/4/8-bit | needs calibration data |
        | **AWQ** | vLLM, AutoAWQ | 4-bit | activation-aware; often beats GPTQ |
        | **EXL2** | ExLlamaV2 | variable | fast on consumer NVIDIA |
        | **ONNX** | ONNX Runtime | various | cross-platform, incl. mobile |
        | **TensorRT-LLM** | Triton | various | fastest on NVIDIA, most work to build |

        The two rules worth memorising:

        - **You cannot feed GGUF to vLLM** — vLLM wants safetensors (or GPTQ/AWQ). GGUF is llama.cpp's.
        - **You cannot feed safetensors to Ollama** — Ollama is llama.cpp underneath and wants GGUF.

        Which is why Lab 6 uses the GGUF and Lab 7 uses the adapter directly.
        """),
        code('''
        disk("end of Lab 5")
        print("\\n  Keep for later:")
        print(f"    {config.ADAPTER_DIR}   (Lab 7, vLLM)")
        print(f"    {config.GGUF_DIR}      (Lab 6, Ollama)")
        '''),

        md(footer("06_serve_ollama.ipynb", "serve it with Ollama")),
    ]


# ==========================================================================
# 06 · Ollama
# ==========================================================================

def nb06() -> list[dict]:
    return [
        md("""
        # Lab 6 · Serving A — Ollama + GGUF

        **~20 minutes.**

        The GGUF from Lab 5 becomes a running model you can chat with, plus an
        OpenAI-compatible API endpoint.

        This lab also brings back every sampling parameter from Lab 1 — but as
        **configuration** rather than Python arguments, which is how you'd
        actually ship them.

        > ↳ Slides: *Model Format: GGUF...* · *Inference Parameters*
        """),

        code(BOOTSTRAP),

        md("""
        ## 7.1 Installing Ollama on Kaggle

        Ollama's installer normally wants root and systemd. On Kaggle:

        - **root** — we have it. Kaggle notebook sessions run as root (`whoami`
          below proves it), so the install script works unmodified.
        - **systemd** — not running. The installer warns and skips service setup,
          so we start the server ourselves as a background process.

        Worth internalising: *"needs root"* usually means *"needs to write to
        `/usr`"*. On a cluster where you have no root, the same binary works fine
        extracted into your own directory — which is exactly what the cluster
        version of this workshop does.
        """),
        code('''
        !whoami
        !curl -fsSL https://ollama.com/install.sh | sh
        '''),

        md("""
        ### Start the server

        No systemd, so we launch it ourselves and poll until the port answers.
        Polling rather than `sleep` — the startup time varies.
        """),
        code('''
        import subprocess, time, os, requests

        os.environ["OLLAMA_HOST"] = f"127.0.0.1:{config.OLLAMA_PORT}"
        os.environ["OLLAMA_MODELS"] = "/kaggle/working/ollama_models"   # not $HOME

        server = subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/kaggle/working/ollama.log", "w"),
            stderr=subprocess.STDOUT,
        )

        url = f"http://127.0.0.1:{config.OLLAMA_PORT}"
        for attempt in range(60):
            try:
                if requests.get(url, timeout=2).status_code == 200:
                    print(f"ollama up after {attempt+1}s")
                    break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            print("ollama did not start - check /kaggle/working/ollama.log")
            !tail -20 /kaggle/working/ollama.log
        '''),

        md("""
        ## 7.2 The Modelfile

        A `Modelfile` is Ollama's packaging format — the model file plus its
        default behaviour, in one place.

        Everything from Lab 1's playground appears here as a `PARAMETER`. That is
        the point of this lab: the sampling knobs aren't a Python detail, they're
        deployment configuration you ship with the model.

        | directive | what it does |
        |---|---|
        | `FROM` | the GGUF file |
        | `TEMPLATE` | the chat template — **third time you've seen this** |
        | `PARAMETER` | default sampling settings |
        | `SYSTEM` | the baked-in system prompt |

        The `TEMPLATE` must match what we trained with. Get it wrong and quality
        degrades quietly — the same failure as Lab 2, in a different costume.
        """),
        code('''
        gguf_files = sorted(config.GGUF_DIR.glob("*.gguf"))
        assert gguf_files, f"No GGUF in {config.GGUF_DIR} - re-run Lab 5."
        GGUF_PATH = gguf_files[0]
        print(f"using {GGUF_PATH.name}  ({GGUF_PATH.stat().st_size/2**30:.2f} GB)")
        '''),

        code('''
        # The TEMPLATE must match the model we actually trained. Llama 3 and
        # Qwen/ChatML use different special tokens, so pick by config - otherwise
        # anyone on the ungated fallback gets a silently wrong template.
        if "llama" in config.CHAT_TEMPLATE:
            template = (
                "<|start_header_id|>system<|end_header_id|>\\n\\n"
                "{{ .System }}<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\\n\\n"
                "{{ .Prompt }}<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>\\n\\n"
            )
            stops = ["<|eot_id|>", "<|start_header_id|>"]
        else:
            template = (
                "<|im_start|>system\\n{{ .System }}<|im_end|>\\n"
                "<|im_start|>user\\n{{ .Prompt }}<|im_end|>\\n"
                "<|im_start|>assistant\\n"
            )
            stops = ["<|im_end|>", "<|im_start|>"]

        q3 = chr(34) * 3        # triple quote, kept out of the f-string below
        modelfile = "\\n".join([
            f"FROM {GGUF_PATH}",
            "",
            f"TEMPLATE {q3}{template}{q3}",
            "",
            f"SYSTEM {q3}{config.SYSTEM_PROMPT}{q3}",
            "",
            f"PARAMETER temperature    {config.TEMPERATURE}",
            f"PARAMETER top_k          {config.TOP_K}",
            f"PARAMETER top_p          {config.TOP_P}",
            f"PARAMETER repeat_penalty {config.REPETITION_PENALTY}",
            f"PARAMETER num_ctx        {config.MAX_SEQ_LENGTH}",
            *[f'PARAMETER stop           "{s}"' for s in stops],
        ])

        open("/kaggle/working/Modelfile", "w").write(modelfile)
        print(modelfile)
        '''),

        md("""
        ## 7.3 Build and run

        `ollama create` reads the Modelfile and registers the model. Takes a
        minute or two — it copies the GGUF into Ollama's store.
        """),
        code('''
        !ollama create {config.OLLAMA_MODEL_NAME} -f /kaggle/working/Modelfile
        !ollama list
        '''),

        code('''
        # Your model, running under its own name.
        !ollama run {config.OLLAMA_MODEL_NAME} "Which partition should I use for a 4-hour A100 job on AURA?"
        '''),

        md("""
        ## 7.4 The OpenAI-compatible endpoint

        Ollama exposes an OpenAI-shaped API on port 11434. Any code written
        against the OpenAI SDK works by changing the `base_url` — which is why
        this is the easy path from a notebook to an actual application.
        """),
        code('''
        import requests, json

        def chat(prompt, **params):
            r = requests.post(f"{url}/v1/chat/completions", json={
                "model": config.OLLAMA_MODEL_NAME,
                "messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                **params,
            }, timeout=180)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        from eval_prompts import EVAL_PROMPTS
        for p in EVAL_PROMPTS[:4]:
            print(f"Q: {p['question']}")
            print(f"A: {chat(p['question']).strip()[:300]}\\n")
        '''),

        md("""
        ### The same parameters, now over HTTP

        Second pass over the *Inference Parameters* section. Same concepts as Lab
        1, different transport — and note the names differ slightly between the
        HuggingFace API, the Modelfile, and the OpenAI API. That inconsistency is
        a real and permanent annoyance.

        | concept | HF `generate()` | Modelfile | OpenAI API |
        |---|---|---|---|
        | randomness | `temperature` | `temperature` | `temperature` |
        | nucleus | `top_p` | `top_p` | `top_p` |
        | top-k | `top_k` | `top_k` | *(not exposed)* |
        | repetition | `repetition_penalty` | `repeat_penalty` | `frequency_penalty` / `presence_penalty` |
        | length | `max_new_tokens` | `num_predict` | `max_tokens` |
        | context | *(model config)* | `num_ctx` | *(server-side)* |
        """),
        code('''
        q = "Explain in one sentence what the aura-debug QoS is for."

        for t in (0.0, 0.8, 1.6):
            print(f"--- temperature={t} ---")
            print(chat(q, temperature=t, max_tokens=90).strip()[:260], "\\n")
        '''),

        md("""
        ## 7.5 Did quantization hurt?

        The Ollama model is `q4_k_m` — 4-bit. Lab 4's answers came from the
        4-bit-base-plus-fp16-adapter version. Different numerics, same training.

        Read a few side by side. For most chat use the difference is hard to
        spot, which is the whole reason 4-bit quantization is the default choice
        for local deployment.

        > ↳ Slide: *What is Quantization*
        """),
        code('''
        import compare
        tuned = compare.load(config.TUNED_ANSWERS)     # from Lab 4

        for p in EVAL_PROMPTS[:3]:
            print("=" * 72)
            print("Q:", p["question"])
            print("-" * 72)
            print("Lab 4 (4-bit + fp16 adapter):\\n ", tuned[p["id"]].strip()[:280])
            print("\\nOllama (q4_k_m GGUF):\\n ", chat(p["question"]).strip()[:280])
            print()
        '''),

        md("""
        ## Take it home

        The GGUF is a single self-contained file. Download it from the Kaggle
        output panel, and on your own machine:

        ```bash
        ollama create aura-support -f Modelfile
        ollama run aura-support
        ```

        No GPU required — llama.cpp runs a 3B q4 model on a laptop CPU at
        readable speed. That's the practical appeal of this format.
        """),
        md(footer("07_serve_vllm.ipynb",
                  "vLLM, throughput, and serving adapters without merging")),
    ]


# ==========================================================================
# 07 · vLLM
# ==========================================================================

def nb07() -> list[dict]:
    return [
        md("""
        # Lab 7 · Serving B — vLLM

        **~25 minutes.**

        > ### ⚠️ Read this first
        >
        > **Start this notebook in a fresh session** (Run → Restart & Clear Cell
        > Outputs, or open it as a new notebook). vLLM pins its own torch build
        > and will fight the Unsloth environment if both are installed.
        >
        > vLLM is also tight on a 15 GB T4. If it won't start, that's a resource
        > limit, not a mistake on your part — read along, and use
        > `docs/troubleshooting.md`. Lab 6 already gave you a working served
        > model, so nothing here is load-bearing.

        Ollama is for running a model. vLLM is for *serving* one — many
        concurrent users, maximum throughput.

        > ↳ Slides: *Parameters / KV Cache* · *Inference Parameters*
        """),

        md("""
        ## 8.1 Why a serving engine exists

        In Lab 1 we watched the KV cache trade memory for speed. Serving many
        users at once turns that into the central engineering problem.

        The naive approach reserves a contiguous block of KV cache per request,
        sized for the longest output it *might* produce. Most requests finish
        early, so most of that memory is never used — reported waste is often
        60–80%.

        **PagedAttention** borrows from operating systems: split the cache into
        fixed-size blocks and keep a block table per sequence, so the cache can be
        non-contiguous and allocated on demand. Waste drops to a few percent, and
        the memory you get back becomes concurrency.

        On top of that, **continuous batching** admits a new request the moment
        any sequence finishes, instead of waiting for the whole batch. That is
        what section 8.6 measures.
        """),

        md("""
        ## Install vLLM

        3–5 minutes. It will install its own torch — expected here, and the
        reason this notebook needs its own session.
        """),
        code('''
        !pip install -q vllm openai
        print("\\ninstalled")
        '''),

        code(BOOTSTRAP),
        code(HF_AUTH),

        md("""
        ## 8.2 Serving the adapter without merging

        vLLM can load a base model once and apply LoRA adapters per request
        (`--enable-lora`). With forty fine-tuned variants that's one 6 GB model in
        memory and forty 90 MB adapters, switchable per request — rather than
        forty full models.

        We point it at the **base** model plus our adapter directory.

        ## 8.3 The flags that matter

        | flag | why |
        |---|---|
        | `--dtype half` | **mandatory on T4** — sm75 has no bf16 |
        | `--max-model-len 2048` | KV cache scales with this; the default would OOM |
        | `--gpu-memory-utilization 0.85` | fraction of VRAM vLLM may claim |
        | `--enable-lora` | turn on adapter support |
        | `--max-lora-rank 16` | must be ≥ the `r` we trained with |
        """),
        code('''
        import subprocess, time, requests, os

        BASE_MODEL = config.MODEL_NAME.replace("unsloth/", "").replace("-bnb-4bit", "")
        BASE_MODEL = f"meta-llama/{BASE_MODEL}" if "Llama" in BASE_MODEL else f"Qwen/{BASE_MODEL}"
        print(f"base model : {BASE_MODEL}")
        print(f"adapter    : {config.ADAPTER_DIR}")

        cmd = [
            "vllm", "serve", BASE_MODEL,
            "--dtype", "half",                       # T4: no bf16
            "--max-model-len", "2048",
            "--gpu-memory-utilization", "0.85",
            "--enable-lora",
            "--lora-modules", f"aura={config.ADAPTER_DIR}",
            "--max-lora-rank", str(config.LORA_R),
            "--port", str(config.VLLM_PORT),
            "--disable-log-requests",
        ]
        print("\\n" + " ".join(cmd))

        log = open("/kaggle/working/vllm.log", "w")
        server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        '''),

        md("""
        ### Wait for the server

        Loading and profiling the KV cache takes **3–6 minutes**. Poll `/health`
        rather than guessing at a sleep.
        """),
        code('''
        url = f"http://127.0.0.1:{config.VLLM_PORT}"
        ready = False

        for attempt in range(90):                     # up to ~7.5 minutes
            if server.poll() is not None:             # process died
                print("server exited - last 30 log lines:")
                !tail -30 /kaggle/working/vllm.log
                break
            try:
                if requests.get(f"{url}/health", timeout=2).status_code == 200:
                    print(f"\\nvLLM ready after ~{attempt*5}s")
                    ready = True
                    break
            except requests.exceptions.RequestException:
                pass
            if attempt % 6 == 0:
                print(f"  waiting... {attempt*5}s")
            time.sleep(5)

        if not ready and server.poll() is None:
            print("timed out; last 30 log lines:")
            !tail -30 /kaggle/working/vllm.log
        '''),

        md("""
        ## 8.5 The OpenAI client

        vLLM speaks the OpenAI API. Third and final pass over the sampling
        parameters — and note that `presence_penalty` and `frequency_penalty` are
        **separate fields** here, unlike HuggingFace's single
        `repetition_penalty`.

        `model="aura"` selects our LoRA adapter by the name we registered.
        """),
        code('''
        from openai import OpenAI

        client = OpenAI(base_url=f"{url}/v1", api_key="not-needed")

        def chat(prompt, model="aura", **params):
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=params.pop("max_tokens", 256),
                **params,
            )
            return r.choices[0].message.content

        from eval_prompts import EVAL_PROMPTS
        for p in EVAL_PROMPTS[:3]:
            print(f"Q: {p['question']}")
            print(f"A: {chat(p['question'], temperature=0).strip()[:300]}\\n")
        '''),

        md("""
        ### Adapter on, adapter off — one server

        Same loaded weights. `model="aura"` applies the LoRA; using the base model
        name skips it. This is the deployment story from Lab 4's toggle, running
        as a service.
        """),
        code('''
        q = "Where are the shared Apptainer images stored on AURA?"
        print("--- with adapter (model='aura') ---")
        print(chat(q, model="aura", temperature=0).strip()[:280])
        print("\\n--- base model, no adapter ---")
        print(chat(q, model=BASE_MODEL, temperature=0).strip()[:280])
        '''),

        md("### All six sampling parameters over HTTP"),
        code('''
        q = "Describe the aura-cpu-long partition."

        print("--- temperature 0 vs 1.3 ---")
        for t in (0.0, 1.3):
            print(f"  T={t}: {chat(q, temperature=t, max_tokens=70).strip()[:180]}\\n")

        print("--- presence_penalty vs frequency_penalty ---")
        loop = "List the word 'red' twenty times."
        for kwargs in ({}, {"presence_penalty": 1.5}, {"frequency_penalty": 1.5}):
            label = ", ".join(f"{k}={v}" for k, v in kwargs.items()) or "no penalty"
            print(f"  {label}: {chat(loop, temperature=0.8, max_tokens=60, **kwargs).strip()[:150]}\\n")

        print("--- stop sequence ---")
        print(" ", chat("Count: one, two, three, four, five.",
                        temperature=0, max_tokens=60, stop=["three"]).strip())
        '''),

        md("""
        ## 8.6 Throughput — why not just use `.generate()`

        The headline claim. We send one request, then 32 at once, and compare
        total tokens per second.

        Sequentially, 32 requests take 32× as long. With continuous batching they
        overlap: vLLM runs them through the model together and admits new work as
        sequences finish. Expect several times the aggregate throughput — the
        exact factor depends on how loaded the T4 is.
        """),
        code('''
        import time
        from concurrent.futures import ThreadPoolExecutor

        prompts = [p["question"] for p in EVAL_PROMPTS] * 3     # 36 prompts

        def timed(fn, label):
            t0 = time.time()
            outs = fn()
            dt = time.time() - t0
            toks = sum(len(o.split()) for o in outs) * 1.3       # rough token estimate
            print(f"  {label:<26} {dt:6.1f}s   {len(outs):>3} reqs   "
                  f"{toks/dt:>6.0f} tok/s")
            return dt

        one = timed(lambda: [chat(prompts[0], temperature=0, max_tokens=128)],
                    "1 request")

        many = timed(lambda: list(ThreadPoolExecutor(16).map(
                        lambda p: chat(p, temperature=0, max_tokens=128), prompts[:32])),
                     "32 concurrent requests")

        print(f"\\n  Sequentially 32 requests would take ~{one*32:.0f}s.")
        print(f"  Batched they took {many:.0f}s - about {one*32/many:.1f}x faster.")
        print("  That gap is continuous batching plus PagedAttention.")
        '''),

        md("### 8.7 Streaming"),
        code('''
        stream = client.chat.completions.create(
            model="aura",
            messages=[{"role": "system", "content": config.SYSTEM_PROMPT},
                      {"role": "user", "content": "How do I load PyTorch on AURA?"}],
            max_tokens=160, temperature=0, stream=True,
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\\n\\n(tokens arrive as generated - the latency users actually feel)")
        '''),

        md("""
        ## 8.8 Ollama or vLLM?

        Not a competition — different jobs.

        | | **Ollama** | **vLLM** |
        |---|---|---|
        | Built for | one user, local | many users, server |
        | Format | GGUF | safetensors / GPTQ / AWQ |
        | CPU inference | yes | no, GPU only |
        | Concurrency | limited | continuous batching |
        | Multi-adapter | one at a time | many, per request |
        | Setup | one command | flags and tuning |
        | Quantization | 2–8 bit K-quants | fp16/bf16, GPTQ, AWQ, fp8 |
        | Use when | laptop, demo, prototype, offline | production API, throughput, cost per token |

        Rule of thumb: **Ollama to try it, vLLM to ship it.** Many teams use both —
        Ollama on developer laptops, vLLM in the cluster.
        """),

        code('''
        server.terminate()
        server.wait(timeout=30)
        print("vLLM stopped, VRAM released")
        '''),

        md("""
        ## End of the practical

        In under four hours you have:

        1. Measured a model's real memory footprint against the arithmetic from the slides
        2. Watched a base model confidently invent facts about a cluster that doesn't exist
        3. Built a dataset in four formats and seen why only two of them survive multi-turn
        4. Fine-tuned 0.7% of a 3B model on a free GPU
        5. Watched it answer questions it had never seen in that wording
        6. Merged, quantized and exported it to GGUF
        7. Served it two ways, with the full sampling parameter surface on each

        **Take home:** the GGUF, the adapter, and this repo. Swap in your own
        dataset at Lab 2 and rerun Labs 2–6 — nothing else has to change.
        """),
    ]


# ==========================================================================
# Build
# ==========================================================================

NOTEBOOKS = [
    ("00_setup_and_gpu_check.ipynb", nb00, "Lab 0 · Environment & Sanity Check"),
    ("01_baseline_inference.ipynb",  nb01, "Lab 1 · Baseline Inference"),
    ("02_dataset_prep.ipynb",        nb02, "Lab 2 · Dataset Preparation"),
    ("03_qlora_finetune.ipynb",      nb03, "Lab 3 · QLoRA Fine-Tuning"),
    ("04_before_after.ipynb",        nb04, "Lab 4 · Before vs After"),
    ("05_save_and_gguf.ipynb",       nb05, "Lab 5 · Saving, Merging & Formats"),
    ("06_serve_ollama.ipynb",        nb06, "Lab 6 · Serving with Ollama"),
    ("07_serve_vllm.ipynb",          nb07, "Lab 7 · Serving with vLLM"),
]


def main() -> None:
    print(f"writing to {OUT}/")
    for filename, builder, title in NOTEBOOKS:
        write(OUT / filename, builder(), title)


if __name__ == "__main__":
    main()
