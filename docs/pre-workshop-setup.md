# Before the Workshop — please do this in advance

**Time needed: about 15 minutes. Please do it at least two days before.**

The hands-on session runs entirely in your browser on Kaggle — nothing to
install, no GPU of your own required. But two of the steps below involve
approvals that are **not instant**, so leaving them to the morning of the
workshop means watching instead of participating.

If you get stuck on any step, don't burn time on it — jump to
[If something doesn't work](#if-something-doesnt-work) at the bottom. There's a
fallback that unblocks you completely.

---

## 1. Kaggle account with phone verification

1. Sign up at [kaggle.com](https://www.kaggle.com).
2. Go to **Settings → Phone Verification** and verify your number.

**Do not skip the phone step.** Kaggle requires it before it will give a notebook
either a **GPU** or **internet access**, and the workshop needs both. Verification
is usually instant but occasionally takes a few hours.

**Check it worked:** create a new notebook, open the right-hand panel, and confirm
you can set *Accelerator* to **GPU T4 ×2** and toggle *Internet* to **On**.

---

## 2. Hugging Face account

Sign up at [huggingface.co](https://huggingface.co). Any plan; the free one is fine.

---

## 3. Accept the Llama 3.2 licence  ← *the one that can take time*

1. Visit [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
2. Fill in the access form and accept the community licence.
3. Wait for the page to show you have access.

Approval is usually granted within minutes, but it is **not guaranteed to be
instant** — it has taken a day for some people. This is exactly why we're asking
you now rather than on the day.

> Don't worry if it hasn't come through by the morning. See the fallback below —
> you'll be able to do every lab regardless.

---

## 4. Create a Hugging Face token

1. Go to **Settings → Access Tokens** → *Create new token*.
2. Type: **Read**.
3. Name it something like `kaggle-workshop`.
4. Copy the token — you only see it once. It starts with `hf_`.

---

## 5. Add the token to Kaggle as a Secret

Do **not** paste your token into a notebook cell. It gets saved into the notebook,
and shared notebooks leak tokens constantly.

1. Open any Kaggle notebook.
2. Right panel → **Add-ons → Secrets**.
3. **Add a new secret**:
   - Label: `HF_TOKEN` *(exactly this, case-sensitive)*
   - Value: your `hf_...` token
4. Make sure the checkbox next to it is **ticked** so the notebook can see it.

---

## 6. Run the check notebook

1. Open `kaggle/00_setup_and_gpu_check.ipynb` (link will be sent with this email).
2. Settings: **Accelerator → GPU T4 ×2**, **Internet → On**.
3. **Run All.**
4. It finishes in about a minute and prints a banner like this:

```
==============================================================
  AURA WORKSHOP · ENVIRONMENT CHECK
==============================================================
  GPU              Tesla T4 x2
  Compute cap.     sm75
  VRAM             14.7 GB
  bfloat16         no
  FlashAttention2  no
  -> training dtype  float16
  -> vllm --dtype    half
  Free disk        19.5 GB
==============================================================
  Ready. Continue to Lab 1.
==============================================================
```

5. **Copy that banner into the form** *(link in the email)*.

`bfloat16: no` and `FlashAttention2: no` are **correct and expected** on a T4 —
they are not errors. We explain why in Lab 0.

---

## Checklist

- [ ] Kaggle account, phone verified
- [ ] Can select GPU T4 ×2 and Internet: On in a notebook
- [ ] Hugging Face account
- [ ] Llama 3.2 licence accepted (or fallback noted)
- [ ] HF read token created
- [ ] Token saved as Kaggle Secret named `HF_TOKEN`
- [ ] Check notebook run, banner submitted

---

## If something doesn't work

**Llama licence still pending, or was declined**
Not a problem. In the first cell of every notebook, change:

```python
USE_UNGATED_MODEL = True
```

That switches to Qwen2.5-3B-Instruct, which needs no licence at all. It is the
same size, trains just as fast, and every lab works identically. You lose
nothing except the Llama name.

**"No GPU available"**
The accelerator isn't set, or phone verification isn't complete. Check the
right-hand panel of the notebook, and Settings → Phone Verification.

**Notebook can't download anything**
Internet is off. Right panel → toggle **Internet: On**. This needs phone
verification too.

**`HF_TOKEN` not found**
Either the secret is named something else (it must be exactly `HF_TOKEN`), or
the checkbox next to it isn't ticked for this notebook.

**401 / "gated repo" error**
The licence hasn't been granted yet, or the notebook isn't using your token.
Use the `USE_UNGATED_MODEL = True` fallback and carry on.

**Anything else**
Reply to this email with a screenshot of the error and the output of the check
notebook. Please don't leave it until the morning — that's the one time we can't
help you quickly.

---

## What to bring

Just a laptop and a browser. Everything runs on Kaggle's GPUs.

If you'd like to fine-tune on **your own data** in the take-home exercise, bring
a few hundred question-and-answer pairs about any subject you know well. We'll
show you the conversion step in Lab 2. Entirely optional.

---

## What we'll build

You'll fine-tune a 3-billion-parameter language model to be a support assistant
for a fictional HPC cluster called AURA — a system that doesn't exist, so the
base model cannot possibly know anything about it.

You'll ask it twelve questions before training and watch it confidently make up
answers. Then you'll train an adapter — about 0.7% of the model's parameters, on
a free GPU, in roughly fifteen minutes — ask the same twelve questions, and get
correct answers. Then you'll export it, quantize it, and serve it over an API.

See you there.
