# Inference Parameters — Cheat Sheet

One page. Hand it out at Lab 1; it stays useful through Labs 6 and 7.

---

## The mental model

The model does **not** choose a word. At each step it outputs a score for every
token in its vocabulary (~128,000 for Llama 3.2). Those scores become
probabilities. Everything below is about **how you pick from that distribution**.

None of it changes the model. Same weights, same distribution — different
sampling.

```
logits  ──temperature──>  scaled  ──top_k / top_p──>  shortlist  ──sample──>  token
                                          ↑
                                    penalties adjust
                                    logits before this
```

---

## The parameters

| Parameter | Range | What it does | Raise it when | Lower it when |
|---|---|---|---|---|
| **temperature** | 0–2 | Scales logits before softmax. Low = sharp/deterministic, high = flat/random | Output is bland, repetitive | Output is unfocused or wrong |
| **top_k** | 1–100 | Keep only the k most likely tokens | You want more variety | You get nonsense words |
| **top_p** | 0–1 | Keep the smallest set summing to p. Size adapts to confidence | Same as top_k, but self-tuning | Same |
| **presence_penalty** | −2–2 | Flat penalty once a token has appeared *at all* | It won't change subject | It's avoiding necessary words |
| **frequency_penalty** | −2–2 | Penalty proportional to *how often* a token appeared | It loops | It won't reuse key terms |
| **max_tokens** | 1–∞ | Hard cap on generated length | Answers get cut off | You want brevity/lower cost |
| **stop** | strings | Halt generation on a match | It runs past the answer | It stops too early |

---

## Sensible starting points

| Task | temperature | top_p | notes |
|---|---|---|---|
| Factual Q&A, extraction | **0.0** | — | Deterministic. Always use for evaluation. |
| Code generation | 0.1–0.3 | 0.95 | Low, but not zero |
| General chat / assistant | **0.7** | 0.9 | The usual default |
| Brainstorming | 0.9–1.1 | 0.95 | |
| Creative writing | 1.0–1.3 | 0.95 | Above ~1.5 coherence falls apart |

**Set temperature *or* top_p, not both.** Tuning both at once makes the effect of
each impossible to reason about. Most people move temperature and leave top_p at
0.9.

---

## temperature=0 is special

It disables sampling entirely — always the highest-probability token. Same
prompt, same output, every time.

Use it for: evaluation, benchmarks, anything reproducible, and any bug report.

> In the HuggingFace API this is `do_sample=False`, not literally
> `temperature=0` — passing zero to a sampler is a division by zero. The `ask()`
> helper in Lab 1 handles the switch for you.

---

## presence vs frequency penalty

The distinction people get wrong:

```
Text so far: "the cat sat on the cat mat the cat"
                                        ↑ "cat" has appeared 3 times

presence_penalty  = 1.0  →  penalty applied once   (flat, appeared-or-not)
frequency_penalty = 1.0  →  penalty applied 3x     (scales with count)
```

- **presence** — "talk about something else"
- **frequency** — "stop saying that word so much"

For a model stuck in a loop, **frequency** is usually the right tool.

> HuggingFace `generate()` exposes neither. It has `repetition_penalty`, which is
> multiplicative and behaves most like a presence penalty. It's **1.0 = off**,
> not 0. Values above ~1.3 start damaging fluency.

---

## Context window vs max_tokens

Constantly confused, entirely different:

| | Context window | max_tokens |
|---|---|---|
| Limits | prompt **+** output | output only |
| Set by | the model architecture | you, per request |
| Llama 3.2 3B | 128,000 | your choice |
| Exceeding it | error, or silently dropped input | output stops mid-sentence |

If your answer is cut off mid-word, it's almost always `max_tokens` — not the
context window.

Note that a longer context costs **memory**, because the KV cache grows with it.
That's why Lab 7 sets `--max-model-len 2048`: not because the model can't do
more, but because the T4 can't hold the cache for more.

---

## Same idea, four different names

The single most annoying thing about this topic.

| Concept | HF `generate()` | Ollama Modelfile | OpenAI / vLLM API | llama.cpp CLI |
|---|---|---|---|---|
| randomness | `temperature` | `temperature` | `temperature` | `--temp` |
| nucleus | `top_p` | `top_p` | `top_p` | `--top-p` |
| top-k | `top_k` | `top_k` | *(not exposed)* | `--top-k` |
| repetition | `repetition_penalty` | `repeat_penalty` | `frequency_penalty`, `presence_penalty` | `--repeat-penalty` |
| output length | `max_new_tokens` | `num_predict` | `max_tokens` | `-n` |
| context size | *(model config)* | `num_ctx` | *(server flag)* | `-c` |
| stop strings | `stopping_criteria` | `stop` | `stop` | `--reverse-prompt` |

---

## Debugging by symptom

| Symptom | Likely cause | Try |
|---|---|---|
| Repeats the same phrase | no repetition control | `frequency_penalty=0.5`, or `repetition_penalty=1.15` |
| Never stops | no stop token honoured | set `stop`, check `eos_token_id` |
| Cut off mid-sentence | token budget | raise `max_tokens` |
| Bland, generic | temperature too low | 0.7–0.9 |
| Incoherent, invented words | temperature too high | drop to 0.7, add `top_p=0.9` |
| Different answer every run | sampling is on | `temperature=0` |
| Ignores the system prompt | template mismatch | check the chat template — see Lab 2 |
| Correct but too verbose | no length guidance | ask for brevity in the prompt; `max_tokens` truncates, it doesn't summarise |

---

## Worth remembering

1. **Sampling parameters cannot fix a bad model.** They change *how* you draw from the distribution, never what's in it. If the facts are wrong, no temperature setting helps — that's what Lab 3 was for.
2. **`temperature=0` for anything you need to measure.** Comparing two models at temperature 0.9 measures mostly luck.
3. **Change one knob at a time.** Otherwise you learn nothing about which one mattered.
