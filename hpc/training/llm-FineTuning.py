"""Fine-tune Qwen2.5-1.5B-Instruct to write SQL, using QLoRA (4-bit base + LoRA adapter).

Same recipe as kaggle/llm-FineTuning.ipynb, minus the explanations, plus DDP:
each GPU gets its own full copy of the model and its own slice of the data.

Submit it with run.sh. Quick smoke test on one node with 2 GPUs:
    torchrun --standalone --nproc_per_node=2 llm-FineTuning.py \
        --model Qwen/Qwen2.5-0.5B-Instruct --samples 50 --max-steps 5 --out /tmp/t
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

SYSTEM = "You are a SQL generator. Given a table schema and a question, reply with one SQL query and nothing else."

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--dataset", default="b-mc2/sql-create-context")
p.add_argument("--samples", type=int, default=2000)
p.add_argument("--out", default="./out")
p.add_argument("--epochs", type=float, default=1.0)
p.add_argument("--max-steps", type=int, default=-1, help="-1 = run the full epochs")
p.add_argument("--lr", type=float, default=2e-4)
args = p.parse_args()

rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
if rank == 0:
    print(f"[setup] world size {os.environ.get('WORLD_SIZE', 1)}, {torch.cuda.device_count()} GPUs on this node", flush=True)

# --- data: flat columns -> chat messages -> one rendered string per row ---
tokenizer = AutoTokenizer.from_pretrained(args.model)


def to_text(row):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"{row['context']}\n\n{row['question']}"},
        {"role": "assistant", "content": row["answer"]},
    ]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}


data = load_dataset(args.dataset, split=f"train[:{args.samples}]")
data = data.map(to_text, remove_columns=data.column_names)

# --- model: 4-bit base, one copy per GPU ---
model = AutoModelForCausalLM.from_pretrained(
    args.model,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # T4/V100 have no bfloat16
    ),
    device_map={"": local_rank},
)
model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

config = SFTConfig(
    output_dir=args.out,
    dataset_text_field="text",
    max_length=1024,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=args.epochs,
    max_steps=args.max_steps,
    learning_rate=args.lr,
    fp16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    ddp_find_unused_parameters=False,
    logging_steps=10,  # prints {'loss': ..., 'epoch': ...} into the slurm .out
    save_strategy="no",
    report_to="none",
    disable_tqdm=True,  # progress bars are unreadable in a log file
)

trainer = SFTTrainer(model=model, args=config, train_dataset=data, peft_config=lora, processing_class=tokenizer)
if rank == 0:
    trainer.model.print_trainable_parameters()

trainer.train()

trainer.save_model(f"{args.out}/adapter")  # writes on rank 0 only, tokenizer included

# --- merge the adapter into a full fp16 model, which is what the servers load ---
# Reload the base in fp16 on CPU: merging into the 4-bit copy would bake the
# quantization error in, and vLLM/SGLang want ordinary fp16 weights.
if rank == 0:
    del trainer, model
    torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, device_map="cpu")
    merged = PeftModel.from_pretrained(base, f"{args.out}/adapter").merge_and_unload()
    merged.save_pretrained(f"{args.out}/merged")
    tokenizer.save_pretrained(f"{args.out}/merged")
    print(f"[done] adapter: {args.out}/adapter | merged model to serve: {args.out}/merged", flush=True)
