"""
3_train.py
==========
Step 3 — Fine-tune a model on the prepared dataset (Step 2 output).

Launch via `accelerate launch` or use the run_training.py convenience script.

Supported base models (tested)
-------------------------------
  google/gemma-1.1-2b-it
  google/gemma-1.1-7b-it
  meta-llama/Meta-Llama-3-8B-Instruct

Training modes
--------------
  lora   — LoRA adapters, base model in bf16
  qlora  — LoRA adapters, base model in 4-bit NF4 (lower VRAM)
  fft    — Full fine-tuning

The `text` field in each dataset record is pre-formatted with the model's
chat template (produced by 2_prepare_dataset.py), so SFTTrainer uses it
directly without any additional templating.
"""

import os
os.environ["BNB_CUDA_VERSION"] = "121"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

from dataclasses import dataclass, field
import random

import torch
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, set_seed,
)
from peft import LoraConfig
from trl import TrlParser, SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Per-model LoRA target modules
# ---------------------------------------------------------------------------

DEFAULT_LORA_TARGETS = {
    "gemma":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":    ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    "llama":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}

TRUST_REMOTE_CODE_PREFIXES = ("microsoft/Phi-3",)


def _needs_trust_remote_code(model_id: str) -> bool:
    return any(model_id.startswith(p) for p in TRUST_REMOTE_CODE_PREFIXES)


def _default_lora_targets(model_id: str) -> list:
    mid = model_id.lower()
    if "phi"   in mid: return DEFAULT_LORA_TARGETS["phi"]
    if "llama" in mid: return DEFAULT_LORA_TARGETS["llama"]
    return DEFAULT_LORA_TARGETS["gemma"]


# ---------------------------------------------------------------------------
# Script arguments
# ---------------------------------------------------------------------------

@dataclass
class ScriptArguments:
    dataset_path:        str   = field(default=None,                     metadata={"help": "Dir with train_dataset.json"})
    model_id:            str   = field(default="google/gemma-1.1-7b-it", metadata={"help": "HuggingFace model ID"})
    hf_token:            str   = field(default=None,                     metadata={"help": "HuggingFace API token"})
    hf_username:         str   = field(default=None,                     metadata={"help": "HuggingFace username"})
    hf_repo_id:          str   = field(default=None,                     metadata={"help": "Full HF Hub repo id"})
    max_seq_length:      int   = field(default=2048,                     metadata={"help": "Max token sequence length"})
    training_mode:       str   = field(default="lora",                   metadata={"help": "lora | qlora | fft"})
    attention_impl:      str   = field(default="sdpa",                   metadata={"help": "sdpa | flash_attention_2"})
    lora_r:              int   = field(default=16,                       metadata={"help": "LoRA rank"})
    lora_alpha:          int   = field(default=32,                       metadata={"help": "LoRA alpha"})
    lora_dropout:        float = field(default=0.05,                     metadata={"help": "LoRA dropout"})
    peft_target_modules: list  = field(default=None,                     metadata={"help": "LoRA target modules (auto-detected if None)"})


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def training_function(script_args: ScriptArguments, training_args: SFTConfig) -> None:
    if script_args.hf_token:
        login(token=script_args.hf_token)

    # Dataset
    train_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, "train_dataset.json"),
        split="train",
    )
    print(f"Train samples: {len(train_dataset)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_id, use_fast=True,
        trust_remote_code=_needs_trust_remote_code(script_args.model_id),
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Log random samples
    with training_args.main_process_first(desc="Log random training samples"):
        for idx in random.sample(range(len(train_dataset)), min(2, len(train_dataset))):
            print(f"\n--- Training sample {idx} ---")
            print(train_dataset[idx][training_args.dataset_text_field][:600])
            print("…")

    # Model
    torch_dtype = torch.bfloat16
    quantization_config = (
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=torch_dtype,
        ) if script_args.training_mode == "qlora" else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        quantization_config=quantization_config,
        attn_implementation=script_args.attention_impl,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        trust_remote_code=_needs_trust_remote_code(script_args.model_id),
    )

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # LoRA / QLoRA
    if script_args.training_mode in ("lora", "qlora"):
        target_modules = script_args.peft_target_modules or _default_lora_targets(script_args.model_id)
        print(f"LoRA target modules: {target_modules}")
        peft_config = LoraConfig(
            lora_alpha=script_args.lora_alpha, lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r, bias="none",
            target_modules=target_modules, task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    # Trainer
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    if (
        trainer.accelerator.is_main_process
        and hasattr(trainer.model, "print_trainable_parameters")
    ):
        trainer.model.print_trainable_parameters()

    checkpoint = training_args.resume_from_checkpoint or None
    trainer.train(resume_from_checkpoint=checkpoint)

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()

    # Push to Hub
    if trainer.accelerator.is_main_process and script_args.hf_username:
        repo_id = script_args.hf_repo_id or (
            f"{script_args.hf_username}/"
            f"{os.path.basename(training_args.output_dir.rstrip('/'))}"
        )
        print(f"Pushing model to HuggingFace Hub: {repo_id}")
        trainer.model.push_to_hub(repo_id, private=True)
        tokenizer.push_to_hub(repo_id, private=True)
        print(f"Done → https://huggingface.co/{repo_id}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch as _torch
    if _torch.cuda.is_available() and _torch.cuda.get_device_capability()[0] >= 8:
        _torch.cuda.is_bf16_supported = lambda: True

    parser = TrlParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_and_config()

    if training_args.dataset_text_field is None:
        training_args.dataset_text_field = "text"

    training_args.max_seq_length = script_args.max_seq_length
    training_args.packing = False
    training_args.dataset_kwargs = {"add_special_tokens": False, "append_concat_token": False}

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    set_seed(training_args.seed)
    training_function(script_args, training_args)
