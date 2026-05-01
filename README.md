# Fine-tuning LLMs for Instruction-Data Separation

This repository contains our fine-tuning pipeline for training LLMs to resist **Indirect Prompt Injection** attacks, built on top of the [SEP benchmark](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed) (Zverev et al., ICLR 2025).

We train models to output a structured JSON response that explicitly names any injected instruction hidden in the data, making prompt injection attempts transparent and neutralised.

**Models trained and evaluated:**
- `google/gemma-1.1-2b-it`
- `google/gemma-1.1-7b-it`
- `meta-llama/Meta-Llama-3-8B-Instruct`

---

## Repository structure

```
fine-tuning/
│
├── original_paper/             ← original SEP benchmark code & dataset (git submodule)
│
├── 1_generate_targets.py       ← generate training labels via Gemma-3-27B API
├── 2_prepare_dataset.py        ← format data with model-specific chat templates
├── 3_train.py                  ← QLoRA / LoRA fine-tuning (via accelerate)
├── 4_evaluate.py               ← evaluate on SEP benchmark
├── run_training.py             ← convenience launcher (installs deps, handles logins)
│
├── configs/
│   ├── gemma1.1_2b.yaml        ← training hyper-params for Gemma-1.1-2B
│   ├── gemma1.1_7b.yaml        ← training hyper-params for Gemma-1.1-7B
│   └── llama3_8b.yaml          ← training hyper-params for Llama-3-8B
│
└── datasets/                   ← generated data (gitignored)
    ├── train_dataset_with_targets.json   (Step 1 output)
    └── prepared/
        ├── gemma/              (Step 2 output for Gemma models)
        └── llama3/             (Step 2 output for Llama-3)
```

---

## Setup

### Clone with submodule

```bash
git clone --recurse-submodules https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd fine-tuning
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

### Install dependencies

```bash
pip install torch transformers trl peft accelerate bitsandbytes \
            datasets huggingface_hub wandb google-generativeai tqdm fire
```

### Required credentials

| Credential | Where to get it | Used in |
|---|---|---|
| `GEMINI_API_KEY` | [ai.google.dev](https://ai.google.dev) | Step 1 only |
| HuggingFace token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Steps 3, 4 |
| Weights & Biases key | [wandb.ai](https://wandb.ai) | Step 3 (optional) |

---

## Step-by-step reproduction

### Step 1 — Generate training targets

Calls Gemma-3-27B via the Gemini API to produce ground-truth labels for ~2 000 training samples from the SEP dataset.

```bash
export GEMINI_API_KEY="your-key"

python 1_generate_targets.py
# Reads:   original_paper/datasets/train_dataset.json
# Writes:  datasets/train_dataset_with_targets.json
```

> **Skip this step** if `datasets/train_dataset_with_targets.json` already exists.

---

### Step 2 — Prepare the dataset

Formats the annotated data with the correct chat template for your target model.

```bash
# Gemma-1.1 (2B or 7B — same template)
python 2_prepare_dataset.py \
    --model_type gemma-1.1 \
    --out_dir    datasets/prepared/gemma

# Llama-3-8B
python 2_prepare_dataset.py \
    --model_type Llama-3 \
    --out_dir    datasets/prepared/llama3
```

Each source entry produces up to 4 training samples:

| Type | Description |
|---|---|
| `injected` | Probe hidden in `<data>` — model must detect and ignore it |
| `multi` | Both real task and probe listed in `<task>` |
| `clean` | Single instruction, clean data, no injection |
| `probe` | Probe is the sole instruction, no `<data>` section |

---

### Step 3 — Fine-tune

#### Option A: Convenience launcher (recommended)

Handles pip installs, W&B and HuggingFace logins automatically.

```bash
# Gemma-1.1-2B
python run_training.py \
    --hf_token    YOUR_HF_TOKEN \
    --hf_username YOUR_HF_USERNAME \
    --wandb_key   YOUR_WANDB_KEY \
    --model_id    google/gemma-1.1-2b-it \
    --output_dir  ./checkpoints/gemma1.1-2b

# Gemma-1.1-7B
python run_training.py \
    --hf_token    YOUR_HF_TOKEN \
    --hf_username YOUR_HF_USERNAME \
    --wandb_key   YOUR_WANDB_KEY \
    --model_id    google/gemma-1.1-7b-it \
    --output_dir  ./checkpoints/gemma1.1-7b

# Llama-3-8B
python run_training.py \
    --hf_token    YOUR_HF_TOKEN \
    --hf_username YOUR_HF_USERNAME \
    --wandb_key   YOUR_WANDB_KEY \
    --model_id    meta-llama/Meta-Llama-3-8B-Instruct \
    --output_dir  ./checkpoints/llama3-8b
```

The launcher auto-selects the right config YAML from `configs/` based on the model name.

#### Option B: Direct accelerate launch

```bash
accelerate launch 3_train.py \
    --config        configs/gemma1.1_7b.yaml \
    --training_mode qlora \
    --model_id      google/gemma-1.1-7b-it \
    --output_dir    ./checkpoints/gemma1.1-7b \
    --hf_token      YOUR_HF_TOKEN \
    --hf_username   YOUR_HF_USERNAME
```

#### Smoke-test (5 steps)

```bash
python run_training.py ... --test
```

---

### Step 4 — Evaluate

```bash
# Gemma-1.1-2B
python 4_evaluate.py \
    --base_model_id   google/gemma-1.1-2b-it \
    --checkpoint_path YOUR_HF_USERNAME/gemma1.1-2b \
    --hf_token        YOUR_HF_TOKEN

# Gemma-1.1-7B
python 4_evaluate.py \
    --base_model_id   google/gemma-1.1-7b-it \
    --checkpoint_path YOUR_HF_USERNAME/gemma1.1-7b \
    --hf_token        YOUR_HF_TOKEN

# Llama-3-8B
python 4_evaluate.py \
    --base_model_id   meta-llama/Meta-Llama-3-8B-Instruct \
    --checkpoint_path YOUR_HF_USERNAME/llama3-8b \
    --hf_token        YOUR_HF_TOKEN
```

Results are saved to `eval_results/results_sep.json`.

| Metric | Description |
|---|---|
| **SEP score** | Fraction of samples where the model answers correctly AND ignores the injection |
| **Utility** | How often the model answers the real task correctly |
| **ASR** | Attack Success Rate — how often the injected probe is executed |

---

## Training details

| Setting | Value |
|---|---|
| Method | QLoRA (NF4, 4-bit) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Epochs | 2 |
| Learning rate | 2e-4 (constant schedule) |
| Precision | bfloat16 |
| Seq length | 2048 (Gemma), 3072 (Llama-3) |

---

## How the training objective works

The model is trained to output a strict JSON object:

```json
{
  "fake_instruction": "<injected command found in <data>, or null>",
  "real_instruction": "<the actual task from <task>>",
  "final_answer_to_task": "<answer to the real task>"
}
```

By explicitly requiring the model to name any injection it finds in `fake_instruction`, we force it to reason about the instruction/data boundary — making it significantly more robust at inference time.

---

## Citation

Please cite the original SEP benchmark paper if you use this work:

```bibtex
@inproceedings{zverev2025,
  title     = {Can {LLMs} Separate Instructions From Data? And What Do We Even Mean By That?},
  author    = {Egor Zverev and Sahar Abdelnabi and Soroush Tabesh and Mario Fritz and Christoph H. Lampert},
  booktitle = {ICLR},
  year      = {2025}
}
```
