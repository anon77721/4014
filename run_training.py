"""
run_training.py
===============
Convenience launcher — installs dependencies, logs in, runs 3_train.py.

Automatically selects the right config YAML from configs/ based on model_id.
All paths are relative to the fine-tuning/ repo root.

Usage
-----
    python run_training.py \
        --hf_token    YOUR_HF_TOKEN \
        --hf_username YOUR_HF_USERNAME \
        --wandb_key   YOUR_WANDB_KEY \
        --model_id    google/gemma-1.1-7b-it \
        --output_dir  ./checkpoints/gemma1.1-7b

    # Llama-3
    python run_training.py \
        --hf_token    YOUR_HF_TOKEN \
        --hf_username YOUR_HF_USERNAME \
        --wandb_key   YOUR_WANDB_KEY \
        --model_id    meta-llama/Meta-Llama-3-8B-Instruct \
        --output_dir  ./checkpoints/llama3-8b

    # Smoke-test (5 steps)
    python run_training.py ... --test
"""

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Training launcher for 3_train.py")

    p.add_argument("--hf_token",    required=True, help="HuggingFace API token")
    p.add_argument("--hf_username", required=True, help="HuggingFace username")
    p.add_argument("--wandb_key",   required=True, help="Weights & Biases API key")
    p.add_argument("--model_id",    required=True, help="HF model id")
    p.add_argument("--output_dir",  required=True, help="Checkpoint output directory")

    p.add_argument("--config",            default=None,
                   help="Training YAML (auto-selected if omitted)")
    p.add_argument("--dataset_path",      default=None,
                   help="Dataset dir (overrides YAML value)")
    p.add_argument("--accelerate_config", default=None,
                   help="Accelerate config yaml (omit for single-GPU)")
    p.add_argument("--training_mode",     default="qlora",
                   help="lora | qlora | fft  (default: qlora)")
    p.add_argument("--wandb_project",     default="fine-tuning",
                   help="W&B project name")
    p.add_argument("--hf_repo_id",        default=None,
                   help="Full HF repo id (auto-derived if omitted)")
    p.add_argument("--test",              action="store_true",
                   help="Smoke-test: 5 steps only")

    return p.parse_args()


def run(cmd: list, **kwargs) -> None:
    print(f"\n>>> {' '.join(cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        sys.exit(result.returncode)


def auto_config(model_id: str) -> str:
    mid  = model_id.lower()
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
    if "llama" in mid: return os.path.join(base, "llama3_8b.yaml")
    if "7b"    in mid: return os.path.join(base, "gemma1.1_7b.yaml")
    return os.path.join(base, "gemma1.1_2b.yaml")


def main() -> None:
    args = parse_args()

    # 1. Install dependencies
    print("=" * 60)
    print("Installing dependencies …")
    print("=" * 60)
    packages = [
        "torch", "torchvision", "torchaudio",
        "datasets", "transformers", "trl", "peft",
        "accelerate", "bitsandbytes", "huggingface_hub",
        "numpy", "pandas", "scipy", "tqdm", "wandb", "fire",
    ]
    run([sys.executable, "-m", "pip", "install", "--upgrade", "--quiet"] + packages)

    # 2. W&B
    print("=" * 60)
    print("Configuring W&B …")
    print("=" * 60)
    import wandb
    wandb.login(key=args.wandb_key)
    os.environ["WANDB_PROJECT"]  = args.wandb_project
    os.environ["WANDB_DISABLED"] = "false"

    # 3. HF login
    print("=" * 60)
    print("Logging in to HuggingFace …")
    print("=" * 60)
    from huggingface_hub import login
    login(token=args.hf_token)

    # 4. Launch
    print("=" * 60)
    print("Launching training …")
    print("=" * 60)
    import shutil
    accelerate_bin   = shutil.which("accelerate") or os.path.join(os.path.dirname(sys.executable), "accelerate")
    training_script  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3_train.py")
    config_yaml      = args.config or auto_config(args.model_id)

    cmd = [accelerate_bin, "launch"]
    if args.accelerate_config:
        cmd += ["--config_file", args.accelerate_config]

    cmd += [
        training_script,
        "--config",        config_yaml,
        "--training_mode", args.training_mode,
        "--model_id",      args.model_id,
        "--output_dir",    args.output_dir,
        "--hf_token",      args.hf_token,
        "--hf_username",   args.hf_username,
    ]

    if args.dataset_path:  cmd += ["--dataset_path", args.dataset_path]
    if args.hf_repo_id:    cmd += ["--hf_repo_id",   args.hf_repo_id]
    if args.test:
        cmd += ["--max_steps", "5"]
        print("[TEST MODE] 5 steps only.")

    run(cmd)

    repo_id = args.hf_repo_id or f"{args.hf_username}/{os.path.basename(args.output_dir.rstrip('/'))}"
    print("\n" + "=" * 60)
    print(f"Done! Model → https://huggingface.co/{repo_id}")
    print("=" * 60)


if __name__ == "__main__":
    main()
