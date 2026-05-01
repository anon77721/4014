"""
1_generate_targets.py
=====================
Step 1 — Generate training targets using Gemma-3-27B via the Gemini API.

For each sample in the SEP training dataset this script calls Gemma-3-27B
twice to produce ground-truth answers that become fine-tuning targets:

  no-probe-res : answer to the real task (task_prompt + data_prompt, no probe)
  probe-res    : answer to the probe instruction alone

What this script does
---------------------
1. Loads  original_paper/datasets/train_dataset.json
2. Filters out samples with vague probe words (summarize, provide, text …)
3. Randomly samples up to 2 000 items
4. Calls Gemma-3-27B for each item (×2 per item)
5. Saves annotated samples → datasets/train_dataset_with_targets.json

Requirements
------------
    pip install google-generativeai tqdm

Environment
-----------
    export GEMINI_API_KEY="your-key-here"

Usage
-----
    # From inside fine-tuning/
    python 1_generate_targets.py

    # With explicit paths
    python 1_generate_targets.py \
        --input  original_paper/datasets/train_dataset.json \
        --output datasets/train_dataset_with_targets.json
"""

import argparse
import json
import os
import random
import time
from collections import deque

from tqdm import tqdm
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


# ---------------------------------------------------------------------------
# Rate limiter  (Gemini free-tier: 30 RPM / 15 000 TPM)
# ---------------------------------------------------------------------------

class RateLimiter:
    def __init__(self, max_rpm: int = 30, max_tpm: int = 15_000):
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.history: deque = deque()

    def _cleanup(self):
        now = time.time()
        while self.history and now - self.history[0][0] > 60.0:
            self.history.popleft()

    def _current_tokens(self):
        return sum(t for _, t in self.history)

    def wait_if_needed(self, estimated_tokens: int):
        self._cleanup()
        while (
            len(self.history) >= self.max_rpm
            or (self._current_tokens() + estimated_tokens) > self.max_tpm
        ):
            if not self.history:
                break
            sleep_time = 60.0 - (time.time() - self.history[0][0])
            if sleep_time > 0:
                tqdm.write(
                    f"Rate limit approaching (tokens: {self._current_tokens()}/{self.max_tpm}, "
                    f"requests: {len(self.history)}/{self.max_rpm}). "
                    f"Sleeping {sleep_time:.1f}s …"
                )
                time.sleep(sleep_time + 0.1)
            self._cleanup()

    def record_usage(self, actual_tokens: int):
        self.history.append((time.time(), actual_tokens))


# ---------------------------------------------------------------------------
# Gemini client
# ---------------------------------------------------------------------------

def build_model(max_output_tokens: int = 150):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Run:  export GEMINI_API_KEY='your-key-here'"
        )
    genai.configure(api_key=api_key)
    model   = genai.GenerativeModel("models/gemma-3-27b-it")
    gen_cfg = genai.GenerationConfig(temperature=0.0, max_output_tokens=max_output_tokens)
    return model, gen_cfg


def generate_with_limits(model, gen_cfg, limiter, prompt_text, max_output_tokens=150, max_retries=3):
    estimated_tokens = (len(prompt_text) // 4) + max_output_tokens
    for attempt in range(max_retries):
        try:
            limiter.wait_if_needed(estimated_tokens)
            response = model.generate_content(prompt_text, generation_config=gen_cfg)
            actual = (
                response.usage_metadata.total_token_count
                if response.usage_metadata else estimated_tokens
            )
            limiter.record_usage(actual)
            try:
                return response.text.strip()
            except ValueError:
                tqdm.write("Response blocked by safety filters.")
                return ""
        except ResourceExhausted:
            sleep_dur = 10 * (attempt + 1)
            tqdm.write(f"429 – backing off {sleep_dur}s …")
            time.sleep(sleep_dur)
        except Exception as exc:
            tqdm.write(f"API error: {exc}")
            return ""
    tqdm.write("Failed after max retries.")
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRICT_INSTRUCTIONS = (
    "INSTRUCTIONS: You are a strict, direct, and concise assistant. "
    "Answer queries in 3-5 sentences. "
    "Do NOT include explanations, justifications, conversational filler, "
    "or follow-up questions. "
    "Provide ONLY the exact answer requested.\n\n"
    "QUERY:\n"
)

FORBIDDEN_WORDS = ["provide", "summarise", "summarize", "text"]


def main():
    p = argparse.ArgumentParser(description="Generate fine-tuning targets via Gemma-3-27B")
    p.add_argument("--input",  default="original_paper/datasets/train_dataset.json")
    p.add_argument("--output", default="datasets/train_dataset_with_targets.json")
    p.add_argument("--sample_size",       type=int, default=2000)
    p.add_argument("--seed",              type=int, default=42)
    p.add_argument("--max_output_tokens", type=int, default=150)
    args = p.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input} …")
    with open(args.input, "r", encoding="utf-8") as f:
        full_dataset = json.load(f)
    print(f"  {len(full_dataset)} entries loaded.")

    filtered = [
        s for s in full_dataset
        if not any(w in s.get("info", {}).get("probe", "").lower() for w in FORBIDDEN_WORDS)
    ]
    print(f"  {len(filtered)} entries after filtering.")

    sample_size = min(args.sample_size, len(filtered))
    sampled     = random.sample(filtered, sample_size)
    print(f"  Sampling {sample_size} items.")

    model, gen_cfg = build_model(args.max_output_tokens)
    limiter        = RateLimiter()

    print("Generating targets …")
    for sample in tqdm(sampled, desc="Annotating"):
        info        = sample.get("info", {})
        task_prompt = info.get("task_prompt", "")
        data_prompt = info.get("data_prompt", "")
        probe       = info.get("probe", "")

        sample["info"]["no-probe-res"] = generate_with_limits(
            model, gen_cfg, limiter,
            f"{STRICT_INSTRUCTIONS}{task_prompt}\n\n{data_prompt}",
            args.max_output_tokens,
        )
        sample["info"]["probe-res"] = generate_with_limits(
            model, gen_cfg, limiter,
            f"{STRICT_INSTRUCTIONS}{probe}",
            args.max_output_tokens,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=4, ensure_ascii=False)
    print(f"\nSaved {len(sampled)} annotated samples → {args.output}")


if __name__ == "__main__":
    main()
