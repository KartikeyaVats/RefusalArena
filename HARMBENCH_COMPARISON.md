# CLS × HarmBench — Apples-to-Apples Comparison Guide

This document explains how to run Contrastive Logit Steering (CLS) through
HarmBench's standardized evaluation pipeline so its ASR numbers are directly
comparable to Arditi et al. ("Refusal in Language Models Is Mediated by a
Single Direction", NeurIPS 2024) and any other method already in HarmBench.

---

## Why this matters

| Axis | Arditi et al. paper | CLS repo (original) |
|---|---|---|
| Mechanism | Activation ablation (residual stream) | Logit-space steering |
| Prompt dataset | AdvBench + MaliciousInstruct + TDC2023 + HarmBench | AdvBench only |
| Judge | JailbreakBench classifier | Llama-3-8B-Instruct |
| Decoding | Greedy, 512 tokens | Multinomial temp=0.7, 150 tokens |
| Teacher forcing | None | "Sure" prefix |

Running both through HarmBench normalizes **dataset**, **judge**, **token budget**,
and **decoding** in one shot, leaving only the mechanism as the variable.

---

## Setup

### 1. Clone HarmBench
```bash
git clone https://github.com/centerforaisafety/HarmBench
cd HarmBench
pip install -r requirements.txt
```

### 2. Install CLS dependencies (if not already done)
```bash
pip install transformers>=4.40.0 accelerate torch tqdm pandas numpy
```

### 3. Copy the adapter next to HarmBench
```bash
cp cls_harmbench_adapter.py ./
```

---

## Running the comparison

### Step A — Run CLS (Step 2 replacement)

The adapter produces a `completions.json` that HarmBench's Step 3 evaluator
reads directly. No Step 1 needed — CLS is a direct-prompt method.

```bash
# Llama-3.1 8B — greedy, alpha=3, 512 tokens (paper-matched settings)
python cls_harmbench_adapter.py \
  --model_name  meta-llama/Meta-Llama-3.1-8B-Instruct \
  --behaviors_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_path   results/cls_llama31_8b_a3_greedy.json \
  --alpha 3.0 \
  --max_new_tokens 512 \
  --teacher_forcing Sure \
  --template llama3 \
  --greedy

# Qwen-2.5 7B
python cls_harmbench_adapter.py \
  --model_name  Qwen/Qwen2.5-7B-Instruct \
  --behaviors_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_path   results/cls_qwen25_7b_a3_greedy.json \
  --alpha 3.0 \
  --max_new_tokens 512 \
  --teacher_forcing Sure \
  --template chatml \
  --greedy

# Gemma-3 4B
python cls_harmbench_adapter.py \
  --model_name  google/gemma-3-4b-it \
  --behaviors_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --save_path   results/cls_gemma3_4b_a3_greedy.json \
  --alpha 3.0 \
  --max_new_tokens 512 \
  --teacher_forcing Sure \
  --template chatml \
  --greedy
```

#### Alpha sweep (optional — reproduces CLS paper's Figure)
```bash
for ALPHA in 0.0 1.0 2.0 3.0 4.0 5.0; do
  python cls_harmbench_adapter.py \
    --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
    --behaviors_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
    --save_path results/cls_llama31_8b_a${ALPHA}_greedy.json \
    --alpha $ALPHA --max_new_tokens 512 --template llama3 --greedy
done
```

---

### Step B — Run Arditi et al.'s method through HarmBench (Step 2)

The Arditi et al. codebase (`github.com/andyrdt/refusal_direction`) includes
a `generate_completions.py` that writes the same JSON format. Follow their
README to run it — or use HarmBench's built-in DirectionAblation baseline
if it exists in your HarmBench version.

---

### Step C — Evaluate completions (HarmBench Step 3)

Run this for **every** completions file from both methods:

```bash
python HarmBench/evaluate_completions.py \
  --cls_path cais/HarmBench-Llama-2-13b-cls \
  --behaviors_path HarmBench/data/behavior_datasets/harmbench_behaviors_text_all.csv \
  --completions_path results/cls_llama31_8b_a3_greedy.json \
  --save_path results/cls_llama31_8b_a3_greedy_evaluated.json
```

The output JSON contains per-behavior verdicts and an overall ASR.

---

## Controlled ablations

To isolate what drives the difference between the two methods, run CLS with
progressively paper-matched settings:

| Run | `--greedy` | `--max_new_tokens` | `--teacher_forcing` |
|---|---|---|---|
| CLS-original | ❌ (temp=0.7) | 150 | Sure |
| CLS-greedy | ✅ | 150 | Sure |
| CLS-greedy-512 | ✅ | 512 | Sure |
| CLS-greedy-512-nopfx | ✅ | 512 | *(empty)* |

Comparing CLS-greedy-512-nopfx against the Arditi et al. numbers isolates the
**mechanism** (logit-space steering vs. activation ablation) from everything else.

---

## Template reference

| Model family | `--template` |
|---|---|
| Llama-3.x Instruct | `llama3` |
| Llama-2 Chat | `llama2` |
| Qwen / Qwen2.5 Instruct | `chatml` |
| Gemma-3 IT | `chatml` |

---

## Expected output structure

```
results/
├── cls_llama31_8b_a3_greedy.json          # Step 2 output (completions)
├── cls_llama31_8b_a3_greedy_evaluated.json # Step 3 output (ASR)
├── cls_qwen25_7b_a3_greedy.json
├── cls_qwen25_7b_a3_greedy_evaluated.json
└── ...
```
