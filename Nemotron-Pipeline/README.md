# Nemotron Pipeline (HF + PEFT)

This pipeline is focused on stable, reproducible LoRA training for the NVIDIA Nemotron reasoning challenge.

## Features

### Data Augmentation
- **Family-specific system prompts** - Tailored instructions for bit, cipher, roman, gravity, unit, symbol tasks
- **Chain-of-Thought (CoT) templates** - Per-family reasoning patterns
- **Example shuffling** - Randomize example order within prompts
- **Example dropping** - Remove random examples for robustness
- **14+ prompt wrappers** - Diverse instruction variations
- **Difficulty estimation** - Score problems by complexity

### Training Enhancements
- **Configurable LoRA** - Rank, alpha, dropout all adjustable
- **Multiple LR schedulers** - Cosine, linear, constant, cosine with restarts
- **Curriculum learning** - Train easy → hard
- **Family balancing** - Oversample underrepresented families
- **Early stopping** - Stop on validation loss plateau

## 1) Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## 2) Build SFT data

Default with all augmentations:

```bash
python build_training_data.py \
  --train-csv train.csv \
  --out-dir data \
  --val-ratio 0.08 \
  --augment-factor 3 \
  --split-mode grouped
```

Fast iteration (limited data):

```bash
python build_training_data.py \
  --train-csv train.csv \
  --out-dir data \
  --val-ratio 0.08 \
  --augment-factor 2 \
  --split-mode grouped \
  --max-train-rows 5000
```

Disable specific augmentations:

```bash
python build_training_data.py \
  --train-csv train.csv \
  --out-dir data \
  --no-cot \
  --no-shuffle \
  --no-drop
```

Outputs:
- `data/train_sft.jsonl`
- `data/val_sft.jsonl`

## 3) Train rank-32 LoRA

Basic training:

```bash
python train_lora.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --train-jsonl data/train_sft.jsonl \
  --val-jsonl data/val_sft.jsonl \
  --out-dir outputs/run1 \
  --max-len 2048 \
  --lr 1.5e-4 \
  --epochs 1.0 \
  --batch-size 1 \
  --grad-accum 16
```

With curriculum learning and family balancing:

```bash
python train_lora.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --train-jsonl data/train_sft.jsonl \
  --val-jsonl data/val_sft.jsonl \
  --out-dir outputs/curriculum_run \
  --curriculum \
  --balance-families \
  --early-stopping
```

Custom LoRA configuration:

```bash
python train_lora.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --train-jsonl data/train_sft.jsonl \
  --val-jsonl data/val_sft.jsonl \
  --out-dir outputs/custom_lora \
  --lora-rank 32 \
  --lora-alpha 96 \
  --lora-dropout 0.03 \
  --lr-scheduler cosine
```

Fast profile:

```bash
python train_lora.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --train-jsonl data/train_sft.jsonl \
  --val-jsonl data/val_sft.jsonl \
  --out-dir outputs/fast_run \
  --profile fast \
  --max-train-samples 4000 \
  --max-eval-samples 300
```

## 4) Local eval (metric-aligned extraction)

```bash
python eval_local.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --adapter-path outputs/run1/adapter \
  --val-jsonl data/val_sft.jsonl
```

## 5) Hyperparameter sweep

Fast sweep (3 configs):

```bash
python run_fast_sweep.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --python-bin python \
  --workspace . \
  --pipeline-dir . \
  --sweep-mode fast
```

Full sweep (9 configs):

```bash
python run_fast_sweep.py \
  --model-path /path/to/nemotron-3-nano-30b \
  --python-bin python \
  --workspace . \
  --pipeline-dir . \
  --sweep-mode full
```

Summary file:
- `outputs/fast_sweep_results.json`

## 6) Kaggle accelerator run

If you do not have local CUDA, run on Kaggle GPU.

```bash
python /kaggle/working/Nemotron-Pipeline/run_on_kaggle.py \
  --workspace /kaggle/working \
  --pipeline-dir /kaggle/working/Nemotron-Pipeline
```

If auto-discovery fails, pass explicit paths:

```bash
python /kaggle/working/Nemotron-Pipeline/run_on_kaggle.py \
  --workspace /kaggle/working \
  --pipeline-dir /kaggle/working/Nemotron-Pipeline \
  --train-csv /kaggle/input/.../train.csv \
  --model-path /kaggle/input/.../model
```

## 7) Package submission

```bash
python package_submission.py \
  --adapter-dir outputs/run1/adapter \
  --out-zip submission.zip
```

The zip includes only:
- `adapter_config.json`
- `adapter_model.safetensors`

## Competition Constraints

| Parameter | Limit |
|-----------|-------|
| max_lora_rank | 32 |
| max_tokens | 7680 |
| max_model_len | 8192 |
| temperature | 0.0 |
| top_p | 1.0 |

## Problem Families

| Family | Detection Pattern |
|--------|-------------------|
| bit | "bit manipulation rule" |
| cipher | "secret encryption rules" |
| roman | "numeral system" |
| gravity | "gravitational constant" or "d = 0.5*g*t^2" |
| unit | "unit conversion" |
| symbol | "transformation rules is applied to equations" |
| other | Default fallback |
