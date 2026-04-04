# Nemotron v2 — Real Chain-of-Thought Training Pipeline

LoRA fine-tuning for the NVIDIA Nemotron-3-Nano-30B reasoning challenge.

**What changed from v1 (score: 0.62):**

| | v1 (Kanye the Bully) | v2 (this) |
|---|---|---|
| CoT traces | Canned 3-line templates per family | Real, problem-specific traces with actual values |
| Solver accuracy | 0% (no solving) | 99.9% (9,493/9,500 puzzles solved) |
| LORA_ALPHA | 64 | 128 |
| MAX_SEQ_LEN | 1536 | 2048 |
| NUM_EPOCHS | 2 | 3 |
| GRAD_ACCUM | 8 | 4 |
| LR | 7e-5 | 5e-5 |
| LORA_DROPOUT | 0.05 | 0.03 |

## Files

```
Nemotron-v2/
  Nemotron_v2_Training.ipynb   # Complete Kaggle notebook (self-contained)
  solvers.py                   # Standalone solver module (for local testing)
  test_solvers.py              # Test solver accuracy against train.csv
  README.md                    # This file
```

## How to Run on Kaggle

### Step 1: Upload the notebook

1. Go to [kaggle.com/competitions/nvidia-nemotron-3-nano-30b-reasoning-challenge](https://www.kaggle.com/competitions/nvidia-nemotron-3-nano-30b-reasoning-challenge)
2. Click **Code** > **New Notebook**
3. Upload `Nemotron_v2_Training.ipynb` (or copy-paste cells)

### Step 2: Configure the notebook

1. **Accelerator**: Select **GPU RTX 6000 Blackwell**
2. **Data sources**: Add the competition dataset (`nvidia-nemotron-3-reasoning-challenge`)
3. **Add model**: Add `metric/nemotron-3-nano-30b-a3b-bf16` from Kaggle Models
4. **Add dataset**: Add `dennisfong/nvidia-nemotron-offline-packages` (for offline pip install)

### Step 3: Run all cells

- Cell 1-4: Setup (installs, imports, config, Triton fixes)
- Cell 5: Puzzle solvers load (~instant)
- Cell 6: Generates real CoT traces for all 9,500 puzzles (~1-2 min on CPU)
- Cell 7-9: Model loading + LoRA setup (~2-3 min)
- Cell 10: **Training** (~2-3 hours for 3 epochs on full dataset)
- Cell 11-13: Save adapter + package submission.zip

### Step 4: Submit

1. After all cells finish, click **Submit** in the notebook
2. Or download `submission.zip` and submit manually

## Local Testing (optional)

Test solver accuracy before uploading to Kaggle:

```bash
cd Nemotron-v2
python test_solvers.py ../Nemotron-Pipeline/train.csv
```

Expected output:
```
bit          1602    1595 ( 99.6%)
cipher       1576    1576 (100.0%)
gravity      1597    1597 (100.0%)
roman        1576    1576 (100.0%)
symbol       1555    1555 (100.0%)
unit         1594    1594 (100.0%)
TOTAL        9500    9493 ( 99.9%)
```

## Hyperparameter Tuning Guide

If you want to experiment further, these are the most impactful knobs:

| Parameter | Current | Try if... |
|---|---|---|
| `NUM_EPOCHS` | 3 | Loss still decreasing → increase to 4-5 |
| `LR` | 5e-5 | Loss diverging → try 2e-5; not converging → try 1e-4 |
| `LORA_ALPHA` | 128 | Overfitting → decrease to 64; underfitting → try 256 |
| `MAX_SEQ_LEN` | 2048 | OOM errors → decrease to 1536; plenty of VRAM → try 2560 |
| `GRAD_ACCUM` | 4 | Want larger effective batch → increase to 8 |

## How the Solvers Work

Each puzzle family has a dedicated solver:

- **Bit (99.6%)**: Tries XOR masks, rotations (left/right 1-7), NOT, bit reversal, nibble swap, combinations (NOT+rotate, rotate+XOR, reverse+XOR), and bit permutation detection
- **Cipher (100%)**: Builds letter-by-letter substitution table from all example pairs, fills gaps via bijection constraint
- **Roman (100%)**: Standard decimal-to-Roman algorithm
- **Gravity (100%)**: Computes g = 2d/t^2 from examples, applies d = 0.5*g*t^2
- **Unit (100%)**: Computes factor = output/input from examples, applies to test
- **Symbol (100%)**: Tries character substitution, constant/positional ASCII offsets, character filtering
