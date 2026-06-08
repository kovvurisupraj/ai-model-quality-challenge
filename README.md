# AI Model Quality Challenge: Submission

## Task 1: Performance Dashboard

**Live URL:** https://cerebras-dashboard.vercel.app/

**Video Walkthrough:** https://drive.google.com/file/d/1TcaditcNezHmJQeNWVG0CvjoQl8CbFeF/view?usp=sharing

**Code:** https://github.com/kovvurisupraj/Cerebras_Dashboard

## Task 2: Benchmark Compression

### Evalscope Fork

**Code:** https://github.com/kovvurisupraj/evalscope

**Evalscope commit SHA developed against:** `c14dbaf`

### What Was Built

Pruned LiveCodeBench v5 and AA-LCR benchmarks using IRT discrimination scoring. Reduces 415 total questions to 93 while preserving the exact model ranking (rank correlation = 1.0).

| Benchmark | Original | Pruned | Reduction |
|---|---|---|---|
| LiveCodeBench v5 | 315 | 63 | 80% |
| AA-LCR | 100 | 30 | 70% |
| Total | 415 | 93 | 78% |

### How to Run

```bash
git clone https://github.com/kovvurisupraj/evalscope.git
cd evalscope && python3 -m pip install -e .

# Pruned LiveCodeBench (63 questions instead of 315)
evalscope eval --model <model_id> --datasets live_code_bench_pruned \
  --dataset-args '{"prune_ratio": 0.2}'

# Pruned AA-LCR (30 questions instead of 100)
evalscope eval --model <model_id> --datasets aa_lcr_pruned \
  --dataset-args '{"prune_ratio": 0.3}'
```

### Files in This Repo

| File | Description |
|---|---|
| `handout_a.md` | Technical write-up covering why this works, pruning results, Part B design, and assumptions |
| `handout_b.md` | Non-technical write-up covering what this changes for customer conversations and how to use it |
| `Task2_Cerebras_Benchmark_Compression.pptx` | Presentation slides summarising the full approach |
| `analyze_evals.py` | Builds score matrices and validates pruning from the review JSONL files |
| `show_discrimination.py` | Shows discrimination scores per question in rank order |
| `lcb_kept_indices.npy` | The 63 LCB question indices selected by the pruner |
| `lcb_score_matrix.npy` | Full 315x3 score matrix (questions x models) |
| `aa_lcr_kept_indices.npy` | The 30 AA-LCR question indices selected by the pruner |
| `aa_lcr_score_matrix.npy` | Full 100x3 score matrix |

### Calibration Models

gpt-oss-120b, kimi-k2.5, minimax-m2.5
