# `data/predictions/` — Full model predictions

Full prediction trajectories for the two models we have not yet graded. Shipped so candidates can read the actual model outputs (reasoning traces, code attempts, tool calls) for stratification, contamination checks, or anything else where the truncated 2 KB `final_answer_text` in `data/slim/` isn't enough.

## Files

| File | Rows | Size |
|---|---|---|
| `live_code_bench_v5__kimi-k2.5.jsonl` | 315 | ~44 MB |
| `live_code_bench_v5__minimax-m2.5.jsonl` | 315 | ~41 MB |
| `aa_lcr__kimi-k2.5.jsonl` | 100 | ~2 MB |
| `aa_lcr__minimax-m2.5.jsonl` | 100 | ~1 MB |

`gpt-oss-120b` is not duplicated here — its full predictions plus scores are reachable through the slim file at `data/slim/{benchmark}__gpt-oss-120b.jsonl`.

## Schema (per row)

Each row is the model's raw inference trajectory with three bulky fields stripped (see "What was stripped" below). Top-level keys:

| Field | Type | Notes |
|---|---|---|
| `index` | int | Sample id. Aligns with `sample_id` in `data/slim/`. |
| `model` | str | `kimi-k2.5` or `minimax-m2.5`. |
| `model_output` | dict | The model's response. `choices[0].message.content` is the structured output (reasoning + text blocks). `usage` has token counts. |
| `messages` | list | The prompt as sent to the model. |
| `metadata` | dict | Per-sample metadata. For AA-LCR: `question`, `data_source_urls`, `input_tokens`. For LCB: typically empty (LCB metadata lives on HuggingFace `livecodebench/code_generation_lite`). |

To get the model's final text from one row:

```python
import json
rec = json.loads(line)
content = rec["model_output"]["choices"][0]["message"]["content"]
# content is a list of {"type": "reasoning"|"text", ...} blocks
text_blocks = [b["text"] for b in content if b.get("type") == "text"]
final = text_blocks[-1] if text_blocks else ""
```

## ⚠️ Scores are NOT included

These files contain only the model's outputs, not graded scores. The corresponding rows in `data/slim/` have `score=null`. If you need scores for stratification, either:

1. Use `gpt-oss-120b` (graded) as your reference and reason about the others by comparison.
2. Re-grade the predictions yourself (LCB has a deterministic sandbox grader; AA-LCR is LLM-judged).
3. Treat ungraded rows as a feature: how well does your strategy hold up when ground truth is missing for some models?

## What was stripped

To keep files under GitHub's 100 MB per-file limit, three fields are dropped from the raw trajectories:

- `model_output.choices[*].logprobs` — per-token logprob distributions (~360 KB/row).
- `model_output.choices[*].token_ids` — generated token ids (~5 KB/row).
- `model_output.prompt_token_ids` — tokenized prompt (~400 KB/row).

This brought files from 100 MB – 2.5 GB down to 1 – 44 MB (~1.7% of original). The model's actual response text, reasoning trace, tool calls, and the prompt that produced them are all preserved.

## Regeneration

Sophie regenerates this data with `python scripts/copy_predictions.py`. Candidates do not run this script.
