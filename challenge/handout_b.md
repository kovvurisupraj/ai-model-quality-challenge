# Handout B: Why This Matters and How to Use It
**For developers, test engineers, product, and customer-facing teams**

---

## What Changes for the Customer Conversation

Right now, when a prospect asks if your model is good enough for their workload, you run the full benchmark suite. That takes days and costs significant compute time. The conversation stalls while you wait.

With the pruner, the same question gets answered in roughly 20% of the time and cost. It skips the 322 questions that every model answers identically (questions that add cost but no information) and runs only the 93 that actually separate strong models from weak ones. The answer you give the customer is the same. The time to produce it is not.

## How to Run It Tomorrow

```bash
# One-time setup
git clone https://github.com/kovvurisupraj/evalscope.git
cd evalscope && python3 -m pip install -e .

# Run against any candidate model
evalscope eval --model <model_id> --datasets live_code_bench_pruned \
  --dataset-args '{"prune_ratio": 0.2}'

evalscope eval --model <model_id> --datasets aa_lcr_pruned \
  --dataset-args '{"prune_ratio": 0.3}'
```

Two commands. Two scores between 0 and 1. Compare against the reference scores for the three shipped models. Above the threshold your team has set, the answer is yes. Below it, the answer is no. There is no manual interpretation step.

## What the Multimodal Probe Gives That Random Sampling Cannot

A model with a broken image encoder can still score around 50% on random MMMU questions by ignoring the image and reasoning from text alone. Random sampling cannot tell you whether the model is bad at images or simply not looking at them at all.

The targeted probe selects only questions where the image is the answer, questions a human cannot solve without seeing the image. A low score on this probe means specifically that the image encoder is failing, not that the model is generally weak. That distinction matters for the next conversation: it tells your team exactly where the problem is and what kind of model to look for.

## Why a Customer-Facing PM Should Care

Speed closes deals. The pruner removes most of the wait between a prospect's technical question and your team's data-backed answer.

The result is also defensible in a way that ad hoc testing is not. When you tell a prospect their model passes the benchmark, you can show them which questions were used, why those questions were selected, and the validation data showing the result matches what the full benchmark would have returned. That is a conversation you can have with a technically sophisticated buyer.
