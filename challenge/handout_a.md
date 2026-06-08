# Handout A: Why This Works
**Technical audience. evalscope fork: github.com/kovvurisupraj/evalscope, commit c14dbaf**

---

## What Problem We Understood Ourselves to Be Solving

The customer needs one answer: is this model good enough for code generation and long-context reasoning? The blocker is not correctness. It is that running 415 benchmark questions costs significant compute time. The real problem is identifying which questions carry discriminative signal that generalises to an unseen model, and keeping only those.

Random sampling does not solve this. We found that 65% of LCB questions and 57% of AA-LCR questions have zero variance across all three shipped models, meaning every model scored identically on them. These questions cannot distinguish a strong model from a weak one regardless of which new model is evaluated next.

## Approach: IRT Discrimination

For each question we compute the absolute Pearson correlation between the per-model pass/fail vector and the per-model overall ability score. A high score means the models that pass this question are consistently the stronger models overall. A zero score means all models scored the same and the question carries no signal.

`disc(i) = |corr(pass_fail_i, ability_vector)|`

This is model-agnostic by construction. The three shipped models are used only as a measuring instrument to characterise question properties, not to select questions that favour specific models. A question that is inherently discriminating will remain so for a fourth model from the same capability distribution.

## How Much Was Pruned and Why the Subset Is Sufficient

| Benchmark | Total | Zero Disc. | Kept | Reduction |
|---|---|---|---|---|
| LiveCodeBench v5 | 315 | 204 (65%) | 63 (20%) | 80% |
| AA-LCR | 100 | 57 (57%) | 30 (30%) | 70% |
| **Combined** | **415** | **261 (63%)** | **93** | **78%** |

Sufficiency was validated by sweeping prune ratios from 5% to 50% and checking whether each subset reproduced the full model ranking. For LCB, rank correlation = 1.0 first held at 20% (63 questions). Below that the ranking broke. For AA-LCR the safe minimum was 30%. At 40% the ranking broke (correlation = 0.5) and recovered at 50%. This is a direct consequence of LLM judge noise pulling borderline questions in and out of the kept set depending on which judged scores are available. We apply `noise_weight = 0.9` to deflate AA-LCR discrimination scores before ranking, reducing over-reliance on marginally discriminating questions whose scores are partly measurement noise.

## Part B: Multimodal Probe Design

The probe must surface image-encoder degradation specifically, not generic capability gaps. Roughly half of MMMU questions can be answered from text alone, so a model with a broken encoder scores around 50% on random MMMU samples by ignoring the image entirely.

**Selection strategy from the full 12K HuggingFace MMMU dataset:**

1. Filter for image-critical questions where the correct answer is unreachable without the image (charts, circuit diagrams, X-rays, microscopy, spectrographs)
2. Prioritise fine-grained visual tasks such as reading numerical values from axes, spatial reasoning, and identifying highlighted components
3. Cover diverse visual modalities including natural photos, technical diagrams, medical imaging, and text-in-image
4. Apply a subject diversity constraint of at least 3 questions per MMMU subject across all 30 subjects

**Measuring encoder quality via the OpenAI interface:** for each probe question, run it twice. First with the image attached. Second with the image replaced by a plain text description of what a human sees. The accuracy gap between these two conditions is the encoder signal. A degraded encoder collapses this gap toward zero because the model effectively stops using the image.

Target: 100 to 150 questions, roughly 1% of the dataset.

## Assumptions

- The three calibration models are representative of the general capability range likely to be encountered
- The LCB sandbox grader is deterministic and scores are treated as ground truth with no noise correction
- AA-LCR judge noise is independent across questions and a uniform noise_weight is applied rather than per-question modelling

## What Would Change With More Resources

**(a) More models:** discrimination scores would be far more reliable. The code accepts an arbitrary score matrix shape so adding a fourth model's review file and re-running the analysis script is a one-command update.

**(b) Live model endpoint:** generalisation could be validated directly by running a fourth model on both the full and pruned sets and comparing rankings empirically. Repeated AA-LCR judgements on the same questions would also allow per-question noise weights instead of a uniform multiplier.

**(c) More time:** ensemble pruning across multiple estimators (IRT, mutual information, classifier feature importance) with questions selected only when they rank highly across all methods. Adaptive prune ratio based on confidence intervals around the model score estimates.
