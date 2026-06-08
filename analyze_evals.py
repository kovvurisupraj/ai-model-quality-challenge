import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

REVIEWS_DIR = Path("Evals/Part 1/reviews")
MODELS = ['gpt-oss-120b', 'kimi-k2.5', 'minimax-m2.5']

def load_score_matrix(benchmark, score_field):
    all_scores = {}
    for model in MODELS:
        path = REVIEWS_DIR / f"{benchmark}__{model}.jsonl"
        with open(path) as f:
            scores = {}
            for line in f:
                row = json.loads(line)
                idx = row['index']
                val = row['sample_score']['score']['value'][score_field]
                scores[idx] = float(val)
        all_scores[model] = scores

    indices = sorted(all_scores[MODELS[0]].keys())
    matrix = np.array([
        [all_scores[m][i] for m in MODELS]
        for i in indices
    ])
    return matrix, indices

print("Loading scores...")
lcb_matrix, lcb_indices = load_score_matrix('live_code_bench_v5', 'pass')
aalcr_matrix, aalcr_indices = load_score_matrix('aa_lcr', 'acc')

print(f"LCB matrix shape:    {lcb_matrix.shape}")
print(f"AA-LCR matrix shape: {aalcr_matrix.shape}")

print("\n=== Model scores (full benchmark) ===")
for i, m in enumerate(MODELS):
    print(f"  {m:25s}  LCB={lcb_matrix[:,i].mean():.3f}  AA-LCR={aalcr_matrix[:,i].mean():.3f}")

def compute_discrimination(matrix):
    model_ability = matrix.mean(axis=0)
    n_questions = matrix.shape[0]
    disc = np.zeros(n_questions)
    for i in range(n_questions):
        q_scores = matrix[i]
        if q_scores.std() == 0:
            disc[i] = 0.0
        else:
            corr = np.corrcoef(q_scores, model_ability)[0, 1]
            disc[i] = abs(corr)
    return disc

lcb_disc = compute_discrimination(lcb_matrix)
aalcr_disc = compute_discrimination(aalcr_matrix)

print(f"\n=== LCB discrimination stats ===")
print(f"  Zero discrimination: {(lcb_disc == 0).sum()} questions")
print(f"  Mean: {lcb_disc.mean():.3f}  Max: {lcb_disc.max():.3f}")

print(f"\n=== AA-LCR discrimination stats ===")
print(f"  Zero discrimination: {(aalcr_disc == 0).sum()} questions")
print(f"  Mean: {aalcr_disc.mean():.3f}  Max: {aalcr_disc.max():.3f}")

def model_ranking(matrix):
    return list(np.argsort(matrix.mean(axis=0))[::-1])

def rank_correlation(matrix, kept):
    corr, _ = spearmanr(matrix.mean(axis=0), matrix[kept].mean(axis=0))
    return corr

print("\n=== Rank preservation vs prune ratio ===")
print(f"{'Benchmark':<12} {'Keep%':>6} {'N kept':>7} {'Rank corr':>10} {'Ranking matches?':>17}")
print("-" * 60)

for benchmark, matrix, disc, total in [
    ('LCB',    lcb_matrix,   lcb_disc,   315),
    ('AA-LCR', aalcr_matrix, aalcr_disc, 100),
]:
    full_ranking = model_ranking(matrix)
    ranked_by_disc = np.argsort(disc)[::-1]
    for ratio in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        n_keep = max(2, int(total * ratio))
        kept = sorted(ranked_by_disc[:n_keep].tolist())
        corr = rank_correlation(matrix, kept)
        pruned_ranking = model_ranking(matrix[kept])
        matches = "YES" if pruned_ranking == full_ranking else "NO"
        print(f"{benchmark:<12} {ratio*100:>5.0f}% {n_keep:>7} {corr:>10.3f} {matches:>17}")
    print()

print("=== Saving score matrices and kept indices ===")
for benchmark, matrix, disc, indices, ratio in [
    ('LCB',    lcb_matrix,   lcb_disc,   lcb_indices,   0.20),
    ('AA-LCR', aalcr_matrix, aalcr_disc, aalcr_indices, 0.30),
]:
    total  = len(indices)
    n_keep = int(total * ratio)
    ranked = np.argsort(disc)[::-1]
    kept_pos = sorted(ranked[:n_keep].tolist())
    kept_idx = [indices[p] for p in kept_pos]
    key = benchmark.lower().replace('-','_')
    np.save(f"{key}_score_matrix.npy", matrix)
    np.save(f"{key}_kept_indices.npy", np.array(kept_idx))
    print(f"  {benchmark}: keeping {n_keep}/{total} questions → saved {key}_kept_indices.npy")

print("\nDone.")