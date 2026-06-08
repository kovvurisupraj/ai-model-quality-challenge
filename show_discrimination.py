import json
import numpy as np
from pathlib import Path

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
                scores[row['index']] = float(row['sample_score']['score']['value'][score_field])
        all_scores[model] = scores
    indices = sorted(all_scores[MODELS[0]].keys())
    matrix = np.array([[all_scores[m][i] for m in MODELS] for i in indices])
    return matrix, indices

def compute_discrimination(matrix):
    model_ability = matrix.mean(axis=0)
    disc = np.zeros(matrix.shape[0])
    for i in range(matrix.shape[0]):
        q = matrix[i]
        if q.std() == 0:
            disc[i] = 0.0
        else:
            disc[i] = abs(np.corrcoef(q, model_ability)[0, 1])
    return disc

lcb_matrix, lcb_indices = load_score_matrix('live_code_bench_v5', 'pass')
lcb_disc = compute_discrimination(lcb_matrix)

ranked = np.argsort(lcb_disc)[::-1]

print("Top 63 LCB questions by discrimination score:")
print(f"{'Rank':>5} {'Q index':>8} {'Disc score':>12} {'gpt':>6} {'kimi':>6} {'minimax':>8}")
print("-" * 55)
for rank, pos in enumerate(ranked[:63]):
    q_idx = lcb_indices[pos]
    disc = lcb_disc[pos]
    gpt = int(lcb_matrix[pos][0])
    kimi = int(lcb_matrix[pos][1])
    minimax = int(lcb_matrix[pos][2])
    print(f"{rank+1:>5} {q_idx:>8} {disc:>12.4f} {gpt:>6} {kimi:>6} {minimax:>8}")

print(f"\nDiscrimination score distribution in top 63:")
top63_disc = lcb_disc[ranked[:63]]
print(f"  Max:    {top63_disc.max():.4f}")
print(f"  Min:    {top63_disc.min():.4f}")
print(f"  Mean:   {top63_disc.mean():.4f}")
print(f"  = 1.0:  {(top63_disc == 1.0).sum()} questions")
print(f"  > 0.9:  {(top63_disc > 0.9).sum()} questions")
print(f"  > 0.5:  {(top63_disc > 0.5).sum()} questions")