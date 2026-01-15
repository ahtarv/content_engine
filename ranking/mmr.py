import numpy as np
def cosine(a,b):
    return float(np.dot(a,b))

def mmr_rerank(items, k=10, lambda_=0.7):
    """
    items: list od fict with keys:
        -vec (embedding)
        -final_score
    """

    if not items:
        return []

    selected = []
    candidates = items.copy()

    selected.append(candidates.pop(0))

    while candidates and len(selected) < k:
        best_item = None
        best_score = -1e9

        for item in candidates:
            relevance = item["final_score"]
            diversity_penalty = max(cosine(item["vec"], s["vec"]) for s in selected)

            score = lambda_ * relevance - (1 - lambda_) * diversity_penalty
            if score > best_score:
                best_item = item
                best_score = score

        selected.append(best_item)
        candidates.remove(best_item)

    return selected 