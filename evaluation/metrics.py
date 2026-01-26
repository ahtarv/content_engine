import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def intra_list_similarity(vectors):
    """
    Average pairwisse cosine similarity inside a ranked list.
    Lower = more diverse
    """

    if len(vectors) < 2:
        return 0.0

    sims = cosine_similarity(vectors)
    n = sims.shape[0]

    return float(
        np.sum(np.triu(sims, k = 1)) / (n * (n - 1)/ 2)
        
    )