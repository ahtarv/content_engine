import numpy as np
import pickle
import sqlite3

def cosine(a,b):
    return float(np.dot(a,b))


def rerank(results, user_profile, alpha = 0.7, beta = 0.5):
    reranked = []

    for item in results:
        score = item["score"]

        if user_profile.interest_vector is not None:
            score += beta * cosine(item["vec"], user_profile.interest_vector)

        if user_profile.avoidance_vector is not None:
            score -= alpha * cosine(item["vec"], user_profile.avoidance_vector)

        new_item = item.copy()
        new_item["final_score"] = score
        reranked.append(new_item)

    return sorted(reranked, key = lambda x: x["final_score"], reverse = True)
