import pickle
from storage.db import init_db, get_conn
from ingest.load_articles import ingest_arxiv
from models.user_model import UserProfile
from ranking.personalized_ranker import rerank
from ranking.mmr import mmr_rerank

QUERY_NAME = "demo_user"
MMR_LAMBDA = 0.6
FINAL_K = 5

def load_items():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT title, abstract, embedding FROM articles")
    rows = cur.fetchall()
    conn.close()

    items = []
    for title, abstract, blob in rows:
        items.append({
            "title": title,
            "abstract": abstract,
            "vec": pickle.loads(blob),
            "score": 0.0
        })
    return items

def simulate_user(items):
    user = UserProfile()

    if len(items) < 2:
        return user

    user.update(items[0]["vec"], dwell_time = 45, scroll_depth=0.8)
    user.update(items[1]["vec"], dwell_time=5, scroll_depth=0.1)

    return user

def main():
    print("Content Discovery Engine Demo")

    init_db()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM articles")
    count = cur.fetchone()[0]
    conn.close()

    if count == 0:
        print("No articles found. Ingesting from arXiv...")
        ingest_arxiv()
    else:
        print(f"Found {count} articles.")

    items = load_items()
    print(f"Loaded {len(items)} items.")

    user = simulate_user(items)
    print(
        f"User vectors -> "
        f"Interest = {user.interest_vector is not None}, "
        f"Avoidance = {user.avoidance_vector is not None}"
    )

    
    ranked = rerank(items, user)

    print("\nTop Personalized Results: ")
    for i, r in enumerate(ranked[:3], 1):
        print(f"{i}. [{r['final_score']:.4f}] {r['title'][:70]}")

    diversified = mmr_rerank(ranked, k=FINAL_K, lambda_=MMR_LAMBDA)

    print("\nFinal Results (MMR Diversified):")
    for i, r in enumerate(diversified, 1):
        print(f"{i}. [{r['final_score']:.4f}] {r['title'][:70]}")

if __name__ == "__main__":
    main()