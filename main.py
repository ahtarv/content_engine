import pickle
import argparse
from storage.db import init_db, get_conn
from ingest.load_articles import ingest_arxiv
from models.user_model import UserProfile
from ranking.personalized_ranker import rerank
from ranking.mmr import mmr_rerank

QUERY_NAME = "demo_user"
MMR_LAMBDA = 0.6
FINAL_K = 5

def parse_args():
    parser = argparse.ArgumentParser(
        description = "Content Discovery Engine CLI"
    )

    parser.add_argument(
        "--query",
        type=str,
        default = "fairness in ranking",
        help = "Search Query (used for future extensions)"
    )

    parser.add_argument(
        "--like",
        type=int,
        default = 0,
        help = "Index of item user reads deeply"
    )

    parser.add_argument(
        "--skip",
        type=int,
        default=1,
        help="Index of item the user skips"
    )

    parser.add_argument(
        "--k",
        type = int,
        default = 5,
        help = "Number of final results"
    )

    parser.add_argument(
        "--lambda",
        dest = "lambda_",
        type=float,
        default=0.6,
        help = "MMR diversity weight (0=max diversity, 1 = max relevance)"
    )

    return parser.parse_args()

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

def simulate_user(items, like_idx, skip_idx):
    user = UserProfile()

    max_index = max(like_idx, skip_idx)

    if len(items) > max_index:
        print(f"    User READS item #{like_idx}")
        user.update(
            items[like_idx]["vec"],
            dwell_time=45,
            scroll_depth=0.8
        )

        print(f"    User SKIPS item #{skip_idx}")
        user.update(
            items[skip_idx]["vec"],
            dwell_time=5,
            scroll_depth=0.1
        )
    else:
        print("    Not enough items to simulate interactions.")

    return user


def main():
    print("Content Discovery Engine Demo")

    args = parse_args()
    print(
    f"CLI settings → "
    f"like={args.like}, "
    f"skip={args.skip}, "
    f"k={args.k}, "
    f"lambda={args.lambda_}"
)

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

    user = simulate_user(items, args.like, args.skip)
    print(
        f"User vectors -> "
        f"Interest = {user.interest_vector is not None}, "
        f"Avoidance = {user.avoidance_vector is not None}"
    )

    ranked = rerank(items, user)

    print("\nTop Personalized Results: ")
    for i, r in enumerate(ranked[:3], 1):
        print(f"{i}. [{r['final_score']:.4f}] {r['title'][:70]}")

    diversified = mmr_rerank(ranked, k=args.k, lambda_=args.lambda_)

    print("\nFinal Results (MMR Diversified):")
    for i, r in enumerate(diversified, 1):
        print(f"{i}. [{r['final_score']:.4f}] {r['title'][:70]}")

if __name__ == "__main__":
    main()