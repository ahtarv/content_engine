import sqlite3
import pickle
from models.user_model import UserProfile


def load_user_profile(user_id: str, db_path="data/content.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
    SELECT a.embedding, i.dwell_time, i.scroll_depth
    FROM interactions i
    JOIN articles a ON i.article_id = a.id
    WHERE i.user_id = ?
    """, (user_id,))

    user = UserProfile()

    for embedding_blob, dwell, scroll in cur.fetchall():
        embedding = pickle.loads(embedding_blob)
        user.update(embedding, dwell, scroll)

    conn.close()
    return user
