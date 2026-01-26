import sqlite3
def log_interaction(user_id, article_id, dwell_time, scroll_depth, db_path = "data/content.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO interactions (user_id, article_id, dwell_time, scroll_depth)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, article_id, dwell_time, scroll_depth)
    )

    conn.commit()
    conn.close()