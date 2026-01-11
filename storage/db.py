import sqlite3

def get_conn(db_path="data/content.db"):
    return sqlite3.connect(db_path)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        url TEXT UNIQUE,
        source TEXT,
        title TEXT,
        abstract TEXT,
        embedding BLOB,
        topics TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS interactions (
        user_id TEXT,
        article_id TEXT,
        dwell_time REAL,
        scroll_depth REAL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()
