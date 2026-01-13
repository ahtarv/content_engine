import json
import pickle
from models.embedder import Embedder
from models.topic_classifier import classify_topics
from storage.db import get_conn
from ingest.arxiv import fetch_arxiv

embedder = Embedder()

def ingest_arxiv():
    conn = get_conn()
    cur = conn.cursor()

    articles  = fetch_arxiv(query = "fairness ranking", max_results = 25)

    for a in articles:
        text = f"{a['title']}\n{a['abstract']}"
        vec = embedder.encode(text)
        topics = classify_topics(text)
        
        cur.execute(""" INSERT OR IGNORE INTO articles 
        VALUES(?,?,?,?,?,?,?)""", (a['id'], a['url'], a['source'], a['title'], a['abstract'], pickle.dumps(vec),json.dumps(topics)))
        conn.commit()

    conn.close()

if __name__ == "__main__":
    ingest_arxiv()