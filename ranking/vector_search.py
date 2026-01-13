import sqlite3
import pickle
import json
import numpy as np
import faiss    

from models.embedder import Embedder

class VectorSearch:
    def __init__(self, db_path = "data/content.db"):
        self.db_path = db_path
        self.embedder = Embedder()

    def _load_filtered_articles(self, required_topics):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        cur.execute("SELECT id, embedding, topics, title, url FROM articles")
        rows = cur.fetchall()
        conn.close()

        article_ids = []
        embeddings = []
        metadata = []

        for article_id, emb_blob, topics_json, title, url in rows:
            topics = json.loads(topics_json)

            if not all(topics.get(t, False) for t in required_topics):
                continue
                
            embeddings.append(pickle.loads(emb_blob))
            article_ids.append(article_id)
            metadata.append({
                "id": article_id,
                "title" : title,
                "url": url,
                "topics": topics
            })

        if not embeddings:
            return None, None, None

        return(
            np.vstack(embeddings).astype("float32"),
            article_ids,
            metadata
        )
    
    def search(self, query_text, required_topics, k=10):
        vectors, ids, metadata = self._load_filtered_articles(required_topics)

        if vectors is None:
            return []

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        query_vec = self.embedder.encode(query_text).astype("float32").reshape(1, -1)
        scores, idxs = index.search(query_vec, min(k, len(ids)))

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            item = metadata[idx].copy()
            item["vec"] = vectors[idx]
            item["score"] = float(score)
            results.append(item)

        return results