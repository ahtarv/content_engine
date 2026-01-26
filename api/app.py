from fastapi import FastAPI 
from pydantic import BaseModel
from typing import List

from ranking.vector_search import VectorSearch
from ranking.personalized_ranker import rerank
from ranking.mmr import mmr_rerank
from models.user_loader import load_user_profile
from storage.interactions import log_interaction

app = FastAPI(title = "Content Discovery Engine")

vs = VectorSearch()

class SearchRequest(BaseModel):
    user_id: str
    query: str
    topics: List[str]
    k: int = 10

class InteractionRequest(BaseModel):
    user_id: str
    article_id: str
    dwell_time: float
    scroll_depth: float

@app.post("/search")
def search(req: SearchRequest):
    user = load_user_profile(req.user_id)

    raw_results =vs.search(
        query_text = req.query,
        required_topics = req.topics,
        k = req.k
    )

    personalized = rerank(raw_results, user)
    diversified = mmr_rerank(personalized, k=req.k)

    for r in diversified:
        r.pop("vec", None)

    return {
        "count": len(diversified),
        "results": diversified
    }

@app.post("/interact")
def interact(req: InteractionRequest):
    log_interaction(
        user_id = req.user_id,
        article_id = req.article_id,
        dwell_time = req.dwell_time,
        scroll_depth = req.scroll_depth
    )

    return {"status": "ok"}
    