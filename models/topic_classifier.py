TOPIC_KEYWORDS = {
    "ml_ethics": ["ethics", "bias", "fairness", "responsiblity"],
    "ranking_systems":["ranking", "rank", "retrieval", "recommendation"],
    "diversity": ["diversity", "representation","inclusion"]

}

def classify_topics(text: str):
    text = text.lower()
    return{
        topic:any(keyword in text for keyword in keywords)
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    