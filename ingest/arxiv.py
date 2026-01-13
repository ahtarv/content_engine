import feedparser
import uuid
import urllib.parse


def fetch_arxiv(query = "ranking fairness", max_results = 20):
    encoded_query = urllib.parse.quote(query)

    url = (
        "http://export.arxiv.org/api/query?"
        f"search_query=all:{encoded_query}"
        f"&start=0&max_results={max_results}"
    )

    print("DEBUG arXiv URL:", url)

    feed = feedparser.parse(url)

    articles = []
    for entry in feed.entries:
        articles.append({
            "id": str(uuid.uuid4()),
            "url": entry.link,
            "source": "arxiv",
            "title": entry.title.strip().replace("\n", " "),
            "abstract": entry.summary.strip().replace("\n", " "),
            "year": entry.published[:4]
        })

    return articles