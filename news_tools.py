"""News source (NewsAPI) and article cleaning.

fetch_newsapi never raises for ordinary failures. It returns (articles, note),
so the fetcher agent can read the note and try a different query.
"""
import html
import os
import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import httpx

from schemas import NewsArticle

NEWSAPI_URL = "https://newsapi.org/v2/everything"


def clean_text(text: str | None) -> str:
    """Strip HTML tags, entities, NewsAPI's '[+123 chars]' marker and extra whitespace."""
    if not text:
        return ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    text = re.sub(r"\s*\[\+\d+ chars\]\s*$", "", text)
    return " ".join(text.split())


# ── Sources ────────────────────────────────────────────────────────────
async def fetch_newsapi(query: str, max_articles: int) -> Tuple[List[NewsArticle], str]:
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return [], "NEWS_API_KEY is not set"
    params = {"q": query, "sortBy": "publishedAt", "pageSize": max(1, min(max_articles, 100)),
              "language": "en", "apiKey": api_key}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(NEWSAPI_URL, params=params)
        data = response.json()
        if response.status_code != 200 or data.get("status") != "ok":
            return [], f"NewsAPI error {response.status_code}: {str(data.get('message', ''))[:160]}"
    except (httpx.HTTPError, ValueError) as exc:
        return [], f"NewsAPI request failed: {type(exc).__name__}"

    articles = [
        NewsArticle(
            title=clean_text(item.get("title")),
            description=clean_text(item.get("description")),
            source=(item.get("source") or {}).get("name") or "Unknown",
            url=item.get("url") or "",
            published_at=item.get("publishedAt") or "",
            content=clean_text(item.get("content")) or clean_text(item.get("description")),
        )
        for item in data.get("articles", [])
    ]
    return articles, "" if articles else "NewsAPI returned no articles for this query"


# ── Curation (plain code, no LLM) ──────────────────────────────────────
def normalize_title(title: str) -> str:
    title = re.sub(r"\s+[-|–—]\s+[^-|–—]{2,40}$", "", title.strip())   # drop " - Source Name"
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


def is_near_duplicate(a: str, b: str) -> bool:
    words_a, words_b = set(a.split()), set(b.split())
    return a == b or bool(words_a and words_b and len(words_a & words_b) / len(words_a | words_b) >= 0.75)


def curate_articles(raw: List[Dict[str, Any]], limit: int = 15) -> List[Dict[str, Any]]:
    """Drop junk and duplicates, sort newest first, and number the articles 1..n."""
    usable = [a for a in raw
              if (a.get("title") or "").strip() and "[removed]" not in a["title"].lower()
              and "removed.com" not in (a.get("url") or "").lower()]
    usable.sort(key=lambda a: (a.get("published_at") or "")[:19], reverse=True)

    curated: List[Dict[str, Any]] = []
    seen_titles: List[str] = []
    seen_urls = set()
    for a in usable:
        title = normalize_title(a["title"])
        parsed = urlparse(a.get("url") or "")
        url_key = f"{parsed.netloc.lower()}{parsed.path.rstrip('/')}" if a.get("url") else ""
        if url_key in seen_urls or any(is_near_duplicate(title, seen) for seen in seen_titles):
            continue
        seen_titles.append(title)
        if url_key:
            seen_urls.add(url_key)
        description = a.get("description") or ""
        curated.append({
            "id": len(curated) + 1,
            "title": a["title"].strip(),
            "summary": description if len(description) <= 400 else description[:400].rstrip() + "...",
            "source": a.get("source") or "Unknown",
            "url": a.get("url") or "",
            "published": a.get("published_at") or "",
            "content": (a.get("content") or description)[:1000],
        })
        if len(curated) >= limit:
            break
    return curated
