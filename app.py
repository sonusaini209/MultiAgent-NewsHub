from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import create_graph, explain_article, NewsState

app = FastAPI(
    title="MultiAgent-NewsHub API",
    description="AI-Powered Multi-Agent News Intelligence for Any Domain",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve Frontend UI ─────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")

# ── API Routes ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str
    num_articles: int = 15
    domain: str = "general"  # NEW: Domain context
    language: str = "en"     # NEW: Language support

class ExplainRequest(BaseModel):
    title: str
    content: str

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "features": ["domain-agnostic", "multi-language", "6-agent-pipeline"]}

@app.get("/api/languages")
async def get_languages():
    """Get supported languages for news fetching"""
    languages = {
        "en": "English",
        "es": "Español (Spanish)",
        "fr": "Français (French)",
        "de": "Deutsch (German)",
        "it": "Italiano (Italian)",
        "pt": "Português (Portuguese)",
        "ru": "Русский (Russian)",
        "ar": "العربية (Arabic)",
        "zh": "中文 (Chinese)",
        "nl": "Nederlands (Dutch)",
        "no": "Norsk (Norwegian)",
        "se": "Svenska (Swedish)"
    }
    return {"languages": languages, "default": "en"}

@app.get("/api/suggested-domains")
async def get_suggested_domains():
    """Get examples of domains you can search (not limited to these)"""
    domains = {
        "Technology": "AI, blockchain, cybersecurity, quantum computing, semiconductors",
        "Business & Finance": "Stock market, cryptocurrency, startups, economics, mergers & acquisitions",
        "Healthcare": "Medical research, pharmaceuticals, vaccines, mental health, biotechnology",
        "Climate & Environment": "Climate change, renewable energy, conservation, sustainability, green tech",
        "Politics & Government": "Elections, policy, international relations, governance, legislation",
        "Science & Research": "Physics, biology, space exploration, scientific breakthroughs",
        "Sports": "Football, basketball, tennis, Olympics, esports, motorsports",
        "Entertainment & Media": "Movies, music, streaming, celebrity, gaming, podcasts",
        "Travel & Culture": "Tourism, cultural events, cities, food, lifestyle, fashion",
        "Education": "Universities, online learning, EdTech, student life",
        "Real Estate": "Housing market, commercial property, urban development, real estate tech",
        "Agriculture": "Farming technology, sustainability, food production, agritech",
        "Transportation": "Electric vehicles, autonomous vehicles, aviation, public transit",
        "Retail & E-commerce": "Online shopping, retail trends, supply chain, consumer behavior",
        "Telecommunication": "5G, internet, telecom companies, broadband"
    }
    return {
        "note": "These are examples. You can search ANY topic, industry, or subject.",
        "suggested_domains": domains
    }

@app.post("/api/generate")
async def generate_report(request: GenerateRequest):
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        if request.num_articles < 1 or request.num_articles > 50:
            raise HTTPException(status_code=400, detail="Articles must be between 1 and 50")

        graph = create_graph()

        initial_state = NewsState(
            query=request.query,
            domain=request.domain,
            language=request.language,
            num_articles=request.num_articles,
            raw_articles=[],
            curated_articles=[],
            blog_content="",
            summary_content="",
            categories_content="",
            trends_content=""
        )

        result = graph.invoke(initial_state)

        return {
            "status": "success",
            "query": request.query,
            "domain": request.domain,
            "language": request.language,
            "articles_count": len(result["curated_articles"]),
            "articles": result["curated_articles"],
            "blog": result["blog_content"],
            "summary": result["summary_content"],
            "categories": result["categories_content"],
            "trends": result["trends_content"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/api/explain")
async def explain_article_endpoint(request: ExplainRequest):
    try:
        if not request.title or len(request.title.strip()) == 0:
            raise HTTPException(status_code=400, detail="Title cannot be empty")

        if not request.content or len(request.content.strip()) == 0:
            raise HTTPException(status_code=400, detail="Content cannot be empty")

        explanation = explain_article(request.title, request.content)

        return {
            "status": "success",
            "title": request.title,
            "explanation": explanation
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
