"""
MultiAgent-NewsHub — app.py (Upgraded)
=======================================
FastAPI backend updated to:
  - Pass thread_id for MemorySaver checkpointing
  - Return supervisor_reasoning and tasks_completed in response
  - Expose new /api/graph-info endpoint for interview demos
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys, os, uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import create_graph, explain_article, NewsState

app = FastAPI(
    title="MultiAgent-NewsHub API",
    description="AI-Powered Multi-Agent News Intelligence — Supervisor + Parallel Execution",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compile graph once at startup (MemorySaver persists across requests)
graph = create_graph()


# ── Frontend ───────────────────────────────────────────────────────────

@app.get("/")
async def serve_ui():
    return FileResponse("index.html")


# ── Request / Response models ──────────────────────────────────────────

class GenerateRequest(BaseModel):
    query: str
    num_articles: int = 15
    language: str = "en"
    thread_id: str | None = None  # optional: pass to resume a checkpointed session


class ExplainRequest(BaseModel):
    title: str
    content: str


# ── Health + metadata ──────────────────────────────────────────────────

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/api/graph-info")
async def graph_info():
    """
    Returns architecture metadata — useful for interview demos and README.
    Shows the multi-agent topology: nodes, edges, supervisor routing.
    """
    return {
        "architecture": "Supervisor Multi-Agent with Parallel Execution",
        "agents": [
            {"name": "Fetcher",     "role": "Pulls articles from NewsAPI; supports retry with broader query"},
            {"name": "Curator",     "role": "Deduplicates and cleans raw articles"},
            {"name": "Supervisor",  "role": "LLM decides which analysis agents to run; triggers retry if articles < 3"},
            {"name": "Blog Agent",  "role": "Writes 1000-1500 word narrative analysis (parallel)"},
            {"name": "Summary Agent","role": "Generates 150-200 word executive summary (parallel)"},
            {"name": "Category Agent","role": "Groups articles into thematic buckets (parallel)"},
            {"name": "Trend Agent", "role": "Detects patterns and forecasts future implications (parallel)"},
            {"name": "Aggregator",  "role": "Collects all parallel outputs into final structured report"},
        ],
        "features": [
            "LLM Supervisor routing (not hardcoded edges)",
            "Parallel agent execution via LangGraph Send API",
            "Conditional retry on insufficient articles",
            "MemorySaver checkpointing (resumable per thread_id)",
            "Single shared LLM instance (not re-created per agent)",
        ],
        "graph_edges": {
            "fixed":       ["START→fetch", "fetch→curate", "curate→supervisor", "analysis_agent→aggregator", "aggregator→END"],
            "conditional": ["supervisor→fetch (retry)", "supervisor→[Send×N analysis_agent] (parallel fan-out)"],
        },
    }


@app.get("/api/topics")
async def get_topics():
    topics = {
        "🤖 AI & Machine Learning":       "Artificial Intelligence OR Machine Learning OR AI",
        "🧠 Deep Learning":                "deep learning OR neural network OR transformer",
        "💬 Large Language Models":        "LLM OR GPT OR language model OR ChatGPT",
        "🔬 AI Research":                  "AI research OR AI breakthrough OR artificial intelligence research",
        "💼 AI in Business":               "AI business OR enterprise AI OR business intelligence",
        "🎨 Generative AI":                "generative AI OR diffusion model OR text generation",
        "🏥 AI in Healthcare":             "AI healthcare OR medical AI OR diagnostic AI",
        "🚗 AI in Autonomous Systems":     "autonomous vehicle OR self-driving OR robotics",
        "📊 Data Science & AI":            "data science OR machine learning analytics",
        "🎵 AI & Creativity":              "AI art OR generative art OR AI music",
    }
    return {"topics": topics}


# ── Main generation endpoint ───────────────────────────────────────────

@app.post("/api/generate")
async def generate_report(request: GenerateRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if not (1 <= request.num_articles <= 50):
        raise HTTPException(status_code=400, detail="num_articles must be between 1 and 50")

    # Use provided thread_id or generate a new one (enables MemorySaver checkpointing)
    thread_id = request.thread_id or str(uuid.uuid4())

    initial_state = NewsState(
        query=request.query,
        domain="",
        language=request.language,
        num_articles=request.num_articles,
        retry_count=0,
        raw_articles=[],
        curated_articles=[],
        tasks_to_run=[],
        supervisor_reasoning="",
        blog_content="",
        summary_content="",
        categories_content="",
        trends_content="",
        final_report={},
    )

    try:
        result = graph.invoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph execution error: {str(e)}")

    report = result.get("final_report", {})

    return {
        "status": "success",
        "thread_id": thread_id,               # return so client can resume
        "query": request.query,
        "articles_count": report.get("articles_count", 0),
        "supervisor_reasoning": report.get("supervisor_reasoning", ""),
        "tasks_completed": report.get("tasks_completed", []),
        "articles": result.get("curated_articles", []),
        "blog": report.get("blog", ""),
        "summary": report.get("summary", ""),
        "categories": report.get("categories", ""),
        "trends": report.get("trends", ""),
    }


# ── Article explainer ──────────────────────────────────────────────────

@app.post("/api/explain")
async def explain_article_endpoint(request: ExplainRequest):
    if not request.title or not request.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    if not request.content or not request.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    try:
        explanation = explain_article(request.title, request.content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainer error: {str(e)}")

    return {"status": "success", "title": request.title, "explanation": explanation}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
