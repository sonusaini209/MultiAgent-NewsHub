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
    description="AI-Powered Multi-Agent News Intelligence",
    version="1.0.0"
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

class ExplainRequest(BaseModel):
    title: str
    content: str

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/api/topics")
async def get_topics():
    topics = {
        " AI & Machine Learning": "Artificial Intelligence OR Machine Learning OR AI",
        " Deep Learning": "deep learning OR neural network OR transformer",
        " Large Language Models": "LLM OR GPT OR language model OR ChatGPT",
        " AI Research": "AI research OR AI breakthrough OR artificial intelligence research",
        " AI in Business": "AI business OR enterprise AI OR business intelligence",
        " Generative AI": "generative AI OR diffusion model OR text generation",
        " AI in Healthcare": "AI healthcare OR medical AI OR diagnostic AI",
        " AI in Autonomous Systems": "autonomous vehicle OR self-driving OR robotics",
        " Data Science & AI": "data science OR machine learning analytics",
        " AI & Creativity": "AI art OR generative art OR AI music",
    }
    return {"topics": topics}

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
