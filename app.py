"""MultiAgent-NewsHub API (FastAPI).

  GET  /                     web UI
  GET  /api/health           liveness check
  POST /api/generate         run the agents, return the finished report as JSON
  POST /api/generate/stream  same, but streams live progress (Server-Sent Events)
  POST /api/explain          explain one article
"""
import json
import logging
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

import llm
from agent import MIN_ARTICLES, NODE_LABELS, create_graph, explain_article, new_run

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = FastAPI(title="MultiAgent-NewsHub", version="3.0.0")
graph = create_graph()
BASE_DIR = Path(__file__).resolve().parent


# ── Request models ─────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=200)
    domain: str = Field(default="", max_length=80)
    num_articles: int = Field(default=15, ge=MIN_ARTICLES, le=50)   # the pipeline needs at least MIN_ARTICLES


class ExplainRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=6000)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Turn Pydantic errors into one readable message the UI can show."""
    first = exc.errors()[0]
    field = ".".join(str(p) for p in first["loc"] if p != "body")
    message = str(first["msg"]).removeprefix("Value error, ")
    return JSONResponse(status_code=400, content={"detail": f"{field}: {message}" if field else message})


# ── Helpers ────────────────────────────────────────────────────────────
def require_llm() -> None:
    if not llm.llm_configured():
        raise HTTPException(503, "The AI service is not configured on this server (GROQ_API_KEY missing).")


def require_news() -> None:
    if not os.getenv("NEWS_API_KEY"):
        raise HTTPException(503, "The news source is not configured on this server (NEWS_API_KEY missing).")


def check_report(report: dict | None) -> dict:
    """Return the report if the run produced one; otherwise raise the matching HTTP error."""
    status = (report or {}).get("status")
    if status == "insufficient_articles":
        raise HTTPException(404, report["message"])
    if status not in ("success", "partial"):
        raise HTTPException(502, "The AI analysis could not be completed. Please try again in a minute.")
    return report


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


GRAPH_CONFIG = {"recursion_limit": 50}


# ── Routes ─────────────────────────────────────────────────────────────
@app.get("/")
async def home():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "healthy", "llm_configured": llm.llm_configured(),
            "news_api_configured": bool(os.getenv("NEWS_API_KEY"))}


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    require_llm()
    require_news()
    try:
        state = await graph.ainvoke(new_run(req.query, req.domain, req.num_articles), GRAPH_CONFIG)
    except llm.LLMError:
        raise HTTPException(503, "The AI service is temporarily unavailable. Please try again shortly.")
    return check_report(state.get("final_report"))


@app.post("/api/generate/stream")
async def generate_stream(req: GenerateRequest):
    """Server-Sent Events: a `progress` event per finished step, then `result` (or `error`)."""
    require_llm()
    require_news()

    async def events():
        report = None
        try:
            yield sse("progress", {"label": "Starting"})
            run = new_run(req.query, req.domain, req.num_articles)
            async for update in graph.astream(run, GRAPH_CONFIG, stream_mode="updates"):
                for node, output in update.items():
                    yield sse("progress", {"node": node, "label": NODE_LABELS.get(node, node)})
                    if node in ("aggregator", "give_up"):
                        report = output["final_report"]
            yield sse("result", check_report(report))
        except HTTPException as exc:
            yield sse("error", {"detail": exc.detail})
        except llm.LLMError:
            yield sse("error", {"detail": "The AI service is temporarily unavailable."})

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/explain")
async def explain(req: ExplainRequest):
    require_llm()
    try:
        explanation = await explain_article(req.title, req.content)
    except llm.LLMError:
        raise HTTPException(503, "The AI service is temporarily unavailable. Please try again shortly.")
    return {"status": "success", "title": req.title, "explanation": explanation}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))



