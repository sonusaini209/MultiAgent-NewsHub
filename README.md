# MultiAgent NewsHub

> AI-powered news intelligence with a LangGraph multi-agent pipeline: fetch, curate, plan, analyse, write, fact-check.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3.3--70b-orange.svg)](https://console.groq.com)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg)](https://multi-agent-news-hub.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live demo:** https://multi-agent-news-hub.vercel.app/

---

## Overview

You give it a topic. It pulls recent articles from NewsAPI, cleans them, and has a team of LLM agents produce a cited report: an executive summary, a long-form blog post, thematic categories and trend analysis. A critic agent fact-checks the drafts against the source articles and sends failed sections back for a rewrite.

---

## Architecture

```
START
  |
  v
fetch --> curate --> supervisor --(too few articles, max 2 retries)--> fetch (rewritten query)
                         |------(still too few)--------------------> give_up --> END
                         v
                  analysts  (parallel: categories, trends)
                         v
                  writers   (parallel: blog, summary; read the analysts' output)
                         v
                  critic --(problems found, max 2 rounds)--> redo ONLY the failed sections
                         |
                         v (all good)
                  aggregator --> END
```

Parallel stages use LangGraph's `Send` API: one worker per task, all running at the same time.

### Agents

| Agent | Uses LLM | Role |
|---|---|---|
| **Fetcher** | Yes (tool calling) | Calls the `search_newsapi` tool, reads results, retries with a better query if needed |
| **Query rewriter** | Yes | Broadens the search query when too few articles come back |
| **Curator** | No | Removes junk and duplicates, sorts newest first, numbers articles 1..n |
| **Supervisor** | Yes (structured output) | Chooses which analysis tasks to run; code enforces minimum-data rules and always runs `summary` |
| **Category analyst** | Yes | Groups articles into 3-6 themes |
| **Trend analyst** | Yes | Finds 3-5 trends with a clearly labelled outlook |
| **Blog writer** | Yes | 1000-1500 word article, uses the analysts' output |
| **Summary writer** | Yes | 150-200 word executive summary |
| **Critic** | Code + LLM | Code checks (valid `[n]` citations, coverage, cut-off text) plus an LLM fact-check of blog, summary and trends |
| **Aggregator** | No | Builds the final report with status and warnings |

### Reliability features

- **Hard rules in code:** the LLM can't skip `summary`, and `categories` / `trends` need at least 5 / 8 articles.
- **Fallbacks:** if the supervisor, query rewriter or critic LLM fails, the pipeline falls back to sensible defaults instead of crashing.
- **Retries:** LLM calls retry with exponential backoff (up to 4 attempts).
- **Honest status:** every report is `success`, `partial` (with warnings) or an error; unresolved critic issues are shown to the user.
- **Prompt-injection guard:** article text is marked as untrusted data in every prompt.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph `StateGraph`, `Send`, conditional edges |
| LLM | Groq, `llama-3.3-70b-versatile` (override with `GROQ_MODEL`) |
| Backend | FastAPI (async, Pydantic validation, Server-Sent Events) |
| News data | NewsAPI `/v2/everything` (English articles) |
| Frontend | Vanilla JS + CSS, dark theme, tabbed UI with live progress |
| Deployment | Vercel (Python serverless) |

---

## API

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Liveness check and whether API keys are configured |
| `POST` | `/api/generate` | Run the pipeline, return the finished report as JSON |
| `POST` | `/api/generate/stream` | Same, but streams a `progress` event per finished step, then `result` (or `error`) |
| `POST` | `/api/explain` | Plain-English explanation of one article |

### `POST /api/generate`

Request:

```json
{
  "query": "Artificial Intelligence",
  "domain": "Technology",
  "num_articles": 15
}
```

`query` is required (1-200 chars). `domain` is optional. `num_articles` is 1-50 (default 15).

Response:

```json
{
  "status": "success",
  "articles_count": 14,
  "articles": [{"id": 1, "title": "...", "summary": "...", "source": "...", "url": "...", "published": "...", "content": "..."}],
  "supervisor_reasoning": "...",
  "tasks_completed": ["summary", "categories", "trends", "blog"],
  "blog": "...",
  "summary": "...",
  "categories": "...",
  "trends": "...",
  "warnings": []
}
```

Error responses: `400` invalid input, `404` too few articles found, `502` analysis failed, `503` service not configured or LLM unavailable.

---

## Project Structure

```
MultiAgent-NewsHub/
├── agent.py          # LangGraph workflow: all agents, prompts, routing
├── app.py            # FastAPI app and routes (plain + streaming)
├── llm.py            # Shared ChatGroq client, retry with backoff, structured output
├── news_tools.py     # NewsAPI client, text cleaning, de-duplication
├── schemas.py        # Pydantic models
├── index.html        # Frontend
├── requirements.txt
├── vercel.json
├── .gitignore
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.10+

```bash
git clone https://github.com/sonusaini209/MultiAgent-NewsHub
cd MultiAgent-NewsHub
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

Create a `.env` file (never commit it):

```env
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=gsk_your_groq_key_here
# optional
GROQ_MODEL=llama-3.3-70b-versatile
```

Get keys at [newsapi.org](https://newsapi.org) and [console.groq.com](https://console.groq.com).

```bash
uvicorn app:app --reload
```

Open http://localhost:8000

---

## License

MIT, see [LICENSE](./LICENSE).
