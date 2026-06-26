

# MultiAgent NewsHub

> AI-Powered News Intelligence with Supervisor-Orchestrated Multi-Agent Pipeline

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama3.3--70b-orange.svg)](https://console.groq.com)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg)](https://multi-agent-news-hub.vercel.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Live Demo

**https://multi-agent-news-hub.vercel.app/**

---

## Overview

MultiAgent NewsHub is an AI news intelligence platform built on a **true multi-agent architecture** — not a sequential pipeline. A LangGraph `StateGraph` with a **Supervisor agent** dynamically decides which analysis agents to activate, then fans them out in **parallel** using LangGraph's `Send` API. All agent runs are checkpointed via `MemorySaver`, making every session resumable by `thread_id`.

The system connects to NewsAPI for real-time articles, passes them through an 8-node agent graph, and returns structured analysis: long-form blog, executive summary, thematic categories, and trend forecasts — all generated concurrently.

---

## Architecture

```
START
  │
  ▼
┌─────────┐     ┌──────────┐     ┌────────────┐
│ Fetcher │────▶│  Curator │────▶│ Supervisor │
└─────────┘     └──────────┘     └─────┬──────┘
                                        │
                          ┌─────────────┼──────────────────┐
                          │  conditional_edges (LLM routes) │
                          │                                  │
                     retry│fetch              Send × N (parallel)
                          │             ┌────────────────────┐
                          │             │  analysis_agent ×4  │
                          │             │  ┌──────────────┐   │
                          │             │  │  Blog Agent  │   │
                          │             │  ├──────────────┤   │
                          │             │  │Summary Agent │   │
                          │             │  ├──────────────┤   │
                          │             │  │Category Agent│   │
                          │             │  ├──────────────┤   │
                          │             │  │ Trend Agent  │   │
                          │             │  └──────────────┘   │
                          │             └────────────┬────────┘
                          │                          │
                          │                    ┌─────▼──────┐
                          │                    │ Aggregator │
                          │                    └─────┬──────┘
                          │                          │
                          └──────────────────────────▼
                                                    END
```

### What makes this genuinely multi-agent

| Feature | Implementation |
|---|---|
| **Supervisor routing** | LLM reads query + article count, returns JSON decision on which agents to run |
| **Parallel execution** | `Send` API fans out to Blog, Summary, Categories, Trends simultaneously |
| **Conditional retry** | If articles < 3, supervisor routes back to Fetcher with a broader query (max 2 retries) |
| **MemorySaver checkpointing** | Every run saved per `thread_id`; resumable across requests |
| **Shared LLM instance** | Single `ChatGroq` client initialized once, reused across all agents |
| **TypedDict state** | Fully typed `NewsState` — not a plain dict subclass |

---

## Agent Roles

| # | Agent | Type | Role |
|---|---|---|---|
| 1 | **Fetcher** | Data | Pulls articles from NewsAPI; uses broader query on retry |
| 2 | **Curator** | Data | Deduplicates and cleans raw articles |
| 3 | **Supervisor** ⭐ | LLM | Decides which agents to run; triggers retry if articles insufficient |
| 4 | **Blog Agent** | LLM (parallel) | Writes 1000–1500 word narrative analysis |
| 5 | **Summary Agent** | LLM (parallel) | Generates 150–200 word executive summary |
| 6 | **Category Agent** | LLM (parallel) | Groups articles into 5–7 thematic buckets |
| 7 | **Trend Agent** | LLM (parallel) | Detects patterns and forecasts implications |
| 8 | **Aggregator** | Logic | Collects all parallel outputs into final structured report |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph `StateGraph`, `Send` API, `conditional_edges`, `MemorySaver` |
| LLM | Groq — `llama-3.3-70b-versatile` |
| Backend | FastAPI (async, CORS, Pydantic validation) |
| News Data | NewsAPI (real-time, 10 languages) |
| Frontend | Vanilla JS + CSS, dark theme, tab-based UI |
| Deployment | Vercel (serverless Python) |

---

## Features

- **LLM Supervisor** — dynamic agent routing, not hardcoded edges
- **Parallel analysis** — 4 agents run concurrently via `Send` API
- **Conditional retry** — auto-broadens query if article count is too low
- **Session checkpointing** — every run resumable by `thread_id`
- **10 languages** — en, es, fr, de, it, pt, ar, zh, nl, ru
- **Article Explainer** — on-demand plain-English breakdown of any article
- **Graph info endpoint** — `/api/graph-info` returns full agent topology as JSON
- **Dark-themed UI** — responsive, tab-based, no framework dependencies

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the frontend UI |
| `GET` | `/api/health` | Health check + version |
| `GET` | `/api/graph-info` | Full agent topology metadata |
| `GET` | `/api/topics` | 10 predefined AI/ML topic queries |
| `POST` | `/api/generate` | Run the multi-agent pipeline |
| `POST` | `/api/explain` | On-demand article explanation |

### `/api/generate` — Request

```json
{
  "query": "Artificial Intelligence",
  "num_articles": 15,
  "language": "en",
  "thread_id": "optional-uuid-to-resume-session"
}
```

### `/api/generate` — Response

```json
{
  "status": "success",
  "thread_id": "uuid",
  "articles_count": 14,
  "supervisor_reasoning": "All 4 tasks activated: 14 articles provide sufficient data for trend analysis.",
  "tasks_completed": ["blog", "summary", "categories", "trends"],
  "articles": [...],
  "blog": "...",
  "summary": "...",
  "categories": "...",
  "trends": "..."
}
```

---

## Project Structure

```
MultiAgent-NewsHub/
├── agent.py          # 8-node multi-agent graph (Supervisor + parallel Send)
├── app.py            # FastAPI backend with thread_id + graph-info endpoint
├── index.html        # Dark-themed frontend UI
├── requirements.txt  # Python dependencies
├── vercel.json       # Vercel deployment config
├── .env              # API keys — never commit this
├── .gitignore        # Excludes .env
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.9+, pip

**1. Clone**
```bash
git clone https://github.com/sonusaini209/MultiAgent-NewsHub
cd MultiAgent-NewsHub
```

**2. Virtual environment**
```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. API keys** — create a `.env` file:
```env
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=gsk_your_groq_key_here
```

Get them free at [newsapi.org](https://newsapi.org) and [console.groq.com](https://console.groq.com).

**5. Run**
```bash
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000)

---

## How the Supervisor Works

On every run, after curation, the Supervisor agent receives the article count and sample titles and calls the LLM with a structured prompt. The LLM responds in JSON:

```json
{
  "tasks": ["blog", "summary", "categories", "trends"],
  "reasoning": "14 articles provide sufficient volume for all analysis types."
}
```

The `supervisor_router` function then uses LangGraph's `Send` API to dispatch each task as a parallel branch — one `analysis_agent` instance per task. If articles < 3, it routes back to the Fetcher with a broader query instead.

---

## License

MIT — see [LICENSE](./LICENSE) for details.
