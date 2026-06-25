"""
MultiAgent-NewsHub — agent.py (Upgraded)
=========================================
Architecture:
  fetch → curate → supervisor → [blog, summary, categories, trends] (parallel) → aggregator → END

Key upgrades over v1:
  1. Supervisor agent  — LLM decides which analysis agents to invoke (not hardcoded)
  2. Parallel branches — blog, summary, categories, trends run simultaneously via Send API
  3. Conditional retry — if articles < 3, supervisor routes back to fetch with broader query
  4. MemorySaver       — full checkpointing; pipeline is resumable per thread_id
  5. Single LLM init   — one ChatGroq instance shared across all agents (not re-created per call)
  6. TypedDict state   — proper typed state instead of plain dict subclass
"""

import os
import requests
import asyncio
from typing import List, Dict, Any, Optional, Literal, Annotated
from typing_extensions import TypedDict

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Shared LLM instance (created once, not per-agent) ─────────────────
_llm: Optional[ChatGroq] = None

def get_llm() -> Optional[ChatGroq]:
    global _llm
    if _llm is None and GROQ_API_KEY:
        _llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048,
        )
    return _llm


# ── Data models ────────────────────────────────────────────────────────

class NewsArticle(BaseModel):
    title: str
    description: str
    source: str
    url: str
    published_at: str
    content: str


class AnalysisTask(TypedDict):
    """Passed via Send to each parallel analysis agent"""
    task: Literal["blog", "summary", "categories", "trends"]
    curated_articles: List[Dict[str, Any]]
    query: str
    domain: str


class NewsState(TypedDict):
    """
    Full pipeline state. All fields are optional at init;
    each node writes only its own outputs.
    """
    query: str
    domain: str
    language: str
    num_articles: int
    retry_count: int                        # tracks fetch retries

    raw_articles: List[NewsArticle]
    curated_articles: List[Dict[str, Any]]

    # Supervisor decision
    tasks_to_run: List[str]                 # e.g. ["blog", "summary", "categories", "trends"]
    supervisor_reasoning: str

    # Parallel agent outputs (aggregated via reducer)
    blog_content: str
    summary_content: str
    categories_content: str
    trends_content: str

    # Final aggregated report
    final_report: Dict[str, Any]


# ── News fetching helpers ──────────────────────────────────────────────

def fetch_top_news(
    query: str,
    max_articles: int,
    language: str = "en",
    broad: bool = False,
) -> List[NewsArticle]:
    """
    Fetch from NewsAPI. If broad=True, strips modifiers for a wider net
    (used on supervisor-triggered retry).
    """
    if not NEWS_API_KEY:
        return []

    valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "nl", "no", "se"]
    if language not in valid_languages:
        language = "en"

    search_query = query.split(" OR ")[0] if broad else query  # broaden on retry

    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": search_query,
                "sortBy": "publishedAt",
                "language": language,
                "pageSize": max_articles,
                "apiKey": NEWS_API_KEY,
            },
            timeout=10,
        )
        data = response.json()
        if data.get("status") != "ok":
            return []

        articles = []
        for article in data.get("articles", []):
            try:
                articles.append(
                    NewsArticle(
                        title=article.get("title", "").strip(),
                        description=article.get("description") or "",
                        source=article.get("source", {}).get("name", "Unknown"),
                        url=article.get("url", ""),
                        published_at=article.get("publishedAt", ""),
                        content=article.get("content") or article.get("description") or "",
                    )
                )
            except Exception:
                continue
        return articles
    except Exception:
        return []


def curate_articles(articles: List[NewsArticle]) -> List[Dict[str, Any]]:
    """Deduplicate and clean articles."""
    curated, seen_titles = [], set()
    for article in articles:
        if not article.title:
            continue
        key = article.title.strip().lower()
        if key in seen_titles:
            continue
        curated.append({
            "title": article.title,
            "summary": (
                article.description[:200] + "..."
                if len(article.description) > 200
                else article.description
            ),
            "source": article.source,
            "url": article.url,
            "published": article.published_at,
            "content": article.content,
        })
        seen_titles.add(key)
    return curated


# ── Prompt helper ──────────────────────────────────────────────────────

def prompt_chain(template: str, data: Dict) -> str:
    llm = get_llm()
    if not llm:
        return "Error: LLM not initialized. Check GROQ_API_KEY."
    try:
        return (ChatPromptTemplate.from_template(template) | llm).invoke(data).content
    except Exception as e:
        return f"Error: {str(e)}"


# ══════════════════════════════════════════════════════════════════════
#  AGENT 1 — Fetcher
# ══════════════════════════════════════════════════════════════════════

def agent_fetch_news(state: NewsState) -> Dict:
    """Fetches articles. On retry (retry_count > 0), broadens the query."""
    broad = state.get("retry_count", 0) > 0
    raw = fetch_top_news(
        state["query"],
        state.get("num_articles", 15),
        state.get("language", "en"),
        broad=broad,
    )
    return {"raw_articles": raw}


# ══════════════════════════════════════════════════════════════════════
#  AGENT 2 — Curator
# ══════════════════════════════════════════════════════════════════════

def agent_curate_news(state: NewsState) -> Dict:
    """Deduplicates and cleans raw articles."""
    return {"curated_articles": curate_articles(state["raw_articles"])}


# ══════════════════════════════════════════════════════════════════════
#  AGENT 3 — Supervisor  ⭐ The new brain
# ══════════════════════════════════════════════════════════════════════

def agent_supervisor(state: NewsState) -> Dict:
    """
    LLM-powered supervisor. Decides:
      - If too few articles → signal retry
      - Which analysis agents to run based on query type
      - Returns tasks_to_run list for the Send dispatcher
    """
    curated = state.get("curated_articles", [])

    # Gate: not enough articles to analyze
    if len(curated) < 3:
        return {
            "tasks_to_run": [],
            "supervisor_reasoning": (
                f"Only {len(curated)} articles found. Insufficient for analysis. "
                "Retrying fetch with broader query."
            ),
            "retry_count": state.get("retry_count", 0) + 1,
        }

    article_titles = "\n".join(f"- {a['title']}" for a in curated[:10])

    reasoning = prompt_chain(
        """You are a news analysis supervisor. Given the query and article titles below,
decide which analysis agents to activate. Always activate all four unless you have a strong reason not to.

Query: {query}
Article count: {count}
Sample titles:
{titles}

Respond in this EXACT format (JSON only, no extra text):
{{
  "tasks": ["blog", "summary", "categories", "trends"],
  "reasoning": "Brief explanation of why these tasks were chosen"
}}

Available tasks: blog, summary, categories, trends
Rules:
- "blog"       → always include; provides narrative analysis
- "summary"    → always include; quick executive overview
- "categories" → include when 5+ articles; groups by theme
- "trends"     → include when 8+ articles; needs enough data to detect patterns
""",
        {"query": state["query"], "count": len(curated), "titles": article_titles},
    )

    # Parse supervisor JSON safely
    import json, re
    tasks = ["blog", "summary", "categories", "trends"]  # safe default
    parsed_reasoning = "Defaulting to all tasks."
    try:
        match = re.search(r"\{.*\}", reasoning, re.DOTALL)
        if match:
            parsed = json.loads(match.group())
            tasks = parsed.get("tasks", tasks)
            parsed_reasoning = parsed.get("reasoning", parsed_reasoning)
    except Exception:
        pass

    return {
        "tasks_to_run": tasks,
        "supervisor_reasoning": parsed_reasoning,
    }


def supervisor_router(state: NewsState):
    """
    Conditional edge after supervisor.
    - If retry needed → back to fetch (max 2 retries)
    - Otherwise → fan out to parallel analysis agents via Send
    """
    tasks = state.get("tasks_to_run", [])
    retry_count = state.get("retry_count", 0)

    # Retry gate (max 2 attempts)
    if not tasks and retry_count <= 2:
        return "fetch"

    # Fan out: one Send per task → parallel execution
    return [
        Send("analysis_agent", {
            "task": task,
            "curated_articles": state["curated_articles"],
            "query": state["query"],
            "domain": state.get("domain", ""),
        })
        for task in tasks
    ]


# ══════════════════════════════════════════════════════════════════════
#  AGENT 4 — Analysis Agent (runs in parallel for each task)
# ══════════════════════════════════════════════════════════════════════

def agent_analysis(task_input: AnalysisTask) -> Dict:
    """
    Single node that handles all four analysis tasks.
    LangGraph spawns a separate instance per Send call → true parallelism.
    """
    task = task_input["task"]
    articles = task_input["curated_articles"]
    query = task_input.get("query", "")
    domain = task_input.get("domain", "")
    domain_ctx = f"Domain focus: {domain}" if domain else ""

    articles_text_long = "\n\n".join(
        f"{i+1}. {a['title']} ({a['source']})\n{a['summary']}"
        for i, a in enumerate(articles)
    )
    articles_text_short = "\n".join(f"- {a['title']}" for a in articles)

    if task == "blog":
        result = prompt_chain(
            f"""Write a comprehensive, well-structured blog post analyzing these news articles.
{domain_ctx}

ARTICLES:
{{articles}}

Requirements:
1. **Headline** — Compelling, 8-12 words
2. **Introduction** — Context and overview (150-200 words)
3. **Main Analysis** — 3-5 sections with key insights
4. **Key Takeaways** — 5-7 bullet points
5. **Future Implications** — What comes next
6. **Conclusion** — Summary and call to action

Style: Professional, engaging, data-driven. Total: 1000-1500 words.

BEGIN:""",
            {"articles": articles_text_long},
        )
        return {"blog_content": result}

    elif task == "summary":
        result = prompt_chain(
            """Create a concise executive summary of these news articles:

ARTICLES:
{articles}

Provide:
1. **Opening Statement** — Main theme in 1-2 sentences
2. **Key Developments** — 3-4 major points with specifics
3. **Takeaway** — What readers should remember (2-3 sentences)

Target: 150-200 words, clear and scannable.""",
            {"articles": articles_text_long},
        )
        return {"summary_content": result}

    elif task == "categories":
        result = prompt_chain(
            """Organize these news articles into logical thematic categories:

ARTICLES:
{articles}

For each category:
1. **[Category Name]**
   - List relevant article titles
   - 1-sentence explanation of the grouping

Maximum 5-7 categories. Be specific and useful.""",
            {"articles": articles_text_short},
        )
        return {"categories_content": result}

    elif task == "trends":
        result = prompt_chain(
            """Analyze trends and patterns across these news articles:

ARTICLES:
{articles}

Cover:
1. **Main Trends** — Recurring patterns
2. **Most Discussed Topics** — What's getting attention and why
3. **Key Developments** — Major breakthroughs or shifts
4. **Market / Industry Impact** — Who benefits or loses
5. **Future Outlook** — What comes next
6. **Global Implications** — Wider world impact

Be specific, insightful, and evidence-based.""",
            {"articles": articles_text_short},
        )
        return {"trends_content": result}

    return {}


# ══════════════════════════════════════════════════════════════════════
#  AGENT 5 — Aggregator
# ══════════════════════════════════════════════════════════════════════

def agent_aggregator(state: NewsState) -> Dict:
    """
    Collects outputs from all parallel analysis agents and assembles
    the final structured report. Also adds supervisor metadata.
    """
    report = {
        "query": state.get("query", ""),
        "articles_count": len(state.get("curated_articles", [])),
        "supervisor_reasoning": state.get("supervisor_reasoning", ""),
        "tasks_completed": state.get("tasks_to_run", []),
        "blog": state.get("blog_content", ""),
        "summary": state.get("summary_content", ""),
        "categories": state.get("categories_content", ""),
        "trends": state.get("trends_content", ""),
    }
    return {"final_report": report}


# ══════════════════════════════════════════════════════════════════════
#  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════

def create_graph():
    """
    Build and compile the multi-agent StateGraph with MemorySaver checkpointing.

    Flow:
      START → fetch → curate → supervisor
                                    ↓ (conditional)
                             [retry → fetch]  OR  [Send × N → analysis_agent (parallel)]
                                                          ↓
                                                     aggregator → END
    """
    checkpointer = MemorySaver()
    workflow = StateGraph(NewsState)

    # Register nodes
    workflow.add_node("fetch", agent_fetch_news)
    workflow.add_node("curate", agent_curate_news)
    workflow.add_node("supervisor", agent_supervisor)
    workflow.add_node("analysis_agent", agent_analysis)
    workflow.add_node("aggregator", agent_aggregator)

    # Fixed edges
    workflow.add_edge(START, "fetch")
    workflow.add_edge("fetch", "curate")
    workflow.add_edge("curate", "supervisor")

    # Conditional: supervisor → retry OR fan-out to parallel analysis agents
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_router,
        {
            "fetch": "fetch",           # retry path
            # Send targets are handled dynamically — no static mapping needed
        },
    )

    # After all parallel analysis_agent nodes complete → aggregator
    workflow.add_edge("analysis_agent", "aggregator")
    workflow.add_edge("aggregator", END)

    return workflow.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════════════
#  ON-DEMAND: Article Explainer (stateless, outside graph)
# ══════════════════════════════════════════════════════════════════════

def explain_article(title: str, content: str) -> str:
    """Explain any single article in plain English. Stateless utility."""
    return prompt_chain(
        """Explain this news article in simple, clear terms:

Title: {title}
Content: {content}

Provide:
1. **What Happened** — 2-3 sentences on the main event
2. **Why It Matters** — Who is affected and how
3. **Impact** — Short-term and long-term consequences

Keep it jargon-free and easy to understand.""",
        {"title": title, "content": content[:2000]},
    )
