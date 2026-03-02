import os
import requests
from typing import List, Dict, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class NewsArticle(BaseModel):
    title: str
    description: str
    source: str
    url: str
    published_at: str
    content: str

class NewsState(dict):
    query: str
    domain: str  # NEW: Track domain for context
    language: str  # NEW: Language support
    num_articles: int
    raw_articles: List[NewsArticle]
    curated_articles: List[Dict[str, Any]]
    blog_content: str
    summary_content: str
    categories_content: str
    trends_content: str

def get_llm():
    if not GROQ_API_KEY:
        return None
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=2048
    )

def fetch_top_news(query: str, max_articles: int, language: str = "en") -> List[NewsArticle]:
    """
    Fetch news from NewsAPI
    Supports: en, es, fr, de, it, pt, ru, ar, zh, nl, no, se
    """
    if not NEWS_API_KEY:
        return []
    
    valid_languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "nl", "no", "se"]
    if language not in valid_languages:
        language = "en"
    
    try:
        response = requests.get(
            "https://newsapi.org/v2/everything",
            params={
                "q": query,
                "sortBy": "publishedAt",
                "language": language,
                "pageSize": max_articles,
                "apiKey": NEWS_API_KEY
            },
            timeout=10
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
                        content=article.get("content") or article.get("description") or ""
                    )
                )
            except:
                continue
        return articles
    except:
        return []

def curate_articles(articles: List[NewsArticle]) -> List[Dict[str, Any]]:
    """Deduplicate and clean articles"""
    curated = []
    seen_titles = set()

    for article in articles:
        if not article.title:
            continue
        title_key = article.title.strip().lower()
        if title_key in seen_titles:
            continue

        curated.append({
            "title": article.title,
            "summary": article.description[:200] + "..." if len(article.description) > 200 else article.description,
            "source": article.source,
            "url": article.url,
            "published": article.published_at,
            "content": article.content
        })
        seen_titles.add(title_key)

    return curated

def agent_fetch_news(state: NewsState) -> NewsState:
    """Agent 1: Fetch latest articles"""
    state["raw_articles"] = fetch_top_news(
        state["query"], 
        state["num_articles"],
        state.get("language", "en")
    )
    return state

def agent_curate_news(state: NewsState) -> NewsState:
    """Agent 2: Deduplicate and clean"""
    state["curated_articles"] = curate_articles(state["raw_articles"])
    return state

def prompt_chain(llm, template: str, data: Dict) -> str:
    """Execute a prompt template"""
    if not llm:
        return "Error: LLM not initialized"
    try:
        return (ChatPromptTemplate.from_template(template) | llm).invoke(data).content
    except Exception as e:
        return f"Error: {str(e)}"

def agent_generate_blog(state: NewsState) -> NewsState:
    """Agent 3: Write comprehensive blog post"""
    llm = get_llm()
    if not llm or not state["curated_articles"]:
        state["blog_content"] = "No content"
        return state
    
    articles_text = "\n\n".join([
        f"{i+1}. {article['title']} ({article['source']})\n{article['summary']}"
        for i, article in enumerate(state["curated_articles"])
    ])
    
    domain_context = f"Topic: {state.get('domain', 'general')}" if state.get('domain') else ""
    
    template = f"""Write a comprehensive, well-structured blog post analyzing these news articles:

{domain_context}

ARTICLES:
{{articles}}

Requirements:
1. **Headline** - Compelling and descriptive (8-12 words)
2. **Introduction** - Context and overview (150-200 words)
3. **Main Analysis** - 3-5 sections covering key points with insights
4. **Key Takeaways** - 5-7 bullet points of important insights
5. **Future Implications** - What comes next and why it matters
6. **Conclusion** - Summary and call to action

Style: Professional, engaging, data-driven. Total: 1000-1500 words.

BEGIN:"""
    
    state["blog_content"] = prompt_chain(llm, template, {"articles": articles_text})
    return state

def agent_generate_summary(state: NewsState) -> NewsState:
    """Agent 4: Generate executive summary"""
    llm = get_llm()
    if not llm or not state["curated_articles"]:
        state["summary_content"] = "No content"
        return state
    
    articles_text = "\n".join([f"{i+1}. {a['title']}: {a['summary']}" for i, a in enumerate(state["curated_articles"])])
    
    template = """Create a concise executive summary of these articles:

ARTICLES:
{articles}

Provide:
1. **Opening Statement** - Main theme in 1-2 sentences
2. **Key Developments** - 3-4 major points with specifics
3. **Takeaway** - What readers should remember (2-3 sentences)

Target: 150-200 words, clear and scannable."""
    
    state["summary_content"] = prompt_chain(llm, template, {"articles": articles_text})
    return state

def agent_categorize(state: NewsState) -> NewsState:
    """Agent 5: Organize by category"""
    llm = get_llm()
    if not llm or not state["curated_articles"]:
        state["categories_content"] = "No content"
        return state
    
    articles_text = "\n".join([f"- {a['title']}" for a in state["curated_articles"]])
    
    template = """Organize these articles into logical categories:

ARTICLES:
{articles}

For each category:
1. **[Category Name]**
   - List relevant articles
   - Brief explanation of why grouped together

Keep categories clear and useful. Maximum 5-7 categories."""
    
    state["categories_content"] = prompt_chain(llm, template, {"articles": articles_text})
    return state

def agent_analyze_trends(state: NewsState) -> NewsState:
    """Agent 6: Identify patterns and forecast"""
    llm = get_llm()
    if not llm or not state["curated_articles"]:
        state["trends_content"] = "No content"
        return state
    
    articles_text = "\n".join([f"- {a['title']}" for a in state["curated_articles"]])
    
    template = """Analyze trends in these articles:

ARTICLES:
{articles}

Identify and explain:
1. **Main Trends** - What patterns emerge?
2. **Most Discussed Topics** - What's getting attention?
3. **Key Developments** - Major breakthroughs or changes
4. **Market/Industry Impact** - Who benefits/loses?
5. **Timeline** - When will key events likely happen?
6. **Future Outlook** - What comes next?
7. **Global Implications** - Wider world impact?

Be specific and insightful."""
    
    state["trends_content"] = prompt_chain(llm, template, {"articles": articles_text})
    return state

def create_graph():
    """Create the LangGraph workflow"""
    workflow = StateGraph(NewsState)
    
    workflow.add_node("fetch", agent_fetch_news)
    workflow.add_node("curate", agent_curate_news)
    workflow.add_node("blog", agent_generate_blog)
    workflow.add_node("summary", agent_generate_summary)
    workflow.add_node("categories", agent_categorize)
    workflow.add_node("trends", agent_analyze_trends)
    
    workflow.set_entry_point("fetch")
    workflow.add_edge("fetch", "curate")
    workflow.add_edge("curate", "blog")
    workflow.add_edge("blog", "summary")
    workflow.add_edge("summary", "categories")
    workflow.add_edge("categories", "trends")
    workflow.add_edge("trends", END)
    
    return workflow.compile()

def explain_article(title: str, content: str) -> str:
    """Explain any article in plain English"""
    llm = get_llm()
    if not llm:
        return "AI not initialized"
    
    template = """Explain this article in simple, clear terms:

Title: {title}
Content: {content}

Provide:
1. **What Happened** (2-3 sentences)
   - What is the main news/event?
   
2. **Why It Matters** (2-3 sentences)
   - Why is this important?
   - Who is affected?
   
3. **Impact** (2-3 sentences)
   - What are the consequences?
   - What should people know?

Keep it clear and easy to understand."""
    
    return prompt_chain(llm, template, {"title": title, "content": content[:2000]})
