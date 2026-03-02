#  MultiAgent NewsHub

> AI-Powered Multi-Agent News Intelligence Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://console.groq.com)
[![Vercel](https://img.shields.io/badge/Deployed%20on-Vercel-black.svg)](https://your-app.vercel.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

##  Live Demo

**https://multi-agent-news-hub.vercel.app/**

> Replace with your actual Vercel URL after deploying.

---

##  Overview

MultiAgent NewsHub orchestrates **6 specialized AI agents** using LangGraph to fetch, analyze, and summarize news articles in real-time. Pick a topic, hit Generate, and get a full blog post, executive summary, categorized articles, and trend analysis — instantly.

---

##  Features

- 🤖 **6 Specialized AI Agents** — each focused on a specific task
- 📰 **10 Predefined AI/ML Topics** — or enter any custom query
- 📊 **5 Output Tabs** — Articles, Blog, Summary, Categories, Trends
- 🔍 **Article Explainer** — plain-English breakdown of any article on demand
- 🌐 **Built-in Web UI** — dark-themed, responsive, no extra setup
- 🚀 **Deployed on Vercel** — accessible from anywhere

---

##  Agent Pipeline

```
NewsAPI
   ↓
1️⃣  Fetcher        →  Pull latest articles from NewsAPI
2️⃣  Curator        →  Deduplicate and clean articles
3️⃣  Blogger        →  Write a comprehensive blog post
4️⃣  Summarizer     →  Generate an executive summary
5️⃣  Categorizer    →  Group articles by topic
6️⃣  Trend Analyzer →  Detect patterns and forecast trends

On Demand:
  Explainer      →  Plain-English explanation of any article
```

---

##  Project Structure

```
MultiAgent-NewsHub/
├── agent.py           # 6 AI agents + LangGraph workflow
├── app.py             # FastAPI backend — serves UI + API routes
├── index.html         # Frontend (dark UI, no framework)
├── requirements.txt   # Python dependencies
├── vercel.json        # Vercel deployment config
├── .env               # API keys — never commit this
├── .gitignore         # Excludes .env
└── README.md
```

---

## 🔑 API Keys Required

You need two free API keys.

**1. NewsAPI** — [newsapi.org](https://newsapi.org)
- Sign up for a free account
- Copy your API key (100 requests/day on free tier)

**2. Groq API** — [console.groq.com](https://console.groq.com)
- Sign up for a free account
- Generate an API key
- Model used: `llama-3.3-70b-versatile`

Create a `.env` file in the project root:

```env
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=gsk_your_groq_key_here
```

> ⚠️ `.gitignore` already excludes `.env` — never push your keys to GitHub.

---

##  Run Locally

**Prerequisites:** Python 3.9+, pip

1. **Clone the repository**
   ```bash
   git clone https://github.com/sonusaini209/MultiAgent-NewsHub
   cd MultiAgent-NewsHub
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv

   # macOS / Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your API keys**

   Create a `.env` file:
   ```env
   NEWS_API_KEY=your_newsapi_key_here
   GROQ_API_KEY=gsk_your_groq_key_here
   ```

5. **Start the server**
   ```bash
   uvicorn app:app --reload
   ```

   Open [http://localhost:8000](http://localhost:8000) in your browser.


##  License

MIT — see [LICENSE](./LICENSE) for details.

</div>
