# 📰 MultiAgent-NewsHub

> **AI-Powered Multi-Agent News Intelligence Platform** | Orchestrated Analysis | Real-Time Insights

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red.svg)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/Powered%20by-LangGraph-green.svg)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://console.groq.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Overview

**MultiAgent-NewsHub** is a sophisticated multi-agent intelligence platform that orchestrates 6 specialized AI agents to analyze, summarize, categorize, and forecast trends from news articles in real-time.

Using **LangGraph** for workflow orchestration and **Groq** for ultra-fast LLM inference, this system delivers enterprise-grade news intelligence with minimal latency.

## Live Demo
https://multiagent-newshub.vercel.app

### ✨ Key Highlights

- 🤖 **6 Specialized AI Agents** - Each focused on specific analysis tasks
- 🔄 **LangGraph Orchestration** - Seamless workflow coordination
- ⚡ **Groq LLM** - Lightning-fast inference (70B parameters)
- 📊 **Multi-Dimensional Analysis** - Blog, Summary, Categories, Trends, Explanations
- 🎨 **Beautiful UI** - Professional Streamlit interface
- 🏗️ **Clean Architecture** - Separated logic and UI modules
- 📈 **10 AI Topics** - Focused on AI/ML domain
- 🔧 **Production Ready** - Error handling, caching, optimization

---

## 🚀 Quick Start

### Prerequisites

```bash
✓ Python 3.9 or higher
✓ pip or conda
✓ NewsAPI Key (free at newsapi.org)
✓ Groq API Key (free at console.groq.com)
```

### Installation

1. **Clone repository**
```bash
git clone https://github.com/yourusername/MultiAgent-NewsHub.git
cd MultiAgent-NewsHub
```

2. **Create environment**
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup API keys**
```bash
cp .env.example .env
# Edit .env with your keys
```

5. **Run application**
```bash
streamlit run ui.py
```

🌐 Opens at http://localhost:8501

---

## 📋 Dependencies

```
streamlit==1.28.1
langchain-groq==0.1.5
langgraph==0.1.15
langchain-core==0.1.35
pydantic==2.5.0
requests==2.31.0
python-dotenv==1.0.0
```

---

## 🔑 API Setup

### NewsAPI (Free)
- Visit: https://newsapi.org
- Create account
- Copy API key
- Add to .env: `NEWS_API_KEY=your_key`

### Groq API (Free)
- Visit: https://console.groq.com
- Create account
- Generate API key
- Add to .env: `GROQ_API_KEY=gsk_your_key`

### .env Template
```env
NEWS_API_KEY=your_newsapi_key_here
GROQ_API_KEY=gsk_your_groq_key_here
```

---

## 🤖 Architecture

### Multi-Agent Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    LANGGRAPH ORCHESTRATION                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  1️⃣ FETCHER          → Get articles from NewsAPI            │
│         ↓                                                     │
│  2️⃣ CURATOR          → Remove duplicates                     │
│         ↓                                                     │
│  3️⃣ BLOGGER          → Write comprehensive blog             │
│         ↓                                                     │
│  4️⃣ SUMMARIZER       → Executive summary                    │
│         ↓                                                     │
│  5️⃣ CATEGORIZER      → Organize by topics                   │
│         ↓                                                     │
│  6️⃣ TREND ANALYZER   → Find patterns & trends               │
│         ↓                                                     │
│  ON-DEMAND: EXPLAINER → Deep dive articles                  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Project Structure

```
MultiAgent-NewsHub/
├── logic.py              # Core agents & business logic
├── ui.py                 # Streamlit interface
├── requirements.txt      # Dependencies
├── .env.example          # Environment template
├── README.md             # Documentation
└── .gitignore            # Git ignore
```

---

## 📖 How to Use

### Step 1: Select Topic
Choose from 10 predefined AI/ML topics or enter custom query

### Step 2: Configure
- Set article count (1-50)
- Enable custom search (optional)

### Step 3: Generate
Click "🚀 Generate" button and wait (1-2 minutes)

### Step 4: Explore
- **Links** - All article URLs
- **Blog** - Comprehensive analysis
- **Summary** - Quick overview
- **Categories** - Organized articles
- **Trends** - Pattern analysis
- **Explain** - Deep dive any article

### Available Topics

```
🤖 AI & Machine Learning     🧠 Deep Learning
🤖 Large Language Models     🔬 AI Research
🏢 AI in Business            🤖 Generative AI
🧬 AI in Healthcare          🚗 AI in Autonomous Systems
📊 Data Science & AI         🎨 AI & Creativity
```

---

## 🔌 Integrations

### NewsAPI
- **Source**: Global news aggregation
- **Limit**: 100 requests/day (free tier)
- **Docs**: newsapi.org/docs

### Groq API
- **Model**: llama-3.3-70b-versatile
- **Speed**: 5x faster than alternatives
- **Cost**: Free tier available
- **Docs**: console.groq.com

---

## ✨ Features

### Intelligence
✅ Smart summarization | ✅ Trend detection | ✅ Auto-categorization
✅ Deep explanations | ✅ Professional blogs

### Technical
✅ LangGraph orchestration | ✅ Fast inference | ✅ Error handling
✅ Session caching | ✅ Clean architecture

### Experience
✅ Beautiful UI | ✅ Light theme | ✅ Responsive | ✅ Interactive tabs
✅ Real-time feedback

---

## 📊 Performance

| Feature | Time |
|---------|------|
| Fetch articles | ~2s |
| Generate blog | ~20s |
| Full report | ~45s |
| Explanation | ~10s |
| Max concurrent users | 5+ |

---

## 🐛 Troubleshooting

**"API Key not set"**
- Create .env file
- Add both API keys
- Restart app

**"No articles found"**
- Verify API key
- Check search query
- Check API limits

**"LLM timeout"**
- Reduce article count
- Check internet
- Try again later

**"Port already in use"**
```bash
streamlit run ui.py --server.port 8502
```

---

## 🚀 Deployment

### Streamlit Cloud (Easiest)
1. Push to GitHub
2. Visit share.streamlit.io
3. Deploy!

### Docker
```bash
docker build -t multiagent-newshub .
docker run -p 8501:8501 multiagent-newshub
```

---

## 🤝 Contributing

Contributions welcome! 

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push & create PR

### Ideas
- Better UI/UX
- New agents
- More news sources
- Data visualizations
- Authentication system

---

## 📝 License

MIT License - Open source for everyone

---

## 🙏 Credits

- **LangChain** - LangGraph orchestration
- **Groq** - Ultra-fast LLM inference
- **Streamlit** - Beautiful web framework
- **NewsAPI** - Global news data

---

<div align="center">

### ⭐ Found it useful? Give us a star!

Made with ❤️ by [Your Name](https://github.com/yourusername)

[GitHub](https://github.com/yourusername/MultiAgent-NewsHub) • [Issues](https://github.com/yourusername/MultiAgent-NewsHub/issues) • [Features](https://github.com/yourusername/MultiAgent-NewsHub/issues)

**Version**: 1.0.0 | **Last Updated**: January 2025

</div>
