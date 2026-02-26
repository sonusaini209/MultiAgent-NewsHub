import streamlit as st
from test import create_graph, explain_article, NewsState

st.set_page_config(page_title="NewsInsight AI", page_icon="📰", layout="wide")

st.markdown("""
<style>
    body { background: #f5f5f5; }
    [data-testid="stSidebar"] { 
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e0e0e0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #1f2937;
    }
    [data-testid="stSidebar"] label { 
        color: #1f2937;
        font-weight: 600;
        font-size: 14px;
    }
    [data-testid="stSidebar"] .stSelectbox div, 
    [data-testid="stSidebar"] .stTextInput input {
        background: white !important;
        border: 1px solid #ddd !important;
        border-radius: 6px !important;
        color: #1f2937 !important;
    }
    [data-testid="stMain"] {
        background: #f5f5f5;
    }
    h1 { 
        color: #1f2937;
        font-weight: 800;
        margin-bottom: 5px;
    }
    h2 { 
        color: #1f2937;
        margin-top: 25px;
        font-weight: 700;
    }
    h3 { 
        color: #374151;
        font-weight: 600;
    }
    .stButton > button { 
        background: #1f2937;
        color: white;
        border-radius: 6px;
        font-weight: 600;
    }
    .stButton > button:hover {
        background: #111827;
    }
    .article-card { 
        background: white;
        padding: 16px;
        margin: 12px 0;
        border-left: 4px solid #1f2937;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .article-title {
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 6px;
        font-size: 1rem;
    }
    .article-source {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 10px;
    }
    .article-link {
        display: inline-block;
        background: #1f2937;
        color: white;
        padding: 8px 16px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .article-link:hover {
        background: #111827;
    }
    .content-box { 
        background: white;
        padding: 24px;
        border-radius: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #1f2937;
        line-height: 1.8;
    }
    .info-box { 
        background: #dbeafe;
        padding: 16px;
        border-left: 4px solid #1f2937;
        border-radius: 6px;
        color: #1f2937;
    }
    .success-box {
        background: #dcfce7;
        padding: 16px;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        color: #16a34a;
    }
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: 600;
        color: #6b7280;
        border: 1px solid #e5e7eb;
    }
    .stTabs [aria-selected="true"] {
        background: #1f2937 !important;
        color: white !important;
        border-color: #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

TOPICS = {
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

def render_controls():
    st.markdown("### Controls")
    
    selected_topic = st.selectbox("AI Topic:", list(TOPICS.keys()), label_visibility="collapsed")
    search_query = TOPICS[selected_topic]
    
    st.divider()
    
    use_custom = st.checkbox("Custom search")
    if use_custom:
        search_query = st.text_input("Query:", search_query, label_visibility="collapsed")
    
    st.divider()
    
    num_articles = st.slider("Articles:", 1, 50, 15, 1, label_visibility="collapsed")
    
    st.divider()
    
    if st.button(" Generate", use_container_width=True, type="primary"):
        st.session_state.should_generate = True
    
    return search_query, num_articles

def render_article_links(articles):
    st.subheader("Article Links")
    for i, article in enumerate(articles, 1):
        st.markdown(f"""
        <div class="article-card">
            <div class="article-title">{i}. {article['title']}</div>
            <div class="article-source">{article['source']}</div>
            <a href="{article['url']}" target="_blank" class="article-link">Read Article</a>
        </div>
        """, unsafe_allow_html=True)

def render_blog(blog_content):
    st.subheader("Blog Post")
    st.markdown(f'<div class="content-box">{blog_content}</div>', unsafe_allow_html=True)

def render_summary(summary_content):
    st.subheader("Summary")
    st.markdown(f'<div class="content-box">{summary_content}</div>', unsafe_allow_html=True)

def render_categories(categories_content):
    st.subheader("Categories")
    st.markdown(f'<div class="content-box">{categories_content}</div>', unsafe_allow_html=True)

def render_trends(trends_content):
    st.subheader("Trends")
    st.markdown(f'<div class="content-box">{trends_content}</div>', unsafe_allow_html=True)

def render_explain(articles):
    st.subheader("Explain Article")
    
    article_titles = [f"{i}. {a['title']}" for i, a in enumerate(articles, 1)]
    selected = st.selectbox("Select article:", article_titles)
    
    if selected:
        article_num = int(selected.split(".")[0]) - 1
        article = articles[article_num]
        
        if st.button(" Get Explanation", use_container_width=True):
            with st.spinner("Explaining..."):
                explanation = explain_article(article['title'], article['content'])
                st.markdown(f'<div class="content-box">{explanation}</div>', unsafe_allow_html=True)

def main():
    st.title(" DeepInsight News AI")
    st.write("Structured News Analysis with Predictive Insight")
    
    col1, col2 = st.columns([0.7, 0.3])
    
    with col2:
        search_query, num_articles = render_controls()
    
    # Generate
    if st.session_state.get("should_generate", False):
        st.session_state.should_generate = False
        
        st.info(" Generating report...")
        graph = create_graph()
        
        initial_state = NewsState(
            query=search_query,
            num_articles=num_articles,
            raw_articles=[],
            curated_articles=[],
            blog_content="",
            summary_content="",
            categories_content="",
            trends_content=""
        )
        
        result = graph.invoke(initial_state)
        
        st.session_state.articles = result["curated_articles"]
        st.session_state.blog = result["blog_content"]
        st.session_state.summary = result["summary_content"]
        st.session_state.categories = result["categories_content"]
        st.session_state.trends = result["trends_content"]
        st.session_state.generated = True
        
        st.success(f" Report ready! {len(result['curated_articles'])} articles analyzed")
        st.rerun()
    
    # Display
    if st.session_state.get("generated", False):
        articles = st.session_state.articles
        
        with col1:
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
                "Links", "Blog", "Summary", "Categories", "Trends", "Explain"
            ])
            
            with tab1:
                render_article_links(articles)
            
            with tab2:
                render_blog(st.session_state.blog)
            
            with tab3:
                render_summary(st.session_state.summary)
            
            with tab4:
                render_categories(st.session_state.categories)
            
            with tab5:
                render_trends(st.session_state.trends)
            
            with tab6:
                render_explain(articles)
    else:
        with col1:
            st.info(" Select topic and click Generate")

if __name__ == "__main__":
    if "should_generate" not in st.session_state:
        st.session_state.should_generate = False
    if "generated" not in st.session_state:
        st.session_state.generated = False
    
    main()