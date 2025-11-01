
"""
AI Knowledge Base Builder
Main Streamlit Application
"""

import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Knowledge Base Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for basic styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧠 AI Knowledge Base Builder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform customer conversations into structured knowledge</p>', unsafe_allow_html=True)

# Welcome message
st.info("👋 Welcome! This app helps you automatically extract knowledge from customer-agent conversations.")

# Feature showcase
st.subheader("✨ What This App Does")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 📁 Upload
    Upload your conversation transcripts in CSV, Excel, or TXT format
    """)

with col2:
    st.markdown("""
    ### 🤖 Extract
    AI automatically identifies question-answer pairs from conversations
    """)

with col3:
    st.markdown("""
    ### 💾 Export
    Download your curated knowledge base in multiple formats
    """)

st.divider()

# Coming soon section
st.subheader("🚧 Development Status")

progress_data = {
    "Setup & Configuration": 100,
    "File Upload Interface": 0,
    "Data Parsing": 0,
    "AI Extraction": 0,
    "Clustering": 0,
    "Export Functionality": 0
}

for feature, progress in progress_data.items():
    st.text(feature)
    st.progress(progress / 100)
    st.caption(f"{progress}% complete")

st.divider()

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This tool uses AI to automatically build knowledge bases from conversation transcripts.
    
    **Perfect for:**
    - Customer support teams
    - Contact centers
    - Sales teams
    - Any organization with conversation data
    """)
    
    st.divider()
    
    st.header("📊 Quick Stats")
    st.metric("Version", "0.1.0")
    st.metric("Status", "In Development")
    
    st.divider()
    
    st.markdown("""
    **🔗 Links**
    - [GitHub Repo](https://github.com/Ravimoon25/knowledge-base-builder)
    - [Documentation](#)
    - [Report Issue](#)
    """)

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit and Claude AI")
