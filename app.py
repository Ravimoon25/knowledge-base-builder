"""
AI Knowledge Base Builder
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import pandas as pd
import sys
sys.path.append('.')

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Knowledge Base Builder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        text-align: center;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    .upload-section {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #667eea;
        margin: 2rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        height: 100%;
        border-left: 4px solid #667eea;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        border: none;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = None
if 'processing_started' not in st.session_state:
    st.session_state.processing_started = False

# Main header
st.markdown('<h1 class="main-header">🧠 AI Knowledge Base Builder</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Transform customer conversations into structured knowledge automatically</p>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "ℹ️ About", "📊 Stats"])

with tab1:
    # File Upload Section
    st.markdown("### 📁 Upload Your Conversation Data")
    
    with st.container():
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop your conversation files here",
            type=['csv', 'xlsx', 'txt'],
            accept_multiple_files=True,
            help="Supported formats: CSV, Excel (.xlsx), or Text files",
            key="file_uploader"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            st.session_state.uploaded_files = uploaded_files
            
            # Show uploaded files
            st.markdown("#### 📄 Uploaded Files:")
            for idx, file in enumerate(uploaded_files, 1):
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.text(f"{idx}. {file.name}")
                with col2:
                    st.text(f"Size: {file.size / 1024:.2f} KB")
                with col3:
                    st.text(f"Type: {file.type}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # File Preview Section
    if uploaded_files:
        st.markdown("---")
        st.markdown("### 👁️ File Preview & Parsing")
        
        # Select file to preview
        file_to_preview = st.selectbox(
            "Select a file to preview:",
            options=[f.name for f in uploaded_files],
            key="preview_selector"
        )
        
        if file_to_preview:
            # Find the selected file
            selected_file = next(f for f in uploaded_files if f.name == file_to_preview)
            
            # Create tabs for raw data and parsed conversations
            preview_tab1, preview_tab2 = st.tabs(["📄 Raw Data", "🔄 Parsed Conversations"])
            
            with preview_tab1:
                try:
                    # Try to read as CSV first
                    if selected_file.name.endswith('.csv'):
                        df = pd.read_csv(selected_file)
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show basic stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(df))
                        with col2:
                            st.metric("Total Columns", len(df.columns))
                        with col3:
                            st.metric("Preview Showing", min(10, len(df)))
                    
                    elif selected_file.name.endswith('.xlsx'):
                        df = pd.read_excel(selected_file)
                        st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show basic stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Rows", len(df))
                        with col2:
                            st.metric("Total Columns", len(df.columns))
                        with col3:
                            st.metric("Preview Showing", min(10, len(df)))
                    
                    elif selected_file.name.endswith('.txt'):
                        # Reset file pointer
                        selected_file.seek(0)
                        content = selected_file.read().decode('utf-8')
                        st.text_area("File Content (First 1000 characters):", 
                                    content[:1000], height=300)
                        st.info(f"Total characters: {len(content)}")
                    
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
            
            with preview_tab2:
                # Import parser
                from src.data_parser import (
                    parse_csv_conversations, 
                    parse_excel_conversations,
                    parse_text_conversations,
                    validate_conversations,
                    format_conversation_for_display
                )
                
                try:
                    # Reset file pointer
                    selected_file.seek(0)
                    
                    # Parse based on file type
                    if selected_file.name.endswith('.csv'):
                        conversations = parse_csv_conversations(selected_file)
                    elif selected_file.name.endswith('.xlsx'):
                        conversations = parse_excel_conversations(selected_file)
                    elif selected_file.name.endswith('.txt'):
                        conversations = parse_text_conversations(selected_file)
                    else:
                        st.error("Unsupported file type")
                        conversations = []
                    
                    # Validate conversations
                    validation = validate_conversations(conversations)
                    
                    if validation['valid']:
                        st.success("✅ Conversations parsed successfully!")
                    else:
                        st.warning("⚠️ Parsing completed with issues:")
                        for issue in validation['issues']:
                            st.warning(f"  • {issue}")
                    
                    # Show statistics
                    st.markdown("#### 📊 Parsing Statistics")
                    stats = validation['stats']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Conversations", stats.get('total_conversations', 0))
                    with col2:
                        st.metric("Total Messages", stats.get('total_messages', 0))
                    with col3:
                        st.metric("Avg Messages", f"{stats.get('avg_messages_per_conversation', 0):.1f}")
                    with col4:
                        st.metric("Unique Speakers", stats.get('unique_speakers', 0))
                    
                    if stats.get('speakers'):
                        st.info(f"👥 Detected speakers: {', '.join(stats['speakers'])}")
                    
                    # Show parsed conversations
                    if conversations:
                        st.markdown("#### 💬 Parsed Conversations Preview")
                        
                        # Select conversation to view
                        conv_options = [f"{conv['id']} ({conv['num_messages']} messages)" 
                                      for conv in conversations]
                        
                        selected_conv_idx = st.selectbox(
                            "Select a conversation to view:",
                            options=range(len(conversations)),
                            format_func=lambda x: conv_options[x],
                            key="conv_selector"
                        )
                        
                        # Display selected conversation
                        if selected_conv_idx is not None:
                            conversation = conversations[selected_conv_idx]
                            
                            # Format and display
                            formatted = format_conversation_for_display(conversation)
                            st.text_area(
                                "Conversation Preview:", 
                                formatted, 
                                height=400,
                                key="conv_display"
                            )
                            
                            # Store in session state for later use
                            if 'parsed_conversations' not in st.session_state:
                                st.session_state.parsed_conversations = {}
                            
                            st.session_state.parsed_conversations[file_to_preview] = conversations
                
                except Exception as e:
                    st.error(f"❌ Error parsing conversations: {str(e)}")
                    st.info("💡 Make sure your file has 'speaker' and 'message' columns (or similar)")        
        st.markdown("---")
        
        # Configuration Section
        st.markdown("### ⚙️ Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            claude_model = st.selectbox(
                "Select Claude Model",
                options=[
                    "claude-sonnet-4-5-20250929",
                    "claude-opus-4"
                ],
                index=0,
                help="Claude Sonnet 4.5 is recommended for best balance of speed and quality"
            )
        
        with col2:
            clustering_method = st.selectbox(
                "Clustering Method",
                options=["Auto-detect", "DBSCAN", "K-Means"],
                index=0,
                help="Auto-detect will choose the best method based on your data"
            )
        
        # Process Button
        st.markdown("### 🚀 Ready to Process?")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                st.session_state.processing_started = True
                st.balloons()
                st.info("🚧 Processing functionality coming soon! Stay tuned.")
    
    else:
        # Show helpful instructions when no files uploaded
        st.info("👆 Upload your conversation files above to get started")
        
        st.markdown("### 📋 Expected File Format")
        
        st.markdown("""
        Your conversation files should contain:
        - **Speaker identification** (Agent/Customer)
        - **Message text** (What was said)
        - **Optional**: Timestamp, conversation ID
        
        **Example CSV format:**
        """)
        
        example_data = pd.DataFrame({
            'speaker': ['Agent', 'Customer', 'Agent', 'Customer'],
            'message': [
                'Hello! How can I help you today?',
                'I need help with my refund policy',
                'I can help you with that. Our refund policy allows...',
                'Thank you, that\'s helpful!'
            ]
        })
        
        st.dataframe(example_data, use_container_width=True)

with tab2:
    # About section
    st.markdown("### 🎯 What This App Does")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
        <h4>📁 Upload</h4>
        <p>Upload your conversation transcripts in CSV, Excel, or TXT format. Multiple files supported!</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
        <h4>🤖 Extract</h4>
        <p>AI automatically identifies question-answer pairs from your conversations using Claude.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
        <h4>💾 Export</h4>
        <p>Download your curated knowledge base in JSON, CSV, or Markdown format.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔄 How It Works")
    
    st.markdown("""
    1. **Upload Conversations** → Upload your customer-agent conversation transcripts
    2. **AI Extraction** → Claude analyzes conversations and extracts meaningful QA pairs
    3. **Smart Clustering** → Similar questions are automatically grouped together
    4. **Deduplication** → Redundant information is identified and removed
    5. **Representative Selection** → Best QA pairs are selected for each topic
    6. **Export** → Download your curated knowledge base
    """)
    
    st.markdown("---")
    
    st.markdown("### 🛠️ Tech Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **Framework:** Streamlit
        - **AI Model:** Anthropic Claude Sonnet 4.5
        - **Embeddings:** OpenAI
        - **Clustering:** scikit-learn (DBSCAN)
        """)
    
    with col2:
        st.markdown("""
        - **Data Processing:** Pandas
        - **Visualization:** Plotly
        - **Language:** Python 3.9+
        """)

with tab3:
    # Stats section
    st.markdown("### 📊 Project Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>0.1.0</h3>
        <p>Version</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>25%</h3>
        <p>Complete</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>2/8</h3>
        <p>Milestones</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
        <h3>🚧</h3>
        <p>In Progress</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🗺️ Development Progress")
    
    progress_data = {
        "✅ Project Setup": 100,
        "✅ File Upload Interface": 100,
        "🚧 Data Parsing": 0,
        "🚧 AI Extraction": 0,
        "🚧 Clustering": 0,
        "🚧 Representative Selection": 0,
        "🚧 Export Functionality": 0,
        "🚧 Final Polish": 0
    }
    
    for feature, progress in progress_data.items():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.text(feature)
            st.progress(progress / 100)
        with col2:
            st.text(f"{progress}%")

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/brain.png", width=150)
    
    st.markdown("---")
    
    st.header("📖 Quick Guide")
    st.markdown("""
    1. Upload your files
    2. Preview the data
    3. Configure settings
    4. Start processing
    5. Download results
    """)
    
    st.markdown("---")
    
    st.header("💡 Tips")
    st.info("""
    **File Format Tips:**
    - Use clear column names
    - Include speaker labels
    - One conversation per file works best
    """)
    
    st.markdown("---")

    
    st.header("🔧 API Status")
    
    # Test Claude connection
    if st.button("Test Claude API", key="test_api"):
        with st.spinner("Testing connection..."):
            try:
                from src.claude_client import get_claude_client, test_claude_connection
                client = get_claude_client()
                
                if test_claude_connection(client):
                    st.success("✅ Claude API connected!")
                else:
                    st.error("❌ Connection test failed")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    st.header("🔗 Links")
    st.markdown("""
    - [GitHub Repo](https://github.com/Ravimoon25/knowledge-base-builder)
    - [Report Issue](https://github.com/Ravimoon25/knowledge-base-builder/issues)
    """)
    
    st.markdown("---")
    
    st.caption("Built with ❤️ using Streamlit & Claude AI")
    st.caption("Version 0.1.0")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🧠 AI Knowledge Base Builder | Powered by Anthropic Claude</p>
</div>
""", unsafe_allow_html=True)
