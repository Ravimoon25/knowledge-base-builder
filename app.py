"""
AI Knowledge Base Builder
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import pandas as pd
import sys
import time
import json
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

        st.markdown("---")
        
        # Single Conversation Extraction Demo
        st.markdown("### 🤖 AI Extraction Demo")
        st.info("👉 Try extracting QA pairs from a single conversation using Claude AI")
        
        if 'parsed_conversations' in st.session_state and file_to_preview in st.session_state.parsed_conversations:
            conversations = st.session_state.parsed_conversations[file_to_preview]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Select conversation for extraction
                conv_options = [f"{conv['id']} ({conv['num_messages']} messages)" 
                              for conv in conversations]
                
                selected_conv_for_extraction = st.selectbox(
                    "Select a conversation to extract knowledge from:",
                    options=range(len(conversations)),
                    format_func=lambda x: conv_options[x],
                    key="extraction_conv_selector"
                )
            
            with col2:
                extraction_model = st.selectbox(
                    "Claude Model:",
                    options=["claude-sonnet-4-5-20250929", "claude-opus-4"],
                    index=0,
                    key="extraction_model"
                )
            
            # Extract button
            if st.button("🚀 Extract QA Pairs", type="primary", key="extract_single"):
                from src.claude_client import get_claude_client
                from src.extractor import extract_qa_pairs
                
                conversation = conversations[selected_conv_for_extraction]
                
                with st.spinner(f"🤖 Claude is analyzing conversation {conversation['id']}..."):
                    try:
                        # Get Claude client
                        client = get_claude_client()
                        
                        # Extract QA pairs
                        result = extract_qa_pairs(
                            client=client,
                            conversation=conversation,
                            model=extraction_model
                        )
                        
                        # Store result in session state
                        if 'extraction_results' not in st.session_state:
                            st.session_state.extraction_results = {}
                        
                        st.session_state.extraction_results[conversation['id']] = result
                        
                        # Display results
                        if result['success']:
                            st.success(f"✅ Extraction complete! Found {result['num_qa_pairs']} QA pairs")
                            
                            # Show statistics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("QA Pairs Extracted", result['num_qa_pairs'])
                            with col2:
                                st.metric("Estimated Cost", f"${result['estimated_cost']:.4f}")
                            with col3:
                                st.metric("Input Tokens", result['input_tokens'])
                            
                            # Display extracted QA pairs
                            if result['qa_pairs']:
                                st.markdown("#### 📚 Extracted Knowledge")
                                
                                for idx, qa in enumerate(result['qa_pairs'], 1):
                                    with st.expander(f"QA Pair {idx}: {qa.get('question', 'No question')[:80]}...", expanded=(idx == 1)):
                                        st.markdown(f"**❓ Question:**")
                                        st.info(qa.get('question', 'No question'))
                                        
                                        st.markdown(f"**✅ Answer:**")
                                        st.success(qa.get('answer', 'No answer'))
                                        
                                        if 'justification' in qa:
                                            st.markdown(f"**💡 Justification:**")
                                            st.caption(qa['justification'])
                            else:
                                st.warning("No QA pairs found in this conversation")
                        
                        else:
                            st.error(f"❌ Extraction failed: {result['error']}")
                    
                    except Exception as e:
                        st.error(f"❌ Error during extraction: {str(e)}")
                        st.exception(e)
            
            # Show previous results if they exist
            if 'extraction_results' in st.session_state:
                stored_results = st.session_state.extraction_results
                if stored_results:
                    st.markdown("---")
                    st.markdown("#### 📊 Previous Extractions")
                    
                    for conv_id, result in stored_results.items():
                        if result['success']:
                            st.text(f"✅ {conv_id}: {result['num_qa_pairs']} QA pairs (${result['estimated_cost']:.4f})")
                        else:
                            st.text(f"❌ {conv_id}: Failed")
        # Configuration Section
        st.markdown("---")
        
        # Batch Processing Section
        st.markdown("### ⚙️ Batch Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_model = st.selectbox(
                "Select Claude Model",
                options=[
                    "claude-sonnet-4-5-20250929",
                    "claude-opus-4"
                ],
                index=0,
                help="Claude Sonnet 4.5 is recommended for best balance of speed and quality",
                key="batch_model"
            )
        
        with col2:
            # Calculate total conversations across all files
            total_convs = 0
            if 'parsed_conversations' in st.session_state:
                for convs in st.session_state.parsed_conversations.values():
                    total_convs += len(convs)
            
            st.metric("Total Conversations Ready", total_convs)
        
        st.markdown("### 🚀 Ready to Extract Knowledge?")
        
        if total_convs > 0:
            # Show cost estimation
            est_cost_per_conv = 0.015  # Rough estimate
            estimated_total_cost = total_convs * est_cost_per_conv
            
            st.info(f"💰 Estimated cost for {total_convs} conversations: ${estimated_total_cost:.2f}")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Start Batch Processing", type="primary", use_container_width=True, key="batch_process"):
                    st.session_state.processing_started = True
        
        # Batch Processing Execution
        if st.session_state.get('processing_started', False):
            st.markdown("---")
            st.markdown("### 🔄 Processing in Progress...")
            
            from src.claude_client import get_claude_client
            from src.batch_processor import process_conversations_batch, aggregate_results, export_qa_pairs_to_dict
            
            try:
                # Get all conversations
                all_conversations = []
                if 'parsed_conversations' in st.session_state:
                    for convs in st.session_state.parsed_conversations.values():
                        all_conversations.extend(convs)
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Get Claude client
                client = get_claude_client()
                
                # Process conversations
                results = []
                total = len(all_conversations)
                
                for idx, conversation in enumerate(all_conversations):
                    # Update progress
                    progress = (idx + 1) / total
                    progress_bar.progress(progress)
                    status_text.text(f"Processing conversation {idx + 1}/{total}: {conversation['id']}")
                    
                    # Extract QA pairs
                    result = extract_qa_pairs(client, conversation, batch_model)
                    results.append(result)
                    
                    # Small delay
                    if idx < total - 1:
                        time.sleep(0.5)
                
                progress_bar.progress(1.0)
                status_text.text("✅ Processing complete!")
                
                # Aggregate results
                summary = aggregate_results(results)
                
                # Store in session state
                st.session_state.batch_results = summary
                st.session_state.processing_started = False
                
                # Show success message
                st.balloons()
                st.success(f"🎉 Extraction complete! Found {summary['total_qa_pairs']} QA pairs from {summary['successful']} conversations")
                
                # Rerun to show results
                st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error during batch processing: {str(e)}")
                st.session_state.processing_started = False
        
        # Display Batch Results
        if 'batch_results' in st.session_state:
            summary = st.session_state.batch_results
            
            st.markdown("---")
            st.markdown("### 📊 Batch Processing Results")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Conversations", summary['total_conversations'])
            with col2:
                st.metric("QA Pairs Extracted", summary['total_qa_pairs'])
            with col3:
                st.metric("Success Rate", f"{summary['success_rate']:.1f}%")
            with col4:
                st.metric("Total Cost", f"${summary['total_cost']:.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Successful", summary['successful'], delta_color="normal")
            with col2:
                st.metric("Failed", summary['failed'], delta_color="inverse")
            
            # Show all extracted QA pairs
            if summary['all_qa_pairs']:
                st.markdown("---")
                st.markdown(f"### 📚 All Extracted QA Pairs ({len(summary['all_qa_pairs'])} total)")
                
                # Search/filter
                search_query = st.text_input("🔍 Search QA pairs:", key="qa_search")
                
                # Filter QA pairs
                filtered_qa_pairs = summary['all_qa_pairs']
                if search_query:
                    filtered_qa_pairs = [
                        qa for qa in summary['all_qa_pairs']
                        if search_query.lower() in qa.get('question', '').lower() or
                           search_query.lower() in qa.get('answer', '').lower()
                    ]
                
                st.info(f"Showing {len(filtered_qa_pairs)} of {len(summary['all_qa_pairs'])} QA pairs")
                
                # Display QA pairs
                for idx, qa in enumerate(filtered_qa_pairs, 1):
                    with st.expander(f"QA {idx}: {qa.get('question', 'No question')[:80]}...", expanded=False):
                        st.markdown(f"**❓ Question:**")
                        st.info(qa.get('question', 'No question'))
                        
                        st.markdown(f"**✅ Answer:**")
                        st.success(qa.get('answer', 'No answer'))
                        
                        if 'justification' in qa:
                            st.markdown(f"**💡 Justification:**")
                            st.caption(qa['justification'])
                        
                        st.caption(f"📝 Source: {qa.get('source_conversation', 'unknown')}")
                
                # Export options
                st.markdown("---")
                st.markdown("### 💾 Export Knowledge Base")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Export as JSON
                    import json
                    export_data = export_qa_pairs_to_dict(summary['all_qa_pairs'])
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        label="📄 Download JSON",
                        data=json_str,
                        file_name="knowledge_base.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    # Export as CSV
                    import pandas as pd
                    df = pd.DataFrame(export_qa_pairs_to_dict(summary['all_qa_pairs']))
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📊 Download CSV",
                        data=csv,
                        file_name="knowledge_base.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    # Export as Markdown
                    md_lines = ["# Knowledge Base\n"]
                    for idx, qa in enumerate(export_qa_pairs_to_dict(summary['all_qa_pairs']), 1):
                        md_lines.append(f"## QA Pair {idx}\n")
                        md_lines.append(f"**Question:** {qa['question']}\n")
                        md_lines.append(f"**Answer:** {qa['answer']}\n")
                        md_lines.append(f"**Source:** {qa['source']}\n")
                        md_lines.append("---\n")
                    
                    md_str = "\n".join(md_lines)
                    st.download_button(
                        label="📝 Download Markdown",
                        data=md_str,
                        file_name="knowledge_base.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
            
            # Show failed conversations if any
            if summary['failed_conversations']:
                st.markdown("---")
                st.markdown("### ⚠️ Failed Conversations")
                for failed in summary['failed_conversations']:
                    st.error(f"❌ {failed['id']}: {failed['error']}")
    
    else:
        # Show helpful instructions when no files uploaded
        st.info("👆 Upload your conversation files above to get started")    
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
