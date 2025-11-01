# 🧠 AI Knowledge Base Builder

> Transform customer-agent conversations into structured knowledge bases automatically using AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Overview

AI Knowledge Base Builder automatically extracts question-answer pairs from historical customer service conversations and creates a curated knowledge base. Perfect for contact centers, customer support teams, and any organization looking to leverage their conversation history.

## ✨ Features

- 📁 **Multi-format Support** - Upload CSV, Excel, or TXT files
- 🤖 **AI-Powered Extraction** - Uses Claude AI to extract meaningful QA pairs
- 🎯 **Smart Clustering** - Groups similar questions automatically
- ✂️ **Deduplication** - Removes redundant information
- 📊 **Interactive Dashboard** - Visualize and explore your knowledge base
- 💾 **Multiple Export Formats** - JSON, CSV, Markdown

## 🚀 Quick Start

### Try it Live
[Launch App →](https://your-app-url.streamlit.app) *(Coming soon)*

### Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/knowledge-base-builder.git
cd knowledge-base-builder

# Install dependencies
pip install -r requirements.txt

# Set up your API keys
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml and add your API keys

# Run the app
streamlit run app.py
```

## 📋 Requirements

- Python 3.9+
- Anthropic API key (Claude)
- OpenAI API key (for embeddings)

## 🏗️ Project Structure
```
knowledge-base-builder/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── src/                      # Source code modules
│   ├── __init__.py
│   ├── data_parser.py       # Data parsing utilities
│   ├── extractor.py         # QA extraction logic
│   ├── clusterer.py         # Clustering algorithms
│   ├── representative.py    # Representative selection
│   └── utils.py             # Helper functions
├── prompts/                  # AI prompts
│   ├── extraction_prompt.txt
│   └── representative_prompt.txt
├── tests/                    # Test data
│   └── sample_conversations.csv
└── docs/                     # Documentation
    └── USER_GUIDE.md
```

## 🎬 How It Works

1. **Upload Conversations** - Upload your customer-agent conversation transcripts
2. **Extract Knowledge** - AI analyzes conversations and extracts QA pairs
3. **Cluster & Deduplicate** - Similar questions are grouped together
4. **Select Representatives** - Best QA pairs are selected for each topic
5. **Export** - Download your curated knowledge base

## 🛠️ Tech Stack

- **Framework:** Streamlit
- **AI Models:** Anthropic Claude (Sonnet 4.5)
- **Embeddings:** OpenAI text-embedding-3-small
- **Clustering:** scikit-learn (DBSCAN)
- **Data Processing:** Pandas

## 📈 Roadmap

- [x] Project setup
- [ ] File upload and parsing
- [ ] Claude integration for extraction
- [ ] Batch processing
- [ ] Clustering implementation
- [ ] Representative selection
- [ ] Interactive dashboard
- [ ] Export functionality
- [ ] Streamlit Cloud deployment

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [Anthropic Claude](https://www.anthropic.com/)
- Inspired by the research paper: "AI Knowledge Assist: An Automated Approach for the Creation of Knowledge Bases for Conversational AI Agents"

## 📞 Contact

Questions? Feel free to [open an issue](https://github.com/YOUR-USERNAME/knowledge-base-builder/issues)

---

**⭐ Star this repo if you find it useful!**
```

**Replace `YOUR-USERNAME` with your actual GitHub username**

---

### Step 3: Verify Repository Structure

After creating the repository, you should see:
```
knowledge-base-builder/
├── .gitignore         ✅ (Python template)
├── LICENSE            ✅ (MIT)
└── README.md          ✅ (Updated with above content)
