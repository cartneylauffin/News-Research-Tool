# 📰 News Research Tool

An AI-powered tool to analyze news articles URL, create a vector-based knowledge index, and answer natural language questions with relevant context.

---

## 🚀 Features
- Ingest multiple news articles via URLs
- Upload and process PDFs
- Automatic text splitting & chunking
- Embeddings with OpenAI (or alternative LLMs)
- Fast semantic search using FAISS
- Ask questions and get contextual answers with sources
- Simple Streamlit user interface

---

## 🛠️ Getting Started

### Prerequisites
- Python **3.8+**
- [OpenAI API key](https://platform.openai.com/)

### Installation
```bash
# Clone the repository
git clone https://github.com/cartneylauffin/News-Research-Tool.git
cd News-Research-Tool

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
````

### Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your_api_key_here"   # Linux / macOS
setx OPENAI_API_KEY "your_api_key_here"     # Windows
```

Or place it in a `.env` file:

```
OPENAI_API_KEY=your_api_key_here
```

### Run the Application

```bash
streamlit run app.py
```

---

## 💡 Usage

1. Enter URLs of news articles.
2. Click **Process** to create embeddings & build the FAISS index.
3. Ask questions in natural language (e.g., *“What is the growth outlook?”*).
4. Get AI-generated answers with relevant source snippets.

---

## ⚠️ Limitations

* Requires OpenAI API (cost + internet access).
* Large articles may lose context due to token limits.
* Parsing issues possible with malformed HTML or PDFs.
* Works best with English text.

---

## 🔮 Future Enhancements

* Support for additional LLMs (Cohere, HuggingFace models)
* Live news integration via RSS feeds
* OCR support for scanned PDFs
* Source highlighting in answers
* Multi-language support


