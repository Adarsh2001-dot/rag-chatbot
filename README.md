# RAG Chatbot 🔍

A full end-to-end **Retrieval-Augmented Generation (RAG)** pipeline built with LangChain, FAISS, and OpenAI. Upload any PDF or text file and ask questions about its content — answers are grounded in the document, not hallucinated.

## What is RAG?

RAG combines LLMs with external knowledge retrieval:

```
Document → Chunk → Embed → FAISS Index
                                ↓
User query → Embed → Similarity Search → Top-k chunks → LLM → Answer
```

Instead of relying on training data alone, the LLM answers using retrieved passages from your document — making responses accurate, traceable, and up-to-date.

## Project Structure

```
rag-chatbot/
├── src/
│   ├── rag_pipeline.py   # Core RAG logic (chunking, embedding, retrieval, generation)
│   ├── chatbot.py        # CLI chatbot interface
│   ├── app.py            # Streamlit web UI
│   └── evaluate.py       # Compare chunking strategies
├── data/
│   └── sample.txt        # Sample document to test with
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
export OPENAI_API_KEY="sk-..."
```

### 3. Run the CLI chatbot

```bash
# Use the included sample document
python src/chatbot.py --file data/sample.txt

# Use your own PDF
python src/chatbot.py --file path/to/your.pdf

# Save the FAISS index so you don't re-embed every time
python src/chatbot.py --file data/sample.txt --save-index faiss_index

# Load a saved index
python src/chatbot.py --index faiss_index
```

### 4. Run the web UI (Streamlit)

```bash
streamlit run src/app.py
```

Opens at `http://localhost:8501` — upload a file, set your API key in the sidebar, and chat.

## CLI Commands

Once the chatbot is running:

| Input | What it does |
|---|---|
| Any question | Runs the full RAG pipeline and returns an answer |
| `sources` | Toggle source chunk display on/off |
| `search <query>` | Show raw similarity scores for a query |
| `quit` | Exit |

## Configuration

```bash
python src/chatbot.py --file doc.pdf \
  --chunk-size 500    \  # Characters per chunk (default: 500)
  --chunk-overlap 50  \  # Overlap between chunks (default: 50)
  --top-k 3           \  # Chunks retrieved per query (default: 3)
  --model gpt-4o-mini    # LLM model (default: gpt-3.5-turbo)
```

## Evaluate Chunking Strategies

Compare how different chunk sizes affect retrieval quality:

```bash
python src/evaluate.py --file data/sample.txt
```

## Tech Stack

| Component | Tool |
|---|---|
| LLM | OpenAI GPT-3.5 / GPT-4o |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector store | FAISS (local) |
| Orchestration | LangChain |
| Web UI | Streamlit |
| Document loading | PyPDF, TextLoader |

## Key Concepts

- **Chunking** — splits documents at natural language boundaries (paragraphs → sentences → words) with configurable overlap so context isn't lost at boundaries
- **Embeddings** — converts text to 1,536-dimensional vectors; semantically similar text clusters together in vector space
- **Cosine similarity** — measures angle between query and chunk vectors to rank relevance; robust to document length differences
- **FAISS** — approximate nearest neighbor search over all chunk vectors in milliseconds using the HNSW algorithm
- **Prompt engineering** — system prompt instructs the LLM to answer only from retrieved context, preventing hallucination

## Author

Built by [Adarsh](https://github.com/YOUR_USERNAME) — MS Information Systems, Binghamton University
