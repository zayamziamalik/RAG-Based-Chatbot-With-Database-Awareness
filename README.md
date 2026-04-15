# RAG Based ChatBot

A production-style Retrieval-Augmented Generation (RAG) chatbot built with Django, LangGraph, and MySQL.  
The system combines document retrieval from local files with SQL-backed analytics to produce fast, context-aware, and privacy-safe responses.

## Features

- LangGraph-driven workflow for routing, retrieval, tool calls, and response generation
- Hybrid retrieval (vector + keyword/BM25) with reranking and context compression
- Groq/OpenAI/Ollama compatible LLM configuration
- MySQL-backed structured analytics for count and aggregate queries
- Privacy guardrails to avoid exposing internal schema metadata in user-facing responses
- Django web UI + CLI support
- Async knowledge refresh to avoid UI blocking during reloads

## Project Structure

- `graph/workflow.py` - main LangGraph orchestration and routing logic
- `retrievers/hybrid.py` - hybrid retrieval pipeline
- `rag/loaders.py` - ingestion for documents and database-backed content
- `utils/tools.py` - SQL/tool layer for analytics and schema-aware operations
- `utils/schema_guard.py` - schema probe detection and privacy-safe responses
- `utils/redact.py` - user-facing redaction utilities
- `rag/service.py` - service entry used by UI/CLI
- `chat_ui/views.py` - Django chat handlers and refresh endpoints
- `init_tables.sql` - SQL schema and seed script
- `init_db.py` - optional database bootstrap utility

## Tech Stack

- Backend: Django, LangChain, LangGraph
- LLM providers: Groq / OpenAI / Ollama
- Embeddings: HuggingFace or OpenAI-compatible endpoint
- Vector store: Chroma or FAISS
- Database: MySQL (via `DATABASE_URL`)

## Setup

### 1) Create and activate virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3) Configure environment variables

Copy `.env.example` to `.env` and update values:

```powershell
copy .env.example .env
```

Minimum important keys:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (Groq example: `https://api.groq.com/openai/v1`)
- `LLM_PROVIDER` and `LLM_MODEL`
- `EMBEDDING_PROVIDER` and `EMBEDDING_MODEL`
- `DATABASE_URL` (MySQL)
- `TEXT_FILES_DIR`

Example MySQL URL:

```env
DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3310/smartphone
```

### 4) Initialize database (optional but recommended)

```powershell
python init_db.py
```

## Running the App

### Django Web UI

```powershell
python manage.py migrate
python manage.py runserver 127.0.0.1:8001
```

Open: [http://127.0.0.1:8001](http://127.0.0.1:8001)

### CLI Mode

```powershell
python main.py
```

## Data and Knowledge Sources

The chatbot can use:

- Local files in `data/text_files` (PDF, TXT, DOCX, JSON, HTML, XML, YAML, etc.)
- MySQL data through the configured SQL tools

Use the UI refresh action to re-index knowledge after adding/updating files.

## Privacy and Safety Behavior

- The assistant is configured to avoid exposing internal table/column names in user-visible replies.
- Schema metadata probes are intercepted and answered with privacy-safe messaging.
- Response redaction is applied to hide restricted identifiers from final output.

## Logging and Debug

- Enable debug in `.env` with `DEBUG_MODE=true`
- Pipeline logs are written to `data/logs/rag_pipeline.log`

## Common Issues

- **Embeddings error with HuggingFace**: install `sentence-transformers`
- **MySQL connection errors**: verify host/port/user/password in `DATABASE_URL`
- **Django DB/version issues**: project is configured for `django>=4.2,<5`
- **Port in use**: run server on another port or stop the existing process

## Build .env in root directory

-OPENAI_API_KEY=Your API Key
-OPENAI_BASE_URL=https://api.groq.com/openai/v1
-OPENAI_MODEL=llama-3.3-70b-versatile
-LLM_PROVIDER=groq
-LLM_MODEL=llama-3.3-70b-versatile
-EMBEDDING_PROVIDER=huggingface
-EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
-RERANKER_TYPE=llm
-DATABASE_URL=mysql+pymysql://root:@127.0.0.1:3310/smartphone
-TEXT_FILES_DIR=data/text_files
-MAX_CHUNK_SIZE=900
-CHUNK_OVERLAP=120
-TOP_K=6
-DJANGO_SECRET_KEY=change_me_for_production
-DJANGO_DEBUG=true
-DJANGO_ALLOWED_HOSTS=*
