# Solution Overview

## Key Insight

After analyzing all 100 questions, I observed they are structured queries (boolean set operations on discrete values), not free-text questions requiring semantic understanding. For example:

- "Dishes WITH ingredient X" -> exact match, not semantic similarity
- "Dishes WITHOUT technique Y" -> set difference
- "At least 2 of [A, B, C]" -> cardinality constraint

This means traditional RAG (vector similarity) is the wrong paradigm for most questions.

## Two Pipelines Implemented

**1. Baseline RAG Pipeline** (`--pipeline rag`):

```
Question -> Embed (OpenAI) -> Retrieve top-20 (Qdrant) -> Rerank top-10 (Cohere) -> LLM Answer -> Dish IDs
```

**2. Structured Pipeline** (`--pipeline structured`) - The improved solution:

```
Question -> LLM parses to JSON filters -> Apply filters on DataFrame -> Dish IDs
```

The structured pipeline parses each question into a structured filter (ingredients, techniques, planet, restaurant, license, distance constraints, technique categories) and applies it programmatically on an in-memory DataFrame of all 287 dishes. This is both faster and more accurate than vector search for this type of query.

I evaluated 3 alternatives before choosing this approach:

- **Hybrid search** (vector + metadata filtering in Qdrant): improves Medium questions but still relies on vector similarity for core retrieval, which is imprecise for exact boolean queries.
- **Query decomposition** (break into sub-queries, combine results): handles NOT/OR better but assumes full recall on each sub-retrieval.
- **Structured search with LLM-parsed filters** (chosen): directly addresses the core problem. Near-perfect precision on exact boolean filtering, deterministic, fast, and easy to debug via the parsed filter JSON.

---

## Project Structure

```
.
├── scripts/
│   ├── run.py              # Main entry point: runs a pipeline on all 100 questions
│   └── ingest.py           # Data ingestion for the RAG pipeline (embed + store in Qdrant)
│
├── src/
│   ├── config.py           # Settings & paths (Pydantic, loads .env)
│   ├── parsing.py          # PDF menu extraction via LLM, with validation
│   ├── knowledge.py        # Auxiliary data: distance matrix, technique categories
│   ├── dataframe.py        # Builds a structured DataFrame from parsed menus
│   ├── filter_engine.py    # Applies structured JSON filters to the DataFrame
│   ├── query_parser.py     # LLM-based question -> JSON filter parsing
│   ├── chunking.py         # Converts menus to embeddings chunks (for RAG)
│   ├── rag.py              # RAG pipeline (retrieve -> rerank -> answer)
│   ├── structured_pipeline.py  # Structured filtering pipeline
│   ├── evaluation.py       # Computes Jaccard similarity score
│   └── metrics/
│       └── jaccard_similarity.py
│
├── notebooks/
│   └── 01_explore_data.ipynb   # Data exploration
│
├── outputs/                # Generated artifacts
│   ├── parsed_menus.json   # Cached LLM-extracted menus
│   ├── technique_categories.json  # Technique categories from Manuale di Cucina
│   ├── submission.csv      # Latest run output
│   └── structured_query_log.json  # Debug log for structured pipeline
│
├── Dataset/                # Provided knowledge base (from the original repo)
│
├── docker-compose.yml      # Qdrant vector store (needed only for RAG pipeline)
├── pyproject.toml          # Dependencies
├── .env.example            # Template for API keys
└── INSTRUCTIONS.md         # This file
```

---

## Setup

### Prerequisites

- Python 3.10+
- Docker (only needed for the RAG pipeline, which uses Qdrant)
- uv

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd <repo-name>

# With uv
uv sync

# For development (jupyter, pytest)
uv sync --extra dev
```

### 2. Configure API keys

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
COHERE_API_KEY=your_cohere_api_key_here
```

| Key | Used for | Required by |
|-----|----------|-------------|
| `OPENAI_API_KEY` | GPT-4o-mini (query parsing, answer generation) + text-embedding-3-small (embeddings) | Both pipelines |
| `COHERE_API_KEY` | rerank-v3.5 (semantic reranking) | RAG pipeline only |

---

## Running

### RAG Pipeline (baseline)

Requires Qdrant running via Docker:

```bash
# 1. Start Qdrant
docker compose up -d

# 2. Ingest data (parse menus, embed, store - one-time)
uv run python scripts/ingest.py

# 3. Run the pipeline
uv run python scripts/run.py --pipeline rag
```

### Structured Pipeline (improved, default)

```bash
uv run python scripts/run.py --pipeline structured
```

This will:

1. Parse all 30 menu PDFs (cached after first run in `outputs/parsed_menus.json`)
2. Build a structured DataFrame of all dishes
3. For each of the 100 questions: parse into JSON filters via LLM, apply filters, return matching dish IDs
4. Save `outputs/submission.csv`
5. Evaluate and print the Jaccard similarity score

### Evaluate a submission manually

```bash
uv run python src/evaluation.py --submission outputs/submission.csv
```

---

## Design Decisions

### PDF Parsing: LLM over regex

29 out of 30 menus use structured bullet lists, but one (Datapizza.pdf) embeds ingredients and techniques in narrative prose. Instead of writing separate parsers, a single GPT-4o-mini prompt handles both formats. The prompt references all 287 canonical dish names from `dish_mapping.json` to guide exact name output, though the extraction currently captures ~279/287 dishes (a few menus return fewer than 10). Results are cached after first run.

### Dish Name Validation

LLMs don't always return exact names. A three-step validation in `src/parsing.py` runs on each extracted name: exact match, then normalized match (whitespace + quote normalization), then fuzzy match (difflib, 0.85 threshold). Of the ~279 extracted dishes, all but 2 are resolved to canonical names automatically.

### Why Structured Search over RAG

The baseline RAG works but has a fundamental mismatch: questions are exact boolean queries, not semantic searches. Vector similarity can't express "dishes WITH X but NOT Y" or "at least 2 of [A, B, C]".

The structured pipeline fixes this: an LLM (temperature=0) converts each question into a JSON filter, which is then applied programmatically on a pandas DataFrame. The filter supports AND/OR/NOT on ingredients and techniques, cardinality constraints, restaurant/planet/license filters, technique category lookups (from Manuale di Cucina), and planet distance constraints. This makes answers deterministic, fast, and debuggable (every parsed filter is logged to `outputs/structured_query_log.json`).

---

## Approach by Question Difficulty

| Difficulty | # Questions | What's needed | Pipeline support |
|-----------|-------------|---------------|-----------------|
| **Easy** | 48 | Ingredients & techniques (AND/OR/NOT) | Fully supported by structured pipeline |
| **Medium** | 28 | + Restaurant, planet, license filters | Fully supported |
| **Hard** | 18 | + Distance constraints, technique categories from Manuale di Cucina | Supported via distance matrix lookup and technique category extraction |
| **Impossible** | 6 | + Regulatory compliance (Codice Galattico), ingredient percentages (blog posts) | Partially supported |
