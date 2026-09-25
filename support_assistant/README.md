# Zepto Support Assistant

## Overview

The Zepto Support Assistant is a local-first policy question-answering system built for the capstone Support Assistant module.

The system uses:

- SentenceTransformers `all-MiniLM-L6-v2` for local embeddings
- ChromaDB for persistent vector retrieval
- LangGraph `StateGraph` for workflow orchestration
- Pydantic for structured responses
- FastAPI for the `/ask` API
- Deterministic `MOCK_LLM=1` mode by default

The default mock mode does not require an API key or external LLM call.

## Architecture

```text
User
  |
  v
POST /ask
  |
  v
FastAPI API Layer
  |
  v
ask(query)
  |
  v
classify_intent
  |
  +-----------------------+
  |                       |
policy                  general
  |                       |
  v                       v
retrieve_and_answer    direct_answer
  |                       |
  v                       v
ChromaDB retrieval     Canned response
  |
  v
Top 3 policy documents
  |
  v
Structured Pydantic output
  |
  v
answer + sources + confidence
```

## Policy Corpus

The assistant uses exactly eight policy documents:

```text
docs/
├── doc_01.txt    Delivery Policy
├── doc_02.txt    Returns & Refunds
├── doc_03.txt    Membership Tiers
├── doc_04.txt    Order Tracking
├── doc_05.txt    Cancellation
├── doc_06.txt    Damaged/Missing Items
├── doc_07.txt    Gift Cards
└── doc_08.txt    Support Hours
```

Each document is embedded locally using `all-MiniLM-L6-v2`.

The embeddings are stored in the persistent ChromaDB collection `zepto_policy` using cosine similarity.

## Intent Classification

The deterministic mock classifier uses these policy keywords:

- `delivery`
- `return`
- `refund`
- `membership`
- `tracking`
- `cancel`
- `gift card`
- `support hours`

Policy questions are routed to `retrieve_and_answer`.

Other questions are routed to `direct_answer`.

## Retrieval

For policy questions, the system:

1. Encodes the user query using `all-MiniLM-L6-v2`.
2. Queries ChromaDB.
3. Retrieves the top 3 policy chunks.
4. Uses the highest-ranked chunk for the deterministic mock response.
5. Returns the retrieved document IDs as sources.

Example:

```text
Query: How long does delivery take?
Top source: doc_01
```

## Structured Output

The assistant returns a Pydantic-validated response:

```json
{
  "answer": "string",
  "sources": ["doc_01"],
  "confidence": 0.4154
}
```

The fields are:

| Field | Description |
|---|---|
| `answer` | Assistant response |
| `sources` | Retrieved policy document/chunk IDs |
| `confidence` | Value between 0 and 1 |

For general questions:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 0.0
}
```

## Structured Prompt

The assistant prompt contains:

- Role
- Context
- Task
- Format
- Length
- Negative constraint
- Few-shot example

The negative constraint prevents unsupported policy information from being invented.

## Mock LLM Mode

The default configuration is:

```text
MOCK_LLM=1
```

This mode is deterministic and does not make an external LLM/API call.

Policy answers use:

```text
Based on the retrieved context: {top_chunk_snippet}
```

General questions use:

```text
I can only answer questions about Zepto policies right now.
```

## Files

```text
support_assistant/
├── README.md
├── Dockerfile
├── api.py
├── support_assistant.py
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
└── chroma_db/
```

## Local Setup

From the repository root:

```bash
pip install -r requirements.txt
```

The vector store must contain the eight policy documents before starting the API.

Default mode:

```bash
export MOCK_LLM=1
```

On Windows PowerShell:

```powershell
$env:MOCK_LLM="1"
```

## Running the API

From `support_assistant/`, run:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

The API will be available on port `8000`.

## API Endpoint

### POST `/ask`

Request:

```json
{
  "query": "How long does delivery take?"
}
```

Response:

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01", "doc_02", "doc_04"],
  "confidence": 0.4154
}
```

## Example API Call

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={"query": "How long does delivery take?"}
)

print(response.json())
```

## General Question Example

Request:

```json
{
  "query": "What is the capital of France?"
}
```

Response:

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 0.0
}
```

## Docker

Build from the repository root:

```bash
docker build -f support_assistant/Dockerfile -t zepto-support-assistant .
```

Run:

```bash
docker run -p 8000:8000 zepto-support-assistant
```

## Validation Performed

The implementation has been tested for:

- Exactly 8 policy documents
- Non-empty UTF-8 policy corpus
- Local `all-MiniLM-L6-v2` embeddings
- Persistent ChromaDB collection
- Cosine similarity retrieval
- Correct delivery-policy retrieval
- Three-node LangGraph workflow
- Deterministic mock intent classification
- Policy routing
- General-question routing
- Pydantic response validation
- FastAPI application import
- `GET /` health endpoint
- `POST /ask` policy query
- `POST /ask` general query
- Empty-query validation

The default submission path is deterministic and does not require an external LLM API key.
