# Agentic RAG Research Assistant

A small but genuinely **agentic** RAG system: instead of blindly retrieving
context for every question, the assistant is given tools (document search,
calculator) and decides for itself when to use them. Built to be finished,
tested, and demoed in about half a day.

---

## Why this isn't "just RAG"

Classic RAG pipelines always do: `retrieve -> stuff into prompt -> generate`.
That wastes tokens and hurts accuracy on questions that don't need retrieval
at all (e.g. "what's 12% of 340?" or "summarize what you just told me").

Here, the LLM is given a small toolbox:

- `search_documents(query, k)` — semantic/keyword search over your ingested
  documents
- `calculator(expression)` — safe arithmetic evaluation

On each turn, the model (via Anthropic's tool-use / function-calling API)
decides whether it needs a tool, calls it, reads the result, and can call
another tool or answer directly. That loop — decide, act, observe, repeat —
is what makes it "agentic" rather than a fixed pipeline.

## Architecture

```
                 ┌────────────────────┐
   user query -> │   Streamlit UI      │
                 │      (app.py)       │
                 └──────────┬──────────┘
                            │
                            v
                 ┌────────────────────┐
                 │   AgenticRAGAgent   │   <- agent.py
                 │  (tool-use loop)    │
                 └────┬───────────┬────┘
                      │           │
             search_documents  calculator
                      │           │
                      v           v
              ┌───────────────┐ ┌──────────────┐
              │  VectorStore  │ │  tools.py     │
              │ (TF-IDF/FAISS)│ │ (safe eval)   │
              └───────┬───────┘ └──────────────┘
                      │
                      v
              ┌───────────────┐
              │   ingest.py    │  <- chunks .txt/.md/.pdf
              └───────────────┘
```

## Retrieval backend — a deliberate trade-off

Two backends are supported, picked by an env var:

- **Default: TF-IDF (scikit-learn).** Zero external downloads, works fully
  offline, deterministic, good enough for small-to-medium document sets.
  This is what ships enabled out of the box so the project runs immediately
  after `pip install`, with no dependency on Hugging Face Hub being
  reachable.
- **Optional: sentence-transformers + FAISS.** Better semantic recall for
  larger or more paraphrase-heavy corpora, but requires downloading an
  embedding model on first run. Enable with `USE_EMBEDDINGS=true` in `.env`
  once you know your deployment environment has internet access to
  Hugging Face.

This is a real engineering decision, not a shortcut: a lot of "RAG demo"
projects fail on a fresh machine because the embedding model download hangs
or is blocked by a firewall. Shipping a working offline default and treating
embeddings as an upgrade avoids that failure mode.

## Project layout

```
agentic_rag_assistant/
├── app.py                 # Streamlit chat UI
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py           # env var / settings loading
│   ├── ingest.py            # document loading + chunking
│   ├── vectorstore.py       # TF-IDF / FAISS retrieval backends
│   ├── tools.py              # tool schemas + executors
│   └── agent.py               # the agentic tool-use loop
├── data/sample_docs/         # sample docs to try it on immediately
├── tests/                    # unit tests (no API key required)
└── run.sh                    # one-command launch
```

## Setup (5 minutes)

```bash
cd agentic_rag_assistant
python -m venv venv && source venv/bin/activate     # optional but recommended
pip install -r requirements.txt
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
bash run.sh
# or directly:
streamlit run app.py
```

Then open the local URL Streamlit prints (usually http://localhost:8501).

1. In the sidebar, click **"Build index from sample docs"** (or upload your
   own `.txt` / `.md` / `.pdf` files).
2. Ask a question in the chat box. Watch the "tool calls" expander to see
   the agent deciding whether to search, calculate, or just answer.

## Running tests (no API key needed)

The ingestion and retrieval layers are fully testable offline:

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Example questions to try against the sample docs

- "What does the sample knowledge base say about onboarding new employees?"
- "Based on the pricing doc, what would 3 seats at the Pro tier cost, plus 8% tax?"
  (forces the agent to combine `search_documents` + `calculator`)
- "What's 47 * 12?" (should NOT trigger document search — pure calculator)

## Extending it

- Swap `calculator` for a real code-execution tool.
- Add a `web_search` tool (mirrors `search_documents`'s schema) for hybrid
  local + live retrieval.
- Swap TF-IDF for FAISS + sentence-transformers once you have a bigger
  corpus (`USE_EMBEDDINGS=true`).
- Add conversation memory persistence (currently in-session only, via
  Streamlit `session_state`).

## Known limitations

- The calculator tool uses a restricted expression evaluator — no variables,
  no imports, arithmetic only. This is intentional (security).
- TF-IDF retrieval is keyword-driven; it won't catch pure paraphrases the
  way embeddings would. Turn on `USE_EMBEDDINGS` for that.
- Single-session memory only; no persistence across restarts.

## License

MIT — do whatever you want with it.
