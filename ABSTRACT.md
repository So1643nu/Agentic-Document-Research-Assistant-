# Abstract

Most "RAG chatbot" demos retrieve context for every single question,
whether it needs it or not. That's wasteful and, more importantly, it
often hurts answer quality — the model gets distracted by irrelevant
chunks when the user just asked it to add two numbers or clarify its own
previous answer.

This project takes a slightly different approach. The language model isn't
handed retrieved context up front; instead it's given two tools — a
document search function and a calculator — and left to decide, turn by
turn, whether it actually needs them. That decide-act-observe loop is what
"agentic" is supposed to mean, and it's a small enough change that the
whole thing was buildable, testable, and demoable in under half a day.

The retrieval layer defaults to TF-IDF rather than neural embeddings on
purpose. It works offline with no model download, which matters more than
it sounds like it should — plenty of RAG side-projects break on a fresh
machine the moment the embedding model can't be pulled from Hugging Face.
Embeddings (FAISS + sentence-transformers) are still available as an
opt-in upgrade for larger, more paraphrase-heavy document sets.

The result is a small Streamlit app: upload a handful of documents, ask
questions, and watch (via an expandable panel) which tools the agent chose
to call and why. It's intentionally scoped to be a solid starting point
rather than a finished product — the tool interface is the part worth
reusing; swapping in a web-search tool, a SQL tool, or a real code
executor is a natural next step and shouldn't require touching the agent
loop itself.
