# PDF Q&A Chatbot (LangGraph)

An actual interactive chatbot now, not a hardcoded question list: upload/point
it at a PDF, then ask it as many questions as you want in a loop.

## Run it
```
pip install langgraph pypdf
python chat.py
```
It'll ask for a PDF filename (try `demo.pdf`, the included Twain story), then
drop you into a chat loop:
```
You: Who is Sadie?
Agent: ...
You: quit
```

## Files
- `agent.py` — the LangGraph pipeline (load → retrieve → answer) plus a
  `PDFAgent` class that loads a PDF once and answers many questions against
  it without re-reading the file each time.
- `chat.py` — the interactive loop. This is the file to run for the actual demo.
- `demo.pdf` — the real assigned PDF (A Dog's Tale, Mark Twain) for testing.

## Honest limitations (same as before, still true)
- Answers are extractive (the actual PDF sentences), not generated, since
  there's no LLM API key in this environment — see `ANSWER_WITH_LLM` in
  agent.py for where to plug one in.
- Retrieval is keyword-overlap, not embeddings, so it can occasionally miss
  the exact right sentence when the question's wording doesn't overlap much
  with the source text.
