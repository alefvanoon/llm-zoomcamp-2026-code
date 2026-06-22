# Homework 01 - Agentic RAG

LLM Zoomcamp 2026, Module 1. A RAG system built over the course lessons,
then turned into an agent. Uses a free OpenRouter model instead of OpenAI.

## Setup

```bash
uv add gitsource minsearch toyaikit openai python-dotenv
```

Get a key from https://openrouter.ai (no card needed) and put it in `.env`:

```dotenv
OPENROUTER_API_KEY=sk-or-...
```

## Run

```bash
uv run main.py
```

It pulls the lessons at commit `8c1834d` and prints the answer to each question.

## Files

- `main.py` - runs Q1 through Q6
- `rag.py` - the RAG helper, adapted from the course's `rag_helper.py`
  (our filename/content schema, OpenRouter's chat API, returns token usage)

## Model

Default is `openrouter/free`. Swap the `MODEL` line in
`main.py` for another free one if it stops working:

- `openrouter/free` (auto-picks a free model)
- `qwen/qwen3-coder:free`
- `openai/gpt-oss-20b:free`

Free tier is ~20 requests/min. The agent in Q6 fires a few calls per run, so a
429 just means wait a few seconds and try again.

## My answers

- Q1: 72
- Q2: 01-agentic-rag/lessons/14-agentic-loop.md
- Q3: ~7000 input tokens (actually 7948)
- Q4: 295 chunks
- Q5: 3x fewer tokens with chunking (actually 3.3x)
- Q6: ~4 search calls (actually 3)
