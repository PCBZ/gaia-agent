---
title: GAIA Agent
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.26.0
app_file: app.py
pinned: false
---

# gaia-agent

A [LlamaIndex](https://docs.llamaindex.ai/)-based agent for the **GAIA benchmark**,
built as the final project of the Hugging Face
[Agents course](https://huggingface.co/learn/agents-course). Solves 18/20 of the
course's GAIA validation questions.

## How it works

One solver per question under `q*.py`, each a `QuestionSolver` subclass (base in
`gaia/base.py`) that declares only its `number`. Shared building blocks:

- `gaia/questions.py` — fetch/cache questions (Pydantic models) + download gated
  GAIA attachments.
- `gaia/web.py` — `web_search` (ddgs), `jina_search`, `read_url` (Jina Reader),
  `read_tables` (pandas), plus `code_interpreter`.
- `gaia/media.py` — `youtube_transcript` (captions) + `transcribe_audio` (Groq Whisper).
- `gaia/web_solver.py` — generic `WebSolver(n)` for retrieval questions.
- `config.toml` — endpoints / model names (non-secret); API keys live in `.env`.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # fill in the keys below
python main.py -n 3     # solve one question
python app.py           # launch the Gradio UI
```

## Keys (Space secrets / .env)

| Key | Used for |
| --- | --- |
| `OPENROUTER_API_KEY` | LLM (gpt-4o-mini / gpt-4o) |
| `GROQ_API_KEY`       | audio transcription (Whisper) |
| `JINA_API_KEY`       | `jina_search` (hard-to-find pages) |
| `HF_TOKEN`           | gated GAIA dataset attachments (accept the dataset terms) |

## Notes

Questions #2 (video bird-species count) and #4 (chess board recognition) are left
unsolved — both need reliable frame/board vision that general tools don't provide.
