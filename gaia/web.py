"""Shared web tools for retrieval questions: search + fetch-page-as-text.

Wrapped as LlamaIndex FunctionTools via `web_tools()`.
"""
from __future__ import annotations

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from llama_index.core.tools import FunctionTool


def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns lines of 'title | url | snippet'."""
    results = list(DDGS().text(query, max_results=max_results, backend="auto"))
    if not results:
        return "No results."
    return "\n".join(
        f"{r.get('title', '')} | {r.get('href', '')} | {r.get('body', '')}"
        for r in results
    )


def read_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a web page and return its visible text (truncated)."""
    resp = requests.get(
        url, timeout=30, headers={"User-Agent": "Mozilla/5.0 (gaia-agent)"}
    )
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "sup"]):
        tag.decompose()
    text = "\n".join(
        line.strip() for line in soup.get_text("\n").splitlines() if line.strip()
    )
    return text[:max_chars]


def web_tools() -> list[FunctionTool]:
    """The search + read toolset for retrieval questions."""
    return [
        FunctionTool.from_defaults(fn=web_search),
        FunctionTool.from_defaults(fn=read_url),
    ]
