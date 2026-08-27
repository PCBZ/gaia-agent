"""Shared web tools for retrieval questions — fully keyless.

- web_search: DuckDuckGo via ddgs (multi-backend fallback), no API key.
- read_url:   Jina Reader (https://r.jina.ai), returns clean Markdown with tables
              preserved, no API key.
- read_tables: pandas.read_html for precise stats tables.
"""
from __future__ import annotations

import httpx
from ddgs import DDGS
from llama_index.core.tools import FunctionTool
from llama_index.tools.code_interpreter import CodeInterpreterToolSpec

from gaia.config import CONFIG


def web_search(query: str, max_results: int = 10) -> str:
    """Search the web. Returns lines of 'title | url | snippet'."""
    try:
        results = list(DDGS().text(query, max_results=max_results, backend="auto"))
    except Exception as exc:  # noqa: BLE001 - ddgs raises on rate limit / no hits
        return f"Search failed ({exc}); try rephrasing the query."
    if not results:
        return "No results."
    return "\n".join(
        f"{r.get('title', '')} | {r.get('href', '')} | {r.get('body', '')}"
        for r in results
    )


def read_url(url: str, max_chars: int = 20000) -> str:
    """Fetch a page as clean Markdown (tables preserved) via Jina Reader."""
    resp = httpx.get(
        f"{CONFIG['api']['jina_reader']}/{url}",
        timeout=90,
        headers={"User-Agent": "gaia-agent"},
        follow_redirects=True,
    )
    resp.raise_for_status()
    return resp.text[:max_chars]


def read_tables(url: str, max_chars: int = 20000) -> str:
    """Fetch a URL and return its HTML tables as text with rows preserved.

    Use this for statistics, rankings or any data where columns must line up with
    the right row.
    """
    import pandas as pd

    try:
        tables = pd.read_html(url)
    except Exception as exc:  # noqa: BLE001 - no tables / fetch failure
        return f"No tables parsed from {url}: {exc}"
    parts = [
        f"### Table {i} (shape {df.shape})\n{df.to_string()}"
        for i, df in enumerate(tables)
    ]
    return "\n\n".join(parts)[:max_chars]


def web_tools() -> list[FunctionTool]:
    """Keyless search + read tools, plus code_interpreter so the agent can compute
    maxima/minima and aggregates over fetched data instead of eyeballing them."""
    return [
        FunctionTool.from_defaults(fn=web_search),
        FunctionTool.from_defaults(fn=read_url),
        FunctionTool.from_defaults(fn=read_tables),
        *CodeInterpreterToolSpec().to_tool_list(),
    ]
