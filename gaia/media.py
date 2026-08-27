"""Media tools for audio/video questions.

Currently: fetch a YouTube video's transcript (captions) via youtube-transcript-api
— no download, no API key. Exposed as a LlamaIndex FunctionTool.
"""
from __future__ import annotations

import os
import re

import httpx
from llama_index.core.tools import FunctionTool
from youtube_transcript_api import YouTubeTranscriptApi

from gaia.config import CONFIG


def youtube_transcript(video: str) -> str:
    """Return the caption transcript of a YouTube video.

    Args:
        video: a YouTube URL (youtube.com/watch?v=... or youtu.be/...) or an
            11-character video id.
    """
    m = re.search(r"(?:v=|youtu\.be/)([\w-]{11})", video)
    vid = m.group(1) if m else video.strip()
    try:
        return " ".join(s.text for s in YouTubeTranscriptApi().fetch(vid))
    except Exception as exc:  # noqa: BLE001 - no captions / blocked / bad id
        return f"Could not fetch transcript for {video}: {exc}"


def transcribe_audio(path: str, model: str = CONFIG["audio"]["model"]) -> str:
    """Transcribe a local audio file to text using Groq's Whisper. Needs GROQ_API_KEY.

    Args:
        path: local path to an audio file (mp3, wav, m4a, ...).
        model: Groq speech model (default from config.toml).
    """
    with open(path, "rb") as fh:
        resp = httpx.post(
            CONFIG["audio"]["endpoint"],
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"},
            files={"file": fh},
            data={"model": model},
            timeout=180,
        )
    resp.raise_for_status()
    return resp.json().get("text", "")


def media_tools() -> list[FunctionTool]:
    """Toolset for audio/video questions."""
    return [
        FunctionTool.from_defaults(fn=youtube_transcript),
        FunctionTool.from_defaults(fn=transcribe_audio),
    ]
