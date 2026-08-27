"""Media tools for audio/video questions.

Currently: fetch a YouTube video's transcript (captions) via youtube-transcript-api
— no download, no API key. Exposed as a LlamaIndex FunctionTool.
"""
from __future__ import annotations

import re

from llama_index.core.tools import FunctionTool
from youtube_transcript_api import YouTubeTranscriptApi


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


def media_tools() -> list[FunctionTool]:
    """Toolset for audio/video questions."""
    return [FunctionTool.from_defaults(fn=youtube_transcript)]
