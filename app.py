"""Gradio app for the HF Agents course final project.

Unlike the course template (one button that runs + submits everything), this lays
out all questions up front: each row has the question, its own Run button, and an
answer box filled in place. A final Submit button sends every answer for scoring.

Space secrets needed: OPENROUTER_API_KEY, GROQ_API_KEY, JINA_API_KEY, HF_TOKEN
(HF_TOKEN must have accepted the gated gaia-benchmark/GAIA dataset for attachments).
"""
from __future__ import annotations

import os
import re

import gradio as gr
import httpx
import pandas as pd

# ZeroGPU Spaces need at least one @spaces.GPU function to initialize the runtime,
# otherwise the app runs internally but HF returns RUNTIME_ERROR / 503. We don't use
# the GPU; this is a never-called stub. Guarded so local imports still work.
try:
    import spaces

    @spaces.GPU
    def _gpu_stub():  # noqa: D401 - satisfies ZeroGPU; never called
        return None
except Exception:  # noqa: BLE001 - not on a ZeroGPU Space
    pass

from gaia.config import CONFIG
from gaia.questions import download_attachment, load_questions
from main import SOLVERS

SCORING_API = CONFIG["api"]["scoring_api"]
AUDIO_EXTS = (".mp3", ".wav", ".m4a", ".ogg", ".flac")


def audio_path(q):
    """Local path to a question's audio attachment, or None (not audio / no access)."""
    if not q.file_name.lower().endswith(AUDIO_EXTS):
        return None
    try:
        return str(download_attachment(q))
    except Exception:  # noqa: BLE001 - no HF_TOKEN / gated access / network
        return None

_FINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL)


def final_answer(text: str) -> str:
    """Extract the exact-match answer after 'FINAL ANSWER:'."""
    text = text or ""
    m = _FINAL_RE.search(text)
    if m:
        ans = m.group(1).strip()
    else:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        ans = lines[-1] if lines else ""
    return ans.strip().strip("\"'").rstrip(".").strip()


def solve_one(number: int):
    """Build the click handler for one question's Run button."""

    def _run() -> str:
        solver = SOLVERS.get(number)
        if solver is None:
            return ""  # unsolved (e.g. #2, #4)
        try:
            return final_answer(solver.resolve())
        except Exception as exc:  # noqa: BLE001
            return f"ERROR: {exc}"

    return _run


def submit_all(profile: gr.OAuthProfile | None, *answers: str):
    """Submit every row's current answer for scoring."""
    if profile is None:
        return "⚠️ Please log in to Hugging Face with the button above."
    username = profile.username

    space_id = os.getenv("SPACE_ID")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "https://github.com/PCBZ/gaia-agent"
    )

    questions = load_questions()
    payload_answers = [
        {"task_id": q.task_id, "submitted_answer": (ans or "").strip()}
        for q, ans in zip(questions, answers)
    ]
    payload = {"username": username, "agent_code": agent_code, "answers": payload_answers}
    try:
        resp = httpx.post(f"{SCORING_API}/submit", json=payload, timeout=180)
        resp.raise_for_status()
        r = resp.json()
        return (
            f"✅ Submitted as {r.get('username')} — Score: {r.get('score')}% "
            f"({r.get('correct_count')}/{r.get('total_attempted')} correct)\n"
            f"{r.get('message', '')}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"❌ Submit failed: {exc}"


with gr.Blocks(title="GAIA Agent") as demo:
    gr.Markdown("# 🤖 GAIA Agent — HF Agents course final project")
    gr.Markdown(
        "Log in, run each question with its **Run** button (answer appears in the "
        "row), edit if needed, then **Submit all** for scoring."
    )
    gr.LoginButton()

    answer_boxes = []
    for i, q in enumerate(load_questions()):
        number = i + 1
        audio = audio_path(q)
        with gr.Row(equal_height=True):
            with gr.Column(scale=6):
                gr.Markdown(f"**#{number}** {q.question}", elem_id=f"q{number}")
                if audio:
                    gr.Audio(value=audio, label="attached audio", interactive=False)
                elif q.file_name.lower().endswith(AUDIO_EXTS):
                    gr.Markdown("🔊 (audio attachment)")
            run_btn = gr.Button("Run", scale=0, min_width=80)
            ans = gr.Textbox(label="answer", show_label=False, scale=2, container=False)
        answer_boxes.append(ans)
        run_btn.click(solve_one(number), outputs=ans)

    submit_btn = gr.Button("Submit all answers", variant="primary")
    status = gr.Markdown()
    submit_btn.click(submit_all, inputs=answer_boxes, outputs=status)


if __name__ == "__main__":
    demo.launch(ssr_mode=False)
