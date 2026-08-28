"""Gradio app for the HF Agents course final project (shaped like the course
template): log in with HF, run our GAIA agent on all questions, submit for scoring.

Space secrets needed: OPENROUTER_API_KEY, GROQ_API_KEY, JINA_API_KEY, HF_TOKEN
(HF_TOKEN must have accepted the gated gaia-benchmark/GAIA dataset for attachments).
"""
from __future__ import annotations

import os
import re

import gradio as gr
import httpx
import pandas as pd

from gaia.config import CONFIG
from gaia.questions import load_questions
from main import SOLVERS

SCORING_API = CONFIG["api"]["scoring_api"]

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


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """Run every registered solver and submit the answers for scoring."""
    if profile is None:
        return "⚠️ Please log in to Hugging Face with the button above.", None
    username = profile.username

    space_id = os.getenv("SPACE_ID")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "https://github.com/PCBZ/gaia-agent"
    )

    questions = load_questions()
    answers, rows = [], []
    for i, q in enumerate(questions):
        number = i + 1
        solver = SOLVERS.get(number)
        if solver is None:
            ans = ""  # unsolved (e.g. #2, #4)
        else:
            try:
                ans = final_answer(solver.resolve())
            except Exception as exc:  # noqa: BLE001 - keep going on any failure
                ans = f"ERROR: {exc}"
        answers.append({"task_id": q.task_id, "submitted_answer": ans})
        rows.append({"#": number, "Question": q.question[:60], "Answer": ans})

    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    try:
        resp = httpx.post(f"{SCORING_API}/submit", json=payload, timeout=180)
        resp.raise_for_status()
        r = resp.json()
        status = (
            f"✅ Submitted as {r.get('username')} — Score: {r.get('score')}% "
            f"({r.get('correct_count')}/{r.get('total_attempted')} correct)\n"
            f"{r.get('message', '')}"
        )
    except Exception as exc:  # noqa: BLE001
        status = f"❌ Submit failed: {exc}"

    return status, pd.DataFrame(rows)


with gr.Blocks(title="GAIA Agent") as demo:
    gr.Markdown("# 🤖 GAIA Agent — HF Agents course final project")
    gr.Markdown(
        "Log in with Hugging Face, then run the LlamaIndex agent on all 20 GAIA "
        "questions and submit for scoring."
    )
    gr.LoginButton()
    run_btn = gr.Button("Run evaluation & submit all answers", variant="primary")
    status = gr.Markdown()
    table = gr.Dataframe(headers=["#", "Question", "Answer"], wrap=True)
    run_btn.click(run_and_submit_all, outputs=[status, table])


if __name__ == "__main__":
    demo.launch()
