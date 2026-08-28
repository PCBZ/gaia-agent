"""Gradio app for the HF Agents course final project.

Runs the GAIA agent on all 20 questions and submits the answers to the course
scoring API. Deployable as a free Hugging Face Space (CPU basic).

Space secrets needed: OPENROUTER_API_KEY, GROQ_API_KEY, JINA_API_KEY, HF_TOKEN
(HF_TOKEN must have accepted the gated gaia-benchmark/GAIA dataset for attachments).
"""
from __future__ import annotations

import re

import gradio as gr
import httpx
import pandas as pd

from gaia.config import CONFIG
from gaia.questions import load_questions
from main import SOLVERS

SCORING_API = CONFIG["api"]["scoring_api"]
DEFAULT_AGENT_CODE = "https://github.com/PCBZ/gaia-agent"

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


def run_and_submit(username: str, agent_code: str, progress=gr.Progress()):
    username = (username or "").strip()
    if not username:
        return "⚠️ Enter your Hugging Face username.", pd.DataFrame()

    questions = load_questions()
    answers, rows = [], []
    for i, q in enumerate(progress.tqdm(questions, desc="Solving")):
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
        rows.append([number, q.question[:60], ans])

    payload = {"username": username, "agent_code": agent_code, "answers": answers}
    try:
        resp = httpx.post(f"{SCORING_API}/submit", json=payload, timeout=120)
        resp.raise_for_status()
        r = resp.json()
        msg = (
            f"✅ Score: {r['score']}%  "
            f"({r['correct_count']}/{r['total_attempted']} correct)\n{r.get('message', '')}"
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"❌ Submit failed: {exc}"

    return msg, pd.DataFrame(rows, columns=["#", "question", "answer"])


with gr.Blocks(title="GAIA Agent") as demo:
    gr.Markdown("# 🤖 GAIA Agent — HF Agents course final project")
    gr.Markdown(
        "Runs a LlamaIndex agent on all 20 GAIA questions and submits for scoring."
    )
    with gr.Row():
        username = gr.Textbox(label="Hugging Face username")
        agent_code = gr.Textbox(label="Agent code URL", value=DEFAULT_AGENT_CODE)
    run_btn = gr.Button("Run evaluation & submit", variant="primary")
    result = gr.Markdown()
    table = gr.Dataframe(headers=["#", "question", "answer"], wrap=True)
    run_btn.click(run_and_submit, [username, agent_code], [result, table])


if __name__ == "__main__":
    demo.launch()
