from docx import Document
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JD_PATH = (
    ROOT / "data" / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge" / "job_description.docx"
)


def load_job_description():
    doc = Document(JD_PATH)
    return "\n".join(para.text for para in doc.paragraphs if para.text.strip())


def load_jd_for_embedding():
    full_text = load_job_description()
    markers = (
        "Things you absolutely need",
        "Things we'd like you to have",
        "What you'd actually be doing",
    )

    chunks = []
    for marker in markers:
        start = full_text.find(marker)
        if start == -1:
            continue
        end = len(full_text)
        for other in markers:
            if other == marker:
                continue
            pos = full_text.find(other, start + len(marker))
            if pos != -1:
                end = min(end, pos)
        chunks.append(full_text[start:end].strip())

    if chunks:
        return "\n".join(chunks)

    return full_text[:4000]
