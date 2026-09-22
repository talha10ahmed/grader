"""Centralised error classification for the grading pipeline.

classify_error(e) → (human_readable_message, show_traceback)

Known/expected errors return a clean one-liner with show_traceback=False.
Unexpected errors return show_traceback=True so the full traceback is logged.
"""

from typing import Tuple

from errors.exceptions import (
    PDFExtractionError,
    QuestionExtractionError,
    ModelAnswerExtractionError,
    StudentAssignmentExtractionError,
    GradingError,
    AnnotationError,
)


def classify_error(e: Exception) -> Tuple[str, bool]:
    """Return (message, show_traceback) for any pipeline exception."""
    msg = str(e)

    # ── API quota / billing ──────────────────────────────────────────────────
    if "insufficient_quota" in msg or ("429" in msg and "quota" in msg.lower()):
        return (
            "OpenAI quota exhausted — check your plan at "
            "https://platform.openai.com/account/billing",
            False,
        )

    # ── Transient rate limit (SDK already retried) ───────────────────────────
    if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
        return "API rate limit hit (all retries exhausted)", False

    # ── Auth / bad API key ───────────────────────────────────────────────────
    if "401" in msg or "invalid_api_key" in msg.lower() or "authentication" in msg.lower():
        return "API authentication failed — check your API key in .env", False

    # ── Network / timeout ────────────────────────────────────────────────────
    if any(k in msg.lower() for k in ("timeout", "timed out", "connection", "network", "ssl")):
        return f"Network error: {msg}", False

    # ── PDF file problems ────────────────────────────────────────────────────
    if "no pages rendered" in msg.lower() or "no such file" in msg.lower() or "filenotfounderror" in msg.lower():
        return f"PDF file error: {msg}", False

    # ── Database ─────────────────────────────────────────────────────────────
    if isinstance(e, (ConnectionError, RuntimeError)) and "mongo" in msg.lower():
        return f"Database error: {msg}", False

    # ── Known pipeline errors (message already descriptive) ─────────────────
    if isinstance(e, (
        PDFExtractionError,
        QuestionExtractionError,
        ModelAnswerExtractionError,
        StudentAssignmentExtractionError,
        GradingError,
        AnnotationError,
    )):
        return msg, False

    # ── Anything else — unexpected, show full traceback ──────────────────────
    return f"Unexpected error: {msg}", True
