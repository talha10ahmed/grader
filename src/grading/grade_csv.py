"""Build a flat CSV of the grader's marks breakdown.

The annotated PDF is the primary output, but the annotator sometimes places a
mark on the wrong line or fails to place it at all. This CSV is an independent,
text-only view of the same model output so a teacher can verify every mark.

Grain — one row per *annotatable unit*, mirroring what the annotator places:
  • numerical mode  → one row per criterion (the score-label unit)
  • holistic mode   → one row per correct point (the tick unit)
A leading TOTAL row carries the question-level score and overall reason.
"""
import csv
import io
from typing import Any


CSV_COLUMNS = [
    "student",
    "question",
    "sub_question",     # holistic only — which sub-part
    "student_label",    # holistic only — how the student labelled the part
    "criterion_id",     # rubric's short id (MM07, TB03 …) — read this, not the prose
    "criterion",        # what is being marked
    "key_phrase",       # holistic only — exact anchor the annotator aims at
    "student_text",     # the student's own words the mark is for (every anchor)
    "marks_awarded",
    "max_possible",
    "OF",               # own-figure partial credit (numerical)
    "reason",
    "comments_summary",  # optional per-item grader note
]


def _evidence_text(item: dict) -> str:
    """All evidence anchors for a numerical criterion, each on its own line.

    Prefers the structured evidence_list (one entry per underline target) so
    every anchor stays individually visible for side-by-side PDF comparison;
    falls back to the joined evidence string.
    """
    raw = item.get("evidence_list")
    if isinstance(raw, list):
        lines = [str(x).strip() for x in raw if x is not None and str(x).strip()]
        if lines:
            return "\n".join(lines)
    return str(item.get("evidence", "") or "")


def _fmt_marks(value: Any) -> str:
    """Format a marks value: drop a trailing .0 but keep 0.5 increments."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "" if value is None else str(value)
    if f == int(f):
        return str(int(f))
    return f"{f:g}"


def build_breakdown_csv(grades_doc: dict) -> str:
    """Return the marks breakdown of *grades_doc* as CSV text."""
    student = str(grades_doc.get("student_id", "") or "")
    question = str(grades_doc.get("question_number", "") or "")
    is_holistic = bool(grades_doc.get("holistic_grading", False))
    breakdown = grades_doc.get("breakdown", []) or []

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()

    # ── Leading summary row: the question total ──────────────────────────────
    writer.writerow({
        "student": student,
        "question": question,
        "criterion": "TOTAL",
        "marks_awarded": _fmt_marks(grades_doc.get("total_marks_awarded")),
        "max_possible": _fmt_marks(grades_doc.get("total_max_possible")),
        "reason": str(grades_doc.get("overall_reason", "") or ""),
    })

    for item in breakdown:
        if not isinstance(item, dict):
            continue

        if is_holistic:
            sub_q = str(item.get("_sub_question", "") or "")
            student_label = str(item.get("_student_label", "") or "")
            criterion = str(item.get("criterion", "") or "")
            points = item.get("_correct_points_with_marks", []) or []

            if points:
                # One row per ticked point (the unit the annotator places).
                for pt in points:
                    if not isinstance(pt, dict):
                        continue
                    writer.writerow({
                        "student": student,
                        "question": question,
                        "sub_question": sub_q,
                        "student_label": student_label,
                        "criterion_id": str(item.get("criterion_id", "") or ""),
                        "criterion": criterion,
                        "key_phrase": str(pt.get("key_phrase", "") or ""),
                        "student_text": str(pt.get("text", "") or ""),
                        "marks_awarded": _fmt_marks(pt.get("marks")),
                        # Per-point max isn't meaningful; sub-q total lives on
                        # the no-points fallback row below / TOTAL row.
                        "max_possible": "",
                        "OF": "",
                        "reason": "",
                    })
            else:
                # Sub-question awarded marks but no per-point breakdown survived —
                # still emit one row so the sub-q score is visible.
                writer.writerow({
                    "student": student,
                    "question": question,
                    "sub_question": sub_q,
                    "student_label": student_label,
                    "criterion_id": str(item.get("criterion_id", "") or ""),
                    "criterion": criterion,
                    "student_text": _evidence_text(item),
                    "marks_awarded": _fmt_marks(item.get("marks_awarded")),
                    "max_possible": _fmt_marks(item.get("max_possible")),
                    "reason": str(item.get("reason", "") or ""),
                    "comments_summary": str(item.get("comments_summary", "") or ""),
                })
        else:
            # Numerical: one row per criterion, with every evidence anchor shown.
            writer.writerow({
                "student": student,
                "question": question,
                "criterion_id": str(item.get("criterion_id", "") or ""),
                "criterion": str(item.get("criterion", "") or ""),
                "student_text": _evidence_text(item),
                "marks_awarded": _fmt_marks(item.get("marks_awarded")),
                "max_possible": _fmt_marks(item.get("max_possible")),
                "OF": "Yes" if item.get("is_of_mark") else "",
                "reason": str(item.get("reason", "") or ""),
                "comments_summary": str(item.get("comments_summary", "") or ""),
            })

    return buf.getvalue()
