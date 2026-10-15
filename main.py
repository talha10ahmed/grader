"""Production grading pipeline.

Loads a pre-saved model answer from MongoDB, extracts the student PDF,
grades, and annotates. No question/rubric PDF extraction happens here.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import asyncio
import hashlib
import json
from typing import Tuple, Optional, List, Literal, Any, Callable
from datetime import datetime
import nest_asyncio

nest_asyncio.apply()

from bson import ObjectId

from logging_config import logger
from extraction.student_assignment_extraction import extract_assignment_pipeline
from grading.grade import grade_student
from grading.grade_csv import build_breakdown_csv
from annotation.annotator import annotate_pdf
from database.mongodb import get_collection
from errors import classify_error


def _build_grades_csv(grades_id: str) -> Optional[str]:
    """Fetch the saved grade doc, render its marks breakdown as CSV text, and
    persist the CSV back onto that document.

    The CSV is stored on the grades document itself (field `breakdown_csv`)
    rather than on disk or in external storage, so it lives with the record it
    describes: it can be re-downloaded or audited later without re-running the
    grader, and it needs no filesystem that survives a container restart.

    Best-effort throughout: the CSV is a verification aid, so neither building
    nor saving it may fail the grading run.
    """
    try:
        coll = get_collection("student_grades")
        grades_doc = coll.find_one({"_id": ObjectId(grades_id)})
        if not grades_doc:
            logger.warning(f"[CSV] No grade doc for _id={grades_id}")
            return None
        csv_text = build_breakdown_csv(grades_doc)
    except Exception as e:
        logger.warning(f"[CSV] Failed to build breakdown CSV: {e}")
        return None

    try:
        coll.update_one(
            {"_id": ObjectId(grades_id)},
            {"$set": {
                "breakdown_csv": csv_text,
                "breakdown_csv_generated_at": datetime.now().isoformat(),
            }},
        )
        logger.info(f"[CSV] Saved breakdown CSV to student_grades/{grades_id} "
                    f"({len(csv_text):,} chars)")
    except Exception as e:
        # Saving is a convenience; the caller still gets the CSV to download.
        logger.warning(f"[CSV] Built CSV but could not save it to Mongo: {e}")

    return csv_text


def _write_grades_csv_file(
    output_dir: str, student_name: str, question_num: str, csv_text: str
) -> Optional[str]:
    """Also drop a .csv next to the annotated PDF for this student.

    Convenience copy only — the authoritative one is on the grades document.
    Never raises.
    """
    try:
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in student_name).strip()
        dest_dir = os.path.join(output_dir, safe or "student")
        os.makedirs(dest_dir, exist_ok=True)
        path = os.path.join(dest_dir, f"{safe or 'student'}_Q{question_num}_grades.csv")
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write(csv_text)
        logger.info(f"[CSV] Wrote {path}")
        return path
    except Exception as e:
        logger.warning(f"[CSV] Could not write CSV file: {e}")
        return None


QuestionType = Literal["numerical", "theoretical"]


# ── Same script, same result ──────────────────────────────────────────────────
# Grading uses a language model, and two runs over the same script can differ
# by a mark. When a script that has already been graded is uploaded again, its
# saved result is returned instead of grading it a second time, so the same
# file always gets the same marks.
#
# The match is on the exact file contents plus everything that decides the
# marks: pages, question, marking scheme (its full contents, so editing the
# scheme regrades), grading model, number of grading passes and GRADER_VERSION.
# Bump GRADER_VERSION whenever the grading logic changes so that scripts are
# regraded under the new logic rather than served an old result.
GRADER_VERSION = "2026-10-15.2"


def _reuse_enabled() -> bool:
    return os.getenv("REUSE_SAVED_GRADES", "1").strip().lower() not in {"0", "false", "no"}


def _submission_fingerprint(
    pdf_path: str,
    pages: List[int],
    question_num: str,
    question_type: str,
    model_answers_id: str,
) -> Optional[str]:
    """Hash identifying one graded submission, or None if it cannot be built."""
    try:
        pdf_hash = hashlib.sha256()
        with open(pdf_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                pdf_hash.update(chunk)

        rubric = get_collection("model_answers").find_one({"_id": ObjectId(model_answers_id)})
        if not rubric:
            return None
        rubric.pop("_id", None)
        rubric_hash = hashlib.sha256(
            json.dumps(rubric, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

        key = {
            "pdf": pdf_hash.hexdigest(),
            "pages": sorted(int(p) for p in pages),
            "question": str(question_num).strip(),
            "type": question_type,
            "model_answers_id": str(model_answers_id),
            "rubric": rubric_hash,
            "grader_version": GRADER_VERSION,
            "grader_model": os.getenv("LLM_GRADER_MODEL", ""),
            "extraction_model": os.getenv("LLM_EXTRACTION_MODEL", ""),
            "votes": os.getenv("GRADING_VOTES", "3"),
        }
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()
    except Exception as e:
        logger.warning(f"[Reuse] Could not fingerprint submission: {e}")
        return None


def _find_saved_grade(fingerprint: str) -> Optional[dict]:
    try:
        return get_collection("student_grades").find_one(
            {"submission_fingerprint": fingerprint},
            sort=[("_id", -1)],
        )
    except Exception as e:
        logger.warning(f"[Reuse] Lookup failed, grading afresh: {e}")
        return None


async def _extract_student_async(
    pdf_path: str,
    pages: List[int],
    student_name: str,
    question_num: str,
) -> Tuple[bool, Optional[str]]:
    loop = asyncio.get_running_loop()
    try:
        logger.info(f"[Student {student_name}] Extracting (pages {pages}, Q{question_num})")
        doc_id = await loop.run_in_executor(
            None,
            lambda: extract_assignment_pipeline(
                pdf_path=pdf_path,
                pages=pages,
                student_name=student_name,
                question_number=question_num,
            )
        )
        if not doc_id:
            logger.error(f"[Student {student_name}] Extraction returned no _id")
            return False, None
        logger.info(f"[Student {student_name}] Done → _id = {doc_id}")
        return True, doc_id
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Student {student_name}] {clean_msg}", exc_info=show_tb)
        return False, None


async def grade_from_db_async(
    model_answers_id: str,
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str,
    question_num: str,
    question_type: str = "numerical",
    reuse_student_answers_id: Optional[str] = None,
    input_pdf_for_annotation: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    force_regrade: bool = False,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Grade a student PDF using a pre-saved model answer from MongoDB.

    A script graded before (same file, pages, question, marking scheme and
    grader settings) gets its saved result back rather than a new grading -
    see GRADER_VERSION. `force_regrade=True` grades it afresh regardless.

    Returns (success, message, annotated_pdf_path, grades_csv_text).
    The CSV is produced as soon as grading succeeds, so it is returned even
    when annotation later fails — that is precisely when it is most useful.

    `reuse_student_answers_id` grades an extraction that already exists instead
    of re-reading the PDF. Extraction is not stable: two passes over the same
    script minutes apart came back 94% similar, one of them splitting the
    £'000 heading into its own column on every row — which changes the text
    every downstream match runs against. Re-extracting on each run therefore
    measures the extractor and the grader at the same time, and a change to
    either is impossible to attribute. Pass this to hold extraction fixed and
    vary only the grading.
    """
    start_time = datetime.now()
    logger.info("=" * 70)
    logger.info(f"GRADE FROM DB → {student_name} | Q{question_num} | type={question_type}")
    logger.info(f"  model_answers_id={model_answers_id}")
    logger.info("=" * 70)

    fingerprint = None
    if not reuse_student_answers_id:
        fingerprint = _submission_fingerprint(
            student_pdf_path, student_pages, question_num, question_type, model_answers_id
        )
    saved = None
    if fingerprint and _reuse_enabled() and not force_regrade:
        saved = _find_saved_grade(fingerprint)
    if saved:
        grades_id = str(saved["_id"])
        logger.info(
            f"[Reuse] {student_name}: this exact script was graded before "
            f"(grades _id={grades_id}, total {saved.get('total_marks_awarded')}) - "
            f"returning the saved result"
        )
        if progress_callback:
            progress_callback("This script was graded before - loading the saved result...", 50)
        grades_csv = saved.get("breakdown_csv") or _build_grades_csv(grades_id)
        if progress_callback:
            progress_callback("Drawing annotations and marks breakdown on PDF...", 85)
        try:
            annotation_ok, annotated_pdf = annotate_pdf(
                input_pdf_path=input_pdf_for_annotation or student_pdf_path,
                output_dir=output_dir,
                student_name=student_name,
                grades_id=grades_id,
                student_pages=student_pages,
            )
        except Exception as e:
            clean_msg, show_tb = classify_error(e)
            logger.error(f"[Annotation] {clean_msg}", exc_info=show_tb)
            return False, clean_msg, None, grades_csv
        if progress_callback:
            progress_callback("Grading and annotation complete!", 100)
        msg = ("Loaded the saved result for this script" if annotation_ok
               else "Annotation failed")
        return annotation_ok, msg, annotated_pdf, grades_csv

    if progress_callback:
        progress_callback("Step 1/3: Extracting student answer from PDF...", 15)

    if reuse_student_answers_id:
        student_answers_id = str(reuse_student_answers_id)
        logger.info(
            f"  reusing extraction {student_answers_id} (PDF not re-read) — "
            f"grading is the only variable"
        )
    else:
        s_ok, student_answers_id = await _extract_student_async(
            student_pdf_path, student_pages, student_name, question_num
        )
        if not s_ok or not student_answers_id:
            return False, "Student answer extraction failed", None, None

    grades_coll = get_collection("student_grades")

    if progress_callback:
        progress_callback("Step 2/3: Grading submission against rubric criteria with AI...", 45)

    loop = asyncio.get_running_loop()
    try:
        grades_id = await loop.run_in_executor(
            None,
            lambda: grade_student(
                student_name=student_name,
                question_number=question_num,
                questions_id=None,
                model_answers_id=model_answers_id,
                student_answers_id=student_answers_id,
                question_type=question_type,
            )
        )
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Grading] {clean_msg}", exc_info=show_tb)
        return False, clean_msg, None, None

    if not grades_id:
        return False, "Grading returned no result", None, None

    if fingerprint:
        try:
            grades_coll.update_one(
                {"_id": ObjectId(grades_id)},
                {"$set": {
                    "submission_fingerprint": fingerprint,
                    "grader_version": GRADER_VERSION,
                }},
            )
        except Exception as e:
            logger.warning(f"[Reuse] Could not save the fingerprint: {e}")

    # Build the CSV now — independent of annotation, so it survives an
    # annotation failure below.
    grades_csv = _build_grades_csv(grades_id)
    if grades_csv and os.getenv("SAVE_LOCAL_GRADES_CSV", "").strip().lower() in {"1", "true", "yes"}:
        _write_grades_csv_file(output_dir, student_name, question_num, grades_csv)

    if progress_callback:
        progress_callback("Step 3/3: Drawing annotations and marks breakdown on PDF...", 85)

    try:
        annotation_ok, annotated_pdf = annotate_pdf(
            input_pdf_path=input_pdf_for_annotation or student_pdf_path,
            output_dir=output_dir,
            student_name=student_name,
            grades_id=grades_id,
            student_pages=student_pages,
        )
        if annotation_ok and annotated_pdf:
            grades_coll.update_one(
                {"_id": ObjectId(grades_id)},
                {"$set": {"annotated_pdf_path": annotated_pdf}},
            )
    except Exception as e:
        clean_msg, show_tb = classify_error(e)
        logger.error(f"[Annotation] {clean_msg}", exc_info=show_tb)
        return False, clean_msg, None, grades_csv

    if progress_callback:
        progress_callback("Grading and annotation complete!", 100)

    duration = (datetime.now() - start_time).total_seconds()
    status = "SUCCESS" if annotation_ok else "PARTIAL (graded, annotation failed)"
    logger.info(f"GRADE FROM DB {status} → {student_name} Q{question_num} | {duration:.2f}s")
    logger.info("=" * 70 + "\n")

    msg = "Grading and annotation complete" if annotation_ok else "Annotation failed"
    return annotation_ok, msg, annotated_pdf, grades_csv


def grade_from_db(
    model_answers_id: str,
    student_pdf_path: str,
    student_pages: List[int],
    student_name: str,
    output_dir: str,
    question_num: str,
    question_type: str = "numerical",
    reuse_student_answers_id: Optional[str] = None,
    input_pdf_for_annotation: Optional[str] = None,
    progress_callback: Optional[Any] = None,
    force_regrade: bool = False,
) -> Tuple[bool, str, Optional[str], Optional[str]]:
    """Sync entry point for the production grading pipeline.

    Pass `reuse_student_answers_id` to grade an existing extraction rather than
    re-reading the PDF — see grade_from_db_async.
    """
    return asyncio.run(
        grade_from_db_async(
            model_answers_id=model_answers_id,
            student_pdf_path=student_pdf_path,
            student_pages=student_pages,
            student_name=student_name,
            output_dir=output_dir,
            question_num=question_num,
            question_type=question_type,
            reuse_student_answers_id=reuse_student_answers_id,
            input_pdf_for_annotation=input_pdf_for_annotation,
            progress_callback=progress_callback,
            force_regrade=force_regrade,
        )
    )

