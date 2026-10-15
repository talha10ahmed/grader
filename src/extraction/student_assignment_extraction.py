import hashlib
import llm_setup

from pathlib import Path
from typing import List, Optional

from pymongo.collection import Collection

from database.mongodb import get_collection
from logging_config import logger
from providers.langchain_pdf_extractor import PDFExtractor
from prompts.student_extraction_prompts import get_student_extraction_prompt
from schemas.student_assignment import StudentAssignmentDocument
from utils.db_utils import add_metadata, validate_and_prepare, save_to_mongodb
from errors import StudentAssignmentExtractionError, PDFExtractionError, classify_error


def compute_pdf_hash(pdf_path: str) -> str:
    """Compute SHA-256 hash of a PDF file to detect identical uploads."""
    hasher = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class StudentAssignmentExtractor:
    COLLECTION_NAME = "student_assignments"

    def __init__(
        self,
        pdf_path: str,
        pages: List[int],
        student_name: Optional[str] = None,
        question_number: Optional[str] = None,
    ):
        self.pdf_path = pdf_path
        self.pages = pages
        self.student_name = student_name
        self.question_number = question_number
        self.collection: Collection = get_collection(self.COLLECTION_NAME)

    def _has_digital_text(self) -> bool:
        """Check if the PDF contains embedded digital text (typed submissions)."""
        try:
            import fitz
            doc = fitz.open(self.pdf_path)
            total_chars = sum(len(doc[p - 1].get_text().strip()) for p in self.pages if 1 <= p <= len(doc))
            doc.close()
            return total_chars >= (50 * len(self.pages))
        except Exception:
            return False

    def _extract_digital_text(self) -> dict:
        """Extract clean digital text directly from PDF without lossy vision OCR."""
        import fitz
        import unicodedata

        doc = fitz.open(self.pdf_path)
        page_texts = []
        for p in self.pages:
            if 1 <= p <= len(doc):
                raw = doc[p - 1].get_text()
                norm = unicodedata.normalize("NFKD", raw).replace("\ufffd", " ")
                page_texts.append({"page": p, "text": norm})
        doc.close()

        full_text = "\n\n".join([f"--- PAGE {pt['page']} ---\n{pt['text']}" for pt in page_texts])
        prompt = get_student_extraction_prompt(self.question_number) + (
            "\n\n═══════════════════════════════════════════════════\n"
            "SOURCE TEXT FROM STUDENT PDF (PRESERVE EXACT CHARACTERS/NUMBERS):\n"
            "═══════════════════════════════════════════════════\n"
            + full_text
        )

        llm = llm_setup._build_chat_model(llm_setup.GRADING_PROVIDER, llm_setup.LLM_EXTRACTION_MODEL, temperature=0)
        structured_llm = llm.with_structured_output(StudentAssignmentDocument)
        response = structured_llm.invoke(prompt)

        data = response.model_dump() if hasattr(response, "model_dump") else response
        if not data.get("page_texts"):
            data["page_texts"] = page_texts
        logger.info(f"Student assignment successfully extracted from digital text ({len(full_text)} chars)")
        return data

    def _extract_with_vision(self) -> dict:
        try:
            prompt = get_student_extraction_prompt(self.question_number)
            extractor = PDFExtractor(
                self.pdf_path,
                self.pages,
                model_name=llm_setup.LLM_EXTRACTION_MODEL,
                render_dpi=llm_setup.LLM_PDF_RENDER_DPI,
            )
            data = extractor.extract(
                instruction_prompt=prompt,
                output_schema=StudentAssignmentDocument,
            )
            logger.info("Student assignment successfully extracted via LangChain")
            return data
        except PDFExtractionError as e:
            raise StudentAssignmentExtractionError(str(e)) from e

    def run(self) -> Optional[str]:
        try:
            pdf_hash = compute_pdf_hash(self.pdf_path)
            pages_sorted = sorted(self.pages)

            # Check if this exact PDF has already been extracted for these pages
            query = {
                "pdf_content_hash": pdf_hash,
                "pages": pages_sorted,
            }
            if self.question_number:
                query["question"] = str(self.question_number)

            cached = self.collection.find_one(query, sort=[("_id", -1)])
            if cached:
                cached_id = str(cached["_id"])
                logger.info(
                    f"Found existing extraction {cached_id} for PDF hash {pdf_hash[:12]}... (pages {pages_sorted}). "
                    f"Holding extraction constant to guarantee marking consistency."
                )
                return cached_id

            if self._has_digital_text():
                logger.info(
                    f"Extracting student answers directly from digital text in {Path(self.pdf_path).name} pages {self.pages}"
                )
                raw_data = self._extract_digital_text()
            else:
                logger.info(
                    f"Extracting student answers via Vision OCR from {Path(self.pdf_path).name} pages {self.pages}"
                )
                raw_data = self._extract_with_vision()
            data_with_meta = add_metadata(
                raw_data,
                self.pdf_path,
                self.pages,
                student_name=self.student_name,
                pdf_content_hash=pdf_hash,
            )
            to_save = validate_and_prepare(data_with_meta, StudentAssignmentDocument)
            return save_to_mongodb(self.collection, to_save, entity_type="student assignment")
        except Exception as e:
            clean_msg, show_tb = classify_error(e)
            logger.error(f"Student assignment extraction failed: {clean_msg}", exc_info=show_tb)
            return None


def extract_assignment_pipeline(
    pdf_path: str,
    pages: List[int],
    student_name: Optional[str] = None,
    question_number: Optional[str] = None,
) -> Optional[str]:
    return StudentAssignmentExtractor(pdf_path, pages, student_name, question_number).run()
