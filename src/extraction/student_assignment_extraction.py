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
            logger.info(
                f"Extracting student answers from {Path(self.pdf_path).name} pages {self.pages}"
            )
            raw_data = self._extract_with_vision()
            data_with_meta = add_metadata(
                raw_data, self.pdf_path, self.pages, student_name=self.student_name
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
