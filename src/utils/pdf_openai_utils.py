import io
import fitz
from typing import List

from llm_setup import client
from logging_config import logger


def create_pdf_subset(pdf_path: str, pages: List[int]) -> io.BytesIO:
    try:
        doc = fitz.open(pdf_path)
        new_doc = fitz.open()
 
        for p in pages:
            if 1 <= p <= len(doc):
                new_doc.insert_pdf(doc, from_page=p - 1, to_page=p - 1)

        pdf_bytes = new_doc.tobytes()
        buf = io.BytesIO(pdf_bytes)
        buf.name = "subset.pdf"
        buf.seek(0)

        doc.close()
        new_doc.close()
        return buf
    except Exception as e:
        logger.error(f"Failed to create PDF subset: {e}", exc_info=True)
        raise RuntimeError(f"PDF subset creation failed for {pdf_path}") from e


def upload_to_openai(pdf_buffer: io.BytesIO, filename: str = "subset.pdf") -> str:
    try:
        file_obj = client.files.create(file=pdf_buffer, purpose="user_data")
        logger.info(f"Uploaded PDF to OpenAI → file_id: {file_obj.id}")
        return file_obj.id
    except Exception as e:
        logger.error(f"OpenAI file upload failed: {e}", exc_info=True)
        raise RuntimeError("Failed to upload PDF to OpenAI") from e