import base64
import os
import re
from typing import Any, Dict, List, Tuple, Type

import fitz
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from logging_config import logger
from llm_setup import _build_chat_model
from errors.exceptions import PDFExtractionError
from errors.classifier import classify_error

load_dotenv()


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    cleaned = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    cleaned = re.sub(r' {3,}', '  ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    lines = [line.rstrip() for line in cleaned.split('\n')]
    return '\n'.join(lines).strip()


def clean_dict_values(data: Any) -> Any:
    """Recursively clean all string values."""
    if isinstance(data, dict):
        return {k: clean_dict_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_dict_values(item) for item in data]
    elif isinstance(data, str):
        return clean_text(data)
    return data


class PDFExtractor:
    def __init__(self, pdf_path: str, pages: List[int], model_name: str, render_dpi: int = 220):
        self.pdf_path = pdf_path
        self.pages = pages
        self.model_name = model_name
        self.render_dpi = render_dpi
        self.llm = self._get_llm()

    def _get_llm(self):
        provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
        if not provider or provider == "auto":
            provider = "openai"
        try:
            logger.info(f"Using {provider} provider via LangChain (model={self.model_name}, dpi={self.render_dpi})")
            return _build_chat_model(provider, self.model_name, temperature=0)
        except Exception as e:
            clean_msg, show_tb = classify_error(e)
            logger.error(f"LLM initialisation failed: {clean_msg}", exc_info=show_tb)
            raise PDFExtractionError(f"LLM initialisation failed: {clean_msg}") from e

    def _render_pages_as_base64(self) -> List[Tuple[int, str]]:
        base64_images: List[Tuple[int, str]] = []
        doc = fitz.open(self.pdf_path)
        try:
            zoom = self.render_dpi / 72.0
            matrix = fitz.Matrix(zoom, zoom)
            for page_num in self.pages:
                if not (1 <= page_num <= len(doc)):
                    continue
                page = doc[page_num - 1]
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                image_bytes = pix.tobytes("jpeg", jpg_quality=90)
                encoded = base64.b64encode(image_bytes).decode("utf-8")
                base64_images.append((page_num, encoded))
            return base64_images
        finally:
            doc.close()

    def extract(
        self,
        instruction_prompt: str,
        output_schema: Type[BaseModel],
    ) -> Dict[str, Any]:
        try:
            rendered_pages = self._render_pages_as_base64()
            if not rendered_pages:
                raise PDFExtractionError("No pages rendered from PDF")

            structured_llm = self.llm.with_structured_output(output_schema)

            content = [{"type": "text", "text": instruction_prompt}]
            for page_num, img_base64 in rendered_pages:
                content.append({
                    "type": "text",
                    "text": f"PAGE {page_num} (in order; this is part of the provided page set)",
                })
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"},
                })

            response = structured_llm.invoke([HumanMessage(content=content)])

            data = response.model_dump() if isinstance(response, BaseModel) else response
            logger.info(f"Successfully extracted from {self.pdf_path}")
            return clean_dict_values(data)

        except PDFExtractionError:
            raise
        except Exception as e:
            clean_msg, show_tb = classify_error(e)
            logger.error(f"PDF extraction failed: {clean_msg}", exc_info=show_tb)
            raise PDFExtractionError(clean_msg) from e
