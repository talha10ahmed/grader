from datetime import datetime
from pathlib import Path
from typing import Any, Type
from logging_config import logger


def add_metadata(data: dict, pdf_path: str, pages: list, **kwargs) -> dict:
    now_iso = datetime.utcnow().isoformat()
    filename = Path(pdf_path).name

    data.update({
        "pages": pages,
        "extracted_at": now_iso,
        "source_filename": filename,
    })
    
    for key, value in kwargs.items():
        if value is not None:
            data[key] = value
    
    return data


def validate_and_prepare(data: dict, schema_class: Type) -> dict:
    try:
        validated = schema_class(**data)
        return validated.model_dump(exclude_none=True)
    except Exception as ve:
        logger.warning(f"Schema validation failed – saving raw data: {ve}", exc_info=True)
        return data


def save_to_mongodb(collection, data: dict, entity_type: str = "document") -> str:
    try:
        result = collection.insert_one(data)
        doc_id = str(result.inserted_id)
        logger.info(f"Saved {entity_type} to MongoDB → _id = {doc_id}")
        return doc_id
    except Exception as e:
        logger.error(f"MongoDB insertion failed: {e}", exc_info=True)
        raise