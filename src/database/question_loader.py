from bson import ObjectId
from database.mongodb import get_collection


def list_available_questions() -> list[dict]:
    """Return summary of all model answers for the question-selector dropdown."""
    coll = get_collection("model_answers")
    cursor = coll.find(
        {},
        {"_id": 1, "question_title": 1, "total_marks": 1, "max_marks": 1, "description": 1},
    )
    result = []
    for doc in cursor:
        result.append({
            "_id": str(doc["_id"]),
            "question_title": doc.get("question_title", "Untitled"),
            "total_marks": doc.get("max_marks") or doc.get("total_marks", "?"),
            "description": doc.get("description", ""),
        })
    return result


def get_question_by_id(question_id: str) -> dict | None:
    """Fetch a single model answer document by its MongoDB _id."""
    coll = get_collection("model_answers")
    doc = coll.find_one({"_id": ObjectId(question_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
    return doc
