"""All custom exception classes for the grading pipeline.

Import from here, not from individual modules.
"""


class PDFExtractionError(Exception):
    """Raised when PDF rendering or LLM extraction fails."""


class QuestionExtractionError(Exception):
    """Raised when question paper extraction fails."""


class ModelAnswerExtractionError(Exception):
    """Raised when model answer / rubric extraction fails."""


class StudentAssignmentExtractionError(Exception):
    """Raised when student assignment extraction fails."""


class GradingError(Exception):
    """Raised when the grading step fails."""


class AnnotationError(Exception):
    """Raised when PDF annotation fails."""
