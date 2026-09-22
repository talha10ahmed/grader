"""Compatibility wrapper.

The PDF annotation implementation lives in `pdf_annotation/annotator.py`.
This module remains to preserve existing imports like:

    from dummy_comments_scenario import annotate_pdf
"""

from pdf_annotation.annotator import annotate_pdf

__all__ = ["annotate_pdf"]
