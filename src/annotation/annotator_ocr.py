# ==================== OCR-AWARE PAGE HELPERS ====================
#
# _ocr_pages: page indices (0-based) that have significant images and need OCR.
# _ocr_tp_cache: page.number → (page_obj, textpage).  Storing the page object
#   prevents GC from destroying it, which keeps the textpage's weak-ref alive.
# All helpers accept the caller's page object and route through the cached
# (identity-matched) page when an OCR textpage is present.

import fitz
from logging_config import logger

_ocr_pages: set[int] = set()
_ocr_tp_cache: dict[int, tuple] = {}  # page.number → (page_obj, textpage)


def _get_ocr_page_and_tp(page):
    """Return (page_to_use, textpage_or_None) for OCR-aware calls.

    When OCR is active for this page the *cached* page object is returned
    (its identity matches tp.parent).  Otherwise returns (page, None).
    """
    if page.number not in _ocr_pages:
        return page, None
    cached = _ocr_tp_cache.get(page.number)
    if cached is not None:
        return cached
    try:
        tp = page.get_textpage_ocr(dpi=200, full=True)
        _ocr_tp_cache[page.number] = (page, tp)
        logger.info(f"  OCR textpage created for page {page.number + 1}")
        return page, tp
    except Exception as e:
        logger.debug(f"  OCR failed for page {page.number + 1}: {e}")
        _ocr_pages.discard(page.number)
        return page, None


def _page_search(page, text, **kwargs):
    """OCR-aware replacement for page.search_for()."""
    ocr_page, tp = _get_ocr_page_and_tp(page)
    if tp is not None:
        return ocr_page.search_for(text, textpage=tp, **kwargs)
    return page.search_for(text, **kwargs)


def _page_words(page):
    """OCR-aware replacement for page.get_text('words')."""
    ocr_page, tp = _get_ocr_page_and_tp(page)
    if tp is not None:
        return ocr_page.get_text("words", textpage=tp)
    return page.get_text("words")


def _page_dict(page):
    """OCR-aware replacement for page.get_text('dict')."""
    ocr_page, tp = _get_ocr_page_and_tp(page)
    if tp is not None:
        return ocr_page.get_text("dict", textpage=tp)
    return page.get_text("dict")


def _page_text(page):
    """OCR-aware replacement for page.get_text('text')."""
    ocr_page, tp = _get_ocr_page_and_tp(page)
    if tp is not None:
        return ocr_page.get_text("text", textpage=tp)
    return page.get_text("text")


def _page_has_significant_images(page) -> bool:
    """Return True if the page contains images large enough to hold text.

    Uses pixel dimensions from image metadata (avoids get_image_bbox failures
    on some PDF producers).  Threshold: > 100×100 pixels.
    """
    try:
        images = page.get_images(full=True)
    except Exception:
        return False
    if not images:
        return False
    for img_info in images:
        try:
            w = int(img_info[2])
            h = int(img_info[3])
            if w > 100 and h > 100:
                return True
        except Exception:
            continue
    return False


def _init_ocr_cache(doc, allowed_pages: list[int]) -> None:
    """Scan pages for significant images and mark them for lazy OCR.

    Does NOT create TextPage objects here — they hold a weak-ref to their
    parent page which is GC'd after this loop.  Instead populates _ocr_pages
    so _get_ocr_page_and_tp() creates them lazily at search time.
    """
    global _ocr_pages, _ocr_tp_cache
    _ocr_pages = set()
    _ocr_tp_cache = {}

    for p in allowed_pages:
        try:
            page = doc[p - 1]
        except Exception:
            continue
        if _page_has_significant_images(page):
            _ocr_pages.add(page.number)
            logger.info(f"  Page {p} marked for OCR (has significant images)")

    if _ocr_pages:
        logger.info(f"  {len(_ocr_pages)} page(s) will be OCR'd on first access")
