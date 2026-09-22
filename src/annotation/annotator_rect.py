# ==================== RECTANGLE & LAYOUT HELPERS ====================
#
# Provides:
#   - Underline drawing primitive
#   - Page-line iteration (text dict-based)
#   - Rect expansion to line / row bounds
#   - Heading and numeric-content detection
#   - Header redirection (move anchor to next numeric row)
#   - Same-line check, next-numeric-line search, number-word refinement

import re
from typing import Iterable, Optional

import fitz

from .annotator_config import CONFIG
from .annotator_ocr import _page_words, _page_dict, _page_search
from .annotator_text import _normalize_text_for_match, _strip_llm_artifacts


# ── Underline drawing ──────────────────────────────────────────────────────────

def _draw_underline_for_rect(page, rect: fitz.Rect, phrase_only: bool = False) -> None:
    """Draw a red underline for the text at *rect*.

    When *phrase_only* is True the underline covers only the matched
    phrase (with minimal padding) — used in holistic mode where each
    key_phrase gets its own targeted underline.  Otherwise it spans the
    full row (original behaviour for standard mode).
    """
    if not page or not rect:
        return
    line_rect = _expand_rect_to_line(page, rect)
    if phrase_only:
        underline_start = max(rect.x0 - 1, 10)
        underline_end = min(rect.x1 + 1, page.rect.width - 10)
    else:
        underline_rect = _expand_rect_to_row(page, rect)
        underline_start = max(underline_rect.x0, 10)
        underline_end = min(underline_rect.x1, page.rect.width - 50)
    underline_y = line_rect.y1 + 2
    page.draw_line(
        (underline_start, underline_y),
        (underline_end, underline_y),
        color=CONFIG['underline_color'],
        width=0.8,
    )


# ── Line / row extraction ──────────────────────────────────────────────────────

def _iter_page_lines(page) -> Iterable[tuple[str, fitz.Rect]]:
    """Yield (line_text, line_rect) for every text line on the page."""
    try:
        data = _page_dict(page)
        for block in data.get("blocks", []) or []:
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                line_text = "".join((s.get("text") or "") for s in spans).strip()
                bbox = line.get("bbox")
                if not line_text or not bbox:
                    continue
                yield line_text, fitz.Rect(bbox)
    except Exception:
        return


def _expand_rect_to_line(page, rect: fitz.Rect) -> fitz.Rect:
    """Return the bounding rect of the full text line that overlaps *rect*."""
    if not rect:
        return rect

    best_line_rect: Optional[fitz.Rect] = None
    best_overlap = 0.0

    for _, line_rect in _iter_page_lines(page):
        if not line_rect.intersects(rect):
            continue
        overlap = (line_rect & rect).get_area()
        if overlap > best_overlap:
            best_overlap = overlap
            best_line_rect = line_rect

    return best_line_rect or rect


def _expand_rect_to_row(page, rect: fitz.Rect) -> fitz.Rect:
    """Return the union of all word bboxes on the same row as *rect*.

    'Same row' means the word's vertical centre is within y_tolerance of
    the given rect's vertical centre.  Stays strict to avoid spanning
    adjacent lines.
    """
    if not rect:
        return rect

    words = _page_words(page)
    tol = max(CONFIG.get('y_tolerance', 6), 6)
    target_yc = (rect.y0 + rect.y1) / 2

    row_rect: Optional[fitz.Rect] = None
    for w in words:
        wrect = fitz.Rect(w[:4])
        w_yc = (wrect.y0 + wrect.y1) / 2
        if abs(w_yc - target_yc) > tol:
            continue
        row_rect = wrect if row_rect is None else row_rect | wrect

    return row_rect or _expand_rect_to_line(page, rect)


def _line_text_for_rect(page, rect: fitz.Rect) -> str:
    """Return the text of the first line that intersects *rect*."""
    if not rect:
        return ""
    for line_text, line_rect in _iter_page_lines(page):
        if line_rect.intersects(rect):
            return line_text.strip()
    return ""


def _text_under_rect(page, rect: fitz.Rect, words=None) -> str:
    """Return the text of the words whose CENTRES fall inside *rect*.

    search_for() rects carry loose vertical bounds — a rect sitting on one
    line routinely overlaps the next line's ascenders — so
    page.get_textbox() bleeds words in from neighbouring lines and cannot
    be used to measure what a hit actually covers.  Testing the centre of
    each word bbox instead keeps the extraction tight to the matched
    fragment.  Pass *words* to reuse a cached _page_words() result across
    repeated calls on the same page.
    """
    if not rect:
        return ""
    if words is None:
        words = _page_words(page)
    picked: list[tuple[float, float, str]] = []
    for w in words:
        wr = fitz.Rect(w[:4])
        cx = (wr.x0 + wr.x1) / 2
        cy = (wr.y0 + wr.y1) / 2
        if rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y0 - 1 <= cy <= rect.y1 + 1:
            picked.append((wr.y0, wr.x0, w[4]))
    picked.sort()
    return re.sub(r"\s+", " ", " ".join(t for _, _, t in picked)).strip()


# Guards for _group_wrapped_hits. A phrase realistically wraps over a
# handful of lines; the cap stops a mis-measured fragment from swallowing
# the rest of the page's hits into one group.
_MAX_WRAP_LINES = 4
# Fraction of the needle that must be covered before a hit counts as
# complete. Slack absorbs glyph/whitespace differences between the search
# variant and the PDF's own rendering.
_WRAP_COVERAGE = 0.9


def _group_wrapped_hits(page, needle: str, hits: list) -> list[list[fitz.Rect]]:
    """Group a flat ``search_for()`` result into logical occurrences.

    PyMuPDF returns ONE rect per PHYSICAL LINE, so a phrase that wraps
    arrives as several consecutive rects that are indistinguishable in
    shape from several separate occurrences of a shorter phrase:

        needle wraps over 2 lines  -> [Rect(y=88), Rect(y=103)]   ONE hit
        short needle appears twice -> [Rect(y=88), Rect(y=148)]   TWO hits

    There is no grouping flag on the API (verified on PyMuPDF 1.26.7), so
    we disambiguate by measuring the text each rect actually covers.  A
    rect whose words already account for the whole needle is a complete
    standalone occurrence; a rect covering only part of it is a fragment,
    so we keep absorbing following rects — which must sit lower on the
    page — until the needle's length is accounted for.

    Returns a list of groups, each a list of one or more rects in reading
    order.  Callers underline every rect of a group so wrapped evidence
    gets an underline beneath each physical line it spans.
    """
    if not hits:
        return []
    target_len = len(re.sub(r"\s+", " ", (needle or "")).strip())
    if target_len <= 0:
        return [[r] for r in hits]

    words = _page_words(page)
    groups: list[list[fitz.Rect]] = []
    i = 0
    total = len(hits)
    while i < total:
        group = [hits[i]]
        covered = len(_text_under_rect(page, hits[i], words))
        i += 1
        while (
            i < total
            and covered < target_len * _WRAP_COVERAGE
            and len(group) < _MAX_WRAP_LINES
            and hits[i].y0 > group[-1].y0 + 1  # strictly a later line
        ):
            group.append(hits[i])
            # +1 for the space the line break stands in for.
            covered += 1 + len(_text_under_rect(page, hits[i], words))
            i += 1
        groups.append(group)
    return groups


def _box_overlaps_page_text(page, box: fitz.Rect) -> bool:
    """Return True if *box* overlaps any word bounding box on the page."""
    if not box:
        return False
    try:
        for w in _page_words(page):
            wrect = fitz.Rect(w[:4])
            if wrect.intersects(box):
                return True
    except Exception:
        return False
    return False


# ── Heading / numeric content detection ───────────────────────────────────────

def _is_heading_like(text: str) -> bool:
    """Heuristic: return True when *text* looks like a section heading rather
    than a table value or answer line.
    """
    if not text or not isinstance(text, str):
        return False
    t = _normalize_text_for_match(text)
    if not t:
        return False

    if t.endswith(":"):
        return True
    if "calculated as follows" in t or "is as follows" in t:
        return True
    if t.startswith("revised consolidated statement"):
        return True
    if t.startswith("the basic eps"):
        return True

    if not re.search(r"\d", t) and len(t.split()) <= 12:
        heading_keywords = (
            "calculated",
            "as follows",
            "accounting",
            "treatment",
            "journal",
            "statement",
            "revised",
            "electrostatic",
        )
        if any(k in t for k in heading_keywords):
            return True
    return False


def _row_has_numeric_content(page, rect: fitz.Rect) -> bool:
    """Return True when the y-band around *rect* contains a meaningful number.

    Filters out table headers (£000, W1, etc.) and zero-only values.
    """
    if not rect:
        return False

    try:
        words = _page_words(page)
    except Exception:
        return False

    tol = max(CONFIG.get('y_tolerance', 6), 6)

    def _has_nonzero_digit(s: str) -> bool:
        return bool(re.search(r"[1-9]", s or ""))

    def _is_value_like(token: str) -> bool:
        t = (token or "").strip()
        if not t:
            return False
        # Short code labels like W1, A12 are not values
        if re.fullmatch(r"[A-Za-z]{1,3}\d{1,3}", t):
            return False
        # Currency/percent: only count if a non-zero digit is present
        if re.search(r"£|\$|%", t) and re.search(r"\d", t):
            digits = re.sub(r"\D", "", t)
            return _has_nonzero_digit(digits)
        # Pure numeric-ish values (allow commas, parens, minus)
        t2 = t.replace(",", "").strip("()")
        if re.fullmatch(r"-?\d+(?:\.\d+)?", t2):
            digits = re.sub(r"\D", "", t2)
            return len(digits) >= 2 and _has_nonzero_digit(digits)
        return False

    for w in words:
        wrect = fitz.Rect(w[:4])
        if abs(wrect.y0 - rect.y0) <= tol or abs(wrect.y1 - rect.y1) <= tol:
            txt = (w[4] or "").strip()
            if _is_value_like(txt):
                return True
    return False


# ── Same-line test ─────────────────────────────────────────────────────────────

def is_on_same_line(r1: fitz.Rect, r2: fitz.Rect) -> bool:
    """Return True when two rects share the same text line (y0 within tolerance)."""
    return abs(r1.y0 - r2.y0) <= CONFIG['y_tolerance']


# ── Next-numeric-line search ───────────────────────────────────────────────────

def _find_next_numeric_line(page, from_rect: fitz.Rect, max_down: float = 140.0) -> Optional[fitz.Rect]:
    """Return the rect of the next line below *from_rect* that contains a
    meaningful (non-zero) number, within *max_down* points.
    """
    if not from_rect:
        return None

    for line_text, line_rect in _iter_page_lines(page):
        if line_rect.y0 <= from_rect.y1 + 1:
            continue
        if line_rect.y0 > from_rect.y1 + max_down:
            break
        t = _normalize_text_for_match(line_text)
        digits = re.sub(r"\D", "", t)
        if digits and re.search(r"[1-9]", digits):
            return line_rect
    return None


# ── Number-word refinement ─────────────────────────────────────────────────────

def _refine_to_numberish_word_on_line(page, line_rect: fitz.Rect) -> Optional[fitz.Rect]:
    """Return the rect of the rightmost numeric/currency word on *line_rect*."""
    if not line_rect:
        return None
    words = _page_words(page)
    candidates: list[tuple[fitz.Rect, str]] = []
    for w in words:
        rect = fitz.Rect(w[:4])
        if abs(rect.y0 - line_rect.y0) > CONFIG['y_tolerance']:
            continue
        txt = (w[4] or "").strip()
        if not txt:
            continue
        if re.search(r"\d", txt) or re.search(r"£|\$|%", txt):
            candidates.append((rect, txt))

    if not candidates:
        return None

    candidates.sort(key=lambda it: it[0].x0)
    return candidates[-1][0]


# ── Header redirection ─────────────────────────────────────────────────────────

def _redirect_if_header_like(
    page,
    rect: fitz.Rect,
    expand_to_line: bool,
    max_down: float = 240.0,
) -> Optional[fitz.Rect]:
    """If *rect* sits on a heading/header row, redirect to the next numeric line.

    Returns the redirected rect, or None when no redirect is needed/possible.
    """
    if not rect:
        return None

    line_rect = _expand_rect_to_line(page, rect)
    line_text = _line_text_for_rect(page, line_rect)

    norm = _normalize_text_for_match(line_text)
    short_label = len(norm.split()) <= 3 and len(norm) <= 24
    wp_ref = any(re.fullmatch(r"[a-z]{1,3}\d{1,3}", tok) for tok in norm.split())
    has_any_digit = bool(re.search(r"\d", norm))
    has_digit_besides_wp = any(
        re.search(r"\d", tok)
        for tok in norm.split()
        if not re.fullmatch(r"[a-z]{1,3}\d{1,3}", tok)
    )

    row_has_values = _row_has_numeric_content(page, line_rect)
    headerish = (
        _is_heading_like(line_text)
        or ((short_label or wp_ref) and not row_has_values)
        or (wp_ref and not has_digit_besides_wp and not row_has_values)
        or (not has_any_digit and not row_has_values and short_label)
    )

    if not headerish:
        return None

    nxt = _find_next_numeric_line(page, line_rect, max_down=max_down)
    if not nxt:
        return None

    if not expand_to_line:
        refined = _refine_to_numberish_word_on_line(page, nxt)
        return refined or nxt
    return nxt
