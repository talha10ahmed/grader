# ==================== MAIN ANNOTATION ORCHESTRATION ====================
#
# annotate_pdf() is the public entry point.  It:
#   1. Loads the grading document and student PDF.
#   2. Detects per-page Y boundaries for the target question.
#   3. In holistic mode:  underlines + tick-marks each correct point,
#                         then places sub-question scores near student labels.
#   4. In standard mode: places criterion scores near evidence anchors.
#   5. Places feedback comment popups.
#   6. Saves the annotated PDF.
#
# Bug fix included:
#   FIX-3: underline is drawn at tick_rect (key_phrase / evidence),
#           guaranteeing tick mark and underline are always co-located on
#           the correct student answer — never on a separate line.

import json
import os
import re
from datetime import datetime
from typing import List, Optional, Tuple

import fitz
from bson import ObjectId

from logging_config import logger
from database.mongodb import get_collection

from .annotator_ocr import _init_ocr_cache, _page_search, _page_text, _page_words
from .annotator_text import (
    _normalize_text_for_match, _strip_llm_artifacts,
    _tokenize, _line_key, _build_anchor_variations, _build_candidate_fragments,
    _normalize_symbols_for_match,
)
from .annotator_rect import (
    _draw_underline_for_rect, _is_heading_like, _iter_page_lines,
    _group_wrapped_hits,
)
from .annotator_match import resolve_anchor_rect, _rank_pages_for_anchor
from .annotator_draw import (
    _safe_float, _fmt_mark_value, _place_score_label,
    _place_ticks, place_score_near_anchor, add_main_score, add_popup_for_comment,
    place_not_required_marker, compute_subq_y_bounds, _strip_subq_prefix,
)


# ── Strict numerical evidence finder ──────────────────────────────────────────


def _try_tabular_row_match(
    doc,
    evidence_text: str,
    ranked_pages: List[int],
    min_y_per_page: Optional[dict],
    max_y_per_page: Optional[dict],
) -> Tuple[List["fitz.Rect"], int]:
    """SOCIE / multi-column table fallback for strict matching.

    The LLM frequently emits evidence like "<row label> <column header>
    <value>" — joining tokens that live on different physical PDF lines
    in a tabular layout (label on line A, values on line B, column header
    "Total" on a separate header row). Strict literal search misses every
    time.

    This fallback extracts the row label's alphabetic prefix (stopping at
    the first number, uppercase abbreviation, or known column-header word
    like "Total"), then searches the PDF strictly for progressively shorter
    versions of that prefix. When the matched label line carries no
    numeric content (the row wraps across two lines), it ALSO returns the
    nearest data line below — so both lines get underlined and the score
    lands on the numeric line via the existing _pick_best_rect_for_score.

    Returns ([label_rect, data_rect], page_num) for a wrapped row,
    ([single_rect], page_num) for a row that fits on one line, or
    ([], -1) when no usable row label is found.
    """
    # Pull the row label from the start of the evidence.
    label_words: List[str] = []
    for w in evidence_text.split():
        if re.search(r"\d", w):
            break
        if w.lower().strip(".,") in {"total", "subtotal", "sub-total"}:
            break
        stripped = w.strip(".,()/:")
        if stripped.isupper() and len(stripped) >= 2:
            break  # column-header abbreviation (NCI, OCI, FX, …)
        label_words.append(w)

    if not label_words:
        return [], -1

    def _outside(page_num: int, rect) -> bool:
        if min_y_per_page:
            lo = min_y_per_page.get(page_num)
            if lo is not None and rect.y0 < lo:
                return True
        if max_y_per_page:
            hi = max_y_per_page.get(page_num)
            if hi is not None and rect.y0 > hi:
                return True
        return False

    def _line_text_at(page, rect) -> str:
        for line_text, line_rect in _iter_page_lines(page):
            if line_rect.intersects(rect):
                return line_text
        return ""

    def _has_numeric(text: str) -> bool:
        return bool(re.search(r"\d", text or ""))

    def _looks_like_column_header(text: str) -> bool:
        # A multi-column header row has many alpha tokens and no numbers.
        # Rejecting these prevents matching "retained earnings" against the
        # SOCIE header row and stamping the mark on the wrong data row.
        if re.search(r"\d", text or ""):
            return False
        words = [w for w in (text or "").split() if len(w) >= 3 and w[0].isalpha()]
        return len(words) >= 4

    def _find_data_rects_for_row(page, label_rect, max_gap_below: float = 18.0):
        # Returns all data-column rects that belong to the row anchored by
        # label_rect. Two cases:
        #   (a) Single-line row, multi-column layout — PyMuPDF treats each
        #       column word at the same y as a separate "line". The values
        #       sit at the same y as the label.
        #   (b) Wrapped row — the label is on physical line A, the values
        #       on physical line B just below (typical SOCIE gap ~13 px).
        # We collect same-y rects first; only if none exist do we look below.
        label_y_mid = (label_rect.y0 + label_rect.y1) / 2
        same_y: List["fitz.Rect"] = []
        for lt, lr in _iter_page_lines(page):
            if not _has_numeric(lt):
                continue
            line_y_mid = (lr.y0 + lr.y1) / 2
            if abs(line_y_mid - label_y_mid) <= 3.0:
                same_y.append(lr)
        if same_y:
            return same_y
        # No same-y values — look just below for a continuation line.
        best, best_gap = None, max_gap_below
        for lt, lr in _iter_page_lines(page):
            if not _has_numeric(lt):
                continue
            gap = lr.y0 - label_rect.y1
            if 0 < gap <= best_gap:
                best_gap = gap
                best = lr
        return [best] if best is not None else []

    for prefix_len in range(len(label_words), 0, -1):
        prefix = " ".join(label_words[:prefix_len])
        if len(prefix) < 6:
            break
        for page_num in ranked_pages:
            page = doc[page_num - 1]
            try:
                hits = _page_search(page, prefix)
            except Exception:
                hits = []
            for label_rect in hits or []:
                if _outside(page_num, label_rect):
                    continue
                line_text = _line_text_at(page, label_rect)
                if line_text and _is_heading_like(line_text):
                    continue
                if line_text and _looks_like_column_header(line_text):
                    continue
                # Find all data-column rects for the row (same-y OR below-y).
                data_rects = _find_data_rects_for_row(page, label_rect)
                if not data_rects and not _has_numeric(line_text):
                    # No values found anywhere for this row — skip this hit.
                    continue
                # Filter out any data rect that falls outside the question's
                # Y bounds (defensive — shouldn't normally happen).
                data_rects = [r for r in data_rects if not _outside(page_num, r)]
                return [label_rect] + data_rects, page_num

    return [], -1


_NUMERIC_TOKEN_RE = re.compile(r"[-−]?\d{1,3}(?:[,\s]\d{3})+(?:\.\d+)?|[-−]?\d{4,}(?:\.\d+)?")

# Evidence carrying at least this many alphabetic words is treated as PROSE.
_PROSE_WORD_MIN = 8
_ALPHA_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def _is_prose_evidence(text: str) -> bool:
    """True when *text* reads as a sentence rather than a working/table line.

    Target-value narrowing shrinks a rect to the cell holding the criterion's
    number. That is right for a table row ("Share capital 250,000 250,000
    250,000") and wrong for a sentence: narrowing "the results up until
    1 March 20X4 e.g. (9/12) will be consolidated, and the remaining
    associate" to its `9` glyph leaves a one-character underline beneath a
    full line of prose. Prose evidence keeps its whole matched span.
    """
    return len(_ALPHA_WORD_RE.findall(text or "")) >= _PROSE_WORD_MIN


def _header_search_candidates(header: str) -> List[str]:
    """Ordered literal search strings for a column-header hint.

    Accounting table headers are routinely STACKED over two physical lines:

        Net assets W1   Year end     Disposal      Acq
                        31 May X4    1 March X4    1 Jan X0

    The model flattens that into a single hint ("Acq 1 Jan X0"), which never
    appears as contiguous text on any line, so one literal search finds
    nothing and column disambiguation is lost entirely. Falling back to
    contiguous word n-grams — longest first — lets the hint still resolve
    via whichever physical line it does appear on ("1 Jan X0").

    Single-token candidates are kept only when they carry real letters, so a
    bare "1" or "X0" can never anchor a column.
    """
    base = re.sub(r"\s+", " ", (header or "")).strip()
    if not base:
        return []
    tokens = base.split(" ")
    cands: List[str] = []
    seen: set[str] = set()
    for n in range(len(tokens), 0, -1):
        for i in range(0, len(tokens) - n + 1):
            gram = " ".join(tokens[i:i + n])
            if len(gram) < 3:
                continue
            if n == 1 and not re.search(r"[A-Za-z]{3}", gram):
                continue
            if gram not in seen:
                seen.add(gram)
                cands.append(gram)
    cands.sort(key=len, reverse=True)
    return cands


def _find_column_header_rect(
    page,
    column_header: str,
    target_y: Optional[float] = None,
    doc=None,
    allowed_pages: Optional[List[int]] = None,
) -> Optional["fitz.Rect"]:
    """Return the RECT of the given column header text, or None if not found.

    Returns the whole rect — not just its x-center — so the caller can use
    both the center and the LEFT EDGE without re-deriving the header itself.
    An earlier version returned only the center, forcing the caller to
    re-search for the left edge; that second search had no idea which PAGE
    the header was finally found on, so for a multi-page table it matched a
    different occurrence entirely (e.g. the "acq" inside "Net assets at acq
    (w1)" at the left margin) and pinned the column to the wrong x.

    Given the LLM's `_column_header` hint (e.g., "acq date" or "Acq"),
    the annotator finds that header's x-position and prefers value hits at
    the same column.

    Two disambiguation signals when a header word appears MULTIPLE times
    on the page (common — e.g. "Disposal" appears in a heading AND in a
    working section AND as the table column header):

      1. `target_y`: prefer a header hit that sits ABOVE the target value's
         y-row and is the CLOSEST above (smallest gap). Column headers are
         always above the data rows they label.

      2. `doc` + `allowed_pages`: if no usable header on the target page
         (e.g. Amy's PDF where the table header is on page 1 and the fair-
         value-uplift row on page 2), scan the PREVIOUS allowed pages and
         return the bottom-most hit (closest to the page-break boundary,
         i.e. closest to the target row on the next page).

    Returns the rect of the best matching header hit.
    """
    if not column_header:
        return None
    header = str(column_header).strip()
    if not header:
        return None

    # Stacked table headers ("Acq" over "1 Jan X0") never match the model's
    # flattened hint literally, so fall back to progressively shorter
    # contiguous n-grams. See _header_search_candidates.
    candidates = _header_search_candidates(header)
    if not candidates:
        return None

    def _hits_on(_page, needle: str) -> List["fitz.Rect"]:
        try:
            return _page_search(_page, needle) or []
        except Exception as e:
            logger.debug(f"  [col-header] search error for {needle!r}: {e}")
            return []

    def _via(cand: str) -> str:
        return "" if cand == header else f" via {cand!r}"

    # STEP 1: Try the target page first, biased by target_y (closest above).
    # When target_y is provided, ONLY accept hits above it. A "header"
    # occurrence found BELOW the target row is by definition not the column
    # header for that row (it's some other text — a working label, an
    # unrelated paragraph, etc.). If no above-target hit exists on this
    # page, fall through to cross-page search rather than accepting a
    # below-target hit.
    for cand in candidates:
        hits = _hits_on(page, cand)
        if not hits:
            continue
        if target_y is not None:
            above = [h for h in hits if h.y1 <= target_y]
            if not above:
                # This candidate occurs only BELOW the target row, so it is
                # not this row's header. Try the next (shorter) candidate
                # before giving up on the page entirely.
                logger.debug(
                    f"  [col-header] {len(hits)} hit(s) for {cand!r} all BELOW "
                    f"y={target_y:.1f} — trying next candidate"
                )
                continue
            above.sort(key=lambda h: target_y - h.y1)  # smallest gap first
            hit = above[0]
            logger.info(
                f"  [col-header] found {header!r}{_via(cand)} on target page "
                f"above y={target_y:.1f}: x={hit.x0:.1f}-{hit.x1:.1f} "
                f"y={hit.y0:.1f}-{hit.y1:.1f} "
                f"(from {len(above)} above-target hits, {len(hits)} total)"
            )
            return hit
        else:
            # No target_y bias — legacy first-hit behaviour.
            hit = hits[0]
            logger.info(
                f"  [col-header] found {header!r}{_via(cand)} on target page "
                f"(first hit): x={hit.x0:.1f}-{hit.x1:.1f} "
                f"y={hit.y0:.1f}-{hit.y1:.1f} "
                f"({len(hits)} total hits, no target_y filter)"
            )
            return hit

    # STEP 2: Not found on target page — try previous allowed pages.
    # Column headers on preceding pages are common for tables that span
    # multiple pages (e.g. Amy's Bauhaus paper — table header on page 1,
    # fair-value-uplift row on page 2).
    if doc is not None and allowed_pages:
        # Determine target page index in allowed_pages (best effort — we
        # don't know it directly here; work backwards from the last
        # allowed page).
        # Simpler: iterate ALL allowed pages EXCEPT the current one, and
        # pick the header hit whose page appears BEFORE the current page.
        target_page_num = None
        try:
            target_page_num = page.number + 1  # 1-indexed
        except Exception:
            pass
        for other_pnum in sorted(allowed_pages, reverse=True):
            if target_page_num is not None and other_pnum >= target_page_num:
                continue
            try:
                other_page = doc[other_pnum - 1]
            except Exception:
                continue
            for cand in candidates:
                other_hits = _hits_on(other_page, cand)
                if not other_hits:
                    continue
                # Bottom-most hit is closest to next page's top → most
                # relevant header for the target row on the next page.
                other_hits.sort(key=lambda h: h.y1, reverse=True)
                hit = other_hits[0]
                logger.info(
                    f"  [col-header] fallback to page {other_pnum} for "
                    f"{header!r}{_via(cand)}: x={hit.x0:.1f}-{hit.x1:.1f} "
                    f"y={hit.y0:.1f}-{hit.y1:.1f} "
                    f"(bottom-most of {len(other_hits)} hits)"
                )
                return hit

    logger.info(f"  [col-header] NOT FOUND: {header!r}")
    return None


def _narrow_rect_to_target_variants(
    page,
    rect: "fitz.Rect",
    target_variants: List[str],
    column_header: Optional[str] = None,
    doc=None,
    allowed_pages: Optional[List[int]] = None,
) -> Optional["fitz.Rect"]:
    """Return a narrower rect covering any of *target_variants* on the same
    line as *rect*, or None if none is found within the rect's row span.

    Column-header disambiguation (when *column_header* is set): if the
    target value appears MULTIPLE times within the rect (e.g. the value
    250,000 appears in both the disposal-date and acq-date columns of a
    tabular row), the annotator prefers the hit whose x-position aligns
    with the column header supplied by the model. Model-side info; no
    heuristic guessing.

    Fallback (no *column_header* or header not found): first occurrence
    within the rect wins.
    """
    if not rect or not target_variants:
        return None
    y_min = float(rect.y0) - 2
    y_max = float(rect.y1) + 2
    # Resolve the column header's x-position (if provided) once up front.
    # Pass the target row's y (rect.y0 = top of the value row) so the
    # header lookup prefers hits ABOVE that y — the actual table header,
    # not another occurrence of the same word elsewhere on the page.
    # Also pass doc + allowed_pages so headers on the PREVIOUS page
    # (multi-page tables) can be located.
    column_x: Optional[float] = None
    column_header_rect: Optional["fitz.Rect"] = None
    if column_header:
        column_header_rect = _find_column_header_rect(
            page, column_header,
            target_y=float(rect.y0),
            doc=doc, allowed_pages=allowed_pages,
        )
        if column_header_rect is not None:
            column_x = float(
                (column_header_rect.x0 + column_header_rect.x1) / 2.0
            )
    # Expand each variant with common accounting decoration.
    expanded: list[str] = []
    seen: set[str] = set()
    for v in target_variants:
        s = str(v).strip()
        if not s:
            continue
        candidates = [s]
        if not s.endswith(".00"):
            candidates.append(s + ".00")
        # Negative forms (accounting PDFs often use `-1,234` or `−1,234`
        # rather than `(1,234)` parens).
        if not s.startswith(("-", "−")):
            candidates.extend(["-" + s, "−" + s])
            if not s.endswith(".00"):
                candidates.extend(["-" + s + ".00", "−" + s + ".00"])
        for c in candidates:
            if c not in seen:
                seen.add(c)
                expanded.append(c)
    # Try longest variants first — a "18,150,000.00" match is more specific
    # than a "18,150" match that might be a substring of another value.
    expanded.sort(key=len, reverse=True)
    for cand in expanded:
        try:
            hits = _page_search(page, cand)
        except Exception:
            hits = []
        # Y-range check only — do NOT require the hit to fall inside the
        # rect's x-range. When PyMuPDF's search_for matches a multi-word
        # string like "land 400000 400000 0", it can return MULTIPLE
        # rects (one per word, spaced across the row because the cells
        # are separated by whitespace/tabs). The caller's `rect` is often
        # just the label rect (x=74-96 for "land"), while the target
        # value lives further right (x=250-360 for the 400000 cells).
        # Restricting hits to the label rect's x-range would falsely
        # reject the value cells. The Y-range check keeps us on the same
        # visual row, and the column_header hint (below) picks the
        # correct COLUMN among same-row hits.
        in_range_hits: list["fitz.Rect"] = []
        for hit in hits or []:
            if hit.y0 >= y_min and hit.y1 <= y_max:
                in_range_hits.append(hit)
        if not in_range_hits:
            continue
        # Column-header disambiguation. Preferred rule: pick the value hit
        # whose LEFT edge sits AT OR TO THE RIGHT OF the header's left edge,
        # minimising the gap. This handles the typical accounting layout
        # where the header word is LEFT-justified in its column while
        # numeric values are RIGHT-justified — x-center comparison fails
        # because the header's center lies BETWEEN two value columns.
        #
        # Example (Amy's Bauhaus paper):
        #   Header "Disposal" x=320-360 (left-justified).
        #   Value cells for row: 250,000 at x_center 289 (year-end col),
        #     427 (disposal col), 500 (acq col).
        #   x_center-distance would pick 289 (closest to 340 header center)
        #     — WRONG (year-end col).
        #   Left-edge rule: values with x0 >= 320-tol are 409 and 482;
        #     closest left-edge to 320 is 409 → correct disposal col value.
        #
        # Fallback (no value has left >= header left, e.g., right-justified
        # header) — fall back to x-center distance so we still return SOME
        # hit rather than nothing.
        if column_x is not None and column_header and len(in_range_hits) > 1:
            # Left edge comes straight off the rect _find_column_header_rect
            # already chose. Re-searching for it here was the bug behind the
            # 400,000 mis-column: this search only ever looked at the TARGET
            # page, so when the header lived on a PREVIOUS page (multi-page
            # table) it silently matched a different occurrence — the "acq"
            # inside "Net assets at acq (w1)" at x=141.5 — and every value in
            # the row counted as "right of header", handing the column to the
            # leftmost (Year end) cell.
            if column_header_rect is not None:
                header_left_x = float(column_header_rect.x0)
            else:
                header_left_x = column_x - 15  # rough fallback offset

            _tol = 3.0  # tolerance so values very slightly left of header are still considered
            right_of_header = [
                h for h in in_range_hits if h.x0 >= header_left_x - _tol
            ]
            if right_of_header:
                right_of_header.sort(key=lambda r: r.x0 - header_left_x)
                chosen = right_of_header[0]
                logger.info(
                    f"  [col-narrow] variant={cand!r} -> picked LEFT-EDGE-aligned "
                    f"x={chosen.x0:.1f}-{chosen.x1:.1f} (header_left={header_left_x:.1f}, "
                    f"{len(right_of_header)}/{len(in_range_hits)} value(s) to right of header)"
                )
                return chosen
            # No value has left >= header left — fall back to x-center distance.
            in_range_hits.sort(
                key=lambda r: abs(((r.x0 + r.x1) / 2.0) - column_x)
            )
            chosen = in_range_hits[0]
            logger.info(
                f"  [col-narrow] variant={cand!r} -> fallback x-center picked "
                f"x={chosen.x0:.1f}-{chosen.x1:.1f} (closest to col_x={column_x:.1f}, "
                f"header_left={header_left_x:.1f}, no value.left >= header.left)"
            )
            return chosen
        # Default: first (leftmost) occurrence — deterministic when no
        # column disambiguation info is available.
        if len(in_range_hits) > 1:
            logger.info(
                f"  [col-narrow] variant={cand!r} -> {len(in_range_hits)} in-range hits but "
                f"no column_x hint; returning FIRST (leftmost) at x={in_range_hits[0].x0:.1f}"
            )
        return in_range_hits[0]
    return None


def _narrow_rect_to_evidence_value(
    page,
    line_rect: "fitz.Rect",
    evidence_text: str,
) -> Optional["fitz.Rect"]:
    """Return a narrower rect covering just the most distinctive numeric
    token from *evidence_text* on the given line, or None if none found.

    Used by Tier-2's normalized-line containment fallback so the underline
    doesn't span the entire table row when the evidence itself references
    only a specific value. Extracts the LONGEST numeric token from the
    evidence (typically the amount, e.g. `18,800,000` or `-11,725,000.00`)
    and searches the page for it, returning a rect that lies on the same
    line as *line_rect* if found.
    """
    if not evidence_text or not line_rect:
        return None
    # Find all numeric tokens; prefer the longest (most distinctive) one.
    tokens = _NUMERIC_TOKEN_RE.findall(evidence_text)
    if not tokens:
        return None
    tokens_sorted = sorted(set(tokens), key=lambda t: -len(t))
    y_min = float(line_rect.y0) - 2
    y_max = float(line_rect.y1) + 2
    for tok in tokens_sorted:
        # Try the token as-is and also with a leading '-' variant since PDFs
        # sometimes render minus as a different glyph.
        candidates = [tok]
        if tok.startswith(("-", "−")):
            candidates.append(tok.lstrip("-−"))
        for cand in candidates:
            try:
                hits = _page_search(page, cand)
            except Exception:
                hits = []
            for hit in hits or []:
                # Same physical line as line_rect?
                if hit.y0 >= y_min and hit.y1 <= y_max:
                    return hit
    return None


def _find_evidence_strict(
    doc,
    evidence_text: str,
    allowed_pages: List[int],
    placed_marks: set,
    page_token_sets: Optional[dict] = None,
    min_y_per_page: Optional[dict] = None,
    max_y_per_page: Optional[dict] = None,
) -> Tuple[List["fitz.Rect"], int, bool]:
    """Numerical-mode evidence resolver.

    Returns (list_of_rects, page_num, is_wrapped) where *is_wrapped* is
    True only when Tier 1 resolved the evidence to a single occurrence
    that spans several physical lines. The caller uses that flag to keep
    the fragments together — target-value narrowing must not shrink one
    fragment to a lone number and let the sibling fragments be dropped.

    Three strict tiers (no fuzzy / token-overlap matching anywhere):
      1. Literal substring search across surface-form variants (GBP↔£,
         USD↔$, percent↔%).
      2. Normalized-line fallback that strips stray backticks and collapses
         whitespace on both sides — rescues PDF font/glyph artifacts.
      3. Tabular row fallback for SOCIE-style evidence where the LLM joined
         a row label with column headers and values from different physical
         lines. Returns label + data line for wrapped rows so both lines
         get underlined and the score lands on the numeric line.

    Single-line matches return [rect]; wrapped tabular matches return
    [label_rect, data_rect]. No match returns ([], -1, False).
    """
    if not evidence_text:
        return [], -1, False

    variants = _build_anchor_variations(evidence_text)
    ranked_pages = _rank_pages_for_anchor(
        page_token_sets or {}, list(allowed_pages), evidence_text
    )

    def _outside(page_num: int, rect) -> bool:
        if min_y_per_page:
            lo = min_y_per_page.get(page_num)
            if lo is not None and rect.y0 < lo:
                return True
        if max_y_per_page:
            hi = max_y_per_page.get(page_num)
            if hi is not None and rect.y0 > hi:
                return True
        return False

    def _line_text_at(page, rect) -> str:
        for line_text, line_rect in _iter_page_lines(page):
            if line_rect.intersects(rect):
                return line_text
        return ""

    # NOTE on duplicates: we deliberately do NOT skip already-claimed lines
    # via placed_marks. When the LLM awards marks for two criteria pointing
    # at the same student line, both score labels must appear on the PDF —
    # _place_score_label handles visual offset to keep them readable.
    # Skipping here would either (a) drop the second criterion's mark, or
    # (b) push it onto the next occurrence of the evidence on another page,
    # which is a worse outcome (misplacement instead of co-located stack).
    for variant in variants:
        for page_num in ranked_pages:
            page = doc[page_num - 1]
            try:
                hits = _page_search(page, variant)
            except Exception:
                hits = []
            if not hits:
                continue
            # search_for() emits one rect per PHYSICAL LINE, so a phrase
            # that wraps arrives as several consecutive rects. Group them
            # back into logical occurrences and return the WHOLE first
            # acceptable one. Returning only hits[0] underlined just the
            # opening fragment of a wrapped sentence ("Given these…") and
            # silently dropped every continuation line.
            for group in _group_wrapped_hits(page, variant, hits):
                anchor = group[0]
                if _outside(page_num, anchor):
                    continue
                line_text = _line_text_at(page, anchor)
                if line_text and _is_heading_like(line_text):
                    continue
                return list(group), page_num, len(group) > 1

    # Tier 2 fallback: normalized-line containment.
    # When the literal substring search fails on every variant, scan page
    # lines with stray backticks stripped and runs of whitespace collapsed
    # on BOTH sides. Still requires the evidence to be a contiguous substring
    # of the cleaned line — no token overlap, no word clustering. This
    # rescues PDF lines like "NCI post-acq'n profits `   3,850" where a font
    # artifact (stray backtick) prevents an exact match.
    # Symbol-insensitive normalizer. The LLM and the PDF frequently disagree
    # on currency symbols, sign glyphs and thousands separators, which blocks
    # the literal search above and drops the whole line. Canonicalising both
    # sides the same way lets the containment check still land the mark:
    # Symbol-insensitive canonicaliser (shared with holistic mode) — folds
    # currency/math glyphs, units and separators so a value differing only in
    # surface form still matches. See _normalize_symbols_for_match for details.
    _norm = _normalize_symbols_for_match

    ev_norm = _norm(evidence_text)
    if len(ev_norm) >= 10:
        for page_num in ranked_pages:
            page = doc[page_num - 1]
            for line_text, line_rect in _iter_page_lines(page):
                if _outside(page_num, line_rect):
                    continue
                if _is_heading_like(line_text):
                    continue
                if ev_norm in _norm(line_text):
                    # Try to narrow the whole-line rect to just the most
                    # distinctive numeric portion of the evidence — teacher-
                    # style underline sits under the value, not across the
                    # entire table row. Falls back to line_rect if no numeric
                    # substring found or its bbox isn't recoverable.
                    narrowed_rect = _narrow_rect_to_evidence_value(
                        page, line_rect, evidence_text
                    )
                    return [narrowed_rect or line_rect], page_num, False

    # Tier 3 fallback: tabular row label (SOCIE / multi-column tables).
    rects, page_num = _try_tabular_row_match(
        doc, evidence_text, ranked_pages, min_y_per_page, max_y_per_page,
    )
    if rects:
        # Tabular matches are label+value rects on DIFFERENT rows, not a
        # wrapped phrase — narrowing is meant to apply there, so False.
        return rects, page_num, False

    # Tier 4 fallback: near-miss line match.
    # Every tier above needs the quoted evidence to appear in the PDF, modulo
    # symbol folding. It sometimes cannot, through no fault of the student: the
    # grading model silently tidies what it quotes. A student who wrote
    # "Considersation (375k shares * £32)" is quoted back as
    # "Consideration (375k shares * £32) 12,000" — a corrected spelling AND a
    # value pulled in from the next column — and the mark went unplaced.
    #
    # Rather than enumerate the ways a quote can drift, accept the single best
    # near-identical line. Two independent conditions must hold, so this cannot
    # wander onto an unrelated line: high character-level similarity, AND a
    # shared multi-digit number, which is what actually identifies a working
    # line in an accounting script.
    best = _find_near_miss_line(
        doc, evidence_text, ranked_pages, min_y_per_page, max_y_per_page,
    )
    if best is not None:
        line_rect, page_num, line_text = best
        logger.debug(
            f"Tier 4 near-miss match on page {page_num}: "
            f"evidence={evidence_text[:60]!r} line={line_text[:60]!r}"
        )
        narrowed_rect = None
        try:
            narrowed_rect = _narrow_rect_to_evidence_value(
                doc[page_num - 1], line_rect, evidence_text
            )
        except Exception:
            pass
        return [narrowed_rect or line_rect], page_num, False

    return [], -1, False


# Minimum character-level similarity for a Tier 4 near-miss line match.
# Measured on real drift: a corrected typo plus an appended column value scores
# 0.91; unrelated lines in the same script sit below 0.65.
_NEAR_MISS_MIN_RATIO = 0.82

# A number worth matching on. Single digits are far too common in an accounting
# script to identify a line — same reasoning as the grader's short-token guard.
_SALIENT_NUM_RE = re.compile(r"\d[\d,.]*\d")


def _salient_numbers(text: str) -> set:
    """Multi-digit numbers in *text*, comma/period stripped, for line identity."""
    out = set()
    for m in _SALIENT_NUM_RE.finditer(text or ""):
        tok = m.group(0).replace(",", "").rstrip(".")
        if len(tok) >= 2:
            out.add(tok)
    return out


def _find_near_miss_line(
    doc,
    evidence_text: str,
    ranked_pages: List[int],
    min_y_per_page: Optional[dict],
    max_y_per_page: Optional[dict],
):
    """Best near-identical line for *evidence_text*, or None.

    Generic by construction: nothing here is tied to a particular paper,
    student or phrasing. It asks only whether some line reads almost exactly
    like the quoted evidence and carries one of the same numbers.
    """
    import difflib

    target = _normalize_symbols_for_match(evidence_text)
    if len(target) < 12:
        return None  # too short to judge similarity safely
    target_nums = _salient_numbers(evidence_text)
    if not target_nums:
        return None  # prose: no independent check available, so don't guess

    best_ratio = 0.0
    best = None
    for page_num in ranked_pages:
        try:
            page = doc[page_num - 1]
        except Exception:
            continue
        for line_text, line_rect in _iter_page_lines(page):
            if min_y_per_page:
                lo = min_y_per_page.get(page_num)
                if lo is not None and line_rect.y0 < lo:
                    continue
            if max_y_per_page:
                hi = max_y_per_page.get(page_num)
                if hi is not None and line_rect.y0 > hi:
                    continue
            if not (target_nums & _salient_numbers(line_text)):
                continue
            ratio = difflib.SequenceMatcher(
                None, _normalize_symbols_for_match(line_text), target
            ).ratio()
            if ratio > best_ratio:
                best_ratio, best = ratio, (line_rect, page_num, line_text)
    if best is not None and best_ratio >= _NEAR_MISS_MIN_RATIO:
        return best
    return None


# ── Best-rect picker for the score label ─────────────────────────────────────

_JOURNAL_DIR_RE = re.compile(r"^\s*(dr|cr)\b", flags=re.IGNORECASE)
_LINE_DEBIT_RE = re.compile(r"^\s*(?:dr\b|debit\b)", flags=re.IGNORECASE)
_LINE_CREDIT_RE = re.compile(r"^\s*(?:cr\b|credit\b)", flags=re.IGNORECASE)
# Working-line prefixes that identify a NON-journal context. When a journal
# criterion (Dr/Cr) has a rect resolving to a line starting with any of
# these, the rect is rejected — teacher never marks a "Cr net assets 18,800"
# journal on the "less net assets -18,800,000" line of the disposal working.
_LINE_WORKING_RE = re.compile(
    r"^\s*(?:less\b|more\b|add\b|plus\b|minus\b|total\b|sub[-\s]?total\b|"
    r"balance\b|proceeds\b|reserves\b|goodwill\b|nci\b(?!\s+at\s+disposal)|"
    r"share\b|land\b|profit\b|loss\b|fair\s+value\b|consideration\b|"
    r"b/f\b|c/f\b|opening\b|closing\b|at\s+acq\b|at\s+disposal\b)",
    flags=re.IGNORECASE,
)


def _filter_rects_by_criterion_context(
    doc,
    resolved_pending: List[Tuple[int, "fitz.Rect", str]],
    criterion_desc: str,
) -> List[Tuple[int, "fitz.Rect"]]:
    """Reject resolved rects whose line context doesn't match the criterion.

    For JOURNAL criteria (description mentions Dr/Cr/debit/credit):
      - Accept lines starting with the SAME direction verb
        (Dr → debit/dr; Cr → credit/cr).
      - Reject lines starting with WORKING keywords (less/more/add/total/
        proceeds/reserves/...) — these are working-area lines, not journal
        entries. Prevents a "Cr net assets 18,800" journal criterion
        anchoring on "less net assets -18,800,000" in disposal w3.
      - If NO rects survive the strict filter (all rejected), fall back to
        returning the original list so the mark doesn't disappear entirely
        — better to place with a warning than lose the annotation.

    For non-journal criteria: pass everything through unchanged.
    """
    if not resolved_pending:
        return []
    if not criterion_desc:
        return [(p, r) for p, r, _ev in resolved_pending]

    desc_l = criterion_desc.lower()
    is_journal = any(k in desc_l for k in (" dr ", "\tdr ", "dr:", " cr ", "cr:", "debit ", "credit "))
    # Also accept criteria starting with Dr/Cr as the first token.
    is_journal = is_journal or bool(re.match(r"^\s*(?:dr|cr|debit|credit)\b", desc_l))
    if not is_journal:
        return [(p, r) for p, r, _ev in resolved_pending]

    wants_debit = "dr " in desc_l or "debit " in desc_l or desc_l.startswith(("dr ", "debit "))
    wants_credit = "cr " in desc_l or "credit " in desc_l or desc_l.startswith(("cr ", "credit "))

    def _line_at(page, rect: "fitz.Rect") -> str:
        try:
            for lt, lr in _iter_page_lines(page):
                if lr.intersects(rect):
                    return lt
        except Exception:
            return ""
        return ""

    kept: list[tuple[int, "fitz.Rect"]] = []
    for page_num, rect, _ev in resolved_pending:
        page = doc[page_num - 1]
        line_text = _line_at(page, rect)
        if not line_text:
            # No line context recoverable → keep (defensive).
            kept.append((page_num, rect))
            continue
        # Reject working-line contexts for journal criteria.
        if _LINE_WORKING_RE.match(line_text):
            continue
        # If we want a specific direction, require the line to match it.
        if wants_debit and _LINE_DEBIT_RE.match(line_text):
            kept.append((page_num, rect))
            continue
        if wants_credit and _LINE_CREDIT_RE.match(line_text):
            kept.append((page_num, rect))
            continue
        # Line doesn't start with a direction verb OR a working keyword — allow
        # it through (may be a Dr/Cr line the regex missed, or an in-line ref).
        if not _LINE_DEBIT_RE.match(line_text) and not _LINE_CREDIT_RE.match(line_text):
            kept.append((page_num, rect))

    # Fallback: if the strict filter dropped everything, keep the originals
    # so the mark still lands somewhere rather than disappearing.
    if not kept and resolved_pending:
        logger.debug(
            "  [belonging] All rects rejected for criterion "
            f"'{criterion_desc[:50]}' — falling back to un-filtered list"
        )
        return [(p, r) for p, r, _ev in resolved_pending]

    return kept


_AGGREGATE_LINE_RE = re.compile(
    r"^\s*(?:total|gain\s+on\s+disposal|loss\s+on\s+disposal|net\s+total|"
    r"grand\s+total|sub[-\s]?total|balance|final|answer)\b",
    flags=re.IGNORECASE,
)


def _pick_best_rect_for_score(
    doc,
    page_rects: List[Tuple[int, "fitz.Rect"]],
    criterion_desc: str = "",
) -> Tuple[int, "fitz.Rect"]:
    """Choose the rect best suited to anchor the criterion's score label.

    Priority order (higher wins):
      A. direction_match — for journal criteria (description mentions "Dr <acct>
         <amt>" or "Cr <acct> <amt>"), lines that START with the SAME direction
         verb (debit/credit) win. Prevents e.g. #29 "Dr NCI 6,975" landing on
         "add back nci 6,975,000" in a working when the actual journal has
         "debit nci 6,975,000".
      B. aggregate_line_boost — for AGGREGATE criteria (description mentions
         "gain on disposal", "compute", "aggregate", "total") lines that START
         with "total"/"gain on disposal"/etc. win over lines that just happen
         to reference an INPUT number. Prevents e.g. #23 (gain on disposal
         computed) landing on "sales proceeds 200000*100" because 200000
         appears in the criterion — instead lands on the "total 10,450"
         line which is the actual gain figure.
      C. matching_num — line contains a distinctive number (≥4 digits, or a
         comma-grouped thousands value) that ALSO appears in the criterion
         description.
      D. earlier_position — the earlier the rect appears in the evidence
         list (as passed by the caller), the higher — the LLM tends to put
         the primary evidence FIRST. Small tie-breaker.
      E. tier — 3 (digit + alpha), 2 (digit only), 1 (alpha only), 0 (empty).
      F. line length — longer line = more context = clearer anchor.
    """
    # Extract distinctive numbers from criterion description.
    crit_nums: set[str] = set()
    if criterion_desc:
        # Distinctive numbers only: comma-grouped thousands OR bare ≥4-digit.
        # Skip small standalone integers (25, 35, 100) which appear too widely.
        for m in re.finditer(r"\d{1,3}(?:,\d{3})+", criterion_desc):
            crit_nums.add(m.group(0).replace(",", ""))
        for m in re.finditer(r"\b\d{4,}\b", criterion_desc):
            crit_nums.add(m.group(0))

    # Detect the criterion's journal direction (Dr → debit lines, Cr → credit).
    # Journal criteria in the rubric read like "…Dr NCI at disposal 6,975, Cr
    # Disposal of subsidiary 6,975"; we prefer the debit direction unless the
    # description leads with Cr.
    desc_l = criterion_desc.lower() if criterion_desc else ""
    crit_wants_debit = "dr " in desc_l or "debit " in desc_l
    crit_wants_credit = " cr " in desc_l or "credit " in desc_l
    # Journal detection: only true when the criterion is CLEARLY a journal
    # entry (starts with Dr/Cr or has "journal" in it). Avoids treating a
    # gain-on-disposal working criterion that merely quotes "Cr P&L" as an
    # example format as a full journal criterion.
    is_journal_crit = bool(re.match(r"^\s*(?:dr|cr|debit|credit)\b", desc_l)) or "journal" in desc_l

    # Detect aggregate/computation criteria — these prefer TOTAL lines over
    # INPUT lines. Keywords: "computed", "aggregate", "gain on disposal",
    # "final", "total".
    is_aggregate_crit = bool(re.search(
        r"\b(?:computed|aggregate|gain\s+on\s+disposal|loss\s+on\s+disposal|"
        r"final\s+figure|total\s+value|working\s+total)\b",
        desc_l,
    ))

    def _score(item: Tuple[int, "fitz.Rect"], index: int) -> Tuple[int, int, int, int, int, int]:
        page_num, rect = item
        page = doc[page_num - 1]
        line_text = ""
        try:
            for lt, lr in _iter_page_lines(page):
                if lr.intersects(rect):
                    line_text = lt
                    break
        except Exception:
            line_text = ""
        # Direction match: for journal criteria only, boost the line starting
        # with the SAME direction verb the criterion mentions.
        direction_match = 0
        if is_journal_crit and line_text:
            if crit_wants_debit and _LINE_DEBIT_RE.match(line_text):
                direction_match = 1
            if crit_wants_credit and _LINE_CREDIT_RE.match(line_text):
                direction_match = 1
        # Aggregate-line boost: for aggregate/computation criteria, prefer
        # lines starting with "total"/"gain on disposal"/etc.
        aggregate_boost = 0
        if is_aggregate_crit and line_text and _AGGREGATE_LINE_RE.match(line_text):
            aggregate_boost = 1
        # Normalise the line for number matching.
        line_digits = re.sub(r"[,\s]+", "", line_text)
        has_matching_num = 0
        if crit_nums:
            for n in crit_nums:
                if n and n in line_digits:
                    has_matching_num = 1
                    break
        # Earlier evidence position wins as small tie-breaker (LLM tends to
        # list the primary evidence first).
        earlier_bonus = max(0, 100 - index)
        has_digit = bool(re.search(r"\d", line_text))
        has_alpha = bool(re.search(r"[A-Za-z]", line_text))
        if has_digit and has_alpha:
            tier = 3
        elif has_digit:
            tier = 2
        elif has_alpha:
            tier = 1
        else:
            tier = 0
        return (direction_match, aggregate_boost, has_matching_num, earlier_bonus, tier, len(line_text))

    return max(
        enumerate(page_rects),
        key=lambda idx_item: _score(idx_item[1], idx_item[0]),
    )[1]


# ── Main annotation function ───────────────────────────────────────────────────

def annotate_pdf(
    input_pdf_path: str,
    output_dir: str,
    student_name: str,
    grades_id: Optional[str] = None,
    grades_doc: Optional[dict] = None,
    student_pages: Optional[List[int]] = None,
) -> Tuple[bool, str]:
    """Annotate a student PDF with scores, underlines, tick marks, and feedback.

    Returns (success, output_pdf_path).
    """
    try:
        student_key = student_name.lower().replace(" ", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_pdf = os.path.join(
            output_dir, student_key, f"{student_key}_annotated_{timestamp}.pdf"
        )
        mapping_json = os.path.join(
            output_dir, student_key, f"{student_key}_mapping_{timestamp}.json"
        )
        os.makedirs(os.path.dirname(output_pdf), exist_ok=True)

        # ── Load grading document ──────────────────────────────────────────────
        if grades_doc is None:
            if not grades_id:
                logger.error("No grades provided (need grades_id or grades_doc)")
                return False, ""
            grades_coll = get_collection("student_grades")
            grades_doc = grades_coll.find_one({"_id": ObjectId(grades_id)})
            if not grades_doc:
                logger.error(f"No grades for _id={grades_id}")
                return False, ""

        total_breakdown = len(grades_doc.get('breakdown', []) or [])
        displayed_count = len(
            [it for it in (grades_doc.get('breakdown', []) or [])
             if float(it.get('marks_awarded', 0) or 0) > 0]
        )
        logger.info(
            f"Annotating {student_name} Q{grades_doc.get('question_number', '?')} "
            f"({displayed_count} displayed / {total_breakdown} total criteria)"
        )

        doc = fitz.open(input_pdf_path)
        allowed_pages = student_pages or list(range(1, len(doc) + 1))

        # OCR must be initialised BEFORE boundary detection so _page_search
        # can read text from scanned pages when locating question headings.
        _init_ocr_cache(doc, allowed_pages)

        # ── Fetch student assignment doc ───────────────────────────────────────
        page_token_sets: dict[int, set[str]] = {}
        student_page_texts: dict[int, str] = {}
        student_question_heading: Optional[str] = None
        try:
            student_answer_id = grades_doc.get('student_answer_id')
            if student_answer_id:
                s_coll = get_collection("student_assignments")
                s_doc = s_coll.find_one({"_id": ObjectId(student_answer_id)})
                if s_doc:
                    student_question_heading = s_doc.get("question_heading_text") or None
                    if isinstance(s_doc.get("page_texts"), list):
                        for item in s_doc.get("page_texts") or []:
                            try:
                                pp = int(item.get("page"))
                                tt = str(item.get("text") or "")
                            except Exception:
                                continue
                            if pp in allowed_pages and tt:
                                student_page_texts[pp] = tt
        except Exception:
            student_page_texts = {}

        # ── Detect per-page Y boundaries for the target question ───────────────
        # Prevents annotations from bleeding into adjacent questions on shared pages.
        q_num = str(grades_doc.get('question_number', '')).strip()
        min_y_per_page: dict[int, float] = {}
        max_y_per_page: dict[int, float] = {}

        if q_num:
            q_heading_patterns: list[str] = []
            if student_question_heading:
                q_heading_patterns.append(student_question_heading)
            q_heading_patterns += [
                f"ANSWER {q_num}", f"Answer {q_num}",
                f"ANSWER: {q_num}", f"Answer: {q_num}",
                f"Q-{q_num.zfill(2)}", f"Q-{q_num}",
                f"Q.{q_num}", f"Q{q_num} ",
                f"Question {q_num}",
            ]

            for page_num in allowed_pages:
                page = doc[page_num - 1]

                # Find this question's heading → min_y
                for pattern in q_heading_patterns:
                    instances = _page_search(page, pattern)
                    for inst in (instances or []):
                        if inst.x0 < 250:
                            min_y_per_page[page_num] = inst.y0 - 5
                            logger.debug(
                                f"Q{q_num} heading found on page {page_num} at y={inst.y0:.0f}"
                            )
                            break
                    if page_num in min_y_per_page:
                        break

                # Find next question's heading → max_y
                # STRICT criteria to avoid false positives from inline text:
                #   • x0 < 100  — heading must be at the far left margin
                #   • y0 must be genuinely below the current question's start
                # We deliberately exclude the ambiguous "Question N" template
                # because students write it inline ("As in Question 3…").
                # We also only use the student_prefix template when the prefix
                # is specific enough (len ≥ 2) — a bare prefix like "" would
                # generate template "3" matching every digit on the page.
                current_min = min_y_per_page.get(page_num, 0)
                best_next_y = float('inf')

                try:
                    q_int = int(q_num)
                    neighbor_nums = [n for n in range(1, 11) if n != q_int]
                except ValueError:
                    neighbor_nums = []

                student_prefix: Optional[str] = None
                student_suffix: Optional[str] = None
                if student_question_heading and q_num in student_question_heading:
                    idx = student_question_heading.find(q_num)
                    student_prefix = student_question_heading[:idx]
                    student_suffix = student_question_heading[idx + len(q_num):]

                for n in neighbor_nums:
                    neighbour_templates = [
                        f"ANSWER {n}", f"Answer {n}",
                        f"ANSWER: {n}", f"Answer: {n}",
                        f"Q-{str(n).zfill(2)}", f"Q.{n}", f"Q{n} ",
                    ]
                    # Only include the student-format template when the prefix is
                    # non-trivially specific (prevents bare-digit templates like "3").
                    if student_prefix and len(student_prefix.strip()) >= 2:
                        neighbour_templates.insert(
                            0, f"{student_prefix}{n}{student_suffix}"
                        )
                    for tmpl in neighbour_templates:
                        hits = _page_search(page, tmpl)
                        for hit in (hits or []):
                            # x0 < 100: only accept hits at the very left margin
                            if hit.x0 < 100 and hit.y0 > current_min + 30:
                                if hit.y0 < best_next_y:
                                    best_next_y = hit.y0
                                    logger.debug(
                                        f"  Neighbor Q{n} '{tmpl}' found on page {page_num}"
                                        f" at y={hit.y0:.0f} x={hit.x0:.0f}"
                                    )

                if best_next_y < float('inf'):
                    max_y_per_page[page_num] = best_next_y - 5
                    logger.debug(
                        f"  Max Y for Q{q_num} on page {page_num}: {best_next_y:.0f}"
                    )

        # ── Build page token sets (for page ranking) ───────────────────────────
        for p in allowed_pages:
            page_text = student_page_texts.get(p, "")
            if not page_text:
                try:
                    page_text = _page_text(doc[p - 1])
                except Exception:
                    page_text = ""
            page_token_sets[p] = set(_tokenize(page_text or ""))

        # ── Shared annotation state ────────────────────────────────────────────
        comment_page_y: dict[int, float] = {}
        comment_used_y: dict[int, list[float]] = {}
        ocr_textpages: dict[int, object] = {}

        placed_lines_per_page = {i - 1: [] for i in allowed_pages}
        placed_marks: set = set()
        line_score_accumulator: dict = {}
        unplaced_items: list = []

        annotation_mapping = {
            'total_score_placed': False,
            'criterion_scores_placed': 0,
            'total_criteria': len(grades_doc.get('breakdown', [])),
            'total_breakdown': len(grades_doc.get('breakdown', []) or []),
            'comments_placed': 0,
            'unplaced_items': [],
            'allowed_pages': allowed_pages,
        }

        # ── Total score label ──────────────────────────────────────────────────
        main_score_text = (
            f"{_fmt_mark_value(grades_doc['total_marks_awarded'])}/"
            f"{_fmt_mark_value(grades_doc['total_max_possible'])}"
        )
        if add_main_score(
            doc, str(grades_doc.get('question_number', '')),
            main_score_text, allowed_pages,
            student_heading_text=student_question_heading,
        ):
            annotation_mapping['total_score_placed'] = True

        # ── Per-criterion annotation ───────────────────────────────────────────
        breakdown = grades_doc.get('breakdown', [])
        is_holistic = grades_doc.get('holistic_grading', False)
        displayed_breakdown = [
            it for it in breakdown
            if float(it.get('marks_awarded', 0) or 0) > 0
            or "Marks given above" in str(it.get('reason', '') or '')
            or "Marks given below" in str(it.get('reason', '') or '')
        ]
        annotation_mapping['total_criteria'] = len(displayed_breakdown)
        criteria_count = 0

        def _is_informative_anchor(text: str) -> bool:
            """Return True when *text* is distinctive enough to anchor to."""
            norm = _normalize_text_for_match(text)
            if len(norm) < 6:
                return False
            key = re.sub(r"[^a-z0-9]+", "", norm)
            if re.fullmatch(r"20x\d", key) or re.fullmatch(r"20\d{2}", key):
                return False
            if re.search(r"\d+\s*/\s*\d+", text):
                return True
            if re.search(r"\d", text) and len(norm) >= 4:
                return True
            return len(_tokenize(text)) >= 2

        # ══════════════════════════════════════════════════════════════════════
        # HOLISTIC GRADING MODE
        # Each breakdown item = one sub-question.
        # _correct_points_with_marks = [{text, key_phrase, marks}, ...]
        # Strategy:
        #   Step 1 — underline each correct point AND tick-mark it, co-located.
        #   Step 2 — place sub-question score near the student's sub-q label.
        # ══════════════════════════════════════════════════════════════════════
        if is_holistic:
            logger.info(
                f"Holistic annotation mode: {len(displayed_breakdown)} "
                f"sub-question(s) with marks"
            )

            for idx, item in enumerate(displayed_breakdown, 1):
                marks = float(item.get('marks_awarded', 0))
                max_marks = float(item.get('max_possible', 0) or 0)
                sub_q = item.get('_sub_question', '')
                student_label = item.get('_student_label', '')

                points_with_marks = item.get('_correct_points_with_marks', [])

                # Fallback: build from flat evidence with even mark distribution
                if not points_with_marks:
                    raw_evidence = item.get('evidence_list')
                    if isinstance(raw_evidence, list):
                        ev_texts = [
                            str(x).strip() for x in raw_evidence
                            if x is not None and str(x).strip()
                        ]
                    else:
                        evidence = (item.get('evidence', '') or '').strip()
                        ev_texts = [
                            p.strip() for p in re.split(r"\s*;\s*", evidence)
                            if p and p.strip()
                        ]
                    if ev_texts:
                        per_point = (
                            round(marks / len(ev_texts) / 0.5) * 0.5
                            if ev_texts else 0.5
                        )
                        per_point = max(per_point, 0.5)
                        points_with_marks = [
                            {"text": t, "marks": per_point} for t in ev_texts
                        ]

                # ── Step 1: Underline + tick each correct point ────────────────
                # Two-pass approach:
                #   Pass A — resolve every correct point to a PDF rect.
                #   Pass B — draw 1 underline + 1 tick per found evidence point.
                #            One correct_point = one model-answer match = one tick.
                logger.info(
                    f"    Processing {len(points_with_marks)} correct points "
                    f"for sub-question {sub_q}"
                )

                # Pass A: resolve all evidence positions.
                # Strategy: try key_phrase FIRST (it's the precise tick target),
                # only fall back to full text if key_phrase search fails.
                #
                # Duplicate handling: dedup is PER-PHRASE, not per-line.
                #   • "qualified opinion" appearing twice → walk to next occurrence.
                #   • "Cost overruns" + "result in delays" on the same line → both
                #     should tick (different phrases, separate dedup namespaces).
                # placed_per_phrase maps phrase → set of line_keys already ticked
                # for THAT phrase. placed_tick_keys is a global safety-net for
                # exact (page, x, y) collisions across all points.
                found_evidence: list[tuple] = []  # (page_obj, tick_rect, pt_marks)
                placed_per_phrase: dict[str, set] = {}  # phrase → line_keys used
                placed_tick_keys: set[tuple] = set()  # global exact-position dedup
                for pt in points_with_marks:
                    pt_text = pt.get("text", "").strip()
                    pt_marks = float(pt.get("marks", 0.5) or 0.5)
                    key_phrase = pt.get("key_phrase", "").strip()
                    if not pt_text:
                        continue
                    if not _is_informative_anchor(pt_text):
                        logger.info(f"    Skipped (not informative): '{pt_text[:60]}'")
                        continue

                    evidence_clean = _strip_llm_artifacts(
                        pt_text.replace("\u2026", " ").replace("...", " ").replace('|', ' ')
                    )
                    if not evidence_clean:
                        continue

                    tick_rect = None
                    found_page_num = -1
                    ev_page = None

                    # Strategy 1: Search for key_phrase directly in the PDF.
                    # Use per-phrase dedup so different phrases on the same line
                    # don't block each other, but a repeated phrase walks past
                    # already-ticked occurrences.
                    if key_phrase and len(key_phrase) >= 3:
                        phrase_key = key_phrase.lower().strip()
                        phrase_marks = placed_per_phrase.setdefault(phrase_key, set())
                        for candidate in _build_anchor_variations(key_phrase):
                            tick_rect, found_page_num = resolve_anchor_rect(
                                doc, candidate, allowed_pages,
                                placed_marks=phrase_marks,
                                skip_duplicates=True,
                                expand_to_line=False,
                                page_token_sets=page_token_sets,
                                redirect_headings=False,
                                use_number_first=False,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                            )
                            if tick_rect and found_page_num > 0:
                                logger.info(f"      Tick via key_phrase: '{key_phrase}'")
                                break

                    # Strategy 2: Fall back to full text search if key_phrase failed.
                    # Per-phrase dedup keyed on the evidence text.
                    if not tick_rect or found_page_num <= 0:
                        text_key = pt_text.lower().strip()
                        text_marks = placed_per_phrase.setdefault(text_key, set())
                        candidate_fragments = _build_candidate_fragments(evidence_clean)
                        for fragment in candidate_fragments:
                            for candidate in _build_anchor_variations(fragment):
                                tick_rect, found_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=text_marks,
                                    skip_duplicates=True,
                                    expand_to_line=False,
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if tick_rect and found_page_num > 0:
                                    break
                            if tick_rect and found_page_num > 0:
                                break

                    if tick_rect and found_page_num > 0:
                        # Refine tick_rect to the exact key_phrase position.
                        # When the full-text fallback returned a line-level rect,
                        # this narrows it to the specific key_phrase within that line
                        # so the tick + underline land on the right words.
                        if key_phrase and len(key_phrase) >= 3:
                            _refine_page = doc[found_page_num - 1]
                            _clip = fitz.Rect(
                                0, max(tick_rect.y0 - 3, 0),
                                _refine_page.rect.width,
                                min(tick_rect.y1 + 3, _refine_page.rect.height),
                            )
                            kp_words = key_phrase.split()
                            for _end in range(len(kp_words), max(1, len(kp_words) - 2) - 1, -1):
                                _sub = " ".join(kp_words[:_end])
                                if len(_sub) < 3:
                                    continue
                                _kp_hits = _page_search(_refine_page, _sub, clip=_clip)
                                if _kp_hits:
                                    tick_rect = _kp_hits[0]
                                    break

                        # Safety-net dedup for exact (page, x, y) collisions across
                        # all phrases. resolve_anchor_rect already walks past
                        # already-ticked occurrences of the same phrase via
                        # placed_per_phrase; this guards against pathological
                        # cross-phrase exact-rect collisions.
                        tick_key = (found_page_num, round(tick_rect.y0, 0), round(tick_rect.x0, 0))
                        if tick_key in placed_tick_keys:
                            logger.info(f"    ⊘ Duplicate tick position, skipping: '{key_phrase or pt_text[:40]}'")
                            continue
                        placed_tick_keys.add(tick_key)

                        # Record this line under the phrase that matched, so a
                        # repeat of the SAME phrase walks to the next occurrence.
                        line_mark = _line_key(found_page_num, tick_rect.y0)
                        if key_phrase and len(key_phrase) >= 3:
                            placed_per_phrase.setdefault(
                                key_phrase.lower().strip(), set()
                            ).add(line_mark)
                        else:
                            placed_per_phrase.setdefault(
                                pt_text.lower().strip(), set()
                            ).add(line_mark)

                        ev_page = doc[found_page_num - 1]
                        found_evidence.append((ev_page, tick_rect, pt_marks))
                        logger.info(f"    ✓ Found: '{key_phrase or pt_text[:60]}'")
                    else:
                        logger.info(f"    ✗ Not found in PDF: '{key_phrase or pt_text[:60]}'")

                # Pass B: draw underlines + place exactly 1 tick per found evidence point.
                # One correct_point = one matched model-answer point = one tick mark.
                # Track how many ticks have already been placed at each rect position
                # so that same-line fallback ticks are offset rather than stacked.
                if found_evidence:
                    tick_offset_per_rect: dict[tuple, int] = {}  # rect_key → ticks_placed
                    for ev_idx, (ev_page, ev_rect, ev_marks) in enumerate(found_evidence):
                        _draw_underline_for_rect(ev_page, ev_rect, phrase_only=True)
                        rect_key = (round(ev_rect.x0), round(ev_rect.y0))
                        offset = tick_offset_per_rect.get(rect_key, 0)
                        # Shift the rect right by (tick_width+gap) × offset so stacked
                        # ticks from the grade.py fallback split appear side by side.
                        tick_w = 10  # approx tick_width + gap from _place_ticks
                        shifted_rect = fitz.Rect(
                            ev_rect.x0 + offset * tick_w,
                            ev_rect.y0,
                            ev_rect.x1 + offset * tick_w,
                            ev_rect.y1,
                        )
                        _place_ticks(ev_page, shifted_rect, 1)
                        tick_offset_per_rect[rect_key] = offset + 1
                        logger.info(
                            f"    ✓ Underlined + 1 tick "
                            f"(evidence {ev_idx + 1}/{len(found_evidence)})"
                        )

                    logger.info(
                        f"    Sub-Q {sub_q}: {len(found_evidence)} tick(s) placed "
                        f"({len(found_evidence)}/{len(points_with_marks)} "
                        f"evidence points found)"
                    )

                # ── Step 1b: Mark off-topic content as "Not required" ──────────
                # No marks affected — purely instructional feedback for the student.
                # Uses the same anchor-resolution flow but with its own per-sub-q
                # line tracker so a "Not required" marker doesn't collide with ticks.
                not_required_points = item.get('_not_required_points', []) or []
                if not_required_points:
                    logger.info(
                        f"    Processing {len(not_required_points)} 'Not required' "
                        f"point(s) for sub-question {sub_q}"
                    )
                    nr_local_marks: set = set()
                    for nr in not_required_points:
                        nr_text = str(nr.get("text", "")).strip()
                        nr_kp = str(nr.get("key_phrase", "")).strip()
                        nr_reason = str(nr.get("reason", "")).strip()
                        if not nr_text:
                            continue

                        nr_clean = _strip_llm_artifacts(
                            nr_text.replace("…", " ").replace("...", " ").replace('|', ' ')
                        )
                        if not nr_clean:
                            continue

                        nr_rect = None
                        nr_page_num = -1

                        # Strategy 1: key_phrase match (precise).
                        if nr_kp and len(nr_kp) >= 3:
                            for candidate in _build_anchor_variations(nr_kp):
                                nr_rect, nr_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=nr_local_marks,
                                    skip_duplicates=True,
                                    expand_to_line=True,  # full line for strikethrough
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if nr_rect and nr_page_num > 0:
                                    break

                        # Strategy 2: full text fallback.
                        if not nr_rect or nr_page_num <= 0:
                            for fragment in _build_candidate_fragments(nr_clean):
                                for candidate in _build_anchor_variations(fragment):
                                    nr_rect, nr_page_num = resolve_anchor_rect(
                                        doc, candidate, allowed_pages,
                                        placed_marks=nr_local_marks,
                                        skip_duplicates=True,
                                        expand_to_line=True,
                                        page_token_sets=page_token_sets,
                                        redirect_headings=False,
                                        use_number_first=False,
                                        min_y_per_page=min_y_per_page,
                                        max_y_per_page=max_y_per_page,
                                    )
                                    if nr_rect and nr_page_num > 0:
                                        break
                                if nr_rect and nr_page_num > 0:
                                    break

                        if nr_rect and nr_page_num > 0:
                            nr_page = doc[nr_page_num - 1]
                            place_not_required_marker(nr_page, nr_rect, nr_reason)
                            nr_local_marks.add(_line_key(nr_page_num, nr_rect.y0))
                            logger.info(f"    ⚑ Not required: '{nr_kp or nr_text[:60]}'")
                        else:
                            logger.info(f"    ✗ NR not found in PDF: '{nr_kp or nr_text[:60]}'")

                # ── Step 2: Place sub-question score near student's label ───────
                # FIX-1 (via draw_underline=False): heading/label lines are never
                # underlined — only the score mark is placed beside them.
                # ORDER: prefer the sub-question NUMBER lookup ("4.1", "4.2", …)
                # over the student_label, because student_label is often a
                # generic single character ("A", "B", "a)") that matches dozens
                # of false positions in the PDF — including the page header.
                score_text = (
                    f"{_fmt_mark_value(marks)}/{_fmt_mark_value(max_marks)}"
                )
                score_placed = False

                if sub_q:
                    for pattern in [
                        sub_q, f"({sub_q})", f"{sub_q})",
                        f"{sub_q}.", f"{sub_q}:",
                    ]:
                        score_placed = place_score_near_anchor(
                            doc, pattern, score_text,
                            allowed_pages, placed_lines_per_page, placed_marks,
                            unplaced_items,
                            page_token_sets=page_token_sets,
                            line_score_accumulator=None,
                            min_y_per_page=min_y_per_page,
                            max_y_per_page=max_y_per_page,
                            draw_underline=False,  # FIX-1
                        )
                        if score_placed:
                            break

                # Fallback to student_label ONLY when it is specific enough.
                # A label without a digit is too ambiguous: "A", "a", "(a)",
                # "(b)", "(i)", "(ii)" all match dozens of arbitrary positions
                # in the PDF (every parenthesised letter, every standalone
                # capital, etc.) and put the score in the wrong region.
                # Requiring a digit keeps "4.1", "1.1", "a.1" but rejects all
                # the letter-only labels that have caused mis-placement.
                def _label_is_specific(label: str) -> bool:
                    s = (label or "").strip()
                    if len(s) < 3:
                        return False
                    return bool(re.search(r"\d", s))

                if not score_placed and _label_is_specific(student_label):
                    score_placed = place_score_near_anchor(
                        doc, student_label.strip(), score_text,
                        allowed_pages, placed_lines_per_page, placed_marks,
                        unplaced_items,
                        page_token_sets=page_token_sets,
                        line_score_accumulator=None,
                        min_y_per_page=min_y_per_page,
                        max_y_per_page=max_y_per_page,
                        draw_underline=False,  # FIX-1: do not underline the sub-q heading
                    )

                # Fallback: anchor to first evidence fragment
                if not score_placed and points_with_marks:
                    for pt in points_with_marks[:2]:
                        pt_text = pt.get("text", "").strip()
                        if pt_text and _is_informative_anchor(pt_text):
                            score_placed = place_score_near_anchor(
                                doc, pt_text, score_text,
                                allowed_pages, placed_lines_per_page, placed_marks,
                                unplaced_items,
                                page_token_sets=page_token_sets,
                                line_score_accumulator=None,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                                draw_underline=False,  # FIX-1: evidence already underlined above
                            )
                            if score_placed:
                                break

                if score_placed:
                    annotation_mapping['criterion_scores_placed'] += 1
                    logger.info(f"  Sub-question {sub_q}: {score_text} placed")
                else:
                    logger.debug(f"  Sub-question {sub_q}: {score_text} NOT placed")
                criteria_count += 1

            logger.info(
                f"✓ Placed {annotation_mapping['criterion_scores_placed']} of "
                f"{criteria_count} sub-question scores"
            )

        # ══════════════════════════════════════════════════════════════════════
        # STANDARD PER-CRITERION ANNOTATION MODE
        # ══════════════════════════════════════════════════════════════════════
        else:
            # Numerical mode rules:
            #   • Underline EVERY entry in evidence_list that resolves to a rect.
            #   • Place exactly ONE score label per criterion, on the criterion's
            #     "best" evidence rect. _pick_best_rect_for_score prefers rects
            #     whose line contains a distinctive number that also appears in
            #     the criterion description (e.g., a "Dr NCI 6,975" criterion
            #     lands on the 6,975 line, not on the "credit profit on
            #     disposal 10,450" line).
            #   • Never fall back to the criterion text as an anchor — the
            #     criterion text often shares words with the student's section
            #     headings and was causing scores to land on headings.
            # NOTE: the grader (grade.py:_apply_line_evidence_dedup) already
            # deduplicates evidence-sharing across criteria and revokes marks
            # from criteria that lose all their evidence, so displayed items
            # here should not collide on the same line.
            for idx, item in enumerate(displayed_breakdown, 1):
                marks = float(item.get('marks_awarded', 0))
                raw_evidence = item.get('evidence_list')
                if isinstance(raw_evidence, list):
                    evidence_lines = [
                        str(x).strip() for x in raw_evidence
                        if x is not None and str(x).strip()
                    ]
                else:
                    evidence = (item.get('evidence', '') or '').strip()
                    evidence_lines = [
                        p.strip() for p in re.split(r"\s*;\s*", evidence)
                        if p and p.strip()
                    ]
                criterion_name = item.get('criterion', '').strip()

                _item_reason = str(item.get('reason', '') or '')
                # OF-marker detection: match "OF" as a standalone word anywhere
                # in the reason (case-sensitive to avoid "of the year" false
                # positives). Previously required "OF" at the START of the
                # reason, but the LLM's newer outputs put OF mid-sentence
                # (e.g., "Cr Net assets (OF 18,800) and Cr Goodwill 11,725
                # both present.").
                is_of = item.get('is_of_mark', False) or bool(
                    re.search(r'\bOF\b', _item_reason)
                )
                if marks == 0 and "Marks given above" in _item_reason:
                    score_label = "Marks given above"
                elif marks == 0 and "Marks given below" in _item_reason:
                    score_label = "Marks given below"
                elif is_of:
                    score_label = f"OF {_fmt_mark_value(marks)}"
                else:
                    score_label = _fmt_mark_value(marks)

                logger.debug(f"  Criterion {idx}: {criterion_name} ({marks}pts)")

                # An evidence string containing a newline describes ONE phrase
                # the student wrote across two physical lines. Search the
                # JOINED form first — Tier 1's wrap grouper resolves it to a
                # single multi-line occurrence, which guarantees the underlines
                # are adjacent parts of the same sentence. Only if the joined
                # form resolves nowhere do we fall back to searching each line
                # independently (the previous behaviour), which can otherwise
                # scatter the halves onto unrelated occurrences.
                # Each entry is [joined_form, *per_line_fallbacks].
                evidence_candidates: list[list[str]] = []
                for ev in evidence_lines:
                    parts = [p.strip() for p in re.split(r"\n+", ev) if p.strip()]
                    if not parts:
                        continue
                    joined = re.sub(r"\s+", " ", " ".join(parts)).strip()
                    evidence_candidates.append(
                        [joined] + parts if len(parts) > 1 else [joined]
                    )

                # Resolve each evidence part via strict exact-substring search.
                # Headings and out-of-bounds rects are rejected.
                # Two-pass: first RESOLVE all rects, then FILTER to only those
                # that semantically BELONG to this criterion (direction match,
                # value match, section context) before drawing anything. This
                # prevents the annotator from placing a `Cr net assets` mark
                # on a `less net assets` working-line just because both share
                # the same amount.
                resolved_pending: list[tuple[int, fitz.Rect, str]] = []
                # Rects whose matched span must be kept WHOLE — never narrowed
                # to a target value, never dropped by the narrowed-page rule.
                # Two sources:
                #   • fragments of ONE wrapped phrase (narrowing one fragment
                #     would let the drop rule delete its siblings, leaving a
                #     wrapped sentence underlined only under a lone number);
                #   • prose evidence (a sentence the student wrote — narrowing
                #     "…e.g. (9/12) will be consolidated…" to its `9` glyph
                #     leaves a one-character underline under a full line).
                keep_whole_keys: set[tuple[int, int, int]] = set()

                def _rect_key(page_num: int, rect: "fitz.Rect") -> tuple[int, int, int]:
                    return (
                        page_num,
                        int(round(rect.y0 * 10)),
                        int(round(rect.x0 * 10)),
                    )

                def _resolve(ev: str) -> bool:
                    """Resolve one evidence string; record rects. True if found."""
                    if not _is_informative_anchor(ev):
                        logger.debug(f"    Evidence skipped (not informative): {ev[:60]}")
                        return False
                    rects, page_num, is_wrapped = _find_evidence_strict(
                        doc, ev, allowed_pages,
                        placed_marks=placed_marks,
                        page_token_sets=page_token_sets,
                        min_y_per_page=min_y_per_page,
                        max_y_per_page=max_y_per_page,
                    )
                    if not (rects and page_num > 0):
                        return False
                    _is_prose = _is_prose_evidence(ev)
                    for rect in rects:
                        resolved_pending.append((page_num, rect, ev))
                        if is_wrapped or _is_prose:
                            keep_whole_keys.add(_rect_key(page_num, rect))
                    if is_wrapped:
                        logger.debug(
                            f"    ↩ Evidence wraps {len(rects)} physical lines: "
                            f"'{ev[:60]}'"
                        )
                    if _is_prose:
                        logger.debug(
                            f"    ¶ Prose evidence — narrowing skipped: '{ev[:60]}'"
                        )
                    return True

                for candidates in evidence_candidates:
                    if _resolve(candidates[0]):
                        continue
                    # Joined form found nothing — fall back to per-line search.
                    found_any = False
                    for part in candidates[1:]:
                        found_any = _resolve(part) or found_any
                    if not found_any:
                        logger.debug(
                            f"    ✗ Evidence not found: '{candidates[0][:60]}'"
                        )

                # BELONGING CHECK: filter resolved rects to those that actually
                # match the criterion's context. Currently checks journal
                # direction (Dr/Cr) — a Cr-criterion should NOT anchor on a
                # line that starts with "less"/"more"/"add" (a working line);
                # only on lines that start with "cr"/"credit" (a journal line).
                # For non-journal criteria this is a no-op.
                filtered: list[tuple[int, fitz.Rect]] = _filter_rects_by_criterion_context(
                    doc, resolved_pending, criterion_name
                )

                # TARGET-VALUE NARROWING (model-side driven, no unit guessing):
                # For each resolved rect, shrink it to just the cell that
                # contains this criterion's target value. The target and its
                # surface-form variants are ALWAYS provided by the model
                # side (grade.py) via `_target_value` and
                # `_target_value_variants` — the annotator never assumes a
                # unit (thousands vs raw). Sources on the model side:
                #   • Rubric `of_value.value` (origin criteria)
                #   • Rubric `of_produces` (aggregate-component criteria)
                #   • sum(subset) (aggregate recovery / merged entries)
                # If neither field is populated the annotator falls back to
                # the wide rect (previous behaviour) — no narrowing lost,
                # no wrong-place hallucination.
                target_variants: list[str] = []
                _stashed_variants = item.get("_target_value_variants")
                if isinstance(_stashed_variants, list):
                    for v in _stashed_variants:
                        s = str(v).strip()
                        if s and s not in target_variants:
                            target_variants.append(s)

                # Model-provided column-header hint for tabular disambiguation
                # (e.g., "acq date" vs "disposal date" when the same value
                # appears in multiple columns of the row). None → annotator
                # falls back to first-hit within the matched rect.
                _col_hint_raw = item.get("_column_header")
                _column_header_hint: Optional[str] = None
                if _col_hint_raw is not None:
                    _stripped = str(_col_hint_raw).strip()
                    _column_header_hint = _stripped or None

                # Narrow each rect; track which ones actually got narrowed
                # so we can drop label-only rects when a value rect exists.
                # This prevents `_pick_best_rect_for_score` from landing the
                # score on a table row's LABEL cell when the same row also
                # contains the target VALUE cell — teacher marks the value,
                # not the label.
                narrowed_rects: list[tuple[int, fitz.Rect]] = []
                unnarrowed_rects: list[tuple[int, fitz.Rect]] = []
                for page_num, rect in filtered:
                    page = doc[page_num - 1]
                    narrowed = None
                    # Prose and wrapped fragments are never narrowed. Narrowing
                    # matches on the row's Y-BAND ONLY (not the rect's
                    # x-range — see _narrow_rect_to_target_variants), so a
                    # sentence collapses onto whatever number happens to share
                    # its visual row. The underline must span the phrase the
                    # student actually wrote; _pick_best_rect_for_score still
                    # positions the score label.
                    if target_variants and _rect_key(page_num, rect) not in keep_whole_keys:
                        narrowed = _narrow_rect_to_target_variants(
                            page, rect, target_variants,
                            column_header=_column_header_hint,
                            doc=doc, allowed_pages=allowed_pages,
                        )
                    if narrowed is not None:
                        narrowed_rects.append((page_num, narrowed))
                    else:
                        unnarrowed_rects.append((page_num, rect))

                # Drop label / non-value rects when at least one rect on the
                # same page got narrowed to a target value — the value rects
                # are the canonical anchors. Rects on OTHER pages (no
                # narrowed match there) are still kept so cross-page
                # evidence still gets some underline coverage.
                # Keep-whole rects are exempt from the drop: they are prose, or
                # continuation lines of the SAME sentence — not competing
                # label/value candidates — so a narrowed rect elsewhere on the
                # page must not silently remove them.
                narrowed_pages = {p for p, _r in narrowed_rects}
                resolved: list[tuple[int, fitz.Rect]] = list(narrowed_rects) + [
                    (p, r) for p, r in unnarrowed_rects
                    if p not in narrowed_pages or _rect_key(p, r) in keep_whole_keys
                ]
                # Defensive: if narrowing dropped everything for some reason,
                # fall back to the full filtered list to keep the mark visible.
                if not resolved:
                    resolved = list(filtered)

                # Draw underlines only for rects that survived the belonging
                # check. phrase_only=True: underline only the matched phrase
                # (not the whole row) — teacher-style narrow underline.
                for page_num, rect in resolved:
                    _draw_underline_for_rect(doc[page_num - 1], rect, phrase_only=True)
                    placed_marks.add(_line_key(page_num, rect.y0))

                if resolved:
                    logger.debug(
                        f"    ✓ Underlined {len(resolved)} line"
                        f"{'s' if len(resolved) != 1 else ''} for criterion "
                        f"(rejected {len(resolved_pending) - len(resolved)} "
                        f"context-mismatch rect"
                        f"{'s' if len(resolved_pending) - len(resolved) != 1 else ''})"
                    )

                if resolved:
                    best_page, best_rect = _pick_best_rect_for_score(
                        doc, resolved, criterion_name,
                    )
                    _place_score_label(
                        doc[best_page - 1], best_rect, best_page - 1,
                        placed_lines_per_page, score_label,
                    )
                    annotation_mapping['criterion_scores_placed'] += 1
                    logger.debug(
                        f"    ✓ Score '{score_label}' placed on page {best_page} "
                        f"(underlined {len(resolved)} evidence line"
                        f"{'s' if len(resolved) != 1 else ''})"
                    )
                else:
                    unplaced_items.append((
                        score_label,
                        (evidence_candidates[0][0] if evidence_candidates else '')[:50],
                    ))
                    logger.debug(
                        f"    ✗ Criterion unplaced — no evidence line matched"
                    )
                criteria_count += 1

            logger.info(
                f"✓ Placed {annotation_mapping['criterion_scores_placed']} of "
                f"{criteria_count} criteria with evidence"
            )

            # ── Standard mode: mark off-topic content as "Not required" ──────
            # No marks affected — purely instructional feedback for the student.
            # Mirrors the holistic-mode Step 1b but reads from the doc-level
            # not_required_points list (numerical questions don't have sub-Qs).
            nr_points = grades_doc.get('not_required_points', []) or []
            if nr_points:
                logger.info(
                    f"Processing {len(nr_points)} 'Not required' point(s) for numerical grading"
                )
                nr_local_marks: set = set()
                for nr in nr_points:
                    if not isinstance(nr, dict):
                        continue
                    nr_text = str(nr.get("text", "")).strip()
                    nr_kp = str(nr.get("key_phrase", "")).strip()
                    nr_reason = str(nr.get("reason", "")).strip()
                    if not nr_text:
                        continue

                    nr_clean = _strip_llm_artifacts(
                        nr_text.replace("…", " ").replace("...", " ").replace('|', ' ')
                    )
                    if not nr_clean:
                        continue

                    nr_rect = None
                    nr_page_num = -1

                    # Strategy 1: key_phrase match (precise).
                    if nr_kp and len(nr_kp) >= 3:
                        for candidate in _build_anchor_variations(nr_kp):
                            nr_rect, nr_page_num = resolve_anchor_rect(
                                doc, candidate, allowed_pages,
                                placed_marks=nr_local_marks,
                                skip_duplicates=True,
                                expand_to_line=True,
                                page_token_sets=page_token_sets,
                                redirect_headings=False,
                                use_number_first=False,
                                min_y_per_page=min_y_per_page,
                                max_y_per_page=max_y_per_page,
                            )
                            if nr_rect and nr_page_num > 0:
                                break

                    # Strategy 2: full text fallback.
                    if not nr_rect or nr_page_num <= 0:
                        for fragment in _build_candidate_fragments(nr_clean):
                            for candidate in _build_anchor_variations(fragment):
                                nr_rect, nr_page_num = resolve_anchor_rect(
                                    doc, candidate, allowed_pages,
                                    placed_marks=nr_local_marks,
                                    skip_duplicates=True,
                                    expand_to_line=True,
                                    page_token_sets=page_token_sets,
                                    redirect_headings=False,
                                    use_number_first=False,
                                    min_y_per_page=min_y_per_page,
                                    max_y_per_page=max_y_per_page,
                                )
                                if nr_rect and nr_page_num > 0:
                                    break
                            if nr_rect and nr_page_num > 0:
                                break

                    if nr_rect and nr_page_num > 0:
                        nr_page = doc[nr_page_num - 1]
                        place_not_required_marker(nr_page, nr_rect, nr_reason)
                        nr_local_marks.add(_line_key(nr_page_num, nr_rect.y0))
                        logger.info(f"  ⚑ Not required: '{nr_kp or nr_text[:60]}'")
                    else:
                        logger.info(f"  ✗ NR not found in PDF: '{nr_kp or nr_text[:60]}'")

        # ── Unplaced items: log only, no margin notes ────────────────────────
        # Placing "Marks given below: X.XXpt (…)" in the margin was confusing
        # because it pollutes the question/answer area with content that has
        # no spatial relationship to where the actual evidence was supposed
        # to be marked. We now only surface unplaced items in the mapping JSON.
        if unplaced_items:
            logger.warning(
                f"{len(unplaced_items)} item(s) unplaced — see mapping JSON for details"
            )

        # ── Feedback comments ──────────────────────────────────────────────────
        all_comments = grades_doc.get('comments', [])
        logger.info(f"Processing {len(all_comments)} comments for feedback...")

        # Pre-compute sub-question Y bounds for every [<sub_question>] tag found
        # in the comments. The grading prompt prepends a tag like "[4.1]" or
        # "[4.3 Payroll Threats]" so the popup lands inside the correct sub-region.
        sub_ids_in_comments: list[str] = []
        seen_ids: set[str] = set()
        for c in all_comments:
            if not isinstance(c, str):
                continue
            sid, _ = _strip_subq_prefix(c)
            if sid and sid not in seen_ids:
                seen_ids.add(sid)
                sub_ids_in_comments.append(sid)
        subq_y_bounds = compute_subq_y_bounds(doc, allowed_pages, sub_ids_in_comments)
        if sub_ids_in_comments:
            pages_with_bounds = sum(1 for sid in sub_ids_in_comments if subq_y_bounds.get(sid))
            if pages_with_bounds:
                logger.info(
                    f"Sub-question Y bounds resolved: {pages_with_bounds}/"
                    f"{len(sub_ids_in_comments)} sub_ids located — comments "
                    f"constrained to their sub-question region"
                )
            else:
                logger.info(
                    f"Sub-question Y bounds: none of {len(sub_ids_in_comments)} "
                    f"sub_id(s) bounded — comments may anchor anywhere in the "
                    f"question (expected when every comment shares one "
                    f"sub-question, or no sub-question labels appear in the PDF)"
                )

        comments_placed = 0
        unplaced_comments: list[str] = []
        for idx, comment in enumerate(all_comments, 1):
            if not comment or not isinstance(comment, str):
                logger.debug(f"  Comment {idx}: INVALID (empty or non-string)")
                continue

            logger.debug(f"  Comment {idx}/{len(all_comments)}: {comment[:70]}...")

            try:
                if add_popup_for_comment(
                    doc,
                    comment.strip(),
                    allowed_pages,
                    placed_marks=placed_marks,
                    page_token_sets=page_token_sets,
                    comment_page_y=comment_page_y,
                    comment_used_y=comment_used_y,
                    ocr_textpages=ocr_textpages,
                    placed_lines_per_page=placed_lines_per_page,
                    min_y_per_page=min_y_per_page,
                    max_y_per_page=max_y_per_page,
                    subq_y_bounds=subq_y_bounds,
                ):
                    comments_placed += 1
                    logger.debug("    ✓ Comment placed")
                else:
                    unplaced_comments.append(comment.strip())
                    logger.warning(
                        f"  ✗ Comment {idx} NOT PLACED on PDF "
                        f"(anchor split or match failed): {comment[:80]!r}"
                    )
            except Exception as e:
                unplaced_comments.append(comment.strip())
                logger.warning(f"  ✗ Comment {idx} error during placement: {e}")

        annotation_mapping['comments_placed'] = comments_placed
        annotation_mapping['unplaced_comments'] = unplaced_comments
        if unplaced_comments:
            logger.warning(
                f"✗ {len(unplaced_comments)} comment(s) failed to place on PDF "
                f"(see 'unplaced_comments' in annotation mapping JSON)"
            )
        logger.info(f"✓ Comments: {comments_placed}/{len(all_comments)} placed")

        # ── Save ───────────────────────────────────────────────────────────────
        doc.save(output_pdf, garbage=4, deflate=True, clean=True)
        doc.close()

        annotation_mapping['unplaced_items'] = unplaced_items[:10]
        with open(mapping_json, 'w') as f:
            json.dump(annotation_mapping, f, indent=2, default=str)

        success_rate = (
            annotation_mapping['criterion_scores_placed']
            / max(annotation_mapping['total_criteria'], 1)
        )
        logger.info(f"Done: {success_rate:.0%} success → {output_pdf}")
        return True, output_pdf

    except Exception as e:
        logger.error(f"Annotation failed for {student_name}: {e}", exc_info=True)
        return False, ""
