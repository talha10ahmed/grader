# ==================== DRAWING & PLACEMENT FUNCTIONS ====================
#
# Provides:
#   _safe_float, _detect_fontsize_at_rect, _fmt_mark_value
#   _place_score_label          – place one criterion score near matched rect
#   place_score_near_anchor     – resolve anchor + optionally underline + place score
#   _place_ticks                – draw tick marks above a rect
#   add_main_score              – place total score near the question heading
#   add_popup_for_comment       – place feedback popup anchored to evidence
#
# Bug fixes included:
#   FIX-1: place_score_near_anchor gains draw_underline param (False for headings)
#   FIX-2: comment popup clamped to q_max_y so it cannot bleed into next question
#   FIX-4: add_main_score's _is_question_heading accepts headings near page bottom
#           (whose answer content continues on the next page)

import re
from typing import Iterable, Optional

import fitz

from logging_config import logger
from .annotator_config import CONFIG
from .annotator_ocr import _page_words, _page_dict, _page_search
from .annotator_text import (
    _strip_llm_artifacts, _normalize_text_for_match, _tokenize,
    _split_comment_arrow, _build_anchor_variations, _line_key,
    _build_candidate_fragments,
)
from .annotator_rect import (
    _draw_underline_for_rect, _iter_page_lines, _box_overlaps_page_text,
)
from .annotator_match import (
    resolve_anchor_rect, _rank_pages_for_anchor,
    clean_anchor_text, extract_number_from_text, _find_best_line_match,
)


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe_float(value) -> float:
    """Parse a float from a value that may be a fraction string like '1.5/3'."""
    try:
        return float(value)
    except (ValueError, TypeError):
        s = str(value).strip()
        if '/' in s:
            try:
                num, den = s.split('/', 1)
                return float(num.strip()) / float(den.strip())
            except Exception:
                pass
    return 0.0


def _detect_fontsize_at_rect(page, rect: fitz.Rect, default: float = 11.0) -> float:
    """Return the font size of the span that best overlaps *rect*.

    Falls back to *default* when nothing matches.  Used to scale score labels
    proportionally to the student's own font choice.
    """
    if not rect:
        return default
    try:
        data = _page_dict(page)
        best_size = default
        best_overlap = 0.0
        for block in data.get("blocks", []) or []:
            for line in block.get("lines", []) or []:
                for span in line.get("spans", []) or []:
                    bbox = span.get("bbox")
                    size = span.get("size") or 0
                    if not bbox or size <= 0:
                        continue
                    sr = fitz.Rect(bbox)
                    if not sr.intersects(rect):
                        continue
                    overlap = (sr & rect).get_area()
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_size = float(size)
        return best_size if best_overlap > 0 else default
    except Exception:
        return default


def _fmt_mark_value(value: float) -> str:
    """Format a mark value, keeping 0.25-step fractions readable."""
    try:
        v = float(value)
    except Exception:
        return str(value)
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return s


# ── Score label placement ──────────────────────────────────────────────────────

def _find_value_x_on_line(page, rect: fitz.Rect) -> Optional[float]:
    """Return the right-edge x-position of the rightmost numeric word on the
    same line as *rect*. Returns None when the line has no numeric token —
    i.e. it is a purely textual sentence.

    "Numeric" here means the word contains at least one digit. Used to
    anchor score labels over the value column on accounting/calc lines
    rather than over the descriptive text on the left.
    """
    try:
        line_y = (rect.y0 + rect.y1) / 2
        best_x: Optional[float] = None
        for w in _page_words(page):
            wx0, wy0, wx1, wy1, text, *_ = w
            if abs((wy0 + wy1) / 2 - line_y) > 4:
                continue
            t = (text or "").strip()
            if t and any(c.isdigit() for c in t):
                if best_x is None or wx1 > best_x:
                    best_x = wx1
        return best_x
    except Exception:
        return None


def _rect_contains_number(page, rect: fitz.Rect) -> bool:
    """True when the anchor rect itself covers text that contains a digit.

    Used by _place_score_label to decide whether to keep the score label
    adjacent to the anchor (rect already IS a value) vs push it to the
    rightmost value on the line (rect is a label pointing at a value).

    Prevents "floating scores past all columns" on table rows where the
    anchor is one specific numeric cell — the score belongs next to THAT
    cell, not at the end of the whole row.
    """
    try:
        # Widen the intersect window slightly to account for glyph vs bbox
        # rounding — otherwise very tight rects sometimes miss all words.
        probe = fitz.Rect(rect.x0 - 0.5, rect.y0 - 0.5, rect.x1 + 0.5, rect.y1 + 0.5)
        for w in _page_words(page):
            wx0, wy0, wx1, wy1, text, *_ = w
            word_rect = fitz.Rect(wx0, wy0, wx1, wy1)
            if not word_rect.intersects(probe):
                continue
            t = (text or "").strip()
            if any(c.isdigit() for c in t):
                return True
        return False
    except Exception:
        return False


def _place_score_label(
    page,
    rect: fitz.Rect,
    page_idx: int,
    placed_lines_per_page: dict,
    score_text: str,
) -> None:
    """Place one red score label near the matched rect, avoiding collisions.

    Placement priority:
      1. If the line has a numeric value further right than the matched
         rect (typical `description … value` layout), place the score
         JUST AFTER that value on the same line — mirrors how theory-line
         scores sit just after the descriptive text, so the reader sees
         "value 0.25" unambiguously associated with that row.
      2. Same case but with no room right of the value → fall back to
         ABOVE the value (better than reverting to the text side).
      3. Full-line evidence reaching the right margin → ABOVE the right
         portion of the rect.
      4. Else default just-right-of-rect — works for plain text answers.
    """
    local_fs = _detect_fontsize_at_rect(page, rect, default=float(CONFIG['criterion_score_fontsize']))
    score_font = max(7.0, min(14.0, local_fs))

    # Right edge of the student's WRITING, not of the paper. These scripts are
    # landscape (842pt) with content ending near 575, so "page.rect.width - 70"
    # parks a mark 200pt out in blank space where it reads as belonging to
    # nothing. Stay just beyond the text instead.
    try:
        _content_x1 = max(
            (float(w[2]) for w in _page_words(page) if (w[4] or "").strip()),
            default=page.rect.width - 70,
        )
    except Exception:
        _content_x1 = page.rect.width - 70
    margin_x = min(_content_x1 + 8, page.rect.width - 60)

    score_y = max(min(rect.y1 - 2, page.rect.height - 10), 10)
    nearby_boxes: list[fitz.Rect] = placed_lines_per_page.get(page_idx, [])

    score_x = rect.x1 + 3
    placement_anchored_to_value = False

    # If the anchor rect ITSELF contains a numeric value, keep the score
    # right after that rect — don't push it to the end of the row. Prevents
    # "floating scores" on wide table rows where the anchor is one specific
    # cell (e.g. `18,150,000.00`) but the rightmost numeric word is a
    # different cell (e.g. `15,400,000.00` in the "post acq" column).
    anchor_is_value = _rect_contains_number(page, rect)

    # Step 1 — line has a value column further right than the rect:
    # place score AFTER the value on the same line (best clarity).
    # Skip when anchor rect already IS a numeric cell — see above.
    if not anchor_is_value:
        value_x = _find_value_x_on_line(page, rect)
        if value_x is not None and value_x > rect.x1 + 5:
            proposed = value_x + 3
            if proposed < page.rect.width - 25:
                score_x = proposed
                placement_anchored_to_value = True
            else:
                # No room to the right of the value — place ABOVE it.
                score_x = min(max(value_x - 25, 50), page.rect.width - 70)
                score_y = max(rect.y0 - (score_font + 2), 10)
                placement_anchored_to_value = True

    # Step 2 — rect itself reaches the right margin (full-line evidence
    # including the value). Default rect.x1 + 3 would push off-page.
    if score_x > page.rect.width - 25 and not placement_anchored_to_value:
        score_x = min(max(rect.x1 - 30, 50), page.rect.width - 70)
        score_y = max(rect.y0 - (score_font + 2), 10)
        placement_anchored_to_value = True

    def _score_box(x: float, y: float) -> fitz.Rect:
        return fitz.Rect(x, y - (score_font + 1), x + 52, y + 3)

    placed_box = _score_box(max(score_x, 50), score_y)
    if any(placed_box.intersects(b) for b in nearby_boxes):
        # When value-anchored, keep alternatives in the value column region
        # so stacked labels (two criteria on same line) stay near the value
        # rather than jumping back to the descriptive-text side.
        if placement_anchored_to_value:
            candidates_x = [
                min(score_x + 20, page.rect.width - 70),
                min(score_x + 40, page.rect.width - 70),
                max(score_x - 25, 50),
                max(score_x - 45, 50),
            ]
        elif anchor_is_value:
            # Rect is a value cell — try positions just right of the rect
            # (progressively further) before falling back to left / above.
            # Keeps stacked labels on the SAME visual row rather than
            # shifting to the next row (which looks like a floating mark).
            candidates_x = [
                min(rect.x1 + 20, page.rect.width - 70),
                min(rect.x1 + 40, page.rect.width - 70),
                min(rect.x1 + 60, page.rect.width - 70),
                max(rect.x0 - 40, 50),
            ]
        else:
            candidates_x = [
                max(rect.x0 - 40, 50),
                min(rect.x1 + 18, page.rect.width - 70),
                min(rect.x1 + 38, page.rect.width - 70),
            ]
        found = False
        for cx in candidates_x:
            cb = _score_box(cx, score_y)
            if not any(cb.intersects(b) for b in nearby_boxes):
                score_x = cx
                placed_box = cb
                found = True
                break

        if not found:
            # BEFORE shifting Y (which can push labels to the next row and
            # look like floating marks with no anchor), try placing ABOVE
            # the rect first — teacher-style "score sits above the value"
            # when there's no room to the right.
            above_y = max(rect.y0 - (score_font + 2), 10)
            above_box = _score_box(max(score_x, 50), above_y)
            # Must clear page TEXT as well as sibling labels. Checking only the
            # labels let a score hop onto the row above and read as marking it:
            # in a table with 12pt rows, two marks on the same data row were
            # pushed up onto the COLUMN HEADER, so the reader saw 0.25 against
            # "$" and "Rate" instead of against the figures underneath.
            if (
                not any(above_box.intersects(b) for b in nearby_boxes)
                and not _box_overlaps_page_text(page, above_box)
            ):
                score_y = above_y
                placed_box = above_box
                found = True

        if not found:
            # Stay CLOSE to the anchor. Stepping straight out to the margin
            # keeps the label on the right ROW but severs it from the value it
            # marks - on a table row carrying three marks they all ended up
            # stacked at the page edge, where a reader cannot tell which figure
            # each belongs to. Walk rightward in small steps, and only reach for
            # the margin once the row is genuinely full.
            for _step in (14, 28, 42, 56):
                _cx = min(score_x + _step, margin_x)
                cb = _score_box(_cx, score_y)
                if (
                    not any(cb.intersects(b) for b in nearby_boxes)
                    and not _box_overlaps_page_text(page, cb)
                ):
                    score_x, placed_box, found = _cx, cb, True
                    break
            if not found:
                mb = _score_box(margin_x, score_y)
                if not any(mb.intersects(b) for b in nearby_boxes):
                    score_x, placed_box, found = margin_x, mb, True

        if not found:
            # Last resort: shift DOWN, but only a tiny amount — half a font
            # height (~5px) — so the label stays visually attached to the
            # anchor rect's row. Larger shifts turn stacked labels into
            # "floating" orphans on the next physical row.
            for _shift in range(3):
                score_y = min(
                    score_y + (score_font * 0.5 + 1),
                    page.rect.height - 10,
                )
                cb = _score_box(max(score_x, 50), score_y)
                if not any(cb.intersects(b) for b in nearby_boxes):
                    placed_box = cb
                    break

    if _box_overlaps_page_text(page, placed_box):
        # Resolve the overlap SIDEWAYS first, staying on the anchor's row. The
        # old behaviour jumped 10pt up, which on a normal 10-12pt row grid is
        # exactly one line - trading an overlap for a label that appears to
        # mark the row above.
        resolved = False
        for _cx in (
            min(placed_box.x0 + 22, margin_x),
            min(placed_box.x0 + 44, margin_x),
            margin_x,
            margin_x + 22,
        ):
            cb = _score_box(_cx, score_y)
            if (
                not any(cb.intersects(b) for b in nearby_boxes)
                and not _box_overlaps_page_text(page, cb)
            ):
                score_x, placed_box, resolved = _cx, cb, True
                break
        if not resolved:
            shifted_y = max(score_y - 10, 10)
            shifted_box = _score_box(max(score_x, 50), shifted_y)
            if not any(shifted_box.intersects(b) for b in nearby_boxes):
                score_y = shifted_y
                placed_box = shifted_box

    page.insert_text(
        (max(score_x, 50), score_y),
        score_text,
        fontsize=score_font,
        color=CONFIG['criterion_score_color'],
    )
    placed_lines_per_page.setdefault(page_idx, []).append(placed_box)


# ── Tick mark drawing ──────────────────────────────────────────────────────────

def _place_ticks(page, rect: fitz.Rect, num_ticks: int) -> None:
    """Draw *num_ticks* red tick marks just above *rect*, scaled to line height.

    Compact sizing to match typical teacher-style check marks.
    """
    if num_ticks <= 0:
        return

    line_h = max(rect.y1 - rect.y0, 6.0)
    # Slightly smaller scale than before for compact teacher-like ticks
    scale = max(0.4, min(1.0, line_h / 14.0))
    tick_width = round(7 * scale)
    tick_gap = round(2 * scale)
    stroke_w = max(0.8, round(1.2 * scale, 1))
    lx = round(2 * scale)   # left-leg x-offset (shorter left leg)
    dn = round(3 * scale)   # left-leg down amount
    up = round(5 * scale)   # right-leg up amount

    # Place tick(s) centered horizontally above the key_phrase rect,
    # just above the text line — matching teacher annotation style.
    center_x = (rect.x0 + rect.x1) / 2
    base_y = rect.y0 - 1  # just above the text

    total_width = num_ticks * tick_width + (num_ticks - 1) * tick_gap
    start_x = center_x - total_width / 2

    for i in range(num_ticks):
        x = start_x + i * (tick_width + tick_gap)
        page.draw_line(
            fitz.Point(x, base_y - 1),
            fitz.Point(x + lx, base_y + dn),
            color=(1, 0, 0), width=stroke_w,
        )
        page.draw_line(
            fitz.Point(x + lx, base_y + dn),
            fitz.Point(x + tick_width, base_y - up),
            color=(1, 0, 0), width=stroke_w,
        )


# ── "Not required" marker ──────────────────────────────────────────────────────

def place_not_required_marker(page, rect: fitz.Rect, reason: str = "") -> None:
    """Mark *rect* as off-topic with a strikethrough + 'Not required' margin label.

    Visual style matches a teacher's red pen: line through the off-topic text,
    label in the right margin, optional popup with the explanation.
    """
    line_h = max(rect.y1 - rect.y0, 6.0)
    mid_y = (rect.y0 + rect.y1) / 2
    red = (1, 0, 0)

    # Strikethrough across the off-topic span.
    page.draw_line(
        fitz.Point(rect.x0, mid_y),
        fitz.Point(rect.x1, mid_y),
        color=red,
        width=max(0.8, round(line_h / 12, 1)),
    )

    # "Not required" label placed in the right margin, vertically aligned with the line.
    page_w = page.rect.width
    label_x = min(rect.x1 + 6, page_w - 80)
    if label_x <= rect.x1:
        label_x = max(page_w - 80, rect.x1 + 4)
    label_y = rect.y1 - 1
    page.insert_text(
        (label_x, label_y),
        "Not required",
        fontsize=max(7.0, min(9.0, line_h * 0.7)),
        color=red,
    )

    # Popup with the reason (only if a reason was provided).
    if reason and reason.strip():
        try:
            popup_anchor = (max(rect.x0 - 12, 4), rect.y0)
            annot = page.add_text_annot(popup_anchor, reason.strip(), icon="Comment")
            annot.set_colors(stroke=red)
            annot.update()
        except Exception:
            pass


# ── Score-near-anchor placement ────────────────────────────────────────────────

def place_score_near_anchor(
    doc,
    anchor_text: str,
    score_text: str,
    allowed_pages: list[int],
    placed_lines_per_page: dict,
    placed_marks: set,
    unplaced_items: list,
    page_token_sets: Optional[dict[int, set[str]]] = None,
    line_score_accumulator: Optional[dict] = None,
    min_y_per_page: Optional[dict[int, float]] = None,
    max_y_per_page: Optional[dict[int, float]] = None,
    draw_underline: bool = True,
    # FIX-1: draw_underline=False when anchor is a question/sub-question heading label.
    # Headings should only receive a score mark, never an underline.
    use_fragments: bool = True,
    use_number_first: bool = True,
    # use_fragments=False  \u2192 search the WHOLE evidence string (no ; / \n / | splitting).
    # use_number_first=False \u2192 skip hybrid number+context. With both off,
    #   placement is driven by exact substring match of the full evidence,
    #   which avoids landing on the wrong occurrence of a shared number.
) -> bool:
    """Resolve *anchor_text* in the PDF and place a score label nearby.

    Returns True when the label was placed successfully.
    """
    evidence_clean = _strip_llm_artifacts(
        (anchor_text or "").replace("\u2026", " ").replace("...", " ").replace('|', ' ')
    )
    if not evidence_clean:
        unplaced_items.append((score_text, "[empty]"))
        logger.debug(f"    [placement] ✗ Empty evidence for score {score_text}")
        return False

    logger.debug(f"    [placement] Searching for anchor: '{evidence_clean[:60]}'...")

    if use_fragments:
        candidate_fragments = _build_candidate_fragments(evidence_clean)
    else:
        # Numerical mode: keep the evidence intact so each criterion anchors
        # to its specific working line, not a sub-string shared with others.
        candidate_fragments = [evidence_clean]

    rect, page_num = None, -1
    used_anchor = evidence_clean
    for fragment in candidate_fragments:
        for candidate in _build_anchor_variations(fragment):
            used_anchor = candidate
            rect, page_num = resolve_anchor_rect(
                doc,
                candidate,
                allowed_pages,
                placed_marks=placed_marks,
                skip_duplicates=(line_score_accumulator is None),
                expand_to_line=False,
                page_token_sets=page_token_sets,
                redirect_headings=False,
                min_y_per_page=min_y_per_page,
                max_y_per_page=max_y_per_page,
                use_number_first=use_number_first,
            )
            if rect and page_num != -1:
                break
        if rect and page_num != -1:
            break

    if rect and page_num != -1:
        page = doc[page_num - 1]
        page_idx = page_num - 1

        # FIX-1: only underline evidence lines, never question heading labels
        if draw_underline:
            _draw_underline_for_rect(page, rect)

        if line_score_accumulator is not None:
            line_key_val = _line_key(page_num, rect.y0)
            try:
                score_value = float(score_text)
                # Pure numeric mark: accumulate so co-located criteria produce
                # one combined label instead of overlapping individual labels.
                entry = line_score_accumulator.get(line_key_val)
                if not entry:
                    line_score_accumulator[line_key_val] = {
                        "page_num": page_num,
                        "page_idx": page_idx,
                        "rect": rect,
                        "marks": [score_value],
                    }
                else:
                    entry["marks"].append(score_value)
                    if rect.x1 > entry["rect"].x1:
                        entry["rect"] = rect
                logger.debug(
                    f"    [placement] ✓ Placed at page {page_num} (grouped) "
                    f"(anchor='{used_anchor[:45]}...')"
                )
            except (ValueError, TypeError):
                # Non-numeric label ("OF 0.5", "Marks given above", "Marks given below"):
                # place directly — these carry contextual meaning that must not be
                # silently converted to 0 and swallowed by the numeric accumulator.
                _place_score_label(page, rect, page_idx, placed_lines_per_page, score_text)
                logger.debug(
                    f"    [placement] ✓ Placed '{score_text}' directly at page {page_num} "
                    f"(anchor='{used_anchor[:45]}...')"
                )
        else:
            line_key_val = _line_key(page_num, rect.y0)
            if line_key_val not in placed_marks:
                _place_score_label(page, rect, page_idx, placed_lines_per_page, score_text)
                placed_marks.add(line_key_val)
            logger.debug(
                f"    [placement] ✓ Placed at page {page_num} "
                f"(anchor='{used_anchor[:45]}...')"
            )
        return True
    else:
        unplaced_items.append((score_text, evidence_clean[:50]))
        logger.debug(
            f"    [placement] ✗ Failed: {score_text} - anchor not found "
            f"for '{evidence_clean[:50]}'"
        )
        return False


# ── Total score placement ──────────────────────────────────────────────────────

def add_main_score(
    doc,
    q_num: str,
    score_text: str,
    allowed_pages: list[int],
    student_heading_text: Optional[str] = None,
) -> bool:
    """Place the total score to the left of the question heading.

    Searches all allowed pages for the heading.  Falls back to the top-left
    of the first page when no heading is found.
    """
    if not q_num:
        logger.warning("No question number for main score placement")
        return False

    q_str = str(q_num).strip()

    strategies: list[tuple[str, str]] = []
    if student_heading_text:
        strategies.append((student_heading_text, "Student exact heading"))
    strategies += [
        (f"Question {q_str}", "Question prefix"),
        (f"Q-{q_str.zfill(2)}", "Q-0N prefix"),
        (f"Q-{q_str}", "Q-N prefix"),
        (f"Q.{q_str}", "Q.N prefix"),
        (f"Q{q_str}", "QN prefix"),
        (f"Answer {q_str}", "Answer prefix"),
        (f"Ans.{q_str}", "Ans. prefix"),
        (f"Ans {q_str}", "Ans prefix"),
    ]

    def _is_question_heading(page, rect: fitz.Rect, search_text: str) -> bool:
        """Validate that a match is an actual question heading.

        Checks:
        1. Match is near the left margin (x < 250)
        2. Not embedded in a longer non-heading word (e.g. "Issue-01")
        3. Substantive content below — OR — heading is near the bottom of the
           page (in which case content continues on the next page).

        FIX-4: relaxed to accept bottom-of-page headings whose answer content
        is on the following page.
        """
        if rect.x0 > 250:
            return False

        # Reject if our match is embedded inside a non-heading label
        for word_obj in _page_words(page):
            word_text = word_obj[4].strip()
            word_y_mid = (word_obj[1] + word_obj[3]) / 2
            rect_y_mid = (rect.y0 + rect.y1) / 2
            if abs(word_y_mid - rect_y_mid) > 5:
                continue
            wl = word_text.lower()
            if any(prefix in wl for prefix in ("issue", "rs.", "rs,", "page", "total")):
                if search_text.lower() in wl and wl != search_text.lower():
                    return False

        # FIX-4: heading near the bottom of the page → content is on the next page
        if rect.y1 > page.rect.height * 0.80:
            return True

        # Validate substantive content below (within 80 pt on the same page)
        min_y_below = rect.y1 + 5
        max_y_below = min(rect.y1 + 80, page.rect.height - 20)
        for word_obj in _page_words(page):
            word_text = word_obj[4].strip()
            word_y = word_obj[1]
            if (
                min_y_below <= word_y <= max_y_below
                and len(word_text) > 1
                and not word_text.isspace()
                and not re.match(r'^[^a-zA-Z0-9]*$', word_text)
            ):
                return True

        return False

    for page_num in allowed_pages:
        page = doc[page_num - 1]

        for search_text, strategy_name in strategies:
            instances = _page_search(page, search_text)
            if not instances:
                continue

            for heading_rect in instances:
                if not _is_question_heading(page, heading_rect, search_text):
                    logger.debug(
                        f"      [main-score] Rejected '{search_text}' at "
                        f"y={heading_rect.y0:.0f} - not a heading"
                    )
                    continue

                x = heading_rect.x0 - CONFIG['main_score_offset_x']
                y = heading_rect.y0 + CONFIG['main_score_offset_y'] - 10

                page.insert_text(
                    (x, y),
                    score_text,
                    fontsize=CONFIG['main_score_fontsize'],
                    color=CONFIG['main_score_color'],
                )
                logger.info(
                    f"Main score '{score_text}' placed (strategy: {strategy_name}) "
                    f"on page {page_num} - content validated"
                )
                return True

    # Fallback: top-left of the first allowed page
    if allowed_pages:
        fallback_page = doc[allowed_pages[0] - 1]
        fallback_page.insert_text(
            (30, 30),
            score_text,
            fontsize=CONFIG['main_score_fontsize'],
            color=CONFIG['main_score_color'],
        )
        logger.info(
            f"Main score '{score_text}' placed at top of page {allowed_pages[0]} "
            f"(fallback - no heading found)"
        )
        return True

    logger.warning(f"Main score not placed - Q{q_str} heading not found or no substantive content")
    return False



# ── Comment popup placement ────────────────────────────────────────────────────

def _strip_subq_prefix(comment: str) -> tuple[Optional[str], str]:
    """Extract a leading [<sub_question>] tag from a comment string.

    Returns (sub_id, comment_without_prefix). sub_id is None if no prefix is present.
    The numeric portion of sub_id (e.g. '4.1' from '4.1 Threats') is what the PDF
    search uses — descriptive suffixes added by the prompt are tolerated.
    """
    if not comment or not isinstance(comment, str):
        return None, comment or ""
    m = re.match(r"^\s*\[\s*([^\]]+?)\s*\]\s*", comment)
    if not m:
        return None, comment
    sub_id = m.group(1).strip()
    remainder = comment[m.end():]
    return sub_id, remainder


def compute_subq_y_bounds(
    doc,
    allowed_pages: list[int],
    sub_ids: list[str],
) -> dict[str, dict[int, tuple[float, float]]]:
    """For each sub_id (e.g. '4.1', '4.2 Threats'), find its Y range on each
    allowed page so comments can be constrained to that sub-question's region.

    The search uses only the leading numeric portion (e.g. '4.1') because student
    handwriting rarely echoes descriptive labels from the rubric. The next
    numerically-sorted sub_id's position gives the max_y; page bottom otherwise.

    Returns {sub_id: {page_num: (min_y, max_y)}}.
    """
    result: dict[str, dict[int, tuple[float, float]]] = {sid: {} for sid in sub_ids}
    if not sub_ids:
        return result

    # Map each sub_id to its numeric prefix (e.g. '4.1 Threats' -> '4.1').
    def _numeric(sid: str) -> Optional[str]:
        m = re.match(r"\s*(\d+(?:\.\d+)*)", sid)
        return m.group(1) if m else None

    sid_numeric: dict[str, str] = {}
    for sid in sub_ids:
        n = _numeric(sid)
        if n:
            sid_numeric[sid] = n

    if not sid_numeric:
        return result

    # NOTHING TO SEPARATE. Bounding exists to stop a comment on 4.1 landing in
    # 4.2's region. When every comment belongs to the SAME sub-question there is
    # no neighbour to protect against, and the whole answer is fair game — so a
    # region can only exclude valid targets. A whole-question paper carries the
    # single prefix "1", and bounding it cut two comments that matched cleanly.
    if len(set(sid_numeric.values())) < 2:
        return result

    # Unique numeric sub-question prefixes in numerical order. Used to look up
    # the next sub-question's start position on each page.
    unique_numerics = sorted(
        set(sid_numeric.values()),
        key=lambda n: tuple(float(p) for p in n.split(".")),
    )

    for page_num in allowed_pages:
        try:
            page = doc[page_num - 1]
        except Exception:
            continue

        # Words on this page, so a numeric label can be required to stand as its
        # OWN token. A raw substring search for "1" matches the 1 inside
        # "1,000,000" or "£1.2M", and one such hit in the left margin silently
        # moved a whole page's allowed region past the content it should have
        # covered. A label is a word like "4.1", "1." or "1)" — never a digit
        # buried in an amount.
        try:
            page_words = _page_words(page) or []
        except Exception:
            page_words = []

        def _label_hits(n: str) -> list[float]:
            ys: list[float] = []
            for w in page_words:
                try:
                    x0, y0, text = float(w[0]), float(w[1]), str(w[4])
                except (IndexError, TypeError, ValueError):
                    continue
                if x0 >= 110:
                    continue  # inline reference, not a left-margin label
                if text.strip().rstrip(".):-") == n:
                    ys.append(y0)
            return sorted(ys)

        numeric_positions: dict[str, float] = {}
        for n in unique_numerics:
            hits = _label_hits(n)
            if hits:
                numeric_positions[n] = hits[0]

        if not numeric_positions:
            continue

        sorted_present = sorted(
            numeric_positions.keys(),
            key=lambda n: tuple(float(p) for p in n.split(".")),
        )
        position_index = {n: i for i, n in enumerate(sorted_present)}

        for sid, numeric in sid_numeric.items():
            if numeric not in numeric_positions:
                continue
            min_y = numeric_positions[numeric] - 5
            max_y = page.rect.height - 20
            idx = position_index[numeric]
            for next_n in sorted_present[idx + 1:]:
                if next_n in numeric_positions:
                    max_y = numeric_positions[next_n] - 5
                    break
            result[sid][page_num] = (min_y, max_y)

    return result


def add_popup_for_comment(
    doc,
    comment: str,
    allowed_pages: list[int],
    placed_marks: Optional[set] = None,
    page_token_sets: Optional[dict[int, set[str]]] = None,
    comment_page_y: Optional[dict[int, float]] = None,
    comment_used_y: Optional[dict[int, list[float]]] = None,
    ocr_textpages: Optional[dict[int, object]] = None,
    placed_lines_per_page: Optional[dict] = None,
    min_y_per_page: Optional[dict[int, float]] = None,
    max_y_per_page: Optional[dict[int, float]] = None,
    subq_y_bounds: Optional[dict[str, dict[int, tuple[float, float]]]] = None,
) -> bool:
    """Place a PDF comment annotation anchored to the feedback text.

    Tries four strategies in order:
      A. Normal PDF-text anchor via resolve_anchor_rect
      B. OCR-based best-line match on likely pages
      C. Sliding sub-phrase search for cross-line anchors
      D. Margin-stacking within the question Y territory (fallback)

    FIX-2: anchored comments are clamped to q_max_y so the popup icon cannot
    bleed visually into the first line of the next question.
    """
    # Strip the [<sub_question>] prefix added by the grading prompts. The sub_id
    # narrows comment placement to its sub-question region; if no prefix is
    # present (older outputs or non-conforming LLM responses), fall back to the
    # whole-question Y range.
    sub_id, comment = _strip_subq_prefix(comment)
    sub_bounds_for_comment: dict[int, tuple[float, float]] = {}
    if sub_id and subq_y_bounds:
        sub_bounds_for_comment = subq_y_bounds.get(sub_id, {}) or {}

    parsed = _split_comment_arrow(comment)
    if not parsed:
        logger.debug(f"  No usable arrow split in comment: '{str(comment)[:40]}...'")
        return False

    if 'TOTAL SCORE' in comment.upper() or re.search(r'\d+\.\d+/\d+', comment):
        logger.debug("  Skipping total score comment")
        return False

    anchor_part, feedback_part = parsed

    if comment_page_y is None:
        comment_page_y = {}
    if comment_used_y is None:
        comment_used_y = {}

    ranked_pages = _rank_pages_for_anchor(
        page_token_sets or {}, list(allowed_pages), anchor_part
    )
    ranked_pages = ranked_pages if ranked_pages else list(allowed_pages)

    # If a sub-question prefix is present and we have bounds for it, prefer the
    # pages where that sub-question actually appears.
    if sub_bounds_for_comment:
        sub_pages = [p for p in ranked_pages if p in sub_bounds_for_comment]
        if sub_pages:
            ranked_pages = sub_pages + [p for p in ranked_pages if p not in sub_pages]

    def _place_note_on_page(
        page,
        page_num: int,
        target_rect: Optional[fitz.Rect] = None,
    ) -> bool:
        q_min_y = float((min_y_per_page or {}).get(page_num, 0))
        q_max_y = float((max_y_per_page or {}).get(page_num, page.rect.height - 20))

        # Tighten to the sub-question's region when we have it for this page.
        if page_num in sub_bounds_for_comment:
            sub_min, sub_max = sub_bounds_for_comment[page_num]
            q_min_y = max(q_min_y, float(sub_min))
            q_max_y = min(q_max_y, float(sub_max))

        # Icon geometry, shared by every collision pass below. add_text_annot()
        # takes its point as the icon's TOP-LEFT, so the footprint runs down and
        # right from (x, y) at the size the viewer actually draws.
        _ICON_W, _ICON_H = 18.0, 18.0

        def _icon_box(cx: float, cy: float) -> fitz.Rect:
            return fitz.Rect(cx, cy, cx + _ICON_W, cy + _ICON_H)

        # Words the icon must not cover. Populated once the anchor row is known.
        row_word_boxes: list[fitz.Rect] = []

        if target_rect is not None:
            # Reject if anchor is outside this question's territory
            if q_min_y > 0 and target_rect.y0 < q_min_y:
                logger.debug(
                    f"  [comment] ✗ target_rect above min_y "
                    f"({target_rect.y0:.0f} < {q_min_y:.0f}), dropped"
                )
                return False
            if q_max_y < page.rect.height - 20 and target_rect.y0 > q_max_y:
                logger.debug(
                    f"  [comment] ✗ target_rect below max_y "
                    f"({target_rect.y0:.0f} > {q_max_y:.0f}), dropped"
                )
                return False
            x = min(target_rect.x1 + 6, page.rect.width - 20)
            x = max(x, 10)
            y = max(min(target_rect.y0 - 1, page.rect.height - 20), 20)
            if y < 90:
                y = 110
            # FIX-2: clamp anchored comment to question territory so icon does
            # not bleed visually into the first line of the next question.
            if q_max_y < page.rect.height - 20:
                y = min(y, q_max_y - 15)

            # FIX-3: keep the note icon on the SAME visual row (same y) but
            # place it on EMPTY horizontal space, not on top of text. The
            # default `target_rect.x1 + 6` sits right after the anchor value
            # which often overlaps the next column of text on wide table
            # rows (e.g. "add back nci 6,975,000" note icon landing on top
            # of the value "6,975,000.00" or on the goodwill row below).
            # Slide the icon rightward on the same row until it's clear of
            # any word-word bounding box.
            try:
                # Collect words over the ICON'S OWN vertical footprint, not the
                # anchor row. An 18pt icon on a 10pt row grid always reaches
                # into the neighbouring rows, so a row-band check declares it
                # clear while it sits squarely on the line below - which is how
                # a note landed on top of "12,125" and hid the student's
                # goodwill figure completely.
                # add_text_annot() treats its point as the icon's TOP-LEFT, so
                # the footprint runs down-and-right from (x, y). Testing a
                # centred box checked an area offset 9pt up and left of where
                # the icon actually lands, which is how a note cleared its
                # collision test and still sat on the row below.
                _pad = 2.0
                _row_y_min = float(y) - _pad
                _row_y_max = float(y) + _ICON_H + _pad
                row_word_boxes = []
                for w in _page_words(page):
                    wx0, wy0, wx1, wy1, wt, *_ = w
                    if wy1 < _row_y_min or wy0 > _row_y_max:
                        continue
                    if not (wt or "").strip():
                        continue
                    row_word_boxes.append(fitz.Rect(wx0, wy0, wx1, wy1))

                _proposed = _icon_box(x, y)
                if any(_proposed.intersects(wb) for wb in row_word_boxes):
                    # Slide right in small steps until we hit clear space
                    # (or the page's right margin).
                    _step = 8.0
                    _limit = page.rect.width - _ICON_W - 4
                    _cx = x
                    while _cx < _limit:
                        _cx += _step
                        _test = _icon_box(_cx, y)
                        if not any(_test.intersects(wb) for wb in row_word_boxes):
                            x = _cx
                            break
                    else:
                        # No clear space on the row's right — fall back to
                        # the page's right margin, still on the same y.
                        x = max(page.rect.width - 20, 10)
            except Exception:
                # Non-fatal: if we can't compute word boxes, keep the
                # original x (previous behaviour).
                pass
        else:
            effective_start = max(120.0, float(page.rect.height) * 0.25, q_min_y + 10)
            y = float(comment_page_y.get(page_num, effective_start))
            x = max(page.rect.width - 24, 10)
            y = max(min(y, page.rect.height - 20), 120)
            if q_min_y > 0:
                y = max(y, q_min_y + 5)
            if q_max_y - 5 > q_min_y + 5:
                y = min(y, q_max_y - 5)

        # Avoid collisions with score labels
        if placed_lines_per_page:
            page_idx = page_num - 1
            score_boxes = placed_lines_per_page.get(page_idx, [])
            # Dodging a score label must not undo the word-avoidance above.
            # This pass previously checked score boxes only, so it happily
            # shifted an icon 20pt sideways back on top of the student's
            # figures - which is how a note ended up covering "(3,000)" and
            # "12,125" after correctly sliding clear of them a moment earlier.
            def _icon_collides(bx: fitz.Rect) -> bool:
                return any(bx.intersects(sb) for sb in score_boxes) or any(
                    bx.intersects(wb) for wb in row_word_boxes
                )

            if _icon_collides(_icon_box(x, y)):
                for shift_x in [20, -20, 35, -35, 55, -55, 75]:
                    nx = max(10, min(x + shift_x, page.rect.width - 20))
                    if not _icon_collides(_icon_box(nx, y)):
                        x = nx
                        break

        # Avoid collisions with existing note icons near the same y.
        # Alternate between shifting down and up to stay close to anchor.
        used = comment_used_y.setdefault(page_num, [])
        original_y = y
        for attempt in range(8):
            if all(abs(y - uy) > 14 for uy in used):
                break
            if attempt % 2 == 0:
                y = min(original_y + (attempt // 2 + 1) * 16, page.rect.height - 20)
            else:
                y = max(original_y - (attempt // 2 + 1) * 16, 20)

        # FIX-2: re-clamp after collision shifts to stay within question territory
        if q_max_y < page.rect.height - 20:
            y = min(y, q_max_y - 15)

        try:
            annot = page.add_text_annot((x, y), "", icon="Note")
            annot.set_colors(stroke=CONFIG['comment_color'])
            annot.set_opacity(0.85)
            annot.set_info(content=feedback_part, title="Feedback")
            annot.update()
            used.append(float(y))
            if target_rect is None:
                comment_page_y[page_num] = float(y) + 22.0
            logger.debug(
                f"  [comment] ✓ Comment popup at page {page_num}, y={y:.1f} "
                f"(anchored={'yes' if target_rect else 'no'})"
            )
            return True
        except Exception as e:
            logger.debug(f"  [comment] ✗ Error adding annotation: {e}")
            return False

    # ── Attempt A: normal PDF-text anchor ──────────────────────────────────────
    rect, page_num = None, -1
    for candidate in _build_anchor_variations(anchor_part):
        rect, page_num = resolve_anchor_rect(
            doc,
            candidate,
            ranked_pages,
            placed_marks=placed_marks,
            skip_duplicates=False,
            page_token_sets=page_token_sets,
            use_number_first=True,
            redirect_headings=True,
            expand_to_line=True,
            min_y_per_page=min_y_per_page,
            max_y_per_page=max_y_per_page,
        )
        if rect and page_num != -1:
            break

    if rect and page_num != -1:
        try:
            page = doc[page_num - 1]
        except Exception:
            page = None
        if page is not None:
            return _place_note_on_page(page, page_num, target_rect=rect)

    # ── Attempt B: OCR-based best-line match ───────────────────────────────────
    def _iter_lines_from_dict(text_dict: dict) -> Iterable[tuple[str, fitz.Rect]]:
        for block in text_dict.get("blocks", []) or []:
            for line in block.get("lines", []) or []:
                spans = line.get("spans", []) or []
                line_text = "".join((s.get("text") or "") for s in spans).strip()
                bbox = line.get("bbox")
                if not line_text or not bbox:
                    continue
                yield line_text, fitz.Rect(bbox)

    def _best_line_rect(page, textpage_obj) -> Optional[fitz.Rect]:
        try:
            td = (
                page.get_text("dict", textpage=textpage_obj)
                if textpage_obj is not None
                else _page_dict(page)
            )
        except Exception:
            return None

        required_number = extract_number_from_text(anchor_part)
        anchor_tokens = set(
            _tokenize(clean_anchor_text(anchor_part, max_words=CONFIG['max_anchor_words']) or anchor_part)
        )
        if not anchor_tokens:
            return None

        best_r: Optional[fitz.Rect] = None
        best_s = 0.0
        local_page_num = page.number + 1
        q_min = float((min_y_per_page or {}).get(local_page_num, 0))
        q_max = float((max_y_per_page or {}).get(local_page_num, float('inf')))

        for lt, lr in _iter_lines_from_dict(td):
            if q_min > 0 and lr.y0 < q_min:
                continue
            if q_max < float('inf') and lr.y0 > q_max:
                continue
            lt_tokens = set(_tokenize(lt))
            overlap = len(anchor_tokens & lt_tokens)
            if overlap < 2:
                continue
            score = overlap / max(len(anchor_tokens), 1)
            if required_number:
                rn = required_number.replace(",", "").replace(" ", "")
                if rn and rn in _normalize_text_for_match(lt).replace(",", "").replace(" ", ""):
                    score += 0.35
            if score > best_s:
                best_s = score
                best_r = lr

        return best_r if best_s >= 0.40 else None

    if ocr_textpages is None:
        ocr_textpages = {}

    for pnum in ranked_pages[:3]:
        try:
            page = doc[pnum - 1]
        except Exception:
            continue

        tp = ocr_textpages.get(pnum)
        if tp is None:
            try:
                has_words = bool(_page_words(page))
            except Exception:
                has_words = False
            if not has_words:
                try:
                    tp = page.get_textpage_ocr(dpi=200, full=True)
                except Exception:
                    tp = None
            ocr_textpages[pnum] = tp

        best_rect = _best_line_rect(page, tp)
        if best_rect is not None:
            return _place_note_on_page(page, pnum, target_rect=best_rect)

    # ── Attempt C: sliding sub-phrase search ───────────────────────────────────
    words = anchor_part.split()
    sub_anchors: list[str] = []
    seen_subs: set[str] = set()
    for window in (4, 3, 2):
        for i in range(len(words) - window + 1):
            phrase = " ".join(words[i:i + window])
            if phrase not in seen_subs:
                seen_subs.add(phrase)
                sub_anchors.append(phrase)
    if not sub_anchors and len(words) >= 2:
        sub_anchors.append(" ".join(words[:2]))

    for sub in sub_anchors:
        for candidate in _build_anchor_variations(sub):
            rect, page_num = resolve_anchor_rect(
                doc,
                candidate,
                ranked_pages,
                placed_marks=placed_marks,
                skip_duplicates=False,
                page_token_sets=page_token_sets,
                use_number_first=False,
                redirect_headings=False,
                expand_to_line=True,
                min_y_per_page=min_y_per_page,
                max_y_per_page=max_y_per_page,
            )
            if rect and page_num != -1:
                try:
                    page = doc[page_num - 1]
                except Exception:
                    page = None
                if page is not None:
                    logger.debug(f"  [comment] Sub-phrase anchor matched: '{sub}'")
                    return _place_note_on_page(page, page_num, target_rect=rect)

    # ── Attempt D: margin-stack within question territory (fallback) ────────────
    page_num = ranked_pages[0] if ranked_pages else (allowed_pages[0] if allowed_pages else 1)
    try:
        page = doc[page_num - 1]
    except Exception:
        logger.debug(f"  [comment] ✗ Invalid page for fallback placement: {page_num}")
        return False

    q_min = float((min_y_per_page or {}).get(page_num, 0))
    q_max = float((max_y_per_page or {}).get(page_num, page.rect.height - 20))
    if q_max <= q_min + 20:
        logger.debug(f"  [comment] ✗ No usable territory on page {page_num}, comment dropped")
        return False

    logger.debug(
        f"  [comment] Fallback margin-stack within Q territory "
        f"y=[{q_min:.0f},{q_max:.0f}] page {page_num}"
    )
    return _place_note_on_page(page, page_num, target_rect=None)
