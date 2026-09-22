# ==================== ANCHOR RESOLUTION ENGINE ====================
#
# Provides resolve_anchor_rect() — a multi-strategy, OCR-aware engine that
# finds where a piece of evidence text lives on the PDF page(s).
#
# Supporting helpers: page ranking, number matching, fuzzy line matching,
# text cleaning, and hybrid number+context search.

import re
from typing import Iterable, Optional

import fitz

from logging_config import logger
from .annotator_config import CONFIG
from .annotator_ocr import _page_search, _page_words, _page_dict
from .annotator_text import (
    _normalize_text_for_match, _strip_llm_artifacts,
    _tokenize, _build_anchor_variations, _line_key,
    _normalize_symbols_for_match,
)
from .annotator_rect import (
    _iter_page_lines, _expand_rect_to_line, _expand_rect_to_row,
    _line_text_for_rect, _is_heading_like, _redirect_if_header_like,
    is_on_same_line, _find_next_numeric_line, _refine_to_numberish_word_on_line,
)


# ── Page ranking ───────────────────────────────────────────────────────────────

def _rank_pages_for_anchor(
    page_token_sets: dict[int, set[str]],
    allowed_pages: list[int],
    anchor_text: str,
) -> list[int]:
    """Return *allowed_pages* sorted by token overlap with *anchor_text* (best first)."""
    anchor_tokens = set(_tokenize(anchor_text))
    if not anchor_tokens or not page_token_sets:
        return allowed_pages

    scored: list[tuple[int, int]] = []
    for p in allowed_pages:
        pset = page_token_sets.get(p)
        score = len(anchor_tokens & pset) if pset else 0
        scored.append((score, p))

    scored.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    return [p for _, p in scored]


# ── Context word extraction ────────────────────────────────────────────────────

def _build_context_words(text: str, max_words: int = 6) -> list[str]:
    """Extract distinctive context tokens for hybrid number+context matching."""
    toks = _tokenize(text)
    if not toks:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_words:
            break
    return out


# ── Number variations ──────────────────────────────────────────────────────────

def _build_number_variations(num_text: str) -> list[str]:
    """Return all common formatting variants of a numeric string.

    Handles thousands separators (comma / space), trailing .00, and
    fractions like 9/12.
    """
    clean = (num_text or "").replace(",", "").replace(" ", "")
    if not clean:
        return []

    if "/" in clean and re.fullmatch(r"\d+/\d+", clean):
        n, d = clean.split("/", 1)
        return [clean, f"{n} / {d}", f"{n}/{d}"]

    if re.fullmatch(r"\d+\.\d+", clean):
        int_part, dec_part = clean.split(".", 1)
    else:
        int_part, dec_part = clean, None

    variants: list[str] = [clean]

    def _add_thousands(int_only: str) -> None:
        if not int_only.isdigit() or len(int_only) <= 3:
            return
        chunks: list[str] = []
        s = int_only
        while len(s) > 3:
            chunks.append(s[-3:])
            s = s[:-3]
        chunks.append(s)
        chunks = list(reversed(chunks))
        comma = ",".join(chunks)
        space = " ".join(chunks)
        variants.extend([comma, space])
        if dec_part is not None:
            variants.extend([f"{comma}.{dec_part}", f"{space}.{dec_part}"])

    _add_thousands(int_part)

    if dec_part is None and int_part.isdigit():
        variants.append(f"{int_part}.00")
        if len(int_part) > 3:
            chunks: list[str] = []
            s = int_part
            while len(s) > 3:
                chunks.append(s[-3:])
                s = s[:-3]
            chunks.append(s)
            chunks = list(reversed(chunks))
            comma = ",".join(chunks)
            space = " ".join(chunks)
            variants.extend([f"{comma}.00", f"{space}.00"])

    dedup: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if v and v not in seen:
            dedup.append(v)
            seen.add(v)
    return dedup


def _search_number_variations(page, number_str: str) -> list[fitz.Rect]:
    """Search for all formatting variants of *number_str* and return unique rects."""
    rects: list[fitz.Rect] = []
    for variation in _build_number_variations(number_str):
        try:
            hits = _page_search(page, variation)
        except Exception:
            hits = []
        if hits:
            rects.extend(hits)

    dedup: list[fitz.Rect] = []
    seen: set[tuple[float, float, float, float]] = set()
    for r in rects:
        key = (round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(r)
    return dedup


# ── Line-level fuzzy match ─────────────────────────────────────────────────────

def _find_best_line_match(
    page,
    anchor_text: str,
    required_number: Optional[str] = None,
) -> Optional[fitz.Rect]:
    """Return the page line whose tokens best overlap with *anchor_text*."""
    best_chunk = clean_anchor_text(anchor_text, max_words=CONFIG['max_anchor_words'])
    anchor_tokens = _tokenize(best_chunk or anchor_text)
    if not anchor_tokens:
        return None

    anchor_set = set(anchor_tokens)
    best_rect: Optional[fitz.Rect] = None
    best_score = 0.0

    required_number_norm = None
    if required_number:
        required_number_norm = required_number.replace(",", "").replace(" ", "")

    for line_text, line_rect in _iter_page_lines(page):
        lt_norm = _normalize_text_for_match(line_text)
        line_tokens = _tokenize(line_text)
        if not line_tokens:
            continue

        line_set = set(line_tokens)
        overlap = len(anchor_set & line_set)
        if overlap < 1:
            continue
        if overlap < 2:
            matching = anchor_set & line_set
            if not any(len(t) >= 6 for t in matching):
                continue

        score = overlap / max(len(anchor_set), 1)

        if required_number_norm:
            lt_digits = lt_norm.replace(",", "").replace(" ", "")
            if required_number_norm in lt_digits:
                score += 0.35

        if score > best_score:
            best_score = score
            best_rect = line_rect

    if best_score >= 0.45:
        return best_rect

    # Fallback: allow a single long keyword to match a value line.
    strong_tokens = [t for t in anchor_set if t.isalpha() and len(t) >= 7]
    if strong_tokens:
        best_rect2: Optional[fitz.Rect] = None
        best_score2 = 0.0
        for line_text, line_rect in _iter_page_lines(page):
            lt_norm = _normalize_text_for_match(line_text)
            line_tokens = _tokenize(line_text)
            if not line_tokens:
                continue
            if not any(st in line_tokens for st in strong_tokens):
                continue
            has_value = bool(re.search(r"\d", lt_norm)) or bool(
                re.search(r"£|\$|%", line_text) and re.search(r"\d", lt_norm)
            )
            if not has_value:
                continue
            score2 = 0.30 + (len(anchor_set & set(line_tokens)) / max(len(anchor_set), 1)) * 0.20
            if score2 > best_score2:
                best_score2 = score2
                best_rect2 = line_rect

        if best_rect2 is not None and best_score2 >= 0.30:
            return best_rect2
    return None


# ── Anchor text cleaning ───────────────────────────────────────────────────────

def clean_anchor_text(text: str, max_words: int = 6) -> Optional[str]:
    """Extract the most distinctive *max_words*-word chunk from *text*.

    Prioritises chunks with numbers, currency symbols, and proper nouns.
    """
    if not text or not isinstance(text, str):
        return None
    text = _strip_llm_artifacts(text)
    words = text.split()
    if len(words) < 2:
        return None

    best_chunk, best_score = None, 0
    for i in range(len(words) - max_words + 1):
        chunk = " ".join(words[i:i + max_words])
        score = (
            chunk.count(',') + chunk.count('£') + chunk.count('$')
            + chunk.count('%') + chunk.count('×')
            + sum(1 for w in chunk.split() if re.match(r'^\d|\d.*\d|FV|NCI|OCI', w))
            + len([w for w in chunk.split() if w[0].isupper() and len(w) > 2])
        )
        if score > best_score:
            best_score, best_chunk = score, chunk
    return best_chunk or " ".join(words[:max_words])


# ── Partial / fuzzy text search ────────────────────────────────────────────────

def find_text_rects_partial(page, search_text: str, full_match: bool = False) -> list[fitz.Rect]:
    """Find text rects with optional fuzzy (word-level) matching."""
    if not search_text:
        return []

    if full_match:
        hits = _page_search(page, search_text)
        return hits if hits else []

    matches: list[fitz.Rect] = []
    seen: set[tuple[float, float, float, float]] = set()

    words = search_text.split()
    for w in _page_words(page):
        word_lower = w[4].lower()
        for search_word in words:
            if search_word.lower() in word_lower or word_lower in search_word.lower():
                rect = fitz.Rect(w[:4])
                rect_key = (rect.x0, rect.y0, rect.x1, rect.y1)
                if rect_key not in seen:
                    matches.append(rect)
                    seen.add(rect_key)
                break
    return matches


# ── Hybrid number + context search ────────────────────────────────────────────

def find_number_with_context(
    page,
    number_str: str,
    context_words: list[str],
) -> Optional[fitz.Rect]:
    """Find *number_str* on the page, then disambiguate by *context_words* on the same line."""
    if not number_str or not context_words:
        return None

    num_rects = _search_number_variations(page, number_str)
    if not num_rects:
        logger.debug(f"  [hybrid] Number not found: {number_str}")
        return None

    logger.debug(f"  [hybrid] Found {len(num_rects)} occurrence(s) of {number_str}, scoring context...")

    words_cache = _page_words(page)
    best_rect: Optional[fitz.Rect] = None
    best_score = -1.0
    best_matches: list[str] = []

    for idx, num_rect in enumerate(num_rects):
        line_words: list[str] = []
        for word_obj in words_cache:
            word_rect = fitz.Rect(word_obj[:4])
            if abs(word_rect.y0 - num_rect.y0) <= CONFIG['y_tolerance']:
                wtxt = (word_obj[4] or "").strip().lower()
                if wtxt:
                    line_words.append(wtxt)

        matched_contexts = [
            ctx for ctx in context_words
            if (ctx or "").lower() and any(
                ctx.lower() in w or w in ctx.lower() for w in line_words
            )
        ]

        frac = len(matched_contexts) / max(len(context_words), 1)
        score = frac
        if _is_heading_like(" ".join(line_words)):
            score -= 0.25

        logger.debug(
            f"  [hybrid] Occurrence #{idx+1}: matched={len(matched_contexts)}/{len(context_words)} score={score:.2f}"
        )

        if score > best_score:
            best_score = score
            best_rect = num_rect
            best_matches = matched_contexts

    if best_rect is not None and best_score >= (1.0 / max(len(context_words), 1)):
        logger.debug(f"  [hybrid] ✓ Best match: {len(best_matches)} context word(s): {best_matches}")
        return best_rect

    logger.debug("  [hybrid] Number found but no sufficiently strong context match")
    return None


# ── Number extraction ──────────────────────────────────────────────────────────

def extract_number_from_text(text: str) -> Optional[str]:
    """Extract the most 'result-like' number from *text* for hybrid anchoring.

    Avoids tiny values (dates, single digits) that produce false matches.
    """
    if not text or not isinstance(text, str):
        return None

    token_re = re.compile(
        r"\b\d[\d,]*(?:\.\d+)?(?:\s*[mk])?\b|\b\d+\s*/\s*\d+\b", re.IGNORECASE
    )
    tokens = token_re.findall(text)
    if not tokens:
        return None

    has_currency = bool(re.search(r"£|\$|gbp|usd|eur", text, flags=re.IGNORECASE))

    def _token_value(tok: str) -> Optional[float]:
        t = (tok or "").strip().lower().replace(",", "").replace(" ", "")
        if not t:
            return None
        if "/" in t and re.fullmatch(r"\d+/\d+", t):
            try:
                n, d = t.split("/", 1)
                den = float(d)
                return float(n) / den if den != 0 else None
            except Exception:
                return None
        mult = 1.0
        if t.endswith("m") and re.fullmatch(r"\d+(?:\.\d+)?m", t):
            mult = 1_000_000.0
            t = t[:-1]
        elif t.endswith("k") and re.fullmatch(r"\d+(?:\.\d+)?k", t):
            mult = 1_000.0
            t = t[:-1]
        try:
            return float(t) * mult
        except Exception:
            return None

    left_text = text.split("=", 1)[0] if "=" in text else text
    pool = token_re.findall(left_text) or tokens

    scored: list[tuple[float, str]] = []
    for raw in pool:
        val = _token_value(raw)
        if val is None:
            continue
        if abs(val) < 1 and "/" not in raw:
            continue
        score = abs(val)
        if "," in raw or re.search(r"\d\s+\d{3}\b", raw):
            score *= 1.15
        if re.search(r"[mk]", raw, flags=re.IGNORECASE):
            score *= 1.10
        if has_currency and abs(val) >= 100:
            score *= 1.05
        scored.append((score, raw.strip()))

    if not scored:
        return None

    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1].replace(" ", "")


# ── Number rect refinement ─────────────────────────────────────────────────────

def find_number_rect_in_text(page, text_rect: fitz.Rect, number_str: str) -> fitz.Rect:
    """Refine a text-level rect to the specific number within it."""
    if not page or not text_rect or not number_str:
        return text_rect

    try:
        search_area = fitz.Rect(
            text_rect.x0,
            text_rect.y0 - 2,
            page.rect.width - 50,
            text_rect.y1 + 2,
        )
        num_hits = _page_search(page, number_str, clip=search_area)
        if num_hits:
            found_rect = num_hits[-1]
            logger.debug(f"      [refined] Number: x={found_rect.x0:.1f} (text was x={text_rect.x0:.1f})")
            return found_rect
    except Exception as e:
        logger.debug(f"      [refined] Error: {e}")

    return text_rect


# ── Main anchor resolution ─────────────────────────────────────────────────────

def resolve_anchor_rect(
    doc,
    anchor_text: str,
    allowed_pages: list[int],
    placed_marks: Optional[set] = None,
    skip_duplicates: bool = True,
    expand_to_line: bool = True,
    page_token_sets: Optional[dict[int, set[str]]] = None,
    use_number_first: bool = True,
    redirect_headings: bool = True,
    min_y_per_page: Optional[dict[int, float]] = None,
    max_y_per_page: Optional[dict[int, float]] = None,
) -> tuple[Optional[fitz.Rect], int]:
    """Find where *anchor_text* lives in the PDF, returning (rect, page_num).

    Tries four strategies in order, respecting per-page Y boundaries:
      1. Hybrid number + context (most reliable for numeric evidence)
      2. Exact phrase match
      3. Cleaned/shortened phrase match
      4. Token-level fuzzy line match
      5. Word-cluster match (last resort)
    """
    if not anchor_text or not isinstance(anchor_text, str):
        return None, -1

    if placed_marks is None:
        placed_marks = set()

    evidence_preview = anchor_text[:50] if len(anchor_text) > 50 else anchor_text
    best_rect, best_page = None, -1

    def _outside_boundary(page_num: int, rect: fitz.Rect) -> bool:
        if min_y_per_page:
            boundary = min_y_per_page.get(page_num)
            if boundary is not None and rect.y0 < boundary:
                return True
        if max_y_per_page:
            boundary = max_y_per_page.get(page_num)
            if boundary is not None and rect.y0 > boundary:
                return True
        return False

    ranked_pages = _rank_pages_for_anchor(page_token_sets or {}, list(allowed_pages), anchor_text)

    def _maybe_redirect(page, chosen_rect: fitz.Rect) -> Optional[fitz.Rect]:
        if not redirect_headings:
            return None
        return _redirect_if_header_like(page, chosen_rect, expand_to_line)

    num = extract_number_from_text(anchor_text) if use_number_first else None
    if num:
        logger.debug(f"    → Number-first approach: searching for '{num}' (extracted from evidence)")

        for page_num in ranked_pages:
            page = doc[page_num - 1]
            context_words = _build_context_words(anchor_text, max_words=6)

            logger.debug(f"    [number-first] Trying hybrid: num={num}, context={context_words[:3]}")
            hybrid_rect = find_number_with_context(page, num, context_words)
            if hybrid_rect and not _outside_boundary(page_num, hybrid_rect):
                mark_key = _line_key(page_num, hybrid_rect.y0)
                if not (skip_duplicates and mark_key in placed_marks):
                    logger.debug(f"    [number-first-hybrid] ✓ Found number with context on page {page_num}")
                    chosen = _expand_rect_to_line(page, hybrid_rect) if expand_to_line else hybrid_rect
                    redirected = _maybe_redirect(page, chosen)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num

            num_hits = _search_number_variations(page, num)
            num_hits = [r for r in num_hits if not _outside_boundary(page_num, r)]
            if len(num_hits) == 1:
                rect = num_hits[0]
                mark_key = _line_key(page_num, rect.y0)
                if not (skip_duplicates and mark_key in placed_marks):
                    logger.debug(f"    [number-first-unique] ✓ Found unique number '{num}' on page {page_num}")
                    chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                    redirected = _maybe_redirect(page, chosen)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num

    # Strategy 1: exact phrase match
    for page_num in ranked_pages:
        page = doc[page_num - 1]
        exact_hits = _page_search(page, anchor_text)
        if exact_hits:
            for rect in exact_hits:
                if _outside_boundary(page_num, rect):
                    continue
                mark_key = _line_key(page_num, rect.y0)
                if skip_duplicates and mark_key in placed_marks:
                    logger.debug(f"    [exact-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                    continue
                if num:
                    refined_rect = find_number_rect_in_text(page, rect, num)
                    if refined_rect and refined_rect.x0 > rect.x0 + 5:
                        rect = refined_rect
                logger.debug(f"    [exact] '{evidence_preview}' → EXACT PHRASE on page {page_num}")
                chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                redirected = _maybe_redirect(page, chosen)
                if redirected is not None:
                    return redirected, page_num
                return chosen, page_num

        # Strategy 2: cleaned phrase
        clean_phrase = clean_anchor_text(anchor_text)
        if clean_phrase and clean_phrase != anchor_text:
            clean_hits = _page_search(page, clean_phrase)
            if clean_hits:
                for rect in clean_hits:
                    if _outside_boundary(page_num, rect):
                        continue
                    mark_key = _line_key(page_num, rect.y0)
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [clean-skip] Rect at y={rect.y0:.1f} already marked, trying next...")
                        continue
                    if num:
                        refined_rect = find_number_rect_in_text(page, rect, num)
                        if refined_rect:
                            rect = refined_rect
                    logger.debug(f"    [clean] '{evidence_preview}' → CLEAN PHRASE on page {page_num}")
                    chosen = _expand_rect_to_line(page, rect) if expand_to_line else rect
                    redirected = _maybe_redirect(page, chosen)
                    if redirected is not None:
                        return redirected, page_num
                    return chosen, page_num

        # Strategy 2b: symbol-insensitive line containment.
        # Rescues anchors that differ from the PDF only by currency/math glyphs,
        # units or separators (£ dropped, "x" vs "×", "/" vs "÷", "14 million"
        # vs "14m", "(1,234)" vs "-1,234"). Canonicalises both sides and requires
        # the anchor to be a contiguous substring of the line — more precise than
        # the token-overlap match below, so it runs first.
        anchor_norm = _normalize_symbols_for_match(anchor_text)
        if len(anchor_norm) >= 6:
            for line_text, line_rect in _iter_page_lines(page):
                if _outside_boundary(page_num, line_rect):
                    continue
                if _is_heading_like(line_text):
                    continue
                if anchor_norm not in _normalize_symbols_for_match(line_text):
                    continue
                mark_key = _line_key(page_num, line_rect.y0)
                if skip_duplicates and mark_key in placed_marks:
                    continue
                logger.debug(
                    f"    [symbol-contain] '{evidence_preview}' → NORMALIZED "
                    f"CONTAINMENT on page {page_num}"
                )
                chosen = _expand_rect_to_line(page, line_rect) if expand_to_line else line_rect
                redirected = _maybe_redirect(page, chosen)
                if redirected is not None:
                    return redirected, page_num
                return chosen, page_num

        # Strategy 3: token-level fuzzy line match
        best_line = _find_best_line_match(page, anchor_text, required_number=num)
        if best_line and not _outside_boundary(page_num, best_line):
            mark_key = _line_key(page_num, best_line.y0)
            if skip_duplicates and mark_key in placed_marks:
                logger.debug(f"    [line-fuzzy-skip] Rect at y={best_line.y0:.1f} already marked, trying next...")
            else:
                logger.debug(f"    [line-fuzzy] '{evidence_preview}' → BEST LINE TOKEN MATCH on page {page_num}")
                chosen = _expand_rect_to_line(page, best_line) if expand_to_line else best_line
                redirected = _maybe_redirect(page, chosen)
                if redirected is not None:
                    return redirected, page_num
                return chosen, page_num

        # Strategy 4: word-by-word clustering (last resort)
        words = anchor_text.split()[:CONFIG['max_anchor_words']]
        rects_by_word: list[list[fitz.Rect]] = []
        for word in words:
            if len(word) > 2:
                hits = find_text_rects_partial(page, word, full_match=False)
                if hits:
                    rects_by_word.append(hits)

        if rects_by_word and len(rects_by_word) >= max(2, len(words) - 2):
            first_matches = [rects[0] for rects in rects_by_word if rects]
            if first_matches:
                first_matches.sort(key=lambda r: r.x0)
                y_vals = [r.y0 for r in first_matches]
                if max(y_vals) - min(y_vals) <= CONFIG['y_tolerance']:
                    combined = first_matches[0]
                    for r in first_matches[1:]:
                        combined = combined | r
                    mark_key = _line_key(page_num, combined.y0)
                    if skip_duplicates and mark_key in placed_marks:
                        logger.debug(f"    [cluster-skip] Rect at y={combined.y0:.1f} already marked, trying next...")
                    else:
                        logger.debug(f"    [cluster] '{evidence_preview}' → WORD CLUSTER on page {page_num}")
                        chosen = _expand_rect_to_line(page, combined) if expand_to_line else combined
                        redirected = _maybe_redirect(page, chosen)
                        if redirected is not None:
                            return redirected, page_num
                        return chosen, page_num

            if not best_rect:
                best_rect, best_page = first_matches[0], page_num
                logger.debug(f"    [fallback] Keeping word cluster as fallback on page {page_num}")

    if best_rect and best_page != -1 and not _outside_boundary(best_page, best_rect):
        mark_key = _line_key(best_page, best_rect.y0)
        if not (skip_duplicates and mark_key in placed_marks):
            logger.debug(f"    [fallback-used] Anchor matched via fallback on page {best_page}")
            return best_rect, best_page

    logger.debug(f"    [FAILED] Could not match: '{evidence_preview}'")
    return None, -1
