# ==================== TEXT NORMALIZATION & PARSING UTILITIES ====================

import re
from typing import Optional

from .annotator_config import STOPWORDS


def _strip_llm_artifacts(text: str) -> str:
    """Remove common LLM-generated noise from a string."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = text
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\[\.\.\.\]|\\n|\\\n", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _line_key(page_num: int, y: float, granularity: float = 2.0) -> tuple[int, int]:
    """Quantize a y-position to a stable per-line key.

    Using raw float y0 is too unstable — quantizing avoids both duplicate marks
    on the same line and accidental de-duplication misses.
    """
    try:
        yy = float(y)
    except Exception:
        yy = 0.0
    return int(page_num), int(yy // float(granularity))


def _normalize_text_for_match(text: str) -> str:
    """Lowercase, strip artifacts, and normalise symbols for fuzzy comparison."""
    if not text or not isinstance(text, str):
        return ""
    cleaned = _strip_llm_artifacts(text)
    cleaned = cleaned.replace("×", "x")
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def _normalize_symbols_for_match(text: str) -> str:
    """Canonicalize math/currency glyphs, units and separators for tolerant
    substring matching.

    The LLM and the student PDF frequently pick different surface forms for the
    same value, which defeats a literal search. Folding BOTH sides the same way
    lets a containment check still locate the line instead of dropping it:
      • ×/✕/⨯ folded to ascii "x"       (model writes "x", PDF renders "×")
      • ÷ folded to "/"                 (division glyph vs slash)
      • £/$/€ made optional            (model drops or adds the symbol)
      • –/—/− unified to plain "-"      (dash / minus glyph mismatch)
      • "(1,234)" ≡ "-1,234" ≡ "1,234"  (accounting negatives: parens & minus dropped)
      • 14 million ≡ 14m, 5 thousand ≡ 5k, 2 billion ≡ 2bn  (magnitude words)
      • 1,250,000 ≡ 1 250 000 ≡ 1250000 (thousands separators)

    This is the shared engine behind both the numerical Tier-2 line-scan and the
    holistic symbol-insensitive containment strategy.
    """
    s = (text or "").replace("`", " ").lower()
    s = re.sub(r"[×✕⨯]", "x", s)
    s = re.sub(r"[–—−]", "-", s)
    s = s.replace("÷", "/")
    s = re.sub(r"[£$€]", " ", s)
    s = re.sub(r"[()\-]", " ", s)
    s = s.replace(",", "")
    s = re.sub(r"\bmillions?\b|\bmn\b", "m", s)
    s = re.sub(r"\bbillions?\b", "bn", s)
    s = re.sub(r"\bthousands?\b", "k", s)
    s = re.sub(r"(?<=\d)\s+(?=(?:bn|[mk])\b)", "", s)
    s = re.sub(r"(?<=\d)\s+(?=\d)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _split_comment_arrow(comment: str) -> Optional[tuple[str, str]]:
    """Split 'anchor → feedback' comment into (anchor, feedback) tuple."""
    if not comment or not isinstance(comment, str):
        return None
    if '→' in comment:
        left, right = comment.split('→', 1)
    elif '->' in comment:
        left, right = comment.split('->', 1)
    else:
        return None
    anchor_part = left.strip().strip('"\'')
    feedback_part = right.strip().strip('"\'')
    if not anchor_part or len(anchor_part) < 3:
        return None
    if not feedback_part:
        return None
    return anchor_part, feedback_part


def _build_anchor_variations(text: str) -> list[str]:
    """Build a small list of surface-form variants for an anchor string.

    Handles symbol/code interchangeability that arises from LLM paraphrasing
    vs. the actual PDF text:
      • percent / per cent / %
      • GBP / £
      • USD / $
    """
    if not text or not isinstance(text, str):
        return []

    base = re.sub(r"\s+", " ", text.replace("|", " ")).strip()
    if not base:
        return []

    variants: list[str] = [base]
    norm = _normalize_text_for_match(base)

    if "percent" in norm or "per cent" in norm:
        v = re.sub(r"\bper\s*cent\b", "%", base, flags=re.IGNORECASE)
        v = re.sub(r"\bpercent\b", "%", v, flags=re.IGNORECASE)
        v = re.sub(r"\s*%\s*", "%", v)
        variants.append(v)

    if "%" in base:
        variants.append(re.sub(r"%", " percent", base))
        variants.append(re.sub(r"%", " per cent", base))

    # GBP <-> £ — LLM evidence frequently uses the currency code "GBP"
    # while student PDFs render the £ symbol (or vice versa).
    if re.search(r"\bGBP\s*(?=\d)", base):
        variants.append(re.sub(r"\bGBP\s*(?=\d)", "£", base))
    if re.search(r"£\s*(?=\d)", base):
        variants.append(re.sub(r"£\s*(?=\d)", "GBP", base))

    # USD <-> $ — same pattern for US dollar evidence.
    if re.search(r"\bUSD\s*(?=\d)", base):
        variants.append(re.sub(r"\bUSD\s*(?=\d)", "$", base))
    if re.search(r"\$\s*(?=\d)", base):
        variants.append(re.sub(r"\$\s*(?=\d)", "USD", base))

    # Currency-symbol-STRIPPED variant. The LLM frequently drops or adds a
    # leading currency symbol (£/$/€) relative to what the student PDF renders.
    # Removing the symbol lets the literal search still hit the number, since
    # search_for matches "3,850" as a substring of "£3,850" (and vice-versa).
    if re.search(r"[£$€]", base):
        stripped = re.sub(r"\s*[£$€]\s*", " ", base)
        variants.append(re.sub(r"\s+", " ", stripped).strip())

    # Dash / minus unification. The LLM interchanges hyphen-minus, en-dash,
    # em-dash and the Unicode minus with whatever glyph the PDF uses. Emit a
    # variant with all of them collapsed to a plain hyphen-minus.
    if re.search(r"[–—−]", base):
        variants.append(re.sub(r"[–—−]", "-", base))

    # Multiplication-sign variants. The LLM commonly writes "175,000 x £80"
    # with an ascii "x" while the PDF renders the Unicode "×" (or vice-versa),
    # which blocks the literal search. Emit both surface forms of a lone,
    # whitespace-surrounded multiplication operator (never touches the "x"
    # inside words like "tax" or "box").
    if re.search(r"[×✕⨯]", base):
        variants.append(re.sub(r"[×✕⨯]", "x", base))
    if re.search(r"(?<=\s)[xX](?=\s)", base):
        variants.append(re.sub(r"(?<=\s)[xX](?=\s)", "×", base))

    # Division-sign variants — "÷" (U+00F7) vs a plain slash "/".
    if "÷" in base:
        variants.append(base.replace("÷", "/"))
    if re.search(r"(?<=\s)/(?=\s)", base):
        variants.append(re.sub(r"(?<=\s)/(?=\s)", "÷", base))

    # Units / magnitude variants — the LLM and the PDF disagree on whether a
    # magnitude is spelled out ("14 million") or abbreviated ("14m"). Emit both
    # the contracted and expanded surface forms so the literal search matches
    # either. Only fires on a magnitude word/letter attached to a number, so
    # ordinary prose is untouched.
    _units = [(r"millions?|mn", "m", "million"),
              (r"billions?", "bn", "billion"),
              (r"thousands?", "k", "thousand")]
    contracted = base
    for word_re, short, _long in _units:
        contracted = re.sub(
            rf"(\d[\d.,]*)\s*(?:{word_re})\b", rf"\g<1>{short}",
            contracted, flags=re.IGNORECASE,
        )
    if contracted != base:
        variants.append(contracted)
    expanded = base
    for _word_re, short, long in _units:
        expanded = re.sub(
            rf"(\d[\d.,]*)\s*{short}\b", rf"\g<1> {long}",
            expanded, flags=re.IGNORECASE,
        )
    if expanded != base:
        variants.append(expanded)

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        vv = re.sub(r"\s+", " ", v).strip()
        if not vv:
            continue
        key = vv.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(vv)
    return out


def _tokenize(text: str) -> list[str]:
    """Extract meaningful tokens for overlap-based matching.

    Strips stopwords and very short tokens; keeps alphanumerics plus a few
    accounting symbols (£, $, %, etc.).
    """
    norm = _normalize_text_for_match(text)
    norm = re.sub(r"[^a-z0-9£$%.,/()\- ]+", " ", norm)
    tokens = [t for t in norm.split() if len(t) > 2 and t not in STOPWORDS]
    return tokens


def _build_candidate_fragments(evidence_text: str) -> list[str]:
    """Split an evidence string into ranked anchor fragments for PDF matching.

    Long evidence strings often contain multiple ';'- or newline-separated
    snippets.  We return them sorted longest-first (more distinctive) and
    de-duplicated, up to 4 candidates.  Falls back to the whole string when
    no valid parts are found.

    DRY helper shared by place_score_near_anchor and the holistic annotation loop.
    """
    parts = [p.strip() for p in re.split(r"\n|;|\|", evidence_text) if p and p.strip()]
    parts = [p for p in parts if len(p) >= 6]
    seen: set[str] = set()
    unique: list[str] = []
    for p in sorted(parts, key=len, reverse=True):
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique[:4] if unique else [evidence_text]
