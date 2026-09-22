import json
import re
import os
import ast
import math
import traceback
from datetime import datetime
from typing import Optional, Any, Tuple
from bson import ObjectId
from pydantic import BaseModel, Field
from prompts.grading_prompts import (
    grade_prompt, holistic_grade_prompt, restatement_prompt,
)
from llm_setup import llm_grader
from logging_config import logger
from schemas.student_grades import StudentGradeDocument, RestatementResponse
from database.mongodb import get_collection
from errors import GradingError, classify_error


class NotRequiredPoint(BaseModel):
    text: str = Field(..., description="Verbatim line/sentence from student that is off-topic / not required")
    key_phrase: str = Field("", description="3-6 word verbatim anchor within text where the 'Not required' marker is placed")
    reason: str = Field("", description="One-sentence reason this content is off-topic or not asked for")


class CorrectPoint(BaseModel):
    text: str = Field(..., description="Verbatim line/sentence from student that earned marks")
    marks: float = Field(..., ge=0, description="Marks this specific point earned (0.5 increments)")
    key_phrase: str = Field("", description="2-5 word core concept within text where tick mark is placed")


class LLMGradingBreakdownItem(BaseModel):
    criterion: str = Field(..., description="Exact criterion description from model marking_criteria")
    marks_awarded: float = Field(..., ge=0, description="Marks awarded for this criterion")
    max_possible: float = Field(..., ge=0, description="Maximum marks possible for this criterion")
    reason: str = Field("", description="Brief reason for award")
    evidence: list[str] = Field(default_factory=list, description="1-3 verbatim quotes from student answer")
    comments_summary: Optional[str] = Field("", description="Optional short note")
    criterion_focus: str = Field(
        "",
        description=(
            "REQUIRED. At most 8 words naming what THIS criterion tests, taken from its "
            "wording BEFORE any 'DISAMBIGUATION' or 'CONTEXT' section. A self-check on "
            "the criterion_id beside it. Name the ROLE, not the figure: a section often "
            "holds two criteria quoting the SAME amount in opposite roles (a gain earned "
            "vs that amount later eliminated; goodwill remeasured vs the exchange "
            "movement to OCI), and those are the pairs that get mispaired. "
            "e.g. 'revaluation gain to OCI', 'goodwill remeasured at closing rate'."
        ),
    )
    column_header: Optional[str] = Field(
        None,
        description=(
            "For tabular data ONLY (e.g. a `disposal date | acq date | post acq` row) "
            "where the target VALUE appears in multiple columns of the row: the exact "
            "column-header text (as it appears in the student's PDF, e.g. 'acq date' "
            "or 'disposal date') that disambiguates WHICH column your target lives in. "
            "The annotator uses this to place the underline on the correct column. "
            "Leave null for non-tabular criteria or when the target value is unique on the row."
        ),
    )


class LLMGradingItem(BaseModel):
    question_number: str = Field(..., description="Question number being graded")
    score: float = Field(..., ge=0, description="Total marks awarded")
    total_marks: float = Field(..., ge=0, description="Maximum marks for this question")
    comments: list[str] = Field(default_factory=list, description="Feedback comments")
    correct_words: list[str] = Field(default_factory=list, description="Verbatim correct phrases")
    breakdown: list[LLMGradingBreakdownItem] = Field(default_factory=list, description="Per-criterion breakdown")
    not_required_points: list[NotRequiredPoint] = Field(
        default_factory=list,
        description="Off-topic / irrelevant content flagged for the student (no marks impact)",
    )


class LLMGradingResponse(BaseModel):
    grades: list[LLMGradingItem] = Field(..., min_length=1, description="Grades array")


# ── Holistic grading models (no-criteria theoretical questions) ──


class HolisticSubQuestionGrade(BaseModel):
    sub_question: str = Field(..., description="Sub-question identifier from model answer, e.g. 'a', '1.1', '(i)'")
    student_label: str = Field("", description="How the student labeled this part, e.g. 'Q1(a)', 'a)', 'Part a'")
    marks_awarded: float = Field(..., ge=0, description="Marks awarded for this sub-question")
    max_marks: float = Field(..., ge=0, description="Maximum marks for this sub-question")
    reason: str = Field("", description="Brief explanation of why marks were awarded/not awarded")
    correct_points: list[CorrectPoint] = Field(default_factory=list, description="Correct points with per-point marks")
    not_required_points: list[NotRequiredPoint] = Field(default_factory=list, description="Off-topic / irrelevant content that earns no marks but is flagged for the student")


class HolisticGradingResponse(BaseModel):
    question_number: str = Field(..., description="Main question number")
    score: float = Field(..., ge=0, description="Total marks awarded")
    total_marks: float = Field(..., ge=0, description="Maximum marks for the entire question")
    sub_grades: list[HolisticSubQuestionGrade] = Field(default_factory=list, description="Per-sub-question grades")
    comments: list[str] = Field(default_factory=list, description="Feedback comments")


# ── Helpers for the parent-calc verification guard ─────────────────────────────

_YEAR_LIKE_RE = re.compile(r"^(19|20)\d{2}$")


def _extract_parent_calc_result(desc: str) -> str | None:
    """Extract the RESULT of a criterion's parent working, if declared.

    Recognises quoted working references embedded in the criterion description:
      • "From the working '25% × £7.2m × 9/12 = 1,350'."     → "1350"
      • "in the 'Less net assets ... = (18,400)' line"        → "18400"
      • "'25% × (£12.75m − £2.75m) = 2,500'"                  → "2500"

    Only the LAST number after the "= " inside single-quoted text is treated
    as the parent result. Commas are stripped. Currency symbols (£/$/€) and a
    trailing scale suffix (m / k) are stripped from the surrounding tokens
    but do NOT scale the returned digits - the rubric writes the numeric
    result plainly (1,350 / 18,400) so we match that form directly in
    student evidence.

    Returns None when no parent-working reference is present - the criterion
    is not verifiable under this rule (leave it alone).
    """
    if not desc:
        return None
    # Prefer the rightmost = inside a quoted span so nested parentheticals
    # like "(£0.25m + £12.75m + £7.2m × 9/12) = (18,400)" pick 18,400.
    for m in re.finditer(
        r"'[^']*=\s*\(?\s*[£$€]?\s*([\d,]+(?:\.\d+)?)\s*[mkbMKB]?\s*\)?\s*'",
        desc,
    ):
        pass  # keep the LAST match
    last: Optional[re.Match] = None
    for m in re.finditer(
        r"'[^']*=\s*\(?\s*[£$€]?\s*([\d,]+(?:\.\d+)?)\s*[mkbMKB]?\s*\)?\s*'",
        desc,
    ):
        last = m
    if last is not None:
        return last.group(1).replace(",", "")
    return None


def _apply_parent_calc_verification(
    normalized_breakdown: list[dict],
    of_source_ids_map: Optional[dict] = None,
    of_component_of_map: Optional[dict] = None,
    of_definitions: Optional[dict] = None,
) -> float:
    """Context-aware crediting via parent-calculation verification.

    For each criterion with marks > 0 that declares a parent working
    (`'X × Y × Z = R'`), check whether the student's evidence contains R -
    the numeric RESULT the working is supposed to produce. If R is absent,
    the student did not perform this specific calculation; revoke the mark.

    Returns total marks revoked.

    This replaces the older "one working = one credit" dedup rule which was
    over-eager: it revoked whenever two criteria SHARED evidence, even if the
    same working legitimately serves both. The parent-calc rule is strictly
    context-driven - it revokes only when the student's evidence proves the
    student did not do THIS specific calculation.

    Criteria without a quoted parent-working reference are skipped (their
    credit stands on the LLM's original judgement).
    """
    revoked_total = 0.0
    for bd in normalized_breakdown:
        try:
            awarded = float(bd.get("marks_awarded", 0) or 0)
        except (TypeError, ValueError):
            continue
        if awarded <= 0:
            continue
        crit = str(bd.get("criterion", "") or "")
        # OWN-FIGURE EXEMPTION. A criterion carrying `of_source_ids` depends on
        # an upstream figure the student is allowed to get wrong and carry
        # forward. Verifying such a criterion against the MODEL's parent result
        # is exactly backwards: the whole point of OF marking is that the
        # student's result legitimately differs. Without this, a student who
        # applies the right method to their own earlier (wrong) number is
        # revoked for the method mark they earned.
        if of_source_ids_map and of_source_ids_map.get(crit):
            continue
        parent_result = _extract_parent_calc_result(crit)
        if not parent_result:
            continue  # no parent working declared - not verifiable

        # COMPONENT EXEMPTION. When the quoted "parent working" is the very
        # TOTAL this criterion is a component of, requiring that total is
        # backwards: the criterion marks one ROW of the working, and the
        # examiner marks that row on its own. The £12.75m component of net
        # assets at disposal quotes the (18,400) total for context, and a
        # student who wrote "Retained earnings b/fwd 12,750" - which earned
        # half a mark from the examiner - was scoring zero because 18,400
        # never appeared. Sub-workings whose quoted result is their OWN output
        # (25% x £7.2m x 9/12 = 1,350) are untouched: there the result IS what
        # the criterion tests.
        _aggregate = (of_component_of_map or {}).get(crit)
        if _aggregate:
            _agg_val = ((of_definitions or {}).get(_aggregate) or {}).get("value")
            try:
                if _agg_val is not None and abs(float(_agg_val)) == abs(
                    float(parent_result)
                ):
                    continue
            except (TypeError, ValueError):
                pass
        ev_blob = " ".join(
            str(e) for e in (bd.get("evidence_list") or []) if e
        )
        if not ev_blob:
            continue  # no evidence to verify against - leave alone

        # Match strategies: literal, comma-formatted, and comma-free scans.
        variants: set[str] = {parent_result}
        try:
            pr_int = int(float(parent_result))
            variants.add(str(pr_int))
            variants.add(f"{pr_int:,}")
        except (ValueError, OverflowError):
            pass
        ev_digits_only = re.sub(r"[,\s]+", "", ev_blob)
        if any(v in ev_blob for v in variants) or parent_result in ev_digits_only:
            continue  # student's evidence produces the parent result

        # SCALE-AWARE SECOND PASS.
        # The variants above are literal only: {"12000", "12,000"}. Rubrics are
        # written in £'000 but students routinely answer in millions, so a
        # student who wrote "Cost: 375,000 shares x £32 = £12m" was judged not
        # to have produced 12,000 and lost a mark the grader had already said
        # was correct. _value_present_in_text knows the scale forms ("£12m",
        # "12 million", "12,000,000") and carries the short-bare-token guard,
        # so reuse it rather than widening the literal set by hand.
        try:
            if _value_present_in_text(int(float(parent_result)), ev_blob):
                continue
        except (ValueError, OverflowError):
            pass

        # Parent result absent from evidence → student did not perform this
        # specific calculation. Revoke and record the reason.
        bd["marks_awarded"] = 0.0
        # Tag the revoke so the aggregate-recovery pass (which runs AFTER this
        # guard) cannot silently hand the mark straight back. Without the tag
        # the two passes contradicted each other inside a single reason string:
        # "Aggregate recovery (+0.25) ... Marks revoked (parent-calc
        # verification): this criterion tests the working that produces 18400,
        # which does not appear in the student's evidence".
        bd["_parent_calc_revoked"] = True
        revoked_total += awarded
        prev = (bd.get("reason", "") or "").strip()
        bd["reason"] = (
            f"Marks revoked (parent-calc verification): this criterion tests "
            f"the working that produces {parent_result}, which does not appear "
            f"in the student's evidence - student used the same input numbers "
            f"in a different calculation. " + prev
        ).strip()
    return revoked_total


# Phrases a grader writes when it has established the student did NOT do
# something. An award carrying one of these in its own reason is self-
# contradicting.
_ABSENCE_REASON_RE = re.compile(
    r"\b(?:no\s+\w+.{0,40}?(?:present|shown|given|calculated|provided)"
    r"|not\s+(?:present|shown|stated|provided|calculated|performed|attempted)"
    r"|does\s+not\s+appear"
    r"|never\s+(?:shown|stated|calculated)"
    r"|absent\s+from)",
    re.IGNORECASE,
)


def _reject_contradictory_awards(normalized_breakdown: list[dict]) -> float:
    """Revoke marks whose own reason says the student didn't do the thing.

    A grader that writes "No 9-month NCI share working present" and awards the
    mark anyway has contradicted itself, and the mark is not defensible to a
    marker reading the breakdown.

    Two exemptions, both load-bearing:
      • PARTIAL awards. A criterion worth 1 mark for two things ("0.5 for the
        equity method and 0.5 for the £630k figure") is correctly scored 0.5
        with a reason noting the half that is missing. The absence phrase
        describes the UNAWARDED half, so it is not a contradiction. Only a
        FULL-credit award can contradict a finding of absence.
      • Aggregate-recovery awards, whose reason legitimately quotes the
        original per-criterion finding of absence and then explains that the
        component was absorbed into a rolled-up figure.

    Returns total marks revoked.
    """
    revoked = 0.0
    for bd in normalized_breakdown:
        try:
            awarded = float(bd.get("marks_awarded", 0) or 0)
            maxp = float(bd.get("max_possible", 0) or 0)
        except (TypeError, ValueError):
            continue
        if awarded <= 0:
            continue
        if maxp <= 0 or awarded < maxp:
            continue  # partial award — the absence describes the missing half
        reason = str(bd.get("reason", "") or "")
        if "aggregate recovery" in reason.lower():
            continue  # recovery explains the absence; not a contradiction
        if not _ABSENCE_REASON_RE.search(reason):
            continue
        bd["marks_awarded"] = 0.0
        revoked += awarded
        bd["reason"] = (
            "Marks revoked (self-contradicting reason): the stated reason "
            "asserts the student did not produce this point. " + reason
        ).strip()
    return revoked


# Currency / spacing that may sit between a sign marker and the digits.
_SIGN_PREFIX_CHARS = " \t£$€"

# An explicit "this is being taken away" label immediately before a figure.
_DEDUCTION_LABEL_RE = re.compile(r"\b(?:less|deduct|minus)\b[^0-9]{0,24}$", re.IGNORECASE)


def _value_is_explicit_deduction(magnitude: int, text: str) -> bool:
    """True only when the student unmistakably wrote `magnitude` as a deduction.

    Deliberately narrow. A rubric's `expected_amount` sign records the MODEL
    ANSWER's presentation, and students legitimately present the same working
    without it — a columnar working writes "net assets b/f 8,000,000 1.6
    5,000,000" where the row label carries the sign, and a journal writes
    "Cr revaluation reserve -300,000" where the minus is the Cr side, not a
    negative amount. Treating either as a sign error revoked correct marks.

    So only two spellings count, and only when they wrap THIS figure:
      • parentheses around the number itself — "(3,125)"
      • a Less / Deduct / Minus label immediately preceding it
    A bare minus is ignored: it is far more often an arithmetic operator
    ("2,000,000-1,875,000") or a Dr/Cr direction marker than a negation.
    """
    if not text:
        return False
    for v in _value_variants_for_search(magnitude):
        if _SHORT_BARE_TOKEN_RE.match(v):
            continue  # too unspecific to attribute a sign to
        for m in re.finditer(r"(?<![\d.,])" + re.escape(v) + r"(?!\d)", text):
            head = text[: m.start()].rstrip(_SIGN_PREFIX_CHARS)
            tail = text[m.end():].lstrip(_SIGN_PREFIX_CHARS)
            if head.endswith("(") and tail[:1] == ")":
                return True
            if _DEDUCTION_LABEL_RE.search(head):
                return True
    return False


def _apply_sign_verification(
    normalized_breakdown: list[dict],
    expected_amount_map: dict[str, float],
    sign_sensitive_criteria: set[str],
) -> float:
    """Revoke awards where the student wrote the right figure with the wrong sign.

    The rubric has always carried `expected_amount` and `sign_sensitive`, but
    nothing read them. They are what separates two criteria that quote the same
    number in opposite roles: the Mission Mouldings goodwill working ADDS NCI at
    acquisition (+3,125) while the net-assets-at-disposal working DEDUCTS it
    ((3,125)). A student who wrote 3,125 once, as a deduction, was credited for
    both — the disambiguation prose in the rubric asked the grader not to, but
    prose is not enforcement.

    One-directional by design. It fires only for a criterion whose figure is an
    ADDITION that the student unmistakably wrote as a DEDUCTION - the direction
    that proves the figure was reused from another working. The reverse (rubric
    shows a deduction, student writes a bare positive) is presentation, not
    error, and is left alone; enforcing it symmetrically revoked seven correct
    marks on the scripts this was regression-tested against.

    Own-figure answers are untouched: a student carrying a different magnitude
    forward never matches the expected magnitude, so the check never fires.

    Returns total marks revoked.
    """
    revoked = 0.0
    for bd in normalized_breakdown:
        try:
            awarded = float(bd.get("marks_awarded", 0) or 0)
        except (TypeError, ValueError):
            continue
        if awarded <= 0:
            continue
        crit = str(bd.get("criterion", "") or "")
        if crit not in sign_sensitive_criteria:
            continue
        expected = expected_amount_map.get(crit)
        # Only the addition direction is checkable — see the docstring.
        if expected is None or expected <= 0:
            continue
        ev_blob = " ".join(str(e) for e in (bd.get("evidence_list") or []) if e)
        if not ev_blob:
            continue
        try:
            magnitude = int(round(abs(float(expected))))
        except (TypeError, ValueError, OverflowError):
            continue
        if not _value_is_explicit_deduction(magnitude, ev_blob):
            continue
        bd["marks_awarded"] = 0.0
        revoked += awarded
        prev = (bd.get("reason", "") or "").strip()
        bd["reason"] = (
            f"Marks revoked (sign verification): this criterion tests "
            f"{magnitude:,} as an addition, but the student's evidence writes "
            f"it as a deduction - the same figure used in a different "
            f"working. " + prev
        ).strip()
    return revoked


# Everything from here on in a criterion description is cross-reference, not
# subject matter: DISAMBIGUATION names the criteria this one must NOT be
# confused with, and CONTEXT reproduces the whole surrounding working. Both
# mention the neighbours by name, so matching against them tells you nothing
# about which criterion you are looking at.
_CRITERION_HEAD_RE = re.compile(
    r"DISAMBIGUATION|CONTEXT|TUTORIAL NOTE|ACCEPT EITHER|NOT A DUPLICATE"
    r"|AWARD ALSO|TWO-METHOD NOTE|ONE JOURNAL ENTRY|NO FURTHER MARKS",
)

_FOCUS_STOPWORDS = frozenset({
    "the", "a", "an", "to", "of", "at", "in", "for", "and", "or", "on", "is",
    "as", "by", "its", "it", "this", "that", "mark", "marks", "row", "line",
})


def _criterion_head(description: str) -> str:
    """The part of a criterion description that states what IT tests."""
    return _CRITERION_HEAD_RE.split(str(description or ""), maxsplit=1)[0]


def _focus_tokens(phrase: str) -> set:
    """Content words of a focus phrase, lowercased."""
    return {
        t.lower() for t in re.findall(r"[A-Za-z]{3,}", str(phrase or ""))
        if t.lower() not in _FOCUS_STOPWORDS
    }


def _focus_match_score(focus_tokens: set, head: str) -> float:
    """Share of the focus phrase's content words present in `head`.

    Prefix-tolerant, so "remeasured" matches "remeasurement" and "eliminated"
    matches "eliminates" - the model paraphrases the criterion rather than
    quoting it exactly.
    """
    if not focus_tokens:
        return 0.0
    head_tokens = {t.lower() for t in re.findall(r"[A-Za-z]{3,}", head or "")}
    if not head_tokens:
        return 0.0
    hits = 0
    for f in focus_tokens:
        if any(h.startswith(f[:5]) or f.startswith(h[:5]) for h in head_tokens):
            hits += 1
    return hits / len(focus_tokens)


# A focus phrase must miss its own criterion this badly, and match a sibling
# this much better, before the pairing is called into question.
_FOCUS_OWN_MAX = 0.40
_FOCUS_RIVAL_MIN = 0.70


def _detect_criterion_focus_mismatch(
    normalized_breakdown: list[dict],
    criterion_id_by_desc: dict,
    of_component_of_map: dict,
    of_ids_map: dict,
    of_source_ids_map: dict,
) -> list[str]:
    """Report entries whose `criterion_focus` describes a DIFFERENT criterion.

    The grading model returns a criterion_id plus its reasoning, and id
    resolution is exact - so whatever id it names receives the mark, the
    evidence and the tick on the PDF. On criteria that quote the same figure in
    opposite roles it pairs correct reasoning with the wrong id: the
    revaluation GAIN criterion comes back reasoned as the LOSS, the goodwill
    REMEASUREMENT as the OCI movement. Marks often net out, but the annotation
    and the feedback land on the wrong point.

    `criterion_focus` is the model's own one-line statement of what it just
    graded. Checking it against the criterion's head - the wording before
    DISAMBIGUATION/CONTEXT, which is the only part that describes THIS
    criterion rather than its neighbours - catches the mispairing without
    depending on the rubric's keyword lists, which are numeric and too sparse
    to identify a concept.

    Reports only; never changes a mark.
    """
    heads: dict[str, str] = {}
    groups: dict[str, str] = {}
    for desc, cid in (criterion_id_by_desc or {}).items():
        heads[desc] = _criterion_head(desc)
        groups[desc] = _criterion_group_key(
            desc, criterion_id_by_desc, of_component_of_map,
            of_ids_map, of_source_ids_map,
        )

    warnings: list[str] = []
    for bd in normalized_breakdown:
        focus = str(bd.get("criterion_focus", "") or "").strip()
        if not focus:
            continue  # model did not supply one — nothing to verify
        desc = str(bd.get("criterion", "") or "")
        if desc not in heads:
            continue
        ftok = _focus_tokens(focus)
        if len(ftok) < 2:
            continue  # too vague to judge either way
        group = groups.get(desc, "")
        own_head = heads[desc]
        # Compare PAIRWISE, on the words that tell that pair apart. Two criteria
        # marking the same working share most of their wording - both the
        # revaluation gain and the revaluation loss are introduced as a
        # "BALANCING FIGURE" - and those shared words drown out the single word
        # that decides which is which. For each rival, keep only the focus words
        # the two heads disagree about.
        own, best, best_desc = 1.0, 0.0, None
        for other_desc, other_head in heads.items():
            if other_desc == desc or groups.get(other_desc, "") != group:
                continue
            disc = {
                f for f in ftok
                if (_focus_match_score({f}, own_head) > 0)
                != (_focus_match_score({f}, other_head) > 0)
            }
            if not disc:
                continue  # indistinguishable on this phrase — nothing to say
            o = _focus_match_score(disc, own_head)
            r = _focus_match_score(disc, other_head)
            if r > best:
                own, best, best_desc = o, r, other_desc
        if best_desc is None or own > _FOCUS_OWN_MAX or best < _FOCUS_RIVAL_MIN:
            continue
        # Detection is pairwise on discriminating words; the SUGGESTION is the
        # sibling the whole phrase fits best, which is the likelier intended
        # criterion when several share the deciding word.
        _suggest, _suggest_score = best_desc, 0.0
        for other_desc, other_head in heads.items():
            if other_desc == desc or groups.get(other_desc, "") != group:
                continue
            sc = _focus_match_score(ftok, other_head)
            if sc > _suggest_score:
                _suggest, _suggest_score = other_desc, sc
        warnings.append(
            f"Criterion-id mismatch: {criterion_id_by_desc.get(desc, '?')} was "
            f"returned with focus \"{focus}\", which does not describe it "
            f"(match {own:.2f}); it looks like "
            f"{criterion_id_by_desc.get(_suggest, '?')} ({_suggest_score:.2f}). "
            f"Marks left unchanged - but this mark, its evidence and its "
            f"annotation may be attached to the wrong criterion."
        )
    return warnings


def _criterion_group_key(
    criterion: str,
    criterion_id_by_desc: dict,
    of_component_of_map: dict,
    of_ids_map: dict,
    of_source_ids_map: dict,
) -> str:
    """Which sub-question a criterion belongs to, as a short key.

    Read from whatever the rubric already provides, most reliable first: the
    criterion id's alphabetic prefix (MM01 -> MM, SOC03 -> SOC), else the
    namespace of any OF id it touches (MM.OF2 -> MM). Returns "" when the rubric
    carries neither, which disables the locality guard for that criterion rather
    than guessing at it.
    """
    cid = str(criterion_id_by_desc.get(criterion, "") or "")
    m = re.match(r"^([A-Za-z]+)", cid)
    if m:
        return m.group(1).upper()
    for src in (of_component_of_map, of_ids_map, of_source_ids_map):
        val = src.get(criterion)
        if not val:
            continue
        ids = [val] if isinstance(val, str) else list(val)
        for oid in ids:
            head = str(oid).split(".", 1)[0].strip()
            if head:
                return head.upper()
    return ""


# A sub-question needs at least this many located awards before its own marks
# are trusted to establish where it lives in the student's script.
_LOCALITY_MIN_SAMPLE = 3


def _apply_section_locality_guard(
    normalized_breakdown: list[dict],
    student_text: str,
    criterion_id_by_desc: dict,
    of_component_of_map: dict,
    of_ids_map: dict,
    of_source_ids_map: dict,
) -> float:
    """Revoke marks awarded from a DIFFERENT sub-question's writing.

    A Mission Mouldings criterion must be earned by what the student wrote under
    Mission Mouldings. It was not: the NCI-at-disposal mark was paid off the
    opening-balance row of the statement of changes in equity, and a Team
    Bauhaus narrative mark off a sentence in the earnings-per-share working.
    Both are whole sub-questions away from the criterion they paid.

    Self-calibrating, so it needs no mapping between rubric topics and whatever
    the student happened to title their sections: each sub-question's own marks
    say where it lives. Take the section holding most of a group's located
    evidence; any award in that group whose evidence sits elsewhere is reading
    another sub-question's answer.

    Deliberately conservative - a group with fewer than three located awards, or
    with no clear majority section, is left alone, as is any script without
    section headers and any criterion whose rubric gives no group key.

    Returns total marks revoked.
    """
    sections = _split_student_sections(student_text)
    if not sections:
        return 0.0

    located: list[tuple[int, str, int]] = []
    for i, bd in enumerate(normalized_breakdown):
        try:
            if float(bd.get("marks_awarded", 0) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        crit = str(bd.get("criterion", "") or "")
        group = _criterion_group_key(
            crit, criterion_id_by_desc, of_component_of_map,
            of_ids_map, of_source_ids_map,
        )
        if not group:
            continue
        idxs = {
            _section_index_for_evidence(sections, str(ev))
            for ev in (bd.get("evidence_list") or [])
        }
        idxs.discard(None)
        if len(idxs) != 1:
            continue  # unlocatable, or straddling sections — not safe to judge
        located.append((i, group, idxs.pop()))

    by_group: dict[str, list[tuple[int, int]]] = {}
    for i, group, sec in located:
        by_group.setdefault(group, []).append((i, sec))

    revoked = 0.0
    for group, items in by_group.items():
        if len(items) < _LOCALITY_MIN_SAMPLE:
            continue
        counts: dict[int, int] = {}
        for _i, sec in items:
            counts[sec] = counts.get(sec, 0) + 1
        home, hits = max(counts.items(), key=lambda kv: kv[1])
        if hits * 2 <= len(items):
            continue  # no clear majority — the group is genuinely spread out
        for i, sec in items:
            if sec == home:
                continue
            bd = normalized_breakdown[i]
            try:
                awarded = float(bd.get("marks_awarded", 0) or 0)
            except (TypeError, ValueError):
                continue
            if awarded <= 0:
                continue
            bd["marks_awarded"] = 0.0
            revoked += awarded
            prev = (bd.get("reason", "") or "").strip()
            bd["reason"] = (
                f"Marks revoked (section locality): this {group} criterion was "
                f"awarded from the student's \"{sections[sec][0]}\" section, "
                f"while {group} is answered under \"{sections[home][0]}\". A "
                f"mark must be earned by the writing under its own "
                f"sub-question. " + prev
            ).strip()
    return revoked


# Canonical column key -> the header wording to look for in the student's table.
# The annotator widens each of these into contiguous word n-grams, so a student
# heading the column "Other Components of Equity" or stacking it over two lines
# still resolves.
_COLUMN_HEADER_TEXT = {
    "retained_earnings": "Retained Earnings",
    "nci": "NCI",
    "share_option_reserve": "Share Option Reserve",
    "other_components": "Other Components",
    "equity_share_capital": "Equity Share Capital",
    "share_premium": "Share Premium",
    "total": "Total",
}


def _canon_column_name(s: str) -> str:
    """Canonical key for a statement column, from a rubric tag or a table header."""
    t = re.sub(r"[^a-z]+", " ", str(s or "").lower()).strip()
    t = re.sub(r"\bpound\b|\b000\b", "", t).strip()
    if not t:
        return ""
    if "non controlling" in t or t == "nci" or t.startswith("nci "):
        return "nci"
    if "retained" in t:
        return "retained_earnings"
    if "share premium" in t or t == "premium":
        return "share_premium"
    if "option" in t:
        return "share_option_reserve"
    if "other component" in t or "other reserve" in t:
        return "other_components"
    if "share capital" in t or "equity share" in t:
        return "equity_share_capital"
    if "total" in t:
        return "total"
    return re.sub(r"\s+", "_", t)


def _parse_table_header(text: str) -> Optional[list[str]]:
    """Canonical column keys from the first pipe-delimited header row in `text`."""
    for line in (text or "").split("\n"):
        if line.count("|") < 3:
            continue
        cells = [c.strip() for c in line.split("|")]
        keys = [_canon_column_name(c) for c in cells]
        known = {
            "retained_earnings", "nci", "share_premium",
            "share_option_reserve", "other_components", "equity_share_capital",
        }
        if len(known & set(keys)) >= 2:
            return keys
    return None


def _apply_column_verification(
    normalized_breakdown: list[dict],
    column_map: dict[str, str],
    expected_amount_map: dict[str, float],
    student_text: str,
) -> float:
    """Revoke awards where the student's figure sits in the wrong table column.

    Statement-of-changes-in-equity marks are column-specific: the same figure
    carries a different meaning in retained earnings than in non-controlling
    interest or other components of equity. The rubric names the column in
    prose only, so the grader matched on the number alone and credited rows
    regardless of which column the student had used.

    Best-effort throughout: no parseable header, no declared column, or no
    identifiable cell leaves the award untouched.

    Returns total marks revoked.
    """
    if not column_map:
        return 0.0
    header = _parse_table_header(student_text)
    if not header:
        return 0.0
    revoked = 0.0
    for bd in normalized_breakdown:
        try:
            awarded = float(bd.get("marks_awarded", 0) or 0)
        except (TypeError, ValueError):
            continue
        if awarded <= 0:
            continue
        crit = str(bd.get("criterion", "") or "")
        want = _canon_column_name(column_map.get(crit, ""))
        if not want or want not in header:
            continue
        expected = expected_amount_map.get(crit)
        if expected is None:
            continue
        try:
            magnitude = int(round(abs(float(expected))))
        except (TypeError, ValueError, OverflowError):
            continue
        want_idx = header.index(want)
        # Look at every evidence line that is a row of this table.
        checked_any = False
        matched = False
        for ev in (bd.get("evidence_list") or []):
            cells = [c.strip() for c in str(ev).split("|")]
            if len(cells) < 2:
                continue
            hit_idxs = [
                i for i, c in enumerate(cells)
                if _value_present_in_text(magnitude, c)
            ]
            if not hit_idxs:
                continue
            checked_any = True
            if want_idx in hit_idxs:
                matched = True
                break
        if not checked_any or matched:
            continue
        bd["marks_awarded"] = 0.0
        revoked += awarded
        prev = (bd.get("reason", "") or "").strip()
        bd["reason"] = (
            f"Marks revoked (column verification): this criterion tests the "
            f"{want.replace('_', ' ')} column, but the student's figure "
            f"{magnitude:,} appears in a different column of the row. " + prev
        ).strip()
    return revoked


_RECOVERY_PER_SUB_MARK = 0.25   # award per un-earned sub-mark

# Recovery cap sizing: the stricter parent-calc verification correctly revokes
# more sub-marks when a student uses an equivalent-method shortcut, so the
# recovery layer needs more room to give the method-equivalent credit back.
# Formula: recover all-but-one un-earned sub-mark, so the student "loses" at
# most one sub-component's worth of credit while their correct aggregate is
# still recognised. Ceiling large enough to cover the 7-sub-mark NCI-at-
# disposal working (recovers up to 6 × 0.25 = 1.5 if 6 are un-earned).
_RECOVERY_HARD_CAP = 1.5


def _recovery_cap_for_group(unearned_count: int) -> float:
    """Cap for aggregate recovery per group.

    Scales with the number of un-earned sub-marks so a working with many
    granular components can still get most sub-marks recovered when the
    student's total is correct.
    """
    if unearned_count <= 1:
        return 0.0
    return min(_RECOVERY_HARD_CAP, (unearned_count - 1) * _RECOVERY_PER_SUB_MARK)


def _value_variants_for_search(value: int) -> set[str]:
    """Numeric variants to look for in the student's raw answer text.

    E.g., 12750 -> {"12750", "12,750", "12,750,000", "12,750,000.00",
    "12.75", "12.75m"} so we catch the same value across scales/formats
    and — crucially for the annotator — the LONG raw form the student
    likely wrote in the PDF ("12,750,000.00" for a rubric value that
    represents £12.75 million).

    Includes:
      • Raw and comma-grouped: "12750", "12,750"
      • ×1000 scaled (rubric convention: value in thousands, PDF shows
        raw amount): "12750000", "12,750,000", "12,750,000.00"
      • Millions abbreviation (only when the value divides cleanly):
        "12.75m", "12.75"
    """
    if not isinstance(value, (int, float)):
        return set()
    abs_v = abs(int(value))
    variants: set[str] = {str(abs_v), f"{abs_v:,}"}
    # ×1000 scaled form — accounting rubrics typically express values in
    # thousands (£000s), while the student PDF renders raw amounts. Adding
    # these variants lets the annotator match the WHOLE cell value
    # ("12,750,000.00") rather than just a substring ("12,750").
    # Skip when the value is already large enough that ×1000 would be
    # unrealistic (10-digit numbers are essentially never accounting
    # figures in a student answer).
    if 0 < abs_v < 10_000_000:
        scaled = abs_v * 1000
        variants.add(str(scaled))
        variants.add(f"{scaled:,}")
        variants.add(f"{scaled:,}.00")
        variants.add(f"{scaled}.00")
    # DECIMAL MILLIONS FORM — for ANY value, not just exact thousands.
    # Rubric values are in £'000 but students routinely answer in millions, and
    # the form they write is usually a decimal: "£3.125m" for 3,125, "12.75m"
    # for 12,750, "0.4m" for 400. Previously the millions form was only
    # generated when the value divided cleanly by 1,000, so every one of those
    # failed to match and the mark was lost — the docstring above claimed
    # "12750 -> 12.75m" but the guard made that unreachable.
    # Emitted both bare ("3.125") and suffixed ("3.125m"); a bare form that is
    # one or two digits is separately gated by _SHORT_BARE_TOKEN_RE.
    if 0 < abs_v < 10_000_000:
        _m = abs_v / 1000.0
        _m_str = f"{_m:g}"          # 3.125 / 12.75 / 5.4 / 0.4 / 12
        variants.add(_m_str)
        variants.add(f"{_m_str}m")
    # Millions form (only when the value divides cleanly)
    if abs_v >= 1000 and abs_v % 1000 == 0:
        thousands = abs_v // 1000
        variants.add(str(thousands))
        variants.add(f"{thousands:,}")
        # 5,400 -> "5.4m" (5,400 / 1,000 = 5.4)
        if abs_v % 100 == 0:
            m_val = abs_v / 1000
            if m_val == int(m_val):
                variants.add(f"{int(m_val)}m")
            else:
                variants.add(f"{m_val}m")
    return variants


# A variant that is just one or two bare digits — the millions abbreviation of
# a thousands value (3,000 -> "3", 13,000 -> "13"). Too unspecific to match on
# its own; see the guard in _value_present_in_text.
_SHORT_BARE_TOKEN_RE = re.compile(r"^\d{1,2}$")

# Above this share of digits (vs letters) a line of student evidence is a
# working or journal line rather than a sentence. Measured on the real cases:
# prose sits at 2-12%, calculation and journal lines at 37-74%.
_PROSE_DIGIT_DENSITY_MAX = 0.35


def _evidence_is_prose(evidence: str) -> bool:
    """True when a piece of cited evidence reads as a SENTENCE rather than a
    working line.

    Used by the duplicate-evidence post-pass: one sentence can legitimately
    earn two different criteria, whereas one arithmetic line represents a
    single step and cannot.
    """
    if not evidence:
        return False
    digits = sum(c.isdigit() for c in evidence)
    letters = sum(c.isalpha() for c in evidence)
    if digits + letters == 0:
        return False
    return (digits / (digits + letters)) <= _PROSE_DIGIT_DENSITY_MAX


def _value_present_in_text(value: int, text: str) -> bool:
    """True if the value appears in *text* under any of its numeric variants,
    bounded to whole-number matches only.

    Uses digit-boundary regex so "2500" doesn't spuriously match inside
    "125000" (a common substring hazard when a smaller sub-working value
    happens to be a suffix of a larger unrelated number). Preceding
    boundary rejects digit / dot / comma (all imply the value is a piece
    of a larger number). Trailing boundary rejects only digits — a
    trailing "." or "," is legitimate ("2,500," or "2,500.00").
    """
    if not text:
        return False
    for v in _value_variants_for_search(value):
        if _SHORT_BARE_TOKEN_RE.match(v):
            # SHORT BARE TOKEN GUARD.
            # _value_variants_for_search abbreviates a thousands value to its
            # millions form, so 3,000 yields "3" and 13,000 yields "13". A
            # naked one- or two-digit token matches almost any accounting
            # script: "3" was found inside "£3.125m" and credited the
            # net-assets-at-acquisition working on an unrelated NCI line, and
            # "13" was found inside "13,913 shares" and credited the
            # net-assets-at-disposal working on the EPS page.
            # Such a token is only meaningful when the student actually wrote
            # it as a millions expression, so require an explicit scale word.
            # The "3m" / "13m" spellings are separate variants and still match
            # on their own.
            pattern = (
                r"(?<![\d.,])" + re.escape(v) + r"\s*(?:m\b|mn\b|million\b)"
            )
        else:
            pattern = r"(?<![\d.,])" + re.escape(v) + r"(?!\d)"
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


# Runs of characters that could form an arithmetic expression. Must start
# with a digit and end with a digit or a closing paren, so prose is skipped.
_EXPR_RUN_RE = re.compile(r"[0-9][0-9,.\s()*/+\-]*[0-9)]")


def _eval_arith_node(node) -> Optional[float]:
    """Evaluate a whitelisted arithmetic AST node, or None if unsupported.

    Deliberately hand-rolled rather than eval()'d: only numeric literals and
    + - * / (plus unary sign) are honoured, so student text can never reach
    an interpreter.
    """
    if isinstance(node, ast.Expression):
        return _eval_arith_node(node.body)
    if isinstance(node, ast.Constant):
        return float(node.value) if isinstance(node.value, (int, float)) else None
    if isinstance(node, ast.UnaryOp):
        val = _eval_arith_node(node.operand)
        if val is None:
            return None
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.UAdd):
            return val
        return None
    if isinstance(node, ast.BinOp):
        left = _eval_arith_node(node.left)
        right = _eval_arith_node(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right else None
        return None
    return None


def _computed_values_in_text(text: str) -> set[int]:
    """Values the student DERIVED via an arithmetic expression.

    _value_present_in_text only finds a value written LITERALLY. A student
    who writes "=12750000+(7200000*9/12)" has demonstrably performed the
    9/12 apportionment, but never writes its result (5,400,000) anywhere in
    the answer — so a literal search misses it and the student scores below
    one who simply wrote the bare figure.

    Every arithmetic run in the text is parsed and EVERY BinOp sub-node is
    evaluated, so intermediate results are captured as well as the final
    one. The example above yields 18,150,000 (the sum), 5,400,000 (the 9/12
    apportionment) and 64,800,000 (the un-divided product).

    Results are returned in BOTH the raw scale and the £000s scale the
    rubric's of_produces values use, so 5,400,000 registers as 5,400.
    """
    if not text:
        return set()
    found: set[int] = set()
    for run in _EXPR_RUN_RE.findall(text):
        expr = run.replace(",", "")
        for candidate in (expr, re.sub(r"\s+", "", expr)):
            try:
                tree = ast.parse(candidate, mode="eval")
            except (SyntaxError, ValueError, MemoryError, RecursionError):
                continue
            for node in ast.walk(tree):
                # Only BinOp nodes — a bare literal is not a "computation"
                # and must not count as demonstrating a working.
                if not isinstance(node, ast.BinOp):
                    continue
                val = _eval_arith_node(node)
                if val is None or not math.isfinite(val):
                    continue
                for scaled in (val, val / 1000.0):
                    if abs(scaled) < 1 or abs(scaled) > 1e12:
                        continue
                    if abs(scaled - round(scaled)) < 0.01:
                        found.add(int(round(scaled)))
            break
    return found


# Journal-line prefix: matches lines the student wrote as journal entries
# ("Debit / Credit / Dr / Cr"). Recovered marks should NOT anchor to these -
# they represent downstream POSTINGS, not the WORKING that produced the value.
_JOURNAL_LINE_RE = re.compile(r"^\s*(debit|credit|dr\b|cr\b)\b", re.IGNORECASE)


def _find_working_line_for_value(value: int, student_text: str) -> Optional[str]:
    """Find a non-journal line in the student's answer that contains `value`.

    Prefers WORKING-area lines (e.g., "total 6,975,000.00" or "share of post
    acq reserves 3,850,000.00") over JOURNAL-entry lines ("debit nci
    6,975,000.00"). Recovered marks anchor here so the visual tick lands on
    the student's working, not on a journal that happens to reference the same
    amount.

    Uses the SAME digit-boundary matching as `_value_present_in_text` so a
    smaller value doesn't spuriously anchor on a line where it only appears
    as a substring of a larger unrelated number. Previously `"25" in "250000"`
    matched, causing an NCI-stake-25% sub-mark to anchor on the share-capital
    line whose value 250 was never about the 25% stake. Regex demands the
    variant sit between non-digit / non-comma / non-dot boundaries on the
    left, and a non-digit boundary on the right — same rule the presence
    check enforces, so the anchor and the presence check stay in sync.
    """
    if not student_text:
        return None
    variants = _value_variants_for_search(value)
    # Pre-compile the boundary-aware pattern per variant. Longest-first so
    # a bare "25" doesn't win over a more-specific "25,000" match on the
    # same line (a bare "25" match is nearly always coincidental noise).
    #
    # SHORT BARE TOKEN GUARD — must mirror _value_present_in_text.
    # _value_variants_for_search abbreviates a thousands value to its millions
    # form, so 8,000 yields "8". Without the guard the anchor search matched
    # the literal "8" inside "Profit to date (8 months) 4,800" and placed a
    # New York Wheels net-assets tick on a Mission Mouldings line. The presence
    # check already required an explicit scale word for such tokens; the anchor
    # search did not, so the two disagreed about what counts as a match.
    patterns = []
    for v in sorted(variants, key=len, reverse=True):
        if _SHORT_BARE_TOKEN_RE.match(v):
            patterns.append(
                re.compile(
                    r"(?<![\d.,])" + re.escape(v) + r"\s*(?:m\b|mn\b|million\b)",
                    re.IGNORECASE,
                )
            )
        else:
            patterns.append(re.compile(r"(?<![\d.,])" + re.escape(v) + r"(?!\d)"))
    for line in student_text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if _JOURNAL_LINE_RE.match(stripped):
            continue
        for pat in patterns:
            if pat.search(stripped):
                return stripped
    return None


# Section header emitted by _format_student_for_prompt: "--- <label> ---".
_STUDENT_SECTION_RE = re.compile(r"^\s*-{3,}\s*(.+?)\s*-{3,}\s*$")


def _split_student_sections(student_text: str) -> list[tuple[str, str]]:
    """Split the flattened student answer into (label, body) sections.

    `_format_student_for_prompt` writes each sub-part as "--- <label> ---"
    followed by that sub-part's verbatim answer, so the boundaries are already
    in the text we search. Anything before the first header becomes a leading
    section labelled "" so no student writing is lost.

    Returns [] when the text carries no headers — callers then fall back to
    whole-text search, preserving behaviour for single-block scripts.
    """
    if not student_text:
        return []
    sections: list[tuple[str, list[str]]] = []
    current_label = ""
    current: list[str] = []
    saw_header = False
    for line in student_text.split("\n"):
        m = _STUDENT_SECTION_RE.match(line)
        if m:
            saw_header = True
            if current or current_label:
                sections.append((current_label, current))
            current_label = m.group(1)
            current = []
        else:
            current.append(line)
    if current or current_label:
        sections.append((current_label, current))
    if not saw_header:
        return []
    return [(lbl, "\n".join(body)) for lbl, body in sections]


# Alphabetic tokens shorter than this are too common to identify a working
# ("of", "at", "the", "nci" is borderline but useful, "net" is not on its own).
_KEYWORD_MIN_TOKEN_LEN = 4

# Tokens that appear on almost every line of an accounting script and so carry
# no evidence about WHICH working a line belongs to.
_KEYWORD_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "less", "add",
    "total", "year", "value", "amount", "figure", "shares", "share",
})


def _keyword_word_tokens(keywords: list[str]) -> set[str]:
    """Alphabetic, discriminating word tokens drawn from a criterion's keywords.

    Numbers are deliberately dropped: the whole point of the gate these feed is
    to require evidence BEYOND a numeric coincidence.
    """
    out: set[str] = set()
    for kw in keywords or []:
        for tok in re.findall(r"[A-Za-z]+", str(kw or "")):
            t = tok.lower()
            if len(t) >= _KEYWORD_MIN_TOKEN_LEN and t not in _KEYWORD_STOPWORDS:
                out.add(t)
    return out


def _line_supports_group_keywords(line: str, wanted: set[str]) -> bool:
    """True when `line` carries at least one of the criteria's word tokens."""
    if not wanted:
        return True
    line_tokens = {t.lower() for t in re.findall(r"[A-Za-z]+", str(line or ""))}
    if not line_tokens:
        return False
    for w in wanted:
        if w in line_tokens:
            return True
        # Tolerate the student's abbreviations ("acq" for "acquisition",
        # "reserves" for "reserve") by accepting a shared prefix of 4+ chars.
        for lt in line_tokens:
            if len(lt) >= _KEYWORD_MIN_TOKEN_LEN and (
                w.startswith(lt) or lt.startswith(w)
            ):
                return True
    return False


def _norm_section_line(s: str) -> str:
    """Normalise a line for cross-referencing evidence against section bodies."""
    return re.sub(r"[\s,;|]+", "", str(s or "")).strip().lower()


def _section_index_for_evidence(
    sections: list[tuple[str, str]], evidence: str
) -> Optional[int]:
    """Index of the section containing `evidence`, or None if not found/ambiguous."""
    needle = _norm_section_line(evidence)
    if not needle or len(needle) < 8:
        # Too short to attribute confidently — a 3-character fragment matches
        # everywhere and would pin the scope to the wrong section.
        return None
    hits = [
        i for i, (_lbl, body) in enumerate(sections)
        if needle in _norm_section_line(body)
    ]
    return hits[0] if len(hits) == 1 else None


def _scope_text_for_group(
    student_text: str,
    sections: list[tuple[str, str]],
    evidence_lines: list[str],
) -> Optional[str]:
    """Narrow the search text to the one student section this working lives in.

    The rubric splinters a single working into many quarter-mark components, and
    the recovery pass below proves a component by finding its value (or a
    subset-sum of sibling values) anywhere in the script. Searching the WHOLE
    script makes that proof worthless across section boundaries: the Mission
    Mouldings NCI-at-disposal group has sub-results {3,125, 2,500, 1,350}, and
    the subset {3,125 + 2,500} sums to 5,625 — which appears in the student's
    statement of changes in equity as the OPENING NCI BALANCE, in a different
    issue entirely. Three MM marks were awarded off that coincidence.

    Scope is derived from where the group's already-credited criteria actually
    point: if every resolvable evidence line sits in one section, search only
    that section.

    Returns None when the script HAS sections but this group's location cannot
    be pinned to one of them - because nothing in the group was credited, or
    its evidence straddles sections. The caller must then skip recovery rather
    than widen the search. Falling back to the whole script in that case is
    what let the Mission Mouldings NCI group, with not one credited component,
    collect a mark off the statement of changes in equity in a different issue:
    the less we know about where a working lives, the less entitled we are to
    go looking for it everywhere.

    Returns the full text unchanged for a script with no section headers at
    all, so single-block papers behave exactly as before.
    """
    if not sections:
        return student_text
    found: set[int] = set()
    for ev in evidence_lines:
        idx = _section_index_for_evidence(sections, ev)
        if idx is not None:
            found.add(idx)
    if len(found) != 1:
        return None
    return sections[found.pop()][1]


def _apply_aggregate_value_recovery(
    normalized_breakdown: list[dict],
    of_component_of_map: dict[str, str],
    of_source_ids_map: dict[str, list[str]],
    of_value_map: dict[str, dict],
    of_definitions: dict[str, dict],
    of_produces_map: dict[str, int] | None = None,
    student_text: str = "",
    keywords_map: dict[str, list[str]] | None = None,
) -> float:
    """Post-LLM aggregate-value recovery via SUBSET-SUM detection.

    When a rubric splits one working into many granular sub-marks and a student
    collapses two or more of those sub-workings into a single aggregated line
    (e.g. writing "share of post acq reserves 3,850" instead of showing 2,500
    and 1,350 separately), the granular sub-marks all evaluate to 0 under a
    strict per-sub-mark rule - yet a real marker would credit the working
    because the student got the answer right via an equivalent decomposition.

    Algorithm per `of_component_of: OFX` group:
      1. For each producer criterion, read `of_produces` (the value produced by
         that criterion's sub-working - e.g. #17/#18/#19 all produce 2,500).
      2. Group producers by their `of_produces` value → sub-workings.
      3. Enumerate every non-empty subset of sub-workings, compute the summed
         value, and check if that sum appears in the student's raw answer
         text.
      4. Confirmed sub-workings = union of every subset whose sum was found.
      5. Award +0.25 recovery to un-earned producer criteria whose sub-working
         is in the confirmed union (up to a cap per group, plus the safeguard
         that at least ONE producer in the group must already be directly
         earned so we're not rewarding lucky-number matches).

    Returns the total marks awarded by the recovery (>= 0).
    """
    if not of_component_of_map or not student_text:
        return 0.0
    of_produces_map = of_produces_map or {}
    keywords_map = keywords_map or {}
    student_sections = _split_student_sections(student_text)

    # Every student line the LLM credited DIRECTLY, mapped to the criterion that
    # claimed it. A size-1 recovery match anchored on a line another criterion
    # already earned is reading someone else's working: the New York Wheels
    # goodwill line "Cost (2M shares * $6) 12,000,000 | 1.60 | 7,500,000" is
    # credited to the cost-of-shares criterion, and it also happens to carry
    # 2,000, the exchange-gain working's average-rate profit result.
    llm_claimed_lines: dict[str, str] = {}
    for _bd in normalized_breakdown:
        try:
            if float(_bd.get("marks_awarded", 0) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            continue
        if "aggregate recovery" in str(_bd.get("reason", "") or "").lower():
            continue
        _owner = str(_bd.get("criterion", "") or "")
        for _ev in (_bd.get("evidence_list") or []):
            _k = _norm_section_line(_ev)
            if len(_k) >= 8:
                llm_claimed_lines.setdefault(_k, _owner)

    # Group producers by their aggregate OF.
    producers_by_of: dict[str, list[int]] = {}
    for idx, bd in enumerate(normalized_breakdown):
        crit = str(bd.get("criterion", "") or "")
        aggregate = of_component_of_map.get(crit)
        if aggregate:
            producers_by_of.setdefault(aggregate, []).append(idx)
    if not producers_by_of:
        return 0.0

    total_recovered = 0.0
    for aggregate_of, producer_indices in producers_by_of.items():
        # Group producer criteria by their sub-working result (`of_produces`).
        # Producers with no `of_produces` marker fall back to a lookup on the
        # aggregate's own value (from `of_definitions`) - this keeps single-
        # producer groups working even when the migration didn't tag them.
        results_to_indices: dict[int, list[int]] = {}
        for idx in producer_indices:
            crit = str(normalized_breakdown[idx].get("criterion", "") or "")
            r = of_produces_map.get(crit)
            if r is None:
                continue
            results_to_indices.setdefault(int(r), []).append(idx)
        if not results_to_indices:
            continue

        # ── Section scoping ──────────────────────────────────────────────────
        # Confine every presence check and anchor search below to the ONE
        # student section this working demonstrably lives in, inferred from
        # where the group's already-credited criteria point. See
        # _scope_text_for_group for why whole-script search is unsafe.
        _scope_evidence: list[str] = []
        for _idx in producer_indices:
            _p = normalized_breakdown[_idx]
            try:
                if float(_p.get("marks_awarded", 0) or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            if "aggregate recovery" in str(_p.get("reason", "") or "").lower():
                continue
            _scope_evidence.extend(
                str(e) for e in (_p.get("evidence_list") or []) if e
            )
        for _cand in normalized_breakdown:
            if aggregate_of not in (
                of_source_ids_map.get(str(_cand.get("criterion", "") or "")) or []
            ):
                continue
            try:
                if float(_cand.get("marks_awarded", 0) or 0) <= 0:
                    continue
            except (TypeError, ValueError):
                continue
            _scope_evidence.extend(
                str(e) for e in (_cand.get("evidence_list") or []) if e
            )
        _scoped = _scope_text_for_group(
            student_text, student_sections, _scope_evidence
        )
        if _scoped is None:
            # Location unknown — see _scope_text_for_group. No recovery.
            continue
        scoped_text: str = _scoped

        # Alphabetic keyword tokens for every producer in this group, indexed by
        # the sub-working value it produces. Used to gate size>=2 subset-sum
        # matches (see _line_supports_group_keywords).
        keywords_by_result: dict[int, set[str]] = {}
        for _r, _idxs in results_to_indices.items():
            _toks: set[str] = set()
            for _i in _idxs:
                _c = str(normalized_breakdown[_i].get("criterion", "") or "")
                _toks.update(_keyword_word_tokens(keywords_map.get(_c) or []))
            keywords_by_result[_r] = _toks

        def _subset_is_supported(subset: list[int]) -> bool:
            """Gate every subset-sum match on the matched line's subject matter.

            A number alone proves nothing about WHICH working it came from, at
            any subset size:
              • size >= 2 — the student's New York Wheels cost line "Cost (2M
                shares * $6) 12,000,000 | 1.60 | 7,500,000" carries 7,500,
                exactly 5,000 + 2,500 from the exchange-gain group, and bought
                0.75 marks for a working never attempted.
              • size 1 — the SAME cost working's "Less: FV of net assets
                (2M * $1) (2,000,000)" carries 2,000, which is the
                average-rate profit translation's result in the exchange-gain
                working, and bought that mark outright. Both lines sit in the
                right SECTION, so section scoping cannot separate them.

            The two sizes need different tests, because they fail differently.

            size >= 2 is pure arithmetic coincidence, so demand subject matter:
            the line must carry a WORD from one of the subset's criteria.

            size 1 cannot use that test. A rubric keyword list is the model
            answer's vocabulary, not the student's - the £12.75m component is
            authored as "reserves" while the student writes "Retained earnings
            b/fwd 12,750", and the examiner awarded that line. Requiring the
            word would revoke it. What the size-1 failures DO share is that the
            number was read off a line another criterion has already been
            credited for, so test that instead.
            """
            line = _find_working_line_for_value(sum(subset), scoped_text)
            if not line:
                return False
            if len(subset) < 2:
                owner = llm_claimed_lines.get(_norm_section_line(line))
                if owner is None:
                    return True
                return owner in {
                    str(normalized_breakdown[i].get("criterion", "") or "")
                    for r in subset
                    for i in results_to_indices.get(r, [])
                }
            wanted: set[str] = set()
            for _r in subset:
                wanted |= keywords_by_result.get(_r, set())
            if not wanted:
                # No alphabetic keywords authored for this group — nothing to
                # verify against, so leave the legacy behaviour untouched.
                return True
            return _line_supports_group_keywords(line, wanted)

        # Enumerate every non-empty subset of sub-working results and mark the
        # sub-workings whose sum appears in the student's answer as confirmed.
        # We also track which sub-workings were confirmed INDIVIDUALLY (subset
        # size 1) - that gives us the engagement safeguard below.
        sub_results = sorted(results_to_indices.keys())
        n = len(sub_results)
        # Track every subset-sum that matches the student's text. matching_subsets
        # is used for the engagement gate; individually_confirmed is used below to
        # differentiate "student wrote this value alone" vs "aggregated with others".
        matching_subsets: list[list[int]] = []
        individually_confirmed: set[int] = set()
        # Cap subset enumeration to keep worst-case complexity bounded. For any
        # realistic exam-rubric group N stays small (≤ 4-5), so 2^N is trivial.
        if n <= 12:
            for mask in range(1, 1 << n):
                subset = [sub_results[i] for i in range(n) if mask & (1 << i)]
                subset_sum = sum(subset)
                if not _value_present_in_text(subset_sum, scoped_text):
                    continue
                if not _subset_is_supported(subset):
                    continue
                matching_subsets.append(subset)
                if len(subset) == 1:
                    individually_confirmed.add(subset[0])
        else:
            # Degenerate fallback: only check each sub-working individually.
            for r in sub_results:
                if _value_present_in_text(r, scoped_text) and _subset_is_supported([r]):
                    matching_subsets.append([r])
                    individually_confirmed.add(r)

        # ── Consumer confirmation (strong engagement signal) ─────────────────
        # If a criterion with of_source_ids: [OFX] is FULLY credited AND its
        # evidence contains OFX's aggregate value, the student demonstrably
        # computed OFX correctly. This is a stronger signal than any subset-sum.
        agg_value = None
        if aggregate_of in of_definitions:
            agg_value = of_definitions[aggregate_of].get("value")
        agg_variants: set[str] = set()
        if agg_value is not None:
            try:
                agg_int = int(round(float(agg_value)))
                agg_variants = _value_variants_for_search(agg_int)
            except (TypeError, ValueError):
                agg_variants = set()

        consumer_confirmed = False
        for cand in normalized_breakdown:
            try:
                _cand_awarded = float(cand.get("marks_awarded", 0) or 0)
                _cand_maxp = float(cand.get("max_possible", 0) or 0)
            except (TypeError, ValueError):
                continue
            if _cand_maxp <= 0 or _cand_awarded < _cand_maxp:
                continue
            cand_crit = str(cand.get("criterion", "") or "")
            cand_sources = of_source_ids_map.get(cand_crit) or []
            if aggregate_of not in cand_sources:
                continue
            _ev_blob = " ".join(str(e) for e in (cand.get("evidence_list") or []) if e)
            if agg_variants and any(v in _ev_blob for v in agg_variants):
                consumer_confirmed = True
                break

        # ── Engagement gate (Fix A) ──────────────────────────────────────────
        # Recovery fires only when ONE of these holds:
        #   (a) Consumer criterion is fully credited AND its evidence contains
        #       the aggregate value (student computed the correct total).
        #   (b) At least 2 DISTINCT sub-working values are individually present
        #       in the student's answer (proves engagement across multiple
        #       parts of the working — not a single cross-context reference).
        #   (c) At least 1 partial subset-sum (size >= 2) matches (proves the
        #       student rolled multiple sub-workings into one combined figure).
        # A single individual match (e.g. 3,125 appearing in the goodwill
        # working only) is NOT enough on its own — it may be a cross-context
        # reference where the value is used in a different working than the OF
        # group being scored.
        _partial_subset_matches = [s for s in matching_subsets if len(s) >= 2]
        engagement_ok = (
            consumer_confirmed
            or len(individually_confirmed) >= 2
            or len(_partial_subset_matches) >= 1
        )
        if not engagement_ok:
            continue

        # Confirmed sub-workings: union of every matching subset (individual or
        # partial). If the consumer is confirmed we ALSO include all sub-workings
        # (student computed the aggregate correctly, so credit the whole group).
        confirmed_results: set[int] = set()
        for subset in matching_subsets:
            confirmed_results.update(subset)
        if consumer_confirmed:
            confirmed_results.update(sub_results)

        if not confirmed_results:
            continue

        # Anchor line for recovered marks - prefer the aggregate line the
        # student ACTUALLY wrote for the components being recovered, so ticks
        # land next to that specific student writing.
        #
        # Priority order:
        #   (1) A subset-sum matching ONLY the sub_results whose producers are
        #       un-earned (the "recovered-only" aggregate). This is the line
        #       the student wrote in place of showing the components. For
        #       Amber's NCI group, sub_results = {3125, 2500, 1350} but 3125
        #       is already awarded (LLM caught "at acq 3,125"), so the
        #       recovered-only aggregate is 2500 + 1350 = 3,850 → anchor on
        #       "share of post acq reserves 3,850,000.00" (in NCI w5 working),
        #       NOT on "add back nci 6,975,000.00" (in disposal w3 working).
        #   (2) Fallback: any confirmed subset-sum (largest first). Catches
        #       the case where every sub_result is recovered — the full
        #       aggregate is then the correct anchor.
        #   (3) Fallback: consumer's own evidence line (journal entry).
        anchor_line: Optional[str] = None

        # Identify the sub_results whose producers are UN-EARNED (i.e., the
        # ones that will receive recovery marks). Producers with marks_awarded
        # already > 0 got direct LLM credit and their sub_result is NOT part
        # of the "recovered aggregate" - anchoring on their line would send
        # the tick to the wrong place.
        recovered_sub_results: set[int] = set()
        for r in confirmed_results:
            for idx in results_to_indices.get(r, []):
                try:
                    _idx_awarded = float(normalized_breakdown[idx].get("marks_awarded", 0) or 0)
                except (TypeError, ValueError):
                    _idx_awarded = 0.0
                if _idx_awarded <= 0:
                    recovered_sub_results.add(r)
                    break

        # ── Per-sub_result subset assignment (teacher-style per-line marking) ─
        # Instead of merging ALL recovered components under a single group
        # anchor, split them by WHICH subset each sub_result actually belongs
        # to in the student's writing:
        #
        #   • sub_result r assigned to the LARGEST matching subset S where
        #     r ∈ S ⊆ recovered_sub_results and sum(S) appears in the
        #     student's text.
        #   • Size-1 subset ({r}) → r appears alone in student text → its
        #     producers stay as INDIVIDUAL entries with their own anchor.
        #   • Size-2+ subset → r was aggregated with others in the student's
        #     writing → its producers merge into ONE combined entry with
        #     the aggregate's anchor line.
        #
        # For Amber's OF12 (recovered = {250, 12750, 5400}):
        #   - 250 → subset {250} → anchor "share cap 250,000" (individual, 0.25)
        #   - 12750, 5400 → subset {12750, 5400}=18150 → anchor "reserves w4
        #     18,150,000" (merged, 0.25 + 0.5 = 0.75)
        # This mirrors teacher's actual marking.

        # Enumerate all matching all-recovered subsets.
        matching_subsets_all_recovered: list[frozenset[int]] = []
        if n <= 12 and recovered_sub_results:
            for mask in range(1, 1 << n):
                subset = [sub_results[i] for i in range(n) if mask & (1 << i)]
                if subset and all(r in recovered_sub_results for r in subset):
                    s = sum(subset)
                    if _value_present_in_text(s, scoped_text) and _subset_is_supported(
                        subset
                    ):
                        matching_subsets_all_recovered.append(frozenset(subset))

        # For each recovered sub_result, find the LARGEST matching subset
        # containing it (tie-broken by higher sum). Sub_results with no
        # matching subset (shouldn't happen since they're in recovered set)
        # fall through with no assignment.
        subset_for_sub_result: dict[int, frozenset[int]] = {}
        for r in recovered_sub_results:
            best: Optional[frozenset[int]] = None
            for S in matching_subsets_all_recovered:
                if r not in S:
                    continue
                if best is None:
                    best = S
                elif len(S) > len(best) or (len(S) == len(best) and sum(S) > sum(best)):
                    best = S
            if best is not None:
                subset_for_sub_result[r] = best

        # Compute per-subset anchor line. Cache so we don't repeat the
        # _find_working_line_for_value call.
        anchor_line_for_subset: dict[frozenset[int], Optional[str]] = {}
        for S in set(subset_for_sub_result.values()):
            anchor_line_for_subset[S] = _find_working_line_for_value(sum(S), scoped_text)

        # Legacy group-level anchor (used when a sub_result has no subset
        # assignment - shouldn't normally happen, but keeps behaviour safe).
        if n <= 12:
            # PRIORITY 1: any recovered-only subset — largest first.
            recovered_only_sums: list[int] = [
                sum(S) for S in matching_subsets_all_recovered
            ]
            for s in sorted(set(recovered_only_sums), reverse=True):
                line = _find_working_line_for_value(s, scoped_text)
                if line:
                    anchor_line = line
                    break

            # PRIORITY 2: fall back to any confirmed subset-sum if no
            # recovered-only match found (e.g. when every sub_result in the
            # group is being recovered - the full aggregate IS the right
            # anchor).
            if not anchor_line:
                candidate_sums: list[int] = []
                for mask in range(1, 1 << n):
                    subset = [sub_results[i] for i in range(n) if mask & (1 << i)]
                    if all(r in confirmed_results for r in subset):
                        s = sum(subset)
                        if _value_present_in_text(s, scoped_text):
                            candidate_sums.append(s)
                for s in sorted(set(candidate_sums), reverse=True):
                    line = _find_working_line_for_value(s, scoped_text)
                    if line:
                        anchor_line = line
                        break

        # Fallback: inherit the consumer's evidence if no working line found.
        consumer_evidence: list[str] = []
        if not anchor_line:
            for cand in normalized_breakdown:
                try:
                    awarded = float(cand.get("marks_awarded", 0) or 0)
                    maxp = float(cand.get("max_possible", 0) or 0)
                except (TypeError, ValueError):
                    continue
                if maxp <= 0 or awarded < maxp:
                    continue
                cand_crit = str(cand.get("criterion", "") or "")
                cand_sources = of_source_ids_map.get(cand_crit) or []
                if aggregate_of not in cand_sources:
                    continue
                _ev_list = [str(e) for e in (cand.get("evidence_list") or []) if e]
                if _ev_list:
                    consumer_evidence = _ev_list
                    break

        # Fix B — anchor-collision guard.
        # Build the set of student-writing lines that ANOTHER producer in this
        # SAME aggregate group has already legitimately claimed (i.e., the LLM
        # awarded that producer marks > 0 directly, not via a prior recovery
        # pass). If our prospective recovery target's anchor line falls in
        # that set, skip the award: the line demonstrates the sibling sub-
        # mark, not this one. Without this, a student who writes ONE working
        # (say "NCI 3,125,000 =25*(500,000-375,000)") triggers a subset-sum
        # match on "25" for another OF2 sub-mark (NCI post-acq stake at 25%)
        # and gets that 0.25 for free — even though they never applied the
        # 25% stake to post-acquisition profits.
        # Same-aggregate scope only. Cross-aggregate legitimate co-cites
        # (e.g. Amber's OF1 reserves 2,750 and OF12 recovered 12,750 both
        # anchored on the tabular row) must still work.
        def _norm_anchor(s: str) -> str:
            # Collapse whitespace / separators AND strip thousands commas so
            # the student's raw "3125000" matches the LLM's quoted
            # "3,125,000". Case-fold too.
            _s = re.sub(r"[\s;|]+", " ", str(s)).strip().lower()
            return _s.replace(",", "")

        llm_awarded_anchors_in_aggregate: set[str] = set()
        for _p_idx in producer_indices:
            _p_bd = normalized_breakdown[_p_idx]
            try:
                _p_awarded = float(_p_bd.get("marks_awarded", 0) or 0)
                _p_maxp = float(_p_bd.get("max_possible", 0) or 0)
            except (TypeError, ValueError):
                continue
            if _p_awarded <= 0 or _p_maxp <= 0:
                continue
            _p_reason = str(_p_bd.get("reason", "") or "").lower()
            if "aggregate recovery" in _p_reason:
                continue  # only LLM-direct awards claim a line for this guard
            for _ev in (_p_bd.get("evidence_list") or []):
                _ev_norm = _norm_anchor(_ev)
                if _ev_norm:
                    llm_awarded_anchors_in_aggregate.add(_ev_norm)

        # Award recovery to un-earned producers in the confirmed sub-workings.
        # Sort so INDIVIDUALLY-CONFIRMED sub-workings go first - those get full
        # marks (student directly wrote the value), then subset-only-confirmed
        # ones (0.25 shortcut credit). Priority sort matters when the cap bites.
        confirmed_indices: list[int] = []
        for r in confirmed_results:
            confirmed_indices.extend(results_to_indices.get(r, []))

        # Values the student DERIVED inside an expression without ever writing
        # the result (e.g. "=12750000+(7200000*9/12)" computes 5,400,000 but
        # never states it). Computed once — the check below runs per criterion
        # and inside a sort key.
        computed_values = _computed_values_in_text(scoped_text)

        def _idx_is_individually_confirmed(idx: int) -> bool:
            crit = str(normalized_breakdown[idx].get("criterion", "") or "")
            r = of_produces_map.get(crit)
            if r is None:
                return False
            if r in individually_confirmed:
                return True
            # Showing the working counts as much as writing the answer. Without
            # this, a student who wrote the 9/12 apportionment out longhand got
            # the 0.25 shortcut credit meant for someone who only produced the
            # rolled-up total — scoring BELOW a student who wrote the bare
            # figure and no working at all.
            return r in computed_values

        # Stable sort: individually-confirmed first, then subset-only.
        confirmed_indices.sort(key=lambda idx: 0 if _idx_is_individually_confirmed(idx) else 1)

        # Hard cap per group so no single working can dominate the total.
        _group_cap = _RECOVERY_HARD_CAP
        group_recovered = 0.0
        for idx in confirmed_indices:
            if group_recovered >= _group_cap:
                break
            bd = normalized_breakdown[idx]
            try:
                awarded = float(bd.get("marks_awarded", 0) or 0)
                maxp = float(bd.get("max_possible", 0) or 0)
            except (TypeError, ValueError):
                continue
            if awarded > 0 or maxp <= 0:
                continue
            # Skip criteria that were zeroed because their content was credited
            # elsewhere ("Marks given above" / "Marks given below"). Recovering
            # these would double-award for the same student writing.
            _reason_l = str(bd.get("reason", "") or "").lower()
            if "marks given above" in _reason_l or "marks given below" in _reason_l:
                continue
            # The parent-calc guard already established that the student never
            # performed THIS working. Recovery must not overturn that finding —
            # it produced self-contradicting awards where one reason string both
            # revoked and restored the same quarter mark.
            if bd.get("_parent_calc_revoked"):
                continue
            # Fix B anchor-collision skip. Compute the prospective anchor for
            # THIS producer up front. If it collides with a line an LLM-
            # awarded sibling in the same aggregate already claims, that
            # sibling captured the student's writing; giving this producer
            # a mark on the same line double-credits one working.
            if llm_awarded_anchors_in_aggregate:
                _crit_here = str(bd.get("criterion", "") or "")
                _sub_result_here = of_produces_map.get(_crit_here)
                _subset_here: Optional[frozenset[int]] = None
                if _sub_result_here is not None:
                    _subset_here = subset_for_sub_result.get(int(_sub_result_here))
                _prospective_anchor = (
                    anchor_line_for_subset.get(_subset_here) if _subset_here else None
                ) or anchor_line
                if _prospective_anchor:
                    _pa_norm = _norm_anchor(_prospective_anchor)
                    if _pa_norm in llm_awarded_anchors_in_aggregate:
                        continue  # sibling in same aggregate already claims this line
            # PER-ITEM AWARD:
            #   • Individually-confirmed sub-working (its result appears alone
            #     in the student's answer) → award FULL max_possible. The
            #     student directly demonstrated this specific value.
            #   • Only confirmed via subset-sum (e.g. absorbed into an
            #     aggregated line like 18,150 = 12,750 + 5,400) → award 0.25
            #     as shortcut credit.
            _confirmed_individually = _idx_is_individually_confirmed(idx)
            if _confirmed_individually:
                award = maxp
            else:
                award = min(_RECOVERY_PER_SUB_MARK, maxp)
            # Don't exceed the group cap.
            if group_recovered + award > _group_cap:
                award = _group_cap - group_recovered
            bd["marks_awarded"] = award
            group_recovered += award
            total_recovered += award
            prev_reason = (bd.get("reason", "") or "").strip()
            if _confirmed_individually:
                _recovery_note = (
                    f"Aggregate recovery (+{award}) - {aggregate_of} sub-working "
                    f"demonstrated in the student's own working; full component "
                    f"mark awarded. "
                )
            else:
                _recovery_note = (
                    f"Aggregate recovery (+{award}) - subset-sum of "
                    f"{aggregate_of} sub-workings matches value in student's "
                    f"answer; component absorbed into rolled-up figure. "
                )
            bd["reason"] = (_recovery_note + prev_reason).strip()
            # Stash aggregate metadata so the post-processing merge pass
            # (_merge_aggregate_recovered_entries) can group components
            # BY SUBSET, not by aggregate. Two components with the same
            # aggregate_of but different subset assignments stay separate
            # (e.g. Amber's #12 alone at 250 vs #13+#14 merged at 18,150).
            bd["_aggregate_of"] = aggregate_of
            # Determine this component's assigned subset via its sub_result.
            _bd_crit = str(bd.get("criterion", "") or "")
            _bd_sub_result = of_produces_map.get(_bd_crit)
            _bd_subset: Optional[frozenset[int]] = None
            if _bd_sub_result is not None:
                _bd_subset = subset_for_sub_result.get(int(_bd_sub_result))
            # Subset-specific anchor line (falls back to group anchor if the
            # sub_result has no subset assignment — defensive).
            _bd_subset_anchor = (
                anchor_line_for_subset.get(_bd_subset) if _bd_subset else None
            ) or anchor_line
            if _bd_subset_anchor:
                bd["_aggregate_anchor_line"] = _bd_subset_anchor
            elif anchor_line:
                bd["_aggregate_anchor_line"] = anchor_line
            # Subset key: a stable, hashable identifier for the subset this
            # component belongs to. Components with the SAME (aggregate_of,
            # subset_key) get merged; different subset keys stay separate.
            # Use sum(S) as the key — unique per subset within a group in
            # practice (since sub_results are distinct positive integers,
            # different subsets can only collide on sum in pathological
            # multi-way ties which don't arise in real rubrics).
            if _bd_subset is not None:
                _subset_sum = sum(_bd_subset)
                bd["_aggregate_subset_key"] = f"{aggregate_of}:{_subset_sum}"
                bd["_aggregate_subset_size"] = len(_bd_subset)
                # Target value = the specific number this component's mark
                # belongs to (in the student's writing). For size-1 subsets
                # this is the sub_result itself; for size-2+ subsets it's
                # the aggregate the student wrote (sum of the subset).
                # Also store search VARIANTS (comma-grouped, scaled, .00
                # suffix, etc.) so the annotator can find the value on
                # the PDF without needing to guess units.
                bd["_target_value"] = int(_subset_sum)
                bd["_target_value_variants"] = sorted(
                    _value_variants_for_search(int(_subset_sum))
                )
            else:
                # No subset assignment — fall back to legacy behaviour
                # (single-group merge on aggregate_of alone).
                bd["_aggregate_subset_key"] = f"{aggregate_of}:legacy"
                bd["_aggregate_subset_size"] = 0
            # Inherit an anchor line so the annotator can place the mark.
            # Priority for anchor:
            #   1. SUBSET-specific anchor (the line for THIS component's
            #      assigned subset, e.g. "share cap 250,000" for a size-1
            #      subset {250}, or "reserves w4 18,150,000" for a size-2+
            #      subset {12750, 5400}).
            #   2. Group-level anchor (fall-back when no subset assignment).
            #   3. Consumer's own evidence (journal line, last resort).
            _anchors: list[str] = []
            if _bd_subset_anchor:
                _anchors.append(_bd_subset_anchor)
            elif anchor_line:
                _anchors.append(anchor_line)
            elif consumer_evidence:
                _anchors.extend(consumer_evidence)
            if _anchors:
                existing = list(bd.get("evidence_list") or [])
                for ev in _anchors:
                    if ev and ev not in existing:
                        existing.append(ev)
                bd["evidence_list"] = existing
                bd["evidence"] = "; ".join(existing)

    return total_recovered


def _merge_aggregate_recovered_entries(
    normalized_breakdown: list[dict],
    of_definitions: dict[str, dict],
) -> list[dict]:
    """Collapse aggregate-recovery entries into teacher-style per-line marks.

    Grouping rule (teacher-style, NOT one-merge-per-aggregate):

    Components are grouped by their assigned SUBSET (via `_aggregate_subset_key`
    set during recovery) - the subset the student's writing actually aggregated
    them into. Two components with the same `_aggregate_of` but DIFFERENT
    subsets stay separate.

    For Amber's OF12 (net assets at disposal, recovered = {250, 12750, 5400}):
      - Component #12 (produces 250) → subset {250} → stays INDIVIDUAL (0.25
        on the "share cap 250,000" line).
      - Components #13, #14 (produces 12750, 5400) → subset {12750, 5400}
        = 18,150 (the aggregate value the student wrote) → MERGED into one
        0.75 entry on the "reserves w4 18,150,000" line.

    Only subsets of SIZE ≥ 2 get merged. Size-1 subsets stay individual
    with their subset-specific anchor line (already set on evidence_list
    during recovery), preserving the granular criterion description so the
    student sees the specific point they earned.

    Args:
        normalized_breakdown: mutable list of grading records (in-place read only).
        of_definitions: sub_answer-level dict of aggregate OF metadata; used to
            look up an aggregate's human-readable label for the merged entry's
            criterion title.

    Returns:
        A new list with merged entries substituted for grouped components.
        Non-aggregate entries and size-1-subset components are passed through
        unchanged. Totals preserved.
    """
    if not normalized_breakdown:
        return list(normalized_breakdown)

    # Group indices by (aggregate_of, subset_key). Components in the same
    # subset get merged; components in different subsets stay separate.
    indices_by_subset: dict[tuple[str, str], list[int]] = {}
    for idx, bd in enumerate(normalized_breakdown):
        agg = bd.get("_aggregate_of")
        if not agg:
            continue
        subset_key = str(bd.get("_aggregate_subset_key") or f"{agg}:legacy")
        indices_by_subset.setdefault((str(agg), subset_key), []).append(idx)

    if not indices_by_subset:
        return list(normalized_breakdown)  # nothing to merge

    indices_to_drop: set[int] = set()
    merged_entries: list[tuple[int, dict]] = []  # (insert_after_idx, entry)

    for (aggregate_of, _subset_key), indices in indices_by_subset.items():
        # Only merge subsets that contained MULTIPLE sub_result values in the
        # student's aggregated line — a single-element subset means the student
        # wrote that value alone, so its criterion stays as its own entry with
        # its own value-specific anchor line (set during recovery).
        _subset_size = 0
        for i in indices:
            try:
                _subset_size = max(_subset_size, int(normalized_breakdown[i].get("_aggregate_subset_size", 0) or 0))
            except (TypeError, ValueError):
                pass
        if _subset_size < 2:
            continue  # single-value subset — keep as individual entry(ies)

        if len(indices) < 2:
            continue  # single component in this subset — no merge benefit

        # Only merge entries whose marks_awarded is > 0 (recovery may have
        # awarded 0 in edge cases; those shouldn't influence the merged mark).
        awarded_indices = [
            i for i in indices
            if float(normalized_breakdown[i].get("marks_awarded", 0) or 0) > 0
        ]
        if len(awarded_indices) < 2:
            continue

        # Aggregate anchor line: use the one stashed on the first entry
        # (all recovered entries in a group share the same anchor). Fall back
        # to any non-empty _aggregate_anchor_line, then to the first evidence.
        anchor_line: Optional[str] = None
        for i in awarded_indices:
            candidate = normalized_breakdown[i].get("_aggregate_anchor_line")
            if candidate:
                anchor_line = str(candidate).strip()
                break
        if not anchor_line:
            for i in awarded_indices:
                ev_list = normalized_breakdown[i].get("evidence_list") or []
                if ev_list:
                    anchor_line = str(ev_list[0]).strip()
                    break

        # Aggregate label (from of_definitions) — used as the merged
        # criterion's human-readable title.
        agg_label = ""
        if isinstance(of_definitions, dict):
            agg_def = of_definitions.get(aggregate_of) or {}
            agg_label = str(agg_def.get("label", "") or "").strip()
        if not agg_label:
            agg_label = f"aggregate {aggregate_of}"

        # Build the components list (full descriptions preserved for teacher
        # traceability). Also compute sums.
        components: list[dict] = []
        total_awarded = 0.0
        total_max = 0.0
        for i in awarded_indices:
            bd = normalized_breakdown[i]
            try:
                awarded = float(bd.get("marks_awarded", 0) or 0)
                maxp = float(bd.get("max_possible", 0) or 0)
            except (TypeError, ValueError):
                awarded, maxp = 0.0, 0.0
            components.append({
                "criterion_description": str(bd.get("criterion", "") or ""),
                "component_marks": awarded,
                "max_possible": maxp,
            })
            total_awarded += awarded
            total_max += maxp

        # Short reason for annotation clarity (full descriptions live in
        # `components` for teacher/CSV view).
        merged_criterion = f"[Combined] {agg_label} ({aggregate_of})"
        merged_reason = (
            f"Marks combined for {len(components)} component criteria of "
            f"{agg_label} ({aggregate_of}). Student's working presents the "
            f"aggregate figure directly on the line quoted below. Full "
            f"credit awarded on this single line (see 'components' for the "
            f"individual sub-marks that make up this total)."
        )

        # Propagate _target_value AND _target_value_variants to the merged
        # entry (unique target across all merged components since they share
        # the same subset key). The annotator uses these to narrow the
        # underline+score rect from the whole matched line down to just the
        # value cell — see _narrow_rect_to_target_value in annotator.py.
        _merged_target_value: Optional[int] = None
        _merged_target_variants: Optional[list[str]] = None
        for i in awarded_indices:
            tv = normalized_breakdown[i].get("_target_value")
            if tv is not None:
                try:
                    _merged_target_value = int(tv)
                    _merged_target_variants = list(
                        normalized_breakdown[i].get("_target_value_variants") or []
                    )
                    break
                except (TypeError, ValueError):
                    pass

        merged_entry = {
            # The aggregate's own OF id doubles as its criterion id. Without
            # one, nothing downstream can refer to this entry - the restatement
            # pass skipped both merged workings on the first live run, 2.0
            # marks' worth, including the two the marker points at.
            "criterion_id": str(aggregate_of or "").strip().upper(),
            "criterion": merged_criterion,
            "marks_awarded": total_awarded,
            "max_possible": total_max,
            "reason": merged_reason,
            "evidence": anchor_line or "",
            "evidence_list": [anchor_line] if anchor_line else [],
            "comments_summary": "",
            "components": components,
            "_merged_from_aggregate": True,
            "_merged_aggregate_of": aggregate_of,
        }
        if _merged_target_value is not None:
            merged_entry["_target_value"] = _merged_target_value
        if _merged_target_variants:
            merged_entry["_target_value_variants"] = _merged_target_variants

        # Insert merged entry at the position of the first component so
        # downstream ordering roughly follows the working section it lives in.
        insert_after = min(awarded_indices)
        merged_entries.append((insert_after, merged_entry))
        indices_to_drop.update(awarded_indices)

    if not merged_entries:
        return list(normalized_breakdown)

    # Build the new breakdown: preserve original order, drop merged
    # components, insert each merged entry at its first-component position.
    new_breakdown: list[dict] = []
    insert_map: dict[int, list[dict]] = {}
    for insert_after, entry in merged_entries:
        insert_map.setdefault(insert_after, []).append(entry)
    for idx, bd in enumerate(normalized_breakdown):
        if idx in indices_to_drop:
            # Emit any merged entries anchored here BEFORE we drop this row.
            for entry in insert_map.pop(idx, []):
                new_breakdown.append(entry)
            continue
        new_breakdown.append(bd)
    # Any merged entries whose insert_after index was not in indices_to_drop
    # (shouldn't happen with current logic, but defensive) - append at end.
    for remaining in insert_map.values():
        new_breakdown.extend(remaining)

    return new_breakdown


def _norm_for_evidence_match_static(s: str) -> str:
    """Module-level twin of the normaliser nested in _build_grade_doc.

    Kept byte-identical so that a line judged "already earning marks" here is
    judged the same way when the note is placed.
    """
    if not s or not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ").replace("\u00d7", "x")
    s = s.replace("\u2013", "-").replace("-", "-")
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = re.sub(r"[^a-z0-9%/().,\- ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


class StudentGrader:

    COLLECTION_NAME = "student_grades"

    def __init__(
        self,
        student_name: str,
        question_number: str,
        questions_id: Optional[str],
        model_answers_id: Optional[str],
        student_answers_id: str,
        question_type: str = "numerical",
    ):
        self.student_name = student_name
        self.question_number = question_number
        self.questions_id = questions_id
        self.model_answers_id = model_answers_id
        self.student_answers_id = student_answers_id
        self.question_type = question_type  # "numerical" or "theoretical"

        # Flag: True when marking criteria were synthesized from answer text
        # (no formal rubric provided). Used to relax strict guardrails.
        self._criteria_were_synthesized: bool = False

        # Flag: True when no marking criteria exist and we use holistic grading
        # (compare full answers) instead of per-criterion grading.
        self._holistic_grading: bool = False
        # Cached holistic sub-question structure for the grading prompt.
        # Each entry: {"sub_question": str, "answer": str, "max_marks": float}
        self._holistic_sub_questions: list[dict] = []

        # Cache of the exact student text passed to the grader in the most recent run.
        # Used for post-grading guardrails (e.g., verify evidence quotes actually exist).
        self._student_text_last_run: str = ""

        # Cache of the question/model payload used in the most recent run.
        # Used to detect "tainted" evidence that is copied from the question/markscheme
        # (e.g., section headings) rather than the student's own work.
        self._question_text_last_run: str = ""
        self._model_text_last_run: str = ""

        # Optional debug trace of raw LLM outputs / parsing errors.
        # Enabled via DEBUG_SAVE_LLM_OUTPUT=1.
        self._llm_debug_trace: list[dict[str, Any]] = []

        # Cache of the rubric criteria (descriptions) used for the most recent run.
        # Used to (a) prevent the LLM from inventing criteria and (b) enforce max_possible.
        self._allowed_criteria_last_run: set[str] = set()
        self._criterion_max_map_last_run: dict[str, float] = {}
        self._criterion_category_map_last_run: dict[str, str] = {}
        # Criteria that require the exact expected number (no OF bypass allowed).
        # Populated from rubric fields with exact_match=True.
        self._exact_match_criteria_last_run: set[str] = set()
        # Cache rubric order (description → position) for post-processing heuristics.
        self._rubric_criteria_order_last_run: list[str] = []
        self._rubric_position_last_run: dict[str, int] = {}
        # Short rubric id -> canonical criterion description. The model returns
        # the id instead of retyping the description, so we can resolve the
        # criterion exactly rather than by matching 600 characters of prose.
        self._criterion_id_map_last_run: dict[str, str] = {}
        # OF metadata caches (populated from rubric each run).
        # `of_ids`: criteria that ORIGINATE OF values (list - 1 or 2 entries).
        # `of_source_ids`: criteria that DEPEND on upstream OFs. Presence is a strong
        # signal that a number mismatch may be a legitimate OF carry (skip revocation).
        # `of_value`: canonical value/unit/label for origin criteria.
        # `of_definitions`: sub_answer-level dict of virtual-origin OFs (working
        # totals like OF1=3,400 that aren't tied to any single criterion).
        self._of_ids_by_criterion_last_run: dict[str, list[str]] = {}
        self._of_source_ids_by_criterion_last_run: dict[str, list[str]] = {}
        self._of_value_by_criterion_last_run: dict[str, dict[str, Any]] = {}
        self._of_definitions_last_run: dict[str, dict[str, Any]] = {}
        # `of_component_of[criterion_desc] = "OF2"` - the aggregate OF this
        # criterion contributes to. Used by the aggregate-value recovery guard.
        self._of_component_of_by_criterion_last_run: dict[str, str] = {}
        # `of_produces[criterion_desc] = 2500` - the numeric value the parent
        # calculation for this criterion produces (e.g., #17 belongs to the
        # "25% × (12.75m − 2.75m) = 2,500" sub-working). Used by the subset-sum
        # aggregate recovery to detect when a student writes an intermediate
        # aggregated value (like 3,850 = 2,500 + 1,350) that implies they did
        # the working via a different decomposition.
        self._of_produces_by_criterion_last_run: dict[str, int] = {}
        self._keywords_by_criterion_last_run: dict[str, list[str]] = {}
        self._expected_amount_by_criterion_last_run: dict[str, float] = {}
        self._sign_sensitive_criteria_last_run: set[str] = set()
        self._column_by_criterion_last_run: dict[str, str] = {}

        self.grades_coll = get_collection(self.COLLECTION_NAME)

        # Grading chain.
        # Prefer strict structured-output when the provider supports it.
        # Some providers/models (e.g., Grok / some Anthropic setups) may return
        # non-conforming JSON; we fall back to text parsing + one repair pass.
        self.grade_chain_structured = None
        try:
            structured_grader = llm_grader.with_structured_output(LLMGradingResponse)
            self.grade_chain_structured = grade_prompt | structured_grader
        except Exception:
            self.grade_chain_structured = None

        self.grade_chain_text = grade_prompt | llm_grader

        # Holistic grading chains (used when no marking criteria exist).
        self.holistic_chain_structured = None
        try:
            holistic_structured = llm_grader.with_structured_output(HolisticGradingResponse)
            self.holistic_chain_structured = holistic_grade_prompt | holistic_structured
        except Exception:
            self.holistic_chain_structured = None
        self.holistic_chain_text = holistic_grade_prompt | llm_grader

        # Restatement pass. Runs AFTER grading, because only then is it known
        # which line each mark landed on - aggregate recovery, subset-sum
        # recovery and the guardrails all move awards after the grading model
        # has replied. Its only job is to find lines where the student repeats
        # an already-credited point, so a marker's "Marks given above/below"
        # note can be placed there.
        self.restatement_chain = None
        try:
            self.restatement_chain = restatement_prompt | llm_grader.with_structured_output(
                RestatementResponse
            )
        except Exception:
            self.restatement_chain = None

    @staticmethod
    def _extract_structured_args_from_message(output: Any) -> Optional[dict]:
        # Newer LangChain: tool_calls is a list of dict-like objects containing `args`.
        try:
            tool_calls = getattr(output, "tool_calls", None)
            if isinstance(tool_calls, list) and tool_calls:
                first = tool_calls[0]
                if isinstance(first, dict):
                    args = first.get("args")
                else:
                    args = getattr(first, "args", None)

                if isinstance(args, dict):
                    return args
                if isinstance(args, str) and args.strip():
                    return json.loads(args)
        except Exception:
            pass

        # Older/alternate: tool calls inside additional_kwargs
        try:
            additional = getattr(output, "additional_kwargs", None) or {}
            if isinstance(additional, dict):
                tc = additional.get("tool_calls")
                if isinstance(tc, list) and tc:
                    func = tc[0].get("function") if isinstance(tc[0], dict) else None
                    if isinstance(func, dict):
                        arguments = func.get("arguments")
                        if isinstance(arguments, str) and arguments.strip():
                            return json.loads(arguments)

                fc = additional.get("function_call")
                if isinstance(fc, dict):
                    arguments = fc.get("arguments")
                    if isinstance(arguments, str) and arguments.strip():
                        return json.loads(arguments)
        except Exception:
            pass

        return None

    @staticmethod
    def _extract_json_from_text(raw: str) -> str:

        if not isinstance(raw, str):
            raw = str(raw)

        cleaned = raw.strip()
        # Remove BOM / zero-width chars that can break json.loads at column 1
        cleaned = cleaned.lstrip("\ufeff\u200b\u200c\u200d")
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return ""

        # Try to isolate the first JSON object/array.
        first_curly = cleaned.find("{")
        first_square = cleaned.find("[")

        starts = [i for i in (first_curly, first_square) if i != -1]
        if not starts:
            return cleaned

        start = min(starts)

        last_curly = cleaned.rfind("}")
        last_square = cleaned.rfind("]")
        ends = [i for i in (last_curly, last_square) if i != -1]
        end = max(ends) + 1 if ends else len(cleaned)

        return cleaned[start:end].strip()

    def _extract_question_max_marks(self, questions_data: dict, main_grade: Optional[dict] = None) -> float:
        """Extract the total maximum marks for the question being graded.

        Priority:
        1. Matching sub-question marks - find the question matching self.question_number
           and use its marks (from the "marks" field, or from trailing "(N)" in content).
           This handles papers where total_marks is the whole-paper total (e.g. 54)
           but each question has its own marks (e.g. 12).
        2. Document-level total_marks - only if there's a single question or no sub-questions.
        3. The LLM grader's own total_marks report in main_grade.
        4. Sum of individual sub-question marks as a last resort.
        """
        def _parse_nums(text: Any) -> list[float]:
            if text is None:
                return []
            return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text))]

        def _best_from_nums(nums: list[float]) -> Optional[float]:
            pos = [n for n in nums if n > 0]
            return max(pos) if pos else None

        def _extract_trailing_marks(text: str) -> Optional[float]:
            """Extract marks from the end of question content, e.g. '...(12)' or '...(12 marks)'."""
            if not text:
                return None
            # Match trailing parenthesized number, optionally followed by "marks"
            m = re.search(r"\((\d+(?:\.\d+)?)\s*(?:[Mm]arks?)?\)\s*$", text.strip())
            if m:
                return float(m.group(1))
            return None

        # 0. Document-level total_marks - try first since the document is already
        # scoped to a single question and its total_marks is the authoritative total.
        if isinstance(questions_data, dict):
            q_total_raw = questions_data.get("total_marks")
            if q_total_raw is not None:
                q_text = str(q_total_raw)
                for pattern in (
                    r"maximum\s*marks?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                    r"max(?:imum)?\s*[:=]?\s*(\d+(?:\.\d+)?)",
                ):
                    m = re.search(pattern, q_text, flags=re.IGNORECASE)
                    if m:
                        logger.info(f"Using document-level total_marks for Q{self.question_number}: {m.group(1)}")
                        return float(m.group(1))
                v = _best_from_nums(_parse_nums(q_text))
                if v:
                    logger.info(f"Using document-level total_marks for Q{self.question_number}: {v}")
                    return v

        # 1. Try to find marks for the specific question being graded.
        if isinstance(questions_data, dict):
            questions_list = questions_data.get("questions")
            if isinstance(questions_list, list) and len(questions_list) > 0:
                q_digit = "".join(re.findall(r"\d+", str(self.question_number)))

                for q in questions_list:
                    if not isinstance(q, dict):
                        continue
                    q_num = str(q.get("question_number", ""))
                    q_num_digit = "".join(re.findall(r"\d+", q_num))

                    if not q_digit or not q_num_digit:
                        continue
                    if q_num_digit != q_digit and not q_num_digit.startswith(q_digit) and not q_digit.startswith(q_num_digit):
                        continue

                    # Found matching question.
                    # Priority: LLM-computed total_marks on the question item (most reliable).
                    llm_total = q.get("total_marks")
                    if llm_total is not None:
                        try:
                            v = float(llm_total)
                            if v > 0:
                                logger.info(f"Using LLM total_marks for Q{self.question_number}: {v}")
                                return v
                        except (TypeError, ValueError):
                            pass

                    # Fallback: recursively sum sub_questions marks -
                    # handles old extractions that pre-date the total_marks field.
                    def _sum_sq_marks(sq_list: list) -> float:
                        """Recursively sum leaf-level marks across all sub_questions."""
                        total = 0.0
                        for sq in sq_list:
                            if not isinstance(sq, dict):
                                continue
                            nested = sq.get("sub_questions")
                            if nested:
                                # Has deeper nesting - recurse instead of reading this level
                                child_total = _sum_sq_marks(nested)
                                if child_total > 0:
                                    total += child_total
                                    continue
                            # Leaf node - read marks directly
                            sq_v = None
                            for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                                raw = sq.get(key)
                                if raw is not None:
                                    sq_v = _best_from_nums(_parse_nums(raw))
                                    if sq_v:
                                        break
                            if sq_v is None:
                                sq_content = sq.get("content", "")
                                if isinstance(sq_content, str):
                                    sq_v = _extract_trailing_marks(sq_content)
                            if sq_v:
                                total += sq_v
                        return total

                    sub_questions = q.get("sub_questions") or []
                    if sub_questions:
                        sq_total = _sum_sq_marks(sub_questions)
                        if sq_total > 0:
                            logger.info(f"Using sum of sub-question marks for Q{self.question_number}: {sq_total}")
                            return sq_total

                    # No sub_questions - use the question-level marks field directly
                    for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                        raw = q.get(key)
                        if raw is not None:
                            v = _best_from_nums(_parse_nums(raw))
                            if v:
                                logger.info(f"Using sub-question marks for Q{self.question_number}: {v} (from '{key}': '{raw}')")
                                return v

                    # Fallback: extract trailing marks from the question content text
                    content = q.get("content", "")
                    if isinstance(content, str):
                        v = _extract_trailing_marks(content)
                        if v:
                            logger.info(f"Using trailing marks from question content for Q{self.question_number}: {v}")
                            return v

        # 2. (Skipped - document-level total_marks already handled in step 0.)

        # 3. LLM-reported total from main_grade.
        if main_grade and isinstance(main_grade, dict):
            for key in ("total_marks", "maximum_marks", "total_marks_available"):
                v = _best_from_nums(_parse_nums(main_grade.get(key)))
                if v:
                    return v

        # 4. Sum individual sub-question marks as a fallback.
        if isinstance(questions_data, dict):
            questions_list = questions_data.get("questions")
            if isinstance(questions_list, list):
                total = 0.0
                for q in questions_list:
                    if not isinstance(q, dict):
                        continue
                    for key in ("marks", "maximum_marks", "max_marks", "total_marks"):
                        nums = _parse_nums(q.get(key))
                        if nums:
                            total += max(n for n in nums if n > 0)
                            break
                if total > 0:
                    return total

        # 5. Last resort: document-level total_marks even for multi-question papers
        if isinstance(questions_data, dict):
            q_total_raw = questions_data.get("total_marks")
            if q_total_raw is not None:
                v = _best_from_nums(_parse_nums(str(q_total_raw)))
                if v:
                    logger.warning(f"Using paper-level total marks as fallback: {v} (may include marks for other questions)")
                    return v

        return 0.0

    # ──────────────────────────────────────────────────────────────────────
    # Criteria synthesis - used when model answers have no marking_criteria
    # but contain inline marks in the answer text (e.g. "SL (3 Marks)")
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_inline_section_marks(answer_text: str) -> list[tuple[str, float, str]]:
        """Parse sections with inline marks from model answer text.

        Looks for patterns like:
          "SL (3 Marks)"  /  "Issue 1 - Peak Estate (4 Marks)"  /  "Required adjustment: (1 Marks)"
        at the start of sections in the answer text.

        Returns list of (section_title, marks, section_body) tuples.
        """
        if not answer_text or not isinstance(answer_text, str):
            return []

        # Pattern: a heading/label followed by (N Marks) or (N marks) or (N Mark)
        # Also handles: "SL (3 Marks)\n..." and "Issue 1 - Peak Estate (4 Marks)\n..."
        section_pattern = re.compile(
            r"^(.+?)\s*\((\d+(?:\.\d+)?)\s*[Mm]arks?\)",
            re.MULTILINE,
        )

        matches = list(section_pattern.finditer(answer_text))
        if not matches:
            return []

        sections: list[tuple[str, float, str]] = []
        for i, m in enumerate(matches):
            title = m.group(1).strip().rstrip("-–-:").strip()
            marks = float(m.group(2))
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
            body = answer_text[body_start:body_end].strip()
            if marks > 0 and body:
                sections.append((title, marks, body))

        return sections

    @staticmethod
    def _split_answer_into_points(section_body: str) -> list[str]:
        """Split a section of model answer text into individual marking points.

        Splits on:
        - Bullet points (-, •, *)
        - Numbered points (1., 2.), (i), (ii))
        - Lines starting with ":" after a keyword
        - Paragraph breaks (double newline)
        - Sentence boundaries for long non-bulleted paragraphs

        Returns list of non-empty point strings.
        """
        if not section_body or not isinstance(section_body, str):
            return []

        lines = section_body.strip().split("\n")
        points: list[str] = []
        current: list[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # Paragraph break - flush current
                if current:
                    points.append(" ".join(current))
                    current = []
                continue

            # Detect bullet / numbered list starts
            is_new_point = bool(re.match(
                r"^(?:[-•*]|\d+[.):]|\([a-z]\)|\([ivx]+\))\s",
                stripped,
                re.IGNORECASE,
            ))

            if is_new_point and current:
                points.append(" ".join(current))
                current = []

            # Clean bullet/number prefix
            cleaned = re.sub(r"^(?:[-•*]|\d+[.):]|\([a-z]\)|\([ivx]+\))\s*", "", stripped).strip()
            if cleaned:
                current.append(cleaned)

        if current:
            points.append(" ".join(current))

        # Filter out very short/meaningless points
        points = [p for p in points if len(p) >= 10]

        # Post-process: split long non-bulleted paragraphs into sentences.
        # This handles model answers where distinct marking points are written
        # as continuous prose rather than bullet lists.
        expanded: list[str] = []
        for p in points:
            if len(p) > 150:
                # Split on sentence boundaries (period followed by space and capital letter,
                # or period followed by newline)
                sentences = re.split(r"(?<=\.)\s+(?=[A-Z])", p)
                sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) >= 10]
                if len(sentences) > 1:
                    expanded.extend(sentences)
                else:
                    expanded.append(p)
            else:
                expanded.append(p)

        return expanded

    def _synthesize_criteria_from_answer(self, answers: list[dict]) -> list[dict[str, Any]]:
        """Generate marking criteria from model answer text when none are provided.

        When the model answer has no marking_criteria but embeds section marks
        inline (e.g. "SL (3 Marks)"), this method:
        1. Parses out each section and its total marks
        2. Splits each section into individual marking points
        3. Distributes the section marks evenly across its points
        4. Returns a list of synthesized criteria dicts
        """
        synthesized: list[dict[str, Any]] = []

        for answer in answers:
            if not isinstance(answer, dict):
                continue
            answer_text = answer.get("answer")
            if not isinstance(answer_text, str) or not answer_text.strip():
                continue

            sections = self._parse_inline_section_marks(answer_text)

            if sections:
                for title, section_marks, body in sections:
                    points = self._split_answer_into_points(body)
                    if not points:
                        # Can't split - use the whole section as one criterion
                        synthesized.append({
                            "marks": section_marks,
                            "description": f"{title}: {body[:200]}",
                        })
                        continue

                    # Distribute marks across points so the total equals section_marks.
                    # Strategy: assign a base of 0.25 per point, then distribute remaining
                    # marks as bonus 0.25 increments to earlier (more important) points.
                    n = len(points)
                    base = 0.25
                    total_at_base = base * n

                    if total_at_base >= section_marks:
                        # More points than marks allow at 0.25 each - only keep enough points
                        max_points = int(section_marks / base)
                        points = points[:max_points] if max_points > 0 else points[:1]
                        n = len(points)
                        total_at_base = base * n

                    # Remaining marks to distribute as bonus 0.25 increments
                    remaining_marks = section_marks - total_at_base
                    bonus_slots = int(round(remaining_marks / 0.25))

                    for j, point in enumerate(points):
                        mark = base
                        if j < bonus_slots:
                            mark += 0.25
                        mark = round(mark / 0.25) * 0.25
                        synthesized.append({
                            "marks": mark,
                            "description": point,
                        })
            else:
                # No inline marks found - try to use question-level marks
                # and split the entire answer into points
                points = self._split_answer_into_points(answer_text)
                if not points:
                    continue

                # Try to get total marks from answer text ending pattern like "(12)" or "12 marks"
                total_marks_match = re.search(
                    r"\((\d+(?:\.\d+)?)\s*(?:[Mm]arks?)?\)\s*$", answer_text.strip()
                )
                if total_marks_match:
                    total = float(total_marks_match.group(1))
                else:
                    # Fallback: can't determine marks, assign equal weight placeholder
                    # These will be scaled in _flatten_model_answers when we know total marks
                    total = float(len(points))  # 1 mark per point as placeholder

                n = len(points)
                base = 0.25
                total_at_base = base * n

                if total_at_base >= total:
                    max_points = int(total / base)
                    points = points[:max_points] if max_points > 0 else points[:1]
                    n = len(points)
                    total_at_base = base * n

                remaining_marks = total - total_at_base
                bonus_slots = int(round(remaining_marks / 0.25))

                for j, point in enumerate(points):
                    mark = base
                    if j < bonus_slots:
                        mark += 0.25
                    mark = round(mark / 0.25) * 0.25
                    synthesized.append({
                        "marks": mark,
                        "description": point,
                    })

        if synthesized:
            logger.info(
                f"Synthesized {len(synthesized)} criteria from model answer text "
                f"(total marks: {sum(c['marks'] for c in synthesized):.1f})"
            )

        return synthesized

    @staticmethod
    def _answer_matches_question(answer_label: str, question_number: str) -> bool:
        """Check if a model answer's question_number matches the question being graded.

        Handles varied labelling conventions: "Ans.1", "Ans 1", "A1", "(a)", "1", "Q.1", etc.
        """
        if not answer_label or not question_number:
            return True  # If either is missing, don't filter

        def _extract_digits(s: str) -> str:
            return "".join(re.findall(r"\d+", s))

        a_digits = _extract_digits(answer_label)
        q_digits = _extract_digits(question_number)

        if not a_digits or not q_digits:
            return True  # Can't compare - don't filter

        # Match if the leading digit(s) agree (e.g. "Ans.1" vs "Q.1" → "1" == "1")
        return a_digits == q_digits or a_digits.startswith(q_digits) or q_digits.startswith(a_digits)

    def _flatten_model_answers(self, model_data: dict, questions_data: Optional[dict] = None) -> dict:
        if not isinstance(model_data, dict):
            return model_data

        answers = model_data.get("answers")
        if not isinstance(answers, list) or not answers:
            return model_data

        # ── Filter answers to only those matching the question being graded ──
        # Skip per-answer filtering when the document-level question_title already
        # matches the target question - all answers in the doc are sub-parts of it.
        doc_title = str(model_data.get("question_title", ""))
        doc_matches_target = self._answer_matches_question(doc_title, self.question_number)

        if len(answers) > 1 and not doc_matches_target:
            filtered = [
                a for a in answers
                if isinstance(a, dict) and self._answer_matches_question(
                    str(a.get("question_number", "")), self.question_number
                )
            ]
            if filtered:
                answers = filtered
                logger.info(
                    f"Filtered model answers to {len(answers)} matching Q{self.question_number} "
                    f"(from {len(model_data['answers'])} total)"
                )
        else:
            logger.info(
                f"Filtered model answers to {len(answers)} matching Q{self.question_number} "
                f"(from {len(model_data['answers'])} total)"
            )

        combined_criteria: list[dict[str, Any]] = []
        combined_answer_parts: list[str] = []

        def _norm_desc_key(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("–", "-").replace("-", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            return s

        def _dedup_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Deduplicate repeated criteria descriptions.

            Some marking guides repeat identical descriptions across nested parts; asking the
            LLM to grade duplicates harms stability and can double-count.
            """
            out: list[dict[str, Any]] = []
            key_to_index: dict[str, int] = {}

            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                if not desc:
                    continue
                key = _norm_desc_key(desc)
                if not key:
                    continue

                marks = it.get("marks")
                marks_num: Optional[float] = None
                if isinstance(marks, (int, float)):
                    marks_num = float(marks)

                if key in key_to_index:
                    existing = out[key_to_index[key]]
                    ex_marks = existing.get("marks")
                    ex_marks_num: Optional[float] = float(ex_marks) if isinstance(ex_marks, (int, float)) else None
                    if marks_num is not None and (ex_marks_num is None or marks_num > ex_marks_num):
                        existing["marks"] = marks_num
                    continue

                _entry: dict[str, Any] = {
                    "marks": marks_num if marks_num is not None else marks,
                    "description": desc,
                }
                if it.get("category"):
                    _entry["category"] = it["category"]
                if it.get("exact_match"):
                    _entry["exact_match"] = it["exact_match"]
                # Preserve OF metadata through dedup.
                for _fld in (
                    "criterion_id",
                    "of_ids", "of_id", "of_source_ids", "of_component_of",
                    "of_produces",
                    "of_value", "of_value_unit", "of_value_label",
                ):
                    if _fld in it and it.get(_fld) is not None:
                        _entry[_fld] = it[_fld]
                out.append(_entry)
                key_to_index[key] = len(out) - 1

            return out

        def _drop_section_heading_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Drop broad section-level criteria when granular micro-criteria exist.

            Many marking guides include a broad, multi-mark "do the whole section" criterion
            alongside numerous micro-criteria. Keeping both encourages double-counting and
            causes PDF marks to anchor to headings.

            This is intentionally heuristic and conservative.
            """
            if not isinstance(criteria, list) or not criteria:
                return criteria

            enabled = os.getenv("DROP_SECTION_HEADING_CRITERIA", "1").strip().lower() not in {"0", "false", "no", "n"}
            if not enabled:
                return criteria

            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "as", "is", "are",
                "was", "were", "be", "been", "being", "with", "from", "by", "this", "that", "these",
                "those", "must", "should", "would", "will", "student", "marks", "mark",
            }
            verbs = (
                "prepare",
                "calculate",
                "compute",
                "present",
                "explain",
                "discuss",
                "show",
                "derive",
                "evaluate",
            )

            def _token_set(desc: str) -> set[str]:
                d = _norm_desc_key(desc)
                if not d:
                    return set()
                toks = [w for w in re.findall(r"[a-z]{4,}", d) if w not in stop]
                return set(toks)

            # Categories that explicitly mark a criterion as a primary marking point.
            # When set, we never heuristically flag the criterion as a "broad section heading"
            # - the rubric author has told us it's a real criterion to grade against.
            PRIMARY_CATEGORIES = {"calculation", "narrative", "journal"}

            # Identify "broad" candidates.
            broad_idxs: list[int] = []
            broad_startswith_verb: set[int] = set()
            forced_drop_idxs: set[int] = set()  # category=="section_header" → unconditional drop
            token_sets: list[set[str]] = []
            for i, it in enumerate(criteria):
                desc = str((it or {}).get("description", "") or "").strip()
                marks = (it or {}).get("marks")
                marks_num: Optional[float] = float(marks) if isinstance(marks, (int, float)) else None
                toks = _token_set(desc)
                token_sets.append(toks)

                if not desc or marks_num is None:
                    continue

                # Trust the explicit `category` marker when present.
                # - "section_header" → unconditionally drop (the author has declared this is
                #   metadata defining the section cap, not a scoring criterion).
                # - Any other set category ("calculation", "narrative", "journal") → never
                #   flag as broad. It's an explicit primary criterion regardless of its
                #   description length or marks.
                explicit_category = str((it or {}).get("category", "") or "").strip().lower()
                if explicit_category == "section_header":
                    forced_drop_idxs.add(i)
                    continue
                if explicit_category in PRIMARY_CATEGORIES:
                    continue

                dnorm = _norm_desc_key(desc)
                starts = any(dnorm.startswith(v + " ") for v in verbs)
                looks_broad = (marks_num >= 2.0) and (len(desc) >= 70 or starts)

                # Also flag short non-numeric "title" criteria with marks >= 1.0
                # e.g., "Electrostatic spraying room" (2/2 = 1.0). These are section
                # headings whose marks should be covered by their sub-criteria.
                desc_words = dnorm.split()
                if (
                    not looks_broad
                    and marks_num >= 1.0
                    and len(desc_words) <= 5
                    and not re.search(r"\d", dnorm)
                    and (not desc_words or desc_words[0] not in ("dr", "cr"))
                ):
                    looks_broad = True

                if looks_broad:
                    broad_idxs.append(i)
                    if starts:
                        broad_startswith_verb.add(i)

            if not broad_idxs and not forced_drop_idxs:
                return criteria

            # Decide which broad criteria have enough overlapping micro-criteria to justify dropping.
            # Start with the explicitly-tagged section_header criteria (always drop those).
            drop: set[int] = set(forced_drop_idxs)
            for i in broad_idxs:
                toks_i = token_sets[i]
                if not toks_i:
                    continue
                needed_overlaps = 1 if i in broad_startswith_verb else 3
                overlap_count = 0
                overlapping_marks_sum = 0.0
                broad_marks = float((criteria[i] or {}).get("marks", 0) or 0) if isinstance((criteria[i] or {}).get("marks"), (int, float)) else 0.0

                # Window-based micro-coverage heuristic (generic, order-aware):
                # Many mark schemes include a broad verb-led criterion (e.g. "Calculate EPS" 4 marks)
                # followed immediately by numerous micro-criteria totalling those marks. These micro
                # criteria often share few keywords with the broad sentence (lots of numeric-only lines),
                # so token overlap alone can fail. If nearby micro-criteria cover most of the marks,
                # drop the broad criterion to prevent double-counting and misleading annotations.
                if i in broad_startswith_verb and broad_marks >= 2.0:
                    window = 18
                    nearby_sum = 0.0
                    nearby_count = 0
                    for j in range(max(0, i - window), min(len(criteria), i + window + 1)):
                        if j == i:
                            continue
                        mj = (criteria[j] or {}).get("marks")
                        if not isinstance(mj, (int, float)):
                            continue
                        mj = float(mj)
                        if mj <= 0 or mj > 1.0:
                            continue
                        nearby_sum += mj
                        nearby_count += 1
                    if nearby_count >= 4 and nearby_sum >= broad_marks * 0.75:
                        drop.add(i)
                        continue

                for j, it in enumerate(criteria):
                    if i == j:
                        continue
                    desc_j = str((it or {}).get("description", "") or "").strip()
                    if not desc_j:
                        continue
                    marks_j = (it or {}).get("marks")
                    marks_j_num: Optional[float] = float(marks_j) if isinstance(marks_j, (int, float)) else None

                    toks_j = token_sets[j]
                    if len(toks_i & toks_j) < 2:
                        continue

                    # Count overlaps primarily against smaller/micro criteria.
                    if (marks_j_num is not None and marks_j_num <= 1.0) or len(desc_j) < 70:
                        overlap_count += 1
                        if marks_j_num is not None:
                            overlapping_marks_sum += marks_j_num
                        if overlap_count >= needed_overlaps:
                            break

                # Only drop if overlapping micro-criteria can cover at least half the
                # broad criterion's marks.  This prevents dropping a "Prepare SOCIE"
                # criterion worth 4 marks when the only overlap is a single vague
                # sub-criterion - keeping it ensures the table actually gets graded.
                if overlap_count >= needed_overlaps and overlapping_marks_sum >= broad_marks * 0.5:
                    drop.add(i)

            if not drop:
                return criteria

            filtered = [it for k, it in enumerate(criteria) if k not in drop]
            logger.info(f"Dropped {len(drop)} broad section-heading criteria to prevent double counting.")
            return filtered

        def _drop_commentary_criteria(criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
            """Drop non-marking commentary mistakenly extracted as criteria.

            Some PDFs embed solution commentary inside marking criteria blocks (eg "Tutorial note",
            "Proof of adjustment", or narrative observations like "appears to have been correctly
            dealt with"). These are not independent marking points and inflate scores.

            Heuristic + conservative: only drop when the phrase strongly signals commentary.
            """
            if not isinstance(criteria, list) or not criteria:
                return criteria

            enabled = os.getenv("DROP_COMMENTARY_CRITERIA", "1").strip().lower() not in {"0", "false", "no", "n"}
            if not enabled:
                return criteria

            commentary_phrases = (
                "tutorial note",
                "proof of adjustment",
                "appears to have been correctly dealt with",
                "this appears to have been correctly dealt with",
                "alternative assumptions",
                "alternative assumption",
            )

            out: list[dict[str, Any]] = []
            dropped = 0
            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                if not desc:
                    continue
                # A criterion_id means a human authored this criterion, so its
                # wording is deliberate. These phrases only signal commentary in
                # rubrics SCRAPED from a marking guide; an authored criterion may
                # legitimately carry a "TUTORIAL NOTE:" or note that a figure
                # "appears to have been correctly dealt with" as part of what it
                # is testing. Matching on a substring deleted those whole.
                if it.get("criterion_id"):
                    out.append(it)
                    continue
                dnorm = _norm_desc_key(desc)
                if any(p in dnorm for p in commentary_phrases):
                    dropped += 1
                    logger.debug(f"Dropped commentary criterion: {desc[:80]}")
                    continue
                out.append(it)

            if dropped:
                logger.info(f"Dropped {dropped} commentary criteria (non-marking text extracted as criteria).")
            return out


        def normalize_marks_value(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)

            text = str(value).strip()

            # Mixed number like "3 1/2"
            mixed = re.fullmatch(r"(\d+)\s+(\d+)\s*/\s*(\d+)", text)
            if mixed:
                whole = float(mixed.group(1))
                num = float(mixed.group(2))
                den = float(mixed.group(3))
                if den != 0:
                    return whole + (num / den)
                return None

            # Fraction like 1/2, 3/12
            frac = re.fullmatch(r"(\d+)\s*/\s*(\d+)", text)
            if frac:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den != 0:
                    return num / den
                return None

            # Fraction at start of a longer marking note, e.g. "1/2 mk each max 4"
            frac_prefix = re.match(r"^(\d+)\s*/\s*(\d+)", text)
            if frac_prefix:
                num = float(frac_prefix.group(1))
                den = float(frac_prefix.group(2))
                if den != 0:
                    return num / den
                return None

            # Plain numeric string
            num_match = re.fullmatch(r"\d+(?:\.\d+)?", text)
            if num_match:
                return float(text)

            # Keep as string for patterns we can't safely normalize
            return value

        def collect_numeric_marks(criteria_items: Any, out: list[float]) -> None:
            if not isinstance(criteria_items, list):
                return
            for it in criteria_items:
                if not isinstance(it, dict):
                    continue
                m = normalize_marks_value(it.get("marks"))
                if isinstance(m, (int, float)):
                    out.append(float(m))
                sub = it.get("sub_criteria")
                if isinstance(sub, list) and sub:
                    collect_numeric_marks(sub, out)

        def iter_answer_nodes(node: Any):
            """Yield answer/sub_answer nodes recursively."""
            if not isinstance(node, dict):
                return
            yield node
            for child in (node.get("sub_answers") or []):
                if isinstance(child, dict):
                    yield from iter_answer_nodes(child)

        # If we have granular micro-criteria, drop broad criteria (e.g. 26 marks learning outcomes)
        # to avoid polluting granular grading.
        all_marks: list[float] = []
        for ans in answers:
            for node in iter_answer_nodes(ans):
                collect_numeric_marks(node.get("marking_criteria"), all_marks)

        has_micro_criteria = any(m < 5 for m in all_marks)

        def flatten_criteria_items(
            criteria_items: Any,
            parent_description: Optional[str] = None,
            sibling_count: int = 0,
        ) -> None:
            if not isinstance(criteria_items, list):
                return

            for criteria_item in criteria_items:
                if not isinstance(criteria_item, dict):
                    continue

                description = str(criteria_item.get("description", "")).strip()
                raw_marks = criteria_item.get("marks")
                marks = normalize_marks_value(raw_marks)
                sub_criteria = criteria_item.get("sub_criteria")

                # Skip entire group when the parent is a known junk OCR/handwriting artefact.
                # Sub-criteria under junk parents (e.g. "Handwritten annotation" → "New York Wheels")
                # are section headings misidentified as criteria, not real marking points.
                _junk_parent_labels = {
                    "handwritten annotation", "annotation", "hr", "told - land.", "told - land",
                    "tutor note", "tutorial note", "marking guide", "mark scheme",
                    "scenario", "memorandum", "memo",
                }
                if description.strip().lower() in _junk_parent_labels and isinstance(sub_criteria, list) and sub_criteria:
                    continue

                # Expand compound criteria like "1/2 mk each max 4" where description
                # contains multiple semicolon/newline-separated line-items.
                if isinstance(raw_marks, str) and re.search(r"\bmk\s*each\b", raw_marks, flags=re.IGNORECASE):
                    per_mark = normalize_marks_value(raw_marks)
                    if isinstance(per_mark, (int, float)) and description:
                        # Extract optional "max N" cap from marks string.
                        max_match = re.search(r"\bmax\s+(\d+(?:\.\d+)?)", raw_marks, flags=re.IGNORECASE)
                        max_total = float(max_match.group(1)) if max_match else None

                        parts: list[str] = []
                        if ";" in description or "\n" in description:
                            parts = [p.strip() for p in re.split(r";|\n", description) if p and p.strip()]
                        else:
                            # Try to split number-heavy descriptions (e.g. SOCIE closing balances:
                            # "80,000 48,000 85,453 1,920 0 0 215,373") into individual number criteria.
                            nums = re.findall(r"\d{1,3}(?:,\d{3})+|\d{4,}", description)
                            if len(nums) >= 3:
                                parts = nums

                        if parts:
                            count = 0
                            for p in parts:
                                if max_total and count * float(per_mark) >= max_total:
                                    break
                                if not self._is_valid_criterion(p):
                                    # Pure numbers are always valid criteria in "mk each" context.
                                    if not re.search(r"\d", p):
                                        continue
                                combined_criteria.append({
                                    "marks": float(per_mark),
                                    "description": p,
                                })
                                count += 1
                            continue
                        elif max_total:
                            # Can't split, but we have a max cap - create one criterion
                            # worth the full max so the LLM can grade holistically.
                            # Only add if the description is meaningful (skip junk like
                            # "handwritten note" which the LLM cannot grade against).
                            if self._is_valid_criterion(description):
                                combined_criteria.append({
                                    "marks": float(max_total),
                                    "description": description,
                                })
                            continue

                # Skip non-numeric marks for grading (e.g., handwritten notes like "HR").
                # These should not be part of numeric breakdown scoring.
                if marks is not None and not isinstance(marks, (int, float)):
                    continue

                # When micro-criteria exist elsewhere, skip broad criteria (>=5 marks).
                if has_micro_criteria and isinstance(marks, (int, float)) and marks >= 5:
                    continue

                if isinstance(sub_criteria, list) and sub_criteria:
                    parent_desc = description if description else (parent_description or "")
                    # Don't propagate junk parent labels from OCR/handwriting artefacts.
                    if not self._is_valid_criterion(parent_desc):
                        parent_desc = ""

                    flattened_subs: list[dict[str, Any]] = []
                    for sub in sub_criteria:
                        if not isinstance(sub, dict):
                            continue

                        sub_desc = str(sub.get("description", "")).strip()
                        if parent_desc and sub_desc:
                            combined_desc = f"{parent_desc} - {sub_desc}"
                        else:
                            combined_desc = sub_desc or parent_desc

                        sub_marks = normalize_marks_value(sub.get("marks"))
                        if sub_marks is None:
                            continue
                        if not self._is_valid_criterion(combined_desc):
                            continue

                        _sub_entry: dict[str, Any] = {
                            "marks": sub_marks,
                            "description": combined_desc,
                        }
                        for _fld in (
                            "criterion_id",
                            "of_ids", "of_id", "of_source_ids",
                            "of_value", "of_value_unit", "of_value_label",
                            "exact_match",
                        ):
                            if _fld in sub and sub.get(_fld) is not None:
                                _sub_entry[_fld] = sub[_fld]
                        flattened_subs.append(_sub_entry)

                    if flattened_subs:
                        combined_criteria.extend(flattened_subs)
                        continue

                    # Sub_criteria exist but none produced usable criteria (e.g., all had
                    # marks=None or failed validation).  Keep the parent as a leaf criterion
                    # if it is itself valid and has marks, so the LLM can still grade against
                    # it.  Example: "Prepare a revised SOCIE" (4 marks) whose only sub-criterion
                    # was a junk handwritten annotation - dropping the parent would lose all
                    # marks for that section.
                    if description and isinstance(marks, (int, float)) and marks > 0:
                        if self._is_valid_criterion(description):
                            _parent_entry: dict[str, Any] = {
                                "marks": marks,
                                "description": description,
                            }
                            # Carry the parent's own metadata. Rebuilding a bare
                            # dict here stripped `category`, so a section_header
                            # came back as a gradeable criterion and the section
                            # cap could be awarded a second time on top of its
                            # own per-row criteria.
                            for _fld in (
                                "criterion_id", "category", "exact_match",
                                "of_ids", "of_id", "of_source_ids",
                                "of_component_of", "of_produces",
                                "of_value", "of_value_unit", "of_value_label",
                            ):
                                if criteria_item.get(_fld) is not None:
                                    _parent_entry[_fld] = criteria_item[_fld]
                            combined_criteria.append(_parent_entry)
                    continue

                # Leaf criterion
                if parent_description and description:
                    description = f"{parent_description} - {description}"
                elif parent_description and not description:
                    description = parent_description

                # If marks are missing and we have other criteria, skip to avoid generic/unweighted points.
                if marks is None and sibling_count > 1:
                    continue

                # Drop invalid/non-descriptive criteria to avoid polluting the grader.
                if not self._is_valid_criterion(description):
                    continue

                # Drop short non-numeric titles with high marks - these are section headings
                # (e.g., "Electrostatic spraying room" 2/2=1.0) whose marks overlap with sub-criteria.
                if isinstance(marks, (int, float)) and marks >= 1.0:
                    desc_words_flat = description.lower().split()
                    desc_stripped = re.sub(r"^\(\d+\)\s*", "", description.lower()).strip()
                    desc_stripped = re.sub(r"^\d+[.)]\s*", "", desc_stripped).strip()
                    if (
                        len(desc_stripped.split()) <= 5
                        and not re.search(r"\d", desc_stripped)
                        and (not desc_words_flat or desc_words_flat[0] not in ("dr", "cr"))
                    ):
                        logger.debug(f"Skipping section heading criterion: '{description}' ({marks} marks)")
                        continue

                _cat = str(criteria_item.get("category", "") or "").strip()
                _crit_entry: dict[str, Any] = {"marks": marks, "description": description}
                if _cat:
                    _crit_entry["category"] = _cat
                # Preserve OF metadata + exact_match flag for downstream caching.
                for _fld in (
                    "criterion_id",
                    "of_ids", "of_id", "of_source_ids", "of_component_of",
                    "of_produces",
                    "of_value", "of_value_unit", "of_value_label",
                    "exact_match",
                ):
                    if _fld in criteria_item and criteria_item.get(_fld) is not None:
                        _crit_entry[_fld] = criteria_item[_fld]
                combined_criteria.append(_crit_entry)

        def _collect_answer_text(node: dict, parts_list: list) -> None:
            """Recursively collect non-empty answer text from a model-answer node and its sub_answers."""
            if not isinstance(node, dict):
                return
            text = node.get("answer")
            label = str(node.get("question_number", "")).strip()
            if isinstance(text, str) and text.strip():
                parts_list.append(f"Part {label}\n{text.strip()}" if label else text.strip())
            for child in (node.get("sub_answers") or []):
                _collect_answer_text(child, parts_list)

        for answer in answers:
            if not isinstance(answer, dict):
                continue

            part_label = str(answer.get("question_number", "")).strip()
            answer_text = answer.get("answer")
            if isinstance(answer_text, str) and answer_text.strip():
                if part_label:
                    combined_answer_parts.append(f"Part {part_label}\n{answer_text.strip()}")
                else:
                    combined_answer_parts.append(answer_text.strip())

            # Also collect text from sub_answers (hierarchical structure for theoretical papers)
            for child in (answer.get("sub_answers") or []):
                _collect_answer_text(child, combined_answer_parts)

            criteria = answer.get("marking_criteria")
            sibling_count = len(criteria) if isinstance(criteria, list) else 0
            flatten_criteria_items(criteria, sibling_count=sibling_count)

            # IMPORTANT: include nested sub_answer criteria (these contain most micro-marking points)
            for node in (answer.get("sub_answers") or []):
                for sub_node in iter_answer_nodes(node):
                    sub_criteria = sub_node.get("marking_criteria")
                    sub_sibling_count = len(sub_criteria) if isinstance(sub_criteria, list) else 0
                    flatten_criteria_items(sub_criteria, sibling_count=sub_sibling_count)

        # Optional compaction (OFF by default): some marking guides create extremely granular criteria.
        # Use COMPACT_ABC_GRADING=1 to enable the older compact behaviour.
        if os.getenv("COMPACT_ABC_GRADING", "").strip() in {"1", "true", "yes"}:
            # (We intentionally keep compaction disabled by default to preserve granular marking.)
            pass

        if not combined_criteria and not combined_answer_parts:
            return model_data

        # ── Holistic grading: for theoretical questions OR when no criteria exist ──
        self._criteria_were_synthesized = False
        self._holistic_grading = False
        use_holistic = (
            (self.question_type == "theoretical" and combined_answer_parts)
            or (not combined_criteria and combined_answer_parts)
        )
        if use_holistic:
            reason = f"question_type='{self.question_type}'" if self.question_type == "theoretical" else "no marking_criteria found"
            logger.info(
                f"Switching to HOLISTIC grading mode ({reason}) - "
                f"full answer comparison instead of per-criterion"
            )
            self._holistic_grading = True

            # Build sub-question structure from model answer for holistic prompt.
            # Each answer entry with a distinct question_number becomes a sub-question.

            # Build a lookup of sub-question marks from the question paper (authoritative).
            # This covers both direct sub_questions and nested structures.
            _paper_sq_marks: dict[str, float] = {}
            if isinstance(questions_data, dict):
                def _collect_sq_marks(sq_list: list) -> None:
                    for sq in sq_list:
                        if not isinstance(sq, dict):
                            continue
                        sq_label = str(sq.get("question_number", "")).strip()
                        raw_marks = sq.get("marks")
                        if sq_label and raw_marks is not None:
                            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw_marks))]
                            if nums:
                                _paper_sq_marks[sq_label] = max(nums)
                        nested = sq.get("sub_questions")
                        if nested:
                            _collect_sq_marks(nested)

                for q in (questions_data.get("questions") or []):
                    if isinstance(q, dict):
                        _collect_sq_marks(q.get("sub_questions") or [])

            holistic_subs: list[dict] = []

            def _add_holistic_sub(node: dict, parent_section: Optional[str] = None, section_cap: Optional[float] = None) -> None:
                """Recursively add leaf sub-questions to holistic_subs.

                For hierarchical model answers (theoretical papers), parent sections
                have answer='' and carry their sub-sections in sub_answers.  We recurse
                until we reach leaf nodes (non-empty answer text) and add those.
                The parent's subsection_max is forwarded as section_cap so the prompt
                can enforce cross-sub-question caps.
                """
                if not isinstance(node, dict):
                    return
                sq_num = str(node.get("question_number", "")).strip()
                sq_answer = str(node.get("answer", "") or "").strip()
                sub_list = node.get("sub_answers") or []

                if sub_list:
                    # Parent section: pass its subsection_max down as the cap for children
                    raw_cap = node.get("subsection_max")
                    child_cap = float(raw_cap) if raw_cap is not None else section_cap
                    for child in sub_list:
                        _add_holistic_sub(child, parent_section=sq_num, section_cap=child_cap)
                    return

                if not (sq_num and sq_answer):
                    return

                # Leaf node - resolve max_marks.
                # Priority: question-paper marks > maximum_marks / subsection_max > total_marks_available
                sq_marks = _paper_sq_marks.get(sq_num, 0.0)
                if not sq_marks:
                    for marks_key in ("maximum_marks", "subsection_max", "marks"):
                        raw = node.get(marks_key)
                        if raw is not None:
                            nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))]
                            if nums:
                                sq_marks = max(nums)
                                break
                if not sq_marks:
                    # Last resort: total_marks_available (may exceed section cap, but better than 0)
                    raw = node.get("total_marks_available")
                    if raw is not None:
                        nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))]
                        if nums:
                            sq_marks = max(nums)

                entry: dict = {
                    "sub_question": sq_num,
                    "answer": sq_answer,
                    "max_marks": sq_marks,
                    "marking_criteria": node.get("marking_criteria") or [],
                }
                rule = node.get("marking_rule")
                if rule:
                    entry["marking_rule"] = rule
                if parent_section is not None:
                    entry["parent_section"] = parent_section
                if section_cap is not None:
                    entry["section_cap"] = section_cap
                holistic_subs.append(entry)

            for answer in answers:
                _add_holistic_sub(answer)

            # If only one sub-question or all share the same number, treat as single block
            if len(holistic_subs) <= 1:
                # Single block: use the main question number
                full_answer = "\n\n".join(combined_answer_parts)
                total_marks = holistic_subs[0]["max_marks"] if holistic_subs else 0.0
                self._holistic_sub_questions = [{
                    "sub_question": self.question_number,
                    "answer": full_answer,
                    "max_marks": total_marks,
                }]
            else:
                self._holistic_sub_questions = holistic_subs

            logger.info(
                f"Holistic grading: {len(self._holistic_sub_questions)} sub-question(s) detected"
            )

            # For holistic mode, we still build a unified answer for the prompt
            # but WITHOUT marking_criteria - the LLM will compare holistically.
            unified_answer = {
                "question_number": self.question_number,
                "answer": "\n\n".join(combined_answer_parts),
                "marking_criteria": [],  # Empty - holistic mode
                "sub_questions": self._holistic_sub_questions,
            }

            flattened = dict(model_data)
            flattened["answers"] = [unified_answer]
            logger.info(
                f"Holistic model data prepared → {len(self._holistic_sub_questions)} sub-questions"
            )
            return flattened

        # ── Criteria exist: standard per-criterion path ──

        # Deduplicate criteria to stabilize marking and prevent repeated grading.
        combined_criteria = _dedup_criteria(combined_criteria)

        # Drop broad section headings when micro-criteria exist.
        combined_criteria = _drop_section_heading_criteria(combined_criteria)

        # Drop non-marking commentary accidentally extracted as criteria.
        combined_criteria = _drop_commentary_criteria(combined_criteria)

        unified_answer = {
            "question_number": self.question_number,
            "answer": "\n\n".join(combined_answer_parts),
            "marking_criteria": combined_criteria,
        }

        # Collect of_definitions from every sub_answer (and any top-level answer)
        # into a single dict on the unified answer, so _cache_rubric_criteria can
        # find virtual-origin OFs regardless of where they lived in the source doc.
        of_definitions_merged: dict[str, dict[str, Any]] = {}
        for _ans in answers:
            if not isinstance(_ans, dict):
                continue
            _top_defs = _ans.get("of_definitions")
            if isinstance(_top_defs, dict):
                of_definitions_merged.update(_top_defs)
            for _sa in (_ans.get("sub_answers") or []):
                if not isinstance(_sa, dict):
                    continue
                _sa_defs = _sa.get("of_definitions")
                if isinstance(_sa_defs, dict):
                    of_definitions_merged.update(_sa_defs)
        if of_definitions_merged:
            unified_answer["of_definitions"] = of_definitions_merged

        flattened = dict(model_data)
        flattened["answers"] = [unified_answer]

        logger.info(
            f"Unified model answers for grading → {len(combined_criteria)} criteria across {len(answers)} top-level answers"
            + (f", {len(of_definitions_merged)} virtual-origin OFs" if of_definitions_merged else "")
        )
        return flattened

    def _cache_rubric_criteria(self, model_data: dict) -> None:
        """Cache allowed criteria and their max marks for strict post-processing.

        We enforce rubric max marks at the post-processing stage and ignore any LLM-supplied
        max_possible values to prevent inflation or hallucinated criteria.
        """
        allowed: set[str] = set()
        max_map: dict[str, float] = {}
        cat_map: dict[str, str] = {}
        exact_match: set[str] = set()
        ordered: list[str] = []
        pos_map: dict[str, int] = {}
        id_map: dict[str, str] = {}
        of_ids_map: dict[str, list[str]] = {}
        of_source_map: dict[str, list[str]] = {}
        of_value_map: dict[str, dict[str, Any]] = {}
        of_defs: dict[str, dict[str, Any]] = {}
        of_component_map: dict[str, str] = {}
        of_produces_map: dict[str, int] = {}
        keywords_map: dict[str, list[str]] = {}
        expected_amount_map: dict[str, float] = {}
        sign_sensitive_set: set[str] = set()
        column_map: dict[str, str] = {}

        def _reset_of_caches() -> None:
            self._of_ids_by_criterion_last_run = {}
            self._of_source_ids_by_criterion_last_run = {}
            self._of_value_by_criterion_last_run = {}
            self._of_definitions_last_run = {}
            self._of_component_of_by_criterion_last_run = {}
            self._of_produces_by_criterion_last_run = {}
            self._keywords_by_criterion_last_run = {}
            self._expected_amount_by_criterion_last_run = {}
            self._sign_sensitive_criteria_last_run = set()
            self._column_by_criterion_last_run = {}

        try:
            answers = (model_data or {}).get("answers")
            if not isinstance(answers, list) or not answers:
                self._allowed_criteria_last_run = set()
                self._criterion_max_map_last_run = {}
                self._criterion_category_map_last_run = {}
                self._exact_match_criteria_last_run = set()
                self._rubric_criteria_order_last_run = []
                self._rubric_position_last_run = {}
                self._criterion_id_map_last_run = {}
                _reset_of_caches()
                return

            # Unified grading payload uses a single answer node.
            criteria = (answers[0] or {}).get("marking_criteria")
            if not isinstance(criteria, list):
                self._allowed_criteria_last_run = set()
                self._criterion_max_map_last_run = {}
                self._criterion_category_map_last_run = {}
                self._exact_match_criteria_last_run = set()
                self._rubric_criteria_order_last_run = []
                self._rubric_position_last_run = {}
                self._criterion_id_map_last_run = {}
                _reset_of_caches()
                return

            # Pull sub-answer-level of_definitions merged onto the unified answer
            # (see _flatten_model_data). Virtual-origin OFs (working totals that
            # aren't criteria) live here.
            _defs = (answers[0] or {}).get("of_definitions")
            if isinstance(_defs, dict):
                for _k, _v in _defs.items():
                    if isinstance(_v, dict):
                        of_defs[str(_k)] = _v

            for it in criteria:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get("description", "") or "").strip()
                marks = it.get("marks")

                if not desc:
                    continue
                if not self._is_valid_criterion(desc):
                    continue

                # Only numeric marks are scoreable.
                if not isinstance(marks, (int, float)):
                    continue
                max_marks = float(marks)
                if max_marks < 0:
                    continue

                allowed.add(desc)
                # CRITERION ID -> canonical description.
                # The model returns the id rather than retyping the whole
                # description, so resolution is exact and a near-identical
                # twin can no longer absorb another criterion's marks.
                _cid = str(it.get("criterion_id", "") or "").strip()
                if _cid and _cid not in id_map:
                    id_map[_cid] = desc
                if desc not in pos_map:
                    pos_map[desc] = len(ordered)
                    ordered.append(desc)
                # In case of duplicates, keep the maximum.
                prev = max_map.get(desc)
                if prev is None or max_marks > prev:
                    max_map[desc] = max_marks
                # Cache category from rubric (LLM output never includes this field).
                cat_val = str(it.get("category", "") or "").strip().lower()
                if cat_val and desc not in cat_map:
                    cat_map[desc] = cat_val
                # Cache exact_match flag - disables OF bypass for this criterion.
                if it.get("exact_match"):
                    exact_match.add(desc)

                # Capture OF metadata (supports both new schema `of_ids: list` and
                # legacy `of_id: scalar`).
                _ids_raw = it.get("of_ids")
                if isinstance(_ids_raw, list) and _ids_raw:
                    of_ids_map[desc] = [str(x) for x in _ids_raw if x is not None]
                elif it.get("of_id"):
                    of_ids_map[desc] = [str(it.get("of_id"))]
                _src_raw = it.get("of_source_ids")
                if isinstance(_src_raw, list) and _src_raw:
                    of_source_map[desc] = [str(x) for x in _src_raw if x is not None]
                _val_raw = it.get("of_value")
                if _val_raw is not None:
                    of_value_map[desc] = {
                        "value": _val_raw,
                        "unit": it.get("of_value_unit"),
                        "label": it.get("of_value_label"),
                    }
                _comp_raw = it.get("of_component_of")
                if _comp_raw:
                    of_component_map[desc] = str(_comp_raw)
                _prod_raw = it.get("of_produces")
                if _prod_raw is not None:
                    try:
                        of_produces_map[desc] = int(round(float(_prod_raw)))
                    except (TypeError, ValueError):
                        pass

                # Authored fields the post-processing guards rely on. These
                # have always been present in the rubric but were never read.
                _kw_raw = it.get("keywords")
                if isinstance(_kw_raw, list) and _kw_raw:
                    keywords_map[desc] = [str(x) for x in _kw_raw if x is not None]
                _amt_raw = it.get("expected_amount")
                if _amt_raw is not None:
                    try:
                        expected_amount_map[desc] = float(_amt_raw)
                    except (TypeError, ValueError):
                        pass
                if it.get("sign_sensitive"):
                    sign_sensitive_set.add(desc)
                _col_raw = str(it.get("column", "") or "").strip()
                if _col_raw:
                    column_map[desc] = _col_raw

        finally:
            self._allowed_criteria_last_run = allowed
            self._criterion_max_map_last_run = max_map
            self._criterion_category_map_last_run = cat_map
            self._exact_match_criteria_last_run = exact_match
            self._rubric_criteria_order_last_run = ordered
            self._rubric_position_last_run = pos_map
            self._criterion_id_map_last_run = id_map
            self._of_ids_by_criterion_last_run = of_ids_map
            self._of_source_ids_by_criterion_last_run = of_source_map
            self._of_value_by_criterion_last_run = of_value_map
            self._of_definitions_last_run = of_defs
            self._of_component_of_by_criterion_last_run = of_component_map
            self._of_produces_by_criterion_last_run = of_produces_map
            self._keywords_by_criterion_last_run = keywords_map
            self._expected_amount_by_criterion_last_run = expected_amount_map
            self._sign_sensitive_criteria_last_run = sign_sensitive_set
            self._column_by_criterion_last_run = column_map

    def _find_restatements(
        self, credited_by_id: dict[str, dict]
    ) -> list[tuple[str, str]]:
        """Where does the student repeat a point that was already credited?

        Returns (criterion_id, student_line) pairs. Best-effort: any failure
        yields nothing, because a missing marker's note is a cosmetic loss
        while a wrong one actively misleads the student.
        """
        logger.info("=" * 62)
        logger.info("RESTATEMENT PASS - finding repeats of already-credited points")
        logger.info("=" * 62)
        if not credited_by_id:
            logger.info("  SKIPPED: no criterion scored above 0, nothing to point at")
            return []
        if self.restatement_chain is None:
            logger.info("  SKIPPED: chain unavailable (provider has no structured output)")
            return []
        student_text = getattr(self, "_student_text_last_run", "") or ""
        if not student_text.strip():
            logger.info("  SKIPPED: student text empty for this run")
            return []

        # Credited points, grouped by the WORKING each belongs to. A criterion
        # description ends with "CONTEXT - <the working>", shared verbatim by
        # every row of that working, so it groups them and carries the
        # working's own result - which is what a prose line usually restates.
        def _ctx_key(desc: str) -> str:
            _m = re.search(r"CONTEXT\s*[-\u2014:]?\s*(.+)$", desc, re.S)
            return re.sub(r"\s+", " ", _m.group(1)).strip()[:300] if _m else ""

        _groups: dict[str, list[tuple[str, str, str]]] = {}
        for _cid, _bi in sorted(credited_by_id.items()):
            _desc = re.sub(r"\s+", " ", str(_bi.get("criterion", "") or ""))
            _ev = str(_bi.get("evidence", "") or "").split(";")[0].strip()[:100]
            if not _ev:
                continue
            _groups.setdefault(_ctx_key(_desc), []).append(
                (_cid, _desc.split("CONTEXT")[0][:100], _ev)
            )

        _lines_out: list[str] = []
        for _ctx, _members in _groups.items():
            if _ctx:
                _lines_out.append(f"WORKING: {_ctx}")
            for _cid, _short, _ev in _members:
                _lines_out.append(f"   {_cid} | {_short} | CREDITED ON: {_ev}")
        if not _lines_out:
            return []

        # Which student lines already earn marks? Containment either way, so a
        # formatting difference cannot leave a scoring line in the candidate
        # set - that is how a note ends up printed on top of a score.
        _rows = [ln.strip() for ln in student_text.splitlines() if ln.strip()]
        _rows_norm = [_norm_for_evidence_match_static(_r) for _r in _rows]
        _paid_idx: set[int] = set()
        for _bi in credited_by_id.values():
            for _e in (_bi.get("evidence_list") or []):
                _en = _norm_for_evidence_match_static(_e)
                if not _en:
                    continue
                _exact = [_i for _i, _rn in enumerate(_rows_norm) if _rn and _rn == _en]
                if _exact:
                    _paid_idx.update(_exact)
                    continue
                # No exact row: the snippet may join or trim one. Claim the
                # single closest row, never every row it happens to contain -
                # a short row repeated later in the script must stay available.
                _cands = [_i for _i, _rn in enumerate(_rows_norm)
                          if _rn and (_rn in _en or _en in _rn)]
                if _cands:
                    _paid_idx.add(max(_cands, key=lambda _i: len(_rows_norm[_i])))
        _unmarked = [_r for _i, _r in enumerate(_rows)
                     if _i not in _paid_idx and _rows_norm[_i]]
        if not _unmarked:
            logger.info("  every student line already earns marks - nothing to flag")
            return []

        logger.info(f"  sending {len(credited_by_id)} credited point(s) in "
                    f"{len(_groups)} working(s); {len(_unmarked)} of {len(_rows)} "
                    f"line(s) earn nothing and are up for decision")
        try:
            _result = self.restatement_chain.invoke({
                "credited": "\n".join(_lines_out),
                "unmarked_lines": "\n".join(
                    f"{_i + 1}. {_l}" for _i, _l in enumerate(_unmarked)
                ),
            })
        except Exception as e:
            logger.warning(f"  FAILED: {e} - grading is unaffected, no notes added")
            return []

        if isinstance(_result, BaseModel):
            _result = _result.model_dump()
        if not isinstance(_result, dict):
            return []

        _raw = _result.get("restatements") or []
        logger.info(f"  model returned {len(_raw)} candidate line(s)")
        _out: list[tuple[str, str]] = []
        for _item in _raw:
            if not isinstance(_item, dict):
                logger.info(f"    DROP  malformed entry: {_item!r:.80}")
                continue
            _cid = str(_item.get("criterion_id", "") or "").strip().upper()
            _line = str(_item.get("line", "") or "").strip()
            _why = str(_item.get("why", "") or "").strip()
            if not _line:
                logger.info(f"    DROP  [{_cid or '?'}] empty line")
                continue
            if _cid not in credited_by_id:
                logger.info(f"    DROP  [{_cid or '?'}] unknown or uncredited id "
                            f"| {_line[:60]}")
                continue
            logger.info(f"    KEEP  [{_cid}] {_line[:60]}")
            if _why:
                logger.info(f"          why: {_why[:80]}")
            _out.append((_cid, _line))
        logger.info(f"  {len(_out)} candidate(s) passed id checks")
        return _out

    def _fetch_doc(self, collection_name: str, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch document by _id."""
        try:
            coll = get_collection(collection_name)
            doc = coll.find_one({"_id": ObjectId(doc_id)})
            if not doc:
                logger.warning(f"No document in {collection_name} for _id={doc_id}")
            return doc
        except Exception as e:
            logger.error(f"Failed to fetch {collection_name} {doc_id}: {e}", exc_info=True)
            return None

    def _clean_for_llm(self, doc: Optional[dict], allowed_keys: list[str]) -> dict:
        if not doc:
            return {}
        return {k: v for k, v in doc.items() if k in allowed_keys}

    def _is_valid_criterion(self, criterion: str) -> bool:
        if not isinstance(criterion, str):
            return False

        clean = criterion.strip().lower()
        if not clean:
            return False

        # Reject pure marking notation patterns
        # Examples: "1/2", "1/4", "2/2", "3/2", "1/2 mk each max 4", "3 1/2", "2 1/2"
        if re.match(r'^\d+/\d+(\s+(mk|marks?).*)?$', clean):
            return False

        # Reject "N 1/2" style mark notations (e.g. "3 1/2" = 3.5 marks)
        if re.match(r'^\d+\s+\d+/\d+\s*$', clean):
            return False

        # Reject if it's only marking notation variations
        if re.match(r'^\s*\d+\s*/\s*\d+\s*(mk|marks)?\s*(each|per)?\s*(max\s*\d+)?\s*$', clean):
            return False

        # Reject handwritten annotation / OCR artefact labels from the marking scheme PDF.
        # These are not real criteria - they are section headings or PDF annotation remnants.
        _junk_labels = {
            "handwritten annotation", "annotation", "hr", "told - land.", "told - land",
            "tutor note", "tutorial note", "marking guide", "mark scheme",
            "scenario", "memorandum", "memo",
        }
        if clean in _junk_labels:
            return False

        words = clean.split()

        # Strip numbered section prefixes like "(4)", "(1)", "1.", "2)" before heading checks.
        # These are section labels, not meaningful numeric content.
        heading_clean = re.sub(r"^\(\d+\)\s*", "", clean)
        heading_clean = re.sub(r"^\d+[.)]\s*", "", heading_clean)
        heading_words = heading_clean.split() if heading_clean else words
        had_number_prefix = (heading_clean != clean)

        # Reject numbered section headings like "(4) Electrostatic spraying room".
        # After stripping the number prefix, if the remaining text is short and non-numeric,
        # it's a section heading - not a grading criterion.
        if had_number_prefix and len(heading_words) <= 5 and not re.search(r"\d", heading_clean):
            if not heading_words or heading_words[0] not in {"dr", "cr"}:
                return False

        # Reject standalone topic/section headings that are just company or section names.
        # These appear when the PDF extractor picks up section titles as criteria.
        # Only reject if the text has no numbers, no Dr/Cr prefix, and looks like a plain heading.
        if len(heading_words) <= 4 and not re.search(r"\d", heading_clean) and heading_words[0] not in {"dr", "cr"}:
            _heading_indicators = {
                "revised", "consolidated", "statement", "changes", "equity",
                "prepare", "calculate", "determine", "explain",
            }
            if any(w in _heading_indicators for w in heading_words):
                return False

        # Journal entry line items are often short but meaningful (e.g., "Dr NCI", "Cr Disposal of subsidiary").
        # Accept common debit/credit prefixes.
        if len(words) >= 2 and words[0] in {"dr", "cr"}:
            return True

        # Accept longer, clearly descriptive criteria
        if len(words) >= 3:
            return True

        # Accept short criteria when they look like genuine line items / calculations
        # (numbers, currency, brackets, etc.)
        if re.search(r"\d", clean):
            return True
        if "(" in clean or ")" in clean:
            return True
        if any(tok in clean for tok in ("gbp", "usd", "eur", "percent")):
            return True

        # Allow-list for common short financial reporting labels
        short_allowlist = {
            "goodwill",
            "reserves",
            "depreciation",
            "impairment",
            "revaluation",
            "nci",
            "oci",
            "eps",
            "investment",
            "associate",
            "subsidiary",
            "sale proceeds",
            "fair value",
            "net assets",
            "share capital",
            "share premium",
            "retained earnings",
            "exchange gain",
            "revaluation gain",
            "revaluation loss",
        }
        if clean in short_allowlist:
            return True

        return False

    def _load_clean_data(self) -> Tuple[dict, dict, dict]:
        q_doc = self._fetch_doc("pac_questions", self.questions_id) if self.questions_id else {}
        m_doc = self._fetch_doc("model_answers", self.model_answers_id) if self.model_answers_id else {}
        s_doc = self._fetch_doc("student_assignments", self.student_answers_id)

        if not s_doc:
            raise GradingError(f"No student answer found for _id={self.student_answers_id}")

        # Only these fields go to LLM - metadata is completely excluded.
        # `max_marks` and `available_marks` are preserved on model_data so
        # _build_grade_doc's max-marks lookup can prefer max_marks (the
        # canonical question total set by the marker) over total_marks (a
        # legacy field that may hold an out-of-date rubric sum).
        q_clean = self._clean_for_llm(q_doc, ["question_title", "description", "total_marks", "questions"])
        m_clean = self._clean_for_llm(m_doc, ["question_title", "description", "total_marks", "max_marks", "available_marks", "answers"])
        s_clean = self._clean_for_llm(s_doc, ["question", "sub_parts"])

        # Grade holistically by combining all sub-answers/criteria into one payload.
        m_clean = self._flatten_model_answers(m_clean, q_clean)

        # Cache rubric criteria for strict post-processing.
        self._cache_rubric_criteria(m_clean)

        return q_clean, m_clean, s_clean

    def _normalize_floating_letter_labels(self, student_data: dict) -> dict:
        """Combine bare letter-labels ('a)', 'b)', '(i)') with their inferred
        numeric parent, in-place on a copy of student_data.

        Some extractions (especially older ones, or when the student omits the
        '4.1' heading because it's pre-printed on the question paper) emit
        sub_parts as 'a)', 'b)', '4.2', '4.3', '4.4' - losing the '4.1'
        parent. The grader then says "Student did not attempt 4.1" even though
        the content is there.

        Inference: if a letter-label appears BEFORE any numeric sub-label, its
        parent is "{main_q}.1". Letter-labels appearing between numeric labels
        inherit the MOST-RECENT prior numeric as their parent. Combined label
        becomes e.g. "4.1(a)" preserving the student's original casing.
        """
        if not isinstance(student_data, dict):
            return student_data
        sub_parts = student_data.get("sub_parts")
        if not isinstance(sub_parts, list) or not sub_parts:
            return student_data

        main_q = str(student_data.get("question", "")).strip() or str(self.question_number)
        if not main_q:
            return student_data

        # Detect: do any bare letter-labels appear BEFORE the first numeric label?
        letter_re = re.compile(r"^[\(\[]?([A-Za-z]+|[ivxIVX]+)[\)\]\.]?$")
        numeric_re = re.compile(r"^\d+(?:\.\d+)*[)\.]?$")

        has_leading_letters = False
        for sp in sub_parts:
            if not isinstance(sp, dict):
                continue
            lab = str(sp.get("question_number", "")).strip()
            if numeric_re.match(lab):
                break
            if letter_re.match(lab):
                has_leading_letters = True
                break

        # Initial inferred parent if there are leading letter-labels with no
        # numeric predecessor: "{main_q}.1".
        current_parent: Optional[str] = f"{main_q}.1" if has_leading_letters else None

        new_sub_parts: list[dict] = []
        renamed_log: list[str] = []
        for sp in sub_parts:
            if not isinstance(sp, dict):
                new_sub_parts.append(sp)
                continue
            lab = str(sp.get("question_number", "")).strip()

            if numeric_re.match(lab):
                # Top-level numeric: update parent context for following letters.
                current_parent = lab.rstrip(")").rstrip(".")
                new_sub_parts.append(sp)
                continue

            m = letter_re.match(lab)
            if m and current_parent:
                letter = m.group(1)
                # Preserve original brackets/casing minimally - combine as parent(letter).
                combined = f"{current_parent}({letter})"
                renamed_log.append(f"{lab!r}→{combined!r}")
                new_sp = dict(sp)
                new_sp["question_number"] = combined
                new_sub_parts.append(new_sp)
                continue

            # Anything else (e.g. already-combined "4.1(a)", or unusual label)
            # passes through unchanged.
            new_sub_parts.append(sp)

        if renamed_log:
            logger.info(
                f"Normalized floating letter-labels in student sub_parts: "
                f"{'; '.join(renamed_log)}"
            )

        out = dict(student_data)
        out["sub_parts"] = new_sub_parts
        return out

    def _format_student_for_prompt(self, student_data: dict) -> str:
        """Flatten student assignment JSON into readable text for the grader.

        Passing a raw dict into the prompt makes it harder for the LLM to reliably
        locate table rows and journal lines.
        """
        if not isinstance(student_data, dict):
            return str(student_data)

        # Normalize floating letter-labels BEFORE flattening into prompt text.
        # This is the fallback for students extracted before the parent-preserving
        # extraction prompt was deployed.
        student_data = self._normalize_floating_letter_labels(student_data)

        def _extract_question_id(label: str) -> Optional[str]:
            """Extract the canonical question number from a sub-part label.

            Handles formats like: "Q-01", "Q.1", "1", "1.1", "1-", "1)", "(a)",
            "Issue-01 Peak State" (not a question-level label).
            Returns the number as a string with leading zeros stripped, or None
            if this doesn't look like a question-level label.
            """
            if not label:
                return None
            lab = label.strip()

            # Skip sub-issue labels like "Issue-01 Peak State" - these are
            # sub-sections within a question, not question-level identifiers.
            if re.match(r"(?:issue|part|section|topic)\s*[-:]?\s*\d", lab, re.IGNORECASE):
                return None

            # "N-" / "N- Topic name" style labels (e.g. "1-", "2- Tech limited:")
            # are scenario/sub-part labels within a question, NOT question IDs.
            # Returning None lets all such sub_parts pass through the filter.
            if re.match(r"^\d+\s*-", lab):
                return None

            # Extract question number from patterns like Q-01, Q.1, Q1, 1), 1.1)
            m = re.match(
                r"^(?:Q(?:uestion)?\.?\s*[-:]?\s*)?(\d+)",
                lab,
                re.IGNORECASE,
            )
            if m:
                return str(int(m.group(1)))  # strip leading zeros: "01" -> "1"

            return None

        target_qid = _extract_question_id(str(self.question_number))
        # Also try plain digit extraction as fallback for target
        if not target_qid:
            digits = re.findall(r"\d+", str(self.question_number))
            target_qid = str(int(digits[0])) if digits else None

        parts: list[str] = []
        q = str(student_data.get("question", "")).strip()
        if q:
            parts.append(f"Question: {q}")

        sub_parts = student_data.get("sub_parts")
        if isinstance(sub_parts, list) and sub_parts:
            # Check if any sub_part has a question-level label matching the target
            any_matches = False
            if target_qid:
                for sp in sub_parts:
                    if not isinstance(sp, dict):
                        continue
                    lab = str(sp.get("question_number", "")).strip()
                    sp_qid = _extract_question_id(lab)
                    if sp_qid == target_qid:
                        any_matches = True
                        break

            in_relevant_block = False
            for sp in sub_parts:
                if not isinstance(sp, dict):
                    continue
                sp_no = str(sp.get("question_number", "")).strip() or q or "(unknown)"

                if any_matches and target_qid:
                    sp_qid = _extract_question_id(sp_no)
                    if sp_qid == target_qid:
                        in_relevant_block = True
                    elif sp_qid is not None and sp_qid != target_qid:
                        # Different question - stop including
                        in_relevant_block = False

                    # Sub-issue labels (Issue-01, etc.) are children of whatever
                    # question block we're currently in. Include them only if
                    # we're in a relevant block.
                    if sp_qid is None:
                        # This is a sub-issue or unlabeled part
                        if not in_relevant_block:
                            continue
                    elif not in_relevant_block:
                        continue

                ans = sp.get("answer")
                ans = ans if isinstance(ans, str) else str(ans or "")
                ans = ans.strip()
                if not ans:
                    continue
                # Preserve tables by keeping answers verbatim.
                parts.append(f"\n--- {sp_no} ---\n{ans}")

        if not parts:
            return json.dumps(student_data, ensure_ascii=False)

        return "\n".join(parts).strip()

    @staticmethod
    def _numbers_in_text(s: str) -> list[str]:
        if not s or not isinstance(s, str):
            return []
        # Pull out sequences that look like accounting numbers.
        return re.findall(r"\d[\d,]*(?:\.\d+)?", s)

    @staticmethod
    def _contains_number_variant(haystack: str, needle: str) -> bool:
        if not haystack or not needle:
            return False
        h = haystack.replace(",", "").replace(" ", "")
        n = needle.replace(",", "").replace(" ", "")
        if n in h:
            return True

        # Expand shorthand magnitudes: "7.2" from "GBP7.2m" may need to match "7200000".
        # Try common multiplier suffixes on the raw needle.
        for suffix, mult in [("m", 1_000_000), ("k", 1_000)]:
            expanded_needle = n + suffix
            parsed = StudentGrader._parse_number_token(expanded_needle)
            if parsed is not None and parsed > 100:
                int_val = int(round(parsed))
                if str(int_val) in h:
                    return True

        return False

    @staticmethod
    def _parse_number_token(token: str) -> Optional[float]:
        """Parse a single numeric token used in marking criteria.

        Supports common forms:
        - 375,000
        - 7.2m (million)
        - 480k (thousand)
        - 50p (pence -> 0.50)
        - 25% (percent -> 0.25)
        - 9/12 (fraction)
        """
        if not token or not isinstance(token, str):
            return None

        t = token.strip().lower()
        t = t.strip("()")
        t = t.replace("£", "").replace("$", "")
        # Handle currency prefixes like "usd3" as well as separate tokens like "USD 3".
        t = re.sub(r"^(gbp|usd|eur)", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\b(gbp|usd|eur)\b", " ", t, flags=re.IGNORECASE)

        # Word magnitudes (common in marking criteria): "3 million" / "8 thousand"
        mult_word = 1.0
        if re.search(r"\bmillion\b", t):
            mult_word = 1_000_000.0
            t = re.sub(r"\bmillion\b", " ", t)
        if re.search(r"\bthousand\b", t):
            mult_word = 1_000.0
            t = re.sub(r"\bthousand\b", " ", t)

        # Word "percent" / "per cent" → treat as % suffix
        t = re.sub(r"\bper\s*cent\b", "%", t)
        t = re.sub(r"\bpercent\b", "%", t)

        t = re.sub(r"\s+", " ", t).strip()
        t = t.replace(",", "").replace(" ", "")

        if not t:
            return None

        # Fraction like 9/12
        frac = re.fullmatch(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", t)
        if frac:
            try:
                num = float(frac.group(1))
                den = float(frac.group(2))
                if den == 0:
                    return None
                return num / den
            except Exception:
                return None

        # Percent
        if t.endswith("%"):
            try:
                return float(t[:-1]) / 100.0
            except Exception:
                return None

        # Pence
        if t.endswith("p") and re.fullmatch(r"\d+(?:\.\d+)?p", t):
            try:
                return float(t[:-1]) / 100.0
            except Exception:
                return None

        # Million / thousand suffixes
        mult = 1.0
        if t.endswith("m") and re.fullmatch(r"\d+(?:\.\d+)?m", t):
            mult = 1_000_000.0
            t = t[:-1]
        elif t.endswith("k") and re.fullmatch(r"\d+(?:\.\d+)?k", t):
            mult = 1_000.0
            t = t[:-1]

        # Plain float/int
        try:
            return float(t) * mult * mult_word
        except Exception:
            return None

    @staticmethod
    def _compute_simple_calc_from_criterion(criterion: str) -> Optional[float]:
        """Compute expected result for simple bracketed calculations.

        Examples:
        - "Cost of investment (375,000 x GBP32)" -> 12000000
        - "Share capital (500,000 x 50p)" -> 250000

        Only handles simple x/* and / expressions inside parentheses. Returns None if
        expression is composite/ambiguous.
        """
        if not criterion or not isinstance(criterion, str):
            return None

        m = re.search(r"\(([^)]*)\)", criterion)
        if not m:
            return None

        expr = m.group(1)
        expr_low = expr.lower()

        # Skip composite expressions; these often list components where evidence may show
        # only a final figure and we can't safely derive a single expected value.
        if "+" in expr_low or "-" in expr_low:
            return None

        # Normalize symbols
        expr_low = expr_low.replace("×", "x")
        expr_low = re.sub(r"\s+", " ", expr_low).strip()

        # Tokenize on operators while keeping them
        parts = re.split(r"\s*(x|\*|/)\s*", expr_low)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) < 3:
            return None

        # Expression must alternate: number, op, number, op, number ...
        # Validate quick
        if parts[1] not in {"x", "*", "/"}:
            return None

        try:
            acc = self_first = StudentGrader._parse_number_token(parts[0])
            if acc is None:
                return None
            i = 1
            while i < len(parts) - 1:
                op = parts[i]
                rhs = StudentGrader._parse_number_token(parts[i + 1])
                if rhs is None:
                    return None
                if op in {"x", "*"}:
                    acc = acc * rhs
                elif op == "/":
                    if rhs == 0:
                        return None
                    acc = acc / rhs
                else:
                    return None
                i += 2
            return acc
        except Exception:
            return None

    @staticmethod
    def _format_expected_number_variants(value: float) -> list[str]:
        """Return a small set of string variants for matching expected values in evidence."""
        try:
            v = float(value)
        except Exception:
            return []

        # If it's very close to an integer, treat it as one.
        if abs(v - round(v)) < 1e-6:
            iv = int(round(v))
            variants = [str(iv)]

            # Also provide compact million/thousand forms commonly used in workings.
            if iv % 1_000_000 == 0:
                m = iv // 1_000_000
                variants.extend([f"{m}m", f"{m} million"])
            elif iv % 1_000 == 0 and iv >= 10_000:
                k = iv // 1_000
                variants.append(f"{k}k")

            # Dedup
            out: list[str] = []
            seen: set[str] = set()
            for s in variants:
                key = s.replace(" ", "").lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(s)
            return out

        # Otherwise keep a couple of sensible formats.
        return [
            (f"{v:.2f}").rstrip("0").rstrip("."),
            (f"{v:.4f}").rstrip("0").rstrip("."),
        ]

    @staticmethod
    def _requires_strict_number_match(criterion: str) -> bool:
        """Heuristic: only enforce strict numeric matching for 'atomic' numeric criteria.

        We avoid enforcing for narrative criteria that merely contain dates/percentages.
        """
        if not isinstance(criterion, str):
            return False

        c = criterion.strip()
        c_low = c.lower()

        # Short currency conversion/FX line-items are typically numeric and should match.
        if c_low.startswith(("usd", "gbp", "eur")):
            nums = StudentGrader._numbers_in_text(c)
            return len(nums) <= 3 and len(c) <= 80

        # Simple bracketed calculations like (375,000 x GBP32).
        # Only enforce on criteria where the calc IS the main content (short to medium length),
        # not long narrative criteria that happen to mention a calc in passing.
        m = re.search(r"\(([^)]*)\)", c)
        if m and len(c) > 100:
            # For long criteria, only apply if the bracket is near the end (the "answer" portion)
            # and the text before the bracket is short.
            bracket_start = m.start()
            text_before = c[:bracket_start].strip()
            if len(text_before) > 60:
                return False
        if m:
            inside = m.group(1).lower()
            if not re.search(r"\d", inside):
                return False
            # Skip composite expressions (they often list components but evidence may show only the final figure)
            if "+" in inside or "-" in inside:
                return False
            if "x" in inside or "*" in inside or "/" in inside:
                return True

        return False



    def _sanitize_holistic_comments(
        self, comments: list, student_text: str
    ) -> list[str]:
        """Pre-flight check on comment anchors before they hit the annotator.

        The annotator dumps any comment whose anchor it cannot find into
        `unanchored_comments` - invisible to the student. Two LLM failure modes
        cause this:
          1. Anchor too long (6+ words) - spans PDF lines, exact match fails.
          2. Hallucinated anchor - not a verbatim substring of the student text.

        This pass trims oversize anchors to a 3-5 word window and verifies
        each anchor is actually present in the student text. Comments that
        can't be salvaged are dropped with a debug log.
        """
        if not isinstance(comments, list) or not comments:
            return []

        # Normalised version of student text for substring search - tolerant of
        # whitespace differences but preserves typos/casing the LLM should copy.
        st = student_text or self._student_text_last_run or ""
        st_norm = re.sub(r"\s+", " ", st)
        st_norm_lower = st_norm.lower()

        def _anchor_present(anchor: str) -> bool:
            if not anchor:
                return False
            a_norm = re.sub(r"\s+", " ", anchor).strip()
            return a_norm.lower() in st_norm_lower

        prefix_re = re.compile(r"^\s*\[([^\]]+)\]\s*")
        out: list[str] = []
        dropped = 0
        trimmed = 0

        for c in comments:
            if not isinstance(c, str) or "→" not in c:
                # Wrong format - pass through; annotator will skip it itself.
                if isinstance(c, str) and c.strip():
                    out.append(c)
                continue

            prefix_match = prefix_re.match(c)
            prefix = prefix_match.group(0) if prefix_match else ""
            body = c[len(prefix):] if prefix else c

            left, right = body.split("→", 1)
            anchor = left.strip().strip('"\'')
            feedback = right.strip()

            if not anchor or not feedback:
                dropped += 1
                logger.debug(f"  Dropping malformed comment: {c[:80]!r}")
                continue

            # 1. Verify the anchor is actually in the student text. If the LLM
            #    hallucinated something the student didn't write, the annotator
            #    will silently fail - drop the comment now.
            if not _anchor_present(anchor):
                # Last-chance salvage: try shorter prefixes (3 words, 4 words).
                a_words = anchor.split()
                salvaged: Optional[str] = None
                for n in (3, 4, 5):
                    if n < len(a_words):
                        candidate = " ".join(a_words[:n])
                        if _anchor_present(candidate):
                            salvaged = candidate
                            break
                if not salvaged:
                    dropped += 1
                    logger.debug(
                        f"  Dropping comment with hallucinated/unverifiable anchor: "
                        f"{anchor!r}"
                    )
                    continue
                anchor = salvaged
                trimmed += 1

            # 2. Trim oversize anchors to a 3-5 word window. Search for a 4-word
            #    or 5-word sub-window that appears verbatim in the student text;
            #    prefer the first such window so the popup lands near the start
            #    of the issue.
            a_words = anchor.split()
            if len(a_words) > 5:
                shorter: Optional[str] = None
                for size in (5, 4, 3):
                    for start in range(0, len(a_words) - size + 1):
                        cand = " ".join(a_words[start:start + size])
                        if _anchor_present(cand):
                            shorter = cand
                            break
                    if shorter:
                        break
                if shorter:
                    anchor = shorter
                    trimmed += 1
                else:
                    # Fall back to first 4 words even if not perfect match.
                    anchor = " ".join(a_words[:4])
                    trimmed += 1

            out.append(f"{prefix}{anchor} → {feedback}")

        if dropped or trimmed:
            logger.info(
                f"Comment sanitization: trimmed {trimmed} anchor(s), "
                f"dropped {dropped} comment(s) with un-locatable anchors"
            )
        return out

    # Stop-words that should never be the FIRST or LAST word of a key_phrase
    # (they make the underline bleed onto a stray article/preposition).
    _KP_STOPWORDS = frozenset({
        "a", "an", "the", "to", "of", "in", "on", "at", "for", "with", "by",
        "and", "or", "but", "so", "as", "is", "are", "was", "were", "be",
        "this", "that", "these", "those", "it", "its", "their",
    })

    @classmethod
    def _trim_stopword_edges(cls, key_phrase: str, sentence: str) -> str:
        """Strip leading/trailing stop-words from key_phrase so the underline
        doesn't bleed onto a stray "a" / "the" / "to" / "of" at the edges.

        Example: "Griffins goals aligning to closely to" → "Griffins goals aligning to closely"
        Example: "unable to identify issues in the" → "unable to identify issues"
        Example: "to keep Nicola on as engagement partner" → "keep Nicola on as engagement partner"
        """
        if not key_phrase:
            return key_phrase
        tokens = key_phrase.split()

        def _is_stop(tok: str) -> bool:
            return re.sub(r"[^\w]+", "", tok).lower() in cls._KP_STOPWORDS

        while tokens and _is_stop(tokens[-1]):
            tokens.pop()
        while tokens and _is_stop(tokens[0]):
            tokens.pop(0)
        return " ".join(tokens) if tokens else key_phrase

    @staticmethod
    def _expand_short_key_phrase(short_kp: str, sentence: str) -> str:
        """Grow a too-short key_phrase (1-3 words) into a 4-6 word window of
        surrounding context from the parent sentence.

        Used to avoid placing ticks on bare fragments like "reviewing payroll"
        or "to Yeti's" - those visually land on stray articles in the rendered
        PDF. We find the short phrase inside the sentence and pad outward
        until the slice is 4-6 words, preferring left-padding (subject context)
        over right-padding when the short phrase already contains the verb.
        """
        if not short_kp or not sentence:
            return short_kp
        kp_tokens = short_kp.split()
        if len(kp_tokens) >= 4:
            return short_kp

        sent_tokens = sentence.split()
        if len(sent_tokens) < 4:
            return short_kp  # whole sentence is too short to grow into

        # Locate the short phrase inside the sentence (case-insensitive, tolerant
        # of trailing/leading punctuation differences like "yrs" vs "yrs.").
        def _norm(s: str) -> str:
            return re.sub(r"[^\w]+", "", s).lower()

        kp_norm = _norm(short_kp)
        sent_norm = [_norm(t) for t in sent_tokens]
        for start in range(len(sent_tokens) - len(kp_tokens) + 1):
            window_norm = "".join(sent_norm[start:start + len(kp_tokens)])
            if window_norm == kp_norm:
                # Found placement. Pad to 4-6 words.
                target = min(6, max(4, len(kp_tokens) + 2))
                # Prefer extending leftward first (gives context/subject).
                lo, hi = start, start + len(kp_tokens)
                while (hi - lo) < target and (lo > 0 or hi < len(sent_tokens)):
                    if lo > 0 and (hi - lo) < target:
                        lo -= 1
                    elif hi < len(sent_tokens):
                        hi += 1
                    else:
                        break
                return " ".join(sent_tokens[lo:hi])

        # Phrase not found by full-match - fall back to any 4-5 word window
        # of the sentence that contains at least one substantive keyword from
        # the original short phrase.
        kp_word_set = {_norm(w) for w in kp_tokens if len(w) >= 3}
        for size in (5, 4):
            for start in range(max(0, len(sent_tokens) - size + 1)):
                window = sent_tokens[start:start + size]
                if len(window) < 4:
                    continue
                window_norm_set = {_norm(w) for w in window}
                if kp_word_set & window_norm_set:
                    return " ".join(window)

        return short_kp  # give up; caller will drop it

    # ── Coverage audit helpers (option B post-processing) ──────────────────

    @staticmethod
    def _audit_split_sentences(text: str) -> list[str]:
        """Split student text into sentence-like chunks for keyword scanning."""
        if not text:
            return []
        chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [c.strip() for c in chunks if c and c.strip()]

    @staticmethod
    def _audit_stem_match(kw_token: str, txt_tokens: set) -> bool:
        """True if any text token shares a stem prefix with kw_token.

        Catches morphological variants like physical/physically, decide/decision,
        consent/consenting, separated/separation. Uses a 4–5-char prefix as the stem.
        """
        kw = kw_token.lower()
        if len(kw) < 4:
            return kw in txt_tokens
        stem = kw[:5] if len(kw) >= 6 else kw[:4]
        for t in txt_tokens:
            if len(t) < 3:
                continue
            if t.startswith(stem) or kw.startswith(t[:max(4, min(len(t), 5))]):
                return True
        return False

    @staticmethod
    def _audit_keyword_in_text(keyword: str, text: str) -> bool:
        """Check if a keyword (or close morphological/partial variant) appears in text.

        Multi-word keywords: pass if the literal phrase appears, OR if at least
        half of the substantive (≥4-char) tokens have a stem match in text.
        Single-word keywords: pass on stem-prefix match (handles plural/-ed/-ing/etc.).
        """
        if not keyword or not text:
            return False
        kw_low = keyword.lower().strip()
        txt_low = text.lower()
        txt_tokens = set(re.findall(r"[a-z]+", txt_low))

        # Multi-word / separator-containing keyword.
        if any(c in kw_low for c in (" ", "/", "-")):
            if kw_low in txt_low:
                return True
            kw_tokens = [t for t in re.findall(r"[a-z]+", kw_low) if len(t) >= 4]
            if not kw_tokens:
                return False
            matches = sum(
                1 for kt in kw_tokens if StudentGrader._audit_stem_match(kt, txt_tokens)
            )
            return matches >= max(1, (len(kw_tokens) + 1) // 2)

        # Single-word keyword.
        if len(kw_low) < 3:
            return False
        return StudentGrader._audit_stem_match(kw_low, txt_tokens)

    def _audit_score_criterion(self, criterion: dict, sentence: str) -> float:
        """0..1 score for how strongly a sentence supports a criterion."""
        keywords = [k for k in (criterion.get("keywords") or []) if isinstance(k, str)]
        if not keywords:
            desc = str(criterion.get("description", "") or "")
            keywords = [w for w in re.findall(r"\w+", desc) if len(w) > 4][:6]
            if not keywords:
                return 0.0
        matched = sum(1 for kw in keywords if self._audit_keyword_in_text(kw, sentence))
        return matched / float(len(keywords))

    def _audit_best_sentence(
        self, criterion: dict, sentences: list[str], threshold: float = 0.34
    ) -> Optional[Tuple[str, float]]:
        best, best_score = None, 0.0
        for s in sentences:
            score = self._audit_score_criterion(criterion, s)
            if score > best_score:
                best, best_score = s, score
        if best is not None and best_score >= threshold:
            return best, best_score
        return None

    def _audit_existing_ticks_for_criterion(
        self, criterion: dict, existing_pts: list[dict]
    ) -> int:
        """Count how many existing correct_points already credit this criterion
        (by keyword presence in the tick's text or key_phrase)."""
        keywords = [k for k in (criterion.get("keywords") or []) if isinstance(k, str)]
        if not keywords:
            return 0
        count = 0
        for pt in existing_pts:
            blob = f"{pt.get('text', '')} {pt.get('key_phrase', '')}"
            if any(self._audit_keyword_in_text(kw, blob) for kw in keywords):
                count += 1
        return count

    def _audit_pick_anchor(
        self, sentence: str, keywords: list[str], used_phrases: list[str]
    ) -> Optional[str]:
        """Pick a 4-6 word slice of sentence containing a keyword, with no word
        overlap against used_phrases. Falls back to any non-overlapping window.

        Minimum 4 words: a 1-3 word slice places the tick on an ambiguous
        fragment that visually looks like a tick on a stray article in the PDF.
        """
        if not sentence:
            return None
        used_words: set = set()
        for up in used_phrases:
            used_words.update(re.findall(r"\w+", (up or "").lower()))

        tokens = sentence.split()
        n = len(tokens)
        if n < 4:
            return None

        kw_positions: list[int] = []
        for kw in keywords or []:
            kw_low = kw.lower().strip()
            if not kw_low:
                continue
            kw_tokens = [t for t in re.findall(r"[a-z]+", kw_low) if len(t) >= 3]
            for i, tok in enumerate(tokens):
                tok_clean = re.sub(r"[^a-z]+", "", tok.lower())
                if not tok_clean:
                    continue
                if any(
                    self._audit_stem_match(kt, {tok_clean}) for kt in kw_tokens
                ):
                    kw_positions.append(i)

        for pos in kw_positions:
            for size in (5, 4, 6):
                for start in range(max(0, pos - size + 1), min(n - size + 1, pos + 1) + 1):
                    if start < 0 or start + size > n:
                        continue
                    window = tokens[start:start + size]
                    if len(window) < 4:
                        continue
                    win_words = set(re.findall(r"\w+", " ".join(window).lower()))
                    if not (win_words & used_words):
                        return " ".join(window)

        for size in (5, 4):
            for start in range(max(0, n - size + 1)):
                window = tokens[start:start + size]
                if len(window) < 4:
                    continue
                win_words = set(re.findall(r"\w+", " ".join(window).lower()))
                if not (win_words & used_words):
                    return " ".join(window)
        return None

    def _audit_holistic_coverage(
        self, breakdown: list, student_text: str
    ) -> list:
        """For each sub-question, ensure rubric criteria with sufficient student
        text support have their full mark value's worth of ticks. Adds ticks for
        under-credited criteria, anchored at non-overlapping key_phrases inside
        the best-matching student sentence. Caps at each sub-question's max_marks.

        Guardrails:
        • Dual threshold - easier to AUGMENT criteria the LLM already credited
          (existing_for_crit > 0) than to introduce NEW credit (existing == 0).
        • Per-sentence global cap - any single student sentence can earn at most
          PER_TEXT_TICK_CAP ticks across all leaves (prevents one sentence from
          being credited for every shared-keyword criterion in the rubric).
        """
        if os.getenv("AUDIT_HOLISTIC_COVERAGE", "1").strip().lower() in {"0", "false", "no"}:
            return breakdown
        if not getattr(self, "_holistic_grading", False):
            return breakdown
        if not (self._holistic_sub_questions and student_text):
            return breakdown

        AUGMENT_THRESHOLD = float(os.getenv("AUDIT_AUGMENT_THRESHOLD", "0.34"))
        NEW_THRESHOLD = float(os.getenv("AUDIT_NEW_THRESHOLD", "0.55"))
        PER_TEXT_TICK_CAP = int(os.getenv("AUDIT_PER_TEXT_TICK_CAP", "4"))
        # When the LLM has already credited a parent section to ≥SECTION_TRUST_RATIO
        # of its section_cap, skip the audit ENTIRELY for that section's leaves
        # (no new credit AND no augmenting). The LLM's coverage call is final.
        # Default 0.75 - at ≥75% of the cap, trust the LLM.
        SECTION_TRUST_RATIO = float(os.getenv("AUDIT_SECTION_TRUST_RATIO", "0.75"))

        sq_meta: dict = {}
        for sq in self._holistic_sub_questions:
            label = str(sq.get("sub_question", "")).strip()
            sq_meta[label] = {
                "criteria": sq.get("marking_criteria") or [],
                "max_marks": float(sq.get("max_marks") or 0),
                "parent_section": sq.get("parent_section"),
                "section_cap": sq.get("section_cap"),
            }

        sentences = self._audit_split_sentences(student_text)

        # Seed per-sentence tick counter with all existing LLM ticks across the
        # whole question so audit additions stay under the cap globally.
        sentence_ticks: dict[str, int] = {}
        for item in breakdown:
            for pt in (item.get("_correct_points_with_marks", []) or []):
                txt = pt.get("text", "")
                if txt:
                    sentence_ticks[txt] = sentence_ticks.get(txt, 0) + 1

        # Pre-compute LLM-awarded marks per parent_section. If the LLM has
        # already credited a section close to its cap, we should not add any
        # NEW criteria there (only augment existing partial credit). This
        # prevents the audit from over-shooting on tightly-capped sections
        # like 4.2 (cap=4, where rubric criteria across leaves are similar).
        section_llm_marks: dict[str, float] = {}
        section_caps: dict[str, float] = {}
        for item in breakdown:
            sq = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sq) or {}
            parent = meta.get("parent_section")
            cap = meta.get("section_cap")
            if parent and cap is not None:
                section_llm_marks[parent] = (
                    section_llm_marks.get(parent, 0.0)
                    + float(item.get("marks_awarded", 0) or 0)
                )
                section_caps[parent] = float(cap)

        section_trusted: set[str] = set()
        for parent, llm_total in section_llm_marks.items():
            cap = section_caps.get(parent, 0.0)
            if cap > 0 and llm_total >= cap * SECTION_TRUST_RATIO:
                section_trusted.add(parent)
                logger.info(
                    f"Audit: section {parent} LLM gave {llm_total}/{cap} "
                    f"(≥{SECTION_TRUST_RATIO:.0%}) - skipping audit entirely"
                )

        # Running marks per parent_section so audit additions stop at the cap.-
        section_running_marks: dict[str, float] = dict(section_llm_marks)

        audit_log: list[str] = []

        for item in breakdown:
            sq = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sq)
            if not meta or not meta["criteria"]:
                continue

            existing_pts = list(item.get("_correct_points_with_marks", []) or [])
            added = 0
            parent_section = meta.get("parent_section")
            section_cap_val = meta.get("section_cap")
            is_trusted = parent_section in section_trusted

            # Trusted-section short-circuit: when the LLM has already credited
            # this parent section close to its cap (≥ SECTION_TRUST_RATIO), skip
            # the audit entirely for this leaf - no new credit AND no augmenting
            # of partial credits. The LLM's coverage call is treated as final.
            # Without this, partial-credit augmentation can still push the
            # section's total to the cap when teacher would have left it lower.
            if is_trusted:
                continue

            for crit in meta["criteria"]:
                crit_marks = float(crit.get("marks") or 1)
                expected_ticks = int(round(crit_marks * 2))
                existing_for_crit = self._audit_existing_ticks_for_criterion(crit, existing_pts)
                if existing_for_crit >= expected_ticks:
                    continue

                threshold = AUGMENT_THRESHOLD if existing_for_crit > 0 else NEW_THRESHOLD
                match = self._audit_best_sentence(crit, sentences, threshold=threshold)
                if not match:
                    continue
                best_sent, _score = match
                keywords = [k for k in (crit.get("keywords") or []) if isinstance(k, str)]
                used_phrases_in_sent = [
                    pt.get("key_phrase", "") for pt in existing_pts
                    if pt.get("text", "") == best_sent
                ]

                need = expected_ticks - existing_for_crit
                for _ in range(need):
                    # Stop if the parent section_cap is already saturated.
                    if (
                        parent_section
                        and section_cap_val is not None
                        and section_running_marks.get(parent_section, 0.0) >= float(section_cap_val)
                    ):
                        break
                    # Per-sentence global cap - protects against over-crediting
                    # the same student sentence under multiple shared-keyword criteria.
                    if sentence_ticks.get(best_sent, 0) >= PER_TEXT_TICK_CAP:
                        break
                    anchor = self._audit_pick_anchor(best_sent, keywords, used_phrases_in_sent)
                    if not anchor:
                        break
                    existing_pts.append({
                        "text": best_sent,
                        "marks": 0.5,
                        "key_phrase": anchor,
                    })
                    used_phrases_in_sent.append(anchor)
                    sentence_ticks[best_sent] = sentence_ticks.get(best_sent, 0) + 1
                    if parent_section:
                        section_running_marks[parent_section] = (
                            section_running_marks.get(parent_section, 0.0) + 0.5
                        )
                    added += 1

            sub_max = meta["max_marks"]
            if sub_max > 0:
                cap_count = int(round(sub_max / 0.5))
                if len(existing_pts) > cap_count:
                    existing_pts = existing_pts[:cap_count]

            item["_correct_points_with_marks"] = existing_pts
            item["marks_awarded"] = len(existing_pts) * 0.5
            item["evidence"] = list(dict.fromkeys(pt["text"] for pt in existing_pts))

            if added > 0:
                audit_log.append(
                    f"{sq}: +{added} ticks (now {len(existing_pts)} = {item['marks_awarded']} marks)"
                )

        if audit_log:
            logger.info(f"Coverage audit added ticks → {'; '.join(audit_log)}")
        return breakdown

    def _aggregate_holistic_breakdown(self, breakdown: list) -> list:
        """Aggregate per-criterion holistic breakdown into parent-section level entries.

        For theoretical papers, the LLM grades fine-grained sub-questions like
        "4.1(a) Consequences", "4.1(b) Recommendations", etc.  The MongoDB
        breakdown should show one entry per top-level section (4.1, 4.2, …)
        with aggregated marks, while keeping all tick annotation data merged.
        """
        from collections import OrderedDict

        # Build lookup: sub_question_label → {parent_section, section_cap}
        sq_meta: dict = {}
        for sq in (getattr(self, "_holistic_sub_questions", None) or []):
            label = str(sq.get("sub_question", "")).strip()
            sq_meta[label] = {
                "parent_section": sq.get("parent_section"),
                "section_cap": sq.get("section_cap"),
            }

        groups: "OrderedDict[str, list]" = OrderedDict()
        group_cap: dict = {}

        for item in breakdown:
            sub_q = str(item.get("_sub_question", "")).strip()
            meta = sq_meta.get(sub_q, {})
            parent = meta.get("parent_section")
            section_cap = meta.get("section_cap")

            group_key = parent if parent else sub_q
            if group_key not in groups:
                groups[group_key] = []
                group_cap[group_key] = section_cap if (parent and section_cap is not None) else item.get("max_possible", 0)
            groups[group_key].append(item)

        # If every item is its own group there is nothing to aggregate.
        if len(groups) == len(breakdown):
            return breakdown

        aggregated = []
        for group_key, items in groups.items():
            total_awarded = sum(float(i.get("marks_awarded", 0) or 0) for i in items)
            max_possible = float(group_cap.get(group_key) or 0)
            total_awarded = min(total_awarded, max_possible)
            total_awarded = round(total_awarded / 0.5) * 0.5

            # Merge evidence (deduplicated, order preserved).
            seen_ev: set = set()
            all_evidence = []
            for item in items:
                for ev in (item.get("evidence") or []):
                    if ev not in seen_ev:
                        all_evidence.append(ev)
                        seen_ev.add(ev)

            # Merge tick-annotation points.
            all_correct_points = []
            for item in items:
                all_correct_points.extend(item.get("_correct_points_with_marks") or [])
            target_count = int(round(total_awarded / 0.5))
            if len(all_correct_points) > target_count:
                all_correct_points = all_correct_points[:target_count]

            # Merge not-required points (deduplicated).
            seen_nr: set = set()
            all_nr: list = []
            for item in items:
                for nr in (item.get("_not_required_points") or []):
                    key = nr.get("text", "")
                    if key not in seen_nr:
                        all_nr.append(nr)
                        seen_nr.add(key)

            reasons = [i.get("reason", "").strip() for i in items if i.get("reason", "").strip()]
            combined_reason = "; ".join(reasons)

            student_label = next(
                (i.get("_student_label", "") for i in items if i.get("_student_label", "")), ""
            )

            aggregated.append({
                "criterion": f"Sub-question {group_key}",
                "marks_awarded": total_awarded,
                "max_possible": max_possible,
                "reason": combined_reason,
                "evidence": all_evidence,
                "comments_summary": "",
                "_sub_question": group_key,
                "_student_label": student_label,
                "_correct_points_with_marks": all_correct_points,
                "_not_required_points": all_nr,
            })

        return aggregated

    def _run_holistic_grading(self, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute holistic grading: compare full answers without per-criterion breakdown.

        Returns a dict in the same shape as the standard grading response so that
        _build_grade_doc() can process it uniformly via a conversion step.
        """
        student_text = self._format_student_for_prompt(student_data)
        self._student_text_last_run = student_text or ""

        compact_json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}
        try:
            self._question_text_last_run = json.dumps(questions_data, **compact_json_kwargs)
        except Exception:
            self._question_text_last_run = str(questions_data)
        try:
            self._model_text_last_run = json.dumps(model_data, **compact_json_kwargs)
        except Exception:
            self._model_text_last_run = str(model_data)

        payload = {
            "model_data": json.dumps(model_data, **compact_json_kwargs),
            "chunks": student_text,
            "questions": json.dumps(questions_data, **compact_json_kwargs),
        }

        debug_enabled = os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}

        def _coerce_holistic(result: Any) -> dict:
            if isinstance(result, BaseModel):
                return result.model_dump()
            if isinstance(result, dict):
                return result
            structured_args = self._extract_structured_args_from_message(result)
            if isinstance(structured_args, dict):
                return structured_args
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            raw = str(content)
            json_text = self._extract_json_from_text(raw)
            if not json_text:
                raise GradingError("Empty holistic grading output")
            return json.loads(json_text)

        # Attempt 1: structured output
        holistic_parsed = None
        try:
            if self.holistic_chain_structured is not None:
                output = self.holistic_chain_structured.invoke(payload)
                holistic_parsed = _coerce_holistic(output)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [structured]")
        except Exception as e:
            logger.warning(f"Holistic structured grading failed; falling back to text: {e}")
            holistic_parsed = None

        # Degenerate-case detection: some providers/structured-output paths return
        # the top-level score but drop the nested sub_grades array entirely. Treat
        # that as a structured-output failure and fall through to the text path,
        # which carries the same content as raw JSON in the message body.
        if (
            holistic_parsed is not None
            and float(holistic_parsed.get("score", 0) or 0) > 0
            and not (holistic_parsed.get("sub_grades") or [])
        ):
            logger.warning(
                "Holistic structured output reported "
                f"score={holistic_parsed.get('score')} but returned 0 sub_grades - "
                "treating as parse failure and retrying via text path"
            )
            holistic_parsed = None

        # Attempt 2: text output + parse
        if holistic_parsed is None:
            try:
                output = self.holistic_chain_text.invoke(payload)
                holistic_parsed = _coerce_holistic(output)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                # Same degenerate-case guard for the text path.
                if (
                    float(holistic_parsed.get("score", 0) or 0) > 0
                    and not (holistic_parsed.get("sub_grades") or [])
                ):
                    raise GradingError(
                        f"Text path returned score={holistic_parsed.get('score')} with 0 sub_grades"
                    )
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [text]")
            except Exception as e:
                logger.warning(f"Holistic text grading failed; attempting repair: {e}")
                holistic_parsed = None

        # Attempt 3: repair
        if holistic_parsed is None:
            try:
                raw_content = getattr(output, "content", None) if "output" in locals() else None
                raw_content = raw_content if raw_content is not None else ""
                repair_prompt = (
                    "You MUST return ONLY valid JSON (no markdown, no commentary). "
                    "Fix the following output to match this exact schema:\n"
                    "{\n"
                    "  \"question_number\": string,\n"
                    "  \"score\": number,\n"
                    "  \"total_marks\": number,\n"
                    "  \"sub_grades\": [\n"
                    "    {\n"
                    "      \"sub_question\": string,\n"
                    "      \"student_label\": string,\n"
                    "      \"marks_awarded\": number,\n"
                    "      \"max_marks\": number,\n"
                    "      \"reason\": string,\n"
                    "      \"correct_points\": [{\"text\": string, \"marks\": number, \"key_phrase\": string}, ...]\n"
                    "    }\n"
                    "  ],\n"
                    "  \"comments\": [string, ...]\n"
                    "}\n\n"
                    "OUTPUT TO FIX:\n"
                    f"{raw_content}"
                )
                fixed = llm_grader.invoke(repair_prompt)
                fixed_content = getattr(fixed, "content", None) or str(fixed)
                fixed_json = self._extract_json_from_text(str(fixed_content))
                holistic_parsed = json.loads(fixed_json)
                validated = HolisticGradingResponse(**holistic_parsed)
                holistic_parsed = validated.model_dump()
                logger.info(f"Holistic grading complete → {self.student_name} (Q{self.question_number}) [repaired]")
            except Exception as e2:
                logger.error("Holistic grading chain failed", exc_info=True)
                raise GradingError("Holistic grading step failed") from e2

        # Convert holistic response to standard grades format for _build_grade_doc().
        # Each sub_grade becomes a breakdown item where:
        #   criterion = "Sub-question <sub_question>" (or just question number for single block)
        #   evidence_list = correct_points texts (for anchoring)
        #   _correct_points_with_marks = full objects with per-point marks (for tick annotation)
        #   marks_awarded / max_possible = sub-question marks
        breakdown = []
        for sg in holistic_parsed.get("sub_grades", []):
            sq_label = sg.get("sub_question", "")

            # Strip artificial chunk separator dashes so annotator finds real PDF text.
            # Chunks are formatted as "--- 4.1 ---\n<answer>" so the LLM returns
            # "--- 4.1 ---" verbatim. Strip to just "4.1" for PDF search.
            student_label = sg.get("student_label", "")
            student_label = re.sub(r'^[-\s]+|[-\s]+$', '', student_label).strip()

            criterion_text = f"Sub-question {sq_label}" if len(holistic_parsed.get("sub_grades", [])) > 1 else f"Question {self.question_number}"

            # Marks awarded for this sub-question - round to nearest 0.5.
            marks_awarded = round(float(sg.get("marks_awarded", 0) or 0) / 0.5) * 0.5

            # correct_points is now list of {"text": str, "marks": float, "key_phrase": str}
            raw_points = sg.get("correct_points", [])
            # Handle both formats: list of objects or list of strings (backward compat)
            evidence_texts = []
            points_with_marks = []
            for pt in raw_points:
                if isinstance(pt, dict):
                    text = str(pt.get("text", "")).strip()
                    # Every correct_point must be exactly 0.5 marks (1 tick).
                    # If the LLM outputs marks > 0.5, split into multiple 0.5 entries.
                    pt_marks = float(pt.get("marks", 0.5) or 0.5)
                    pt_marks = max(0.5, round(pt_marks / 0.5) * 0.5)
                    key_phrase = str(pt.get("key_phrase", "")).strip()
                    # Enforce 4-6 word window. Long phrases (>6 words) span PDF
                    # lines and can't be found; short phrases (<4 words) place
                    # the tick on an ambiguous fragment that visually looks like
                    # a tick on a stray article ("a", "the") in the rendered PDF.
                    kp_words = key_phrase.split()
                    if len(kp_words) > 6:
                        key_phrase = " ".join(kp_words[:6])
                    elif 0 < len(kp_words) < 4 and text:
                        # Try to grow the slice in-place by extending into the
                        # surrounding sentence words.
                        key_phrase = self._expand_short_key_phrase(key_phrase, text)
                        kp_words = key_phrase.split()
                        if len(kp_words) < 4:
                            # Couldn't grow to ≥4 words - drop this tick rather
                            # than place it on a misleading fragment.
                            logger.debug(
                                f"  Dropping tick with un-growable short key_phrase: "
                                f"{pt.get('key_phrase', '')!r} (text: {text[:60]!r})"
                            )
                            continue
                    # Trim trailing/leading stop-words so the underline doesn't
                    # extend onto a stray article/preposition ("to", "the", "a",
                    # "of", "in"). This is what creates the visual "tick on a"
                    # complaint - the rect ends on a stop word and the underline
                    # bleeds onto it. After trim, re-expand if we fell under 4.
                    key_phrase = self._trim_stopword_edges(key_phrase, text)
                    kp_words = key_phrase.split()
                    if len(kp_words) < 4 and text:
                        key_phrase = self._expand_short_key_phrase(key_phrase, text)
                        kp_words = key_phrase.split()
                        if len(kp_words) < 4:
                            logger.debug(
                                f"  Dropping tick after stop-word trim left short phrase: "
                                f"{pt.get('key_phrase', '')!r} (text: {text[:60]!r})"
                            )
                            continue
                    if text:
                        evidence_texts.append(text)
                        if pt_marks == 0.5:
                            points_with_marks.append({
                                "text": text,
                                "marks": 0.5,
                                "key_phrase": key_phrase,
                            })
                        else:
                            # Split into N × 0.5 entries. First entry keeps key_phrase;
                            # subsequent entries reuse the same text (annotator underlines
                            # the same line, placing additional ticks alongside).
                            n_ticks = int(round(pt_marks / 0.5))
                            for tick_i in range(n_ticks):
                                points_with_marks.append({
                                    "text": text,
                                    "marks": 0.5,
                                    "key_phrase": key_phrase if tick_i == 0 else "",
                                })
                elif isinstance(pt, str) and pt.strip():
                    evidence_texts.append(pt.strip())
                    points_with_marks.append({"text": pt.strip(), "marks": 0.5, "key_phrase": ""})

            # Strict consistency: marks_awarded MUST equal len(points_with_marks) × 0.5.
            # Each correct_point = 0.5 marks = 1 tick on the PDF, so the totals
            # rendered to the student cannot diverge from the visible tick count.
            if marks_awarded > 0:
                target_count = int(round(marks_awarded / 0.5))
                if len(points_with_marks) > target_count:
                    # Too many evidenced ticks - LLM reported lower marks; raise marks
                    # to match the evidence it produced (each tick is 0.5 of evidence).
                    marks_awarded = len(points_with_marks) * 0.5
                elif len(points_with_marks) < target_count:
                    # Fewer evidenced ticks than LLM-reported marks - trust the
                    # evidence: marks must equal the visible tick count × 0.5.
                    marks_awarded = len(points_with_marks) * 0.5
            elif points_with_marks:
                # LLM reported 0 marks but produced ticks - trust the ticks.
                marks_awarded = len(points_with_marks) * 0.5

            # Re-cap at this sub-question's own max_marks after the alignment.
            sq_max = float(sg.get("max_marks", 0) or 0)
            if sq_max > 0 and marks_awarded > sq_max:
                marks_awarded = sq_max
                target_count = int(round(marks_awarded / 0.5))
                if len(points_with_marks) > target_count:
                    points_with_marks = points_with_marks[:target_count]

            # Off-topic ("Not required") points - flagged by LLM, no marks.
            # Same key_phrase truncation rule as correct_points.
            raw_nr = sg.get("not_required_points", []) or []
            not_required_points: list[dict] = []
            for nr in raw_nr:
                if not isinstance(nr, dict):
                    continue
                nr_text = str(nr.get("text", "")).strip()
                if not nr_text:
                    continue
                nr_kp = str(nr.get("key_phrase", "")).strip()
                kp_words = nr_kp.split()
                if len(kp_words) > 6:
                    nr_kp = " ".join(kp_words[:6])
                not_required_points.append({
                    "text": nr_text,
                    "key_phrase": nr_kp,
                    "reason": str(nr.get("reason", "")).strip(),
                })

            breakdown.append({
                "criterion": criterion_text,
                "marks_awarded": marks_awarded,
                "max_possible": float(sg.get("max_marks", 0) or 0),
                "reason": sg.get("reason", ""),
                "evidence": evidence_texts,
                "comments_summary": "",
                # Extra fields for annotation
                "_sub_question": sq_label,
                "_student_label": student_label,
                "_correct_points_with_marks": points_with_marks,
                "_not_required_points": not_required_points,
            })

        # Guard: if the LLM reported a non-zero score but produced no sub_grades,
        # the structured output parsing silently dropped the breakdown. Fail loudly
        # so the text-based fallback / repair path can try instead of saving 0/20.
        llm_score_raw = float(holistic_parsed.get("score", 0) or 0)
        if llm_score_raw > 0 and not breakdown:
            raise GradingError(
                f"Holistic LLM reported score={llm_score_raw} but returned "
                f"0 sub_grades - structured output likely failed to parse"
            )

        # Coverage audit: add ticks for rubric criteria the LLM under-credited.
        # The audit becomes the authoritative source of marks for each sub-question;
        # the LLM's reported score is no longer used as an upper bound after this.
        breakdown = self._audit_holistic_coverage(breakdown, student_text)

        # Aggregate fine-grained sub-question entries into parent-section level
        # (e.g. "4.1(a) Consequences" + "4.1(b) Recommendations" + … → "4.1").
        breakdown = self._aggregate_holistic_breakdown(breakdown)

        # Recompute total score from audited+aggregated breakdown so the
        # student-facing total matches the tick counts shown in each section.
        total_max = float(holistic_parsed.get("total_marks", 0) or 0)
        audited_score = sum(float(b.get("marks_awarded", 0) or 0) for b in breakdown)
        if total_max > 0:
            audited_score = min(audited_score, total_max)
        audited_score = round(audited_score / 0.5) * 0.5

        comments = self._sanitize_holistic_comments(
            holistic_parsed.get("comments", []) or [],
            student_text,
        )

        converted = {
            "grades": [{
                "question_number": holistic_parsed.get("question_number", self.question_number),
                "score": audited_score,
                "total_marks": total_max,
                "comments": comments,
                "correct_words": [],
                "breakdown": breakdown,
            }]
        }

        logger.info(
            f"Holistic grading converted to standard format: "
            f"{len(breakdown)} section-level breakdown items, "
            f"score={audited_score}/{total_max} "
            f"(LLM-only score was {holistic_parsed.get('score', 0)})"
        )
        return converted

    def _run_grading(self, student_data: dict, model_data: dict, questions_data: dict) -> dict:
        """Execute grading chain with clean content (holistic evaluation against all criteria)."""

        # Stash the model-answer doc so _build_grade_doc can read its top-level
        # max_marks / total_marks fields (the canonical question total set by
        # the marker) without threading model_data through another parameter.
        self._model_data_last_run = model_data

        # Route to holistic grading when no marking criteria exist.
        if self._holistic_grading:
            return self._run_holistic_grading(student_data, model_data, questions_data)

        student_text = self._format_student_for_prompt(student_data)
        self._student_text_last_run = student_text or ""
        # Use compact JSON to reduce prompt/token bloat.
        # (This can materially reduce latency/cost on large rubrics.)
        compact_json_kwargs = {"ensure_ascii": False, "separators": (",", ":")}

        # Store the raw payload blobs so downstream guardrails can detect evidence that
        # originates from the question / marking guide rather than student work.
        try:
            self._question_text_last_run = json.dumps(questions_data, **compact_json_kwargs)
        except Exception:
            self._question_text_last_run = str(questions_data)
        try:
            self._model_text_last_run = json.dumps(model_data, **compact_json_kwargs)
        except Exception:
            self._model_text_last_run = str(model_data)

        payload = {
            "model_data": json.dumps(model_data, **compact_json_kwargs),
            "chunks": student_text,
            "questions": json.dumps(questions_data, **compact_json_kwargs),
        }

        debug_enabled = os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}
        debug_dir = os.getenv("DEBUG_LLM_OUTPUT_DIR", "logs").strip() or "logs"
        debug_max_chars = int(os.getenv("DEBUG_LLM_OUTPUT_MAX_CHARS", "20000"))

        def _truncate(s: str) -> str:
            if not isinstance(s, str):
                s = str(s)
            if debug_max_chars <= 0:
                return s
            return s[:debug_max_chars]

        def _capture_debug(
            attempt: str,
            stage: str,
            output_obj: Any = None,
            raw_text: Optional[str] = None,
            error: Optional[BaseException] = None,
        ) -> None:
            if not debug_enabled:
                return

            try:
                if raw_text is None and output_obj is not None:
                    raw_text = getattr(output_obj, "content", None)
                    if raw_text is None:
                        raw_text = str(output_obj)
                raw_text = _truncate(raw_text or "")

                record: dict[str, Any] = {
                    "ts": datetime.utcnow().isoformat(),
                    "attempt": attempt,
                    "stage": stage,
                    "provider": os.getenv("GRADING_PROVIDER") or os.getenv("LLM_PROVIDER") or "",
                    "model": os.getenv("LLM_GRADER_MODEL") or "",
                    "error": str(error) if error else "",
                    "traceback": traceback.format_exc() if error else "",
                    "raw": raw_text,
                }
                self._llm_debug_trace.append(record)

                os.makedirs(debug_dir, exist_ok=True)
                fname = f"llm_grading_debug_{self.student_name}_Q{self.question_number}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}.json"
                safe_fname = re.sub(r"[^a-zA-Z0-9._-]+", "_", fname)
                fpath = os.path.join(debug_dir, safe_fname)
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(record, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved LLM debug trace → {fpath}")
            except Exception:
                # Never let debug capture break grading.
                pass

        def _norm_for_evidence_match(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("×", "x")
            s = s.replace("–", "-").replace("-", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            # Remove most punctuation while keeping separators meaningful for ratios.
            s = re.sub(r"[^a-z0-9%/().,\- ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        def _is_distinctive_short_evidence(ev_norm: str) -> bool:
            if not ev_norm or not isinstance(ev_norm, str):
                return False
            compact = ev_norm.replace(" ", "")
            if re.search(r"\d", ev_norm):
                return len(compact) >= 4
            if "/" in ev_norm or "%" in ev_norm:
                return len(compact) >= 3
            words = ev_norm.split()
            return len(words) <= 3 and any(len(w) >= 7 for w in words)

        def _evidence_present_in_student(evidence_items: list[str]) -> bool:
            student_blob = _norm_for_evidence_match(self._student_text_last_run)
            if not student_blob:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue

                # Fast path: full normalized substring.
                if ev_norm in student_blob:
                    return True

                # Sliding-window fallback: LLMs sometimes add/remove minor punctuation in
                # verbatim quotes.  A contiguous N-word window in the student text is
                # sufficient to confirm the quote is genuine.
                words = [w for w in ev_norm.split() if len(w) >= 3]
                if len(words) >= 10:
                    for i in range(0, min(len(words) - 9, 8)):
                        window = " ".join(words[i:i + 10])
                        if window in student_blob:
                            return True
                elif len(words) >= 5:
                    for i in range(0, min(len(words) - 4, 6)):
                        window = " ".join(words[i:i + 5])
                        if window in student_blob:
                            return True
            return False

        def _unwrap_stringified_lists(d: dict) -> dict:
            """Fix a common LLM output quirk where list-typed fields come back as
            JSON-encoded strings instead of native lists (Anthropic tool-call
            format sometimes stringifies large nested arrays under load).

            Applied to the top-level `grades` field and its nested `breakdown`
            and `comments`. Non-string values pass through unchanged. Invalid
            JSON strings pass through so the pydantic validator produces its
            normal error rather than a silent swallow.
            """
            if not isinstance(d, dict):
                return d
            g = d.get("grades")
            if isinstance(g, str):
                try:
                    d["grades"] = json.loads(g)
                except Exception:
                    pass  # let pydantic raise the descriptive validation error
            if isinstance(d.get("grades"), list):
                for gi in d["grades"]:
                    if not isinstance(gi, dict):
                        continue
                    for k in ("breakdown", "comments", "correct_words", "not_required_points"):
                        v = gi.get(k)
                        if isinstance(v, str):
                            try:
                                parsed_v = json.loads(v)
                                if isinstance(parsed_v, list):
                                    gi[k] = parsed_v
                            except Exception:
                                pass
            return d

        def _coerce_to_dict(result: Any) -> dict:
            if isinstance(result, dict):
                return _unwrap_stringified_lists(result)

            if isinstance(result, BaseModel):
                dumped = result.model_dump()
                # LangChain message classes (AIMessage, ChatMessage, ...) are
                # BaseModels too, so model_dump gives us `{'content': ...,
                # 'response_metadata': ...}` — NOT the grading dict. When the
                # dump doesn't carry a `grades` key, fall through to the
                # content-extraction path below instead of treating the
                # envelope as the grading response.
                if isinstance(dumped, dict) and "grades" in dumped:
                    return _unwrap_stringified_lists(dumped)

            structured_args = self._extract_structured_args_from_message(result)
            if isinstance(structured_args, dict):
                return _unwrap_stringified_lists(structured_args)

            # Try content-based JSON parsing (common for non-tool providers)
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            raw = str(content)
            json_text = self._extract_json_from_text(raw)
            if not json_text:
                raise GradingError("Empty grading output")
            try:
                return _unwrap_stringified_lists(json.loads(json_text))
            except Exception as je:
                raise GradingError(f"Invalid JSON from grader: {je}") from je

        # Attempt 1: structured output if available
        try:
            if self.grade_chain_structured is not None:
                output = self.grade_chain_structured.invoke(payload)
                _capture_debug("structured", "received", output_obj=output)
                parsed = _coerce_to_dict(output)
                _capture_debug("structured", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
                # Validate schema
                validated = LLMGradingResponse(**parsed)
                logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [structured]")
                result = validated.model_dump()
                # Guardrail: evidence must be a verbatim quote that exists in the student text.
                try:
                    for g in result.get("grades", []) or []:
                        for bi in g.get("breakdown", []) or []:
                            if float(bi.get("marks_awarded", 0) or 0) <= 0:
                                continue
                            ev = bi.get("evidence", [])
                            ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                            if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                                bi["marks_awarded"] = 0.0
                                bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
                except Exception:
                    # Non-fatal: fallback guardrails will still run in _build_grade_doc.
                    pass
                return result
        except Exception as e:
            _capture_debug("structured", "error", output_obj=locals().get("output"), error=e)
            logger.warning(f"Structured grading failed; falling back to text parsing: {e}")

        # Attempt 2: text output + parse
        try:
            output = self.grade_chain_text.invoke(payload)
            _capture_debug("text", "received", output_obj=output)
            parsed = _coerce_to_dict(output)
            _capture_debug("text", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
            validated = LLMGradingResponse(**parsed)
            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [text]")
            result = validated.model_dump()
            try:
                for g in result.get("grades", []) or []:
                    for bi in g.get("breakdown", []) or []:
                        if float(bi.get("marks_awarded", 0) or 0) <= 0:
                            continue
                        ev = bi.get("evidence", [])
                        ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                        if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                            bi["marks_awarded"] = 0.0
                            bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
            except Exception:
                pass
            return result
        except Exception as e:
            _capture_debug("text", "error", output_obj=locals().get("output"), error=e)
            logger.warning(f"Text grading parse failed; attempting one JSON repair pass: {e}")
            # Save the exception for the repair block below - `e` is scoped to
            # this except in Py3 and would be cleared on exit.
            _text_grade_err = e

        # Attempt 3: repair by asking the same model to output strict JSON only
        try:
            # Get the raw text from the previous output if possible
            raw_content = getattr(output, "content", None) if "output" in locals() else None
            raw_content = raw_content if raw_content is not None else str(_text_grade_err)

            repair_prompt = (
                "You MUST return ONLY valid JSON (no markdown, no commentary). "
                "Fix the following output to match this exact schema:\n"
                "{\n"
                "  \"grades\": [\n"
                "    {\n"
                "      \"question_number\": string,\n"
                "      \"score\": number,\n"
                "      \"total_marks\": number,\n"
                "      \"comments\": [string, ...],\n"
                "      \"correct_words\": [string, ...],\n"
                "      \"breakdown\": [\n"
                "        {\n"
                "          \"criterion\": string,\n"
                "          \"marks_awarded\": number,\n"
                "          \"max_possible\": number,\n"
                "          \"reason\": string,\n"
                "          \"evidence\": [string, ...],\n"
                "          \"comments_summary\": string\n"
                "        }\n"
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}\n\n"
                "OUTPUT TO FIX:\n"
                f"{raw_content}"
            )

            fixed = llm_grader.invoke(repair_prompt)
            _capture_debug("repair", "received", output_obj=fixed)
            fixed_content = getattr(fixed, "content", None)
            fixed_content = fixed_content if fixed_content is not None else str(fixed)
            fixed_json = self._extract_json_from_text(str(fixed_content))
            parsed = json.loads(fixed_json)
            _capture_debug("repair", "parsed", raw_text=json.dumps(parsed, ensure_ascii=False)[:debug_max_chars] if debug_enabled else None)
            validated = LLMGradingResponse(**parsed)
            logger.info(f"Grading complete → {self.student_name} (Q{self.question_number}) [repaired]")
            result = validated.model_dump()
            try:
                for g in result.get("grades", []) or []:
                    for bi in g.get("breakdown", []) or []:
                        if float(bi.get("marks_awarded", 0) or 0) <= 0:
                            continue
                        ev = bi.get("evidence", [])
                        ev_list = ev if isinstance(ev, list) else ([str(ev)] if ev else [])
                        if not _evidence_present_in_student([str(x) for x in ev_list if x is not None]):
                            bi["marks_awarded"] = 0.0
                            bi["reason"] = ("Marks revoked by guardrails: evidence not found verbatim in student answer")
            except Exception:
                pass
            return result
        except Exception as e2:
            _capture_debug("repair", "error", output_obj=locals().get("fixed"), error=e2)
            logger.error("Grading chain failed", exc_info=True)
            raise GradingError("Grading step failed") from e2

    def _build_grade_doc(self, parsed_grades: dict, questions_data: dict) -> dict:
        now_iso = datetime.utcnow().isoformat()

        # Conservative heuristic partial credit is OFF by default.
        # Enable explicitly with ENABLE_PARTIAL_CREDIT=1/true/yes.
        partial_credit_enabled = os.getenv("ENABLE_PARTIAL_CREDIT", "0").strip().lower() in {"1", "true", "yes", "y"}

        allowed_criteria = self._allowed_criteria_last_run or set()
        rubric_max_map = self._criterion_max_map_last_run or {}

        # Build a normalized lookup so paraphrased / truncated LLM output can
        # still resolve back to the canonical rubric criterion text.  Without
        # this, long narrative criteria (~1000+ chars) that the LLM trims when
        # echoing back fail the `criterion not in allowed_criteria` exact-match
        # check and their marks are silently dropped from the breakdown.
        def _norm_crit_key(s: str) -> str:
            if not isinstance(s, str):
                return ""
            t = s.replace(" ", " ").replace("–", "-").replace("-", "-")
            t = re.sub(r"\s+", " ", t).strip().lower()
            return t

        _norm_to_canonical: dict[str, str] = {}
        # List of (normalized_key, canonical_text) for bidirectional prefix matching:
        #   - LLM truncates canonical to its title  -> canon_nk.startswith(llm_nk)
        #   - LLM extends canonical with extra text -> llm_nk.startswith(canon_nk)
        _canonical_norm_pairs: list[tuple[str, str]] = []
        for _canon in allowed_criteria:
            _nk = _norm_crit_key(_canon)
            if not _nk:
                continue
            if _nk not in _norm_to_canonical:
                _norm_to_canonical[_nk] = _canon
            _canonical_norm_pairs.append((_nk, _canon))
        _dropped_out_of_rubric: list[str] = []  # collected for an INFO summary
        # Canonical criteria already resolved during this pass. Used to break
        # ties when an LLM-returned string matches SEVERAL rubric criteria:
        # prefer one nothing has claimed yet, so two different LLM entries
        # cannot both collapse onto the same criterion and silently discard
        # the other one's marks. (Seen with two criteria whose normalized text
        # shares its first 40 chars: "...0.25 for the 'Less net assets..." and
        # "...0.25 for the 'Less goodwill...".)
        _claimed_canonicals: set[str] = set()

        # Minimum normalized length required to attempt prefix matching.
        # Set just high enough that two distinct rubric criteria sharing a
        # short common prefix (e.g., "Profitability - ", "Pre-IPO - ")
        # cannot both match a single LLM-returned string.
        _PREFIX_MATCH_MIN_LEN = 20

        # Length of the distinctive leading prefix used for the third matching
        # layer (LLM and canonical share a long leading prefix but diverge
        # mid-string because the LLM paraphrased the middle/end of a long
        # rubric description). Set long enough that the prefix uniquely
        # identifies one rubric criterion in normal rubrics, short enough to
        # tolerate minor wording variations in the leading title sentence.
        _SHARED_LEADING_PREFIX_LEN = 40

        _STOPWORDS = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "for",
            "on",
            "at",
            "as",
            "is",
            "are",
            "be",
            "been",
            "being",
            "with",
            "from",
            "by",
            "this",
            "that",
            "these",
            "those",
            "should",
            "would",
            "will",
            "must",
            "therefore",
            "figure",
            "million",
            "thousand",
            "gbp",
        }

        def _partial_overlap_ok(criterion_text: str, evid_items: list[str]) -> bool:
            """Heuristic for partial credit.

            Returns True when evidence suggests the student is addressing the criterion but
            the response is incomplete/incorrect. Conservative: requires multiple keyword/number overlaps.
            """
            if not criterion_text or not evid_items:
                return False
            crit = _norm_for_evidence_match(criterion_text)
            ev_blob = _norm_for_evidence_match(" ".join(evid_items))
            if not crit or not ev_blob:
                return False

            # Numbers in criterion (e.g., 300,000 / 9/12) are strong signals.
            crit_nums = self._numbers_in_text(criterion_text)
            num_hits = 0
            for n in crit_nums:
                if self._contains_number_variant(ev_blob, n):
                    num_hits += 1

            # Keyword overlap: long-ish content words.
            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit) if w not in _STOPWORDS]
            # Emphasize accounting nouns that often matter for journals.
            boosted = {"goodwill", "disposal", "associate", "subsidiary", "investment", "nci", "retained", "earnings", "assets", "surplus", "revaluation", "oci", "impairment", "profit", "loss"}
            hits = 0
            for w in crit_words:
                if w in ev_blob:
                    hits += 1
            boosted_hits = sum(1 for w in boosted if w in crit and w in ev_blob)

            # Require either:
            # - strong keyword overlap, OR
            # - some keyword overlap + at least one number match.
            return (hits + boosted_hits) >= 2 or ((hits + boosted_hits) >= 1 and num_hits >= 1)

        def _quantize_to_quarter(mark: float) -> float:
            try:
                m = float(mark)
            except Exception:
                return 0.0
            return round(m / 0.25) * 0.25

        def _extract_expected_gbp_amount(criterion_text: str) -> Optional[float]:
            """Extract a single expected GBP amount from a criterion.

            Handles patterns like:
            - "GBP5.4 million"
            - "GBP630,000"
            Returns a float in absolute GBP units.

            Intentionally conservative: returns only the first clear GBP amount.
            """
            if not criterion_text or not isinstance(criterion_text, str):
                return None

            # Prefer amounts explicitly prefixed by GBP.
            m = re.search(
                r"\bgbp\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(million|m|thousand|k)?\b",
                criterion_text,
                flags=re.IGNORECASE,
            )
            if not m:
                return None

            raw = m.group(1)
            suffix = (m.group(2) or "").lower().strip()
            try:
                val = float(raw.replace(",", ""))
            except Exception:
                return None

            if suffix in ("million", "m"):
                val *= 1_000_000.0
            elif suffix in ("thousand", "k"):
                val *= 1_000.0
            return val

        def _safe_eval_arithmetic(expr: str) -> Optional[float]:
            """Safely evaluate simple arithmetic expressions.

            Allowed: + - * / parentheses, numeric literals, unary +/-.
            Also supports suffixes like 7.2m / 300k and percent literals like 25%.
            """
            if not expr or not isinstance(expr, str):
                return None

            s = expr.strip().lower()
            if len(s) > 120:
                return None

            # Normalize common tokens
            s = s.replace(",", "")
            s = s.replace("×", "*").replace("x", "*")

            # Convert percents: 25% => (25/100)
            s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)

            # Convert k/m suffixes: 7.2m => (7.2*1000000)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*m\b", r"(\1*1000000)", s)
            s = re.sub(r"(\d+(?:\.\d+)?)\s*k\b", r"(\1*1000)", s)

            # Strip currency symbols/words
            s = re.sub(r"[£$]", "", s)
            s = re.sub(r"\bgbp\b", "", s)
            s = re.sub(r"\s+", "", s)

            # Some extracted snippets include extra outer parentheses or a trailing ')'
            # (e.g., when pulled from inside a larger expression like 12750000+(7200000*9/12)).
            # Stripping outer parens makes parsing robust while keeping inner grouping.
            s = s.strip("()")

            # Must contain at least one operator.
            if not re.search(r"[+\-*/]", s):
                return None

            try:
                tree = ast.parse(s, mode="eval")
            except Exception:
                return None

            allowed_nodes = (
                ast.Expression,
                ast.BinOp,
                ast.UnaryOp,
                ast.Add,
                ast.Sub,
                ast.Mult,
                ast.Div,
                ast.USub,
                ast.UAdd,
                ast.Constant,
                ast.Load,
            )

            for node in ast.walk(tree):
                if not isinstance(node, allowed_nodes):
                    return None

            def _eval(n: ast.AST) -> float:
                if isinstance(n, ast.Expression):
                    return _eval(n.body)
                if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                    return float(n.value)
                if isinstance(n, ast.UnaryOp):
                    v = _eval(n.operand)
                    if isinstance(n.op, ast.USub):
                        return -v
                    if isinstance(n.op, ast.UAdd):
                        return v
                    raise ValueError("bad unary")
                if isinstance(n, ast.BinOp):
                    left = _eval(n.left)
                    right = _eval(n.right)
                    if isinstance(n.op, ast.Add):
                        return left + right
                    if isinstance(n.op, ast.Sub):
                        return left - right
                    if isinstance(n.op, ast.Mult):
                        return left * right
                    if isinstance(n.op, ast.Div):
                        return left / right
                    raise ValueError("bad binop")
                raise ValueError("bad node")

            try:
                out = float(_eval(tree))
                if not math.isfinite(out):
                    return None
                return out
            except Exception:
                return None

        def _award_if_calc_matches_expected(
            criterion_text: str,
            evid_items: list[str],
            max_possible: float,
        ) -> tuple[bool, str]:
            """Return (award?, reason_suffix) if evidence contains calc matching expected GBP amount."""
            expected = _extract_expected_gbp_amount(criterion_text)
            if expected is None:
                return False, ""

            ev_blob = " ".join([e for e in evid_items if isinstance(e, str)])
            if not ev_blob:
                return False, ""

            # Context guard: a criterion that explicitly names profit/contribution/revenue/income
            # as the concept being measured requires the evidence to also contain that vocabulary.
            # Without this, a calculation like "7,200,000*9/12" in the student's net-assets working
            # table would be incorrectly credited for a criterion about profit contribution, purely
            # because the arithmetic result equals the expected GBP amount.
            _INCOME_TERMS_RE = re.compile(
                r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
            )
            if _INCOME_TERMS_RE.search(criterion_text) and not _INCOME_TERMS_RE.search(ev_blob):
                return False, ""

            # Find candidate expressions like 7200000*9/12 or (7200000*9/12)
            expr_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[+\-*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )

            # Also extract pure multiplication/division sub-expressions, which may be nested
            # inside a larger '+' working line (e.g., 12750000+(7200000*9/12)).
            muldiv_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )

            candidates = [m.group(0) for m in expr_re.finditer(ev_blob)]
            candidates.extend([m.group(0) for m in muldiv_re.finditer(ev_blob)])
            # Also pull sub-expressions after '=' which often contain the calc.
            if "=" in ev_blob:
                rhs = ev_blob.split("=", 1)[1]
                candidates.extend([m.group(0) for m in expr_re.finditer(rhs)])
                candidates.extend([m.group(0) for m in muldiv_re.finditer(rhs)])

            # De-dup while preserving order.
            seen = set()
            uniq: list[str] = []
            for c in candidates:
                cc = c.strip()
                if not cc:
                    continue
                key = re.sub(r"\s+", "", cc.lower())
                if key in seen:
                    continue
                seen.add(key)
                uniq.append(cc)

            # Evaluate candidates and check for match.
            for expr in uniq[:12]:
                val = _safe_eval_arithmetic(expr)
                if val is None:
                    continue
                if math.isclose(val, expected, rel_tol=1e-6, abs_tol=2.0):
                    return True, f"Awarded by numeric-calc check (evidence calc {expr.strip()} = {expected:,.0f} GBP)."

            return False, ""

        def _evidence_has_calc_result(ev_blob: str, expected_value: float) -> bool:
            """Return True if evidence contains an arithmetic expression that evaluates to expected_value."""
            if expected_value is None or not isinstance(expected_value, (int, float)):
                return False
            if not ev_blob or not isinstance(ev_blob, str):
                return False

            expr_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[+\-*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )
            muldiv_re = re.compile(
                r"[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?(?:\s*[*/x×]\s*\(?\s*[0-9][0-9,]*(?:\.[0-9]+)?(?:\s*[mk])?\s*\)?)+",
                flags=re.IGNORECASE,
            )
            candidates = [m.group(0) for m in expr_re.finditer(ev_blob)]
            candidates.extend([m.group(0) for m in muldiv_re.finditer(ev_blob)])
            if "=" in ev_blob:
                rhs = ev_blob.split("=", 1)[1]
                candidates.extend([m.group(0) for m in expr_re.finditer(rhs)])
                candidates.extend([m.group(0) for m in muldiv_re.finditer(rhs)])

            seen = set()
            for expr in candidates:
                key = re.sub(r"\s+", "", (expr or "").lower())
                if not key or key in seen:
                    continue
                seen.add(key)
                val = _safe_eval_arithmetic(expr)
                if val is None:
                    continue
                if math.isclose(float(val), float(expected_value), rel_tol=1e-6, abs_tol=2.0):
                    return True
            return False

        def _find_calc_snippet_in_student(criterion_text: str) -> Optional[str]:
            """If criterion contains a simple x/* and / calc in parentheses, try to find the same calc in the student text.

            Returns a short expression snippet (e.g., "7200000*9/12") if found.
            """
            if not isinstance(criterion_text, str) or not criterion_text:
                return None

            m = re.search(r"\(([^)]*)\)", criterion_text)
            if not m:
                return None

            expr = m.group(1)
            expr_low = expr.lower().replace("×", "x")
            if "+" in expr_low or "-" in expr_low:
                return None

            # Tokenize similarly to _compute_simple_calc_from_criterion
            parts = re.split(r"\s*(x|\*|/)\s*", expr_low)
            parts = [p.strip() for p in parts if p and p.strip()]
            if len(parts) < 3:
                return None

            # Build a regex that matches the numeric expression in student text.
            # Convert operands to canonical numeric strings where possible (e.g., 7.2 million -> 7200000).
            def _int_with_commas_pattern(ival: str) -> str:
                """Match either the raw digits or a correctly comma-grouped variant.

                Example: 7200000 matches "7200000" or "7,200,000" (allowing optional spaces around commas).
                """
                if not ival or not ival.isdigit():
                    return re.escape(ival or "")
                if len(ival) <= 3:
                    return re.escape(ival)

                first_len = len(ival) % 3
                if first_len == 0:
                    first_len = 3
                groups = [ival[:first_len]]
                for j in range(first_len, len(ival), 3):
                    groups.append(ival[j:j + 3])
                grouped = r"\s*,\s*".join(re.escape(g) for g in groups)
                return rf"(?:{re.escape(ival)}|{grouped})"

            pattern_parts: list[str] = []
            i = 0
            while i < len(parts):
                token = parts[i]
                if token in {"x", "*", "/"}:
                    if token == "/":
                        pattern_parts.append(r"\s*/\s*")
                    else:
                        pattern_parts.append(r"\s*[x\*]\s*")
                    i += 1
                    continue

                # Operand
                tok_compact = re.sub(r"\s+", "", token.lower())
                # Special-case fractions (e.g., 9/12) so we can match either the literal
                # fraction form or its decimal equivalent in student workings.
                if re.fullmatch(r"\d+(?:\.\d+)?/\d+(?:\.\d+)?", tok_compact):
                    try:
                        num_s, den_s = tok_compact.split("/", 1)
                    except Exception:
                        return None
                    frac_pat = rf"{re.escape(num_s)}\s*/\s*{re.escape(den_s)}"
                    val = StudentGrader._parse_number_token(token)
                    if val is not None:
                        # Keep a conservative decimal string (avoid scientific notation).
                        dec = (f"{float(val):.6f}").rstrip("0").rstrip(".")
                        if dec:
                            pattern_parts.append(rf"(?:{frac_pat}|{re.escape(dec)})")
                        else:
                            pattern_parts.append(frac_pat)
                    else:
                        pattern_parts.append(frac_pat)
                    i += 1
                    continue

                val = StudentGrader._parse_number_token(token)
                if val is None:
                    return None

                # Prefer integer representation when close.
                if abs(val - round(val)) < 1e-6:
                    ival = str(int(round(val)))
                    pattern_parts.append(_int_with_commas_pattern(ival))
                else:
                    # Keep a conservative float token
                    sval = str(val)
                    pattern_parts.append(re.escape(sval))
                i += 1

            if not pattern_parts:
                return None

            regex = re.compile("".join(pattern_parts), flags=re.IGNORECASE)
            hay = getattr(self, "_student_text_last_run", "") or ""
            m2 = regex.search(hay)
            if not m2:
                # Try a more permissive match on the normalized blob
                m3 = regex.search(student_blob_norm)
                if not m3:
                    return None
                snippet = m3.group(0)
            else:
                snippet = m2.group(0)

            snippet = (snippet or "").strip()
            snippet = snippet.replace(" ", "")
            snippet = snippet.replace("×", "*").replace("x", "*")
            # Keep snippet short.
            return snippet[:60] if snippet else None

        def _norm_for_evidence_match(s: str) -> str:
            if not s or not isinstance(s, str):
                return ""
            s = s.replace("\u00a0", " ")
            s = s.replace("×", "x")
            s = s.replace("–", "-").replace("-", "-")
            s = re.sub(r"\s+", " ", s).strip().lower()
            s = re.sub(r"[^a-z0-9%/().,\- ]+", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            return s

        student_blob_norm = _norm_for_evidence_match(getattr(self, "_student_text_last_run", "") or "")

        # Normalized question/model blobs for detecting evidence that is copied from the question
        # text or marking guide (e.g., headings). If evidence only appears in these sources,
        # we treat it as "tainted" and revoke marks.
        question_blob_norm = _norm_for_evidence_match(getattr(self, "_question_text_last_run", "") or "")
        model_blob_norm = _norm_for_evidence_match(getattr(self, "_model_text_last_run", "") or "")
        reference_blob_norm = " ".join([t for t in (question_blob_norm, model_blob_norm) if t])

        def _evidence_has_untainted_snippet(evidence_items: list[str]) -> bool:
            """Return True if at least one evidence snippet is present in student text AND not present in reference text."""
            if not student_blob_norm:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue
                if ev_norm not in student_blob_norm:
                    continue
                if reference_blob_norm and ev_norm in reference_blob_norm:
                    # Evidence also appears in question/markscheme.
                    # Only flag as "tainted" when the snippet is pure text (no numbers).
                    # Evidence containing a meaningful number (3+ digits) almost certainly
                    # represents the student's own calculation or stated value - the fact that
                    # the same number appears in the model answer just means the student got it right.
                    # Pure-text phrases (e.g. section headings like "statement of financial position")
                    # with 2+ content words ARE tainted - they were likely extracted from the PDF template.
                    has_meaningful_number = bool(re.search(r"\d{3,}|\d+\.\d{2,}", ev_norm))
                    if not has_meaningful_number:
                        alpha_words = re.findall(r"[a-z]{4,}", ev_norm)
                        if len(alpha_words) >= 2:
                            continue
                return True
            return False

        def _is_distinctive_short_evidence(ev_norm: str) -> bool:
            if not ev_norm or not isinstance(ev_norm, str):
                return False
            compact = ev_norm.replace(" ", "")
            if re.search(r"\d", ev_norm):
                return len(compact) >= 4
            if "/" in ev_norm or "%" in ev_norm:
                return len(compact) >= 3
            words = ev_norm.split()
            return len(words) <= 3 and any(len(w) >= 7 for w in words)

        def _criterion_evidence_alignment_ok(criterion_text: str, evid_items: list[str]) -> bool:
            """Heuristic guardrail: ensure awarded evidence actually aligns to the criterion.

            This is NOT a correctness check vs the model answer.
            It prevents awarding marks when the evidence only weakly relates to a long narrative criterion
            (common failure mode: giving partial credit for mentioning a vague related term).

            Keep it conservative and only apply to longer/descriptive criteria.
            """
            if not isinstance(criterion_text, str) or not evid_items:
                return True

            crit_norm = _norm_for_evidence_match(criterion_text)
            if not crit_norm:
                return True

            # Only enforce this alignment guardrail for narrative criteria.
            # Numeric/method criteria (with GBP amounts, large calculations) are better handled by the
            # other guardrails (evidence-present, taint check, strict-number match for full marks).
            # Enforcing alignment here tends to incorrectly revoke legitimate own-figure work.
            # HOWEVER: criteria whose only digits are dates or small ordinals (e.g. "31 May 20X4",
            # "within 30 days") are still narrative criteria - keep alignment enforcement for those.
            if re.search(r"\d", crit_norm):
                # Has a large number (3+ digits) or explicit GBP/currency marker → numeric criterion.
                has_large_num = bool(re.search(r"\b\d{3,}\b", crit_norm))
                has_currency = bool(re.search(r"[£$]|\bgbp\b", crit_norm, re.IGNORECASE))
                if has_large_num or has_currency:
                    return True
                # Only small numbers (≤ 2 digits) present - treat as narrative (date-qualified).
                # Fall through to apply alignment check.
            if crit_norm.strip().startswith("dr ") or crit_norm.strip().startswith("cr "):
                return True

            # Do not enforce on short labels/headings.
            if len(crit_norm) < 50 and len(crit_norm.split()) < 8:
                return True

            ev_blob = _norm_for_evidence_match(" ".join([e for e in evid_items if isinstance(e, str)]))
            if not ev_blob:
                return False

            # Numbers/ratios in evidence are strong anchors.
            has_number = bool(re.search(r"\d", ev_blob))
            has_ratio = bool(re.search(r"\d+\s*/\s*\d+", ev_blob))

            stop = {
                "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "at", "as", "is", "are",
                "was", "were", "be", "been", "being", "with", "from", "by", "this", "that", "these",
                "those", "should", "would", "will", "must", "therefore", "account", "accounts", "year",
                "statement", "financial", "reporting", "treatment",
            }

            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit_norm) if w not in stop]
            if not crit_words:
                return True

            ev_words = set(re.findall(r"[a-z]{4,}", ev_blob))

            # If evidence is predominantly numeric (no/one content word) and contains meaningful
            # numbers (3+ digits), trust the LLM's criterion-evidence pairing.
            # Numeric evidence like "25%*(12,750,000-2,750,000) 2,500,000.00" is inherently
            # specific - the earlier guards already confirmed it exists in the student text
            # and is not tainted from the question/markscheme.
            if len(ev_words) <= 1 and bool(re.search(r"\d{3,}", ev_blob)):
                return True

            # Evidence with few content words (2-3) can pass IF a specific number from the
            # criterion also appears in the evidence. This handles accounting synonym issues
            # (e.g., "staff expense 480,000" for criterion about "GBP480,000 charged to P/L")
            # without being overly permissive.
            if 2 <= len(ev_words) <= 3 and bool(re.search(r"\d{3,}", ev_blob)):
                crit_nums_raw = re.findall(r"\d[\d,]*\.?\d*", crit_norm)
                ev_blob_compact = ev_blob.replace(",", "").replace(" ", "")
                for cn in crit_nums_raw:
                    cn_stripped = cn.replace(",", "")
                    if len(cn_stripped) >= 3 and cn_stripped in ev_blob_compact:
                        return True

            # Accounting domain synonym groups for alignment matching.
            # Words in the same group are treated as equivalent when computing
            # overlap, so "FX gain" matches "exchange movement", "close" matches
            # "closing", "debited" matches "charged", etc.
            _SYNONYM_GROUPS = [
                frozenset({"exchange", "forex", "currency", "translation", "retranslation"}),
                frozenset({"movement", "gain", "loss", "difference", "change"}),
                frozenset({"charged", "debited", "expensed", "recognised", "recognized", "impact"}),
                frozenset({"closing", "close", "closed", "yearend"}),
                frozenset({"revaluation", "revalued", "remeasured", "remeasurement", "restated"}),
                frozenset({"consolidation", "consolidated", "consolidate"}),
                frozenset({"profit", "income", "earnings"}),
                frozenset({"comprehensive", "reserve", "surplus"}),
                # Personnel synonyms: "Four cyclists departed / 8 riders" ↔ "8 staff members"
                frozenset({"cyclist", "rider", "employee", "staff", "member", "player", "worker"}),
                # Associate / equity method synonyms
                frozenset({"associate", "equity", "accounted"}),
            ]
            _WORD_TO_GROUP: dict[str, int] = {}
            for _gi, _grp in enumerate(_SYNONYM_GROUPS):
                for _sw in _grp:
                    _WORD_TO_GROUP[_sw] = _gi
                    # Also map common plural form so "cyclists" matches "cyclist", etc.
                    if not _sw.endswith("s"):
                        _WORD_TO_GROUP[_sw + "s"] = _gi

            # Compute overlap: exact word matches first.
            exact_overlap = {w for w in crit_words if w in ev_words}
            overlap = len(exact_overlap)

            # Add synonym-based overlap: for each unmatched criterion word that
            # belongs to a synonym group, check if any evidence word is in the
            # same group.  Each group contributes at most one additional point.
            matched_groups: set[int] = set()
            for w in crit_words:
                if w in exact_overlap:
                    continue
                g = _WORD_TO_GROUP.get(w)
                if g is None or g in matched_groups:
                    continue
                if any(_WORD_TO_GROUP.get(ew) == g for ew in ev_words):
                    overlap += 1
                    matched_groups.add(g)

            # Require at least 2 overlapping content words, OR 1 overlap plus a strong anchor.
            if overlap >= 2:
                return True
            if overlap >= 1 and (has_number or has_ratio):
                return True
            # A single long domain-specific word (≥7 chars) is a strong enough anchor on its own.
            # e.g. "associate" in both criterion and evidence passes without a numeric anchor.
            long_exact = {w for w in exact_overlap if len(w) >= 7}
            if long_exact:
                return True
            return False

        def _find_journal_line_in_student(criterion_text: str) -> Optional[str]:
            """Try to recover a verbatim journal/workings line from the student text.

            Generic recovery for criteria that are journal-line-shaped (Dr/Cr) and include a numeric amount.
            Returns a short snippet from the student's submitted text.
            """
            if not isinstance(criterion_text, str) or not criterion_text:
                return None

            crit = criterion_text.strip()
            crit_low = crit.lower().strip()
            if not (crit_low.startswith("dr ") or crit_low.startswith("cr ")):
                return None

            crit_nums = self._numbers_in_text(crit)
            if not crit_nums:
                return None

            # Extract a couple of account-ish tokens to increase precision.
            crit_words = [w for w in re.findall(r"[a-z]{4,}", crit_low) if w not in {"dr", "cr", "gbp", "usd", "eur", "million", "thousand"}]
            crit_words = crit_words[:4]

            hay = getattr(self, "_student_text_last_run", "") or ""
            if not isinstance(hay, str) or not hay.strip():
                return None

            # Work line-by-line to preserve journals/tables.
            lines = [ln.strip() for ln in hay.splitlines() if ln and ln.strip()]

            # Precompute normalized versions for matching.
            for ln in lines:
                ln_low = ln.lower()
                if not ("dr" in ln_low.split() or "cr" in ln_low.split() or ln_low.startswith("dr ") or ln_low.startswith("cr ")):
                    continue

                # Must contain correct direction.
                if crit_low.startswith("dr ") and not (ln_low.startswith("dr ") or "dr" in ln_low.split()):
                    continue
                if crit_low.startswith("cr ") and not (ln_low.startswith("cr ") or "cr" in ln_low.split()):
                    continue

                # Must contain at least one of the criterion numbers.
                if not any(self._contains_number_variant(ln, n) for n in crit_nums):
                    continue

                # Must contain at least one criterion content word (if any) to avoid matching random journal lines.
                if crit_words and not any(w in ln_low for w in crit_words):
                    continue

                # Return verbatim line, truncated.
                return ln[:160]

            return None

        def _evidence_present(evidence_items: list[str]) -> bool:
            if not student_blob_norm:
                return False
            for ev in evidence_items:
                if not ev or not isinstance(ev, str):
                    continue
                ev_norm = _norm_for_evidence_match(ev)
                if len(ev_norm) < 12 and not _is_distinctive_short_evidence(ev_norm):
                    continue
                if ev_norm in student_blob_norm:
                    return True
                # Sliding-window fallback: LLMs sometimes add/remove minor punctuation in
                # verbatim quotes.  A contiguous N-word window in the student text is
                # sufficient to confirm the quote is genuine.
                words = [w for w in ev_norm.split() if len(w) >= 3]
                if len(words) >= 10:
                    for i in range(0, min(len(words) - 9, 8)):
                        window = " ".join(words[i:i + 10])
                        if window in student_blob_norm:
                            return True
                elif len(words) >= 5:
                    for i in range(0, min(len(words) - 4, 6)):
                        window = " ".join(words[i:i + 5])
                        if window in student_blob_norm:
                            return True
            return False

        grades = parsed_grades.get("grades", [])
        if not grades:
            raise GradingError("No grades returned from LLM")

        main_grade = grades[0]  # assuming single main grade object
        all_comments = main_grade.get("comments", []) if isinstance(main_grade.get("comments", []), list) else []

        def _is_annotation_comment(s: str) -> bool:
            if not s or not isinstance(s, str):
                return False
            if "\n" in s or "\r" in s:
                return False
            if "→" in s:
                left, right = s.split("→", 1)
            elif "->" in s:
                left, right = s.split("->", 1)
            else:
                return False
            left = (left or "").strip()
            right = (right or "").strip()
            if not left or not right:
                return False
            # Expect two short sentences after the arrow; require at least 2 periods.
            if right.count(".") < 2:
                return False
            return True

        annotation_comments: list[str] = []
        unanchored_comments: list[str] = []
        for c in all_comments:
            try:
                (annotation_comments if _is_annotation_comment(c) else unanchored_comments).append(str(c))
            except Exception:
                continue

        normalized_breakdown = []
        evidence_warnings = []
        sum_awarded_calc = 0.0  # debug only

        _id_map = self._criterion_id_map_last_run or {}
        for item in main_grade.get("breakdown", []) or []:
            criterion = item.get("criterion", "Unknown")
            criterion = str(criterion or "").strip()

            # CRITERION ID takes precedence over the echoed text.
            # The model returns e.g. "TB03" and we substitute the rubric's own
            # wording, so a criterion can no longer be lost to a near-identical
            # twin, paraphrased into unrecognisability, or dropped because
            # retyping 600 characters was too costly.
            if _id_map and not self._holistic_grading:
                _rid = str(item.get("criterion_id", "") or "").strip().upper()
                _canon_by_id = _id_map.get(_rid)
                if _canon_by_id:
                    if criterion != _canon_by_id:
                        logger.debug(
                            f"Resolved criterion by id {_rid}: "
                            f"'{criterion[:60]}' -> '{_canon_by_id[:60]}'"
                        )
                    criterion = _canon_by_id
                    item["criterion"] = _canon_by_id
                    _claimed_canonicals.add(_canon_by_id)
                elif _rid:
                    logger.debug(f"Unknown criterion_id '{_rid}' - falling back to text match")

            # Strictly require that the criterion exists in the rubric we provided.
            # This prevents the LLM from inventing criteria or grading headings/commentary.
            # When criteria were synthesized from answer text, relax this check since the
            # LLM may reasonably rephrase the auto-generated criterion descriptions.
            # For holistic grading, skip rubric validation entirely - breakdown items are
            # sub-questions, not rubric criteria.
            if not criterion:
                continue
            if allowed_criteria and not self._criteria_were_synthesized and not self._holistic_grading:
                if criterion in allowed_criteria:
                    # Exact hit — claim it so a later ambiguous match prefers a
                    # different criterion rather than collapsing onto this one.
                    _claimed_canonicals.add(criterion)
                else:
                    # Try normalized whole-string match (case + whitespace insensitive).
                    nk = _norm_crit_key(criterion)
                    canonical = _norm_to_canonical.get(nk)
                    if canonical is None and len(nk) >= _PREFIX_MATCH_MIN_LEN:
                        # Bidirectional prefix match: handle BOTH
                        #   (a) LLM truncated a long canonical to just its title
                        #       -> canon_nk.startswith(nk)
                        #   (b) LLM appended extra commentary after the canonical
                        #       -> nk.startswith(canon_nk)
                        # If exactly one rubric criterion matches, accept it.
                        # If multiple match, pick the one whose normalized
                        # length is closest to the LLM's text (best-fit) so
                        # ambiguity tends to land on the right criterion.
                        _matches: list[str] = []
                        for _canon_nk, _canon_full in _canonical_norm_pairs:
                            if _canon_nk.startswith(nk) or nk.startswith(_canon_nk):
                                _matches.append(_canon_full)
                        if len(_matches) == 1:
                            canonical = _matches[0]
                        elif len(_matches) > 1:
                            _pool = [c for c in _matches if c not in _claimed_canonicals] or _matches
                            canonical = min(
                                _pool,
                                key=lambda c: abs(len(_norm_crit_key(c)) - len(nk)),
                            )
                    if canonical is None and len(nk) >= _SHARED_LEADING_PREFIX_LEN:
                        # Shared-leading-prefix fallback: handles the common case
                        # where the LLM echoes the first ~40 chars of the rubric
                        # criterion verbatim (the distinctive title) and then
                        # PARAPHRASES the middle/end of the long description.
                        # Neither startswith direction catches this, but two
                        # strings sharing a long distinctive leading prefix are
                        # the same criterion in practice.
                        _llm_head = nk[:_SHARED_LEADING_PREFIX_LEN]
                        _matches = [
                            c for _ck, c in _canonical_norm_pairs
                            if _ck.startswith(_llm_head)
                        ]
                        if len(_matches) == 1:
                            canonical = _matches[0]
                        elif len(_matches) > 1:
                            _pool = [c for c in _matches if c not in _claimed_canonicals] or _matches
                            canonical = min(
                                _pool,
                                key=lambda c: abs(len(_norm_crit_key(c)) - len(nk)),
                            )
                    if canonical is not None:
                        # Rewrite to canonical so downstream lookups (rubric_max_map,
                        # category map, position map) all hit.
                        criterion = canonical
                        item["criterion"] = canonical
                        _claimed_canonicals.add(canonical)
                    else:
                        _dropped_out_of_rubric.append(criterion[:90])
                        logger.debug(f"Skipping out-of-rubric criterion: '{criterion}'")
                        continue

            if not self._holistic_grading and not self._is_valid_criterion(criterion):
                logger.debug(f"Skipping invalid criterion: '{criterion}'")
                continue

            original_marks_awarded = float(item.get("marks_awarded", 0) or 0)

            # Enforce max_possible from the rubric (ignore LLM-supplied max_possible).
            # If the rubric doesn't have a numeric max for this criterion, treat it as non-scoreable.
            # For holistic grading, use the authoritative max_marks from _holistic_sub_questions
            # (sourced from the question paper), falling back to the LLM's value only if not found.
            # Note: model answer sub-criteria marks (e.g. 0.5/point) are still used by the LLM to
            # score individual points - the cap only limits the final marks_awarded total.
            rubric_max = rubric_max_map.get(criterion)
            if self._holistic_grading:
                sq_label = item.get("_sub_question", "")
                authoritative_max = 0.0
                for hq in (self._holistic_sub_questions or []):
                    if str(hq.get("sub_question", "")).strip() == str(sq_label).strip():
                        authoritative_max = float(hq.get("max_marks", 0) or 0)
                        break
                if authoritative_max > 0:
                    max_possible = authoritative_max
                else:
                    llm_max = float(item.get("max_possible", 0) or 0)
                    max_possible = llm_max if llm_max > 0 else original_marks_awarded
            elif rubric_max is None and self._criteria_were_synthesized:
                # For synthesized criteria, the LLM may rephrase - trust the LLM's max_possible
                llm_max = float(item.get("max_possible", 0) or 0)
                max_possible = llm_max if llm_max > 0 else original_marks_awarded
            else:
                max_possible = float(rubric_max) if isinstance(rubric_max, (int, float)) else 0.0
            # Clamp marks_awarded: never exceed max_possible and quantize to 0.25 steps.
            marks_awarded = min(original_marks_awarded, max_possible)
            marks_awarded = round(marks_awarded / 0.25) * 0.25

            raw_evidence = item.get("evidence", [])
            evid_list = []

            # Super-robust parsing
            if isinstance(raw_evidence, str):
                cleaned = raw_evidence.replace('\\n', '\n').replace('\\r', '').replace('\\t', ' ').strip()
                if cleaned:
                    # Do NOT split on '|' because that destroys table rows.
                    evid_list = [s.strip() for s in re.split(r'\n|;', cleaned) if s.strip()]
            elif isinstance(raw_evidence, list):
                for ev in raw_evidence:
                    if ev is None:
                        continue
                    if isinstance(ev, str):
                        cleaned_ev = ev.replace('\\n', '\n').strip()
                        if cleaned_ev:
                            evid_list.append(cleaned_ev)
                    else:
                        cleaned_ev = str(ev).strip()
                        if cleaned_ev:
                            evid_list.append(cleaned_ev)
            else:
                # fallback for odd types
                cleaned = str(raw_evidence).strip()
                if cleaned:
                    evid_list = [cleaned]

            # Evidence expansion for tables: keep original evidence, but also add
            # OCR-friendly variants when evidence contains pipe-separated columns.
            # This helps PDF annotation anchor marks within tables.
            try:
                expanded: list[str] = []
                for ev in list(evid_list or []):
                    if not isinstance(ev, str) or "|" not in ev:
                        continue
                    cells = [c.strip() for c in ev.split("|")]
                    cells = [c for c in cells if c]
                    if len(cells) < 2:
                        continue

                    # Prefer a compact "label value" variant when last cell looks numeric.
                    last = cells[-1]
                    if re.search(r"\d", last):
                        compact = f"{cells[0]} {last}".strip()
                        if compact and compact not in evid_list and compact not in expanded:
                            expanded.append(compact)

                    joined = " ".join(cells).strip()
                    if joined and joined not in evid_list and joined not in expanded:
                        expanded.append(joined)

                if expanded:
                    evid_list = expanded + evid_list
            except Exception:
                pass

            # Enforce evidence: if we can't parse evidence, we cannot justify awarding marks.
            # This prevents incorrect awards when the model "guesses".
            # For holistic grading, still require evidence but don't revoke - it's possible
            # the LLM awarded marks for overall understanding without pinpointing exact lines.
            if marks_awarded > 0 and not evid_list and not self._holistic_grading:
                evidence_warnings.append(f"Marks revoked (missing evidence): {criterion}")
                marks_awarded = 0.0

            # Stronger guardrail: evidence must actually exist in the student answer text.
            # This prevents marks being awarded when the LLM fabricates an evidence quote.
            # SKIP for holistic grading - the LLM's holistic comparison is trusted.
            if marks_awarded > 0 and evid_list and student_blob_norm and not self._holistic_grading:
                if not _evidence_present(evid_list):
                    evidence_warnings.append(f"Marks revoked (evidence not found in student answer): {criterion}")
                    marks_awarded = 0.0

            # Strongest guardrail: do not award marks based on evidence copied from the question/rubric.
            # This prevents awarding marks from section headings that appear in the PDF but contain no student work.
            # SKIP for synthesized criteria and holistic grading - theoretical answers naturally share
            # terminology with the model answer.
            if marks_awarded > 0 and evid_list and student_blob_norm and not self._criteria_were_synthesized and not self._holistic_grading:
                if reference_blob_norm and not _evidence_has_untainted_snippet(evid_list):
                    evidence_warnings.append(f"Marks revoked (evidence appears in question/markscheme, not student work): {criterion}")
                    marks_awarded = 0.0

            # Income/profit context revocation: a criterion that explicitly names profit,
            # contribution, revenue, income, or earnings as the concept measured requires the
            # evidence to also contain that vocabulary.  Without this, a calculation like
            # "7,200,000*9/12" in the student's net-assets working table would remain credited
            # for a profit-contribution criterion after the LLM incorrectly awarded marks.
            # This is intentionally narrow (only income-context terms) to avoid revoking
            # legitimate marks for criteria that describe profit without using that exact word.
            # Limit to micro-criteria (≤ 1 mark); section-total criteria with large marks
            # are less likely to be wrongly awarded and should not be revoked this way.
            # SKIP for journal and calc criteria - they have dedicated direction/number guards.
            # Those criteria contain "profit or loss" as an ACCOUNT NAME not a concept check,
            # so the income guard fires spuriously (e.g. "Dr Profit or loss 167" → evidence
            # shows "Dr Revaluation Loss (PL)" which is equivalent but lacks the word "profit").
            # Look up category from rubric cache (LLM output never includes category).
            _income_cat = self._criterion_category_map_last_run.get(
                criterion, str(item.get("category", "") or "")
            ).lower()
            # Skip for journal/calc (dedicated guards) and for narrative criteria
            # - narrative criteria mention profit/revenue as CONCEPTS being explained,
            # not as values to compute; the student's evidence naturally paraphrases
            # ("associate instead of subsidiary", "consolidated for the whole year")
            # without needing the specific vocabulary tokens.
            _skip_income_guard = _income_cat in ("journal", "calculation", "calc", "narrative")
            if (
                not _skip_income_guard
                and marks_awarded > 0
                and evid_list
                and isinstance(criterion, str)
                and float(max_possible) <= 1.0
                and not self._holistic_grading
            ):
                _INCOME_REVOKE_RE = re.compile(
                    r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
                )
                if _INCOME_REVOKE_RE.search(criterion):
                    _ev_income_ctx = " ".join(evid_list)
                    # Accept P&L, P/L, PL, loss, OCI, exchange/FX reserve, comprehensive,
                    # NCI (non-controlling interest share of profit), EPS (earnings per share),
                    # and plural forms (profits, earnings) as valid income-context evidence.
                    _INCOME_EVID_RE = re.compile(
                        r"\b(profits?|contribution|revenue|income|earnings?|loss|oci|"
                        r"exchange|reserve|comprehensive|nci|eps)\b"
                        r"|p[&/]l|\bpl\b|\bfx\b",
                        re.IGNORECASE,
                    )
                    if not _INCOME_EVID_RE.search(_ev_income_ctx):
                        evidence_warnings.append(
                            f"Marks revoked (profit-criterion evidence lacks profit context): {criterion}"
                        )
                        marks_awarded = 0.0

            # Alignment guardrail: for longer narrative criteria, require evidence to actually align.
            # Prevents awarding marks for vague mentions (e.g., saying "exchange difference" but not stating the required treatment).
            # Skip for large-number / calculation criteria; those are handled by numeric guardrails.
            # Also skip for Dr/Cr journal criteria, synthesized criteria, and holistic grading.
            # BUT apply even when criterion has small numbers (dates, ordinals) - those are still narrative.
            def _criterion_has_large_number(crit: str) -> bool:
                c = _norm_for_evidence_match(crit)
                return bool(re.search(r"\b\d{3,}\b", c)) or bool(
                    re.search(r"[£$]|\bgbp\b", c, re.IGNORECASE)
                )

            if (
                marks_awarded > 0
                and evid_list
                and student_blob_norm
                and isinstance(criterion, str)
                and criterion.strip()
                and not _criterion_has_large_number(criterion)
                and not criterion.strip().lower().startswith(("dr ", "cr "))
                and float(max_possible) < 1.0
                and not self._criteria_were_synthesized
                and not self._holistic_grading
            ):
                if not _criterion_evidence_alignment_ok(str(criterion), evid_list):
                    evidence_warnings.append(f"Marks revoked (weak evidence alignment to criterion): {criterion}")
                    marks_awarded = 0.0

            # Negation/contradiction guard: if a criterion states something "is required"
            # or "must" occur, and the evidence explicitly states the opposite (e.g., "no
            # impairment review", "not required"), revoke marks.
            # This catches the case where the alignment guardrail was bypassed (e.g., because
            # the criterion contains a date like "31 May 20X4") but the evidence contradicts
            # the criterion's core assertion.
            if marks_awarded > 0 and evid_list and isinstance(criterion, str) and not self._holistic_grading:
                _crit_lower = criterion.lower()
                if re.search(r"\brequired\b|\bnecessary\b|\bmust\b", _crit_lower):
                    _NEG_STOP = {
                        "required", "necessary", "must", "should", "would", "which", "this",
                        "that", "also", "have", "been", "will", "need", "needs", "being",
                        "review", "report", "audit", "before", "after",
                    }
                    _crit_key = [
                        w for w in re.findall(r"[a-z]{4,}", _crit_lower) if w not in _NEG_STOP
                    ]
                    _ev_combined = " ".join(evid_list).lower()
                    # "no of ord shares" is an abbreviation of "number of",
                    # not a negation. Left in, the guard reads it as "no ...
                    # shares" and revokes a correct answer - it cost EPS03 its
                    # 0.5 on a share count that was right. Same for "no." and
                    # "nos.", which students use for the same thing.
                    _ev_combined = re.sub(r"\bnos?\.?\s+of\b", "number of", _ev_combined)
                    for _kw in _crit_key[:6]:
                        # Match "no <optional words> <keyword-prefix>" - covers "no impairments"
                        # when keyword is "impairment" and similar plural/suffix variations.
                        _kw_prefix = _kw[:min(len(_kw), 7)]
                        if re.search(rf"\bno\s+(?:\w+\s+){{0,2}}{re.escape(_kw_prefix)}", _ev_combined) or \
                           re.search(rf"\bnot\s+(?:\w+\s+){{0,2}}{re.escape(_kw_prefix)}", _ev_combined):
                            evidence_warnings.append(
                                f"Marks revoked (evidence contradicts required criterion): {criterion}"
                            )
                            marks_awarded = 0.0
                            break

            # Evidence recovery for numeric-calc criteria (SKIP for holistic grading):
            # Sometimes the LLM attaches unhelpful evidence (e.g., just "Profit after tax 7,200,000")
            # even though the student has the exact calculation elsewhere (e.g., "7200000*9/12").
            # If marks are currently 0 and we can find the calc in the student's text, use it as evidence
            # and award full marks.
            if marks_awarded == 0 and max_possible > 0 and student_blob_norm and not self._holistic_grading:
                expected_gbp = _extract_expected_gbp_amount(str(criterion))
                expected_calc = self._compute_simple_calc_from_criterion(str(criterion))
                if expected_gbp is not None and expected_calc is not None:
                    if math.isclose(float(expected_gbp), float(expected_calc), rel_tol=1e-6, abs_tol=2.0):
                        snippet = _find_calc_snippet_in_student(str(criterion))
                        # Context guard: if the criterion is about profit/contribution/income,
                        # only accept the recovered snippet when the evidence or snippet itself
                        # also contains profit-context vocabulary.  A "7,200,000*9/12" pattern
                        # found in a net-assets table must not be credited for a profit criterion.
                        if snippet:
                            _INCOME_TERMS_RE = re.compile(
                                r"\b(profit|contribution|revenue|income|earning)\b", re.IGNORECASE
                            )
                            if _INCOME_TERMS_RE.search(str(criterion)):
                                ev_ctx = _norm_for_evidence_match(
                                    " ".join((evid_list or []) + [snippet])
                                )
                                if not _INCOME_TERMS_RE.search(ev_ctx):
                                    snippet = None
                        if snippet:
                            # Attach recovered evidence and award.
                            if snippet not in evid_list:
                                evid_list = [snippet] + evid_list
                            marks_awarded = float(max_possible)
                            existing = (item.get("reason", "") or "").strip()
                            add = f"Awarded by calc-evidence recovery (found '{snippet}' in student answer)."
                            item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # Deterministic credit for numeric-result criteria (SKIP for holistic grading):
            # If the criterion states an expected GBP amount and the evidence contains a clear calculation
            # that evaluates to that amount, award FULL marks (even if the LLM awarded 0 or partial).
            # Kept narrow + only when evidence exists in the student answer to avoid false positives.
            if marks_awarded < max_possible and max_possible > 0 and evid_list and student_blob_norm and not self._holistic_grading:
                if _evidence_present(evid_list):
                    award, suffix = _award_if_calc_matches_expected(str(criterion), evid_list, max_possible)
                    if award:
                        marks_awarded = float(max_possible)
                        existing = (item.get("reason", "") or "").strip()
                        saved_reason = existing
                        if saved_reason:
                            saved_reason = f"{saved_reason} {suffix}".strip()
                        else:
                            saved_reason = suffix
                        item["reason"] = saved_reason

            # Optional partial credit (disabled by default).
            # This is intended for cases where the student is clearly addressing the criterion
            # but the full correct treatment isn't provided.
            if (
                partial_credit_enabled
                and marks_awarded == 0
                and max_possible >= 0.5
                and evid_list
                and student_blob_norm
                and _evidence_present(evid_list)
            ):
                if _partial_overlap_ok(str(criterion), evid_list):
                    partial = _quantize_to_quarter(max_possible / 2)
                    partial = max(0.25, min(float(max_possible), float(partial)))
                    # Only award partial if it is strictly less than full marks.
                    if 0 < partial < float(max_possible):
                        marks_awarded = float(partial)
                        existing = (item.get("reason", "") or "").strip()
                        add = f"Partial credit awarded by heuristic ({partial}/{max_possible})."
                        item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # Lightweight numeric consistency guard:
            # If the criterion is an atomic numeric criterion, require evidence to contain those numbers too.
            # This is intentionally narrow to avoid revoking narrative marks that mention dates/percentages.
            # Skip this guard when the LLM already gave partial credit - it's signalling an "own figure"
            # scenario where the student used the correct method but got a different number.
            if marks_awarded > 0 and marks_awarded >= max_possible:
                if self._requires_strict_number_match(str(criterion)):
                    crit_nums = self._numbers_in_text(str(criterion))
                    if crit_nums:
                        ev_blob = " ".join(evid_list)
                        # Accept either:
                        # 1) all literal numbers in the criterion appear in evidence, OR
                        # 2) evidence contains the derived result of a simple bracketed calc
                        literal_ok = all(self._contains_number_variant(ev_blob, n) for n in crit_nums)
                        expected_val = self._compute_simple_calc_from_criterion(str(criterion))
                        expected_ok = False
                        if expected_val is not None:
                            for variant in self._format_expected_number_variants(expected_val):
                                if self._contains_number_variant(ev_blob, variant):
                                    expected_ok = True
                                    break
                            # If the student shows the working (e.g., 7200000*9/12) but not the final
                            # number, accept it if the calculation evaluates to the expected value.
                            if not expected_ok and _evidence_has_calc_result(ev_blob, expected_val):
                                expected_ok = True

                        # Also accept: if the criterion explicitly states "= X" (the direct answer),
                        # check that stated answer against evidence. This handles criteria where
                        # _compute_simple_calc returns None or a unit-mismatch value (e.g., £k vs £).
                        # Example: "NCI column = 1,350 (25% × £5,400k × 9/12)" - the "= 1,350" is
                        # the canonical answer; evidence "1350" should pass even if the bracketed
                        # calc computes to a different unit scale.
                        if not (literal_ok or expected_ok):
                            _eq_match = re.search(r"=\s*\(?([\d,]+)\)?", str(criterion))
                            if _eq_match:
                                _stated_ans = _eq_match.group(1)
                                if self._contains_number_variant(ev_blob, _stated_ans):
                                    expected_ok = True

                        # If LLM explicitly identified this as own-figure (OF), skip number
                        # mismatch revocation. In UK professional exams, OF for CALC criteria
                        # awards FULL marks - the student is not penalised twice for one wrong
                        # input. The numbers in evidence will differ from the criterion by design.
                        _reason_text_nm = str(item.get("reason", "")).lower()
                        _is_of_nm = bool(re.search(
                            r"\bof\b|\bown.?figure\b|\bown.?fig\b", _reason_text_nm
                        ))
                        # Rubric-driven OF signal: any criterion with `of_source_ids`
                        # populated (i.e. downstream of an upstream OF) is a candidate for
                        # OF bypass - a number mismatch here can be a legitimate carry from
                        # the student's own wrong upstream value.
                        if not _is_of_nm and self._of_source_ids_by_criterion_last_run.get(str(criterion)):
                            _is_of_nm = True
                        # exact_match criteria (e.g. SOCIE financial-statement rows) must show
                        # the rubric's expected number - OF bypass is not allowed because the
                        # amount itself is the assessable element, not a downstream carry-forward.
                        if _is_of_nm and criterion in self._exact_match_criteria_last_run:
                            _is_of_nm = False

                        if not (literal_ok or expected_ok) and not _is_of_nm:
                            evidence_warnings.append(f"Marks revoked (number mismatch): {criterion}")
                            marks_awarded = 0.0

            # Journal-direction guard: if criterion starts with Dr/Cr, evidence must include that direction.
            if marks_awarded > 0 and isinstance(criterion, str):
                crit_clean = criterion.strip().lower()
                if crit_clean.startswith("dr ") or crit_clean.startswith("cr "):
                    ev_blob = " ".join(evid_list).lower()
                    required = "dr" if crit_clean.startswith("dr ") else "cr"
                    # Use word-boundary regex so "Dr.", "DR", "dr" all match
                    if not re.search(rf"\b{re.escape(required)}\b", ev_blob):
                        evidence_warnings.append(f"Marks revoked (missing {required.upper()} in evidence): {criterion}")
                        marks_awarded = 0.0

                    # Also require the journal amount to be present when the criterion includes a number.
                    # BUT: if the LLM already gave partial credit (marks < max), it likely recognised
                    # an "own figure" scenario (correct journal structure, wrong amount). Don't override that.
                    # If the LLM gave FULL marks but the amount is wrong, award 50% OF instead of
                    # revoking to 0 - the student demonstrated correct journal structure (OF mark).
                    crit_nums = self._numbers_in_text(criterion)
                    if marks_awarded > 0 and marks_awarded >= max_possible and crit_nums:
                        if not any(self._contains_number_variant(" ".join(evid_list), n) for n in crit_nums):
                            _ev_lower = " ".join(evid_list).lower()
                            _crit_lower2 = str(criterion).strip().lower()
                            _direction_present = (
                                (_crit_lower2.startswith("dr ") and re.search(r"\bdr\b", _ev_lower)) or
                                (_crit_lower2.startswith("cr ") and re.search(r"\bcr\b", _ev_lower))
                            )
                            if _direction_present:
                                _of_val = max(0.25, round(float(max_possible) * 0.5 / 0.25) * 0.25)
                                _of_val = min(_of_val, float(max_possible) - 0.25)
                                marks_awarded = _of_val
                                evidence_warnings.append(
                                    f"OF mark (correct journal direction, own figure amount): {criterion}"
                                )
                            else:
                                evidence_warnings.append(
                                    f"Marks revoked (missing journal amount in evidence): {criterion}"
                                )
                                marks_awarded = 0.0

            # Journal-line recovery: if a Dr/Cr criterion got 0 but a matching journal line exists in the student text,
            # attach that verbatim line as evidence and award full marks.
            if marks_awarded == 0 and max_possible > 0 and student_blob_norm and isinstance(criterion, str):
                crit_clean = criterion.strip().lower()
                if crit_clean.startswith("dr ") or crit_clean.startswith("cr "):
                    snippet = _find_journal_line_in_student(criterion)
                    if snippet:
                        evid_list = [snippet] + (evid_list or [])
                        marks_awarded = float(max_possible)
                        existing = (item.get("reason", "") or "").strip()
                        add = "Awarded by journal-line recovery (matched Dr/Cr + amount in student answer)."
                        item["reason"] = (f"{existing} {add}".strip() if existing else add)

            # If we revoked marks in post-processing, ensure the saved reason doesn't contradict the score.
            saved_reason = item.get("reason", "")
            if original_marks_awarded > 0 and marks_awarded == 0:
                if saved_reason:
                    saved_reason = f"Marks revoked by guardrails: {saved_reason}"
                else:
                    saved_reason = "Marks revoked by guardrails"

            # Deduplicate evidence_list: the LLM sometimes produces both a plain-text
            # and a pipe-table version of the same quote.  Keep the more descriptive
            # form (longer after normalisation) and drop near-duplicates.
            if evid_list:
                seen_ev_norm: dict[str, str] = {}
                deduped_ev: list[str] = []
                for _ev in evid_list:
                    if not isinstance(_ev, str):
                        continue
                    _ev_norm = re.sub(r"[|\s]+", " ", _ev).strip().lower()
                    if not _ev_norm:
                        continue
                    if _ev_norm not in seen_ev_norm:
                        seen_ev_norm[_ev_norm] = _ev
                        deduped_ev.append(_ev)
                evid_list = deduped_ev

            sum_awarded_calc += marks_awarded

            # Detect "own figure" (OF) scenarios: LLM gave partial credit on a
            # calc/journal criterion, or journal guard downgraded to partial.
            # This flag is surfaced in annotations so students see "OF" clearly.
            # Use rubric category cache - LLM output items never include category.
            _item_category = self._criterion_category_map_last_run.get(
                criterion, str(item.get("category", "") or "")
            ).lower()
            _is_of_mark = (
                0 < float(marks_awarded) < float(max_possible)
                and _item_category in ("calculation", "journal", "calc")
                and not self._holistic_grading
            )

            bd_item = {
                "criterion": criterion,
                "marks_awarded": marks_awarded,
                "max_possible": max_possible,
                "reason": saved_reason,
                "evidence": "; ".join(evid_list) if evid_list else "",
                "evidence_list": evid_list if evid_list else [],
                "comments_summary": item.get("comments_summary", ""),
                "is_of_mark": _is_of_mark,
            }
            # Carry the rubric's own id onto the saved entry. Without it the
            # graded document gives no way to tell which criterion an entry is,
            # short of diffing 600 characters of description - and no way to
            # tell whether the model returned an id at all.
            _rubric_cid = ""
            for _cid_k, _cid_desc in (_id_map or {}).items():
                if _cid_desc == criterion:
                    _rubric_cid = _cid_k
                    break
            if _rubric_cid:
                bd_item["criterion_id"] = _rubric_cid
            bd_item["_llm_returned_criterion_id"] = str(
                item.get("criterion_id", "") or "").strip().upper()
            # The model's own one-line statement of what it just graded, used
            # below to verify it paired its reasoning with the right id.
            _focus = str(item.get("criterion_focus", "") or "").strip()
            if _focus:
                bd_item["criterion_focus"] = _focus
            # Lines where the student repeats this point for no extra marks.
            # Turned into "Marks given above/below" notes further down.
            _restated = item.get("restated_at")
            if isinstance(_restated, list):
                _clean = [str(r).strip() for r in _restated if str(r or "").strip()]
                if _clean:
                    bd_item["_restated_at"] = _clean[:4]
            # LLM-provided column header for tabular disambiguation. Only
            # forwarded when the LLM populated it (non-empty string). The
            # annotator uses this to disambiguate values that appear in
            # multiple columns of a table row (e.g. `share cap | 250,000 |
            # 250,000 | 0` — "acq date" vs "disposal date").
            _col_hdr = item.get("column_header")
            if _col_hdr is not None:
                _col_hdr_str = str(_col_hdr).strip()
                if _col_hdr_str:
                    bd_item["_column_header"] = _col_hdr_str
            # RUBRIC COLUMN WINS when the mark scheme declares one. The LLM's
            # hint is a guess at what the student's table looks like; `column`
            # is the author's statement of which column the mark is FOR. Without
            # it the annotator takes the first matching number on the row, so a
            # revaluation-loss mark for OTHER COMPONENTS underlined the (300) in
            # retained earnings and put its score beside the (600) total.
            _rubric_col = (self._column_by_criterion_last_run or {}).get(
                str(bd_item.get("criterion", "") or "")
            )
            if _rubric_col:
                _hdr = _COLUMN_HEADER_TEXT.get(
                    _canon_column_name(_rubric_col), ""
                )
                if _hdr:
                    bd_item["_column_header"] = _hdr
            # Populate _target_value from the rubric's of_value / of_produces
            # so the annotator can narrow the underline+score to the specific
            # value cell on the resolved evidence line. Model-side info — no
            # unit guessing in the annotator.
            #
            # Priority order (first non-None wins):
            #   1. of_value.value — the raw target value for an origin
            #      criterion (e.g. share cap 250,000 or NCI at acq 3,125,000).
            #   2. of_produces — the sub-working result value for a component
            #      criterion in an aggregate group (e.g. 250, 12,750, 5,400
            #      for OF12 components).
            # For non-OF criteria (no of_value, no of_produces), the target
            # stays unset and the annotator falls back to whole-rect underline
            # (existing behaviour).
            _rubric_target: Optional[int] = None
            _val_meta = (self._of_value_by_criterion_last_run or {}).get(criterion)
            if isinstance(_val_meta, dict):
                _raw_val = _val_meta.get("value")
                if _raw_val is not None:
                    try:
                        _rubric_target = int(round(float(_raw_val)))
                    except (TypeError, ValueError):
                        pass
            if _rubric_target is None:
                _prod_val = (self._of_produces_by_criterion_last_run or {}).get(criterion)
                if _prod_val is not None:
                    try:
                        _rubric_target = int(round(float(_prod_val)))
                    except (TypeError, ValueError):
                        pass
            if _rubric_target is not None and _rubric_target != 0:
                bd_item["_target_value"] = _rubric_target
                bd_item["_target_value_variants"] = sorted(
                    _value_variants_for_search(_rubric_target)
                )

            # For holistic grading, preserve sub-question metadata for the annotator.
            if self._holistic_grading:
                bd_item["_sub_question"] = item.get("_sub_question", "")
                bd_item["_student_label"] = item.get("_student_label", "")
                bd_item["_correct_points_with_marks"] = item.get("_correct_points_with_marks", [])
                bd_item["_not_required_points"] = item.get("_not_required_points", [])
            normalized_breakdown.append(bd_item)

        # "Marks given above / below" guard - SKIP for holistic grading.
        # When two criteria in the same grading run were both awarded marks and
        # their evidence strings are identical (after whitespace normalisation),
        # keep only the one with the higher max_possible and zero out the other.
        # This mirrors the teacher's annotation rule: marks are awarded once, at
        # the location where the actual working appears; subsequent references to
        # the same result (e.g., restating a goodwill figure in narrative or in
        # a journal after calculating it in a working) do NOT earn extra marks.
        # Exception: a journal entry criterion citing the same number as a calc
        # criterion is a DIFFERENT skill and keeps its marks - we only zero out
        # when the criteria descriptions themselves overlap significantly.
        if normalized_breakdown and not self._holistic_grading:
            try:
                # Build the index: normalised evidence key → list of criterion
                # indices that cite it (only those with awarded > 0).
                _ev_per_idx: dict[int, str] = {}
                _ev_index: dict[str, list[int]] = {}
                for _idx, _bd in enumerate(normalized_breakdown):
                    _ev_raw = _bd.get("evidence", "") or ""
                    _awarded = float(_bd.get("marks_awarded", 0) or 0)
                    if not _ev_raw or _awarded <= 0:
                        continue
                    _ev_key = re.sub(r"[\s;|]+", " ", _ev_raw).strip().lower()
                    if len(_ev_key) < 12:
                        continue
                    _ev_per_idx[_idx] = _ev_key
                    _ev_index.setdefault(_ev_key, []).append(_idx)

                # Substring-containment merge. The LLM sometimes cites a bare
                # working line "7200 x 9/12 x 0.25 = 1350" for one criterion
                # and the SAME line with a narrative prefix ("The profit
                # attributable to NCI for the year will be ..." + same calc)
                # for another. Strict equality misses that these are the same
                # student writing. If key A is fully contained inside key B
                # after normalisation, merge A's indices into B's group so the
                # co-crediting is detected. Cap length ratio at 4× so we don't
                # merge a short common substring into an unrelated long one.
                _keys_by_len = sorted(_ev_index.keys(), key=len)
                _absorbed: set[str] = set()
                for _k_short in _keys_by_len:
                    if _k_short in _absorbed:
                        continue
                    for _k_long in _keys_by_len:
                        if _k_long is _k_short or _k_long in _absorbed:
                            continue
                        if len(_k_long) <= len(_k_short):
                            continue
                        if len(_k_long) > len(_k_short) * 4:
                            continue
                        if _k_short in _k_long:
                            _ev_index[_k_long].extend(_ev_index[_k_short])
                            _absorbed.add(_k_short)
                            break
                for _k in _absorbed:
                    _ev_index.pop(_k, None)

                def _has_component(i: int) -> bool:
                    _crit = str(normalized_breakdown[i].get("criterion", "") or "")
                    return bool(self._of_component_of_by_criterion_last_run.get(_crit, ""))

                # Where each piece of evidence sits in the student's script, so
                # the annotation can point the reader the right way. A marker
                # writes "Marks given BELOW" against an earlier mention whose
                # marks are awarded further down (the student states the
                # goodwill figure in prose, then shows the working underneath),
                # and "Marks given ABOVE" when the credit came first.
                _student_hay = _norm_for_evidence_match(
                    getattr(self, "_student_text_last_run", "") or ""
                )

                def _evidence_pos(i: int) -> int:
                    _ev = _norm_for_evidence_match(_ev_per_idx.get(i, "") or "")
                    if not _ev or not _student_hay:
                        return -1
                    return _student_hay.find(_ev)

                def _direction_word(keeper_i: int, dup_i: int) -> str:
                    _kp, _dp = _evidence_pos(keeper_i), _evidence_pos(dup_i)
                    # Fall back to "above" when either side cannot be located -
                    # that is the historical wording and the commoner case.
                    if _kp < 0 or _dp < 0 or _kp == _dp:
                        return "above"
                    return "below" if _kp > _dp else "above"

                for _ev_key, _idxs in _ev_index.items():
                    # De-dupe while preserving order (a criterion could land
                    # in both its own group and the absorbed-into group).
                    _seen: set[int] = set()
                    _idxs = [i for i in _idxs if not (i in _seen or _seen.add(i))]
                    if len(_idxs) < 2:
                        continue
                    # Keeper selection. Priority order:
                    #   1. Aggregate sub-mark (of_component_of set) — student's
                    #      granular calc lines are what the mark scheme prizes;
                    #      a wrapper narrative criterion that cites the same
                    #      line via containment loses to the sub-marks.
                    #   2. Highest max_possible (existing rule for the case
                    #      where nothing has of_component_of).
                    #   3. Highest marks_awarded (tie-break).
                    _idxs_sorted = sorted(
                        _idxs,
                        key=lambda i: (
                            1 if _has_component(i) else 0,
                            float(normalized_breakdown[i].get("max_possible", 0) or 0),
                            float(normalized_breakdown[i].get("marks_awarded", 0) or 0),
                        ),
                        reverse=True,
                    )
                    _keeper_crit = str(normalized_breakdown[_idxs_sorted[0]].get("criterion", "") or "")
                    _keeper_component = self._of_component_of_by_criterion_last_run.get(_keeper_crit, "")
                    _keeper_ev = _ev_per_idx.get(_idxs_sorted[0], "")
                    for _dup_idx in _idxs_sorted[1:]:
                        _dup_crit = str(normalized_breakdown[_dup_idx].get("criterion", "") or "").lower()
                        _keep_crit_lower = _keeper_crit.lower()
                        _dup_ev = _ev_per_idx.get(_dup_idx, "")
                        # Was this pair grouped via SUBSTRING CONTAINMENT rather
                        # than exact evidence equality? Containment is a stronger
                        # signal than word overlap — the two criteria demonstrably
                        # cite the SAME student writing, one just with extra
                        # narrative wrapping. When true we skip the word-overlap
                        # gate that guards the exact-equality path.
                        _via_containment = (
                            bool(_dup_ev) and bool(_keeper_ev)
                            and _dup_ev != _keeper_ev
                            and (_dup_ev in _keeper_ev or _keeper_ev in _dup_ev)
                        )
                        # Context guard: aggregate sub-marks (of_component_of set)
                        # never cross-null each other.
                        #   • DIFFERENT aggregates → different working contexts
                        #     testing the same VALUE at different POINTS
                        #     (e.g., share cap 250 at acq date AND at disposal
                        #     date). Both keep credit.
                        #   • SAME aggregate → different sub-marks of the same
                        #     rolled-up figure, which are DISTINCT sub-workings
                        #     by construction (each has its own of_produces
                        #     value like 250, 12,750, 5,400 for OF12). The
                        #     student legitimately co-cites them on the one
                        #     aggregated line the mark scheme expects. Both
                        #     keep credit.
                        # If only one side is an aggregate sub-mark (or
                        # neither), fall through to the substring / word-overlap
                        # checks below.
                        _dup_component = self._of_component_of_by_criterion_last_run.get(
                            str(normalized_breakdown[_dup_idx].get("criterion", "") or ""),
                            "",
                        )
                        if _keeper_component and _dup_component:
                            continue  # both aggregate sub-marks — never cross-null
                        # ONE SIDE IS A BUCKET SUB-MARK, THE OTHER IS NOT.
                        # These live in different structures: a sub-mark of a
                        # rolled-up working total versus a standalone narrative
                        # or calculation point. The rubric says so in as many
                        # words - "do NOT treat it as the same point as the
                        # narrative statement ... which is a separate 0.5
                        # criterion" - and this guard used to fall through and
                        # null the standalone one anyway. On Amber that cost
                        # 1.5 marks across MM09, NYW05 and NYW12, and printed
                        # "Marks given above" on the very line that was being
                        # awarded 0.25.
                        if bool(_keeper_component) != bool(_dup_component):
                            continue
                        # DIFFERENT COLUMNS OF THE SAME TABLE ROW.
                        # A SOCIE row carries one figure per column and the
                        # rubric marks several of them separately. Both items
                        # quote the whole row, so they look identical here -
                        # only the resolved column tells them apart.
                        _keep_col = str(normalized_breakdown[_idxs_sorted[0]].get(
                            "_column_header", "") or "").strip().lower()
                        _dup_col = str(normalized_breakdown[_dup_idx].get(
                            "_column_header", "") or "").strip().lower()
                        if _keep_col and _dup_col and _keep_col != _dup_col:
                            continue
                        _dup_crit_raw = str(
                            normalized_breakdown[_dup_idx].get("criterion", "") or ""
                        )
                        # PROSE EVIDENCE CARRIES MORE THAN ONE POINT.
                        # What the two criteria share decides this, not what
                        # they are. A sentence of prose can legitimately make
                        # two different points at once — "treated as an
                        # associate from 1 March 20X4" is both the
                        # becomes-an-associate mark AND the equity-accounting
                        # mark, and revoking one of them cost a student 0.5.
                        # A CALCULATION LINE cannot: the student performed one
                        # arithmetic step, so a narrative criterion citing that
                        # same line is restating the calculation rather than
                        # making its own point. Crediting both there awarded
                        # 1.5 marks a marker would not give — the narrative
                        # "MM contributes £5.4m to profit" and the 1-mark "NCI
                        # is allocated 25%" both scored on working lines the
                        # student never wrote a statement about.
                        # Digit density separates the two cleanly: the prose
                        # cases sit at 2-12%, the working and journal lines at
                        # 37-74%.
                        if _evidence_is_prose(_keeper_ev or _dup_ev):
                            continue
                        # JOURNAL vs WORKING never cross-null. A journal
                        # legitimately re-presents figures derived in a working;
                        # a marker credits the working line AND the journal
                        # entry. Only one side being category 'journal' means
                        # the two are testing different things.
                        _keeper_cat = (self._criterion_category_map_last_run.get(
                            _keeper_crit, "") or "").lower()
                        _dup_cat = (self._criterion_category_map_last_run.get(
                            _dup_crit_raw, "") or "").lower()
                        if (_keeper_cat == "journal") != (_dup_cat == "journal"):
                            continue
                        # Zero out when EITHER:
                        #   (a) the pair was grouped via substring containment
                        #       (dup's evidence sits inside keeper's or vice
                        #       versa) — same student writing, direct signal, no
                        #       word-overlap gate needed;
                        #   (b) the duplicate criterion shares 3+ significant
                        #       words with the keeper (existing exact-evidence
                        #       path — protects against random collisions).
                        _dup_words = set(re.findall(r"[a-z]{4,}", _dup_crit))
                        _keep_words = set(re.findall(r"[a-z]{4,}", _keep_crit_lower))
                        _overlap = _dup_words & _keep_words
                        _stop = {"mark", "marks", "amount", "value", "total", "year", "each", "from", "with", "that", "this", "have", "been"}
                        _overlap -= _stop
                        if _via_containment or len(_overlap) >= 2:
                            _dup_marks = float(normalized_breakdown[_dup_idx].get("marks_awarded", 0) or 0)
                            if _dup_marks > 0:
                                normalized_breakdown[_dup_idx]["marks_awarded"] = 0.0
                                sum_awarded_calc -= _dup_marks
                                _prev_reason = (normalized_breakdown[_dup_idx].get("reason", "") or "").strip()
                                _dir = _direction_word(_idxs_sorted[0], _dup_idx)
                                # A marker writes "Marks given above/below" to
                                # point AWAY from the line - "you made this
                                # point here too, the marks are over there".
                                # When both criteria resolved to the SAME line
                                # there is nowhere to point, and the annotator
                                # draws the note on top of the score already
                                # printed there. Zero it, but say so plainly
                                # rather than printing a pointer to itself.
                                _same_line = (
                                    _evidence_pos(_idxs_sorted[0]) ==
                                    _evidence_pos(_dup_idx)
                                )
                                if _same_line:
                                    normalized_breakdown[_dup_idx]["reason"] = (
                                        f"Credited once: this line is already marked for "
                                        f"'{_keeper_crit[:60]}'. " + _prev_reason
                                    ).strip()
                                else:
                                    normalized_breakdown[_dup_idx]["reason"] = (
                                        f"Marks given {_dir}: same evidence already credited for "
                                        f"'{_keeper_crit[:60]}'. " + _prev_reason
                                    ).strip()
            except Exception:
                pass  # Guard must never fail the grader

        # ── Cross-context un-revoke: restore LLM's "Marks given above" zeros ──
        # The LLM sometimes emits "Marks given above" (marks_awarded = 0) on a
        # criterion whose value legitimately appears at a DIFFERENT working
        # context from the criterion it references (e.g., share cap 250 tested
        # at both acquisition date AND disposal date - the rubric wants both
        # credited). If the criterion has an `of_component_of` group and the
        # LLM's cited-first criterion has a DIFFERENT `of_component_of`, treat
        # the "Marks given above" as an over-revocation and restore full marks.
        if normalized_breakdown and not self._holistic_grading:
            try:
                _crit_to_idx: dict[str, int] = {
                    str(bd.get("criterion", "") or ""): i
                    for i, bd in enumerate(normalized_breakdown)
                }
                _restored_total = 0.0
                for _idx, _bd in enumerate(normalized_breakdown):
                    try:
                        _awarded = float(_bd.get("marks_awarded", 0) or 0)
                        _maxp = float(_bd.get("max_possible", 0) or 0)
                    except (TypeError, ValueError):
                        continue
                    if _awarded > 0 or _maxp <= 0:
                        continue
                    _reason = str(_bd.get("reason", "") or "")
                    if "Marks given above" not in _reason and "Marks given below" not in _reason:
                        continue
                    _self_crit = str(_bd.get("criterion", "") or "")
                    _self_comp = self._of_component_of_by_criterion_last_run.get(_self_crit, "")
                    if not _self_comp:
                        continue  # no OF context to compare - leave alone
                    # Try to identify the sibling the LLM referenced. Its
                    # description prefix is embedded in the reason like
                    # "credited for 'Share capital (500,000 x 50p) = 250'".
                    _ref_match = re.search(r"credited for ['\"]([^'\"]{5,80})", _reason)
                    if not _ref_match:
                        continue
                    _ref_prefix = _ref_match.group(1).strip().lower()
                    _sib_comp = ""
                    for _c, _i in _crit_to_idx.items():
                        if _c.lower().startswith(_ref_prefix[:40]):
                            _sib_comp = self._of_component_of_by_criterion_last_run.get(_c, "")
                            break
                    if not _sib_comp or _sib_comp == _self_comp:
                        continue  # same context (or unknown) - leave the zero
                    # Different working contexts: restore this criterion's marks.
                    _bd["marks_awarded"] = _maxp
                    _restored_total += _maxp
                    _bd["reason"] = (
                        f"Restored (cross-context): value appears in a different "
                        f"working section ({_self_comp}) than the referenced "
                        f"sibling ({_sib_comp}); both criteria legitimately earn "
                        f"marks. " + _reason
                    ).strip()
                sum_awarded_calc += _restored_total
            except Exception:
                pass  # Guard must never fail the grader

        # ── Parent-calculation verification guard (numerical mode only) ──────
        # Enforces context-aware crediting: for each criterion that references
        # a specific parent working (e.g. "From the working '25% × £7.2m × 9/12
        # = 1,350'"), the student's evidence must contain that parent working's
        # RESULT (here 1,350). If not, the student did not perform this specific
        # calculation - award 0 regardless of which individual numbers appear.
        # This mirrors how a teacher marks: they ask "did the student actually
        # DO this working?", not "do any of these numbers appear somewhere?".
        # Criteria without a quoted parent-working reference are not touched -
        # their credit stands on the LLM's original judgement.
        if normalized_breakdown and not self._holistic_grading:
            try:
                _revoked = _apply_parent_calc_verification(
                    normalized_breakdown,
                    of_source_ids_map=self._of_source_ids_by_criterion_last_run,
                    of_component_of_map=self._of_component_of_by_criterion_last_run,
                    of_definitions=self._of_definitions_last_run,
                )
                sum_awarded_calc -= _revoked
            except Exception:
                pass  # Guard must never fail the grader

        # ── Aggregate-value recovery guard (numerical mode only) ─────────────
        # When a student uses an equivalent-method shortcut (rolls several
        # rubric sub-components into one aggregated figure) they compute the
        # right answer via a different decomposition. The dedup guard above
        # can strip credits from the "absorbed" sub-marks because their evidence
        # duplicates a sibling's. Real markers still reward the working when
        # the aggregate total is correct.
        #
        # Trigger - ALL must hold, otherwise no recovery for that OF group:
        #   (i)   The rubric has criteria tagged `of_component_of: OFX` (i.e.,
        #         producers of some aggregate OFX).
        #   (ii)  A downstream consumer criterion (has `of_source_ids: [OFX]`)
        #         is fully credited (marks_awarded == max_possible) AND its
        #         evidence contains OFX's value from `of_definitions` or from
        #         the origin's `of_value`.
        #   (iii) At least ONE producer sub-mark in the group is directly
        #         earned (proves method knowledge, not a lucky number).
        #
        # When triggered: award +0.25 recovery to each un-earned producer
        # sub-mark in the group, capped at a total of 0.5 marks per group.
        if normalized_breakdown and not self._holistic_grading:
            try:
                _recovered = _apply_aggregate_value_recovery(
                    normalized_breakdown,
                    of_component_of_map=self._of_component_of_by_criterion_last_run,
                    of_source_ids_map=self._of_source_ids_by_criterion_last_run,
                    of_value_map=self._of_value_by_criterion_last_run,
                    of_definitions=self._of_definitions_last_run,
                    of_produces_map=self._of_produces_by_criterion_last_run,
                    student_text=self._student_text_last_run or "",
                    keywords_map=self._keywords_by_criterion_last_run,
                )
                sum_awarded_calc += _recovered
            except Exception:
                pass  # Guard must never fail the grader

        # ── Sign / column / self-consistency guards (numerical mode only) ────
        # These read rubric fields that were authored from the start but never
        # enforced: `expected_amount` + `sign_sensitive` separate two criteria
        # quoting the same figure in opposite roles, and `column` pins a
        # statement row to the column the mark is actually for. Run AFTER
        # recovery so a recovered mark is held to the same standard as a direct
        # one. Each is best-effort and independently guarded.
        if normalized_breakdown and not self._holistic_grading:
            for _guard_name, _guard in (
                (
                    "sign",
                    lambda: _apply_sign_verification(
                        normalized_breakdown,
                        expected_amount_map=self._expected_amount_by_criterion_last_run,
                        sign_sensitive_criteria=self._sign_sensitive_criteria_last_run,
                    ),
                ),
                (
                    "column",
                    lambda: _apply_column_verification(
                        normalized_breakdown,
                        column_map=self._column_by_criterion_last_run,
                        expected_amount_map=self._expected_amount_by_criterion_last_run,
                        student_text=self._student_text_last_run or "",
                    ),
                ),
                (
                    "focus-mismatch",
                    lambda: (
                        evidence_warnings.extend(
                            _detect_criterion_focus_mismatch(
                                normalized_breakdown,
                                criterion_id_by_desc={
                                    d: c for c, d in
                                    (self._criterion_id_map_last_run or {}).items()
                                },
                                of_component_of_map=self._of_component_of_by_criterion_last_run,
                                of_ids_map=self._of_ids_by_criterion_last_run,
                                of_source_ids_map=self._of_source_ids_by_criterion_last_run,
                            )
                        ) or 0.0
                    ),
                ),
                (
                    "section-locality",
                    lambda: _apply_section_locality_guard(
                        normalized_breakdown,
                        student_text=self._student_text_last_run or "",
                        criterion_id_by_desc={
                            d: c for c, d in
                            (self._criterion_id_map_last_run or {}).items()
                        },
                        of_component_of_map=self._of_component_of_by_criterion_last_run,
                        of_ids_map=self._of_ids_by_criterion_last_run,
                        of_source_ids_map=self._of_source_ids_by_criterion_last_run,
                    ),
                ),
                ("contradiction", lambda: _reject_contradictory_awards(
                    normalized_breakdown
                )),
            ):
                try:
                    sum_awarded_calc -= _guard()
                except Exception:
                    logger.debug(f"{_guard_name} guard skipped", exc_info=True)

        # Broad-criterion gating (generic) - SKIP for holistic grading:
        # If a high-mark narrative criterion sits next to many micro-criteria (<= 0.5 each),
        # don't award the broad marks unless the student scores well on the micro-criteria.
        # This prevents over-awarding for broad statements when the detailed workings are wrong
        # (common in EPS / table-style sections) while still allowing broad criteria in sections
        # that have no micro breakdown.
        try:
            pos_map = self._rubric_position_last_run or {}
            if pos_map and normalized_breakdown and not self._holistic_grading:
                # position → indices (handle duplicates conservatively)
                pos_to_indices: dict[int, list[int]] = {}
                for idx, bi in enumerate(normalized_breakdown):
                    crit = bi.get("criterion")
                    if not isinstance(crit, str):
                        continue
                    p = pos_map.get(crit)
                    if isinstance(p, int):
                        pos_to_indices.setdefault(p, []).append(idx)

                def _is_broad_narrative(crit: str, maxp: float) -> bool:
                    if not isinstance(crit, str) or not crit.strip():
                        return False
                    if float(maxp) < 1.0:
                        return False
                    # Only gate medium-sized broad criteria (EPS-style, 1–2 marks).
                    # Do not gate large statement criteria (e.g., SOCIE totals 4 marks).
                    if float(maxp) > 2.0:
                        return False
                    c = crit.strip().lower()
                    if c.startswith("dr ") or c.startswith("cr "):
                        return False
                    if re.search(r"\d", crit):
                        return False
                    return True

                def _is_micro(maxp: float) -> bool:
                    try:
                        return 0 < float(maxp) <= 0.5
                    except Exception:
                        return False

                window = 18
                micro_success_threshold = float(os.getenv("BROAD_CRITERION_MICRO_SUCCESS_THRESHOLD", "0.8") or 0.8)
                for idx, bi in enumerate(normalized_breakdown):
                    marks = float(bi.get("marks_awarded") or 0.0)
                    maxp = float(bi.get("max_possible") or 0.0)
                    crit = bi.get("criterion")
                    if marks <= 0 or not isinstance(crit, str):
                        continue
                    if not _is_broad_narrative(crit, maxp):
                        continue

                    p = pos_map.get(crit)
                    if not isinstance(p, int):
                        continue

                    micro_max_sum = 0.0
                    micro_awarded_sum = 0.0
                    for q in range(p - window, p + window + 1):
                        if q == p:
                            continue
                        for j in pos_to_indices.get(q, []):
                            bj = normalized_breakdown[j]
                            maxj = float(bj.get("max_possible") or 0.0)
                            if not _is_micro(maxj):
                                continue
                            micro_max_sum += maxj
                            micro_awarded_sum += float(bj.get("marks_awarded") or 0.0)

                    # Only gate if there is meaningful micro coverage nearby.
                    if micro_max_sum <= 0:
                        continue
                    if micro_max_sum < maxp * 0.75:
                        continue

                    ratio = (micro_awarded_sum / micro_max_sum) if micro_max_sum > 0 else 0.0
                    if ratio < micro_success_threshold:
                        # Revoke broad marks (but do not change micro marks).
                        revoked = marks
                        bi["marks_awarded"] = 0.0
                        sum_awarded_calc -= revoked
                        existing = (bi.get("reason") or "").strip()
                        add = (
                            f"Marks revoked by guardrails: broad criterion gated by micro-criteria performance "
                            f"({micro_awarded_sum:.2f}/{micro_max_sum:.2f} nearby micro marks)."
                        )
                        bi["reason"] = f"{existing} {add}".strip() if existing else add
                        evidence_warnings.append(f"Marks revoked (broad criterion gated by micro performance): {crit}")
        except Exception:
            pass

        # Post-grading dedup: when one awarded criterion's description fully
        # contains the text of 2+ other awarded criteria, it is a duplicate
        # "paragraph total" annotation from the marking guide.  Zero it out
        # to prevent double-counting.  This does NOT remove criteria from the
        # LLM prompt (which causes instability), only revokes marks afterward.
        for i, bi in enumerate(normalized_breakdown):
            if bi["marks_awarded"] <= 0:
                continue
            crit_i = _norm_for_evidence_match(bi.get("criterion", ""))
            if not crit_i or len(crit_i) < 40:
                continue
            contained_count = 0
            for j, bj in enumerate(normalized_breakdown):
                if i == j or bj["marks_awarded"] <= 0:
                    continue
                crit_j = _norm_for_evidence_match(bj.get("criterion", ""))
                if not crit_j or len(crit_j) >= len(crit_i) * 0.85:
                    continue
                if crit_j in crit_i:
                    contained_count += 1
            if contained_count >= 2:
                revoked = bi["marks_awarded"]
                sum_awarded_calc -= revoked
                bi["marks_awarded"] = 0.0
                bi["reason"] = f"Marks revoked (duplicate superset criterion): {bi['reason']}"
                evidence_warnings.append(
                    f"Marks revoked (superset criterion duplicates {contained_count} sub-criteria): {bi['criterion'][:80]}"
                )
                logger.info(f"Superset dedup: revoked {revoked} from '{bi['criterion'][:60]}...'")

        # Evidence-reuse dedup (OPTIONAL): Some earlier versions revoked marks when the same
        # evidence snippet was used across multiple micro-criteria. This is too aggressive for
        # official marking guides: a single table row can legitimately earn multiple 0.25/0.5
        # marks (eg a number and its supporting rate/working).
        #
        # Default is OFF. Enable explicitly with DEDUP_EVIDENCE_REUSE=1.
        if os.getenv("DEDUP_EVIDENCE_REUSE", "0").strip().lower() in {"1", "true", "yes", "y"}:
            _evidence_owner: dict[str, int] = {}  # norm_snippet → index of first awardee
            # Sort by marks descending so the highest-value award "claims" the evidence.
            _award_order = sorted(
                range(len(normalized_breakdown)),
                key=lambda idx: normalized_breakdown[idx]["marks_awarded"],
                reverse=True,
            )
            for idx in _award_order:
                bi = normalized_breakdown[idx]
                if bi["marks_awarded"] <= 0:
                    continue
                evid_list = bi.get("evidence_list") or []
                for ev in evid_list:
                    if not isinstance(ev, str):
                        continue
                    ev_key = _norm_for_evidence_match(ev)
                    if len(ev_key) < 20:
                        continue
                    if ev_key in _evidence_owner:
                        first_idx = _evidence_owner[ev_key]
                        if first_idx != idx:
                            revoked = bi["marks_awarded"]
                            sum_awarded_calc -= revoked
                            bi["marks_awarded"] = 0.0
                            first_crit = normalized_breakdown[first_idx]["criterion"][:60]
                            bi["reason"] = (
                                f"Marks revoked (evidence already used for '{first_crit}'): "
                                f"{bi['reason']}"
                            )
                            evidence_warnings.append(
                                f"Marks revoked (duplicate evidence reuse): "
                                f"{bi['criterion'][:80]}"
                            )
                            logger.info(
                                f"Evidence dedup: revoked {revoked} from "
                                f"'{bi['criterion'][:60]}' (evidence claimed by "
                                f"'{first_crit}')"
                            )
                            break  # already revoked this criterion, move on
                    else:
                        _evidence_owner[ev_key] = idx

        # Priority for total_max: the model-answer doc's `max_marks` (canonical
        # question total per the marker, e.g. 28 for Bauhaus Q1) wins over any
        # computed sum-of-rubric-criteria. Fall back to the questions_data /
        # LLM-reported total via _extract_question_max_marks only if the model
        # doc doesn't carry a max_marks field. Model data was stashed on self
        # by _run_grading - accessing it here avoids threading it through the
        # method signature.
        total_max: float = 0.0
        try:
            _model_data = getattr(self, "_model_data_last_run", None)
            _mdoc_max = None
            if isinstance(_model_data, dict):
                # Prefer max_marks; fall back to total_marks then available_marks.
                for _key in ("max_marks", "total_marks", "available_marks"):
                    raw = _model_data.get(_key)
                    if raw is None:
                        continue
                    _nums = [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(raw))]
                    if _nums:
                        _mdoc_max = max(n for n in _nums if n > 0) if any(n > 0 for n in _nums) else None
                        if _mdoc_max:
                            logger.info(
                                f"Using model_answer.{_key} for Q{self.question_number} "
                                f"max: {_mdoc_max}"
                            )
                            break
            if _mdoc_max and _mdoc_max > 0:
                total_max = float(_mdoc_max)
        except Exception:
            total_max = 0.0
        if total_max <= 0:
            total_max = self._extract_question_max_marks(questions_data, main_grade)

        # Use the post-processed breakdown sum (after any revocations) as the source of truth.
        # Round to nearest 0.5 (standard mathematical rounding, half-up) to match marking
        # convention (only .5 or whole marks allowed).  Previous ceiling rounding systematically
        # inflated scores; round-to-nearest is fairer.
        rounded_total = math.floor(sum_awarded_calc * 2 + 0.5) / 2
        if total_max > 0:
            rounded_total = min(rounded_total, total_max)

        # ── Rubric completeness ─────────────────────────────────────────────
        # Every scoreable rubric criterion must appear in the saved breakdown,
        # even when the grading model never returned it. Without this an
        # omission is SILENT: the criterion simply vanishes from the CSV and
        # the annotated PDF, so a student can lose a mark for work they plainly
        # did and nobody can see why it happened. Observed repeatedly on
        # "Fair value adjustment (land) = 400", where the student wrote
        # "Land fair value uplift 0.4m" and the criterion was never assessed.
        #
        # Appended at 0 marks with an explicit reason, so this NEVER changes a
        # score — total_max comes from the model answer's max_marks, not from
        # summing the breakdown. It only makes the gap visible for review.
        if not self._holistic_grading and self._allowed_criteria_last_run:
            try:
                _seen_crits: set[str] = set()
                for _bi in normalized_breakdown:
                    _c = _bi.get("criterion")
                    if isinstance(_c, str):
                        _seen_crits.add(_c)
                    # Components folded into a [Combined] entry are represented
                    # there and must not be re-added as missing.
                    for _comp in (_bi.get("components") or []):
                        _cd = _comp.get("criterion_description")
                        if isinstance(_cd, str):
                            _seen_crits.add(_cd)

                _cat_map = self._criterion_category_map_last_run or {}
                _max_map = self._criterion_max_map_last_run or {}
                _pos_map = self._rubric_position_last_run or {}
                _never_returned = [
                    _c for _c in self._allowed_criteria_last_run
                    if _c not in _seen_crits
                    and (_cat_map.get(_c, "") or "").lower() != "section_header"
                ]
                # Keep rubric order so the CSV reads in mark-scheme sequence.
                _never_returned.sort(key=lambda c: _pos_map.get(c, 10 ** 6))

                for _c in _never_returned:
                    normalized_breakdown.append({
                        "criterion": _c,
                        "marks_awarded": 0.0,
                        "max_possible": float(_max_map.get(_c, 0) or 0),
                        "reason": (
                            "NOT ASSESSED - the grading model did not return "
                            "this criterion, so no judgement was made on it "
                            "either way. Added at 0 for visibility; review "
                            "manually before releasing the mark."
                        ),
                        "evidence": "",
                        "evidence_list": [],
                        "comments_summary": "",
                        "_not_returned_by_model": True,
                    })

                if _never_returned:
                    _lost = sum(float(_max_map.get(_c, 0) or 0) for _c in _never_returned)
                    logger.warning(
                        f"Rubric completeness: {len(_never_returned)} of "
                        f"{len(self._allowed_criteria_last_run)} criteria were never "
                        f"returned by the grading model ({_lost:g} marks unassessed). "
                        f"Added at 0 marks for visibility:"
                    )
                    for _c in _never_returned[:20]:
                        logger.warning(f"  not-returned: '{_c[:100]}...'")
            except Exception:
                # Visibility aid only - must never break a grading run.
                logger.warning("Rubric completeness check failed", exc_info=True)

        # ── "Marks given above / below" notes ────────────────────────────────
        # A marker writes these beside a line where the student REPEATS a point
        # that was credited somewhere else, pointing the reader to where the
        # marks actually are. The breakdown carries exactly one entry per
        # criterion, and that entry's evidence is the credited line - so the
        # repeat has no slot of its own. We give it one here: a zero-mark,
        # zero-max entry anchored on the repeated line, which the annotator
        # already knows how to render as the note.
        #
        # Scores are untouched: marks_awarded and max_possible are both 0, and
        # the totals above are already final.
        try:
            _student_hay_notes = _norm_for_evidence_match(
                getattr(self, "_student_text_last_run", "") or ""
            )
            _already_anchored = {
                _norm_for_evidence_match(_e)
                for _bi in normalized_breakdown
                for _e in (_bi.get("evidence_list") or [])
                if str(_e or "").strip()
            }
            # Ask the restatement pass where the student repeats a credited
            # point, and merge anything the grading model happened to volunteer
            # in `restated_at`. Both feed the same list.
            _by_id: dict[str, dict] = {}
            for _bi in normalized_breakdown:
                _cid_k = str(_bi.get("criterion_id", "") or "").strip().upper()
                if _cid_k and float(_bi.get("marks_awarded", 0) or 0) > 0:
                    _by_id.setdefault(_cid_k, _bi)
            for _cid_k, _line in self._find_restatements(_by_id):
                _tgt = _by_id.get(_cid_k)
                if _tgt is not None:
                    _tgt.setdefault("_restated_at", [])
                    if _line not in _tgt["_restated_at"]:
                        _tgt["_restated_at"].append(_line)

            _notes: list[dict] = []
            for _bi in list(normalized_breakdown):
                _lines = _bi.get("_restated_at") or []
                if not _lines or float(_bi.get("marks_awarded", 0) or 0) <= 0:
                    continue
                _credited_ev = _norm_for_evidence_match(_bi.get("evidence", "") or "")
                _credited_pos = (_student_hay_notes.find(_credited_ev)
                                 if _credited_ev else -1)
                _cid_log = _bi.get("criterion_id", "") or "?"
                for _line in _lines:
                    _ln_norm = _norm_for_evidence_match(_line)
                    # Never put a note on a line that earns marks somewhere.
                    if not _ln_norm:
                        logger.info(f"    DROP  [{_cid_log}] line normalises to nothing")
                        continue
                    if _ln_norm in _already_anchored:
                        logger.info(f"    DROP  [{_cid_log}] line already earns marks "
                                    f"elsewhere | {_line[:55]}")
                        continue
                    _ln_pos = _student_hay_notes.find(_ln_norm)
                    if _ln_pos < 0:
                        logger.info(f"    DROP  [{_cid_log}] not verbatim in the script "
                                    f"| {_line[:55]}")
                        continue
                    if _credited_pos < 0 or _credited_pos == _ln_pos:
                        _dir_note = "above"
                    else:
                        _dir_note = "below" if _credited_pos > _ln_pos else "above"
                    _already_anchored.add(_ln_norm)
                    logger.info(f"    NOTE  [{_cid_log}] \"Marks given {_dir_note}\" on: "
                                f"{_line[:55]}")
                    logger.info(f"          marks are at: "
                                f"{(_bi.get('evidence') or '')[:55]}")
                    _notes.append({
                        "criterion_id": _bi.get("criterion_id", ""),
                        "criterion": _bi.get("criterion", ""),
                        "marks_awarded": 0.0,
                        "max_possible": 0.0,
                        "reason": (
                            f"Marks given {_dir_note}: this point is credited at "
                            f"'{(_bi.get('evidence') or '')[:60]}'."
                        ),
                        "evidence": _line,
                        "evidence_list": [_line],
                        "comments_summary": "",
                        "_restated_note": True,
                    })
            if _notes:
                normalized_breakdown.extend(_notes)
                _above = sum(1 for _n in _notes if "given above" in _n["reason"])
                logger.info(f"  ADDED {len(_notes)} note(s): {_above} above, "
                            f"{len(_notes) - _above} below")
            else:
                logger.info("  ADDED 0 notes - no repeated point survived the checks")
            logger.info("=" * 62)
        except Exception:
            logger.warning("Restatement note generation failed", exc_info=True)

        # Debug logs
        logger.info(f"Raw LLM breakdown count: {len(main_grade.get('breakdown', []))}")
        if _dropped_out_of_rubric:
            logger.info(
                f"Dropped {len(_dropped_out_of_rubric)} LLM-returned criteria that did not "
                f"match any rubric criterion (even via normalized/prefix lookup):"
            )
            for _txt in _dropped_out_of_rubric[:20]:
                logger.info(f"  out-of-rubric: '{_txt}...'")
        logger.info(f"Saved breakdown count: {len(normalized_breakdown)}")
        logger.info(f"Calc sum (post-check): {sum_awarded_calc}")
        logger.info(f"Question max marks used: {total_max}")
        logger.info(f"Final saved total: {rounded_total}")

        # Capture top-level not_required_points from the numerical LLM output.
        # In holistic mode these are stored per-sub-question on the breakdown items
        # (under _not_required_points) and the top-level field stays empty.
        nr_points_top: list[dict] = []
        if not self._holistic_grading:
            raw_nr = main_grade.get("not_required_points", []) or []
            if isinstance(raw_nr, list):
                for nr in raw_nr:
                    if not isinstance(nr, dict):
                        continue
                    text = str(nr.get("text", "") or "").strip()
                    if not text:
                        continue
                    nr_points_top.append({
                        "text": text,
                        "key_phrase": str(nr.get("key_phrase", "") or "").strip(),
                        "reason": str(nr.get("reason", "") or "").strip(),
                    })

        # ── Aggregate-recovery merge (numerical mode only) ────────────────
        # After all guardrails/dedup/gating have run, collapse aggregate-
        # recovered component entries into teacher-style "one line, one
        # combined mark" records. See _merge_aggregate_recovered_entries.
        # Marks totals are preserved (only display shape changes).
        if normalized_breakdown and not self._holistic_grading:
            try:
                _pre_sum = sum(
                    float(b.get("marks_awarded", 0) or 0) for b in normalized_breakdown
                )
                normalized_breakdown = _merge_aggregate_recovered_entries(
                    normalized_breakdown,
                    of_definitions=self._of_definitions_last_run,
                )
                _post_sum = sum(
                    float(b.get("marks_awarded", 0) or 0) for b in normalized_breakdown
                )
                # Sanity: the merge must be totals-preserving.
                if abs(_pre_sum - _post_sum) > 1e-6:
                    logger.warning(
                        f"Aggregate merge changed total (pre={_pre_sum} "
                        f"post={_post_sum}) - this should never happen; "
                        f"falling back to un-merged breakdown."
                    )
            except Exception:
                logger.warning("Aggregate merge pass failed - keeping un-merged breakdown", exc_info=True)

        # ── Restore mark-scheme order ────────────────────────────────────────
        # Until here the breakdown sits in the order the grading model happened
        # to emit criteria, so the CSV and the marker's review jump around the
        # paper - MM22 before MM15, ESR05 before ESR03, the restatement notes
        # all bunched at the end away from the criteria they annotate. Sort by
        # the criterion's position in the rubric so both read in mark-scheme
        # sequence. A merged aggregate entry has no rubric description of its
        # own, so it takes the earliest position among its components.
        try:
            _pos = self._rubric_position_last_run or {}
            if _pos:
                _END = 10 ** 6

                def _rubric_pos(item: dict) -> int:
                    crit = str(item.get("criterion", "") or "")
                    if crit in _pos:
                        return _pos[crit]
                    best = _END
                    for comp in (item.get("components") or []):
                        cd = comp.get("criterion_description")
                        if isinstance(cd, str) and cd in _pos:
                            best = min(best, _pos[cd])
                    return best

                # Stable, so criteria sharing a position (a criterion plus its
                # "marks given above/below" notes) keep their existing order.
                normalized_breakdown.sort(key=_rubric_pos)
        except Exception:
            logger.warning("Rubric-order sort failed - keeping emitted order", exc_info=True)

        doc = {
            "student_id": self.student_name,
            "question_number": self.question_number,
            "total_marks_awarded": rounded_total,
            "total_max_possible": total_max,
            "overall_reason": main_grade.get("reason", "Graded automatically"),
            "breakdown": normalized_breakdown,
            # Keep comments annotation-friendly; store guardrail and unanchored notes separately.
            "comments": annotation_comments,
            "guardrail_warnings": evidence_warnings,
            "unanchored_comments": unanchored_comments,
            "not_required_points": nr_points_top,
            "extracted_at": now_iso,
            "question_id": self.questions_id,
            "model_answer_id": self.model_answers_id,
            "student_answer_id": self.student_answers_id,
        }

        # Flag for the annotator: holistic grading uses sub-question level annotation.
        if self._holistic_grading:
            doc["holistic_grading"] = True

        if os.getenv("DEBUG_SAVE_LLM_OUTPUT", "").strip().lower() in {"1", "true", "yes", "y"}:
            doc["llm_debug"] = self._llm_debug_trace[-5:]
            doc["llm_grading_provider"] = os.getenv("GRADING_PROVIDER")
            doc["llm_grader_model"] = os.getenv("LLM_GRADER_MODEL")

        try:
            validated = StudentGradeDocument(**doc)
            return validated.model_dump(exclude_none=True, by_alias=True)
        except Exception as ve:
            logger.warning(f"Pydantic validation failed - saving raw: {ve}")
            return doc

    def _save(self, doc: dict) -> str:
        try:
            result = self.grades_coll.insert_one(doc)
            doc_id = str(result.inserted_id)
            logger.info(f"Grading results saved → _id = {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"MongoDB save failed: {e}", exc_info=True)
            raise GradingError("Failed to save grading result") from e

    def grade(self) -> Optional[str]:
        try:
            logger.info(f"Starting grading → {self.student_name} | Q{self.question_number}")

            q_clean, m_clean, s_clean = self._load_clean_data()
            grades_parsed = self._run_grading(s_clean, m_clean, q_clean)
            grade_doc = self._build_grade_doc(grades_parsed, q_clean)
            doc_id = self._save(grade_doc)

            return doc_id

        except GradingError as ge:
            logger.error(f"Grading pipeline failed: {ge}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"Unexpected grading error: {e}", exc_info=True)
            return None


# Public API
def grade_student(
    student_name: str,
    question_number: str,
    questions_id: Optional[str],
    model_answers_id: Optional[str],
    student_answers_id: str,
    question_type: str = "numerical",
) -> Optional[str]:
    grader = StudentGrader(
        student_name=student_name,
        question_number=question_number,
        questions_id=questions_id,
        model_answers_id=model_answers_id,
        student_answers_id=student_answers_id,
        question_type=question_type,
    )
    return grader.grade()