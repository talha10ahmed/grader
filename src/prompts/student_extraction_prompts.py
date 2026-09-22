STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE = """You are extracting a student's typed/handwritten answer from PDF page images.

The pages provided ARE this student's answer. They have already been pre-selected
upstream — your job is to faithfully extract every word, NOT to filter by question
label. Whatever the student wrote on these pages is the answer; never return empty
output when the pages contain text.

═══════════════════════════════════════════════════
OUTPUT FIELDS
═══════════════════════════════════════════════════

1. `question`: set this to "{question_number}".

2. `sub_parts`: an array, ALWAYS containing at least one entry.

   • If the student used explicit sub-part labels (e.g. "4.1", "4.2", "a)", "b)",
     "(i)", "(ii)", "1-", "2-", "Case 1", "Scenario 2", "Part A"), create ONE
     entry per label:
         - `question_number`: the label EXACTLY as the student wrote it
           (preserve casing, punctuation, and any typos)
         - `answer`: the full content under that label up to (but not including)
           the next sub-part label or the end of the pages, preserving paragraphs,
           tables, bullet points, and workings

   • NESTED LABELS — PRESERVE THE PARENT (CRITICAL):
     If a top-level numeric sub-part label (e.g. "4.1") is followed by letter-style
     sub-sub-labels ("a)", "b)", "(i)", "(ii)") BEFORE the next numeric label
     appears, the letter-labels are CHILDREN of the numeric label. You MUST
     combine the two into the `question_number` so the parent is preserved.

       Student writes:                    Output should be:
         4.1                              question_number "4.1(a)" with the a) answer,
         a) ...consequences...            then "4.1(b)" with the b) answer.
         b) ...recommendations...
         4.2                              question_number "4.2"
         ...
         4.3                              question_number "4.3(a)" / "4.3(b)" if
         A) ...threat...                  A)/b) appear under 4.3.
         b) ...other threat...

     The combined label uses the EXACT casing the student used (`4.3(A)` if
     student wrote `A)`, `4.1(a)` if lowercase). NEVER drop the numeric parent
     — letter-only labels like "a)" floating without context cannot be matched
     to the rubric's sub-question structure downstream.

   • MISSING PARENT — INFER IT (CRITICAL):
     Students often omit the top-level numeric heading (e.g. "4.1") because
     it's already printed in the question paper, and write only their letter
     labels ("a)", "b)"). The question heading may also live on a different
     page from the answer content. In BOTH cases you must INFER the missing
     numeric parent from context:

       Case A — letter-labels appear at the START of the answer with NO
                preceding numeric label, and the next numeric label is
                e.g. "4.2":

         Student writes:                Output should be:
           a) ...consequences...        question_number "4.1(a)" — INFERRED
           b) ...recommendations...     question_number "4.1(b)" — INFERRED
           4.2 ...                      question_number "4.2"
           4.3 ...                      question_number "4.3"

       Case B — between two numeric labels, letter-labels appear that
                clearly belong to a SKIPPED numeric (e.g. "4.1" missing
                between section start and "4.2"):

         Apply the same inference. The skipped numeric is "(first numeric
         seen) − 1" within the main question's structure.

     For Question {question_number} specifically: if letter-labels appear
     BEFORE the first numeric sub-label, infer their parent as "{question_number}.1".
     If letter-labels appear between "{question_number}.N" and "{question_number}.M",
     the parent is "{question_number}.N" (the most-recent prior numeric).

     NEVER emit a bare letter-label ("a)", "b)", "A)", "(i)") as a top-level
     `question_number` when a numeric parent can be reasonably inferred from
     context — the grader will treat it as a separate question and fail to
     match against the rubric.

   • If NO sub-part labels are present, output EXACTLY ONE entry with:
         - `question_number`: "{question_number}"
         - `answer`: ALL content from ALL provided pages, concatenated in order

   Descriptive topic headings ("Materiality", "Auditor action", "Conclusion",
   "Discussion of X") are NEVER sub-parts — they are section headings inside a
   sub-part. Sub-part identifiers are numbered or lettered labels only.

3. `question_heading_text`: the EXACT verbatim heading the student wrote at the
   start of their answer (e.g. "Quiestion 4", "Q4.", "QUESTION 4"), preserving
   typos and spacing. Set to null if no visible heading.

4. `page_texts`: one entry per provided page:
         - `page`: 1-based page number
         - `text`: ALL visible text on the page, verbatim

═══════════════════════════════════════════════════
ADJACENT-QUESTION TRIMMING (only when clearly present)
═══════════════════════════════════════════════════

If — and ONLY if — the first or last page contains a clearly-labelled OTHER
top-level question (e.g. an explicit "Question 5" or "Q.3" written at the left
margin), exclude only that other question's content from `sub_parts`. The full
raw text still goes into `page_texts`. If no such other-question label appears,
treat all content as belonging to this answer.

Sub-part numbering like "4.1" / "4.2" or scenario numbering like "1-" / "2-" is
NEVER an other-question boundary — those are sub-parts of the current answer.

═══════════════════════════════════════════════════
TEXT & TABLE FORMATTING
═══════════════════════════════════════════════════

- Use clean printable ASCII: letters, digits, standard punctuation (. , ; : ! ? - ( ) [ ]),
  spaces, newlines. No control characters or Unicode artifacts.
- Replace bullet symbols with "-" or "*".
- Preserve paragraph breaks with \\n\\n.
- Preserve every word, number, currency symbol, and unit exactly as written.
- Tables: keep row and column structure as plain text using " | " or aligned
  spaces. Keep all values and headings. NEVER drop a row — if a cell is
  unreadable, write "[unclear]" in place.
- Ignore page headers, footers, watermarks, "Continued...", candidate numbers.

═══════════════════════════════════════════════════
HARD CONSTRAINTS
═══════════════════════════════════════════════════

- `sub_parts` MUST contain at least one entry whenever the pages have any
  student writing. Returning empty `sub_parts` (or an entry with an empty
  `answer` while `page_texts` has content) is a failure.
- NEVER invent sub-parts from descriptive topic headings.
- NEVER paraphrase, summarise, or skip content — copy verbatim.
- The label the student wrote for a sub-part may differ from the model's
  expected numbering (e.g. student writes "4.1" while the system asks for
  question "{question_number}"). Use the STUDENT'S label, not the system's.

Return ONLY valid JSON — no markdown, no commentary, no preamble.
"""


def get_student_extraction_prompt(question_number: str = None) -> str:
    """Return the student extraction prompt, optionally scoped to a specific question number."""
    if question_number:
        return STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
            question_number=question_number
        )
    return STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
        question_number="(unknown)"
    )


# Backward compatibility: default prompt without question filtering
STUDENT_ASSIGNMENT_EXTRACTION_PROMPT = STUDENT_ASSIGNMENT_EXTRACTION_PROMPT_TEMPLATE.format(
    question_number="(the main question on these pages)"
)
