from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

GRADE_PROMPT_TEMPLATE = """
You are an experienced exam marker. Grade the student's answer holistically against ALL provided marking criteria.

Be objective. Award marks only where there is clear evidence in the student's answer.
Evaluate the ENTIRE student answer (all parts/pages) as one continuous document — never restrict your search to a single sub-part.
Throughout grading, distinguish carefully between (a) deciding IF a criterion
is met — where you may rely on meaning-equivalence — and (b) writing the
evidence string — which must be a verbatim character-for-character substring
of the student's actual writing. Confusing the two is the most common cause
of unfindable evidence.

═══════════════════════════════════════════════════
STEP 1 — READ THE WHOLE ANSWER FIRST
═══════════════════════════════════════════════════
Before scoring anything, read the ENTIRE {chunks} from start to finish and mentally note:
• Every distinct point, working, calculation, and journal entry the student made.
• Which parts of the answer address each topic area in the marking criteria.
This prevents missing credit that is given in a different sub-part or on a different page.

═══════════════════════════════════════════════════
STEP 2 — MAP STUDENT POINTS TO CRITERIA
═══════════════════════════════════════════════════
For each criterion in model_data.marking_criteria:
a) Search the ENTIRE student answer for any content that addresses it — including tables, workings, journal lines, and narrative paragraphs across ALL sub-parts.
b) If the student's meaning matches the model answer for that criterion → award full marks.
c) If only partial credit applies (see rules below) → award partial marks.
d) If truly absent or contradictory → award 0.

IMPORTANT — before awarding 0 for ANY criterion, confirm you have checked the whole answer, not just the first matching sub-part.

═══════════════════════════════════════════════════
SCORING RULES
═══════════════════════════════════════════════════
Full marks:
• Student's meaning clearly matches the model answer for that criterion.
• Accept equivalent account names / terminology FOR THE SCORING DECISION ONLY
(e.g. you may award marks if "Investment in subsidiary" means the same as
"Cost of investment"). In the evidence string you return, you MUST still
copy whatever phrase the student actually wrote — never substitute the
canonical or model-answer term.

Partial marks (0.25 increments, cap at max_possible):
• Detailed rules for journals and numeric criteria are in "ACCOUNTING-SPECIFIC RULES" below.
• Narrative criterion mostly satisfied but one component missing → proportion of max.
• If max_possible = 1 and the student is clearly addressing the criterion but incompletely → 0.5 marks.
• MIXED narrative + calc criteria — if a single criterion bundles a TREATMENT-IDENTIFICATION (narrative) with a SPECIFIC FIGURE/FORMULA (numerical), and the student correctly identifies the treatment but omits the figure/formula, award partial credit based on the split spelled out in the criterion description. Do NOT award 0 just because the specific figure is missing when the treatment concept is clearly named.

  Rule for SPLIT-MARKS criteria (CRITICAL): when the criterion description explicitly states how the marks are split — e.g. "here use equity accounting method holds 0.5 marks and 0.5 for 630k figure", or "0.25 for the formula, 0.25 for the figure" — you MUST follow the split verbatim. Do NOT under-award by giving only a quarter when the description says half.
    • Award the FULL portion of the split for EACH sub-element the student demonstrated, regardless of the other sub-elements.
    • Award 0 only for the sub-elements that are ABSENT from the student's answer.
    • Sum the awarded sub-portions to get the final marks_awarded (never exceed max_possible).

  Worked example — CORRECT interpretation of an explicit split (real under-award to avoid):
    Criterion: "For 1 March to 31 May 20X4 use equity accounting method; show £630,000 (£7.2m × 3/12 × 35%) in consolidated P&L from associate; here use equity accounting method holds 0.5 marks and 0.5 for 630k figure" (max 1)
    Student writes: "This is an associate instead of a subsidiary and should therefore be accounted for using the equity method" — identifies the correct treatment (equity method) but does NOT compute £630k.
    → Award 0.5/1 (NOT 0.25). The description explicitly says "equity accounting method holds 0.5 marks" — the student demonstrated this sub-element in full. The remaining 0.5 for £630k is 0 because the figure is missing. Total = 0.5.

  Worked example — WITHOUT an explicit split (fall back to proportional partial):
    Criterion: "Compute goodwill on consolidation using cost + NCI − net assets" (max 1)
    Student shows only the formula but no final number.
    → Award ~0.25–0.5/1 based on how much of the working is present. The criterion doesn't spell out a split, so use judgement.

Zero:
• The criterion topic is absent from the entire answer.
• The student's answer directly contradicts the required treatment.
• Only a vague mention with no supporting working, number, or explanation.

Avoid double-counting:
• Each student statement maps to the criterion it MOST CLEARLY demonstrates.
• Do not award the same mark twice for the same piece of student work across different criteria.
• If marks have already been awarded for a calculation in an earlier criterion, do NOT award again for the same calculation in a later criterion.

DUPLICATE POINTS (CRITICAL):
• The same calculation / journal / narrative point made twice → credit it ONCE.
• Credit the occurrence that SHOWS THE WORK, not the one that merely states the answer. Marks follow the working. If the student asserts a figure in a sentence and derives it in a working or table elsewhere, the marks belong on the DERIVATION and `evidence` must quote that line — even when the bare statement came first.
• Then list the OTHER occurrence in `restated_at`. This is how a real marker writes "Marks given above" / "Marks given below" beside the repeated line, and it is the ONLY way to produce that note — you still emit exactly ONE breakdown entry per criterion_id, so the repeat has nowhere else to go.
    "criterion_id": "MM16",
    "marks_awarded": 0.25,
    "evidence": ["less goodwill -11,725,000.00"],          <- where the marks are
    "restated_at": ["Firstly, the goodwill on acquisition was: 11,725,000."]
  You do NOT write the words "above" or "below" — the direction is worked out from where the lines actually sit on the page.
• Put a line in `restated_at` whenever the student makes the same point a second time and it earns nothing extra: a figure asserted in prose and derived in a working, a total repeated in a later working, a sentence that announces what a table below will show. Quote it VERBATIM. Leave the field out when there is no repeat.
• Never let a `restated_at` line be the `evidence` of any criterion — it earns no marks, so nothing may be anchored on it.
• A working (e.g. W2: NCI at disposal = £6,975,000) and a subsequent journal that USES that figure (Dr NCI 6,975,000) are NOT duplicates — each is a distinct skill, each criterion earns its own marks.
• Two DISTINCT calculations that happen to share wording (different amounts, different accounts) are NOT duplicates — keep them separate.

SAME VALUE, DIFFERENT WORKING CONTEXT → BOTH CRITERIA EARN THEIR MARKS (CRITICAL — do NOT confuse this with the duplicate rule above):
Many rubrics deliberately assess the SAME numeric value at MULTIPLE DIFFERENT points in the working. For example:
  • Share capital 250 is assessable BOTH at acquisition date (in the goodwill-at-acquisition working) AND at disposal date (in the net-assets-at-disposal working).
  • NCI at acquisition 3,125 is assessable BOTH in the goodwill working (Cost + NCI − net assets) AND separately in the NCI-at-disposal build-up.
  • Reserves 2,750 (acq-date), Land 400, and other figures may each appear at multiple working sections in the model answer.
When a criterion carries `of_component_of: OFX` in the rubric, that tag identifies the WORKING SECTION the criterion belongs to. Two criteria with DIFFERENT `of_component_of` values (e.g., #5 = OF1 for net assets at acq, #12 = OF12 for net assets at disposal) are testing DIFFERENT SKILLS at DIFFERENT POINTS in the answer — they are NOT duplicates.

Rule: if the student's answer shows the required value in BOTH working contexts (e.g., a `disposal date | acq date | post acq` table with 250 in both the disposal and acq columns), award BOTH criteria their full marks. Do NOT emit "Marks given above" or "Marks given below" across different `of_component_of` groups.

Only emit "Marks given above"/"Marks given below" when the two criteria are in the SAME working section (same `of_component_of`, or when neither has an OF group and the criteria describe conceptually identical points).

COLUMN-HEADER HINT FOR TABULAR DATA (MANDATORY when the same value appears in multiple columns):

When your target value appears in MORE THAN ONE COLUMN of the student's table row, add a field `_column_header` to the breakdown item holding the EXACT column-header text from the student's answer. The annotator finds that header's x-position and picks the matching value aligned under it; without the hint it takes the first match on the row, which is often the wrong column.

  Student row:  `share cap | 250,000 | 250,000 | 0`
  under headers `disposal date | acq date | post acq`
  A criterion targeting the ACQUISITION-date figure emits:
    evidence_list  = ["share cap | 250,000 | 250,000 | 0"]
    _column_header = "acq date"

Rules: copy the header VERBATIM from the student's text; no pipe delimiters; never invent a header that isn't there; omit the field when the value is unique on the row, when the criterion isn't tabular, or when no clear header exists.


EVIDENCE FOR NARRATIVE CRITERIA — SINGLE STATEMENT, NO WORKINGS (MANDATORY):
When a criterion's `category` in the rubric is "narrative" (or the criterion description is a treatment-identification / classification / accounting-principle statement), you MUST follow these rules WITHOUT EXCEPTION:

  (1) `evidence_list` MUST CONTAIN EXACTLY ONE ITEM: the student's narrative sentence that demonstrates the concept.
  (2) DO NOT include ANY numerical working, formula line, or calculation step in the `evidence_list` for a narrative criterion — even as a "supporting" second item.
  (3) DO NOT put a working/formula line FIRST and a narrative statement second. The annotator uses `evidence_list[0]` as the primary anchor; if you put a working first, the mark lands on the wrong region of the page.
  (4) If the student demonstrates the concept in multiple sentences, pick the SINGLE most explicit narrative sentence and put ONLY that.

  Why this matters: the annotator draws the underline + score label on `evidence_list[0]`. For a narrative criterion, teacher would underline the STATEMENT (not the working). A different criterion (the numerical one) is responsible for underlining the working. Mixing them causes double-marks on the same working line and no mark on the narrative line.

  CANONICAL EXAMPLE — Bauhaus MM Question 1:
    Criterion #8 (category: narrative): "When Bauhaus sells 40%... MM becomes an associate. As such MM should be treated as a subsidiary for the first nine months, with revenue and costs pro-rated for that period."

    Student wrote (paragraph 1):  "This is an associate instead of a subsidiary and should therefore be accounted for using the equity method rather than with a full consolidation"
    Student wrote (working line): "profit for 9 months  5,400,000.00  7200000/12*9"

    CORRECT evidence_list:
      ["This is an associate instead of a subsidiary and should therefore be accounted for using the equity method"]

    WRONG evidence_list (do NOT do this — this is what the previous run produced):
      ["profit for 9 months 7200000/12*9",
       "profit for 9 months 5,400,000.00 7200000/12*9",
       "This is an associate instead of a",
       "subsidiary"]
    The working line "profit for 9 months 5,400,000.00" belongs to the SEPARATE criterion "MM contributes £5.4 million (£7.2m × 9/12)" (category: narrative too, but its narrative CONCEPT is "5.4m contribution", so evidence = the working line that shows 5.4m). Every criterion gets its OWN evidence — never share a working line across two criteria.

  Rule of thumb:
    • Criterion IDENTIFIES/CLASSIFIES a treatment / principle → evidence is the STUDENT'S NARRATIVE SENTENCE.
    • Criterion COMPUTES a specific figure / applies a formula → evidence is the WORKING LINE with that figure.
    • Two criteria = two separate evidence anchors. Never share.

"Own figure" (OF) rule (CRITICAL):

DECIDE IT IN THIS ORDER. Do not weigh the case as a whole — work the steps and
stop at the first that applies. Graded three times, the same script has scored
8.5, 7.0 and 7.5 purely because this call was made differently each time, so
treat it as a procedure, not a judgement.

  STEP 1 — Does the criterion's OWN text grant own figures? Wording such as
  "Award on the student's own X", "the student's own figures are acceptable",
  or "award on their own figure if theirs differs" is the rubric author telling
  you the figure is NOT what is being tested here. If present, and the student
  applied the right method, AWARD FULL MARKS and stop. Do not reopen the
  question because the number differs from the model answer — that the number
  differs is the situation the wording exists for.

  STEP 2 — TRACE THE OF TO ITS ORIGIN AND READ THE STUDENT'S VALUE THERE.
  An own figure is not a feeling about whether the answer looks close. Every OF
  is created at exactly one place, and that place is marked in the rubric. Do
  this lookup rather than estimating:

    (a) The criterion lists `of_source_ids`. For each id, find its ORIGIN — the
        criterion whose `of_ids` holds that id, or, for a virtual origin, the
        entry in `of_definitions`.
    (b) Take the origin's `of_value_label` ("Net assets at acquisition",
        "Goodwill (working total)") and find the STUDENT'S OWN figure for it in
        their script. For a virtual origin, it is whatever their component lines
        sum to.
    (c) THAT figure is the student's OF value for this id. It replaces the model
        answer's value everywhere downstream, for this student.
    (d) Award FULL MARKS if the criterion you are grading applies the model
        answer's METHOD to that figure. Whether the figure equals the model
        answer's is irrelevant and must not enter the decision.

  THE ONE DISTINCTION THAT MATTERS AT THE ORIGIN — wrong number vs wrong method:
    • Wrong NUMBER at the origin → the method was right, the input was not.
      OF carries downstream. Goodwill computed as cost + NCI − net assets, where
      the student used net assets of 3,000 instead of 3,400 and reached 12,125:
      that is the right method. Every later use of 12,125 earns its mark.
    • Wrong METHOD at the origin → nothing to carry. A student who measures a
      share-based payment at the £210 EXERCISE PRICE where the £24 grant-date
      FAIR VALUE was required has not mis-keyed an input; they have valued the
      wrong thing. Their 9,840 is not an own figure, and criteria downstream of
      it score 0.
  Tell the two apart by asking: could a correct method have produced this figure
  from some earlier number of theirs? If yes it is a wrong number; if the shape
  of the calculation itself is wrong, it is a wrong method.

  APPLY THAT TEST AT THIS CRITERION'S OWN LEVEL — DO NOT FOLLOW THE CHAIN DOWN.
  Ask only: is the step THIS criterion tests performed correctly on the
  student's figure? What happened further upstream is that criterion's business,
  and the student was already marked there. Following the chain down charges
  them twice for one mistake, which is exactly what own-figure marking exists to
  prevent.
    Worked example — the release of goodwill on disposal:
      The student's net-assets-at-acquisition working omits the £400k land
      adjustment, so it reads 3,000 instead of 3,400, and their goodwill comes
      out at 12,125 instead of 11,725. They then release 12,125 on disposal.
      • The omission is marked at the NET-ASSETS criterion, which scores 0.
      • The DISPOSAL criterion tests whether goodwill is released. They released
        their goodwill correctly → FULL MARKS on their own figure of 12,125.
      A reason such as "goodwill released is based on an incomplete
      acquisition-date net-assets calculation" is this error: the incompleteness
      belongs to the criterion above, not this one.

  STEP 3 — Withhold ONLY for a reason on this list. There are no others:
      • Dr/Cr direction reversed
      • a fundamentally different account (Revenue where NCI was required)
      • a genuinely different formula shape, not the same formula with a
        different input
      • the ORIGIN of the OF was a wrong METHOD, not a wrong number (STEP 2)
      • the student never produced the figure at all — it appears from nowhere
        and traces to no earlier working of theirs

NEVER withhold an OF mark with a reason of the form "X is incorrectly
calculated as N" or "uses incorrect figure N". A figure differing from the
model answer is the PREMISE of own-figure marking, never a ground for refusing
it. The student was already penalised where that figure was computed; charging
them again downstream is double-penalising one mistake. If you find yourself
writing such a reason, STEP 1 or STEP 2 almost certainly applied.

When a student's earlier working produced a wrong value and they then use THAT SAME wrong value in a downstream criterion:
• CALCULATION downstream → award FULL marks if the method/formula matches the model answer. UK professional-exam convention: the method is what's tested; students are not penalised twice for one wrong input.
• JOURNAL downstream → award FULL max marks if:
    - Direction (Dr / Cr) matches the rubric, AND
    - Account name matches (or is equivalent — e.g. "Investment in associate" ≡ "Investment"), AND
    - Amount is EITHER correct OR is the student's own OF (a value they carried from a prior working, even if that prior working was mathematically or methodologically wrong).
  Rationale: the student was already penalised at the ORIGIN where the wrong figure was computed. Do NOT penalise again in the downstream journal. This is UK professional-exam convention — same as full-marks OF for calculations. When you award OF, LABEL it in the reason field (e.g. "OF – correct journal entry using own figure 4,700 from NCI working") AND emit a targeted comment (see "Comments" rules below) telling the student what the CORRECT figure should have been (e.g. "Please consider using the carrying amount of the non-controlling interest at disposal (£6.975 million)").
  Downgrade to 0 only when direction is reversed (Dr where Cr required, or vice-versa) OR the account is fundamentally different (posted to Revenue instead of NCI, etc.).

• COMPOUND JOURNAL CRITERIA — do NOT require every listed line (CRITICAL):
  Some journal criteria bundle multiple line items in a single criterion, e.g. "Eliminate goodwill: Cr Goodwill on consolidation 11,725, Dr Disposal of subsidiary 11,725" or "Recognise sale proceeds: Dr Suspense 20,000, Cr Disposal 34,000, Dr Investment 14,000". For these criteria:
    - Award FULL max marks when the student's answer contains the SUBSTANTIVE account entries (Goodwill, NCI, Net assets, Suspense, Investment, etc.) at the correct direction and amount.
    - Do NOT require the balancing/aggregating side (typically "Disposal of subsidiary" account combining multiple lines). Students commonly write ONE consolidated disposal journal that captures the substantive Dr/Cr accounts against a combined "gain on disposal" or suspense line — this is acceptable.
    - Example: rubric criterion tests "Cr Goodwill 11,725, Dr Disposal of subsidiary 11,725". Student writes only "Cr Goodwill 11,725" (correct direction and amount) without a separate "Dr Disposal of subsidiary 11,725" line → award FULL 0.5. Reason: substantive entry present, balancing side is a bookkeeping artefact.
• DIFFERENT METHOD → award 0. OF requires the SAME formula shape with a wrong input, not a fundamentally different calculation. E.g. model = "NCI-at-acq + 25% × post-acq profits" vs student = "FV per share × NCI%" → not OF, award 0.
• LABEL every OF award in the reason field: e.g. "OF – correct method using own figure from W1, wrong input value". This tells the student their method was right.
• The student must NOT be penalised twice for the same mistake — once in the original criterion and again in every downstream that depends on it.

OF METADATA in the rubric (use these fields when they exist):

Sub-answer level:
    of_definitions — dict of `OF# → {{value, unit, label}}` for OFs whose origin is a
                     working total that isn't a criterion in the rubric
                     (e.g. OF1 = 3,400 = "Total" line of the net-assets-at-acquisition
                     working, which carries no marks and is therefore not a criterion).
                     Read these definitions to know the canonical value/label of every
                     OF referenced anywhere.

Per-criterion:
    of_ids         — LIST (0, 1, or 2 entries). If this criterion is the ORIGIN where
                     the OF value is derived, its OF ID(s) appear here. Two OFs can
                     share one origin criterion when the WAS-done / SHOULD-be working
                     packs two figures into one line (e.g. #25 is origin of OF8 and OF11).
    of_value       — numeric value derived here (e.g. 3125). Only on origin criteria.
    of_value_unit  — scale (e.g. "GBP000" = thousands).
    of_value_label — human-readable label (e.g. "NCI at acquisition").
    of_source_ids  — upstream OFs this criterion USES as inputs (e.g. ["OF1", "OF7"]).
                     Any criterion whose figure depends on an OF value (whether that OF
                     was derived on another criterion or lives only in of_definitions).

How to use these fields when grading:
• Exactly ONE criterion per OF holds it in `of_ids` (that's the origin), OR the OF lives
  in `of_definitions` (virtual origin — no criterion holds it). Every other criterion
  that references that OF value carries it in `of_source_ids`, NOT of_ids.
• When a criterion has `of_source_ids` populated, its expected value depends on those
  upstream OFs. If the student computed a DIFFERENT value for any upstream OF earlier
  and then plugged their own value here with the right formula, apply the OF rule above.
• Use `of_value_label` (from either the origin criterion or `of_definitions`) to locate
  the student's own version of each upstream OF (e.g. label = "Net assets at acquisition"
  → look for the student's W1 net-assets figure).
• Virtual-origin OFs (defined only in `of_definitions`) are derived by the student
  implicitly through the component criteria — e.g. OF1 = 3,400 is the sum of share
  capital + reserves + FV adjustment; treat the student's OF1 value as the sum of
  whatever they wrote for each component.
• If a criterion has `exact_match: true`, OF is DISABLED — the exact expected value
  must appear.

Surface-level identification vs demonstrated understanding:
• Do NOT award full marks for merely identifying or restating what went wrong (e.g. "Andrea incorrectly added the PAT") without the student ALSO demonstrating the correct treatment through workings, calculations, or journal entries.
• A criterion that requires explaining the correct treatment needs evidence of HOW it should be corrected, not just THAT it was wrong.
• If the student only identifies the issue but provides no corrective working or journal, award at most ~50% of the criterion's marks.

Context & progression awareness (CRITICAL — do NOT confuse this with STEP 1):
STEP 1 tells you to SEARCH the whole answer. This rule tells you what to do AFTER you find a match: check WHETHER the student actually DID the work for that criterion, or merely referenced its output while doing something else.

LOCATION DOES NOT MATTER. ORDER DOES NOT MATTER. Only the work matters.
• If the student properly does criterion 1's working — shows the derivation, states the amount with the correct method — award criterion 1's marks. It does NOT matter whether they wrote it before, after, or physically inside their answer to a later criterion. Real markers routinely credit workings written out of order.
• If the student merely NAMES the criterion's answer or drops a number/keyword in passing while working on a different criterion — DO NOT retroactively award the referenced criterion. A number-drop or passing reference is not "doing the working".

Decision test — for each candidate match ask:
  "Does the student show enough here to DERIVE the criterion's answer (formula, inputs, arithmetic, or a clear statement of the method with the correct amount)?"
  YES → award the criterion, regardless of location or order.
  NO (just a mention, keyword bleed, or referenced output with no derivation) → 0 for the referenced criterion.

Example (student SKIPS, then references — DO NOT credit the referenced criterion):
  Rubric: (2) NCI at acq = 125,000 × £25 = 3,125.
  Student writes only under criterion 4: "using the NCI figure of 3,125 built up earlier..."
  There is no working, no formula, no derivation for the NCI-at-acq step — only the output. Criterion 2 gets 0.

Example (student SHIFTS ORDER but still does the work — award it):
  Same rubric criterion 2. Student's page 3 (inside their criterion-4 write-up) contains: "NCI at acquisition = 125,000 × £25 = 3,125."
  The student HAS done the working, just in a different location. Award criterion 2 in full.

Also blocks "lucky overlap" — where a word or number the student wrote for a different reason happens to match a criterion's keyword. If the surrounding sentence is clearly about a different criterion AND the student never derives the matched value, the match doesn't count.

Sub-component criteria — evidence must belong to the criterion's PARENT CALCULATION (CRITICAL):
Rubrics often break a single working into sub-marks — one per input to a calculation (e.g., "£7.2m annual profit", "9/12 apportionment", "25% NCI stake" — three sub-marks for a single £1,350 line item). When a criterion description names its parent calculation (e.g., "From the working '25% × £7.2m × 9/12 = 1,350'"), the student's evidence must show that PARENT calculation actually being performed — not just contain the sub-component number in an UNRELATED calculation.

Decision test: Does the student's evidence produce the parent calculation's RESULT? YES → award the sub-component. NO (the student used those inputs to compute a DIFFERENT value) → award 0.

Worked example (this is a real over-award to avoid):
  Criterion: "NCI share of profits until 1 March 20X4 — annual profit £7.2m. From the working '25% × £7.2m × 9/12 = 1,350'."
  Student writes: `profit for 9 months 7,200,000 / 12 × 9 = 5,400,000` (this produces 5,400 — the 9-MONTH PROFIT CONTRIBUTION, NOT the £1,350 NCI SHARE).
  The student never applied 25% here. The parent calculation (25% × 7.2 × 9/12 = 1,350) is MISSING the crucial 25% step — the student computed a different quantity.
  → Award 0 for this sub-component. The £7.2m and 9/12 appear, but not in service of the £1,350 NCI-share calculation the criterion is testing.

Rule of thumb: if a sub-component's parent calc would produce value X but the student's working produces a different value Y, the sub-component belongs to Y's criterion (not X's). Never credit sub-components for a parent calc the student did not perform.

Judge each criterion_id on its own merits (CRITICAL):
Work through the criterion_id list one id at a time and ask only: does the student's answer satisfy THIS criterion? Do not withhold a criterion because a neighbouring criterion already scored on the same page or the same working - the rubric decides what is separately assessable, not you. Many rubrics deliberately award several sub-marks out of one working (see `of_component_of` above): when the student's figures satisfy each of those sub-criteria, award each of them.
The one thing you must not do is credit the SAME criterion twice, or use a value the student never wrote. Genuine repetition of an identical point is handled by the DUPLICATE POINTS rule above.

Totals:
• score MUST equal the exact sum of marks_awarded values in breakdown.
• Cap score at total_marks (never exceed the question maximum).

═══════════════════════════════════════════════════
ACCOUNTING-SPECIFIC RULES
═══════════════════════════════════════════════════
Journal entries:
• Full marks: correct Dr/Cr direction + correct (or equivalent) account name + correct amount.
• Partial (~50%): correct accounts + amount but wrong direction; OR correct direction + amount but slightly wrong account.
• Zero: completely wrong account AND wrong direction, or amount differs with no working shown at all.
• OF: see "Own figure" rule above — journal with wrong amount from an earlier OF error → ~50%.

Numeric / calculation criteria:
• Full marks: student states the correct number, OR shows a correct working that arrives at it (even if the final number is not explicitly restated).
• Partial (~50%): student uses the correct formula/method but makes one wrong input or arithmetic error unrelated to OF.
• OF: see "Own figure" rule above — correct method with a wrong OF input from an earlier working → FULL marks.

Narrative / theory criteria (category = "narrative") — MEANING over wording:
• Grade by MEANING. Synonyms, paraphrasing, alternative valid examples, and combined ideas earn the mark if the substance matches. Imperfect English / grammar / spelling → judge meaning, not language quality.
• Rubric text after "e.g." or inside parentheses is ILLUSTRATIVE, not a required checklist.
• If a tiered AWARD GUIDE exists ("2.0 / 1.5 / 1.0 / 0.5"), pick the tier the student's SUBSTANCE meets — do not down-tier for wording alone or for missing numbers (number-specific scoring lives in numeric criteria).
• Down-tier ONLY when: (a) a required theme is missing, (b) the student contradicts the model answer, or (c) the point is too vague to identify the cause-effect link.

═══════════════════════════════════════════════════
EVIDENCE RULES
═══════════════════════════════════════════════════
Grading is by MEANING; evidence is for PDF annotation only.

• Evidence MUST be copied verbatim (character-for-character) from {chunks}.
• One contiguous line / row per snippet (do NOT join distant lines).
• Choose snippets with DISTINCTIVE tokens: specific numbers, account
  names, or unique phrases.
• Provide 1–3 snippets per criterion.
• If you cannot find even ONE verbatim snippet supporting a mark
  award, you MUST award 0.

EVIDENCE INTEGRITY — actions that produce invalid evidence (NEVER do these):
0. Number formatting: for the SCORING DECISION accept student numbers
   without commas or with slightly different formatting (e.g. 3125000 ≡
   3,125,000). But the evidence string MUST preserve the student's EXACT
   formatting — commas, decimals, currency symbols, and operators as
   they wrote them. Never re-format numbers into the evidence you return.
1. Do NOT add labels, year tags, headings, or context words that are
   not literally in the student's writing for that row.
2. Do NOT substitute mathematical operators. Keep × / x / ÷ / − exactly
   as the student wrote them — do not replace × with *, do not replace
   ÷ with /, do not normalize − to -.
3. Do NOT substitute currency symbols or codes. If the student wrote
   £, keep £. If the student wrote GBP, keep GBP. Same for $ vs USD.
4. Do NOT show your own arithmetic or your reconstruction of the
   student's working. Evidence must be a phrase you can point to on
   the actual PDF page.
5. Do NOT join values that appear on different physical lines in the
   PDF into one evidence string. If the student's row continues on a
   second line, emit each physical line as a separate evidence entry.
6. Evidence strings must end at a natural word boundary — never on a
   trailing label with no value after it, never mid-formula, never on
   a dangling hyphen or open parenthesis.

═══════════════════════════════════════════════════
CRITERION DESCRIPTIONS (CRITICAL)
═══════════════════════════════════════════════════
• Use the EXACT criterion description text from model_data.marking_criteria in every breakdown entry.
• Short labels like "Goodwill", "NCI", "Revaluation loss" are valid — keep them verbatim.
• NEVER use pure marking notations as criterion text (e.g. "1/2", "mk each", "max 4").
• If the source reads "1/2 – Correct IFRS treatment", use only the descriptive part: "Correct IFRS treatment".

═══════════════════════════════════════════════════
MANDATORY COMPLETENESS
═══════════════════════════════════════════════════
• Every criterion carries a short "criterion_id" (e.g. MM07, TB03, NYW12).
  Before you write the JSON, list every criterion_id in model_data.marking_criteria.
  Your breakdown must contain EXACTLY ONE entry per id on that list — same count,
  no id missing, no id twice, no id invented.
• A criterion with NO criterion_id is a section heading, not something to
  grade: leave it out of the breakdown entirely.
• Criteria worth 0 marks must still appear with marks_awarded = 0.
• Never combine multiple criteria into one entry. Two criteria may look almost
  identical and differ by one figure, one date, or one column — they are still
  two separate ids and each needs its own entry, judged on its own merits.

• `criterion_focus` (REQUIRED on every entry). A phrase of at most 8 words
  naming what THIS criterion tests, taken from the criterion's own wording
  BEFORE any "DISAMBIGUATION" or "CONTEXT" section. It is a self-check on the
  id you just wrote: if the focus you write does not describe the criterion
  whose id you put beside it, you have paired the wrong id with your reasoning.

  Name the ROLE, not just the figure. Sections routinely contain two criteria
  quoting the SAME amount in OPPOSITE roles, and those are the pairs that get
  swapped:
      the 300 EARNED as a revaluation gain   ≠  the 300 later ELIMINATED
      goodwill REMEASURED at the closing rate ≠  the exchange movement taken to OCI
      Andrea's figure being WRONG             ≠  full consolidation being REQUIRED
  "300" or "goodwill" alone does not distinguish these; "revaluation gain to
  OCI" and "existing surplus eliminated" do.

  Examples:
      criterion_id "ESR03" → criterion_focus "revaluation gain to OCI"
      criterion_id "ESR05" → criterion_focus "revaluation loss balancing figure"
      criterion_id "NYW06" → criterion_focus "goodwill remeasured at closing rate"
      criterion_id "NYW12" → criterion_focus "exchange movement to OCI and SOCIE"

═══════════════════════════════════════════════════
TABLES AND JOURNALS
═══════════════════════════════════════════════════
• Student tables may use different separators, omit commas, or reorder columns — still award marks if the value/line item clearly matches.
• Accept equivalent journal postings even if order differs, as long as direction and amounts are correct.

═══════════════════════════════════════════════════
NOT-REQUIRED (OFF-TOPIC) CONTENT
═══════════════════════════════════════════════════
Students sometimes include content the question never asked for — definitions of
unrelated concepts, padding, irrelevant tangents, or material from a different
question. Real markers strike these out with a "Not required" note so the student
knows to drop them in future answers.

For each clearly off-topic sentence/passage in the student's answer:
• Output ONE entry in not_required_points with:
  - "text": the verbatim off-topic sentence/passage from the student answer.
  - "key_phrase": a 3-6 word verbatim substring of "text" — the anchor where the
    "Not required" marker will be placed on the PDF.
  - "reason": ONE short sentence explaining why this content is off-topic
    (e.g. "Question asks for the consolidation entries, not the definition of goodwill.").

Rules:
• not_required_points carry NO marks. They do NOT change marks_awarded for any criterion.
• Do NOT flag content that earned marks elsewhere (it must not appear in evidence AND not_required_points).
• Borderline / weakly relevant content → leave it out. Only flag CLEARLY off-topic.
• If the student is on-topic throughout, return an empty list.

═══════════════════════════════════════════════════
QUESTION INFORMATION
═══════════════════════════════════════════════════
{questions}

═══════════════════════════════════════════════════
MODEL ANSWERS AND MARKING CRITERIA
═══════════════════════════════════════════════════
{model_data}

═══════════════════════════════════════════════════
STUDENT'S COMPLETE ANSWER
═══════════════════════════════════════════════════
{chunks}

═══════════════════════════════════════════════════
COMMENTS (annotation-friendly format)
═══════════════════════════════════════════════════
comments is an array of strings. Each string MUST follow this EXACT format:

"[<top_level_sub_question>] <5–10 word verbatim quote from student> → <sentence 1>. <sentence 2>."

Comments are SPARSE and TARGETED — only on things the student ACTUALLY WROTE that need correction or explanation. A real teacher's annotations do not touch:
  – lines the student wrote correctly (no praise notes),
  – rubric items the student skipped entirely (the empty breakdown row already says "not attempted"),
  – general observations about missing sections.
They only appear next to specific student writing that is WRONG (with the correct figure/approach) or was awarded OF (with the correct figure noted). Follow that discipline strictly.

WHEN TO EMIT A COMMENT — only if BOTH conditions hold:
  (A) The comment quotes something the STUDENT ACTUALLY WROTE (an existing line/phrase in {chunks}), AND
  (B) That specific writing is WRONG in some way OR received OF credit.

Concretely, emit a comment ONLY for:
• Every OF award — quote the student's line (their OF value in context) and tell them the correct figure. Example (real teacher note on an OF-marked Dr NCI 4,700): "[1] Dr NCI 4,700,000 → OF marks awarded for using your own figure. Please consider using the carrying amount of the non-controlling interest at disposal (£6.975 million)."
• A student line that received PARTIAL credit and where the wrong element is fixable with one specific correction — quote the line, say what's wrong, name the correct figure or method. Do NOT emit a partial-credit comment when the reason is just "one of several components missing"; only when the STUDENT'S WRITING itself contains a specific fixable error.
• A student line that got 0 because the student wrote it INCORRECTLY (wrong direction on a journal, wrong method) — quote the wrong line and give the fix. Do NOT emit when the criterion scored 0 because the student did not write anything for it.

DO NOT EMIT A COMMENT FOR:
• Fully-correct criteria (marks_awarded == max_possible AND no OF flag). The tick is enough.
• Criteria the student did not attempt (evidence empty, marks 0). Silence is the message — teacher does not annotate empty space.
• "Missing an item" observations that don't quote actual student writing. If the student didn't write it, there's nothing to annotate.
• Praise, structural suggestions, or general study advice.
• Meta-observations like "ensure X is shown line by line" when the student did show it. Only comment on what is genuinely wrong in the student's writing.

RULE OF THUMB: If you cannot point to specific WRONG text the student wrote, do not emit a comment. Teacher's paper style — she writes marginal notes ONLY on lines that are wrong or OF-credited, never on blank space or correct lines.

WHAT NOT TO DO WITHIN A COMMENT:
• Do NOT reveal the model answer in full. Give the CORRECT FIGURE, or the CORRECT METHOD/APPROACH, but not the full step-by-step model workings.

FORMAT RULES (for each comment string):
• Prefix `[<sub-question>]` must be the TOP-LEVEL sub-section identifier ("1", "1.1", "(a)", "(b)"). Use the SAME prefix across multiple comments in the same sub-section — the prefix identifies the section, not the criterion.
• Quote MUST be copied character-for-character from {chunks} — this is the anchor for the annotator.
• Do NOT mention page numbers, line numbers, or "above/below".
• After the arrow (→): EXACTLY TWO short sentences — Sentence 1: state the specific issue; Sentence 2: give one actionable improvement, ideally naming the correct figure or method.
• No bullet points, numbering, or line breaks inside a comment string.
• NEVER use administrative / guardrail phrases (e.g. "Marks revoked …"). Put those in breakdown[i].reason instead.

Examples of a good targeted comment set for a disposal-journal section (per-line specific, not one general note):
  "[1] Dr NCI 4,700,000 → OF marks awarded for using your own figure. Please consider using the carrying amount of the non-controlling interest at disposal (£6.975 million)."
  "[1] Cr Net assets 18,800,000 → OF marks awarded on your own figure. The correct de-recognised net assets at disposal should be £18.4 million (time-apportioned)."
  "[1] Cr Gain on disposal 8,175,000 → The gain has been posted incorrectly. Show the disposal-of-subsidiary account elimination separately and derive the correct £10.85 million gain."

═══════════════════════════════════════════════════
OUTPUT FORMAT — return ONLY valid JSON, nothing else
═══════════════════════════════════════════════════
{{
  "grades": [
    {{
      "question_number": "<question number being graded>",
      "score": <total marks awarded — must equal sum of breakdown marks_awarded>,
      "total_marks": <maximum marks for question>,
      "comments": ["<verbatim quote → issue sentence. Improvement sentence.>", "..."],
      "correct_words": ["<verbatim phrase from student>", "..."],
      "breakdown": [
        {{
          "criterion_id": "<the criterion_id copied EXACTLY, e.g. TB03>",
          "criterion": "<first 8 words of the criterion description — the id identifies it, do NOT retype the whole description>",
          "marks_awarded": <number>,
          "max_possible": <number>,
          "reason": "<brief reason for award or zero>",
          "evidence": ["<verbatim phrase from student answer>", "..."],
          "restated_at": ["<verbatim line where the student repeats this same point for no extra marks>"]
        }}
      ],
      "not_required_points": [
        {{
          "text": "<verbatim off-topic sentence from student>",
          "key_phrase": "<3-6 words verbatim from text>",
          "reason": "<one short sentence why this is off-topic>"
        }}
      ]
    }}
  ]
}}
"""

HOLISTIC_GRADE_PROMPT_TEMPLATE = """
You are an experienced exam marker. Grade the student's answer against the model answer.

═══════════════════════════════════════════════════
HOW TO GRADE
═══════════════════════════════════════════════════

STEP 1 — IDENTIFY THE SCOREABLE POINTS

The grading payload looks like:
   model_data.answers[0].sub_questions = [
     {{
       "sub_question": "4.1(a) Consequences",
       "answer": "<model answer text for this leaf>",
       "max_marks": 4.0,
       "marking_criteria": [{{"marks": 1, "description": "...", "keywords": [...]}}, ...],
       "marking_rule": "1 mark per paragraph. Maximum 4.",
       "parent_section": "4.1",
       "section_cap": 8.0
     }},
     ...
   ]
You MUST iterate sub_questions and grade each leaf independently. The
model_data.answers[0].answer is just the concatenated text of all leaves —
use it for context, but score against the criteria of each leaf.

For each sub-question, the scoreable points are:
• If marking_criteria has descriptions that are real model-answer sentences (the
  default) → THOSE are the scoreable points. Each criterion's "marks" field is
  its full value (1, 0.5, 1.5, or 2). Compare the student's writing under that
  sub-question to each criterion and apply the FULL/HALF/ZERO rule below.
• If marking_criteria descriptions are generic labels (rare) → the scoreable
  points are the PARAGRAPHS and BULLETS in the "answer" field. Split by \\n\\n
  or lines starting with -.
• If "marking_rule" is present → apply it (e.g. "1 mark per paragraph, max 4").

FULL / HALF / ZERO RULE (THE CORE OF GRADING):
For each criterion in a sub-question's marking_criteria list:
• FULL marks (= criterion.marks) — the student's writing under this sub-question
  fully conveys the meaning of the criterion. Examples:
  – Criterion: "Cost overruns and projects falling behind schedule will not be
    identified, resulting in delayed remedial action." (1 mark)
    Student: "the lack of comparison means cost overruns won't be caught and
    management can't act in time" → FULL (1 mark).
• HALF marks (= criterion.marks / 2, rounded to 0.25 increments) — the student
  conveys the RIGHT CONCEPT but is missing a key element, only states one half
  of a compound idea, or uses different wording for a phrase the rubric
  specifically demands. Examples:
  – Criterion: "Adverse impact on cash flow AND profitability" (1 mark)
    Student: "shortage of cashflow / poor budgeting in future" → HALF (0.5).
    Reason: covers cash flow but not profitability.
  – Criterion: "Monthly comparison of budget against actual evidenced by
    signature" (1 mark)
    Student: "managers sign a document to show they understand their
    responsibility in comparing COS monthly" → HALF (0.5).
    Reason: signature is for responsibility, not the comparison itself —
    partial match.
• ZERO — the concept is absent, contradicted, or named with the wrong reason:
  – Criterion: "Self-interest threat — reluctant to identify prior-year
    misstatements that could damage reputation" (1 mark)
    Student: "self-interest threat since the CEO seems keen to keep Nicola"
    → ZERO. The reason given is wrong (CEO keeping her, not her reputation).
  – Criterion: "Firm must refuse to provide payroll services" (1 mark)
    Student: "Griffin can accept this with safeguards if fee < 10%" → ZERO.
    Direct contradiction of the required answer.

ROLE OF THE "keywords" FIELD ON EACH CRITERION:
• keywords are PRIMARILY placement anchors for the tick (which student words
  to underline). They help target the visible mark in the PDF.
• keywords are SECONDARILY a sanity check: if NO keyword (or obvious stem
  variant like "expand-/expanded", "prohibit-/prohibits", "cashflow/cash flow")
  appears in the student's text for this sub-question, that's a strong signal
  the criterion is at best a HALF, not a FULL. Use this as a tiebreaker — don't
  treat keywords as a hard match filter.

Determine marks per point:
• marking_rule present → follow it exactly (e.g. "1 mark per paragraph, max 4" = 1 mark each, stop at 4).
• "1 per para" / "one each" / "one per para" → 1 mark per point.
• "half for each item" → 0.5 marks per point.
• Otherwise → total_marks_available ÷ number of points.
• "max N" → award up to N marks for valid matches. Even 1 valid match earns marks.

SECTION CAPS (CRITICAL):
• Some sub-questions carry a "section_cap" and "parent_section" field.
• section_cap = the MAXIMUM marks that ALL sub-questions with the same parent_section can earn COMBINED.
• Track a running total per parent_section. BEFORE awarding marks to a sub-question in that section, check the running total: only award up to (section_cap − running_total). Once the total reaches section_cap, set marks_awarded = 0 for any remaining sub-questions in that section.
• Sub-questions without section_cap are capped only by their own max_marks.
• Also honour each sub-question's own max_marks (e.g. "maximum_marks": 3) even when it is lower than total_marks_available.

WORKED EXAMPLE — section cap with max_marks:
Suppose 4.2 has parent section_cap = 4, and contains two sub-questions:
   - "4.2 Threats" (max_marks 3, four 1-mark criteria available)
   - "4.2 Response" (no max_marks, five 1-mark criteria available)
If the student matches 3 of the 4 threat criteria → award 3 marks for "4.2 Threats" (its own max).
Running total for 4.2 is now 3. Section cap is 4, so only 1 mark remains.
Even if the student matches 4 of the 5 response criteria → award only 1 mark for "4.2 Response" (cap exhausted).
Tick count must equal awarded marks ÷ 0.5; trim extra correct_points so the count matches.

STEP 2 — COMPARE STUDENT WRITING TO THE MODEL ANSWER, THEN ANNOTATE THE MATCHES

You are NOT checking off criteria in isolation. You are COMPARING the student's
writing to the model answer, finding the specific student WORDS that demonstrate
each scoreable point, and placing ticks on those exact words.

The loop, for each criterion in the rubric:

(1) READ THE CRITERION'S SUBSTANCE.
    The criterion's "description" tells you the concept; its "keywords" list
    highlights its DISTINGUISHING ELEMENT (the specific fact/figure/relationship
    that separates this criterion from every other criterion in the rubric —
    e.g. "five years", "reasonable third party", "FRC ES prohibits for listed
    clients", "expanded review of audit work").

(2) FIND IT IN THE STUDENT'S TEXT, OR DON'T.
    Scan the student answer for words/phrases that EXPRESS the distinguishing
    element — verbatim or as clear paraphrase of the substance, not of the
    topic. If you find a substantive match, you have a credit. If you only
    find a generic gesture toward the topic (no diinguishing element), you
    have NO credit. There is no half-credit here — either the substance is
    there or it isn't.

(3) ANCHOR THE TICK ON THE MATCHING STUDENT WORDS.
    The student's matching phrase becomes the `key_phrase` for the tick. Pick
    3-6 verbatim words from the student's text — the slice that contains (or
    paraphrases) the distinguishing element. THAT is where the tick lands.

STRICTNESS — four rules that prevent over-marking:

A. DISTINGUISHING-ELEMENT REQUIRED — LITERAL OR STEM PRESENCE.
   If the criterion's description or keywords contain a SPECIFIC distinguishing
   word/phrase (a precise term that identifies THIS criterion, not a generic
   topic word), the student's text MUST contain that word/phrase — verbatim,
   pluralised, or as a clear morphological/stem variant of the SAME root.

   Semantic synonyms with DIFFERENT word roots are NOT acceptable. A teacher
   following this rubric expects the specific term to appear; equivalent
   concepts expressed in completely different vocabulary do not earn the mark.

   EARN: "audited for 5 yrs" matches "five years" (stem match: 5/five → years/yrs).
   EARN: "expanded the review" matches "expanded review of audit work"
         (stem match: expanded → expanded).
   EARN: "FRC ES prohibits payroll for listed clients" matches "FRC Ethical
         Standard expressly prohibits payroll services for listed clients"
         (literal match on FRC, prohibits, listed).
   ZERO: "external quality review" does NOT match "expanded review of audit
         work" — "external" is a different root from "expanded"; the
         distinguishing element ("expanded"/"audit work"/"discrepancies") is
         absent. Different words, different mark.
   ZERO: "outside observer" does NOT match "reasonable and informed third
         party" — different root words; the distinguishing phrase "third party"
         is absent.
   ZERO: "there is a familiarity threat as she has known the client a while"
         does NOT match "five-year continuous audit partner" — no five-year
         or continuity word present.
   ZERO: "the firm shouldn't do payroll" — no FRC / listed element.

   Rule of thumb: scan the criterion's keywords list. Does ANY keyword (or its
   obvious stem like `expand-`, `prohibit-`, `disclos-`, `scept-`) appear in
   the student's text? If NO keyword has even a stem-match in the student
   text, the criterion is NOT matched — even if a generic synonym is used.

   SPACING / HYPHENATION EQUIVALENCE (read this carefully — common false-strict):
   Treat single-word, hyphenated, and space-separated forms of the SAME root as
   matching. The student is not penalised for spacing choices.

   EARN: "cashflow" matches keyword "cash flow" (single word vs two words).
   EARN: "non-compliant" matches keyword "non compliant" (hyphen vs space).
   EARN: "self review" matches keyword "self-review" (space vs hyphen).
   EARN: "userfriendly" matches keyword "user friendly".
   EARN: "TCWG" matches keyword "those charged with governance" (acronym).
   ZERO: "cash position" does NOT match "cash flow" — "flow" stem absent.

   COMMON PARAPHRASE FAILURES (these are real teacher fail-modes — do NOT
   accept them):
   ZERO: "challenge management" does NOT match "insufficiently sceptical of
         financial statements" — different roots ("challenge" ≠ "scept-"),
         and the substantive object ("financial statements") is absent.
   ZERO: "external quality review" ≠ "expanded review of audit work".
   ZERO: "outside observer" ≠ "reasonable and informed third party".
   ZERO: "sign a document to show responsibility" ≠ "monthly comparison
         evidenced by signature" — the signing is for responsibility, NOT
         for the monthly comparison. Different referent.
   ZERO: "informed that expensing outside X will not be reimbursed" ≠ "do
         not process expense claims that bypassed the website" — different
         remedy (employee communication vs back-office process control).
   ZERO: "authorization controls on working papers" ≠ "confidential and
         secure data filing" — different concept (access control vs
         physical filing system).

B. NO DOUBLE-CREDIT FOR ONE CONCEPT.
   Two criteria that describe causally-linked aspects of the SAME concept
   (familiarity → trust → reduced scepticism) collapse to ONE credit when
   the student treats them as one thought.

   ZERO (double-credit): student writes "familiarity threat as she may be too
   trusting and scepticism is impaired" — DO NOT credit BOTH "familiarity
   threat" AND "too trusting/insufficiently sceptical" criteria. Pick the
   single best match. To earn both, the student must develop each point
   SEPARATELY with distinct substantive content.

B'. RESPECT THE LEAF — NO CROSS-LEAF CREDIT FOR THE SAME SENTENCE.
   When the rubric splits a sub-question into leaves with similar/overlapping
   criteria (e.g. 4.3 has BOTH a "Recruitment Threats" leaf and a "Payroll
   Threats" leaf, both with a "management threat" / "too closely aligned"
   criterion), a single student sentence MUST credit criteria in AT MOST
   ONE leaf — the leaf whose CONTEXT matches the sentence.

   Identify the leaf from the student's context cues:
   • The student's own sub-label (e.g. "4.3(A)" → Recruitment leaf,
     "4.3(b)" → Payroll leaf) is the strongest signal — respect it.
   • Substantive content cues — "decision roles", "candidate selection" →
     Recruitment; "reviewing payroll", "payroll information" → Payroll.

   WRONG (cross-leaf double-credit): student writes under 4.3(A): "management
   threat ... Griffin taking on decision roles ... Griffin's goals aligning
   to closely to Yeti's." Credit ONLY the Recruitment Threats leaf's
   management/closely-aligned criterion. DO NOT also credit the Payroll
   Threats leaf's "closely aligned with views and interests of management"
   criterion using the same sentence — that's the same concept in a leaf
   the student is not addressing.

   RIGHT: each leaf earns marks ONLY from sentences whose context matches
   that leaf. If the student wrote nothing about payroll's "closely aligned"
   point, the payroll leaf earns 0 for it — even if a recruitment sentence
   contains matching keywords.

B''. EVIDENCE LOCALITY — NO CROSS-SECTION EVIDENCE LEAKAGE (CRITICAL).
   Evidence supporting any criterion in sub-question 4.X MUST be drawn from
   the student's writing UNDER section 4.X. You are NOT allowed to harvest a
   sentence the student wrote under 4.2 and use it as evidence for a 4.3
   criterion, even if the words happen to match a 4.3 keyword.

   How to determine which student section a sentence belongs to:
   1. Section headings in the student's text — "4.2", "4.3", "(a)", "(b)",
      "A)", "b)" — define a boundary. Every sentence between heading X and
      the next heading belongs to X.
   2. If a sentence appears BEFORE the first heading, attribute it to the
      first sub-question listed in the rubric (usually 4.1(a)).
   3. The student's own paragraph layout matters more than keyword matches —
      a sentence under "4.2" is 4.2's, even if it mentions payroll.

   COMMON LEAK CASES TO AVOID:
   • "should refuse Nicola to continue on this engagement" lives in 4.2 (the
     auditor-rotation question). It is NOT evidence for 4.3's "must refuse
     payroll services" or for any 4.3 criterion.
   • "as an external quality review" lives in 4.2. NOT evidence for 4.3
     or 4.4.
   • "poor management decisions are made re. costs" lives in 4.1(a). NOT
     evidence for 4.3's "management threat" criterion.
   • "authorization controls on audit working papers" lives in 4.4. NOT
     evidence for 4.1.

   Before adding a correct_point with a given `text`, ask: which section of
   the student's answer did this sentence come from? If it isn't the
   sub-question you're currently grading, REJECT the evidence — do not
   award the credit, even if a keyword matches.

C. NO CREDIT FOR LABEL WITHOUT SUBSTANCE.
   "there is a self-interest threat" — naming the threat without explaining
   why it arises → 0. Criterion names by themselves are not assessable points.

   EXTRA EXAMPLES (these are the failure modes we have seen):
   • "There may also be a self-interest threat since the CEO seems keen to
     keep Nicola." → 0 for the self-interest criterion. The reason given
     ("CEO keen to keep Nicola") does NOT match the model answer's reason
     ("Nicola may be reluctant to identify prior-year misstatements that
     could damage her reputation"). Wrong rationale = label only.
   • "management threat" appearing alone without explaining HOW the firm
     becomes aligned → 0. The student must state the mechanism
     (decision-making, candidate selection, alignment with management's
     views) to earn the half-mark.

   To earn a label-bearing half-mark, the student's sentence must contain
   the LABEL token AND a substantive clause that matches the rubric's
   stated reason for that label. If only the label is present, or the
   reason is wrong, award 0.

D. WRONG-ANSWER CONTAMINATION VOIDS THE POINT (AND OFTEN THE WHOLE LEAF).
   If the student includes a CORRECT phrase alongside an INCORRECT conclusion
   on the same point, do not award the point.

   ENFORCEMENT — the "how to address" leaves of 4.3 are the canonical case:
   When the model answer says the firm MUST REFUSE / DECLINE a service (e.g.
   FRC ES prohibition on payroll services for listed clients), and the
   student says the firm CAN ACCEPT WITH SAFEGUARDS (or any variant that
   reverses the directive — "acceptable if fee < X%", "okay with team
   segregation", etc.), you MUST award 0 for the entire "How to address" leaf
   for that service, regardless of how many keywords match. The student has
   misunderstood the rule, not learned it. Do not award even 0.5 for naming
   the safeguard name correctly when the safeguard is the WRONG remedy.

   The corresponding "Threats" leaf is graded separately — the student may
   still earn marks there for correctly identifying threats (self-review,
   management, self-interest) even if their conclusion is wrong. The
   contamination only voids the leaf where the wrong conclusion lives.

E. EXACT-PHRASE INERTIA FOR DISTINGUISHING TERMS.
   Some criteria are anchored on a SPECIFIC LEXICAL TERM the rubric author
   chose deliberately ("expanded review", "FRC Ethical Standard", "five years
   continuously", "reasonable and informed third party", "evidenced by
   signature", "confidential and secure data filing", etc.). For these
   criteria, the student's text should contain a keyword stem variant of the
   SAME ROOT word — otherwise downgrade FULL → HALF (not always zero).

   Apply the FULL/HALF/ZERO mapping for concept-level paraphrase mismatches:
   • "external quality review" vs "expanded review" → HALF (right concept,
     wrong specific term). If the student also fails other parts, → ZERO.
   • "outside observer" vs "reasonable and informed third party" → HALF.
   • "managers should sign acknowledgement of responsibility" vs "monthly
     comparison evidenced by signature" → HALF (signing is happening but
     for the wrong thing). Could go to ZERO if the comparison itself isn't
     described.
   • "authorization controls on working papers" vs "confidential and secure
     data filing" → ZERO (different procedural concept — access control vs
     filing system).
   • Wrong-direction conclusions (Rule D) → ZERO regardless of keyword match.

DUPLICATE OCCURRENCES of the SAME point by the student (repeated for emphasis)
— credit ONLY the first occurrence; on the second, do not output another
correct_point. Optionally add a comment "<anchor> → Marks already given above
for this point. <improvement>."

STEP 3 — PLACE THE TICKS ON THE STUDENT'S MATCHING WORDS

Each correct_point = 0.5 marks = ONE tick (✓) on the annotated PDF, drawn over
the EXACT student words that demonstrated the criterion. The tick is the
visible record of the comparison: "this phrase in your answer matched this
point in the model answer".

Marks → tick count (each tick = 0.5 marks visible on PDF):
• Award FULL marks for a criterion = output (criterion.marks / 0.5) correct_points.
  Examples: 1 mark FULL → 2 ticks. 1.5 marks FULL → 3 ticks. 0.5 mark FULL → 1 tick. 2 marks FULL → 4 ticks.
• Award HALF marks for a criterion = output (criterion.marks / 0.5 / 2) correct_points.
  Examples: 1 mark HALF (= 0.5 awarded) → 1 tick. 1.5 marks HALF (= 0.75 awarded — round to 0.5) → 1 tick. 0.5 mark HALF (= 0.25 awarded — round to 0.5) → 1 tick.
• Award ZERO → no correct_points for that criterion.
• Every correct_point in the output JSON MUST have `marks: 0.5`. The number of
  correct_points × 0.5 must equal the sub-question's marks_awarded.

WHERE THE TICK LANDS (key_phrase selection):
1. The key_phrase MUST be a 5-8 word verbatim slice of the student's sentence
   that captures the WHOLE substantive clause demonstrating the criterion —
   the part a teacher would underline with a red pen. NEVER 1-3 words and
   AVOID 4-word fragments unless the 4 words clearly stand alone as a clause.
   Short anchors render as a tick on a single word in the PDF (e.g. on
   "a", "the", "of") and look like noise. Underline the WHOLE point, not
   just the cue word.

   Examples of GOOD anchor selection (5-8 words, full substantive clause):
   • Criterion: "Adverse impact on cash flow"
     GOOD:  "shortage of cashflow / poor budgeting in future"
     BAD:   "shortage of cashflow" (too short — extend to the predicate)
   • Criterion: "Will not hit stated target of net zero"
     GOOD:  "inability to meet net zero goals and Yeti"
     BAD:   "inability to meet net zero" (cuts off before the consequence)
   • Criterion: "Increased scope for fraud"
     GOOD:  "also a risk of fraud, due to uncontrolled"
     BAD:   "risk of fraud" (3 words — too short)
   • Criterion: "Familiarity (trust) threat"
     GOOD:  "Familiarity threat to objectivity, since Nicola has"
     BAD:   "Familiarity threat" (2 words — too short)
2. The slice must contain the SUBSTANTIVE WORDS that demonstrate the criterion.
   If the criterion is "five-year continuous audit partner" and the student wrote
   "audit partner for 5 yrs", the key_phrase should be the WHOLE clause
   like "Nicola has been on the audit for 5 yrs" (7 words) — the full
   distinguishing clause with subject + predicate, not just "5 yrs".
3. For 1-mark criteria (2 ticks), pick TWO DIFFERENT 5-8 word spans inside the
   same sentence — each must independently be a meaningful clause. Do not split
   one short clause into two even shorter halves.
4. If the criterion has a `keywords` list AND a keyword (or its obvious
   paraphrase) appears in the student text, the key_phrase MUST contain that
   keyword/paraphrase plus enough surrounding context to make a clause.
5. If no keyword appears verbatim (pure paraphrase case), pick the 5-8 word
   slice that most directly demonstrates the criterion's substance.

BAD key_phrases — DO NOT EMIT:
• "a", "the", "to", "of" — articles/prepositions
• 1-3 word fragments that lack a subject or predicate
  – "reviewing payroll" (2 words, no subject — expand to "audit team will be reviewing payroll", 6 words)
  – "closely to Yeti's" (3 words, no subject — expand to "Griffins goals aligning to closely to Yeti's", 7 words)
  – "Griffins goals aligning" (3 words, no completion — expand to "Griffins goals aligning to closely to Yeti's", 7 words)
  – "risk of fraud" (3 words — expand to "also a risk of fraud, due to uncontrolled", 8 words)
  – "Familiarity threat" (2 words — expand to "Familiarity threat to objectivity, since Nicola has", 7 words)

GOOD key_phrases (5-8 words, meaningful clause with subject + predicate):
• "Nicola has been on the audit for 5 yrs" (8 words)
• "audit team will be reviewing payroll" (6 words)
• "Griffin should decline to assist with recruitment" (6 words)
• "decisions on behalf of management" (5 words)
• "shortage of cashflow / poor budgeting in future" (7 words)
• "inability to meet net zero goals" (6 words)

key_phrase mechanics:
• 5-8 words only. Shorter underlines look like ticks on stray articles; longer phrases span PDF lines and cannot be found.
• Verbatim substring of "text" (the student sentence).
• Two key_phrases from the same text must target STRICTLY DIFFERENT word spans — no word may appear in both.

GOOD (criterion keywords were ["shortage of cashflow", "budgeting"]):
   "shortage of cashflow/ poor" + "poor budgeting in future"
BAD overlap (the word "non-compliant" appears in both — REJECT):
   "non-compliant" + "Yeti may been non-compliant"
BAD overlap (same span, no anchor split):
   "shortage of cashflow/ poor budgeting" + "cashflow/ poor budgeting in future"

marks_awarded for the sub-question = count of correct_points × 0.5.
Cap at max_marks, then apply section_cap (Step 1). score = sum of all marks_awarded.

STEP 3.5 — SELF-AUDIT BEFORE RETURNING (do this for every sub-question):
For each sub-question you just built:
1. List the criteria you classified FULL or HALF (from STEP 2 reasoning).
2. Sum: for each FULL → +criterion.marks; for each HALF → +criterion.marks/2.
   That total is expected_marks.
3. Count your correct_points for this sub-question → tick_count.
4. Verify: tick_count == expected_marks / 0.5. If LESS, you under-marked — add
   the missing tick(s) on a different anchor word in the same sentence.
   If MORE, you over-marked — remove ticks until balanced.
5. Apply max_marks and section_cap LAST. Trim ticks to match the capped
   marks_awarded.
6. Final check: marks_awarded must EXACTLY equal len(correct_points) × 0.5.
   If not, fix one or the other before returning.

STEP 3.6 — TEACHER-PASS (theory only — last sanity check before returning):
Read each AWARDED criterion one more time with a teacher's mindset:
  a) Does the student's writing in this sub-question REALLY cover the
     model-answer point, or only gesture at the topic?
  b) Is the conclusion correct (e.g. "must refuse payroll" vs "can accept
     with safeguards")?
  c) For label-bearing points like "self-interest threat" — is the REASON
     the student gives the same as the model answer's reason?
If the answer to any of these is "no", downgrade FULL → HALF or HALF → ZERO,
and remove ticks accordingly. The teacher does NOT credit:
  • "external quality review" for "expanded review of audit work" (different
    word root) — downgrade to HALF at most, often ZERO if accompanied by
    other errors.
  • "sign a document showing responsibility" for "monthly comparison
    evidenced by signature" (signing the wrong thing) — HALF.
  • "self-interest threat" with the wrong reason — ZERO.
  • "can accept payroll with safeguards" for "must refuse payroll" — ZERO
    (entire "How to address" leaf goes to zero per Rule D).

After this teacher-pass, RE-VERIFY the marks_awarded == len(correct_points)
× 0.5 invariant before returning.

STEP 4 — IDENTIFY NOT-REQUIRED (OFF-TOPIC) CONTENT

Students sometimes include content that the question never asked for — definitions
of unrelated concepts, padding, tangents, or material from a different question.
Real teachers mark these areas with "Not required" so the student knows to drop
them in future answers.

For each off-topic sentence/passage in the student's answer:
• Output ONE entry in not_required_points with:
  - "text": the verbatim off-topic sentence/passage from the student answer
  - "key_phrase": a 3-6 word verbatim substring of "text" — the anchor where the
    "Not required" marker will be placed on the PDF
  - "reason": ONE short sentence explaining why this content is off-topic
    (e.g. "Question asks about audit procedures, not internal control design.")

Rules:
• not_required_points carry NO marks. They do NOT reduce marks_awarded.
• Do NOT mark a point as "not required" if it earned credit elsewhere (would never
  appear in correct_points AND not_required_points).
• Borderline / weakly relevant content → leave it out. Only flag CLEARLY off-topic.
• If the student is on-topic throughout, return an empty list.

═══════════════════════════════════════════════════
QUESTION INFORMATION
═══════════════════════════════════════════════════
{questions}

═══════════════════════════════════════════════════
MODEL ANSWER
═══════════════════════════════════════════════════
{model_data}

═══════════════════════════════════════════════════
STUDENT'S ANSWER
═══════════════════════════════════════════════════
{chunks}

═══════════════════════════════════════════════════
COMMENTS
═══════════════════════════════════════════════════
Array of strings, each formatted as:
"[<top_level_sub_question>] <3-5 word verbatim quote from student> → <what was wrong>. <one improvement>."

QUOTA — ONE comment per TOP-LEVEL sub-question where marks were lost.
• For EACH top-level sub-question that lost marks → emit EXACTLY ONE consolidated
  comment. Do not skip a sub-question that lost marks; do not emit two for one.
• If a top-level sub-question is FULLY credited (no marks lost) → omit it entirely.
• Prefix MUST be the TOP-LEVEL sub-question only ("4.1", "4.2", "4.3", "4.4") —
  NEVER a leaf label ("4.1(a) Consequences", "4.3 Payroll Threats" etc.).

ANCHOR REQUIREMENTS — these are how the PDF annotator finds where to place
the comment popup. Getting these wrong puts the comment in `unanchored_comments`
where the student never sees it.

• 3-5 words ONLY. Longer phrases (6+ words) often span PDF lines and the
  exact-match search fails. Pick the SHORTEST distinctive slice you can.
• MUST be a character-for-character verbatim substring of {chunks}. Copy
  exactly — preserve typos ("hsould", "biith", "specipitcal"), preserve
  punctuation, preserve casing. If you paraphrase or "clean up" the spelling,
  the search will not find the anchor in the PDF.
• MUST come from a single line in {chunks} — never join words across lines.
• MUST be from the SAME sub-question as the prefix. A `[4.3]` comment cannot
  anchor on a phrase the student wrote under 4.4.
• If you can't find a distinctive 3-5 word phrase, pick the first 3-4 words
  of the most relevant sentence in that sub-question — but never fabricate.

Other rules:
• No praise-only comments. Do not reveal the model answer.
• Use full form of words in comments, no abbreviations.

Example (good — top-level prefix, one comment for the whole sub-question):
"[4.1] Cost overruns and projects falling behind → Consequences are limited to
delays and customer dissatisfaction; missed work-in-progress overvaluation, breach
of contract, and net-zero/reputational impact. Recommendations also miss progress
reports to senior management and not processing claims that bypassed the website.
Cover the wider commercial and procedural consequences for full marks."

Example (BAD — leaf prefix, multiple comments per sub-question):
"[4.1(a) Consequences] Cost overruns → ..."  ← REJECT, must be "[4.1]"
"[4.1(a) Recommendations] All large variances → ..."  ← REJECT, redundant with above

═══════════════════════════════════════════════════
OUTPUT — return ONLY valid JSON
═══════════════════════════════════════════════════
{{
  "question_number": "<main question number>",
  "score": <sum of all marks_awarded>,
  "total_marks": <max marks for question>,
  "sub_grades": [
    {{
      "sub_question": "<identifier from model answer>",
      "student_label": "<verbatim label from student's answer, or empty string>",
      "marks_awarded": <number>,
      "max_marks": <number>,
      "reason": "<brief explanation>",
      "correct_points": [
        {{
          "text": "<verbatim student sentence>",
          "marks": 0.5,
          "key_phrase": "<3-6 words verbatim from text>"
        }}
      ],
      "not_required_points": [
        {{
          "text": "<verbatim off-topic sentence>",
          "key_phrase": "<3-6 words verbatim from text>",
          "reason": "<one short sentence why this is off-topic>"
        }}
      ]
    }}
  ],
  "comments": ["<quote → issue. improvement.>"]
}}

Constraints:
• One sub_grades entry per model answer sub-question.
• marks_awarded = count of correct_points × 0.5 (capped at max_marks).
• score = sum of all marks_awarded (capped at total_marks).
• Every correct_point has marks: 0.5. Every key_phrase is 3-6 words.
• student_label must be verbatim from {chunks} or empty string.
"""

grade_prompt = ChatPromptTemplate.from_template(GRADE_PROMPT_TEMPLATE)
holistic_grade_prompt = ChatPromptTemplate.from_template(HOLISTIC_GRADE_PROMPT_TEMPLATE)


RESTATEMENT_PROMPT_TEMPLATE = """
You are helping a marker finish annotating an exam script. The marking is DONE
and you must not change any of it.

A marker reads down the page. Every so often they meet a line where the student
makes a point they have ALREADY given marks for somewhere else, and they write
"Marks given above" or "Marks given below" beside it, so the student knows the
point was seen and credited, just not there.

Your job is that read-through. Below is every line of the student's answer that
earned NOTHING. Go through them IN ORDER and decide, for each one: is this the
student making a point that already got marks elsewhere?

You never write "above" or "below" - the direction is worked out afterwards
from the page positions. You only say WHICH line and WHICH point.

═══════════════════════════════════════════════════
THE POINTS THAT WERE CREDITED, GROUPED BY WORKING
═══════════════════════════════════════════════════
{credited}

═══════════════════════════════════════════════════
LINES THAT EARNED NOTHING - DECIDE ABOUT EACH ONE
═══════════════════════════════════════════════════
{unmarked_lines}

═══════════════════════════════════════════════════
HOW TO DECIDE
═══════════════════════════════════════════════════
Flag the line when a marker reading it would think "the student is making that
same point again here". The usual shapes:

  • A figure ANNOUNCED in a sentence and then derived in a working below it.
    "Firstly, the goodwill on acquisition was: 11,725,000." announces the result
    of the goodwill working. The working's credited rows are its components -
    12,000, 3,125, (3,400) - and none of them contains 11,725,000. It is still
    the same point. This is why the working's RESULT is given to you above.
  • A sentence announcing what a table below is about to show.
    "The journal entries should be as follows:" restates the journal beneath it.
  • A total built in one working and carried into a later one.
    "b/f reserves 18,150,000.00" where that figure was credited in the net
    assets working; "add back nci 6,975,000.00" where the NCI working earned
    the marks.
  • A conclusion restated in a summary after the work was done.

Do NOT flag a line when:
  • It is a DIFFERENT step that happens to use the same number. A working that
    produces a figure and a journal that posts it are two different pieces of
    work, each with its own marks - not one point made twice.
  • It merely contains a similar number with no connection to the credited point.
  • It is simply wrong, or an attempt that earned nothing anywhere. A line that
    was never credited at all gets no pointer - there is nothing to point to.

Match on the POINT, not on the digits. Work through the whole list rather than
stopping at the first few; a long script usually has ten to fifteen of these,
and returning only one or two means the list was skimmed. Where you genuinely
cannot tell, leave it out.

═══════════════════════════════════════════════════
OUTPUT FORMAT - return ONLY valid JSON, nothing else
═══════════════════════════════════════════════════
{{
  "restatements": [
    {{
      "criterion_id": "<id of the credited point being repeated>",
      "line": "<the line, copied EXACTLY as it appears in the list above>",
      "why": "<a few words: what makes this the same point>"
    }}
  ]
}}

Copy each line character-for-character, without its leading number. A line that
cannot be matched back to the list is discarded.
"""

restatement_prompt = ChatPromptTemplate.from_template(RESTATEMENT_PROMPT_TEMPLATE)
