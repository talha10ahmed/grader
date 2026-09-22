from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class GradeBreakdownItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    criterion_id: Optional[str] = Field(
        None,
        description="The criterion_id copied EXACTLY from the marking criteria (e.g. 'TB03'). Always include it when the criterion has one."
    )
    criterion: str = Field(..., description="Marking criterion or point description")
    marks_awarded: Union[float, int] = Field(..., description="Marks given")
    max_possible: Union[float, int] = Field(..., description="Maximum marks for this item")
    reason: str = Field(..., description="Explanation for awarded marks")
    # Keep the historical string field for compatibility, but also preserve the
    # original list of evidence snippets for reliable PDF anchoring.
    evidence: Optional[str] = Field(None, description="Relevant student text snippet(s), possibly joined")
    evidence_list: Optional[List[str]] = Field(
        default=None,
        description="Optional list of 1-3 verbatim evidence snippets from the student answer"
    )
    restated_at: Optional[List[str]] = Field(
        default=None,
        description=(
            "Verbatim student lines where this SAME point is stated a second time but earns "
            "nothing extra, because the marks were given at the `evidence` line instead. "
            "Each becomes a 'Marks given above/below' note beside that line. Leave empty "
            "unless the student genuinely repeats the point somewhere else."
        )
    )
    comments_summary: Optional[str] = Field(None, description="Grader comment for this specific item (if any)")
    criterion_focus: Optional[str] = Field(
        default=None,
        description=(
            "Short phrase (<= 8 words) naming what THIS criterion tests, taken from the "
            "wording BEFORE 'DISAMBIGUATION'/'CONTEXT' in its description - e.g. "
            "'revaluation gain to OCI', 'goodwill at closing rate'. Used to verify the "
            "criterion_id you returned is the one you actually graded. Two criteria in a "
            "section often quote the SAME figure in opposite roles (a gain earned vs the "
            "same amount later eliminated), so name the ROLE, not just the number."
        )
    )

    # Holistic grading fields (only populated when holistic_grading=True on the parent doc)
    sub_question: Optional[str] = Field(None, alias="_sub_question", description="Sub-question identifier (holistic grading)")
    student_label: Optional[str] = Field(None, alias="_student_label", description="Student's label for this sub-question (holistic grading)")
    correct_points_with_marks: Optional[List[dict]] = Field(None, alias="_correct_points_with_marks", description="Per-point marks for holistic annotation [{text, marks}]")

    # Aggregate-recovery merge fields (only populated on entries produced by
    # _merge_aggregate_recovered_entries — one entry per aggregate OF group
    # instead of one entry per component criterion). Preserves full component
    # traceability without cluttering annotations.
    components: Optional[List[dict]] = Field(
        default=None,
        description="Per-component sub-marks that make up a merged aggregate entry. Each entry: {criterion_description, component_marks, max_possible}."
    )
    merged_from_aggregate: Optional[bool] = Field(
        default=None, alias="_merged_from_aggregate",
        description="True when this entry was produced by collapsing multiple aggregate-recovered components onto a single student-written line."
    )
    merged_aggregate_of: Optional[str] = Field(
        default=None, alias="_merged_aggregate_of",
        description="The aggregate OF id (e.g. 'OF2') this merged entry represents."
    )
    target_value: Optional[int] = Field(
        default=None, alias="_target_value",
        description="The specific numeric value in the student's writing that this mark belongs to (for annotator to narrow underline+score placement to this exact cell). Populated from rubric of_value / of_produces (model side) — no unit guessing by annotator."
    )
    target_value_variants: Optional[List[str]] = Field(
        default=None, alias="_target_value_variants",
        description="Surface-form variants of _target_value (raw, comma-grouped, ×1000 form, '.00' suffix, 'm' millions) — supplied by model side so annotator can find the value on the PDF without unit assumptions."
    )
    column_header: Optional[str] = Field(
        default=None, alias="_column_header",
        description="Column-header text (exact substring from student's PDF, e.g. 'acq date') that disambiguates WHICH column the target value belongs to when the same value appears in multiple columns of the row. Set by the LLM per the SHORTEST SUFFIX RULE in the prompt."
    )


class StudentGradeDocument(BaseModel):
    student_id: str = Field(..., description="Student name or ID")
    question_number: str = Field(..., description="Question number (e.g. '1', '4.2')")
    total_marks_awarded: Union[float, int] = Field(..., description="Total marks awarded")
    total_max_possible: Union[float, int] = Field(..., description="Total possible marks")
    overall_reason: str = Field(..., description="Summary reason for total score")

    breakdown: List[GradeBreakdownItem] = Field(
        ..., description="Detailed per-criterion breakdown"
    )

    comments: List[str] = Field(
        default_factory=list,
        description="Structured list of LLM-generated feedback comments for the whole question"
    )

    # Notes that are useful for debugging or reporting but are not suitable for PDF anchoring.
    guardrail_warnings: List[str] = Field(
        default_factory=list,
        description="Warnings generated by post-processing guardrails (not intended for PDF annotation)"
    )
    unanchored_comments: List[str] = Field(
        default_factory=list,
        description="LLM comments that did not meet the strict annotation-friendly format"
    )

    # Off-topic content flagged by the LLM. Carries no marks impact but is annotated
    # on the PDF with a strikethrough + 'Not required' margin label so the student
    # knows to drop the content in future answers. Holistic mode stores the same
    # information per sub-question on each breakdown item (under `_not_required_points`).
    not_required_points: List[dict] = Field(
        default_factory=list,
        description="Verbatim off-topic snippets {text, key_phrase, reason} for PDF annotation"
    )

    # Holistic grading flag
    holistic_grading: Optional[bool] = Field(None, description="True when holistic grading mode was used (no marking criteria)")

    # References & Metadata
    extracted_at: str = Field(..., description="ISO timestamp of grading")
    question_id: Optional[str] = Field(None, description="pac_questions _id")
    model_answer_id: Optional[str] = Field(None, description="model_answers _id")
    student_answer_id: Optional[str] = Field(None, description="student_assignments _id")


class RestatementItem(BaseModel):
    """One student line that repeats an already-credited point."""
    model_config = ConfigDict(populate_by_name=True)

    criterion_id: str = Field(..., description="Id of the credited point being repeated")
    line: str = Field(..., description="The student's line, verbatim")
    why: Optional[str] = Field(None, description="Short note on why it is the same point")


class RestatementResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    restatements: List[RestatementItem] = Field(default_factory=list)
