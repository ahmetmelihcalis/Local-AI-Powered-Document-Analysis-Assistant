from .service import (
    DEFAULT_TOP_K,
    MIN_SIMILARITY,
    QuestionScope,
    RetrievedChunk,
    classify_question_scope,
    is_comparison_question,
    retrieve_relevant_chunks,
)
from .expansion import (
    focused_clause_indices as _focused_clause_indices,
    paragraph_anchor_index as _paragraph_anchor_index,
)
from .scoring import (
    definition_scores as _definition_scores,
    mentions_multiple_documents,
    question_type_adjustment as _question_type_adjustment,
)


def __getattr__(name: str):
    """Keep internal test and integration imports working during the package split."""
    from . import service

    return getattr(service, name)
