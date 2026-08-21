"""Reusable retrieval-context guardrails for the voiseAI backend.

This module provides a lightweight, dependency-free guard layer that sits
between a retrieval step and the answer-generation (LLM) step.  Import it
from any router or service that needs to assert that at least one relevant
document was found before proceeding.

Typical usage::

    from backend.guardrails import ensure_context

    results = await retriever.search(query)
    results = ensure_context(results)   # raises NoContextError if empty
    answer  = await llm.generate(query, context=results)
"""

from __future__ import annotations

from typing import Sequence, TypeVar

from backend.errors import AppError

# ---------------------------------------------------------------------------
# TypeVar — keeps ensure_context generic so it works with any result type
# (plain dicts, Pydantic models, dataclasses, …) without importing schemas.
# ---------------------------------------------------------------------------
T = TypeVar("T")


class NoContextError(AppError):
    """Raised when a retrieval step returns no usable results.

    Signals to the caller (typically a router) that the system cannot
    generate a grounded answer because no relevant context was found.
    The error maps to HTTP 422 so that API clients can distinguish it from
    a generic server failure.

    Raise with no arguments::

        raise NoContextError()
    """

    def __init__(self) -> None:
        super().__init__(
            message="I don't have enough information to answer that question.",
            status_code=422,
            error_code="no_context",
        )


def ensure_context(results: Sequence[T]) -> Sequence[T]:
    """Assert that *results* contains at least one item, then return it unchanged.

    Call this immediately after a retrieval step and before passing results
    to the LLM / answer-generation step.  If the retrieval returned nothing,
    :class:`NoContextError` is raised so that the router can surface a
    meaningful 422 response to the client instead of forwarding an empty
    context to the model.

    Args:
        results: The sequence of retrieval results (any element type).

    Returns:
        The same *results* object, unmodified, when it is non-empty.

    Raises:
        NoContextError: If *results* is empty (length zero / falsy).

    Example::

        docs = await vector_store.search(query, top_k=5)
        docs = ensure_context(docs)   # safe to use docs below this line
    """
    if not results:
        raise NoContextError()
    return results
