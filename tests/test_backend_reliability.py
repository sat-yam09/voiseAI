"""Tests for Member 3 reliability features.

Covers:
- backend.settings  (defaults)
- backend.errors    (AppError)
- backend.guardrails (NoContextError, ensure_context)
- backend.schemas   (RetrieveRequest validation, Pydantic v2)
- backend.routers.retrieve.run_with_retry (retry logic, backoff, timeout propagation)

Isolation strategy
------------------
Heavy ML deps (faiss, src.retrieval, backend.pipeline …) are temporarily
inserted into sys.modules only while this file's tests run, then fully removed
so they cannot bleed into other test files that import the real packages.

No FastAPI TestClient, no FAISS, no model weights, no real pipeline.
pytest-asyncio is NOT installed; async tests use asyncio.run() directly.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import types
import os

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Names of modules we stub so heavy ML deps are not required
# ---------------------------------------------------------------------------
_STUB_NAMES = [
    "src", "src.retrieval", "backend.pipeline",
    "faiss", "sentence_transformers", "datasets", "rank_bm25",
]

# Backend modules imported using the stubs — must be evicted on teardown
_BACKEND_MODULES = [
    "backend.schemas",
    "backend.routers",
    "backend.routers.retrieve",
]


@pytest.fixture(scope="module", autouse=True)
def _stub_heavy_deps():
    """Install stubs before any test in this module, clean up after all finish.

    scope="module" ensures cleanup happens before other test files are run,
    so src.retrieval (and friends) are re-importable with their real
    implementations by test_retrieval.py, test_vector_store.py etc.
    """
    # 1. Save what was already in sys.modules
    saved = {name: sys.modules.get(name) for name in _STUB_NAMES + _BACKEND_MODULES}

    # 2. Install lightweight stubs
    for name in _STUB_NAMES:
        mod = types.ModuleType(name)
        sys.modules[name] = mod

    # Set minimal attributes the backend imports need
    sys.modules["src.retrieval"].RetrievalPipeline = object   # type: ignore[attr-defined]
    sys.modules["backend.pipeline"].get_pipeline = lambda: None  # type: ignore[attr-defined]

    # 3. Force-reload backend modules so they bind against our stubs
    for name in _BACKEND_MODULES:
        sys.modules.pop(name, None)

    yield  # ← tests run here

    # 4. Restore / remove everything we touched
    for name in _STUB_NAMES + _BACKEND_MODULES:
        if saved[name] is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = saved[name]


# ---------------------------------------------------------------------------
# Lazy imports — resolved after the fixture has installed the stubs.
# We use module-level variables populated in a session-autouse fixture so
# that the imports happen after _stub_heavy_deps() has run.
# ---------------------------------------------------------------------------
settings = None
AppError = None
NoContextError = None
ensure_context = None
RetrieveRequest = None
run_with_retry = None
ValidationError = None


@pytest.fixture(scope="module", autouse=True)
def _import_modules(_stub_heavy_deps):  # noqa: F811  depends on stub fixture
    """Import backend modules after stubs are installed."""
    global settings, AppError, NoContextError, ensure_context
    global RetrieveRequest, run_with_retry, ValidationError

    from backend.settings import settings as _settings
    from backend.errors import AppError as _AppError
    from backend.guardrails import NoContextError as _NCE, ensure_context as _ec
    from backend.schemas import RetrieveRequest as _RR
    from backend.routers.retrieve import run_with_retry as _rwr
    from pydantic import ValidationError as _VE

    settings = _settings
    AppError = _AppError
    NoContextError = _NCE
    ensure_context = _ec
    RetrieveRequest = _RR
    run_with_retry = _rwr
    ValidationError = _VE


# ===========================================================================
# 1. Settings defaults
# ===========================================================================

class TestSettingsDefaults:
    def test_max_query_length(self):
        assert settings.max_query_length == 5000

    def test_request_timeout_seconds(self):
        assert settings.request_timeout_seconds == 30.0

    def test_max_retries(self):
        assert settings.max_retries == 3

    def test_retry_base_delay_seconds(self):
        assert settings.retry_base_delay_seconds == 1.0


# ===========================================================================
# 2. RetrieveRequest validation (Pydantic v2)
# ===========================================================================

class TestRetrieveRequestValidation:
    def _make(self, **kwargs):
        return RetrieveRequest.model_validate(kwargs)

    def test_valid_query_and_top_k(self):
        req = self._make(query="hello world", top_k=5)
        assert req.query == "hello world"
        assert req.top_k == 5

    def test_top_k_defaults_to_5(self):
        req = self._make(query="test")
        assert req.top_k == 5

    def test_top_k_lower_boundary_valid(self):
        assert self._make(query="test", top_k=1).top_k == 1

    def test_top_k_upper_boundary_valid(self):
        assert self._make(query="test", top_k=50).top_k == 50

    def test_query_exactly_max_length_valid(self):
        q = "a" * settings.max_query_length
        assert len(self._make(query=q).query) == settings.max_query_length

    def test_empty_query_raises(self):
        with pytest.raises(ValidationError):
            self._make(query="")

    def test_query_too_long_raises(self):
        with pytest.raises(ValidationError):
            self._make(query="a" * (settings.max_query_length + 1))

    def test_top_k_zero_raises(self):
        with pytest.raises(ValidationError):
            self._make(query="test", top_k=0)

    def test_top_k_negative_raises(self):
        with pytest.raises(ValidationError):
            self._make(query="test", top_k=-1)

    def test_top_k_above_max_raises(self):
        with pytest.raises(ValidationError):
            self._make(query="test", top_k=51)


@pytest.mark.parametrize("top_k", [0, -1, -100, 51, 100])
def test_retrieve_request_invalid_top_k_parametrized(top_k):
    with pytest.raises(ValidationError):
        RetrieveRequest.model_validate({"query": "test", "top_k": top_k})


# ===========================================================================
# 3. Guardrails
# ===========================================================================

class TestEnsureContext:
    def test_non_empty_returns_same_object(self):
        data = [{"text": "hello"}, {"text": "world"}]
        assert ensure_context(data) is data

    def test_non_empty_returns_equal_content(self):
        data = [1, 2, 3]
        assert ensure_context(data) == data

    def test_empty_raises_no_context_error(self):
        with pytest.raises(NoContextError):
            ensure_context([])

    def test_no_context_error_status_code(self):
        with pytest.raises(NoContextError) as exc_info:
            ensure_context([])
        assert exc_info.value.status_code == 422

    def test_no_context_error_error_code(self):
        with pytest.raises(NoContextError) as exc_info:
            ensure_context([])
        assert exc_info.value.error_code == "no_context"

    def test_no_context_error_is_app_error(self):
        assert issubclass(NoContextError, AppError)


# ===========================================================================
# 4. AppError
# ===========================================================================

class TestAppError:
    def test_default_status_code(self):
        assert AppError(message="oops").status_code == 500

    def test_default_error_code(self):
        assert AppError(message="oops").error_code == "internal_error"

    def test_default_message_stored(self):
        assert AppError(message="something broke").message == "something broke"

    def test_custom_status_code(self):
        assert AppError(message="gone", status_code=410, error_code="gone_error").status_code == 410

    def test_custom_error_code(self):
        assert AppError(message="gone", status_code=410, error_code="gone_error").error_code == "gone_error"

    def test_custom_message(self):
        assert AppError(message="custom", status_code=409, error_code="conflict").message == "custom"

    def test_is_exception(self):
        assert isinstance(AppError("test"), Exception)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(AppError) as exc_info:
            raise AppError("boom", status_code=503, error_code="unavailable")
        assert exc_info.value.status_code == 503
        assert exc_info.value.error_code == "unavailable"


# ===========================================================================
# 5. run_with_retry  (no pytest-asyncio; uses asyncio.run())
# ===========================================================================

class TestRunWithRetry:
    """All async tests driven via asyncio.run() — no pytest-asyncio needed."""

    @staticmethod
    def _patch_sleep(monkeypatch):
        """Replace asyncio.sleep in the retrieve module with an instant no-op."""
        async def _fast_sleep(*args, **kwargs):
            return None
        import backend.routers.retrieve as _retrieve_mod
        monkeypatch.setattr(_retrieve_mod.asyncio, "sleep", _fast_sleep)

    def test_immediate_success_returns_result(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        assert asyncio.run(run_with_retry(fn)) == "ok"
        assert call_count == 1

    def test_immediate_success_called_exactly_once(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        calls = []

        async def fn():
            calls.append(1)
            return 42

        asyncio.run(run_with_retry(fn))
        assert len(calls) == 1

    def test_eventual_success_after_transient_failures(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 3)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("transient")
            return "recovered"

        assert asyncio.run(run_with_retry(fn)) == "recovered"
        assert call_count == 3

    def test_exhaustion_reraises_final_exception(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 2)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"fail #{call_count}")

        with pytest.raises(ValueError) as exc_info:
            asyncio.run(run_with_retry(fn))

        assert call_count == settings.max_retries + 1
        assert "fail #3" in str(exc_info.value)

    def test_exhaustion_call_count_matches_max_retries_plus_one(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 2)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always")

        with pytest.raises(RuntimeError):
            asyncio.run(run_with_retry(fn))

        assert call_count == settings.max_retries + 1

    def test_timeout_error_propagates_immediately(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 3)
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            raise asyncio.TimeoutError()

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(run_with_retry(fn))

        assert call_count == 1

    def test_sleep_is_actually_mocked(self, monkeypatch):
        import time
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 1)
        monkeypatch.setattr(settings, "retry_base_delay_seconds", 60.0)

        async def fn():
            raise RuntimeError("force retry")

        start = time.monotonic()
        with pytest.raises(RuntimeError):
            asyncio.run(run_with_retry(fn))
        assert time.monotonic() - start < 2.0

    def test_exception_not_wrapped(self, monkeypatch):
        self._patch_sleep(monkeypatch)
        monkeypatch.setattr(settings, "max_retries", 1)

        class _CustomError(Exception):
            pass

        async def fn():
            raise _CustomError("raw")

        with pytest.raises(_CustomError):
            asyncio.run(run_with_retry(fn))
