"""
Shared fixtures and test helpers for the BanditDB Python SDK test suite.

Architecture
------------
All HTTP calls are intercepted by replacing ``client.session`` with a
``MagicMock`` *after* construction.  This lets us test the real ``__init__``
logic (URL normalisation, header injection, retry adapter mounting) while
keeping every other test fully isolated from the network.
"""

from unittest.mock import MagicMock

import pytest

from banditdb import Client

# ── Constants used across test modules ────────────────────────────────────────

BASE_URL = "http://test.banditdb.internal:8080"
API_KEY  = "test-secret-key-abc123"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_response(status_code: int, json_body=None, text: str = "") -> MagicMock:
    """
    Build a minimal mock that mimics ``requests.Response``.

    Parameters
    ----------
    status_code:
        The HTTP status code the mock will report via ``.status_code``.
    json_body:
        The decoded value returned by ``.json()``.  Pass ``None`` when the
        response body is irrelevant to the test being written.
    text:
        The raw text returned by ``.text``, used in error-message assertions.
        When omitted, falls back to ``str(json_body)`` so the attribute is
        never an unresolved ``MagicMock``.
    """
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body
    resp.text = text if text else str(json_body)
    return resp


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def client() -> Client:
    """
    A ``Client`` instance wired to a stable test URL with its HTTP session
    replaced by a ``MagicMock``.

    Individual tests configure the mock as needed before exercising the method
    under test::

        client.session.get.return_value    = make_response(200, ...)
        client.session.post.return_value   = make_response(200, ...)
        client.session.delete.return_value = make_response(200, ...)

    No live BanditDB server or network access is required.
    """
    c = Client(url=BASE_URL, api_key=API_KEY, timeout=2.0)
    c.session = MagicMock()
    return c
