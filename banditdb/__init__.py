from .client import (
    Client,
    NeuralLinUCBConfig,
    ProgressiveConfig,
    Algorithm,
    normalize_context,
    MAX_CONTEXT_MAGNITUDE,
)
from .exceptions import BanditDBError, ConnectionError, TimeoutError, APIError

__version__ = "0.2.0"


def __getattr__(name):
    """
    Load ``banditdb.eval`` on first access rather than at import time.

    ``eval`` imports numpy, which lives under the optional ``[eval]`` extra. Importing
    it eagerly meant a plain ``pip install banditdb-python`` produced a package where
    ``import banditdb`` raised ModuleNotFoundError — the base install was unusable
    unless numpy happened to be present for other reasons.

    Deferring it keeps the base install dependency-light and turns a missing extra
    into an actionable message at the point of use.
    """
    if name == "eval":
        # `import_module`, not `from . import eval`: the latter performs an
        # attribute lookup on this package, which re-enters __getattr__ and
        # recurses until the interpreter gives up.
        import importlib

        try:
            _eval = importlib.import_module(".eval", __name__)
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise ImportError(
                "banditdb.eval requires optional dependencies. "
                "Install them with:  pip install 'banditdb-python[eval]'"
            ) from exc
        globals()["eval"] = _eval
        return _eval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Client
    "Client",
    # Algorithm configs
    "NeuralLinUCBConfig",
    "ProgressiveConfig",
    "Algorithm",
    # Context helpers
    "normalize_context",
    "MAX_CONTEXT_MAGNITUDE",
    # Exceptions
    "BanditDBError",
    "ConnectionError",
    "TimeoutError",
    "APIError",
    # Sub-modules
    "eval",
    # Metadata
    "__version__",
]
