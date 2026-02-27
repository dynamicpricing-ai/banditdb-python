class BanditDBError(Exception):
    """Base exception for all BanditDB errors."""
    pass

class ConnectionError(BanditDBError):
    """Raised when the database cannot be reached."""
    pass

class TimeoutError(BanditDBError):
    """Raised when the database takes too long to respond."""
    pass

class APIError(BanditDBError):
    """Raised when the database returns an error (e.g., Campaign Not Found)."""
    pass
