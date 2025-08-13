import time
from collections import defaultdict
from .config import MAX_QUERY_CHARS, RATE_LIMIT_QPM

# in-memory rate limiter (best-effort)
_hits = defaultdict(list)

BANNED_PATTERNS = (
    "drop table", "delete from", "shutdown", "sudo ", "rm -rf",  # silly but illustrative
)

def rate_limit_ok(ip: str) -> bool:
    now = time.time()
    _hits[ip] = [t for t in _hits[ip] if now - t < 60.0]
    if len(_hits[ip]) >= RATE_LIMIT_QPM:
        return False
    _hits[ip].append(now)
    return True

def sanitize_and_check(q: str) -> str:
    q = (q or "").strip()
    if not q:
        raise ValueError("Empty query")
    if len(q) > MAX_QUERY_CHARS:
        raise ValueError(f"Query too long (>{MAX_QUERY_CHARS} chars)")
    lo = q.lower()
    if any(p in lo for p in BANNED_PATTERNS):
        raise ValueError("Query rejected by safety filter")
    return q

