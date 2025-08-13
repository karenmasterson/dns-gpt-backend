import os
from dotenv import load_dotenv

load_dotenv()

ZILLIZ_URI   = os.getenv("ZILLIZ_URI", "")
ZILLIZ_TOKEN = os.getenv("ZILLIZ_TOKEN", "")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "dns_gpt_anomalies_v1")

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM   = int(os.getenv("EMBED_DIM", "384"))

TOP_K       = int(os.getenv("TOP_K", "20"))
RETURN_K    = int(os.getenv("RETURN_K", "6"))

# Optional LLM re-rank (OpenAI)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # cheap+fast
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"

# Guards
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "300"))
RATE_LIMIT_QPM  = int(os.getenv("RATE_LIMIT_QPM", "30"))   # per IP

