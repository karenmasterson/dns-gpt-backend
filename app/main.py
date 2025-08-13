# app/main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from .schemas import AskRequest, AskResponse, Hit
from .guards import rate_limit_ok, sanitize_and_check
from .embeddings import embed_texts
from .milvus_client import get_collection, search_vectors
from .config import TOP_K, RETURN_K

app = FastAPI(title="DNS-GPT Backend", version="0.1.0")

# NOTE: The URL you shared (vercel.com/karen-mastersons-projects) is the dashboard.
# Replace ALLOWED_ORIGINS with your actual deployed app URL, e.g.:
#   https://dns-gpt.vercel.app
ALLOWED_ORIGINS = [
    "https://karen-mastersons-projects.vercel.app",   # update once you have the app URL
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Startup: verify Milvus connectivity & collection availability ---
@app.on_event("startup")
async def startup_event():
    try:
        coll = get_collection()  # connects, loads, and validates dim in milvus_client
        # Touch a light stat to ensure the handle works
        _ = coll.num_entities
        print(f"✅ Milvus ready. Collection='{coll.name}', entities={coll.num_entities}")
    except Exception as e:
        # Crash startup so deployment doesn't go live in a broken state
        raise RuntimeError(f"Milvus connection failed at startup: {e}")

@app.get("/")
def root():
    return {"ok": True, "service": "dns-gpt-backend"}

@app.get("/health")
def health():
    # App process is up
    return {"ok": True}

@app.get("/ready")
def ready():
    """
    Liveness + dependency readiness: verifies Milvus responds *now*.
    Returns 503 if the collection can't be reached.
    """
    try:
        coll = get_collection()
        # very light live check: ask for num_entities (no search cost)
        entities = coll.num_entities
        return {"ok": True, "collection": coll.name, "entities": int(entities)}
    except Exception as e:
        # 503 so load balancers can pull it out of rotation temporarily
        raise HTTPException(status_code=503, detail=f"Milvus not ready: {e}")

@app.post("/search", response_model=AskResponse)
async def search(req: Request, body: AskRequest):
    ip = req.client.host if req.client else "unknown"
    if not rate_limit_ok(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    q = sanitize_and_check(body.query)
    top_k = max(1, min(body.top_k, 50))
    return_k = max(1, min(body.return_k, top_k))

    qv = embed_texts([q])
    candidates = search_vectors(qv.tolist(), top_k, output_fields=[
        "doc_text", "event_hour", "prb_id", "rdata_trimmed",
        "country_code", "anomaly_type", "median_rtt_hour",
        "p95_rtt_hour", "error_rate_hour", "robust_z_rtt"
    ])

    # optional re-rank (imports lazily to speed cold start)
    from .rerank import rerank_async
    order = await rerank_async(q, candidates, k=return_k)

    hits: List[Hit] = []
    for idx in order:
        c = candidates[idx]
        hits.append(Hit(**c))

    return AskResponse(query=q, hits=hits)

# Convenience alias for UI
@app.post("/ask", response_model=AskResponse)
async def ask(req: Request, body: AskRequest):
    return await search(req, body)
