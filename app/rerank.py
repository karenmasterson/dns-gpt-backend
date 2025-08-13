import os, json, httpx
from typing import List, Dict
from .config import OPENAI_API_KEY, OPENAI_MODEL, RERANK_ENABLED

PROMPT = """You are re-ranking DNS anomaly snippets for relevance to the user's query.

User query:
{query}

For each candidate, you will receive a JSON with fields:
- score: cosine similarity (higher is better)
- doc_text: the anomaly text
- metadata: event_hour, rdata_trimmed, country_code, anomaly_type

Return a JSON array of the TOP {k} items with fields:
- idx: index in the provided list
- final: a float score between 0 and 1 (your relevance)
Think briefly; prefer precise temporal/location matches, concrete anomaly reasons, and consistent metrics."""

async def rerank_async(query: str, candidates: List[Dict], k: int) -> List[int]:
    """Return indices of top-k items. Falls back to vector-only if no OPENAI key."""
    if not RERANK_ENABLED or not OPENAI_API_KEY:
        # simple fallback: keep original order (vector score)
        return list(range(min(k, len(candidates))))

    # Build compact prompt with truncated docs
    items = []
    for i, c in enumerate(candidates):
        items.append({
            "idx": i,
            "score": round(float(c["score"]), 4),
            "doc_text": (c.get("doc_text") or "")[:800],
            "event_hour": c.get("event_hour"),
            "rdata_trimmed": c.get("rdata_trimmed"),
            "country_code": c.get("country_code"),
            "anomaly_type": c.get("anomaly_type"),
        })

    sys_prompt = PROMPT.format(query=query, k=k)
    content = f"{sys_prompt}\n\nCANDIDATES:\n{json.dumps(items, ensure_ascii=False)}"

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    # Use responses API (chat) to get JSON tool output; keep simple with text parsing
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": OPENAI_MODEL,
                "messages": [{"role":"user","content":content}],
                "temperature": 0.0,
            },
        )
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
    # Try to parse indices from JSON; fallback: keep vector order
    try:
        data = json.loads(txt)
        order = [int(d["idx"]) for d in data][:k]
        return order
    except Exception:
        return list(range(min(k, len(candidates))))

