from pymilvus import connections, utility, Collection, DataType
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional
from .config import ZILLIZ_URI, ZILLIZ_TOKEN, COLLECTION_NAME, EMBED_DIM

_coll = None

def get_collection():
    global _coll
    if _coll is not None:
        return _coll

    if not ZILLIZ_URI or not ZILLIZ_TOKEN:
        raise RuntimeError("Missing ZILLIZ_URI or ZILLIZ_TOKEN")
    connections.connect(alias="default", uri=ZILLIZ_URI, token=ZILLIZ_TOKEN)

    if not utility.has_collection(COLLECTION_NAME):
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found")

    coll = Collection(COLLECTION_NAME)

    # sanity: vector dim
    vec_field = next((f for f in coll.schema.fields if f.name == "embedding"), None)
    dim_in_coll = getattr(vec_field, "dim", None) or (vec_field.params.get("dim") if hasattr(vec_field, "params") else None)
    if int(dim_in_coll) != int(EMBED_DIM):
        raise RuntimeError(f"Dim mismatch: collection={dim_in_coll}, app={EMBED_DIM}")

    coll.load()
    _coll = coll
    return _coll

def search_vectors(query_vectors, top_k, output_fields=None):
    coll = get_collection()
    res = coll.search(
        data=query_vectors,
        anns_field="embedding",
        param={"metric_type":"COSINE","params":{"ef":64}},
        limit=top_k,
        output_fields=output_fields or ["doc_text","event_hour","prb_id","rdata_trimmed","country_code","anomaly_type","median_rtt_hour","p95_rtt_hour","error_rate_hour","robust_z_rtt"]
    )
    # Milvus returns a list per query; we assume single query
    hits = []
    for h in res[0]:
        fields = {k: h.entity.get(k) for k in (output_fields or [])}
        hits.append({
            "score": float(h.distance),
            **fields
        })
    return hits


def recent_anomalies(n: int = 20, hours: int = 48, *, country_code: Optional[str] = None,
                     rdata_trimmed: Optional[str] = None, anomaly_only: bool = False) -> List[Dict]:
    """
    Fetch up to `n` most recent anomalies within the last `hours`.
    Uses ISO hour strings so lexicographic comparisons work as time comparisons.
    """
    coll = get_collection()

    # Build Milvus boolean expr (note: event_hour is stored as VARCHAR)
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:00:00")
    expr = f'event_hour >= "{cutoff_str}"'
    if country_code:
        expr += f' && country_code == "{country_code}"'
    if rdata_trimmed:
        expr += f' && rdata_trimmed == "{rdata_trimmed}"'
    if anomaly_only:
        expr += ' && anomaly_type != "unknown"'

    # Pull a reasonable upper bound, then sort/trim in app
    rows = coll.query(
        expr=expr,
        output_fields=[
            "event_hour","prb_id","rdata_trimmed","country_code","anomaly_type",
            "median_rtt_hour","p95_rtt_hour","avg_rtt_hour","error_rate_hour","robust_z_rtt","doc_text"
        ],
        limit=5000,  # safe upper bound; tune if needed
    )

    # Sort newest first, then stronger anomalies (higher z, higher error)
    def safe_float(v):
        try:
            return float(v)
        except Exception:
            return 0.0

    rows.sort(
        key=lambda r: (
            r.get("event_hour", ""),
            safe_float(r.get("robust_z_rtt")),
            safe_float(r.get("error_rate_hour")),
        ),
        reverse=True,
    )
    return rows[:n]

