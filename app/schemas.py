from pydantic import BaseModel, Field
from typing import List, Optional, Any

class AskRequest(BaseModel):
    query: str = Field(..., description="Natural language question")
    top_k: int = 20
    return_k: int = 6

class Hit(BaseModel):
    score: float
    event_hour: Optional[str] = None
    prb_id: Optional[int] = None
    rdata_trimmed: Optional[str] = None
    country_code: Optional[str] = None
    anomaly_type: Optional[str] = None
    median_rtt_hour: Optional[float] = None
    p95_rtt_hour: Optional[float] = None
    error_rate_hour: Optional[float] = None
    robust_z_rtt: Optional[float] = None
    doc_text: Optional[str] = None

class AskResponse(BaseModel):
    query: str
    hits: List[Hit]

