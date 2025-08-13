from sentence_transformers import SentenceTransformer
from .config import EMBED_MODEL

_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed_texts(texts):
    m = get_model()
    vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vecs

