# DNS-GPT Backend (FastAPI + Zilliz)

## Local dev
1) Python 3.10+
2) `pip install -r requirements.txt`
3) Copy `.env.example` to `.env` and set ZILLIZ_URI/TOKEN
4) Run: `uvicorn app.main:app --reload --port 8000`

## Integration tests
`pytest -q`

## Deploy
- Any Python host (Render, Fly, EC2). Example:
  - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Set env vars in your host:
  - ZILLIZ_URI, ZILLIZ_TOKEN, COLLECTION_NAME, etc.

## Vercel frontend
- Create a Next.js app on Vercel that calls:
  - `POST https://<your-backend>/ask` with `{ query, top_k, return_k }`
- Restrict CORS in `main.py` to your Vercel domain.

