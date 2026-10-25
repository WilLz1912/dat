"""Web server localhost cho ứng dụng phân tích review Steam.

Chạy (từ thư mục DAT):  python app/server.py   → mở http://localhost:8050  (đổi cổng: python app/server.py 9000)
Tài liệu API tự sinh:    http://localhost:8050/docs
"""
import io
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
MAX_BATCH = 5000

app = FastAPI(title="Steam Review Analyzer", description="API cho 2 model Keras + gợi ý game (môn DAT)")


class ReviewIn(BaseModel):
    text: str
    voted_up: bool | None = None
    playtime: float | None = None


@app.post("/api/analyze")
def analyze(r: ReviewIn):
    if not r.text.strip():
        raise HTTPException(400, "Review trống")
    return core.score_reviews([r.text], [r.voted_up], [r.playtime])[0]


def _parse_bool(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "recommended", "positive", "x"):
        return True
    if s in ("0", "false", "no", "n", "not recommended", "negative"):
        return False
    return None


@app.post("/api/batch")
def batch(csv_text: str = Body(..., media_type="text/plain")):
    """Nhận nội dung file CSV. Cột text bắt buộc (review_text / text / review / text_clean);
    cột tuỳ chọn: voted_up (recommend), playtime (giờ chơi)."""
    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise HTTPException(400, f"Không đọc được CSV: {e}")
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = next((cols[c] for c in ("review_text", "text", "review", "text_clean", "content") if c in cols), None)
    if tcol is None:
        raise HTTPException(400, "CSV cần một cột text: review_text, text, review hoặc text_clean")
    vcol = next((cols[c] for c in ("voted_up", "recommend", "recommended", "label", "sentiment") if c in cols), None)
    pcol = next((cols[c] for c in ("playtime", "playtime_forever_hours", "playtime_hours", "hours") if c in cols), None)
    df = df.dropna(subset=[tcol]).head(MAX_BATCH)
    voted = df[vcol].map(_parse_bool).tolist() if vcol else None
    pt = pd.to_numeric(df[pcol], errors="coerce").tolist() if pcol else None
    return _batch_result(df[tcol].astype(str).tolist(), voted, pt)


@app.get("/api/batch/sample")
def batch_sample(game: str):
    texts, voted, pt = core.game_reviews(game)
    if not texts:
        raise HTTPException(404, "Không có game này")
    return _batch_result(texts, voted, pt)


def _batch_result(texts, voted, pt):
    rows = core.score_reviews(texts, voted, pt)
    return {"summary": core.summarize(rows), "rows": rows}


@app.get("/api/games")
def games():
    return core.game_names()


@lru_cache
def _overview():
    return core.dataset_overview()


@app.get("/api/overview")
def overview():
    return _overview()


@app.get("/api/groups")
def groups():
    return core.game_groups()


@app.get("/api/recommend")
def recommend(app_id: int, liked: bool = True):
    try:
        return core.recommend(app_id, liked)
    except IndexError:
        raise HTTPException(404, "Không có app_id này")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8050
    print("Đang nạp model ...")
    core.models()
    print(f"Mở http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port)
