"""Logic của ứng dụng: làm sạch text, tạo feature, gọi 2 model Keras, gợi ý game.

Làm sạch và feature chép đúng quy tắc trong preprocessing.ipynb §2 và model.ipynb §1,
để review mới đi qua cùng đường ống với dữ liệu lúc train.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS = ROOT / "data", ROOT / "models"

# ---------- làm sạch (preprocessing.ipynb §2.2–2.5) ----------
BB_TAGS = r"h1|h2|h3|b|i|u|s|strike|spoiler|noparse|hr|list|olist|\*|quote|code|table|tr|td|th|img|url|previewyoutube"
BB_RE = re.compile(rf"\[/?(?:{BB_TAGS})(?:=[^\]]*)?\]", re.I)
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\b[\w.-]+\.(?:com|net|org|gg|io|ly|be)/\S*", re.I)
CENSOR_RE = re.compile(r"[\w♥]*♥{2,}[\w♥]*")
EMOJI_CLASS = "\U0001F000-\U0001FAFF☀-☉☓-➿⭐⭕⬛⬜⤴⤵〰〽"
EMOJI_RE = re.compile(f"[{EMOJI_CLASS}](?:[️‍\U0001F3FB-\U0001F3FF]*[{EMOJI_CLASS}]?)*")
TRANS = str.maketrans({"‘": "'", "’": "'", "´": "'", "“": '"', "”": '"',
                       "—": " - ", "–": "-", "‑": "-", "…": "...", " ": " "})
SHORT_WORDS, SPAM_CONSEC, SPAM_MIN_TOKENS, SPAM_UNIQUE_RATIO = 5, 5, 10, 0.3

try:
    from langdetect import DetectorFactory, detect_langs
    DetectorFactory.seed = 0
except ImportError:          # không có langdetect thì chỉ dùng tỉ lệ chữ non-Latin
    detect_langs = None


def clean_text(s):
    s = URL_RE.sub(" ", BB_RE.sub(" ", str(s)))
    has_censored = bool(CENSOR_RE.search(s))
    s = CENSOR_RE.sub(" <censored> ", s)
    s = unicodedata.normalize("NFKC", s).translate(TRANS)
    s = re.sub("[̵-̸]", "", s)
    s = "".join(ch for ch in s if ch == "‍" or unicodedata.category(ch) not in ("Cc", "Cf") or ch in "\n\t")
    s = EMOJI_RE.sub(lambda m: f" {m.group(0)} ", s)
    return re.sub(r"\s+", " ", s).strip(), has_censored


def rule_flags(text):
    """Cờ chất lượng theo luật. Trả về danh sách lý do (rỗng = không dính cờ)."""
    reasons = []
    words = text.split()
    letters = [ch for ch in text if ch.isalpha()]
    non_latin = sum(not unicodedata.name(ch, "").startswith("LATIN") for ch in letters)
    lang = "en"
    if detect_langs and letters:
        try:
            lang = detect_langs(text)[0].lang
        except Exception:
            pass
    # langdetect đoán sai với text ngắn / lặp từ → chỉ tin khi có đủ từ khác nhau
    if letters and (non_latin / len(letters) > 0.3 or (lang != "en" and len(set(words)) >= 5)):
        reasons.append("không phải tiếng Anh (model chỉ hiểu tiếng Anh)")
    toks = re.findall(r"\w+|<censored>", text.lower())
    consec = best = 1
    for a, b in zip(toks, toks[1:]):
        consec = consec + 1 if a == b else 1
        best = max(best, consec)
    if best >= SPAM_CONSEC or (len(toks) >= SPAM_MIN_TOKENS and len(set(toks)) / len(toks) < SPAM_UNIQUE_RATIO):
        reasons.append("lặp từ kiểu spam")
    if len(words) < SHORT_WORDS:
        reasons.append(f"quá ngắn (< {SHORT_WORDS} từ)")
    return reasons


# ---------- feature meta (model.ipynb §1) ----------
SCORE_RE = re.compile(r"(?<![\d/])(\d{1,2}(?:\.\d)?)\s*/\s*10\b")
PT_BINS = [-0.01, 2, 10, 50, 200, np.inf]
PT_LABELS = ["pt_lt2", "pt_2_10", "pt_10_50", "pt_50_200", "pt_ge200"]
META = ["log_len", "has_score", "score_x10", "has_question", "has_censored"] + PT_LABELS
HELPFUL_AGE_DAYS = 30        # model hữu ích cần tuổi review → chấm "sau khoảng 30 ngày đăng"


def first_score(s):
    m = SCORE_RE.search(s)
    return float(m.group(1)) if m and float(m.group(1)) <= 10 else np.nan


def meta_frame(texts, censored, playtime):
    d = pd.DataFrame({"text": texts})
    d["log_len"] = np.log1p(d.text.str.split().str.len())
    d["score_x10"] = d.text.map(first_score)
    d["has_score"] = d.score_x10.notna().astype(int)
    d["score_x10"] = d.score_x10.fillna(0)
    d["has_question"] = d.text.str.contains("?", regex=False).astype(int)
    d["has_censored"] = np.asarray(censored, dtype=int)
    pt_bin = pd.cut(pd.Series(playtime, dtype=float).fillna(20), PT_BINS, labels=PT_LABELS)
    for c in PT_LABELS:
        d[c] = (pt_bin == c).astype(int).values
    return d


# ---------- model ----------
_models = {}


def models():
    if not _models:
        _models["sent"] = tf.keras.models.load_model(MODELS / "sentiment_best.keras")
        _models["help"] = tf.keras.models.load_model(MODELS / "helpful_m4.keras")
    return _models["sent"], _models["help"]


def score_reviews(texts, voted_up=None, playtime=None):
    """Chấm một loạt review. voted_up: list bool/None (None = người dùng không cho biết)."""
    n = len(texts)
    voted_up = list(voted_up) if voted_up is not None else [None] * n
    playtime = list(playtime) if playtime is not None else [None] * n
    cleaned = [clean_text(t) for t in texts]
    txt = [c[0] or "." for c in cleaned]
    meta = meta_frame(txt, [c[1] for c in cleaned], playtime)
    sent, helpm = models()
    X = meta[META].values.astype("float32")
    p_pos = sent.predict({"text": tf.constant(txt), "meta": X}, verbose=0, batch_size=256).ravel()

    # model hữu ích dùng nhãn recommend; nếu không có thì lấy nhãn do model sentiment đoán
    label = np.array([int(v) if v is not None else int(p >= .5) for v, p in zip(voted_up, p_pos)], dtype="float32")
    age = np.full(n, np.log1p(HELPFUL_AGE_DAYS), dtype="float32")
    Xh = np.column_stack([X, age, label]).astype("float32")
    p_help = helpm.predict({"text": tf.constant(txt), "meta": Xh}, verbose=0, batch_size=256).ravel()

    out = []
    for i in range(n):
        reasons = rule_flags(txt[i])
        mismatch = None
        if voted_up[i] is True and p_pos[i] <= .1:
            mismatch = "nội dung chê nhưng lại bấm Recommend (nghi mỉa mai / đùa)"
        elif voted_up[i] is False and p_pos[i] >= .9:
            mismatch = "nội dung khen nhưng lại Không recommend (nghi mỉa mai / bấm nhầm)"
        if mismatch:
            reasons.insert(0, mismatch)
        # lệch nhẹ (chưa tới ngưỡng 0,9 / 0,1 của notebook Bài 2) → chỉ ghi chú, không gắn cờ
        note = None
        if not mismatch and voted_up[i] is not None and voted_up[i] != (p_pos[i] >= .5):
            note = ("nội dung nghiêng về " + ("chê" if voted_up[i] else "khen") + " nhưng người viết bấm "
                    + ("Recommend" if voted_up[i] else "Không recommend") + " — chưa đủ mạnh để gắn cờ")
        out.append({
            "text_clean": txt[i],
            "p_positive": round(float(p_pos[i]), 4),
            "sentiment": "positive" if p_pos[i] >= .5 else "negative",
            "confidence": "phân vân" if abs(p_pos[i] - .5) < .1 else ("chắc chắn" if abs(p_pos[i] - .5) > .35 else "khá chắc"),
            "p_helpful": round(float(p_help[i]), 4),
            "voted_up": voted_up[i],
            "playtime": None if playtime[i] is None or pd.isna(playtime[i]) else float(playtime[i]),
            "suspicious": bool(reasons),
            "reasons": reasons,
            "note": note,
        })
    return out


def summarize(rows):
    """Thống kê cho trang kiểm duyệt hàng loạt."""
    d = pd.DataFrame(rows)
    pt = pd.cut(d.playtime.astype(float), PT_BINS, labels=["< 2h", "2–10h", "10–50h", "50–200h", "≥ 200h"])
    by_pt = (d.assign(pt=pt, neg=d.sentiment.eq("negative")).dropna(subset=["pt"])
             .groupby("pt", observed=False).agg(n=("neg", "size"), neg=("neg", "mean")).fillna(0))
    return {
        "n": len(d),
        "n_positive": int(d.sentiment.eq("positive").sum()),
        "n_negative": int(d.sentiment.eq("negative").sum()),
        "n_suspicious": int(d.suspicious.sum()),
        "n_uncertain": int(d.confidence.eq("phân vân").sum()),
        "by_playtime": [{"bin": str(k), "n": int(r.n), "neg_pct": round(float(r.neg) * 100, 1)} for k, r in by_pt.iterrows()],
        "keywords_neg": top_terms(d[d.sentiment == "negative"].text_clean),
        "keywords_pos": top_terms(d[d.sentiment == "positive"].text_clean),
    }


STOP = set("""a an the and or but if of to in on at for with is are was were be been it its it's this that these those
i me my we you your he she they them their our so not no just very really too can can't don't dont do does did have
has had will would get got game games play played playing like one all more some time from as about than then there
what when out up even only also much any after before because which who into over still other first make by im i'm
""".split())


def top_terms(texts, k=12):
    if len(texts) == 0:
        return []
    cnt = pd.Series([w for t in texts for w in set(re.findall(r"[a-z][a-z']+", t.lower())) if w not in STOP and len(w) > 2])
    vc = cnt.value_counts().head(k)
    return [{"term": t, "n": int(n)} for t, n in vc.items()]


# ---------- dữ liệu cho dashboard ----------
def dataset_overview():
    df = pd.read_csv(DATA / "steam_reviews_clean.csv")
    sus = pd.read_csv(DATA / "steam_reviews_suspicious.csv")
    df = df.merge(sus[["review_id", "suspicion_score", "flag_mismatch", "suspicious"]], on="review_id", how="left")
    g = (df.groupby("game_name").agg(n=("label", "size"), pos_pct=("label", "mean"),
                                     mismatch_pct=("flag_mismatch", "mean"), sus_pct=("suspicious", "mean"))
           .mul({"n": 1, "pos_pct": 100, "mismatch_pct": 100, "sus_pct": 100}).round(1).reset_index())
    pt = pd.cut(df.playtime_forever_hours, PT_BINS, labels=["< 2h", "2–10h", "10–50h", "50–200h", "≥ 200h"])
    by_pt = df.groupby(pt, observed=False).label.agg(["size", "mean"])
    examples = (df[df.flag_mismatch.fillna(False).astype(bool)].nlargest(12, "suspicion_score")
                [["game_name", "text_clean", "label", "suspicion_score"]])
    return {
        "n_reviews": len(df), "n_games": int(df.game_name.nunique()),
        "n_suspicious": int(df.suspicious.fillna(False).sum()), "n_mismatch": int(df.flag_mismatch.fillna(False).sum()),
        "pos_pct": round(float(df.label.mean()) * 100, 1),
        "games": g.to_dict("records"),
        "by_playtime": [{"bin": str(k), "n": int(r["size"]), "pos_pct": round(float(r["mean"]) * 100, 1)} for k, r in by_pt.iterrows()],
        "suspicious_examples": [{"game": r.game_name, "text": r.text_clean[:300], "recommend": bool(r.label),
                                 "score": round(float(r.suspicion_score), 3)} for r in examples.itertuples()],
        "metrics": [
            {"model": "Chỉ meta (LR)", "A": .72, "B": .71},
            {"model": "TF-IDF + LR (baseline)", "A": .90, "B": .91},
            {"model": "M1 Embedding + Pooling", "A": .84, "B": .88},
            {"model": "M2 BiLSTM", "A": .88, "B": .89},
            {"model": "M3 CNN 1D", "A": .87, "B": .88},
            {"model": "M4 TF-IDF + Dense (dùng trong app)", "A": .89, "B": .89},
            {"model": "M5 Ensemble M2 + M4", "A": .90, "B": .91},
        ],
    }


def game_reviews(game_name):
    df = pd.read_csv(DATA / "steam_reviews_clean.csv")
    d = df[df.game_name == game_name]
    return d.text_clean.tolist(), d.label.astype(bool).tolist(), d.playtime_forever_hours.tolist()


def game_names():
    return sorted(pd.read_csv(DATA / "steam_reviews_clean.csv", usecols=["game_name"]).game_name.unique().tolist())


# ---------- gợi ý game (feedback của thầy, dữ liệu Data Stream v2) ----------
W_GENRE, W_PUBLISHER = 2.0, 1.0


def _v2():
    games = pd.read_csv(DATA / "v2_games.csv")
    users = pd.read_csv(DATA / "v2_reviews_clean.csv")
    return games, users


def game_groups():
    games, users = _v2()
    pub_n = games.publisher.value_counts()
    genre = (games[games.is_game].groupby("genre")
             .agg(n=("app_id", "size"), pos=("game_pos_pct", "mean")).round(1).sort_values("pos", ascending=False))
    pub = (games[games.is_game & games.publisher.map(pub_n).ge(2)].groupby("publisher")
           .agg(n=("app_id", "size"), pos=("game_pos_pct", "mean")).round(1).sort_values("pos", ascending=False))
    return {
        "games": games.fillna("").to_dict("records"),
        "by_genre": [{"name": k, **r} for k, r in genre.to_dict("index").items()],
        "by_publisher": [{"name": k, **r} for k, r in pub.to_dict("index").items()],
        "users": users[["user_id", "game_name", "app_id", "voted_up", "reviewer_tier", "review_text"]].to_dict("records"),
        "multi_publishers": pub_n[pub_n >= 2].index.tolist(),
    }


def recommend(app_id, liked=True, k=8):
    """Content-based: điểm = 2·cùng thể loại + 1·cùng NPH, hoà điểm xếp theo % đánh giá tích cực của game.
    Nếu user KHÔNG thích game gốc → bỏ thể loại đó, gợi ý game điểm cao ở thể loại khác (vẫn ưu tiên cùng NPH)."""
    games, _ = _v2()
    src = games[games.app_id == app_id].iloc[0]
    cand = games[games.is_game & (games.app_id != app_id)].copy()
    same_genre = cand.genre.eq(src.genre)
    same_pub = cand.publisher.eq(src.publisher)          # NaN không bằng NaN → game thiếu NPH không ghép nhau
    if liked:
        cand["score"] = W_GENRE * same_genre + W_PUBLISHER * same_pub
        cand = cand[cand.score > 0]
    else:
        cand = cand[~same_genre]
        cand["score"] = W_PUBLISHER * same_pub[~same_genre]
    cand = cand.sort_values(["score", "game_pos_pct"], ascending=False).head(k)
    rows = []
    for r in cand.itertuples():
        why = []
        if r.genre == src.genre:
            why.append(f"cùng thể loại {r.genre}")
        if isinstance(src.publisher, str) and r.publisher == src.publisher:
            why.append(f"cùng NPH {r.publisher}")
        if not why:
            why.append(f"thể loại khác ({r.genre}), được đánh giá cao")
        rows.append({"app_id": int(r.app_id), "game_name": r.game_name, "genre": r.genre,
                     "publisher": r.publisher if isinstance(r.publisher, str) else "",
                     "game_pos_pct": float(r.game_pos_pct), "score": float(r.score), "why": why})
    return {"source": {"app_id": int(src.app_id), "game_name": src.game_name, "genre": src.genre,
                       "publisher": src.publisher if isinstance(src.publisher, str) else ""},
            "liked": liked, "items": rows}
