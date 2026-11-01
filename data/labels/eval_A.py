"""Phương án A, phần chưa cần transformer: từ khoá nhận khía cạnh + M4 hiện có chấm khen/chê, chấm trên nhãn của Claude.
Chỉ đọc data và model, không train, không sửa model."""
import os, re, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)
sys.path.insert(0, p("..", "..", "app"))
from core import META, meta_frame
sys.path.insert(0, HERE)
from aspect_keywords import detect

ASPECTS = ["GFX", "AUD", "PLAY", "STORY", "PERF", "PRICE", "ONLINE", "DEV"]

def parse(a):
    return {} if a == "NONE" else {x.split(":")[0]: x.split(":")[1] for x in a.split("|")}

df = pd.read_csv(p("..", "steam_reviews_clean.csv")).astype({"text_clean": str})
_, te = train_test_split(range(len(df)), test_size=.15, stratify=df.label, random_state=0)
test = df.iloc[te].set_index("review_id")
model = tf.keras.models.load_model(p("..", "..", "models", "sentiment_best.keras"))

def score(texts, playtime):
    texts = [t if t else "." for t in texts]
    cens = ["<censored>" in t for t in texts]
    X = meta_frame(texts, cens, playtime)[META].values.astype("float32")
    return model.predict({"text": tf.constant(texts), "meta": X}, verbose=0, batch_size=256).ravel()

test["p"] = score(test.text_clean.tolist(), test.playtime_forever_hours.tolist())
out = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)

say("== M4 trên toàn tập test A, nhãn Steam: acc", round(accuracy_score(test.label, test.p >= .5), 3), "AUC", round(roc_auc_score(test.label, test.p), 3), "n", len(test))

# ---- 500 review
r = pd.read_csv(p("reviews_key.csv")).merge(pd.read_csv(p("reviews_labels.csv"), keep_default_na=False), on="sample_id")
r["y"] = test.label.loc[r.review_id].values
r["pm"] = test.p.loc[r.review_id].values
r["wrong"] = (r.pm >= .5).astype(int) != r.y
def grp(name, m):
    d = r[m]
    auc = round(roc_auc_score(d.y, d.pm), 3) if d.y.nunique() == 2 and len(d) >= 10 else None
    say(f"  {name}: n={len(d)} acc={1 - d.wrong.mean():.3f} auc={auc}")
say("\n== 500 review: M4 theo nhãn Steam, tách theo nhãn Claude")
grp("tất cả", r.sentiment != "")
clear = r.sentiment.isin(["K", "C"]) & (r.kind == "BT")
grp("rõ ràng (K/C + Bình thường)", clear)
grp("còn lại", ~clear)
for k, n in [("U", "Không rõ"), ("L", "Vừa khen vừa chê")]:
    grp(n, r.sentiment == k)
for k, n in [("DUA", "Đùa/meme"), ("MIA", "Mỉa mai"), ("RAC", "Lạc đề"), ("NGAN", "Quá ngắn")]:
    grp(n, r.kind == k)
kc = r[r.sentiment.isin(["K", "C"])]
agree = (kc.sentiment == "K").astype(int) == kc.y
say(f"\nClaude K/C khớp nhãn Steam: {agree.sum()}/{len(kc)} = {agree.mean():.3f}")
for s, y, n in [("K", 0, "Claude thấy khen nhưng Steam Not Recommended"), ("C", 1, "Claude thấy chê nhưng Steam Recommended")]:
    say(f"  {n}: {((r.sentiment == s) & (r.y == y)).sum()}")
w = r[r.wrong]
hum = (w.sentiment == "K").astype(int)
kcw = w.sentiment.isin(["K", "C"])
say(f"\n== Nguyên nhân khi M4 đoán sai ({len(w)} review)")
say("  model sai thật (Claude khớp Steam):", int((kcw & (hum == w.y)).sum()))
say("  nhãn nhiễu (Claude ngược Steam):", int((kcw & (hum != w.y)).sum()))
say("  vừa khen vừa chê:", int((w.sentiment == "L").sum()))
say("  không rõ:", int((w.sentiment == "U").sum()))

# ---- câu
key = pd.read_csv(p("sentences_key.csv")).merge(pd.read_csv(p("sentences_labels.csv"), keep_default_na=False), on="sample_id")
ext = pd.read_csv(p("sentences_extra_key.csv")).merge(pd.read_csv(p("sentences_extra_labels.csv"), keep_default_na=False), on="sample_id")
for d in (key, ext):
    d["gold"] = d.aspects.map(parse)
    d["pred"] = d.sentence.map(detect)
    d["p"] = score(d.sentence.tolist(), test.playtime_forever_hours.loc[d.review_id].tolist())

say("\n== Từ khoá nhận khía cạnh, chỉ 500 câu ngẫu nhiên")
tp_all = fp_all = fn_all = 0
rows = []
for a in ASPECTS:
    g = key.gold.map(lambda x: a in x); h = key.pred.map(lambda x: a in x)
    tp, fp, fn = int((g & h).sum()), int((~g & h).sum()), int((g & ~h).sum())
    tp_all += tp; fp_all += fp; fn_all += fn
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    say(f"  {a:7s} thật={tp + fn:3d} đoán={tp + fp:3d} precision={pr:.2f} recall={rc:.2f} F1={2 * pr * rc / max(pr + rc, 1e-9):.2f}")
pr, rc = tp_all / (tp_all + fp_all), tp_all / (tp_all + fn_all)
say(f"  TỔNG    precision={pr:.2f} recall={rc:.2f} F1={2 * pr * rc / (pr + rc):.2f}")

say("\n== M4 chấm khen/chê cả câu, trên khía cạnh đúng (K/C), 500 câu + 110 câu thêm")
al = pd.concat([key, ext])
rec = []
for _, x in al.iterrows():
    vals = {v for v in x.gold.values() if v in "KC"}
    for a, v in x.gold.items():
        if v in "KC":
            rec.append((a, v, x.p, len(x.gold) > 1, len(vals) > 1, a in x.pred, x.sample_id))
e = pd.DataFrame(rec, columns=["a", "v", "p", "multi", "conflict", "found", "id"])
e["ok"] = (e.p >= .5) == (e.v == "K")
say(f"  tất cả: n={len(e)} acc={e.ok.mean():.3f} | luôn đoán Chê: {(e.v == 'C').mean():.3f}")
say(f"  câu 1 khía cạnh: n={(~e.multi).sum()} acc={e[~e.multi].ok.mean():.3f}")
say(f"  câu nhiều khía cạnh, cùng cảm xúc: n={(e.multi & ~e.conflict).sum()} acc={e[e.multi & ~e.conflict].ok.mean():.3f}")
say(f"  câu nhiều khía cạnh, trái cảm xúc: n={e.conflict.sum()} acc={e[e.conflict].ok.mean():.3f}")
for v, n in [("K", "khen"), ("C", "chê")]:
    say(f"  chỉ khía cạnh được {n}: n={(e.v == v).sum()} acc={e[e.v == v].ok.mean():.3f}")
for a in ASPECTS:
    d = e[e.a == a]
    say(f"  {a:7s} n={len(d):3d} acc={d.ok.mean():.3f} (khen {int((d.v == 'K').sum())}, chê {int((d.v == 'C').sum())})")

say("\n== Toàn bộ đường ống A (từ khoá + M4), 500 câu ngẫu nhiên, cặp khía cạnh + cảm xúc (K/C)")
tp = fp = fn = 0
for _, x in key.iterrows():
    gold = {(a, v) for a, v in x.gold.items() if v in "KC"}
    skip = {a for a, v in x.gold.items() if v not in "KC"}
    pred = {(a, "K" if x.p >= .5 else "C") for a in x.pred if a not in skip}
    tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
say(f"  đúng={tp} thừa={fp} thiếu={fn} precision={pr:.2f} recall={rc:.2f} F1={2 * pr * rc / max(pr + rc, 1e-9):.2f}")

say("\n== Ví dụ từ khoá bỏ sót")
n = 0
for _, x in key.iterrows():
    miss = set(x.gold) - x.pred
    if miss and n < 14:
        say(" ", sorted(miss), x.sentence[:130]); n += 1
say("\n== Ví dụ từ khoá gán thừa")
n = 0
for _, x in key.iterrows():
    extra = x.pred - set(x.gold)
    if extra and n < 14:
        say(" ", sorted(extra), x.sentence[:130]); n += 1
say("\n== Ví dụ M4 chấm sai trên câu trái cảm xúc")
c = e[e.conflict & ~e.ok].drop_duplicates("id").head(8)
for i in c.id:
    say(" ", i, al[al.sample_id == i].aspects.iloc[0], al[al.sample_id == i].sentence.iloc[0][:140])
pd.concat([pd.DataFrame({"id": test.index.astype(str), "p": test.p.values}),
           pd.DataFrame({"id": key.sample_id, "p": key.p}), pd.DataFrame({"id": ext.sample_id, "p": ext.p})]).to_csv(p("m4_preds.csv"), index=False)
with open(p("ket_qua_A_tu_khoa_M4.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(out))
