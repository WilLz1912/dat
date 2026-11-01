"""Lần chạy 3: lấy thêm câu để gán nhãn, trộn lẫn nên lúc gán không biết câu thuộc phần nào.
test2: 500 câu ngẫu nhiên từ review test A, chưa dùng; danh sách từ khoá đã khoá trước khi gán nên chấm công bằng.
dev2: 300 câu ngẫu nhiên từ review chưa có câu nào trong bộ train cũ, dùng chọn ngưỡng.
train2: 1.500 câu ngẫu nhiên + 200 câu có từ nối đối lập và ≥ 2 khía cạnh theo từ khoá (nhắm vào câu trái cảm xúc)."""
import os, random, re
import pandas as pd
from sklearn.model_selection import train_test_split
from aspect_keywords import detect

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)
df = pd.read_csv(p("..", "steam_reviews_clean.csv")).astype({"text_clean": str})
tr, te = train_test_split(range(len(df)), test_size=.15, stratify=df.label, random_state=0)
old = [pd.read_csv(p(f)) for f in ("sentences_key.csv", "sentences_extra_key.csv", "train_sentences_key.csv")]
used = {s.strip().lower() for d in old for s in d.sentence}
old_train_rev = set(old[2].review_id)

def split(frame):
    out = {}
    for rid, t in zip(frame.review_id, frame.text_clean):
        ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if 3 <= len(x.split()) and len(x) <= 400]
        ss = [x for x in ss if x.lower() not in used]
        if ss:
            out[rid] = ss
    return out
rng = random.Random(3)
def pool(by, k=2):
    return [(r, s) for r, ss in by.items() for s in rng.sample(ss, min(k, len(ss)))]

te_by, tr_by = split(df.iloc[te]), split(df.iloc[tr])
rows = [x + ("test2",) for x in rng.sample(pool(te_by), 500)]
fresh = sorted(r for r in tr_by if r not in old_train_rev)
rng.shuffle(fresh)
dev_rev = set(fresh[:300])
rows += [(r, rng.choice(tr_by[r]), "dev2") for r in fresh[:300]]
rest = {r: ss for r, ss in tr_by.items() if r not in dev_rev}
seen = {x[1].lower() for x in rows}
rnd = [x for x in pool(rest) if x[1].lower() not in seen]
rows += [x + ("train2",) for x in rng.sample(rnd, 1500)]
seen = {x[1].lower() for x in rows}
OPP = re.compile(r"\b(but|however|although|though|except|while|yet)\b", re.I)
cf = [(r, s) for r, ss in rest.items() for s in ss if OPP.search(s) and len(detect(s)) >= 2 and s.lower() not in seen]
rows += [x + ("train2_trai",) for x in rng.sample(cf, min(200, len(cf)))]
print("ung vien trai cam xuc", len(cf))
rng.shuffle(rows)
out = pd.DataFrame({"sample_id": [f"n{i:04d}" for i in range(1, len(rows) + 1)], "review_id": [r[0] for r in rows],
                    "sentence": [r[1] for r in rows], "part": [r[2] for r in rows]})
assert not set(out[out.part != "test2"].review_id) & set(df.iloc[te].review_id)
out.to_csv(p("run3", "sentences_key.csv"), index=False, encoding="utf-8")
with open(p("to_read", "run3_sentences.txt"), "w", encoding="utf-8") as f:
    for a, b in zip(out.sample_id, out.sentence):
        f.write(f"{a}: {' '.join(b.split())}\n")
print(out.part.value_counts().to_dict(), "mean words", out.sentence.str.split().str.len().mean())
