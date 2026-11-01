"""Phương án B: lấy 1.500 câu để gán nhãn làm dữ liệu train cho model khía cạnh.
Chỉ lấy từ review KHÔNG thuộc tập test A để tránh rò rỉ. 900 câu ngẫu nhiên + 600 câu chọn theo từ khoá cho khía cạnh hiếm, rồi trộn lẫn."""
import os, random, re
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "steam_reviews_clean.csv")).astype({"text_clean": str})
tr, te = train_test_split(range(len(df)), test_size=.15, stratify=df.label, random_state=0)
train = df.iloc[tr]
assert not set(train.review_id) & set(df.iloc[te].review_id)

sents = []
for rid, t in zip(train.review_id, train.text_clean):
    sents += [(rid, p.strip()) for p in re.split(r"(?<=[.!?])\s+|\n+", t) if 3 <= len(p.split()) and len(p) <= 400]
rng = random.Random(2)
by_review = {}
for rid, s in sents:
    by_review.setdefault(rid, []).append(s)
pool = [(rid, s) for rid, ss in by_review.items() for s in rng.sample(ss, min(2, len(ss)))]
rows, used = [], set()
for x in rng.sample(pool, 900):
    rows.append(x + ("random",)); used.add(x[1])

KW = {"AUD": (r"\bsounds?\b|\baudio\b|\bmusic\b|soundtrack|\bost\b|voice ?act|\bvoices?\b", 170),
      "GFX": (r"graphic|visual|art ?style|textures?\b|animation|beautiful|gorgeous|\bugly\b|aesthetic", 170),
      "STORY": (r"\bstor(y|ies|yline)\b|\bplot\b|narrative|\bcharacters?\b|\bwriting\b|\bending\b|\bquests?\b|\blore\b", 130),
      "PRICE": (r"\bprice|\bmoney\b|\brefund|overpriced|\bdlcs?\b|microtransaction|pay to win|\bcost|battle ?pass", 130)}
for k, (rx, n) in KW.items():
    c = [x for x in sents if re.search(rx, x[1], re.I) and x[1] not in used]
    print(k, "ung vien", len(c))
    for x in rng.sample(c, min(n, len(c))):
        rows.append(x + ("keyword",)); used.add(x[1])
rng.shuffle(rows)
out = pd.DataFrame({"sample_id": [f"t{i:04d}" for i in range(1, len(rows) + 1)], "review_id": [r[0] for r in rows],
                    "sentence": [r[1] for r in rows], "source": [r[2] for r in rows]})
out.to_csv(os.path.join(HERE, "train_sentences_key.csv"), index=False, encoding="utf-8")
path = os.path.join(HERE, "to_read", "train_sentences.txt")
with open(path, "w", encoding="utf-8") as f:
    for a, b in zip(out.sample_id, out.sentence):
        f.write(f"{a}: {b}\n")
print("tong", len(out), out.source.value_counts().to_dict(), "kich thuoc", os.path.getsize(path))
