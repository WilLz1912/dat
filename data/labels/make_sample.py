"""Lấy mẫu để gán nhãn thủ công. Chỉ ghi text + mã mẫu; nhãn Steam, tên game, dự đoán model đều không xuất hiện trong file cần đọc."""
import os, random, re, textwrap
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 0
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "steam_reviews_clean.csv"))
df["text_clean"] = df.text_clean.astype(str)

# Cùng tập test kịch bản A với model.ipynb
_, te = train_test_split(range(len(df)), test_size=.15, stratify=df.label, random_state=SEED)
test = df.iloc[te]
rng = random.Random(SEED)

rows = rng.sample(list(test.review_id), 500)
reviews = pd.DataFrame({"sample_id": [f"r{i:03d}" for i in range(1, 501)], "review_id": rows})
text_of = test.set_index("review_id").text_clean

def sentences(t):
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", t) if len(p.split()) >= 3]

pool = []
for rid, t in text_of.items():
    s = sentences(t)
    pool += [(rid, x) for x in rng.sample(s, min(2, len(s)))]   # tối đa 2 câu/review để review dài không lấn át
picked = rng.sample(pool, 500)
sents = pd.DataFrame({"sample_id": [f"s{i:03d}" for i in range(1, 501)], "review_id": [p[0] for p in picked],
                      "sentence": [p[1] for p in picked]})

check_r = sorted(rng.sample(list(reviews.sample_id), 30))
check_s = sorted(rng.sample(list(sents.sample_id), 20))

reviews.to_csv(os.path.join(HERE, "reviews_key.csv"), index=False)
sents.to_csv(os.path.join(HERE, "sentences_key.csv"), index=False, encoding="utf-8")
pd.DataFrame({"sample_id": check_r + check_s}).to_csv(os.path.join(HERE, "check_ids.csv"), index=False)

def wrap(t):
    return "\n".join(textwrap.wrap(t, 160)) or t

os.makedirs(os.path.join(HERE, "to_read"), exist_ok=True)
for b in range(10):
    part = reviews.iloc[b * 50:(b + 1) * 50]
    with open(os.path.join(HERE, "to_read", f"reviews_{b + 1:02d}.txt"), "w", encoding="utf-8") as f:
        for sid, rid in zip(part.sample_id, part.review_id):
            f.write(f"### {sid}\n{wrap(text_of[rid])}\n")
with open(os.path.join(HERE, "to_read", "sentences.txt"), "w", encoding="utf-8") as f:
    for sid, s in zip(sents.sample_id, sents.sentence):
        f.write(f"{sid}: {s[:600]}\n")
print("test", len(test), "pool cau", len(pool), "review", len(reviews), "cau", len(sents))
print("so tu trung binh review mau", round(test.set_index("review_id").loc[rows].word_count.mean(), 1), "tong", int(test.set_index("review_id").loc[rows].word_count.sum()))
