"""Lấy thêm câu cho khía cạnh hiếm (<30 câu trong mẫu ngẫu nhiên). Chọn theo từ khoá nên KHÔNG phải mẫu ngẫu nhiên: chỉ dùng đo độ chính xác theo khía cạnh."""
import os, random, re
import pandas as pd
from sklearn.model_selection import train_test_split
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "..", "steam_reviews_clean.csv")).astype({"text_clean": str})
_, te = train_test_split(range(len(df)), test_size=.15, stratify=df.label, random_state=0)
test = df.iloc[te]
seen = set(pd.read_csv(os.path.join(HERE, "sentences_key.csv")).sentence)
KW = {"AUD": (r"\bsound|\baudio|\bmusic|soundtrack|\bost\b|voice act|\bvoices?\b", 40),
      "GFX": (r"graphic|visual|art style|artstyle|textures?\b|animation|beautiful|\bugly\b", 30),
      "STORY": (r"\bstory|\bplot\b|narrative|characters?\b|\bending\b|\bquests?\b|\blore\b", 25),
      "PRICE": (r"\bprice|\bmoney\b|\brefund|overpriced|\bdlc\b|microtransaction|pay to win|\bcost", 15)}
sents = []
for rid, t in zip(test.review_id, test.text_clean):
    sents += [(rid, p.strip()) for p in re.split(r"(?<=[.!?])\s+|\n+", t) if len(p.split()) >= 3 and p.strip() not in seen and len(p) <= 600]
rng = random.Random(1)
rows, used = [], set()
for k, (rx, n) in KW.items():
    c = [x for x in sents if re.search(rx, x[1], re.I) and x[1] not in used]
    print(k, "ung vien", len(c))
    for x in rng.sample(c, min(n, len(c))):
        rows.append((x[0], x[1])); used.add(x[1])
rng.shuffle(rows)   # trộn để lúc gán không biết câu được chọn theo khía cạnh nào
out = pd.DataFrame({"sample_id": [f"x{i:03d}" for i in range(1, len(rows) + 1)], "review_id": [r[0] for r in rows], "sentence": [r[1] for r in rows]})
out.to_csv(os.path.join(HERE, "sentences_extra_key.csv"), index=False, encoding="utf-8")
with open(os.path.join(HERE, "to_read", "sentences_extra.txt"), "w", encoding="utf-8") as f:
    for a, b in zip(out.sample_id, out.sentence):
        f.write(f"{a}: {b}\n")
print("tong", len(out))
