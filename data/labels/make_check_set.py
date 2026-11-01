"""Kiểm tra file nhãn của Claude, đếm phân bố, xuất bộ kiểm tra đã giấu nhãn. Không đọc nhãn Steam hay dự đoán model."""
import os, re
from collections import Counter
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda n: os.path.join(HERE, n)
r = pd.read_csv(p("reviews_labels.csv"), keep_default_na=False)
s = pd.read_csv(p("sentences_labels.csv"), keep_default_na=False)
assert list(r.sample_id) == [f"r{i:03d}" for i in range(1, 501)] and list(s.sample_id) == [f"s{i:03d}" for i in range(1, 501)]
assert r.sentiment.isin(list("KCLU")).all() and r.kind.isin(["BT", "MIA", "DUA", "RAC", "NGAN"]).all()
pat = re.compile(r"^(NONE|(GFX|AUD|PLAY|STORY|PERF|PRICE|ONLINE|DEV):[KCTU](\|(GFX|AUD|PLAY|STORY|PERF|PRICE|ONLINE|DEV):[KCTU])*)$")
assert s.aspects.map(lambda a: bool(pat.match(a))).all()
print("sentiment", r.sentiment.value_counts().to_dict()); print("kind", r.kind.value_counts().to_dict())
c = Counter(); e = Counter()
for a in s.aspects:
    if a == "NONE": c["NONE"] += 1; continue
    for x in a.split("|"):
        k, v = x.split(":"); c[k] += 1; e[(k, v)] += 1
print("aspect", dict(c.most_common())); print("co >=1 khia canh", int((s.aspects != "NONE").sum()))
print("cam xuc", {k: {v: e[(k, v)] for v in "KCTU" if e[(k, v)]} for k in c if k != "NONE"})

ids = pd.read_csv(p("check_ids.csv")).sample_id
key = pd.read_csv(p("reviews_key.csv"))
text = pd.read_csv(p(os.path.join("..", "steam_reviews_clean.csv"))).astype({"text_clean": str}).set_index("review_id").text_clean
cr = key[key.sample_id.isin(ids)]
pd.DataFrame({"sample_id": cr.sample_id, "review": text.loc[cr.review_id].values, "sentiment": "", "kind": ""}).to_csv(p("KIEM_TRA_reviews.csv"), index=False, encoding="utf-8-sig")
sk = pd.read_csv(p("sentences_key.csv"))
cs = sk[sk.sample_id.isin(ids)]
pd.DataFrame({"sample_id": cs.sample_id, "sentence": cs.sentence, "aspects": ""}).to_csv(p("KIEM_TRA_sentences.csv"), index=False, encoding="utf-8-sig")
print("kiem tra", len(cr), len(cs))
