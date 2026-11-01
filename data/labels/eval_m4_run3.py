"""Chấm M4 trên 500 câu Test 2 để so sánh trong aspect_model_run3.ipynb (M4 chạy bằng TensorFlow, notebook chạy nền PyTorch)."""
import os, sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import pandas as pd, tensorflow as tf

HERE = os.path.dirname(os.path.abspath(__file__))
p = lambda *a: os.path.join(HERE, *a)
sys.path.insert(0, p("..", "..", "app"))
from core import META, meta_frame

df = pd.read_csv(p("..", "steam_reviews_clean.csv")).set_index("review_id")
k = pd.read_csv(p("run3", "sentences_key.csv"))
k = k[k.part == "test2"]
model = tf.keras.models.load_model(p("..", "..", "models", "sentiment_best.keras"))
texts = k.sentence.tolist()
X = meta_frame(texts, ["<censored>" in t for t in texts], df.playtime_forever_hours.loc[k.review_id].tolist())[META].values.astype("float32")
s = model.predict({"text": tf.constant(texts), "meta": X}, verbose=0, batch_size=256).ravel()
pd.DataFrame({"id": k.sample_id, "p": s}).to_csv(p("run3", "m4_test2.csv"), index=False)
print("ok", len(k))
