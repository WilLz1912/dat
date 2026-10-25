# DAT — Phân tích review Steam

Phân tích cảm xúc, phát hiện review đáng ngờ và dự đoán mức hữu ích của review Steam (Keras / TensorFlow).

## Cấu trúc
| File | Nội dung |
|---|---|
| `preprocessing.ipynb` | EDA + làm sạch `data/steam_reviews.csv` → `data/steam_reviews_clean.csv` |
| `preprocessing_v2.ipynb` | Làm sạch + EDA `data/Data Stream_v2.xlsx` → `data/v2_reviews_clean.csv`, `data/v2_games.csv` |
| `model.ipynb` | Bài 1 sentiment, Bài 2 review đáng ngờ, Bài 3 mức hữu ích |
| `models/sentiment_best.keras` | Model sentiment (TF-IDF + Dense), ROC-AUC ≈ 0,89–0,90 |
| `models/helpful_m4.keras` | Model dự đoán review có vote hữu ích, ROC-AUC ≈ 0,76 |
| `data/steam_reviews_suspicious.csv` | Review có nội dung mâu thuẫn với nhãn (mỉa mai, bấm nhầm...) |

## Chạy
```
pip install tensorflow==2.20 pandas scikit-learn
```
Load model: `tf.keras.models.load_model("models/sentiment_best.keras")`, đầu vào là `tf.constant([text, ...])`.
