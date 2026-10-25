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
| `app/` | Ứng dụng web localhost dùng 2 model trên (xem dưới) |

## Chạy ứng dụng
```
pip install -r requirements.txt
python app/server.py          # hoặc bấm đúp run_app.bat
```
Mở http://localhost:8050 (đổi cổng: `python app/server.py 9000`). Tài liệu API: http://localhost:8050/docs

| Trang | Dùng để làm gì |
|---|---|
| Phân tích review | Dán 1 review tiếng Anh → khen/chê, khả năng được vote hữu ích, cảnh báo nếu nội dung ngược với nút Recommend (mỉa mai / troll / bấm nhầm) |
| Kiểm duyệt hàng loạt | Chọn 1 game trong dữ liệu hoặc tải CSV (`review_text`, tuỳ chọn `voted_up`, `playtime`) → tỉ lệ khen/chê, % chê theo giờ chơi (newbie vs oldbie), từ hay gặp trong review chê, lọc review đáng ngờ, xếp review hữu ích, tải kết quả CSV |
| Tổng quan dữ liệu | Số liệu tập train, game có nhiều review mâu thuẫn, so sánh các model |
| Gợi ý game | Đồ thị nhóm game theo thể loại / nhà phát hành; chọn user hoặc game → gợi ý game cùng nhóm, xếp theo % đánh giá tích cực (dữ liệu minh hoạ Data Stream v2) |

Cấu trúc: `app/core.py` (làm sạch text giống `preprocessing.ipynb`, tạo feature giống `model.ipynb`, gọi model, gợi ý),
`app/server.py` (API FastAPI), `app/static/` (giao diện HTML/JS, Chart.js + vis-network để sẵn trong `vendor/`, chạy không cần mạng).

Load model trong code: `tf.keras.models.load_model("models/sentiment_best.keras")`, đầu vào
`{"text": tf.constant([...]), "meta": mảng 10 cột}` (xem `app/core.py`).
