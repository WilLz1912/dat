# DAT — Phân tích review Steam

Phân loại cảm xúc, lọc review đáng ngờ, chấm mức hữu ích, phân tích theo khía cạnh (DistilBERT kết hợp từ khoá) và gợi ý game từ review Steam. Báo cáo đầy đủ: [BAO_CAO.md](BAO_CAO.md).

## Kết quả chính

| Bài toán | Mô hình cơ sở | Mô hình tốt nhất | Kết quả |
|---|---|---|---|
| Khen/chê cả review, ROC-AUC kịch bản A / B | TF-IDF + LR: 0,903 / 0,906 | DistilBERT | 0,937 / 0,930 |
| Nhận khía cạnh, micro F1 trên Test 2 | Từ khoá: 0,557 | Từ khoá trong đầu vào DistilBERT | 0,682 |
| Khen/chê theo khía cạnh, độ chính xác | M4 cả câu: 0,742 | DistilBERT theo cặp | 0,898 |
| Cả đường ống, F1 trên Test 2 | Từ khoá + M4: 0,451 | Từ khoá trong đầu vào + theo cặp | 0,622 |

Nhãn dùng để chấm phần khía cạnh do mô hình ngôn ngữ Claude gán và chưa được người kiểm tra độc lập. Câu có các khía cạnh trái cảm xúc vẫn chưa được cải thiện (độ chính xác khoảng 0,55, n=55). Chi tiết ở mục 5 và 8 của báo cáo.

## Cấu trúc

| File | Nội dung |
|---|---|
| `preprocessing.ipynb` | EDA + làm sạch `data/steam_reviews.csv` → `data/steam_reviews_clean.csv` |
| `preprocessing_v2.ipynb` | Làm sạch + EDA `data/Data Stream_v2.xlsx` → `data/v2_reviews_clean.csv`, `data/v2_games.csv` |
| `model.ipynb` | Mô hình Keras: Bài 1 cảm xúc, Bài 2 review đáng ngờ, Bài 3 mức hữu ích |
| `aspect_model.ipynb` | Lần chạy 1: DistilBERT khen/chê ở hai kịch bản, phân tích lỗi, khía cạnh bước đầu |
| `aspect_model_run2.ipynb` | Lần chạy 2: đường cong học, các cách kết hợp từ khoá với transformer |
| `aspect_model_run3.ipynb` | Lần chạy 3: thêm 2.500 câu có nhãn, chấm trên Test 2, kết quả cuối theo game |
| `data/labels/` | Nhãn tay, quy tắc gán nhãn (`QUY_TAC_GAN_NHAN.md`), danh sách từ khoá (`aspect_keywords.py`), kết quả từng lần chạy (`run2/`, `run3/`), bộ kiểm tra giấu nhãn (`KIEM_TRA_*.csv`) |
| `report/` | `make_figures.py` vẽ lại 11 hình của báo cáo vào `report/figures/` |
| `models/sentiment_best.keras` | Mô hình cảm xúc M4 (TF-IDF + Dense), ROC-AUC ≈ 0,89–0,90 |
| `models/helpful_m4.keras` | Mô hình dự đoán review có vote hữu ích, ROC-AUC ≈ 0,76 |
| `models/transformer/` | Trọng số DistilBERT (khen/chê, nhận khía cạnh, theo cặp), khoảng 800 MB, **không đưa lên git** |
| `data/steam_reviews_suspicious.csv` | Review có nội dung mâu thuẫn với nhãn (mỉa mai, bấm nhầm...) |
| `app/` | Ứng dụng web localhost, dùng 2 mô hình Keras ở trên |

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

## Chạy các notebook transformer

Ba notebook `aspect_model*.ipynb` cần môi trường riêng với `requirements.txt`: Keras 3, keras-hub và PyTorch có CUDA, chạy với nền `KERAS_BACKEND=torch`. Đã chạy trên GTX 1650 4 GB; lần chạy 3 mất khoảng 2,5 giờ. Notebook đã lưu sẵn toàn bộ kết quả, không cần chạy lại để xem.
