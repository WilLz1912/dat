# Quy tắc gán nhãn (viết trước khi gán, chốt với Quang 2026-10-31)

Người gán: Claude Opus 5.5. Nhãn chỉ dùng để đánh giá, không đưa vào train. Khi gán không xem nhãn Steam, tên game, dự đoán model.

## Chung
- Chỉ dựa vào chữ trong review, không dùng kiến thức ngoài về game để đoán thêm.
- Cần ngữ cảnh mới hiểu (inside joke, meme lạ) thì ghi "Không rõ", không đoán.
- `<censored>` là từ tục bị Steam che: chỉ coi là tín hiệu chê khi câu chữ xung quanh đang chê.

## Bộ 1: review (`reviews_labels.csv`: sample_id, sentiment, kind)
Sentiment, là thái độ với game thể hiện trong chữ:
- `K` Khen; `C` Chê.
- `L` Vừa khen vừa chê: cả hai đều đáng kể và không thể nói bên nào rõ ràng thắng. Có kết luận rõ ("hay nhưng có vài lỗi, vẫn nên mua") thì theo kết luận.
- `U` Không rõ.

Kind:
- `BT` Bình thường.
- `MIA` Mỉa mai: nghĩa thật ngược nghĩa chữ, và chữ đã đủ cho thấy điều đó. Vẫn gán sentiment theo nghĩa thật.
- `DUA` Đùa, meme, inside joke, không nhằm đánh giá nghiêm túc.
- `RAC` Lạc đề, rác, ASCII art, không nói về game.
- `NGAN` Quá ngắn để kết luận.

## Bộ 2: câu (`sentences_labels.csv`: sample_id, aspects)
`aspects` là các cặp `khía_cạnh:cảm_xúc` cách nhau bởi `|`; câu không thuộc khía cạnh nào ghi `NONE`.

Khía cạnh: `GFX` đồ hoạ, hình ảnh, phong cách, hoạt ảnh. `AUD` nhạc, lồng tiếng, hiệu ứng. `PLAY` cơ chế, điều khiển, độ khó, cày cuốc. `STORY` truyện, nhân vật, nhiệm vụ.
`PERF` crash, lag, tối ưu, bug. `PRICE` giá, DLC, hoàn tiền, đáng tiền hay không. `ONLINE` server, ghép trận, hack, người chơi khác. `DEV` patch, hỗ trợ, nhà phát triển, bỏ rơi game.

Cảm xúc: `K` khen, `C` chê, `T` trung tính hoặc lẫn lộn, `U` không rõ.

Câu chung chung ("game hay", "fun") không gắn khía cạnh, ghi `NONE`. "Fun" chỉ tính `PLAY` khi nói rõ về cách chơi.
