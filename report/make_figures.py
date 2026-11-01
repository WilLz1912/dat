"""Sinh hình cho báo cáo vào report/figures/. Chạy từ thư mục gốc: python report/make_figures.py
Hình của lần chạy 3 chỉ được vẽ khi data/labels/run3/ đã có kết quả."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path("report/figures"); OUT.mkdir(parents=True, exist_ok=True)
RUN = Path("data/labels/run3")
BLUE, GREY, LIGHT, INK, MUTED = "#4C78A8", "#8A94A6", "#C9CFD8", "#1F2733", "#5B6472"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LIGHT, "axes.grid": True, "grid.color": "#E6E9EE", "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
VI = dict(GFX="Đồ hoạ", AUD="Âm thanh", PLAY="Gameplay", STORY="Cốt truyện", PERF="Hiệu năng, lỗi", PRICE="Giá, trả phí", ONLINE="Online, cộng đồng", DEV="Nhà phát triển")

def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print(name)

def hbars(ax, labels, series, colors, fmt="{:.3f}", xlim=None, h=.8):
    # series: {tên: giá trị}, vẽ ngang, nhãn trên cùng
    y = np.arange(len(labels))[::-1]; n = len(series); bh = h / n
    for i, ((name, vals), col) in enumerate(zip(series.items(), colors)):
        pos = y + h / 2 - bh * (i + .5)
        ax.barh(pos, vals, bh * .86, color=col, label=name)
        for p, v in zip(pos, vals):
            if not np.isnan(v):
                ax.text(v + (xlim[1] - xlim[0]) * .008 if xlim else v, p, fmt.format(v), va="center", fontsize=8.5, color=MUTED)
    ax.set_yticks(y, labels); ax.grid(axis="y", visible=False)
    if xlim:
        ax.set_xlim(*xlim)
    if n > 1:
        ax.legend(loc="lower right", fontsize=9)

# ---- Quy trình
def flow():
    rows = [("Bài toán review", ["Steam API\n10.000 review, 99 game", "Làm sạch\n9.971 review", "EDA, chọn đặc trưng", "Mô hình Keras M1–M5", "Khen/chê, lọc\nđáng ngờ, hữu ích"]),
            ("Mở rộng theo góp ý", ["Gán nhãn tay\n500 review, 4.610 câu", "Từ khoá theo khía cạnh", "DistilBERT\nnhận khía cạnh", "DistilBERT theo cặp\nchấm khen/chê", "Bảng khen/chê\ntheo game"]),
            ("Gợi ý game", ["Data Stream v2\n275 review, 92 game", "Gán thể loại,\nnhà phát hành", "Nhóm game", "Tìm game cùng nhóm", "Gợi ý cho user"])]
    fig, ax = plt.subplots(figsize=(12, 4.1)); ax.axis("off"); ax.set_xlim(-.05, 11.4); ax.set_ylim(0, 4.05)
    w, hh, step = 1.95, .72, 2.35
    for r, (title, steps) in enumerate(rows):
        y = 3.0 - r * 1.33
        ax.text(0, y + hh + .08, title, fontweight="bold", fontsize=10)
        for i, s in enumerate(steps):
            x = i * step
            ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#EEF3F8" if i < 4 else BLUE, ec="none"))
            ax.text(x + w / 2, y + hh / 2, s, ha="center", va="center", fontsize=8.8, color=INK if i < 4 else "white")
            if i:
                ax.annotate("", (x - .03, y + hh / 2), (x - step + w + .03, y + hh / 2), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
    save(fig, "hinh01_quy_trinh")

# ---- EDA
def eda():
    d = pd.read_csv("data/steam_reviews_clean.csv")
    pt = pd.cut(d.playtime_forever_hours, [-1, 2, 10, 50, 200, 1000, 1e9], labels=["< 2", "2–10", "10–50", "50–200", "200–1000", "≥ 1000"], right=False)
    wc = pd.cut(d.word_count, [0, 5, 10, 20, 50, 100, 1e9], labels=["≤ 5", "6–10", "11–20", "21–50", "51–100", "> 100"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, g, t, xl in [(axs[0], pt, "Theo giờ chơi", "Giờ chơi"), (axs[1], wc, "Theo độ dài review", "Số từ")]:
        s = d.groupby(g, observed=True).label.mean() * 100
        b = ax.bar(s.index.astype(str), s.values, .62, color=BLUE)
        ax.bar_label(b, fmt="%.0f%%", fontsize=9, color=MUTED, padding=2)
        ax.set_title(t); ax.set_xlabel(xl); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", labelsize=8.5)
    axs[0].set_ylabel("Tỉ lệ review khen")
    save(fig, "hinh02_eda")

# ---- Khen/chê toàn review
def sentiment():
    names = ["Chỉ đặc trưng phụ + LR", "M1 Embedding + Pooling", "M3 CNN 1D", "M2 BiLSTM", "M4 TF-IDF + Dense", "M5 Ensemble M2 + M4", "TF-IDF + LR", "TF-IDF + phụ + LR", "DistilBERT"]
    A = [.716, .836, .871, .884, .891, .902, .903, .908, .937]; B = [.711, .883, .883, .893, .893, .905, .906, .909, .930]
    fig, ax = plt.subplots(figsize=(7.4, 4)); y = np.arange(len(names))[::-1]
    ax.hlines(y, [min(a, b) for a, b in zip(A, B)], [max(a, b) for a, b in zip(A, B)], color=LIGHT, lw=2)
    ax.scatter(A, y, s=62, color=BLUE, label="A: game đã biết", zorder=3, edgecolor="white")
    ax.scatter(B, y, s=62, color=GREY, marker="s", label="B: game mới", zorder=3, edgecolor="white")
    ax.set_yticks(y, names); ax.set_xlabel("ROC-AUC trên tập test"); ax.set_xlim(.69, .95); ax.grid(axis="y", visible=False); ax.legend(loc="center left", bbox_to_anchor=(.08, .3))
    ax.text(.9335, y[-1] + .32, ".937 / .930", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "hinh03_khen_che_auc")
    fig, ax = plt.subplots(figsize=(7, 2.9))
    hbars(ax, ["Model sai thật", "Nhãn Steam ngược nội dung", "Review vừa khen vừa chê", "Không xác định được"], {"M4": [46, 13, 10, 28], "DistilBERT": [26, 14, 7, 22]}, [LIGHT, BLUE], "{:.0f}", (0, 52))
    ax.set_xlabel("Số review đoán sai, trong 500 review đọc tay")
    save(fig, "hinh04_nguyen_nhan_loi")

# ---- Lần chạy 3
def run3():
    D = pd.read_csv(RUN / "nhan_khia_canh.csv")
    cv = D[D.phương_án.str.startswith("Từ khoá trong")].groupby("số_câu")[["test2", "dev"]].mean()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(cv.index, cv.test2, marker="o", ms=7, lw=2, color=BLUE, label="Từ khoá trong đầu vào")
    pl = D[D.phương_án.str.startswith("DistilBERT")].groupby("số_câu").test2.mean()
    ax.plot(pl.index, pl.values, marker="s", ms=7, lw=2, color=GREY, label="DistilBERT")
    kw = D[D.phương_án == "Từ khoá"].test2.iloc[0]
    ax.axhline(kw, color=GREY, lw=1.2, ls="--", label="Từ khoá")
    for x, v in cv.test2.items():
        ax.annotate(f"{v + 1e-9:.3f}", (x, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.set(xlabel="Số câu train", ylabel="Micro F1 trên Test 2", ylim=(.4, .85)); ax.legend(loc="upper left", ncol=3, fontsize=9)
    save(fig, "hinh05_duong_cong_hoc")

    m = D.groupby("phương_án")[["test2", "test1", "số_câu"]].mean()
    m = m[(m.số_câu == 0) | (m.số_câu == D.số_câu.max())].sort_values("test2", ascending=False)
    lab = [n.split(" (")[0] for n in m.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hbars(ax, lab, {"Test 2 (từ khoá đã khoá trước)": m.test2.values, "Test 1 (từ khoá viết sau khi đọc)": m.test1.values}, [BLUE, LIGHT], "{:.3f}", (0, .9))
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2, fontsize=9); ax.set_xlabel("Micro F1 nhận khía cạnh")
    save(fig, "hinh06_nhan_khia_canh")

    per = pd.read_csv(RUN / "f1_theo_khia_canh.csv", index_col=0)
    cols = [c for c in per.columns if c != "số câu thật"]
    M = per[cols].values.astype(float)
    fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.grid(False)
    ax.imshow(M, cmap="Blues", vmin=.2, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [c.split(" (")[0].replace("Từ khoá trong đầu vào", "Từ khoá\ntrong đầu vào").replace("TF-IDF + LR", "TF-IDF\n+ LR") for c in cols], fontsize=9)
    ax.xaxis.tick_top(); ax.set_yticks(range(len(per)), [f"{i} ({n})" for i, n in zip(per.index, per["số câu thật"])], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > .68 else INK)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    save(fig, "hinh07_f1_theo_khia_canh")

    P = pd.read_csv(RUN / "khen_che_theo_cap.csv", index_col=0)
    k = json.load(open(RUN / "chon.json", encoding="utf-8"))["BEST_K"]
    col = [c for c in P.columns if c.startswith("Theo cặp") and ("trái" in c) == (k == "D")][0]
    grp = ["Tất cả", "Câu 1 khía cạnh", "Nhiều khía cạnh, cùng cảm xúc", "Nhiều khía cạnh, trái cảm xúc", "Thật sự là Khen", "Thật sự là Chê"]
    Q = P.loc[grp]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    hbars(ax, [f"{g} (n={int(n)})" for g, n in zip(grp, Q["số cặp"])], {"M4 cả câu": Q["M4 cả câu"].values, "DistilBERT cả câu": Q["DistilBERT cả câu"].values, "DistilBERT theo cặp": Q[col].values}, [LIGHT, GREY, BLUE], "{:.2f}", (0, 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.13), ncol=3, fontsize=9); ax.set_xlabel("Độ chính xác")
    save(fig, "hinh08_khen_che_theo_cap")

    C = pd.read_csv(RUN / "ca_duong_ong.csv", index_col=0); C = C[C.tập == "Test 2"]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    hbars(ax, list(C.index), {"F1": C.F1.values}, [BLUE], "{:.3f}", (0, .85)); ax.set_xlabel("F1 cả đường ống trên Test 2, đúng cả khía cạnh lẫn cảm xúc")
    save(fig, "hinh09_ca_duong_ong")

    A = pd.read_csv(RUN / "aspect_by_game.csv"); order = list(VI)
    top = A.groupby("game").so_cau.sum().sort_values(ascending=False).head(15).index
    H = A[A.game.isin(top)].pivot(index="game", columns="aspect", values="pct_che").reindex(index=top, columns=order)
    Nc = A[A.game.isin(top)].pivot(index="game", columns="aspect", values="so_cau").reindex(index=top, columns=order)
    H = H.where(Nc >= 5)
    fig, ax = plt.subplots(figsize=(8.6, 5.2)); ax.grid(False)
    ax.imshow(H.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(8), [VI[a].replace("Nhà phát triển", "Nhà phát\ntriển").replace(", ", ",\n") for a in order], fontsize=9); ax.xaxis.tick_top(); ax.set_yticks(range(len(top)), [g[:30] for g in top], fontsize=9)
    for i in range(H.shape[0]):
        for j in range(8):
            v = H.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=9, color=GREY if np.isnan(v) else ("white" if v > 60 else INK))
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    save(fig, "hinh10_khen_che_theo_game")

# ---- Gợi ý
def recommend():
    g = pd.read_csv("data/v2_games.csv"); u = pd.read_csv("data/v2_reviews_clean.csv")
    s = g.genre.value_counts().sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    b = axs[0].barh(s.index, s.values, .68, color=BLUE); axs[0].bar_label(b, fontsize=8.5, color=MUTED, padding=2); axs[0].grid(axis="y", visible=False)
    axs[0].set_title("Số game theo thể loại"); axs[0].tick_params(axis="y", labelsize=8.5)
    order = u.groupby("reviewer_tier").games_owned.median().sort_values().index
    t = (1 - u.groupby("reviewer_tier").voted_up.mean()).reindex(order) * 100
    b = axs[1].bar([{"Ít game nhất": "Ít game", "Tầm trung": "Tầm trung", "Nhiều game nhất": "Nhiều game"}.get(x, x) for x in t.index], t.values, .55, color=BLUE)
    axs[1].bar_label(b, fmt="%.1f%%", fontsize=9, color=MUTED, padding=2); axs[1].set_title("Tỉ lệ review chê theo nhóm"); axs[1].set_ylim(0, 45); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "hinh11_du_lieu_goi_y")

flow(); eda(); sentiment(); recommend()
if (RUN / "aspect_by_game.csv").exists():
    run3()
