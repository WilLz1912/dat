"""English figures for the LaTeX report -> report/latex/figures/.
Run from the project root:  python -I report/latex/make_figures_en.py
Same data and numbers as report/make_figures.py (Vietnamese draft), plus four extra figures."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

OUT = Path("report/latex/figures"); OUT.mkdir(parents=True, exist_ok=True)
RUN = Path("data/labels/run3")
BLUE, GREY, LIGHT, INK, MUTED, RED = "#4C78A8", "#8A94A6", "#C9CFD8", "#1F2733", "#5B6472", "#C8553D"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 200, "font.size": 10, "text.color": INK, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": INK,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LIGHT, "axes.grid": True, "grid.color": "#E6E9EE", "axes.axisbelow": True,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False})
EN = dict(GFX="Graphics", AUD="Audio", PLAY="Gameplay", STORY="Story", PERF="Performance, bugs", PRICE="Price, monetisation", ONLINE="Online, community", DEV="Developer")
VI2EN = {"Đồ hoạ": "Graphics", "Âm thanh": "Audio", "Gameplay": "Gameplay", "Cốt truyện": "Story", "Hiệu năng, lỗi": "Performance, bugs",
         "Giá, trả phí": "Price, monetisation", "Online, cộng đồng": "Online, community", "Nhà phát triển": "Developer"}


def method_en(n):
    """Translate method names stored in the run-3 result files."""
    rep = [("Từ khoá trong đầu vào", "Keywords-in-input"), ("Hợp từ khoá với DistilBERT", "Union of keywords and DistilBERT"), ("Từ khoá", "Keywords"),
           ("(nhãn cũ)", "(old labels)"), ("(+ 50% train 2)", "(+50% Train 2)"), ("(+ toàn bộ train 2)", "(+ all Train 2)"), ("(+ câu trái cảm xúc)", "(+ conflict sentences)"),
           ("+ M4 (bản cũ)", "+ M4 (first version)"), ("DistilBERT cả câu", "whole-sentence DistilBERT"), ("theo cặp", "pairwise")]
    for a, b in rep:
        n = n.replace(a, b)
    return n


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", facecolor="white"); plt.close(fig); print(name)


def hbars(ax, labels, series, colors, fmt="{:.3f}", xlim=None, h=.8):
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


def flow():
    rows = [("Review-level tasks", ["Steam API\n10,000 reviews, 99 games", "Cleaning\n9,971 reviews", "EDA, feature\nselection", "Keras models\nM1–M5", "Sentiment, suspicious\nreviews, helpfulness"]),
            ("Extension (instructor feedback)", ["LLM-assisted labels\n500 reviews, 4,461 sent.", "Aspect keyword\nlists", "DistilBERT\naspect detection", "Pairwise DistilBERT\naspect polarity", "Aspect sentiment\nper game"]),
            ("Game recommendation", ["Data Stream v2\n275 reviews, 92 games", "Genre and\npublisher tagging", "Game groups\n(graph)", "Same-group\ncandidates", "Recommendations\nfor a user"])]
    fig, ax = plt.subplots(figsize=(12, 4.1)); ax.axis("off"); ax.set_xlim(-.05, 11.4); ax.set_ylim(0, 4.05)
    w, hh, step = 1.95, .72, 2.35
    for r, (title, steps) in enumerate(rows):
        y = 3.0 - r * 1.33
        ax.text(0, y + hh + .08, title, fontweight="bold", fontsize=10)
        for i, s in enumerate(steps):
            x = i * step
            ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.06", fc="#EEF3F8" if i < 4 else BLUE, ec="none"))
            ax.text(x + w / 2, y + hh / 2, s, ha="center", va="center", fontsize=8.6, color=INK if i < 4 else "white")
            if i:
                ax.annotate("", (x - .03, y + hh / 2), (x - step + w + .03, y + hh / 2), arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.2))
    save(fig, "fig01_workflow")


def eda():
    d = pd.read_csv("data/steam_reviews_clean.csv")
    pt = pd.cut(d.playtime_forever_hours, [-1, 2, 10, 50, 200, 1000, 1e9], labels=["< 2", "2–10", "10–50", "50–200", "200–1000", "≥ 1000"], right=False)
    wc = pd.cut(d.word_count, [0, 5, 10, 20, 50, 100, 1e9], labels=["≤ 5", "6–10", "11–20", "21–50", "51–100", "> 100"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), sharey=True)
    for ax, g, t, xl in [(axs[0], pt, "By playtime", "Playtime (hours)"), (axs[1], wc, "By review length", "Number of words")]:
        s = d.groupby(g, observed=True).label.mean() * 100
        b = ax.bar(s.index.astype(str), s.values, .62, color=BLUE)
        ax.bar_label(b, fmt="%.0f%%", fontsize=9, color=MUTED, padding=2)
        ax.set_title(t); ax.set_xlabel(xl); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False); ax.tick_params(axis="x", labelsize=8.5)
    axs[0].set_ylabel("Share of positive reviews")
    save(fig, "fig02_eda")


def sentiment():
    names = ["Meta features only + LR", "M1 Embedding + Pooling", "M3 CNN 1D", "M2 BiLSTM", "M4 TF-IDF + Dense", "M5 Ensemble M2 + M4", "TF-IDF + LR", "TF-IDF + meta + LR", "DistilBERT"]
    A = [.716, .836, .871, .884, .891, .902, .903, .908, .937]; B = [.711, .883, .883, .893, .893, .905, .906, .909, .930]
    fig, ax = plt.subplots(figsize=(7.4, 4)); y = np.arange(len(names))[::-1]
    ax.hlines(y, [min(a, b) for a, b in zip(A, B)], [max(a, b) for a, b in zip(A, B)], color=LIGHT, lw=2)
    ax.scatter(A, y, s=62, color=BLUE, label="Scenario A: known games", zorder=3, edgecolor="white")
    ax.scatter(B, y, s=62, color=GREY, marker="s", label="Scenario B: unseen games", zorder=3, edgecolor="white")
    ax.set_yticks(y, names); ax.set_xlabel("Test ROC-AUC"); ax.set_xlim(.69, .95); ax.grid(axis="y", visible=False); ax.legend(loc="center left", bbox_to_anchor=(.08, .3))
    ax.text(.9335, y[-1] + .32, ".937 / .930", ha="center", fontsize=8.5, color=MUTED)
    save(fig, "fig03_sentiment_auc")
    fig, ax = plt.subplots(figsize=(7, 2.9))
    hbars(ax, ["Genuine model error", "Steam label contradicts text", "Mixed review", "Cannot be determined"], {"M4": [46, 13, 10, 28], "DistilBERT": [26, 14, 7, 22]}, [LIGHT, BLUE], "{:.0f}", (0, 52))
    ax.set_xlabel("Misclassified reviews among the 500 labelled test reviews")
    save(fig, "fig04_error_causes")


def run3():
    D = pd.read_csv(RUN / "nhan_khia_canh.csv")
    cv = D[D.phương_án.str.startswith("Từ khoá trong")].groupby("số_câu")[["test2", "dev"]].mean()
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.plot(cv.index, cv.test2, marker="o", ms=7, lw=2, color=BLUE, label="Keywords-in-input")
    pl = D[D.phương_án.str.startswith("DistilBERT")].groupby("số_câu").test2.mean()
    ax.plot(pl.index, pl.values, marker="s", ms=7, lw=2, color=GREY, label="DistilBERT")
    kw = D[D.phương_án == "Từ khoá"].test2.iloc[0]
    ax.axhline(kw, color=GREY, lw=1.2, ls="--", label="Keywords only")
    for x, v in cv.test2.items():
        ax.annotate(f"{v + 1e-9:.3f}", (x, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=MUTED)
    ax.set(xlabel="Number of training sentences", ylabel="Micro F1 on Test 2", ylim=(.4, .85)); ax.legend(loc="upper left", ncol=3, fontsize=9)
    save(fig, "fig05_learning_curve")

    m = D.groupby("phương_án")[["test2", "test1", "số_câu"]].mean()
    m = m[(m.số_câu == 0) | (m.số_câu == D.số_câu.max())].sort_values("test2", ascending=False)
    lab = [method_en(n.split(" (")[0]) for n in m.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hbars(ax, lab, {"Test 2 (keywords frozen beforehand)": m.test2.values, "Test 1 (keywords written after reading it)": m.test1.values}, [BLUE, LIGHT], "{:.3f}", (0, .9))
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2, fontsize=9); ax.set_xlabel("Aspect detection micro F1")
    save(fig, "fig06_aspect_detection")

    per = pd.read_csv(RUN / "f1_theo_khia_canh.csv", index_col=0)
    cols = [c for c in per.columns if c != "số câu thật"]
    M = per[cols].values.astype(float)
    xl = {"Từ khoá": "Keywords", "TF-IDF + LR": "TF-IDF\n+ LR", "DistilBERT": "DistilBERT", "Từ khoá trong đầu vào": "Keywords-\nin-input"}
    fig, ax = plt.subplots(figsize=(7.2, 3.5)); ax.grid(False)
    ax.imshow(M, cmap="Blues", vmin=.2, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)), [xl[c.split(" (")[0]] for c in cols], fontsize=9)
    ax.xaxis.tick_top(); ax.set_yticks(range(len(per)), [f"{VI2EN[i]} ({n})" for i, n in zip(per.index, per["số câu thật"])], fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > .68 else INK)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    save(fig, "fig07_f1_per_aspect")

    P = pd.read_csv(RUN / "khen_che_theo_cap.csv", index_col=0)
    k = json.load(open(RUN / "chon.json", encoding="utf-8"))["BEST_K"]
    col = [c for c in P.columns if c.startswith("Theo cặp") and ("trái" in c) == (k == "D")][0]
    grp = ["Tất cả", "Câu 1 khía cạnh", "Nhiều khía cạnh, cùng cảm xúc", "Nhiều khía cạnh, trái cảm xúc", "Thật sự là Khen", "Thật sự là Chê"]
    grp_en = ["All pairs", "Single-aspect sentences", "Multi-aspect, same polarity", "Multi-aspect, conflicting polarity", "Gold positive", "Gold negative"]
    Q = P.loc[grp]
    fig, ax = plt.subplots(figsize=(7.4, 4))
    hbars(ax, [f"{g} (n={int(n)})" for g, n in zip(grp_en, Q["số cặp"])], {"M4 whole sentence": Q["M4 cả câu"].values, "DistilBERT whole sentence": Q["DistilBERT cả câu"].values, "Pairwise DistilBERT": Q[col].values}, [LIGHT, GREY, BLUE], "{:.2f}", (0, 1.08))
    ax.legend(loc="upper center", bbox_to_anchor=(.45, -.13), ncol=3, fontsize=9); ax.set_xlabel("Accuracy")
    save(fig, "fig08_polarity_by_group")

    C = pd.read_csv(RUN / "ca_duong_ong.csv", index_col=0); C = C[C.tập == "Test 2"]
    names = ["Keywords + M4 (first version)", "Keywords + whole-sentence DistilBERT", "Keywords + pairwise DistilBERT", "Keywords-in-input + pairwise DistilBERT"]
    fig, ax = plt.subplots(figsize=(7, 2.5))
    hbars(ax, names, {"F1": C.F1.values}, [BLUE], "{:.3f}", (0, .85)); ax.set_xlabel("End-to-end F1 on Test 2 (aspect and polarity both correct)")
    save(fig, "fig09_pipeline")

    A = pd.read_csv(RUN / "aspect_by_game.csv"); order = list(EN)
    top = A.groupby("game").so_cau.sum().sort_values(ascending=False).head(15).index
    H = A[A.game.isin(top)].pivot(index="game", columns="aspect", values="pct_che").reindex(index=top, columns=order)
    Nc = A[A.game.isin(top)].pivot(index="game", columns="aspect", values="so_cau").reindex(index=top, columns=order)
    H = H.where(Nc >= 5)
    fig, ax = plt.subplots(figsize=(8.6, 5.2)); ax.grid(False)
    ax.imshow(H.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(8), [EN[a].replace(", ", ",\n") for a in order], fontsize=9); ax.xaxis.tick_top(); ax.set_yticks(range(len(top)), [g[:30] for g in top], fontsize=9)
    for i in range(H.shape[0]):
        for j in range(8):
            v = H.values[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.0f}", ha="center", va="center", fontsize=9, color=GREY if np.isnan(v) else ("white" if v > 60 else INK))
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    save(fig, "fig10_aspect_by_game")

    T = pd.read_csv(RUN / "aspect_tong.csv", index_col=0)
    T.index = [VI2EN[i] for i in T.index]; T = T.sort_values("pct_che")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True, gridspec_kw=dict(width_ratios=[1.2, 1]))
    b = axs[0].barh(T.index, T.so_cau, .62, color=GREY); axs[0].bar_label(b, labels=[f"{v:,}" for v in T.so_cau], fontsize=8.5, color=MUTED, padding=2)
    axs[0].set_title("Detected aspect mentions"); axs[0].set_xlim(0, T.so_cau.max() * 1.18); axs[0].grid(axis="y", visible=False)
    b = axs[1].barh(T.index, T.pct_che, .62, color=BLUE); axs[1].bar_label(b, fmt="%.1f%%", fontsize=8.5, color=MUTED, padding=2)
    axs[1].set_title("Share of predicted negative mentions"); axs[1].set_xlim(0, 110); axs[1].grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "fig11_aspect_totals")


def recommend():
    g = pd.read_csv("data/v2_games.csv"); u = pd.read_csv("data/v2_reviews_clean.csv")
    s = g.genre.value_counts().sort_values()
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    b = axs[0].barh(s.index, s.values, .68, color=BLUE); axs[0].bar_label(b, fontsize=8.5, color=MUTED, padding=2); axs[0].grid(axis="y", visible=False)
    axs[0].set_title("Games per genre"); axs[0].tick_params(axis="y", labelsize=8.5)
    order = u.groupby("reviewer_tier").games_owned.median().sort_values().index
    t = (1 - u.groupby("reviewer_tier").voted_up.mean()).reindex(order) * 100
    tier = {"Ít game nhất": "Fewest games", "Tầm trung": "Mid-range", "Nhiều game nhất": "Most games"}
    b = axs[1].bar([tier.get(x, x) for x in t.index], t.values, .55, color=BLUE)
    axs[1].bar_label(b, fmt="%.1f%%", fontsize=9, color=MUTED, padding=2); axs[1].set_title("Negative reviews by reviewer tier"); axs[1].set_ylim(0, 45); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig12_recommendation_data")


def suspicious():
    d = pd.read_csv("data/steam_reviews_suspicious.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw=dict(width_ratios=[1, 1.15]))
    axs[0].hist(d.suspicion_score, bins=40, color=BLUE); axs[0].set_yscale("log")
    axs[0].axvline(.9, color=RED, ls="--", lw=1.2); axs[0].text(.885, axs[0].get_ylim()[1] * .5, "threshold 0.9", ha="right", fontsize=8.5, color=RED)
    axs[0].set(xlabel="Suspicion score (model probability against the Steam label)", ylabel="Reviews (log scale)"); axs[0].set_title("Distribution of suspicion scores")
    g = d.groupby("game_name").flag_mismatch.mean().sort_values(ascending=False).head(10) * 100
    b = axs[1].barh([x.replace("Tom Clancy's ", "")[:30] for x in g.index[::-1]], g.values[::-1], .62, color=BLUE); axs[1].bar_label(b, fmt="%.0f%%", fontsize=8.5, color=MUTED, padding=2)
    axs[1].set_title("Games with the highest share of mismatched reviews"); axs[1].set_xlabel("% of the game's reviews"); axs[1].grid(axis="y", visible=False); axs[1].tick_params(axis="y", labelsize=8.5)
    fig.tight_layout(); save(fig, "fig13_suspicious")


def helpful():
    d = pd.read_csv("data/steam_reviews_clean.csv")
    age = (d.timestamp_created.max() - d.timestamp_created) / 86400
    b_age = pd.cut(age, [-1, 1, 3, 7, 14, 30, 90, 1e5], labels=["≤ 1", "1–3", "3–7", "7–14", "14–30", "30–90", "> 90"])
    has = (d.votes_helpful > 0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[1.4, 1]))
    r = has.groupby(b_age, observed=True).agg(["mean", "size"])
    b = axs[0].bar(r.index.astype(str), r["mean"] * 100, .62, color=BLUE); axs[0].bar_label(b, labels=[f"n={n}" for n in r["size"]], fontsize=8, color=MUTED, padding=2)
    axs[0].set(xlabel="Review age at crawl time (days)", ylabel="% with ≥ 1 helpful vote", ylim=(0, 100)); axs[0].set_title("By review age"); axs[0].grid(axis="x", visible=False)
    s = has.groupby(d.label).mean().rename({0: "Negative", 1: "Positive"}) * 100
    b = axs[1].bar(s.index, s.values, .5, color=[GREY, BLUE]); axs[1].bar_label(b, fmt="%.1f%%", fontsize=9, color=MUTED, padding=2)
    axs[1].set(ylabel="% with ≥ 1 helpful vote", ylim=(0, 100)); axs[1].set_title("By Steam label"); axs[1].grid(axis="x", visible=False)
    fig.tight_layout(); save(fig, "fig14_helpful")


def labels():
    r = pd.read_csv("data/labels/reviews_labels.csv")
    sent = r.sentiment.map(dict(K="Positive", C="Negative", L="Mixed", U="Unclear")).value_counts()
    kind = r.kind.map(dict(BT="Normal", DUA="Joke / meme", MIA="Sarcasm", RAC="Off-topic", NGAN="Too short")).value_counts()
    fig, axs = plt.subplots(1, 2, figsize=(10, 2.8))
    for ax, s, t in [(axs[0], sent, "Sentiment of the review text"), (axs[1], kind, "Review type")]:
        s = s.sort_values()
        b = ax.barh(s.index, s.values, .6, color=BLUE); ax.bar_label(b, fontsize=8.5, color=MUTED, padding=2)
        ax.set_title(t); ax.set_xlim(0, s.max() * 1.15); ax.grid(axis="y", visible=False)
    fig.tight_layout(); save(fig, "fig15_review_labels")


flow(); eda(); sentiment(); recommend(); suspicious(); helpful(); labels()
if (RUN / "aspect_by_game.csv").exists():
    run3()
