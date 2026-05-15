from pathlib import Path
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

try:
    from wordcloud import WordCloud
except ImportError:  # pragma: no cover
    WordCloud = None


LABEL_ORDER = ["positif", "netral", "negatif"]


def fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    plt.close(fig)
    return img_str


def _normalisasi_label(nilai) -> str:
    label = str(nilai).lower().strip()
    return label if label in LABEL_ORDER else ""


def _cari_kolom(df: pd.DataFrame, kandidat: list[str]) -> str | None:
    kolom_map = {str(kolom).lower().strip(): kolom for kolom in df.columns}
    for nama in kandidat:
        if nama in kolom_map:
            return kolom_map[nama]
    return None


def _hasil_kosong(pesan: str = "") -> dict:
    return {
        "plot_bar": "",
        "plot_pie": "",
        "wc_positif": "",
        "wc_negatif": "",
        "jml_positif": 0,
        "jml_netral": 0,
        "jml_negatif": 0,
        "total": 0,
        "label_sumber": "-",
        "kolom_teks": "-",
        "pesan": pesan,
    }


def _pilih_kolom_label(df: pd.DataFrame) -> str | None:
    for kolom in ["label_sentimen", "label_otomatis", "label_pakar"]:
        if kolom in df.columns:
            label_bersih = df[kolom].apply(_normalisasi_label)
            if label_bersih.ne("").sum() > 0:
                df[kolom] = label_bersih
                return kolom
    return None


def _pilih_kolom_teks(df: pd.DataFrame) -> str | None:
    return _cari_kolom(
        df,
        ["text_clean", "text_stopword", "text_stemmed", "full_text_bersih", "full_text", "text"],
    )


def _buat_bar_chart(counts: pd.Series) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    warna = ["#059669", "#64748b", "#e11d48"]
    bars = ax.bar(["Positif", "Netral", "Negatif"], counts.tolist(), color=warna)
    ax.set_title("Distribusi Sentimen")
    ax.set_ylabel("Jumlah Data")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            str(int(bar.get_height())),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    return fig_to_base64(fig)


def _buat_pie_chart(counts: pd.Series) -> str:
    if counts.sum() == 0:
        return ""

    fig, ax = plt.subplots(figsize=(5, 4))
    warna = ["#10b981", "#94a3b8", "#fb7185"]
    ax.pie(
        counts.tolist(),
        labels=["Positif", "Netral", "Negatif"],
        autopct="%1.1f%%",
        startangle=90,
        colors=warna,
        textprops={"fontsize": 9},
    )
    ax.set_title("Proporsi Sentimen")
    return fig_to_base64(fig)


def _buat_wordcloud(teks: str, background: str, colormap: str) -> str:
    if WordCloud is None or not teks.strip():
        return ""

    fig = plt.figure(figsize=(8, 4))
    wordcloud = WordCloud(
        width=900,
        height=450,
        background_color=background,
        colormap=colormap,
        max_words=120,
        collocations=False,
    ).generate(teks)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig_to_base64(fig)


def jalankan_visualisasi(input_path: str | Path = "data_berlabel.csv") -> dict:
    input_path = Path(input_path)
    if not input_path.exists():
        return _hasil_kosong("File data_berlabel.csv belum ada.")

    df = pd.read_csv(input_path)
    kolom_label = _pilih_kolom_label(df)
    kolom_teks = _pilih_kolom_teks(df)

    if kolom_label is None:
        return _hasil_kosong("Kolom label sentimen belum tersedia.")
    if kolom_teks is None:
        return _hasil_kosong("Kolom teks belum tersedia.")

    df[kolom_label] = df[kolom_label].apply(_normalisasi_label)
    df = df[(df[kolom_label] != "") & df[kolom_teks].notna()].copy()

    counts = df[kolom_label].value_counts().reindex(LABEL_ORDER, fill_value=0)
    teks_positif = " ".join(df[df[kolom_label] == "positif"][kolom_teks].astype(str))
    teks_negatif = " ".join(df[df[kolom_label] == "negatif"][kolom_teks].astype(str))

    return {
        "plot_bar": _buat_bar_chart(counts),
        "plot_pie": _buat_pie_chart(counts),
        "wc_positif": _buat_wordcloud(teks_positif, "white", "Greens"),
        "wc_negatif": _buat_wordcloud(teks_negatif, "white", "Reds"),
        "jml_positif": int(counts["positif"]),
        "jml_netral": int(counts["netral"]),
        "jml_negatif": int(counts["negatif"]),
        "total": int(counts.sum()),
        "label_sumber": kolom_label,
        "kolom_teks": kolom_teks,
        "pesan": "",
    }
