from pathlib import Path
import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


LABEL_ORDER = ["positif", "netral", "negatif"]


def _empty_result(pesan: str) -> dict:
    return {
        "akurasi_nb": 0,
        "akurasi_svm": 0,
        "grafik": "",
        "total_data": 0,
        "data_latih": 0,
        "data_uji": 0,
        "rasio_uji": 20,
        "label_sumber": "-",
        "kolom_teks": "-",
        "pesan": pesan,
        "tabel_prediksi": [],
        "laporan_nb": {},
        "laporan_svm": {},
    }


def _fig_to_base64(fig) -> str:
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


def _pilih_kolom_teks(df: pd.DataFrame) -> str | None:
    return _cari_kolom(
        df,
        ["text_stemmed", "text_stopword", "text_clean", "full_text_bersih", "full_text", "text"],
    )


def _pilih_kolom_label(df: pd.DataFrame) -> str | None:
    for kolom in ["label_pakar", "label_sentimen", "label_otomatis"]:
        if kolom in df.columns:
            label_bersih = df[kolom].apply(_normalisasi_label)
            if label_bersih.ne("").sum() > 0 and label_bersih.nunique() >= 2:
                df[kolom] = label_bersih
                return kolom
    return None


def _buat_grafik(nb_accuracy: float, svm_accuracy: float) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    algoritma = ["Naive Bayes", "SVM"]
    akurasi = [nb_accuracy, svm_accuracy]
    warna = ["#2563eb", "#dc2626"]
    bars = ax.bar(algoritma, akurasi, color=warna, width=0.55)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Perbandingan Akurasi Model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(yval + 2, 102),
            f"{yval:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    return _fig_to_base64(fig)


def jalankan_komparasi(
    input_path: str | Path = "data_berlabel.csv",
    output_path: str | Path = "hasil_klasifikasi.csv",
) -> dict:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        return _empty_result("File data_berlabel.csv belum ada.")

    df = pd.read_csv(input_path)
    kolom_teks = _pilih_kolom_teks(df)
    kolom_label = _pilih_kolom_label(df)

    if kolom_teks is None:
        return _empty_result("Kolom teks tidak ditemukan.")
    if kolom_label is None:
        return _empty_result("Minimal perlu dua kelas label untuk melatih model.")

    df_model = df.copy()
    df_model[kolom_label] = df_model[kolom_label].apply(_normalisasi_label)
    df_model[kolom_teks] = df_model[kolom_teks].fillna("").astype(str).str.strip()
    df_model = df_model[(df_model[kolom_label] != "") & (df_model[kolom_teks] != "")]

    total_data = len(df_model)
    if total_data < 4:
        return _empty_result("Jumlah data terlalu sedikit untuk pembagian latih/uji.")

    jumlah_kelas = df_model[kolom_label].nunique()
    if jumlah_kelas < 2:
        return _empty_result("Variasi sentimen kurang. Model perlu minimal dua kelas.")

    min_per_kelas = df_model[kolom_label].value_counts().min()
    if min_per_kelas < 2:
        return _empty_result("Setiap kelas label perlu minimal dua data agar pembagian latih/uji stabil.")

    X = df_model[kolom_teks]
    y = df_model[kolom_label]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_vector = vectorizer.fit_transform(X)

    test_size = min(0.4, max(0.2, jumlah_kelas / total_data))
    split_result = train_test_split(
        X_vector,
        y,
        df_model.index,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )
    X_train, X_test, y_train, y_test, idx_train, idx_test = split_result

    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred) * 100

    svm_model = SVC(kernel="linear", C=1.0)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred) * 100

    hasil_uji = df_model.loc[idx_test].copy()
    hasil_uji["label_prediksi_nb"] = nb_pred
    hasil_uji["label_prediksi_svm"] = svm_pred
    hasil_uji = hasil_uji.reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hasil_uji.to_csv(output_path, index=False)

    laporan_nb = classification_report(y_test, nb_pred, output_dict=True, zero_division=0)
    laporan_svm = classification_report(y_test, svm_pred, output_dict=True, zero_division=0)

    kolom_tabel = [
        kolom for kolom in ["username", "full_text", kolom_teks, kolom_label, "label_prediksi_nb", "label_prediksi_svm"]
        if kolom in hasil_uji.columns
    ]
    tabel_prediksi = hasil_uji[kolom_tabel].head(100).to_dict(orient="records")

    return {
        "akurasi_nb": round(nb_accuracy, 2),
        "akurasi_svm": round(svm_accuracy, 2),
        "grafik": _buat_grafik(nb_accuracy, svm_accuracy),
        "total_data": total_data,
        "data_latih": int(len(idx_train)),
        "data_uji": int(len(idx_test)),
        "rasio_uji": round(len(idx_test) / total_data * 100, 2),
        "label_sumber": kolom_label,
        "kolom_teks": kolom_teks,
        "pesan": "",
        "tabel_prediksi": tabel_prediksi,
        "laporan_nb": laporan_nb,
        "laporan_svm": laporan_svm,
    }
