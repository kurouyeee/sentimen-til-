import pandas as pd

from klasifikasi import prediksi_teks_tunggal
from kompirasi import jalankan_komparasi
from visualisasi import jalankan_visualisasi


def prediksi_teks(teks_baru: str) -> str:
    return prediksi_teks_tunggal(teks_baru)


def data_komparasi(input_path: str = "data_berlabel.csv") -> dict:
    return jalankan_komparasi(input_path=input_path)


def buat_visualisasi_wordcloud(df: pd.DataFrame | None = None) -> str:
    if df is not None:
        df.to_csv("data_berlabel.csv", index=False)
    hasil = jalankan_visualisasi()
    return hasil.get("wc_positif") or hasil.get("wc_negatif") or ""
