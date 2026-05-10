import pandas as pd

from klasifikasi import prediksi_teks_tunggal


LABEL_VALID = {"positif", "netral", "negatif"}


def _cari_kolom_teks(df: pd.DataFrame) -> str | None:
    kandidat = [
        "text_stemmed",
        "text_stopword",
        "text_clean",
        "full_text_bersih",
        "full_text",
        "text",
    ]
    kolom_map = {str(kolom).lower().strip(): kolom for kolom in df.columns}
    for nama in kandidat:
        if nama in kolom_map:
            return kolom_map[nama]
    return None


def normalisasi_label(nilai, default: str = "netral") -> str:
    label = str(nilai).lower().strip()
    if label in LABEL_VALID:
        return label
    return default


def jalankan_labeling_otomatis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    kolom_teks = _cari_kolom_teks(df)

    if kolom_teks is None:
        df["label_otomatis"] = "netral"
        return df

    df["label_otomatis"] = df[kolom_teks].fillna("").astype(str).apply(prediksi_teks_tunggal)

    if "label_pakar" in df.columns:
        df["label_pakar"] = df["label_pakar"].apply(normalisasi_label)

    return df
