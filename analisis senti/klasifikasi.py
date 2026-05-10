import re
from pathlib import Path

import pandas as pd


KATA_POSITIF = {
    "bagus", "baik", "bantu", "aman", "cepat", "mudah", "murah",
    "untung", "keren", "puas", "suka", "cinta", "hebat", "mantap",
    "selamat", "ramah", "solusi", "sukses", "tanggap", "seru",
    "cekatan", "sigap", "pahlawan", "berani", "jasa", "jasanya",
    "terbantu", "profesional", "siaga", "responsif", "salut",
    "respect", "makasih", "terimakasih", "ngebantu", "menolong",
    "pertolongan", "penyelamat", "amanah", "sopan", "santun",
    "mendidik", "bermanfaat", "berguna", "lucu", "gesit", "mulia",
    "berjasa", "tertolong", "terkendali",
}

KATA_NEGATIF = {
    "buruk", "jelek", "susah", "lambat", "mahal", "rugi", "kecewa",
    "marah", "benci", "parah", "gagal", "bohong", "tipu", "sulit",
    "sesal", "lamban", "telat", "lama", "bahaya", "korban", "panik",
    "takut", "musibah", "lambatnya", "kurang", "payah", "lelet",
    "nyebelin", "mengerikan", "hancur", "macet", "halang", "rusuh",
    "mainmain", "kesel", "kesal", "najis", "bajingan", "keparat",
    "bangsat", "goblok", "tolol", "terbakar", "kebakaran",
}

FRASA_POSITIF = {
    "terima kasih", "terima kasih banyak", "gerak cepat", "respon cepat",
    "cepat tanggap", "sangat membantu",
}

FRASA_NEGATIF = {
    "terlalu lama", "tidak tanggap", "kurang cepat", "belum datang",
    "tidak datang", "susah dihubungi",
}

LABEL_ORDER = ["positif", "netral", "negatif"]


def _bersihkan_teks(teks: str) -> str:
    teks = str(teks).lower()
    teks = re.sub(r"http\S+|www\S+|https\S+", " ", teks)
    teks = re.sub(r"@\w+|#", " ", teks)
    teks = re.sub(r"[^a-zA-Z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


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


def pastikan_kolom_teks(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    df = df.copy()
    kolom_teks = _cari_kolom_teks(df)
    if kolom_teks is None:
        raise ValueError("Dataset harus memiliki kolom teks seperti full_text, text, atau text_clean.")

    if "full_text" not in df.columns:
        df["full_text"] = df[kolom_teks].astype(str)

    if "full_text_bersih" not in df.columns:
        df["full_text_bersih"] = df[kolom_teks].astype(str)

    if "username" not in df.columns:
        df["username"] = "Anonim"

    return df, kolom_teks


def hitung_skor(teks) -> int:
    if not isinstance(teks, str) or not teks.strip():
        return 0

    teks_bersih = _bersihkan_teks(teks)
    token = teks_bersih.split()
    skor = 0

    skor += sum(1 for frasa in FRASA_POSITIF if frasa in teks_bersih)
    skor -= sum(1 for frasa in FRASA_NEGATIF if frasa in teks_bersih)

    for kata in token:
        if kata in KATA_POSITIF:
            skor += 1
        elif kata in KATA_NEGATIF:
            skor -= 1

    return skor


def tentukan_label(skor: int) -> str:
    if skor > 0:
        return "positif"
    if skor < 0:
        return "negatif"
    return "netral"


def prediksi_teks_tunggal(teks_input: str) -> str:
    return tentukan_label(hitung_skor(teks_input))


def jalankan_klasifikasi(
    input_path: str | Path = "data_sementara.csv",
    output_path: str | Path = "data_berlabel.csv",
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"File {input_path.name} tidak ditemukan.")

    df = pd.read_csv(input_path)
    df, kolom_teks = pastikan_kolom_teks(df)
    df = df.dropna(subset=[kolom_teks]).copy()

    df["skor_sentimen"] = df[kolom_teks].apply(hitung_skor)
    df["label_sentimen"] = df["skor_sentimen"].apply(tentukan_label)

    if "label_otomatis" not in df.columns:
        df["label_otomatis"] = df["label_sentimen"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
