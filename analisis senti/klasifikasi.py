import pandas as pd

# ---------------------------------------------------------
# 1. PERSIAPAN KAMUS LEXICON (SENTIMEN)
# Ditaruh di luar agar hanya dimuat sekali dan siap dipakai
# ---------------------------------------------------------
kata_positif = [
    "bagus", "baik", "bantu", "aman", "cepat", "mudah", "murah", 
    "untung", "keren", "puas", "suka", "cinta", "hebat", "mantap", 
    "selamat", "ramah", "solusi", "terima kasih", "sukses"
]

kata_negatif = [
    "buruk", "jelek", "susah", "lambat", "mahal", "rugi", "kecewa", 
    "marah", "benci", "rusak", "parah", "hancur", "gagal", "bohong", 
    "tipu", "sulit", "sesal", "lamban"
]

# ---------------------------------------------------------
# 2. FUNGSI DASAR
# ---------------------------------------------------------
def hitung_skor(teks):
    skor = 0
    if isinstance(teks, str):
        # Mengubah teks menjadi huruf kecil semua (lowercase) agar cocok dengan kamus
        kata_kata = teks.lower().split()
        for kata in kata_kata:
            if kata in kata_positif:
                skor += 1
            elif kata in kata_negatif:
                skor -= 1
    return skor

def tentukan_label(skor):
    if skor > 0:
        return "Positif"
    elif skor < 0:
        return "Negatif"
    else:
        return "Netral"

# ---------------------------------------------------------
# 3. FUNGSI UNTUK WEB (YANG BISA DIPANGGIL MAIN.PY)
# ---------------------------------------------------------

# Fungsi A: Untuk memproses seluruh file CSV
def jalankan_klasifikasi():
    print("Menghitung sentimen file CSV...")
    
    # Membaca data
    df = pd.read_csv('data_jadi.csv')
    df = df.dropna(subset=['full_text_bersih'])

    # Menghitung skor & label
    df['skor_sentimen'] = df['full_text_bersih'].apply(hitung_skor)
    df['label_sentimen'] = df['skor_sentimen'].apply(tentukan_label)

    # Menyimpan hasil
    df.to_csv('data_berlabel.csv', index=False)
    print("Berhasil! Data telah disimpan ke 'data_berlabel.csv'")
    
    # Mengembalikan dataframe agar bisa dimunculkan ke tabel di web
    return df

# Fungsi B: Khusus jika user mengetik 1 kalimat di Web
def prediksi_teks_tunggal(teks_input):
    skor = hitung_skor(teks_input)
    label = tentukan_label(skor)
    return label