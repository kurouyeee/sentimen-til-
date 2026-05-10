import re
import pandas as pd

# ==========================================
# 1. KAMUS MANUAL (TIDAK BUTUH LIBRARY)
# ==========================================

# Kamus Normalisasi (Bahasa Gaul/Slang -> Baku)
kamus_alay = {
    "yg": "yang", "dgn": "dengan", "bgmn": "bagaimana",
    "tdk": "tidak", "gak": "tidak", "ga": "tidak", "g": "tidak",
    "jg": "juga", "kalo": "kalau", "sampe": "sampai",
    "bgt": "banget", "udh": "sudah", "sdh": "sudah",
    "tp": "tapi", "dlm": "dalam", "utk": "untuk",
    "sy": "saya", "aku": "saya", "gue": "saya", "gw": "saya"
}

# Daftar Stopword Manual (Kata hubung yang tidak punya sentimen)
# Anda bisa menambahkan kata lain ke dalam daftar ini
daftar_stopword = [
    "yang", "di", "ke", "dari", "pada", "dalam", "untuk", 
    "dengan", "dan", "atau", "ini", "itu", "juga", "sudah", 
    "saya", "dia", "mereka", "kita", "kami", "adalah", "sebuah"
]

# ==========================================
# 2. FUNGSI PEMBERSIHAN MURNI (PYTHON BAWAAN)
# ==========================================

def clean_text(text):
    if type(text) != str: return ""
    text = text.lower() # Ubah ke huruf kecil
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Hapus Link/URL
    text = re.sub(r'\@\w+|\#', '', text) # Hapus @mention dan hashtag
    text = re.sub(r'[^a-zA-Z\s]', ' ', text) # Hapus angka dan tanda baca
    text = re.sub(r'\s+', ' ', text).strip() # Hapus spasi berlebih
    return text

def normalize_text(text):
    words = text.split()
    # Jika kata ada di kamus alay, ganti. Jika tidak, biarkan.
    normalized_words = [kamus_alay.get(word, word) for word in words]
    return ' '.join(normalized_words)

def remove_stopwords(text):
    words = text.split()
    # Hanya simpan kata yang TIDAK ada di daftar_stopword
    filtered_words = [word for word in words if word not in daftar_stopword]
    return ' '.join(filtered_words)

def stem_text_manual(text):
    
    words = text.split()
    stemmed_words = []
    
    for word in words:
        # Hapus partikel dan kepunyaan (lah, kah, nya, ku, mu)
        word = re.sub(r'(lah|kah|nya|ku|mu)$', '', word)
        # Hapus awalan sederhana (me, di, pe, ter, ke)
        word = re.sub(r'^(me|di|pe|ter|ke)', '', word)
        # Hapus akhiran (kan, i)
        word = re.sub(r'(kan|i)$', '', word)
        stemmed_words.append(word)
        
    return ' '.join(stemmed_words)

# ==========================================
# 3. FUNGSI UTAMA UNTUK DIPANGGIL OLEH MAIN.PY
# ==========================================

def bersihkan_semua_data(df):
    # Cari nama kolom teks (biasanya full_text atau text)
    kolom_target = 'full_text' if 'full_text' in df.columns else 'text' if 'text' in df.columns else None
    
    if not kolom_target:
        raise ValueError("Gagal! Dataset tidak memiliki kolom 'full_text' atau 'text'.")

    # Jalankan proses berurutan
    df['text_clean'] = df[kolom_target].apply(clean_text)
    df['text_normal'] = df['text_clean'].apply(normalize_text)
    df['text_stopword'] = df['text_normal'].apply(remove_stopwords)
    df['text_stemmed'] = df['text_stopword'].apply(stem_text_manual)
    
    # Jaga-jaga agar HTML tidak error jika kolom username kosong
    if 'username' not in df.columns:
        df['username'] = 'Anonim'
        
    if 'full_text' not in df.columns:
        df['full_text'] = df[kolom_target]

    return df