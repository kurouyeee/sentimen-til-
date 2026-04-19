import pandas as pd
import matplotlib
# PENTING UNTUK WEB: Gunakan 'Agg' agar Matplotlib tidak mencoba membuka jendela popup
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Tambahan untuk web (mengubah gambar jadi teks)
import io
import base64

def jalankan_komparasi():
    print("Membaca data untuk komparasi...")
    # 1. PERSIAPAN DATA
    df = pd.read_csv('data_berlabel.csv')
    df = df.dropna(subset=['full_text_bersih', 'label_sentimen'])

    X = df['full_text_bersih']
    y = df['label_sentimen']

    # 2. EKSTRAKSI FITUR (TF-IDF)
    vectorizer = TfidfVectorizer()
    X_vector = vectorizer.fit_transform(X)

    # 3. PEMBAGIAN DATA
    X_train, X_test, y_train, y_test = train_test_split(X_vector, y, test_size=0.2, random_state=42)

    # 4. MODELING: NAIVE BAYES
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred) * 100

    # 5. MODELING: SVM
    svm_model = SVC(kernel='linear')
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred) * 100

    # 6. VISUALISASI KOMPARASI (UNTUK WEB)
    algoritma = ['Naive Bayes', 'SVM']
    akurasi = [nb_accuracy, svm_accuracy]

    # Membuat figure
    plt.figure(figsize=(8, 5))
    bars = plt.bar(algoritma, akurasi, color=['#3498db', '#e74c3c'])
    plt.ylim(0, 110)
    plt.title('Perbandingan Akurasi Naive Bayes vs SVM', fontsize=14)
    plt.ylabel('Akurasi (%)', fontsize=12)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # --- BAGIAN PALING PENTING UNTUK WEB ---
    # Simpan plot ke dalam buffer memori, ubah jadi base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    grafik_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close() # Tutup memori gambar agar tidak bocor

    # 7. KEMBALIKAN DATA KE main.py
    # Kita mengirimkan akurasi dan gambarnya sekaligus
    return {
        "akurasi_nb": round(nb_accuracy, 2),
        "akurasi_svm": round(svm_accuracy, 2),
        "grafik": grafik_base64
    }