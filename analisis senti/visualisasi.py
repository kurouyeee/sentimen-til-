import pandas as pd
import matplotlib
# PENTING: Gunakan 'Agg' agar Matplotlib tidak membuka jendela baru
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64

# Fungsi pembantu untuk mengubah plot Matplotlib menjadi string Base64
def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close(fig) # Tutup figure untuk menghemat memori
    return img_str

def jalankan_visualisasi():
    # 1. BACA DATA
    df = pd.read_csv('data_berlabel.csv') 
    kolom_teks = 'full_text_bersih'

    # ---------------------------------------------------------
    # 2. GRAFIK DISTRIBUSI SENTIMEN (BAR CHART)
    # ---------------------------------------------------------
    fig1 = plt.figure(figsize=(8, 6))
    sns.countplot(x='label_sentimen', data=df, palette='viridis', order=['Positif', 'Netral', 'Negatif'])
    plt.title('Distribusi Sentimen Pemadam Kebakaran', fontsize=16)
    plt.xlabel('Sentimen', fontsize=12)
    plt.ylabel('Jumlah Cuitan', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    gambar_bar = fig_to_base64(fig1)

    # ---------------------------------------------------------
    # 3. WORD CLOUD
    # ---------------------------------------------------------
    teks_positif = ' '.join(df[df['label_sentimen'] == 'Positif'][kolom_teks].dropna())
    teks_negatif = ' '.join(df[df['label_sentimen'] == 'Negatif'][kolom_teks].dropna())

    # Word Cloud Positif
    gambar_wc_pos = ""
    if len(teks_positif) > 0:
        fig_pos = plt.figure(figsize=(10, 5))
        wc_pos = WordCloud(width=800, height=400, background_color='white', colormap='hsv', max_words=100).generate(teks_positif)
        plt.imshow(wc_pos, interpolation='bilinear')
        plt.title('Word Cloud Sentimen Positif', fontsize=16)
        plt.axis('off')
        gambar_wc_pos = fig_to_base64(fig_pos)

    # Word Cloud Negatif
    gambar_wc_neg = ""
    if len(teks_negatif) > 0:
        fig_neg = plt.figure(figsize=(10, 5))
        wc_neg = WordCloud(width=800, height=400, background_color='black', colormap='hsv', max_words=100).generate(teks_negatif)
        plt.imshow(wc_neg, interpolation='bilinear')
        plt.title('Word Cloud Sentimen Negatif', fontsize=16)
        plt.axis('off')
        gambar_wc_neg = fig_to_base64(fig_neg)

    # Mengembalikan ketiga gambar dalam satu dictionary
    return {
        "plot_bar": gambar_bar,
        "wc_positif": gambar_wc_pos,
        "wc_negatif": gambar_wc_neg
    }