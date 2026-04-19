from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.templating import Jinja2Templates
import uvicorn
import pandas as pd
import io

# Import semua fungsi dari file mesin buatanmu
from klasifikasi import jalankan_klasifikasi
from kompirasi import jalankan_komparasi
from visualisasi import jalankan_visualisasi

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------------------------------------------------------
# 1. HALAMAN UTAMA (TAMPILAN FORM UPLOAD)
# ---------------------------------------------------------
@app.post("/upload")
async def proses_upload(request: Request, file: UploadFile = File(...)):
    # Membaca data yang diupload
    isi_file = await file.read()
    df_temp = pd.read_csv(io.BytesIO(isi_file))
    
    # Menjalankan mesin ML
    jalankan_klasifikasi()
    hasil_kmpr = jalankan_komparasi()
    hasil_viz = jalankan_visualisasi()
    
    # --- TAMBAHAN BARU: Menghitung Info untuk 4 Kartu ---
    total_data = len(df_temp)
    data_latih = int(total_data * 0.8) # Karena di komparasi.py test_size=0.2 (80:20)
    data_uji = total_data - data_latih

    info_kartu = {
        "total_data": total_data,
        "data_latih": data_latih,
        "data_uji": data_uji
    }
    # -----------------------------------------------------

    data_tabel = df_temp.head(10).to_dict(orient="records")
    kolom = df_temp.columns.tolist()

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "komparasi": hasil_kmpr,
            "visualisasi": hasil_viz,
            "data_tabel": data_tabel,
            "kolom": kolom,
            "nama_file": file.filename,
            "info_kartu": info_kartu # Jangan lupa kirimkan info_kartu ke HTML
        }
    )

# ---------------------------------------------------------
# 2. PROSES UPLOAD & ANALISIS (MENJALANKAN ML)
# ---------------------------------------------------------
@app.post("/upload")
async def proses_upload(request: Request, file: UploadFile = File(...)):
    # 1. Simpan file CSV di RAM untuk dibaca sementara ( opsional, 
    # di Python ML kamu sudah membaca file 'data_jadi.csv', 
    # jadi upload ini hanya formalitas untuk memicu proses )
    isi_file = await file.read()
    df_temp = pd.read_csv(io.BytesIO(isi_file))
    
    # 2. JALANKAN MESIN ML SECARA BERURUTAN
    # Proses klasifikasi (menghasilkan data_berlabel.csv)
    jalankan_klasifikasi()
    
    # Proses komparasi model
    hasil_kmpr = jalankan_komparasi()
    
    # Proses visualisasi data
    hasil_viz = jalankan_visualisasi()
    
    # Ambil sebagian data untuk tabel agar tidak berat
    data_tabel = df_temp.head(10).to_dict(orient="records")
    kolom = df_temp.columns.tolist()

    # 3. KIRIM SEMUA HASIL KE HTML
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "komparasi": hasil_kmpr,
            "visualisasi": hasil_viz,
            "data_tabel": data_tabel,
            "kolom": kolom,
            "nama_file": file.filename
        }
    )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)