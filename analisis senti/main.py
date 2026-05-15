from pathlib import Path
import glob
import io
import json
import os
import shutil
import subprocess

import pandas as pd
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.templating import Jinja2Templates

from klasifikasi import jalankan_klasifikasi, prediksi_teks_tunggal
from kompirasi import jalankan_komparasi
from labeling import jalankan_labeling_otomatis, normalisasi_label
from preprocessing import bersihkan_semua_data
from visualisasi import jalankan_visualisasi


BASE_DIR = Path(__file__).resolve().parent
DATA_TEMP = BASE_DIR / "data_sementara.csv"
DATA_PREPROCESSING = BASE_DIR / "data_preprocessing.csv"
DATA_BERLABEL = BASE_DIR / "data_berlabel.csv"
HASIL_KLASIFIKASI = BASE_DIR / "hasil_klasifikasi.csv"
SPLIT_CONFIG = BASE_DIR / "split_config.json"
TWEETS_DIR = BASE_DIR / "tweets-data"

app = FastAPI()
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


def baca_csv_fleksibel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    percobaan = [
        {"encoding": "utf-8"},
        {"encoding": "utf-8-sig"},
        {"sep": ";", "encoding": "utf-8"},
        {"sep": ";", "encoding": "utf-8-sig"},
        {"encoding": "latin1"},
        {"sep": ";", "encoding": "latin1"},
    ]
    error_terakhir = None
    for opsi in percobaan:
        try:
            df = pd.read_csv(path, **opsi)
            if len(df.columns) == 1 and opsi.get("sep") != ";":
                continue
            return df
        except Exception as exc:  # pragma: no cover
            error_terakhir = exc
    raise ValueError(f"Gagal membaca CSV {path.name}: {error_terakhir}")


def _cari_kolom(df: pd.DataFrame, kandidat: list[str]) -> str | None:
    kolom_map = {str(kolom).lower().strip(): kolom for kolom in df.columns}
    for nama in kandidat:
        if nama in kolom_map:
            return kolom_map[nama]
    return None


def siapkan_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(kolom).strip() for kolom in df.columns]
    kolom_teks = _cari_kolom(df, ["full_text_bersih", "full_text", "text", "tweet", "content"])

    if kolom_teks is None:
        raise ValueError("Dataset harus memiliki kolom teks seperti full_text, text, atau full_text_bersih.")

    if "full_text" not in df.columns:
        df["full_text"] = df[kolom_teks].astype(str)

    if "full_text_bersih" not in df.columns:
        df["full_text_bersih"] = df[kolom_teks].astype(str)

    if "username" not in df.columns:
        df["username"] = "Anonim"

    return df


def baca_konfigurasi_split() -> dict:
    default = {"rasio_latih": 80, "rasio_uji": 20}
    if not SPLIT_CONFIG.exists():
        return default

    try:
        config = json.loads(SPLIT_CONFIG.read_text(encoding="utf-8"))
        rasio_latih = int(config.get("rasio_latih", 80))
        return normalisasi_konfigurasi_split(rasio_latih)
    except Exception:
        return default


def normalisasi_konfigurasi_split(rasio_latih: int) -> dict:
    rasio_latih = max(50, min(int(rasio_latih), 90))
    return {"rasio_latih": rasio_latih, "rasio_uji": 100 - rasio_latih}


def simpan_konfigurasi_split(rasio_latih: int) -> dict:
    config = normalisasi_konfigurasi_split(rasio_latih)
    SPLIT_CONFIG.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def info_pembagian(total_data: int, rasio_latih: int | None = None) -> dict:
    if rasio_latih is None:
        rasio_latih = baca_konfigurasi_split()["rasio_latih"]
    rasio_uji = 100 - rasio_latih
    data_latih = int(total_data * (rasio_latih / 100))
    data_uji = max(total_data - data_latih, 0)
    return {
        "total_data": total_data,
        "data_latih": data_latih,
        "data_uji": data_uji,
        "rasio_latih": rasio_latih,
        "rasio_uji": rasio_uji,
    }


def _ambil_tabel(path: Path, limit: int = 20) -> tuple[list[dict], list[str]]:
    if not path.exists():
        return [], []
    df = baca_csv_fleksibel(path).fillna("")
    return df.head(limit).to_dict(orient="records"), df.columns.tolist()


def hitung_stats_label(df: pd.DataFrame) -> dict:
    df = df.copy()
    if "label_pakar" not in df.columns:
        df["label_pakar"] = "netral"
    if "label_otomatis" not in df.columns:
        df["label_otomatis"] = "-"

    df["label_pakar"] = df["label_pakar"].apply(normalisasi_label)
    df["label_otomatis"] = df["label_otomatis"].apply(lambda nilai: normalisasi_label(nilai, default="-"))

    return {
        "pakar_pos": int((df["label_pakar"] == "positif").sum()),
        "pakar_net": int((df["label_pakar"] == "netral").sum()),
        "pakar_neg": int((df["label_pakar"] == "negatif").sum()),
        "auto_pos": int((df["label_otomatis"] == "positif").sum()),
        "auto_net": int((df["label_otomatis"] == "netral").sum()),
        "auto_neg": int((df["label_otomatis"] == "negatif").sum()),
    }


def buat_kesimpulan(visualisasi: dict | None = None, komparasi: dict | None = None) -> dict:
    if visualisasi is None and DATA_BERLABEL.exists():
        visualisasi = jalankan_visualisasi(DATA_BERLABEL)
    visualisasi = visualisasi or {}
    komparasi = komparasi or {}

    counts = {
        "positif": int(visualisasi.get("jml_positif", 0) or 0),
        "netral": int(visualisasi.get("jml_netral", 0) or 0),
        "negatif": int(visualisasi.get("jml_negatif", 0) or 0),
    }
    total = sum(counts.values())
    dominan = max(counts, key=counts.get) if total else "-"
    persen = round((counts[dominan] / total) * 100, 2) if total and dominan != "-" else 0

    akurasi_nb = float(komparasi.get("akurasi_nb", 0) or 0)
    akurasi_svm = float(komparasi.get("akurasi_svm", 0) or 0)
    if akurasi_nb == akurasi_svm == 0:
        model_terbaik = "-"
    else:
        model_terbaik = "Naive Bayes" if akurasi_nb >= akurasi_svm else "SVM"

    narasi = "Belum ada data berlabel yang bisa dirangkum."
    if total:
        narasi = (
            f"Sentimen dominan adalah {dominan} sebanyak {counts[dominan]} dari {total} data "
            f"({persen}%)."
        )
        if model_terbaik != "-":
            narasi += f" Model dengan akurasi tertinggi pada data uji adalah {model_terbaik}."

    return {
        "counts": counts,
        "total": total,
        "dominan": dominan,
        "persen": persen,
        "model_terbaik": model_terbaik,
        "narasi": narasi,
    }


def konteks_dasar(status_tab: str = "dashboard") -> dict:
    split_config = baca_konfigurasi_split()
    konteks: dict = {"status_tab": status_tab, "split_config": split_config}

    data_tabel, kolom = _ambil_tabel(DATA_TEMP, limit=10)
    if data_tabel:
        total_data = len(baca_csv_fleksibel(DATA_TEMP))
        konteks.update(
            {
                "data_tabel": data_tabel,
                "kolom": kolom,
                "info_kartu": info_pembagian(total_data, split_config["rasio_latih"]),
                "nama_file": "Data aktif",
            }
        )

    data_preprocessing, _ = _ambil_tabel(DATA_PREPROCESSING, limit=100)
    if data_preprocessing:
        konteks["data_preprocessing"] = data_preprocessing
        konteks["jumlah_data_prep"] = len(baca_csv_fleksibel(DATA_PREPROCESSING))

    if DATA_BERLABEL.exists():
        df_label = baca_csv_fleksibel(DATA_BERLABEL).fillna("")
        if "label_pakar" not in df_label.columns:
            df_label["label_pakar"] = "netral"
        if "label_otomatis" not in df_label.columns:
            df_label["label_otomatis"] = "-"
        if "text_clean" not in df_label.columns:
            df_label["text_clean"] = df_label["full_text"] if "full_text" in df_label.columns else "-"
        konteks["data_label"] = df_label.head(200).to_dict(orient="records")
        konteks["stats_label"] = hitung_stats_label(df_label)

    if DATA_BERLABEL.exists():
        hasil_visualisasi = jalankan_visualisasi(DATA_BERLABEL)
        hasil_komparasi = jalankan_komparasi(
            DATA_BERLABEL,
            HASIL_KLASIFIKASI,
            rasio_uji=split_config["rasio_uji"],
        )
        konteks["visualisasi"] = hasil_visualisasi
        konteks["komparasi"] = hasil_komparasi
        konteks["kesimpulan"] = buat_kesimpulan(hasil_visualisasi, hasil_komparasi)

    if DATA_BERLABEL.exists() and not konteks.get("komparasi", {}).get("pesan"):
        klasifikasi_tabel, _ = _ambil_tabel(HASIL_KLASIFIKASI, limit=100)
        if klasifikasi_tabel:
            konteks["klasifikasi_tabel"] = klasifikasi_tabel

    return konteks


def jalankan_pipeline_lengkap(df: pd.DataFrame, nama_file: str, status_tab: str = "dataset") -> dict:
    df = siapkan_dataframe(df)
    df.to_csv(DATA_TEMP, index=False)

    try:
        df_preprocessing = bersihkan_semua_data(df)
        df_preprocessing.to_csv(DATA_PREPROCESSING, index=False)
        input_klasifikasi = DATA_PREPROCESSING
    except Exception:
        input_klasifikasi = DATA_TEMP

    df_berlabel = jalankan_klasifikasi(input_klasifikasi, DATA_BERLABEL)
    if "label_pakar" not in df_berlabel.columns:
        df_berlabel["label_pakar"] = df_berlabel["label_sentimen"]
    if "label_otomatis" not in df_berlabel.columns:
        df_berlabel["label_otomatis"] = df_berlabel["label_sentimen"]
    df_berlabel.to_csv(DATA_BERLABEL, index=False)

    split_config = baca_konfigurasi_split()
    hasil_komparasi = jalankan_komparasi(
        DATA_BERLABEL,
        HASIL_KLASIFIKASI,
        rasio_uji=split_config["rasio_uji"],
    )
    hasil_visualisasi = jalankan_visualisasi(DATA_BERLABEL)

    konteks = konteks_dasar(status_tab)
    konteks.update(
        {
            "komparasi": hasil_komparasi,
            "visualisasi": hasil_visualisasi,
            "nama_file": nama_file,
            "status_analisis": "selesai",
            "kesimpulan": buat_kesimpulan(hasil_visualisasi, hasil_komparasi),
        }
    )
    return konteks


def pastikan_data_berlabel() -> None:
    if DATA_BERLABEL.exists():
        return
    if DATA_PREPROCESSING.exists():
        shutil.copy(DATA_PREPROCESSING, DATA_BERLABEL)
        return
    if DATA_TEMP.exists():
        df = baca_csv_fleksibel(DATA_TEMP)
        df = siapkan_dataframe(df)
        df_preprocessing = bersihkan_semua_data(df)
        df_preprocessing.to_csv(DATA_PREPROCESSING, index=False)
        shutil.copy(DATA_PREPROCESSING, DATA_BERLABEL)
        return
    raise FileNotFoundError("Belum ada data. Unggah CSV atau lakukan scraping terlebih dahulu.")


def siapkan_data_klasifikasi() -> None:
    if DATA_BERLABEL.exists():
        df = baca_csv_fleksibel(DATA_BERLABEL)
    else:
        input_path = DATA_PREPROCESSING if DATA_PREPROCESSING.exists() else DATA_TEMP
        if not input_path.exists():
            raise FileNotFoundError("Belum ada data untuk diklasifikasi.")
        df = jalankan_klasifikasi(input_path, DATA_BERLABEL)

    df = siapkan_dataframe(df).fillna("")

    label_otomatis_kosong = "label_otomatis" not in df.columns or df["label_otomatis"].apply(
        lambda nilai: normalisasi_label(nilai, default="")
    ).eq("").all()
    if label_otomatis_kosong:
        df = jalankan_labeling_otomatis(df)

    if "label_sentimen" not in df.columns:
        df["label_sentimen"] = df["label_otomatis"]

    if "label_pakar" not in df.columns:
        df["label_pakar"] = df["label_sentimen"]
    else:
        df["label_pakar"] = df["label_pakar"].apply(normalisasi_label)

    df["label_otomatis"] = df["label_otomatis"].apply(normalisasi_label)
    df.to_csv(DATA_BERLABEL, index=False)


def konteks_model_klasifikasi(model: str, komparasi: dict | None = None) -> dict:
    komparasi = komparasi or {}
    model = str(model).lower().strip()

    if model == "svm":
        return {
            "active_model_key": "svm",
            "active_model": "SVM",
            "active_prediction_column": "label_prediksi_svm",
            "active_accuracy": komparasi.get("akurasi_svm", 0),
            "active_model_color": "rose",
        }

    return {
        "active_model_key": "nb",
        "active_model": "Naive Bayes",
        "active_prediction_column": "label_prediksi_nb",
        "active_accuracy": komparasi.get("akurasi_nb", 0),
        "active_model_color": "blue",
    }


@app.get("/")
async def halaman_awal(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context=konteks_dasar())


@app.post("/scraping")
async def proses_scraping(
    request: Request,
    kata_kunci: str = Form(...),
    jumlah_data: int = Form(...),
    auth_token: str = Form(...),
):
    try:
        env_sistem = os.environ.copy()
        env_sistem["TWITTER_AUTH_TOKEN"] = auth_token
        TWEETS_DIR.mkdir(exist_ok=True)

        perintah_terminal = f'npx --yes tweet-harvest@latest -s "{kata_kunci}" -l {jumlah_data}'
        proses = subprocess.run(
            perintah_terminal,
            shell=True,
            cwd=BASE_DIR,
            env=env_sistem,
        )

        if proses.returncode != 0:
            raise RuntimeError("Scraping gagal. Cek terminal untuk detail error dari tweet-harvest.")

        semua_file_csv = glob.glob(str(TWEETS_DIR / "*.csv"))
        if not semua_file_csv:
            raise FileNotFoundError("Scraping selesai, tetapi file CSV tidak ditemukan di folder tweets-data.")

        path_hasil = max(semua_file_csv, key=os.path.getmtime)
        df_baru = siapkan_dataframe(baca_csv_fleksibel(path_hasil))
        df_baru.to_csv(DATA_TEMP, index=False)
        pesan = f"Sukses menarik data '{kata_kunci}'. Data sudah siap dianalisis."

        konteks = konteks_dasar("dashboard")
        konteks["pesan_scraping"] = pesan
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("dashboard")
        konteks["pesan_scraping"] = f"Gagal scraping: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/analisis-otomatis")
async def proses_otomatis(request: Request):
    try:
        if DATA_TEMP.exists():
            df_temp = baca_csv_fleksibel(DATA_TEMP)
            nama_file = "Data hasil scraping terakhir"
        else:
            list_file_csv = glob.glob(str(TWEETS_DIR / "*.csv"))
            if not list_file_csv:
                raise FileNotFoundError("Belum ada file CSV di folder tweets-data.")
            path_file_scraping = max(list_file_csv, key=os.path.getctime)
            df_temp = baca_csv_fleksibel(path_file_scraping)
            nama_file = f"Data scraping: {Path(path_file_scraping).name}"

        konteks = jalankan_pipeline_lengkap(df_temp, nama_file, status_tab="dataset")
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("dashboard")
        konteks["pesan_scraping"] = f"Terjadi kesalahan saat analisis: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        with DATA_TEMP.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df_temp = baca_csv_fleksibel(DATA_TEMP)
        konteks = jalankan_pipeline_lengkap(df_temp, f"File upload: {file.filename}", status_tab="dataset")
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("dataset")
        konteks["pesan_error"] = str(exc)
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/jalankan-preprocessing")
async def proses_preprocessing(request: Request):
    try:
        if not DATA_TEMP.exists():
            raise FileNotFoundError("Belum ada data. Unggah CSV atau lakukan scraping terlebih dahulu.")

        df = siapkan_dataframe(baca_csv_fleksibel(DATA_TEMP))
        df_preprocessing = bersihkan_semua_data(df)
        df_preprocessing.to_csv(DATA_PREPROCESSING, index=False)

        konteks = konteks_dasar("preprocessing")
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("preprocessing")
        konteks["pesan_error"] = f"Gagal preprocessing: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/atur-rasio")
async def atur_rasio(request: Request, rasio_latih: int = Form(...)):
    try:
        split_config = simpan_konfigurasi_split(rasio_latih)

        if DATA_BERLABEL.exists():
            jalankan_komparasi(
                DATA_BERLABEL,
                HASIL_KLASIFIKASI,
                rasio_uji=split_config["rasio_uji"],
            )

        konteks = konteks_dasar("split")
        konteks["pesan_rasio"] = (
            f"Rasio berhasil diubah menjadi {split_config['rasio_latih']}:"
            f"{split_config['rasio_uji']}."
        )
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("split")
        konteks["pesan_error"] = f"Gagal mengubah rasio: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/upload-label-pakar")
async def upload_label_pakar(request: Request, file: UploadFile = File(...)):
    try:
        isi = await file.read()
        df = pd.read_csv(io.BytesIO(isi))
        df = siapkan_dataframe(df)
        if "label_pakar" not in df.columns:
            kolom_label = _cari_kolom(df, ["label_sentimen", "sentimen", "label"])
            df["label_pakar"] = df[kolom_label].apply(normalisasi_label) if kolom_label else "netral"
        df.to_csv(DATA_BERLABEL, index=False)
        return await render_label_page(request)
    except Exception as exc:
        konteks = konteks_dasar("label")
        konteks["pesan_error"] = f"Gagal upload label pakar: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/label-otomatis")
async def label_otomatis(request: Request):
    try:
        pastikan_data_berlabel()
        df = baca_csv_fleksibel(DATA_BERLABEL)
        df = jalankan_labeling_otomatis(df)
        if "label_pakar" not in df.columns:
            df["label_pakar"] = "netral"
        df.to_csv(DATA_BERLABEL, index=False)
        return await render_label_page(request)
    except Exception as exc:
        konteks = konteks_dasar("label")
        konteks["pesan_error"] = f"Gagal labeling otomatis: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/update-label")
async def update_label(request: Request, index: int = Form(...), label_baru: str = Form(...)):
    try:
        pastikan_data_berlabel()
        df = baca_csv_fleksibel(DATA_BERLABEL)
        if "label_pakar" not in df.columns:
            df["label_pakar"] = "netral"
        if 0 <= index < len(df):
            df.at[index, "label_pakar"] = normalisasi_label(label_baru)
        df.to_csv(DATA_BERLABEL, index=False)
        return await render_label_page(request)
    except Exception as exc:
        konteks = konteks_dasar("label")
        konteks["pesan_error"] = f"Gagal update label: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


async def render_label_page(request: Request):
    try:
        pastikan_data_berlabel()
        df = baca_csv_fleksibel(DATA_BERLABEL).fillna("")
        if "label_pakar" not in df.columns:
            df["label_pakar"] = "netral"
        if "label_otomatis" not in df.columns:
            df["label_otomatis"] = "-"
        if "text_clean" not in df.columns:
            df["text_clean"] = df["full_text"] if "full_text" in df.columns else "-"
        df["label_pakar"] = df["label_pakar"].apply(normalisasi_label)
        df.to_csv(DATA_BERLABEL, index=False)

        konteks = konteks_dasar("label")
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("label")
        konteks["pesan_error"] = str(exc)
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


async def render_klasifikasi_model(request: Request, model: str):
    try:
        siapkan_data_klasifikasi()

        konteks = konteks_dasar("klasifikasi")
        konteks.update(konteks_model_klasifikasi(model, konteks.get("komparasi")))
        konteks["pesan_klasifikasi"] = f"Klasifikasi {konteks['active_model']} berhasil dijalankan."
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("klasifikasi")
        konteks.update(konteks_model_klasifikasi(model, konteks.get("komparasi")))
        konteks["pesan_error"] = f"Gagal klasifikasi: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/jalankan-klasifikasi")
async def proses_klasifikasi(request: Request):
    return await render_klasifikasi_model(request, "nb")


@app.post("/jalankan-klasifikasi-nb")
async def proses_klasifikasi_nb(request: Request):
    return await render_klasifikasi_model(request, "nb")


@app.post("/jalankan-klasifikasi-svm")
async def proses_klasifikasi_svm(request: Request):
    return await render_klasifikasi_model(request, "svm")


@app.post("/prediksi-teks")
async def proses_prediksi_teks(request: Request, teks_input: str = Form(...)):
    konteks = konteks_dasar("klasifikasi")
    konteks["prediksi"] = {
        "teks": teks_input,
        "label": prediksi_teks_tunggal(teks_input),
    }
    return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/jalankan-komparasi")
async def proses_komparasi(request: Request):
    try:
        pastikan_data_berlabel()
        split_config = baca_konfigurasi_split()
        hasil_komparasi = jalankan_komparasi(
            DATA_BERLABEL,
            HASIL_KLASIFIKASI,
            rasio_uji=split_config["rasio_uji"],
        )
        konteks = konteks_dasar("komparasi")
        konteks["komparasi"] = hasil_komparasi
        konteks["kesimpulan"] = buat_kesimpulan(komparasi=hasil_komparasi)
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("komparasi")
        konteks["pesan_error"] = f"Gagal komparasi: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/jalankan-visualisasi")
async def proses_visualisasi(request: Request):
    try:
        pastikan_data_berlabel()
        hasil_visualisasi = jalankan_visualisasi(DATA_BERLABEL)
        konteks = konteks_dasar("visualisasi")
        konteks["visualisasi"] = hasil_visualisasi
        konteks["kesimpulan"] = buat_kesimpulan(hasil_visualisasi)
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("visualisasi")
        konteks["pesan_error"] = f"Gagal visualisasi: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.post("/kesimpulan")
async def proses_kesimpulan(request: Request):
    try:
        pastikan_data_berlabel()
        split_config = baca_konfigurasi_split()
        hasil_komparasi = jalankan_komparasi(
            DATA_BERLABEL,
            HASIL_KLASIFIKASI,
            rasio_uji=split_config["rasio_uji"],
        )
        hasil_visualisasi = jalankan_visualisasi(DATA_BERLABEL)
        konteks = konteks_dasar("kesimpulan")
        konteks["komparasi"] = hasil_komparasi
        konteks["visualisasi"] = hasil_visualisasi
        konteks["kesimpulan"] = buat_kesimpulan(hasil_visualisasi, hasil_komparasi)
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)
    except Exception as exc:
        konteks = konteks_dasar("kesimpulan")
        konteks["pesan_error"] = f"Gagal membuat kesimpulan: {exc}"
        return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.get("/klasifikasi-nb")
async def halaman_nb(request: Request):
    konteks = konteks_dasar("klasifikasi")
    konteks.update(konteks_model_klasifikasi("nb", konteks.get("komparasi")))
    return templates.TemplateResponse(request=request, name="index.html", context=konteks)


@app.get("/klasifikasi-svm")
async def halaman_svm(request: Request):
    konteks = konteks_dasar("klasifikasi")
    konteks.update(konteks_model_klasifikasi("svm", konteks.get("komparasi")))
    return templates.TemplateResponse(request=request, name="index.html", context=konteks)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
