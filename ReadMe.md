# Dicoding Dashboard

## Setup Environment

Ikuti langkah-langkah berikut untuk mengatur lingkungan dan menjalankan Streamlit Dashboard:

### 1. Buat dan Aktifkan Virtual Environment (Conda)

Jika menggunakan **Conda**, jalankan perintah berikut di terminal:

```sh
conda create --name py39 python=3.9
conda activate py39
```

### 2. Instalasi Dependensi

Setelah environment aktif, install dependensi yang diperlukan dengan:

```sh
pip install pandas numpy matplotlib seaborn streamlit plotly scikit-learn statsmodels
```

### 3. Menjalankan Streamlit

Setelah semua dependensi terinstal, jalankan Streamlit dengan perintah berikut:

```sh
streamlit run Dashboard/AnalisisDataBike.py
```

## Troubleshooting

Jika mengalami error **ModuleNotFoundError**, pastikan semua dependensi telah terinstal dengan benar. Jika masih mengalami masalah, coba jalankan:

```sh
pip install --upgrade pip
pip install -r requirements.txt  # Jika tersedia
```

Jika ada kendala lain, periksa error log dan pastikan Anda berada di dalam environment yang benar dengan:

```sh
conda info --envs
```
