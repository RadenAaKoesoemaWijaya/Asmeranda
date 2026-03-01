# Asmeranda: Aplikasi Analisis Data & Machine Learning Komprehensif

## 📝 Deskripsi
Asmeranda adalah platform berbasis web (Streamlit) untuk analisis data mendalam dan pengembangan model Machine Learning. Dirancang khusus untuk membantu peneliti dalam mengolah data dari tahap eksplorasi (EDA), preprocessing, pelatihan model, hingga interpretasi hasil menggunakan XAI (SHAP & LIME).

> **⚠️ PERINGATAN:** DILARANG MENDISTRIBUSIKAN TANPA IZIN DARI PT. ASMER SAHABAT SUKSES.

---

## 🚀 Fitur Utama

1. **Sistem Keamanan & Autentikasi**: Login dengan verifikasi OTP Email, Captcha, dan Dashboard Super Admin untuk manajemen pengguna.
2. **Eksplorasi Data (EDA)**: Visualisasi otomatis distribusi, korelasi, dan deteksi pola data (termasuk Time Series).
3. **Preprocessing Canggih**: Penanganan nilai hilang (*Missing Values*), deteksi *outlier*, seleksi fitur, hingga normalisasi data otomatis.
4. **Pelatihan Model ML**: Mendukung 15+ algoritma Klasifikasi dan Regresi (Random Forest, XGBoost, LightGBM, dll) dengan optimasi hyperparameter (GridSearch, Optuna).
5. **Interpretasi Model (XAI)**: Penjelasan keputusan model secara global dan lokal menggunakan **SHAP** dan **LIME**.
6. **Time Series Analysis**: Forecasting (ARIMA, LSTM, Prophet) dan deteksi anomali pada data berbasis waktu.
7. **AI-Powered Analysis**: Rekomendasi metode penelitian otomatis berbasis karakteristik dataset.

---

## 🛠️ Cara Menjalankan Aplikasi

1. **Instalasi Dependensi**:
   ```bash
   pip install -r requirements.txt
   python -m pip install "tensorflow"
   ```

2. **Menjalankan Aplikasi**:
   ```bash
   streamlit run app.py
   ```
   Akses melalui browser di `http://localhost:8501`.

---

## 🔐 Panduan Singkat

### Login & OTP
- Daftar akun baru -> Login -> Cek email untuk kode OTP -> Masukkan kode di tab Verifikasi.
- **Mode Demo**: Jika SMTP belum disetel, kode OTP akan muncul sementara di antarmuka aplikasi.

### Dashboard Super Admin
- Hanya dapat diakses oleh akun dengan status Super Admin.
- Digunakan untuk konfigurasi SMTP, memantau aktivitas pengguna, dan manajemen akun.

### Unduh Model
- Setelah pelatihan model selesai, tombol **"Unduh Model (.pkl)"** akan muncul secara otomatis untuk menyimpan model hasil latih.

---

## 📂 Struktur Proyek Singkat
- `app.py`: File utama aplikasi.
- `utils.py`: Fungsi utilitas untuk pemrosesan data.
- `auth_db.py`: Manajemen database pengguna dan autentikasi.
- `requirements.txt`: Daftar pustaka Python yang diperlukan.
- `database/`: Penyimpanan database SQLite untuk pengguna.

---
**PT. ASMER SAHABAT SUKSES**
