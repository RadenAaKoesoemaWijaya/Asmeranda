# Asmeranda: Aplikasi Analisis Data & Machine Learning Komprehensif

## 📝 Deskripsi
Asmeranda adalah platform berbasis web (Streamlit) untuk analisis data mendalam dan pengembangan model Machine Learning. Dirancang khusus untuk membantu peneliti dalam mengolah data dari tahap eksplorasi (EDA), preprocessing, pelatihan model, hingga interpretasi hasil menggunakan XAI (SHAP & LIME).

> **⚠️ PERINGATAN:** DILARANG MENDISTRIBUSIKAN TANPA IZIN DARI PT. ASMER SAHABAT SUKSES.

---

## 🚀 Fitur Utama & Metodologi

1. **Sistem Keamanan & Autentikasi**: Login dengan verifikasi OTP Email, Captcha, dan Dashboard Super Admin untuk manajemen pengguna.
2. **Eksplorasi Data (EDA) & Validasi**: Visualisasi otomatis dan sistem **Workflow Validator** yang memastikan data siap sebelum lanjut ke tahap ML.
3. **Preprocessing & Deteksi Tipe Data**: Penanganan nilai hilang, outlier, dan **Deteksi Tipe Data Otomatis** untuk rekomendasi model yang lebih akurat.
4. **Pelatihan Model ML**: Mendukung 15+ algoritma Klasifikasi dan Regresi dengan optimasi hyperparameter (GridSearch, Optuna).
5. **Best Practice Imbalanced Handling**: Penanganan data tidak seimbang (SMOTE, dll) dilakukan **setelah Train-Test Split** untuk mencegah *Data Leakage*.
6. **Interpretasi Model (XAI)**: Penjelasan keputusan model secara global dan lokal menggunakan **SHAP** dan **LIME**.
7. **Time Series & AI Analysis**: Forecasting (ARIMA, LSTM, Prophet) dan rekomendasi metode penelitian berbasis AI.

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

## 🔐 Panduan Singkat Penggunaan

### 1. Login & Verifikasi OTP
- Daftar akun baru -> Login -> Cek email untuk kode OTP -> Masukkan kode di tab Verifikasi.
- **Mode Demo**: Jika SMTP belum disetel, kode OTP akan muncul di terminal/antarmuka aplikasi.

### 2. Alur Kerja (Workflow)
- Pastikan data telah divalidasi di tab **Data Upload** sebelum berpindah ke **EDA**.
- Penanganan dataset *imbalanced* kini tersedia di tab **Supervised ML** setelah pembagian data (split) untuk menjamin validitas pengujian.

### 3. Dashboard Admin & Ekspor
- **Super Admin**: Akses penuh untuk manajemen pengguna dan konfigurasi sistem (SMTP, Logs).
- **Unduh Model**: Model hasil latih dapat diunduh dalam format `.pkl` untuk digunakan di lingkungan produksi.

---

## 📂 Struktur Proyek Inti
- `app.py`: Logika antarmuka dan alur kerja utama.
- `utils.py`: Mesin pemrosesan data, clustering, dan forecasting.
- `workflow_validator.py`: Sistem validasi kesiapan data antar tahap.
- `data_type_detector.py`: Modul deteksi otomatis karakteristik kolom data.
- `auth_db.py` & `database/`: Sistem manajemen keamanan dan basis data pengguna.

---
**PT. ASMER SAHABAT SUKSES**
