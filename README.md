# Aplikasi Analisis Data dan Machine Learning Komprehensif

## Deskripsi
Aplikasi ini adalah alat analisis data dan machine learning berbasis web yang dibangun dengan Streamlit. Aplikasi ini memungkinkan pengguna untuk menganalisis data, melakukan preprocessing, melatih model machine learning, dan menginterpretasikan hasil menggunakan nilai SHAP.

## Fitur Utama
Aplikasi ini menyediakan beberapa fitur utama yang dibagi dalam beberapa tab:

### 1. 📤 Unggah Data
- Unggah file CSV
- Melihat pratinjau data
- Melihat informasi dan statistik data
- Identifikasi otomatis kolom numerik dan kategorikal

### 2. 📊 Analisis Data Eksploratori
- Analisis nilai yang hilang (missing values)
- Analisis korelasi dengan heatmap
- Visualisasi distribusi fitur numerik (histogram dan boxplot)
- Visualisasi distribusi fitur kategorikal
- Analisis bivariat untuk melihat hubungan antar variabel

### 3. 🔄 Preprocessing Data
- Pemilihan variabel target
- Penentuan tipe masalah (klasifikasi atau regresi)
- Penanganan nilai yang hilang
- Encoding fitur kategorikal
- Penskalaan fitur numerik (pilihan Standard Scaler atau MinMax Scaler)
- **Seleksi fitur dengan berbagai metode:**  
  - Manual  
  - Mutual Information  
  - Pearson Correlation  
  - Recursive Feature Elimination (RFE)  
  - LASSO  
  - Gradient Boosting Importance  
  - Random Forest Importance  
  - Ensemble Feature Selection (gabungan dua metode dengan union/intersection)

### 4. 🛠️ Feature Engineering & Pelatihan Model
- Pemilihan fitur dengan berbagai metode
- Pemilihan algoritma machine learning
- Pembagian data training dan testing
- Pelatihan model dengan parameter yang dapat disesuaikan
- **Optimasi Parameter Hyperparameter Otomatis**:
  - **GridSearchCV**: Pencarian eksaustif parameter optimal
  - **RandomizedSearchCV**: Pencarian parameter acak efisien
  - **Bayesian Optimization**: Optimasi berbasis probabilistik (Optuna)
  - **Preset Parameter**: Konfigurasi parameter siap pakai untuk berbagai skenario
  - **Validasi Parameter Cerdas**: Validasi otomatis parameter berbasis karakteristik data
  - **Rentang Parameter Kustom**: Fleksibilitas mendefinisikan rentang parameter spesifik
  - **Impor/Ekspor Konfigurasi**: Simpan dan bagikan konfigurasi parameter favorit

### 5. 🔍 Interpretasi Model dengan SHAP
- Visualisasi nilai SHAP untuk memahami kontribusi fitur
- Analisis pengaruh fitur terhadap prediksi model

### 6. 📈 Analisis Data Baru & Prediksi
- Input data baru secara manual atau upload file CSV untuk prediksi batch
- Preprocessing otomatis pada data baru (encoding & scaling sesuai model)
- Hasil prediksi dapat diunduh dalam format CSV atau PDF laporan prediksi
- Fitur untuk memuat model yang sudah disimpan dan melakukan prediksi ulang

### 7. 📈 Time Series Forecasting & Analisis Prediktif
- **Algoritma Forecasting Lengkap**:
  - **ARIMA/SARIMA**: Model statistik klasik untuk time series
  - **Exponential Smoothing**: Holt-Winters untuk data dengan tren dan musiman
  - **LSTM (Long Short-Term Memory)**: Deep learning untuk prediksi jangka panjang
  - **Random Forest Regressor**: Ensemble learning untuk data time series
  - **Gradient Boosting**: XGBoost-style untuk forecasting kompleks
  - **Linear Regression**: Baseline untuk perbandingan performa

- **Fitur Analisis Time Series**:
  - Deteksi otomatis kolom tanggal/waktu dan frekuensi data
  - Analisis stasioneritas data dengan Augmented Dickey-Fuller test
  - Identifikasi komponen tren, musiman, dan siklus
  - Visualisasi dekomposisi time series
  - Plot ACF dan PACF untuk analisis autokorelasi

- **Parameter Konfigurasi Interaktif**:
  - Order ARIMA (p,d,q) dan seasonal order (P,D,Q,s) yang dapat disesuaikan
  - Hyperparameter LSTM (units, dropout, epochs, batch size)
  - Window size dan lag features untuk model ML
  - Proporsi train-test split yang fleksibel

- **Evaluasi Model Forecasting**:
  - Metrik evaluasi komprehensif: MAE, MSE, RMSE, MAPE, R²
  - Visualisasi prediksi vs aktual dengan confidence intervals
  - Forecast horizon yang dapat dikonfigurasi
  - Perbandingan performa antar model dalam tabel interaktif

- **Visualisasi dan Interpretasi**:
  - Plot time series dengan area prediksi
  - Residual analysis dan diagnostic plots
  - Feature importance untuk model tree-based
  - Export hasil forecasting ke CSV dengan interval kepercayaan

### 8. ⚠️ Deteksi Anomali Time Series
- **Algoritma State-of-the-Art**: Isolation Forest, One-Class SVM, Statistical (Z-Score), dan Ensemble Method
- Deteksi otomatis kolom tanggal/waktu dalam dataset
- Konfigurasi parameter interaktif (tingkat kontaminasi, threshold Z-score)
- Visualisasi komprehensif hasil deteksi anomali
- Analisis perbandingan antar metode deteksi
- Ekspor hasil dalam format CSV dengan flag anomali untuk setiap metode
- Dukungan bilingual (Bahasa Indonesia & English)
- Penanganan missing values otomatis

### 9. 🎯 Optimasi Parameter & Preset Konfigurasi
Aplikasi ini menyediakan sistem optimasi parameter yang canggih untuk meningkatkan performa model:

#### Fitur Optimasi Parameter:
- **GridSearchCV**: Pencarian parameter optimal secara menyeluruh dalam rentang yang ditentukan
- **RandomizedSearchCV**: Pencarian parameter acak yang efisien untuk ruang parameter besar
- **Bayesian Optimization**: Optimasi berbasis probabilistik menggunakan Optuna untuk hasil optimal dengan iterasi minimal
- **Validasi Parameter Cerdas**: Validasi otomatis parameter berbasis karakteristik data
  - `max_depth` divalidasi berdasarkan jumlah fitur
  - `n_neighbors` divalidasi berdasarkan jumlah sampel training
  - `max_features` divalidasi berdasarkan dimensi data

#### Sistem Preset Parameter:
- **Preset Siap Pakai**: Konfigurasi parameter optimal untuk berbagai skenario:
  - **Default**: Konfigurasi standar untuk pemula
  - **Fast Training**: Parameter untuk pelatihan cepat pada dataset kecil
  - **High Accuracy**: Parameter untuk akurasi maksimal (komputasi lebih intensif)
  - **Small Dataset**: Parameter yang dioptimalkan untuk dataset dengan sampel terbatas
- **Preset Kustom**: Buat dan simpan konfigurasi parameter favorit
- **Impor/Ekspor Preset**: Bagikan konfigurasi parameter dengan tim atau simpan untuk digunakan kembali

#### Rentang Parameter Kustom:
- Definisikan rentang parameter spesifik untuk setiap model
- Fleksibilitas penuh dalam menentukan nilai minimum dan maksimum
- Integrasi dengan semua metode optimasi (GridSearchCV, RandomizedSearchCV, Bayesian Optimization)
- Antarmuka yang intuitif untuk mengatur parameter berdasarkan kebutuhan spesifik

## Model yang Didukung
### Klasifikasi
- Random Forest
- Gradient Boosting
- Logistic Regression
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Neural Network (MLP)
- Decision Tree
- Naive Bayes

### Regresi
- Random Forest
- Gradient Boosting
- Linear Regression
- Support Vector Regression (SVR)

### Time Series Forecasting
- **ARIMA/SARIMA**: Untuk data dengan pola musiman
- **Exponential Smoothing (Holt-Winters)**: Untuk data dengan tren dan musiman
- **LSTM**: Deep learning untuk prediksi kompleks
- **Random Forest Regressor**: Untuk data time series dengan fitur eksternal
- **Gradient Boosting**: Untuk forecasting non-linear kompleks
- **Linear Regression**: Baseline untuk perbandingan performa

## Keunggulan Aplikasi
- **Antarmuka yang Ramah Pengguna**: Tampilan yang intuitif dan mudah digunakan, cocok untuk pemula maupun pengguna berpengalaman
- **Analisis Komprehensif**: Menyediakan berbagai metode analisis dari eksplorasi data hingga interpretasi model
- **Machine Learning Lengkap**: Mendukung klasifikasi, regresi, dan time series forecasting
- **Interpretasi Model yang Canggih**: Menggunakan SHAP untuk interpretasi model yang dapat dipercaya
- **Dukungan Multi-bahasa**: Mendukung Bahasa Indonesia dan Bahasa Inggris
- **Export Hasil yang Fleksibel**: Dapat mengekspor hasil dalam berbagai format (CSV, PDF, gambar)
- **Sistem Optimasi Parameter Canggih**: 
  - Validasi parameter otomatis berbasis karakteristik data
  - Preset parameter siap pakai untuk berbagai skenario
  - Fitur impor/ekspor konfigurasi untuk kolaborasi tim
  - Integrasi dengan berbagai metode optimasi (GridSearchCV, RandomizedSearchCV, Bayesian Optimization)

## Cara Menjalankan Aplikasi
1. Pastikan semua persyaratan telah terpenuhi dengan menjalankan:

pip install -r requirements.txt
python -m pip install "tensorflow"

2. Jalankan aplikasi dengan perintah:

streamlit run app.py

3. Aplikasi akan terbuka di browser web Anda secara otomatis (biasanya di http://localhost:8501 )

## Struktur File
- `app.py` - File utama aplikasi Streamlit
- `param_presets.py` - Modul manajemen preset parameter untuk optimasi model
- `auth_db.py` - Modul autentikasi pengguna
- `utils.py` - Fungsi utilitas untuk time series dan preprocessing
- `anomaly_detection_utils.py` - Fungsi untuk deteksi anomali
- `forecasting_utils.py` - Fungsi untuk forecasting time series
- `requirements.txt` - Daftar dependensi Python
- `users.db` - Database SQLite untuk autentikasi
- `models/` - Folder untuk menyimpan model yang dilatih
- `DFI/`, `KIMIA DARAH/`, `PROLANIS/` - Folder dataset contoh

## Persyaratan Sistem
- Python 3.7 atau lebih baru
- Streamlit untuk antarmuka web
- Scikit-learn untuk machine learning
- Pandas dan NumPy untuk manipulasi data
- Matplotlib dan Seaborn untuk visualisasi
- Plotly untuk visualisasi interaktif
- SHAP untuk interpretasi model
- Optuna untuk optimasi Bayesian (opsional)
- TensorFlow untuk LSTM forecasting (opsional)
- LIME untuk interpretasi model alternatif (opsional)

## Alur Kerja Penggunaan
1. **Unggah dataset Anda di tab "Data Upload"**
   - Dukungan untuk data CSV, Excel, dan format lainnya
   - Identifikasi otomatis kolom tanggal/waktu untuk time series

### Contoh Penggunaan Fitur Optimasi Parameter:

#### Menggunakan Preset Parameter:
1. Setelah memilih model di tab "Feature Engineering & Model Training"
2. Aktifkan "Gunakan Rentang Parameter Kustom"
3. Pilih preset dari dropdown (Default, Fast Training, High Accuracy, atau Small Dataset)
4. Klik "Terapkan Preset" untuk mengisi parameter otomatis
5. Lihat detail preset dengan klik "Lihat Detail"

#### Membuat Konfigurasi Kustom:
1. Aktifkan "Gunakan Rentang Parameter Kustom"
2. Atur rentang parameter sesuai kebutuhan di form yang tersedia
3. Simpan konfigurasi dengan fitur impor/ekspor
4. Bagikan konfigurasi dengan tim menggunakan file JSON

#### Optimasi Parameter Otomatis:
1. Pilih metode optimasi (GridSearchCV untuk hasil optimal, RandomizedSearchCV untuk efisiensi, atau Bayesian Optimization untuk keseimbangan)
2. Sistem akan secara otomatis memvalidasi parameter berdasarkan data Anda
3. Parameter yang tidak valid akan disesuaikan secara otomatis
4. Hasil optimasi ditampilkan dengan metrik performa terbaik

2. **Lakukan eksplorasi data di tab "Exploratory Data Analytic"**
   - Analisis tren, musiman, dan pola time series
   - Visualisasi dekomposisi data untuk forecasting

3. **Lakukan preprocessing data di tab "Preprocessing"**
   - Penanganan missing values otomatis
   - Seleksi fitur untuk model forecasting
   - Scaling dan transformasi data sesuai kebutuhan

4. **Time Series Forecasting di tab "Time Series Forecasting"**
   - Pilih algoritma forecasting (ARIMA, LSTM, Exponential Smoothing, dll)
   - Konfigurasi parameter interaktif
   - Visualisasi prediksi dengan confidence intervals
   - Evaluasi performa model dengan berbagai metrik

5. **Latih dan evaluasi model klasifikasi/regresi di tab "Feature Engineering & Model Training"**
   - Untuk data non-time series atau sebagai pendamping forecasting

6. **Interpretasikan hasil model dengan SHAP di tab "SHAP Model Interpretation"**
   - Analisis feature importance untuk model forecasting
   - Pemahaman kontribusi fitur terhadap prediksi

7. **Analisis anomali pada data time series di tab "Time Series Anomaly Detection"**
   - Deteksi outlier dalam data time series
   - Identifikasi event abnormal yang dapat mempengaruhi forecasting

8. **Optimasi Parameter Hyperparameter di tab "Feature Engineering & Model Training"**
   - Pilih metode optimasi (GridSearchCV, RandomizedSearchCV, atau Bayesian Optimization)
   - Gunakan preset parameter siap pakai atau buat konfigurasi kustom
   - Validasi parameter otomatis akan memastikan parameter yang dipilih sesuai dengan data Anda
   - Simpan konfigurasi parameter favorit untuk digunakan kembali di masa depan
   - Bandingkan performa model dengan berbagai konfigurasi parameter

9. **Prediksi data baru dan ekspor hasil**
   - Gunakan model terlatih untuk prediksi masa depan
   - Export hasil forecasting ke CSV dengan interval kepercayaan
   - Generate laporan PDF komprehensif untuk hasil analisis

## Tips dan Trik

### Untuk Optimasi Parameter yang Efektif:
- **Dataset Kecil (< 1000 sampel)**: Gunakan preset "Small Dataset" atau RandomizedSearchCV untuk menghindari overfitting
- **Dataset Besar (> 10000 sampel)**: Gunakan GridSearchCV atau Bayesian Optimization untuk hasil optimal
- **Waktu Terbatas**: Gunakan preset "Fast Training" atau RandomizedSearchCV dengan iterasi minimal
- **Akurasi Maksimal**: Gunakan preset "High Accuracy" atau Bayesian Optimization dengan iterasi banyak
- **Kolaborasi Tim**: Export konfigurasi parameter yang berhasil ke JSON dan bagikan dengan tim

### Troubleshooting:
- Jika optimasi parameter memakan waktu lama, coba gunakan RandomizedSearchCV dengan jumlah iterasi yang lebih sedikit
- Untuk parameter yang tidak valid, sistem akan secara otomatis menyesuaikan dengan batasan data Anda
- Gunakan fitur "Lihat Detail" pada preset untuk memahami konfigurasi parameter sebelum menerapkannya

## Pembaruan Terbaru (v2.0)

### 🎯 Fitur Optimasi Parameter Baru:
- **Sistem Preset Parameter**: Konfigurasi parameter siap pakai untuk berbagai skenario (Default, Fast Training, High Accuracy, Small Dataset)
- **Validasi Parameter Cerdas**: Validasi otomatis parameter berbasis karakteristik data (jumlah fitur, sampel, dll)
- **Impor/Ekspor Konfigurasi**: Bagikan konfigurasi parameter dengan tim menggunakan file JSON
- **Rentang Parameter Kustom**: Fleksibilitas penuh dalam menentukan rentang parameter untuk optimasi
- **Integrasi dengan Semua Model**: Mendukung semua model klasifikasi dan regresi yang tersedia

### 📊 Peningkatan Performa:
- Validasi parameter otomatis mencegah error saat optimasi
- Preset parameter menghemat waktu konfigurasi
- Fitur impor/ekspor memungkinkan kolaborasi tim yang lebih baik
- Antarmuka yang lebih intuitif untuk pengaturan parameter

## Kontribusi
Kontribusi sangat dipersilakan! Silakan buat pull request atau laporkan issue di repository ini.

## Lisensi
Proyek ini dilisensikan di bawah lisensi MIT. Lihat file [LICENSE](LICENSE) untuk detail lebih lanjut.

## Dukungan
Jika Anda menemui masalah atau memiliki pertanyaan, silakan buat issue di repository ini atau hubungi melalui email yang tersedia di profil.

---
**Catatan**: Aplikasi ini terus dikembangkan. Pastikan untuk selalu memperbarui ke versi terbaru untuk mendapatkan fitur dan perbaikan terbaru.
