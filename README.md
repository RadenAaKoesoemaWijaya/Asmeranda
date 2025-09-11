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
- Evaluasi performa model

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

## Cara Menjalankan Aplikasi
1. Pastikan semua persyaratan telah terpenuhi dengan menjalankan:

pip install -r requirements.txt
python -m pip install "tensorflow"

2. Jalankan aplikasi dengan perintah:

streamlit run app.py

3. Aplikasi akan terbuka di browser web Anda secara otomatis (biasanya di http://localhost:8501 )

## Alur Kerja Penggunaan
1. **Unggah dataset Anda di tab "Data Upload"**
   - Dukungan untuk data CSV, Excel, dan format lainnya
   - Identifikasi otomatis kolom tanggal/waktu untuk time series

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

8. **Prediksi data baru dan ekspor hasil**
   - Gunakan model terlatih untuk prediksi masa depan
   - Export hasil forecasting ke CSV dengan interval kepercayaan
   - Generate laporan PDF komprehensif untuk hasil analisis
