# Aplikasi Analisis Data dan Machine Learning Komprehensif

## Deskripsi
Aplikasi ini adalah alat analisis data dan machine learning berbasis web yang dibangun dengan Streamlit. Aplikasi ini memungkinkan pengguna untuk menganalisis data, melakukan preprocessing, melatih model machine learning, dan menginterpretasikan hasil menggunakan nilai SHAP dan LIME (Local Interpretable Model-agnostic Explanations).

## Fitur Utama
Aplikasi ini menyediakan beberapa fitur utama yang dibagi dalam beberapa tab:

### 1. 🔐 Sistem Autentikasi & Keamanan
- **Registrasi & Login Pengguna**: Sistem autentikasi lengkap dengan database SQLite
- **Captcha Security**: Perlindungan dari bot dengan captcha visual
- **Session Management**: Pengelolaan sesi pengguna yang aman
- **Multi-bahasa**: Dukungan Bahasa Indonesia dan English

### 2. 📤 Unggah Data
- **Multi-format Support**: Unggah file CSV, Excel (XLSX), dan ZIP (dataset gabungan)
- **Dataset Integration**: Support untuk dataset train/test split dalam format ZIP
- **Automatic Data Detection**: Identifikasi otomatis kolom numerik, kategorikal, dan datetime
- **Data Preview & Statistics**: Pratinjau data dan statistik deskriptif
- **Missing Values Analysis**: Deteksi dan analisis nilai hilang otomatis

### 3. 📊 Analisis Data Eksploratori
- **Comprehensive EDA**: Analisis eksploratori data lengkap
- **Correlation Analysis**: Heatmap korelasi dengan berbagai metrik
- **Distribution Visualization**: Histogram, boxplot, dan visualisasi distribusi
- **Bivariate Analysis**: Analisis hubungan antar variabel
- **Time Series Detection**: Deteksi otomatis data time series

### 4. 🎯 Rekomendasi Metode Penelitian & AI Analysis
- **Research Type Recommendations**: Rekomendasi jenis penelitian berbasis karakteristik dataset
- **Methodology Suggestions**: Saran metodologi penelitian yang tepat
- **Journal References**: Referensi jurnal ilmiah dengan impact factor dan DOI
- **AI-Powered Dataset Analysis**: Analisis potensi keberhasilan dataset untuk penelitian
- **ML Approach Recommendations**: Saran pendekatan machine learning yang sesuai

### 5. 🔄 Preprocessing Data Canggih
- **Automated Preprocessing**: Pipeline preprocessing otomatis
- **Feature Selection**: 8 metode seleksi fitur (Manual, MI, Pearson, RFE, LASSO, GB, RF, Ensemble)
- **Encoding Methods**: One-hot encoding, label encoding, target encoding
- **Scaling Options**: StandardScaler, MinMaxScaler, RobustScaler, Normalizer
- **Advanced Missing Value Handling**: 
  - Mean, median, mode, forward/backward fill, interpolation
  - **NEW**: Time-weighted interpolation untuk data time series
  - **NEW**: Multiple imputation strategies (auto, drop, impute)
  - **NEW**: Robust handling untuk missing data >30%
  - **NEW**: Smart detection dan handling untuk short time series
- **Outlier Detection**: IQR, Z-score, Isolation Forest untuk deteksi outlier

### 6. 🛠️ Feature Engineering & Pelatihan Model
- **Advanced Feature Engineering**: Polynomial features, interaction terms, binning
- **Comprehensive ML Models**: 10+ algoritma klasifikasi dan regresi
- **Hyperparameter Optimization**: GridSearchCV, RandomizedSearchCV, Bayesian Optimization (Optuna)
- **Smart Parameter Validation**: Validasi parameter otomatis berbasis karakteristik data
- **Preset Configurations**: Konfigurasi siap pakai untuk berbagai skenario
- **Custom Parameter Ranges**: Fleksibilitas definisi rentang parameter

### 7. 🔍 Interpretasi Model dengan SHAP & LIME
- **SHAP Values Analysis**: Visualisasi kontribusi fitur terhadap prediksi
- **Feature Importance**: Analisis kepentingan fitur global dan lokal
- **Model Interpretation**: Penjelasan interpretabel untuk hasil model
- **Prediction Explanation**: Detail kontribusi setiap fitur untuk prediksi individual
- **LIME Integration**: Local Interpretable Model-agnostic Explanations untuk interpretasi model
- **Multi-Model Support**: SHAP & LIME untuk klasifikasi, regresi, dan forecasting
- **Interactive Visualizations**: Plot interaktif untuk interpretasi model
- **Local & Global Explanations**: Penjelasan lokal per prediksi dan global per model
- **LIME Workflow**: Langkah-langkah menggunakan LIME dalam alur kerja penggunaan

### 8. 📈 Time Series Forecasting & Analisis Prediktif
- **Comprehensive Forecasting Models**:
  - **ARIMA/SARIMA**: Model statistik klasik untuk time series dengan **NEW**: robust handling untuk missing data
  - **Exponential Smoothing**: Holt-Winters untuk data dengan tren dan musiman
  - **LSTM**: Deep learning untuk prediksi kompleks jangka panjang dengan **NEW**: enhanced missing data imputation
  - **Random Forest & Gradient Boosting**: Ensemble learning untuk forecasting dengan **NEW**: improved data validation
  - **Linear Regression**: Baseline untuk perbandingan performa
  - **NEW**: Prophet Integration untuk forecasting dengan komponen tren dan musiman

- **Advanced Time Series Analysis**:
  - **Stationarity Testing**: Augmented Dickey-Fuller test otomatis
  - **Decomposition Analysis**: Analisis tren, musiman, dan komponen siklis
  - **ACF/PACF Analysis**: Plot autokorelasi untuk identifikasi order
  - **Seasonal Detection**: Deteksi otomatis pola musiman
  - **NEW**: VAR Models untuk multivariate time series

- **Enhanced Missing Data Handling**:
  - **NEW**: Time-weighted interpolation untuk time series dengan missing values
  - **NEW**: Smart handling untuk short time series (< 10 data points)
  - **NEW**: Multiple imputation strategies dengan validasi otomatis
  - **NEW**: Robust processing untuk high percentage missing data (>30%)

- **Forecasting Evaluation**: MAE, MSE, RMSE, MAPE, R² dengan confidence intervals
- **Multi-horizon Forecasting**: Prediksi multi-step ke depan
- **External Features**: Support fitur eksternal untuk forecasting

### 9. ⚠️ Deteksi Anomali Time Series
- **Multiple Algorithms**: 
  - **Isolation Forest**: Dengan **NEW**: robust missing data handling
  - **One-Class SVM**: Dengan **NEW**: enhanced feature engineering untuk missing values
  - **Statistical (Z-Score)**: Dengan **NEW**: improved statistical calculations
  - **LSTM Autoencoder**: Deep learning untuk deteksi kompleks dengan **NEW**: missing data imputation
  - **Prophet-based Detection**: **NEW**: Anomali detection menggunakan Prophet decomposition
  - **Ensemble Methods**: Kombinasi multiple algorithms untuk hasil optimal
- **Enhanced Missing Data Support**:
  - **NEW**: Semua algoritma mendukung data dengan missing values
  - **NEW**: Flexible fill methods (auto, drop, impute, interpolation)
  - **NEW**: Time-aware interpolation untuk time series data
  - **NEW**: Robust handling untuk high missing data percentages
- **Real-time Detection**: Deteksi anomali real-time pada data streaming
- **Configurable Parameters**: Threshold adjustment dan parameter kontrol
- **Comparative Analysis**: Perbandingan performa antar metode deteksi
- **Anomaly Visualization**: Visualisasi komprehensif hasil deteksi
- **Export Results**: Export hasil deteksi dengan flag anomali

### 10. 📊 Advanced Visualization & Reporting
- **Interactive Charts**: Plot interaktif dengan zoom, pan, dan hover
- **Multi-format Export**: Export hasil dalam CSV, PDF, PNG, dan HTML
- **Comprehensive Reports**: Laporan lengkap dengan metrik dan visualisasi
- **Dashboard Customization**: Panel kontrol yang dapat disesuaikan
- **Real-time Updates**: Update visualisasi secara real-time selama analisis

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
- XGBoost
- CatBoost
- LightGBM
- Extra Trees
- AdaBoost

### Regresi
- Random Forest
- Gradient Boosting
- Linear Regression
- Support Vector Regression (SVR)
- XGBoost Regressor
- CatBoost Regressor
- LightGBM Regressor
- Extra Trees Regressor
- Ridge Regression
- Lasso Regression
- Elastic Net
- Neural Network Regressor

### Time Series Forecasting
- **ARIMA/SARIMA**: Untuk data dengan pola musiman dengan **NEW**: robust missing data handling
- **Exponential Smoothing (Holt-Winters)**: Untuk data dengan tren dan musiman
- **LSTM**: Deep learning untuk prediksi kompleks dengan **NEW**: enhanced missing data imputation
- **Random Forest Regressor**: Untuk data time series dengan fitur eksternal dengan **NEW**: improved data validation
- **Gradient Boosting**: Untuk forecasting non-linear kompleks
- **Linear Regression**: Baseline untuk perbandingan performa
- **Prophet**: Facebook Prophet untuk forecasting dengan komponen tren dan musiman
- **VAR**: Vector Autoregression untuk multivariate time series

## Keunggulan Aplikasi
Aplikasi ini memiliki beberapa keunggulan dibandingkan tools sejenis:

### 1. **Advanced Model Interpretability with SHAP & LIME**
- **Dual Interpretation Engine**: Kombinasi SHAP untuk interpretasi global dan LIME untuk interpretasi lokal
- **Multi-Model Support**: Berfungsi untuk klasifikasi, regresi, dan time series forecasting
- **Interactive Visualizations**: Plot kontribusi fitur yang interaktif dan informatif
- **Local Explanations**: Penjelasan detail untuk setiap prediksi individual
- **Model-Agnostic**: Bekerja dengan semua jenis algoritma machine learning
- **Compliance Ready**: Interpretasi model untuk kebutuhan audit dan regulatory compliance

### 2. **All-in-One Research Platform**
- Platform lengkap untuk penelitian berbasis data dari awal hingga akhir
- Integrasi autentikasi pengguna, unggah data, analisis, modeling, dan reporting
- Tidak perlu berpindah-pindah tools atau environment
- Workflow penelitian yang seamless dan terintegrasi

### 3. **AI-Powered Research Recommendations**
- Rekomendasi jenis penelitian otomatis berbasis karakteristik dataset
- Saran metodologi dan pendekatan ML yang tepat
- Referensi jurnal ilmiah dengan impact factor dan DOI
- Analisis potensi keberhasilan dataset untuk penelitian

### 4. **Advanced Security & User Management**
- Sistem autentikasi lengkap dengan registrasi dan login
- Perlindungan captcha untuk keamanan dari bot
- Manajemen sesi pengguna yang aman dan terenkripsi
- Support multi-pengguna dengan isolasi data

### 5. **Comprehensive Data Support**
- Multi-format data: CSV, Excel (XLSX), ZIP (dataset gabungan)
- Support dataset train/test split dalam format ZIP
- Deteksi otomatis tipe data, kolom datetime, dan time series
- Penanganan missing values yang canggih dan otomatis

### 6. **Advanced Feature Engineering & Selection**
- 8+ metode seleksi fitur dengan ensemble capabilities
- Feature engineering otomatis (polynomial, interaction, binning)
- Encoding methods lengkap: one-hot, label, target encoding
- Multiple scaling options dan outlier detection

### 7. **State-of-the-Art ML Algorithms**
- 15+ algoritma klasifikasi dan regresi termasuk XGBoost, LightGBM, CatBoost
- Neural networks dan deep learning untuk pola kompleks
- Time series forecasting dengan ARIMA, LSTM, Prophet, VAR
- Ensemble methods untuk performa optimal

### 8. **Advanced Hyperparameter Optimization**
- 3 metode optimasi parameter: GridSearchCV, RandomizedSearchCV, Bayesian Optimization
- Smart parameter validation berbasis karakteristik data
- Preset configurations untuk berbagai skenario (Fast, High Accuracy, Small Dataset)
- Custom parameter ranges dengan fleksibilitas penuh

### 9. **SHAP & LIME Integration & Model Interpretability**
- Interpretasi model yang mendalam dengan SHAP values dan LIME explanations
- Visualisasi kontribusi fitur global dan lokal untuk semua jenis model
- Prediction explanation untuk setiap instance dengan detail kontribusi fitur
- Model interpretability untuk compliance dan audit
- Local Interpretable Model-agnostic Explanations (LIME) untuk interpretasi yang lebih intuitif
- Support untuk klasifikasi, regresi, dan time series forecasting

### 10. **Advanced Time Series & Anomaly Detection**
- Multiple forecasting algorithms dengan auto-parameter selection
- Anomaly detection real-time dengan berbagai metode
- Time series decomposition dan stationarity testing
- Multi-horizon forecasting dengan confidence intervals

### 11. **Interactive Visualization & Reporting**
- Visualisasi interaktif dengan zoom, pan, dan hover capabilities
- Multi-format export: CSV, PDF, PNG, HTML reports
- Real-time dashboard updates selama analisis
- Customizable dashboard layouts dan themes

### 12. **Bilingual & Accessibility Support**
- Dukungan Bahasa Indonesia dan English
- Interface yang dapat disesuaikan preferensi bahasa
- Accessibility features untuk pengguna dengan kebutuhan khusus
- Dokumentasi dan help system bilingual

### 13. **Scalability & Performance**
- Optimasi performa untuk dataset besar
- Parallel processing untuk training dan hyperparameter tuning
- Memory efficient untuk resource yang terbatas
- Caching system untuk hasil analisis yang cepat

### 14. **Enhanced Missing Data Handling (NEW)**
- **Advanced Imputation Methods**: Time-weighted interpolation, multiple imputation strategies
- **Robust Processing**: Handle missing data >30% dengan otomatis
- **Time Series Aware**: Smart interpolation untuk temporal data
- **Flexible Strategies**: Auto, drop, impute, atau custom handling
- **Validation System**: `validate_data_for_ml()` untuk data quality assurance
- **Short Series Support**: Special handling untuk time series pendek (<10 data points)
- **Cross-Method Consistency**: Missing data handling konsisten di semua algoritma

## Cara Menjalankan Aplikasi
1. Pastikan semua persyaratan telah terpenuhi dengan menjalankan:

pip install -r requirements.txt
python -m pip install "tensorflow"

2. Jalankan aplikasi dengan perintah:

streamlit run app.py

3. Aplikasi akan terbuka di browser web Anda secara otomatis (biasanya di http://localhost:8501 )

## Struktur File
```
├── app.py                 # File utama aplikasi Streamlit (5000+ baris)
├── requirements.txt       # Daftar dependensi Python
├── README.md              # File dokumentasi ini
├── test_missing_data_fixes_fixed.py  # **NEW**: Test script untuk missing data handling
├── utils.py               # **UPDATED**: Fungsi utilitas dengan enhanced missing data handling
├── forecasting_utils.py   # **UPDATED**: Forecasting dengan robust missing data support
├── anomaly_detection_utils.py  # **UPDATED**: Anomaly detection dengan missing data handling
├── 📁 assets/               # Folder untuk assets
│   ├── 📁 images/           # Folder untuk gambar dan ikon
│   ├── 📁 styles/           # Folder untuk styling CSS
│   ├── 📁 fonts/            # Folder untuk font kustom
├── 📁 models/               # Folder untuk menyimpan model ML
├── 📁 data/                 # Folder untuk data sample dan dataset
├── 📁 exports/              # Folder untuk hasil export (CSV, PDF, PNG)
├── 📁 temp/                 # Folder temporary untuk file sementara
├── 📁 logs/                  # Folder untuk log aplikasi dan error tracking
├── 📁 config/               # Folder untuk konfigurasi aplikasi
├── 📁 database/             # Folder untuk database SQLite
└── 📁 cache/                # Folder untuk caching hasil analisis

# File utama app.py berisi:
# - Sistem autentikasi dan manajemen pengguna
# - Fungsi-fungsi analisis data dan visualisasi
# - Implementasi 15+ algoritma ML untuk klasifikasi/regresi
# - Time series forecasting dengan 6+ metode
# - Anomaly detection dengan 4+ algoritma
# - SHAP integration untuk interpretasi model
# - Research recommendation system dengan AI analysis
# - Multi-language support (ID/EN)
# - Export system untuk multiple formats
```

## Fitur Baru & Update Terbaru

### 🆕 Enhanced Missing Data Handling (Update Terbaru)
- **Time-Weighted Interpolation**: Interpolasi cerdas untuk data time series dengan missing values
- **Multiple Imputation Strategies**: Opsi auto, drop, impute dengan validasi otomatis
- **Robust High Missing Data Processing**: Handle missing data >30% dengan algoritma adaptif
- **Short Time Series Support**: Special handling untuk time series pendek (<10 data points)
- **Cross-Algorithm Consistency**: Missing data handling konsisten di semua metode ML dan forecasting
- **Smart Validation System**: `validate_data_for_ml()` untuk quality assurance data
- **Enhanced Error Handling**: Validasi parameter dan error handling yang lebih robust

### 🔍 LIME Model Interpretation (Update Terbaru)
- **LIME for Classification**: Local Interpretable Model-agnostic Explanations untuk model klasifikasi
- **LIME for Regression**: Interpretasi lokal untuk model regresi dengan visualisasi kontribusi fitur
- **LIME for Forecasting**: Support LIME untuk time series forecasting models (ARIMA, LSTM, Prophet, dsb)
- **Multi-Algorithm Support**: LIME untuk Random Forest, Gradient Boosting, Neural Networks, dan semua model ML
- **Interactive Visualizations**: Plot kontribusi fitur interaktif dengan penjelasan detail
- **Feature Importance Local**: Analisis kepentingan fitur untuk prediksi individual
- **Model-Agnostic**: Bekerja dengan semua jenis model machine learning
- **Integration with SHAP**: Kombinasi LIME dan SHAP untuk interpretasi komprehensif

### 🔍 AI-Powered Research Recommendations
- **Analisis Otomatis Dataset**: Sistem AI yang menganalisis karakteristik dataset dan memberikan rekomendasi jenis penelitian yang sesuai
- **Rekomendasi Metodologi**: Saran metodologi penelitian berbasis tipe data dan domain penelitian
- **Referensi Jurnal Ilmiah**: Database jurnal internasional dengan impact factor, indexing, dan DOI lengkap
- **Potensi Keberhasilan Analisis**: Evaluasi keberhasilan dataset untuk penelitian dengan indikator khusus

### 🤖 Advanced Machine Learning Integration
- **15+ Algoritma ML Terbaru**: Termasuk CatBoost, LightGBM, dan Extra Trees untuk performa optimal
- **Neural Network Support**: Deep learning untuk pola kompleks dan dataset besar
- **AutoML Features**: Otomatisasi pemilihan model dan hyperparameter tuning
- **Ensemble Methods**: Kombinasi multiple algorithms untuk hasil terbaik

### 📊 Enhanced Time Series Analysis
- **Prophet Integration**: Facebook Prophet untuk forecasting dengan komponen tren dan musiman
- **VAR Models**: Vector Autoregression untuk multivariate time series
- **Multi-horizon Forecasting**: Prediksi multi-step dengan confidence intervals
- **Real-time Anomaly Detection**: Deteksi anomali pada data streaming

### 🛡️ Enterprise-Grade Security
- **User Authentication System**: Registrasi, login, dan session management yang aman
- **Captcha Protection**: Perlindungan dari automated attacks dan bot
- **Multi-user Support**: Isolasi data antar pengguna untuk keamanan privasi
- **Encrypted Data Storage**: Keamanan data dengan enkripsi terbaru

### 🌐 Multi-language & Accessibility
- **Bahasa Indonesia & English**: Dukungan penuh untuk kedua bahasa
- **Dynamic Language Switching**: Perubahan bahasa secara real-time
- **Accessibility Features**: Support untuk pengguna dengan kebutuhan khusus
- **Localized Interface**: Antarmuka yang disesuaikan dengan preferensi bahasa

### 📈 Advanced Visualization & Export
- **Interactive Dashboards**: Visualisasi interaktif dengan Plotly dan Altair
- **Multi-format Export**: CSV, PDF, PNG, HTML, dan format laporan khusus
- **Real-time Updates**: Dashboard yang update secara real-time selama analisis
- **Customizable Reports**: Template laporan yang dapat disesuaikan

## Persyaratan Sistem
- **Python**: 3.8 atau lebih baru (3.9+ direkomendasikan)
- **RAM**: Minimum 8GB (16GB direkomendasikan untuk dataset besar)
- **Storage**: Minimum 5GB ruang kosong (termasuk cache dan model storage)
- **OS**: Windows 10/11, macOS 10.15+, atau Linux (Ubuntu 18.04+)
- **Browser**: Chrome 90+, Firefox 88+, Safari 14+, atau Edge 90+
- **Internet**: Diperlukan untuk download dependencies dan update model

## Dependencies Utama
- **Streamlit**: 1.28+ untuk web interface
- **Pandas**: 1.5+ untuk data manipulation
- **Scikit-learn**: 1.3+ untuk machine learning
- **XGBoost**: 1.7+ untuk gradient boosting
- **LightGBM**: 3.3+ untuk high-performance boosting
- **CatBoost**: 1.2+ untuk categorical feature handling
- **SHAP**: 0.42+ untuk model interpretability
- **LIME**: 0.2.0+ untuk local model interpretability
- **Plotly**: 5.17+ untuk interactive visualization
- **Prophet**: 1.1+ untuk time series forecasting
- **SQLite**: Untuk user database dan session management

## Alur Kerja Penggunaan

### 1. **Autentikasi & Setup Awal**
   - Jalankan aplikasi dengan `streamlit run app.py`
   - Akses melalui browser di `http://localhost:8501`
   - Registrasi akun baru atau login dengan akun existing
   - Selesaikan captcha untuk keamanan
   - Pilih preferensi bahasa (Bahasa Indonesia/English)

### 2. **Unggah Data**
   - Pilih tab "Unggah Data"
   - Upload file CSV, Excel (XLSX), atau ZIP (dataset gabungan)
   - Sistem otomatis mendeteksi tipe data dan struktur
   - Lihat pratinjau data dan statistik deskriptif
   - Identifikasi missing values dan outliers otomatis

### 3. **AI-Powered Research Recommendations**
   - Gunakan tombol "Generate Rekomendasi Jenis Penelitian"
   - Sistem AI akan menganalisis karakteristik dataset
   - Dapatkan rekomendasi: jenis penelitian, metodologi, jurnal referensi
   - Lihat impact factor dan DOI jurnal yang direkomendasikan
   - Gunakan tombol "Generate Analisis Dataset" untuk evaluasi potensi keberhasilan

### 4. **Analisis Eksploratori Data (EDA)**
   - Pindah ke tab "Analisis Data"
   - Eksplorasi komprehensif dengan visualisasi interaktif
   - Analisis korelasi, distribusi, dan hubungan antar variabel
   - Deteksi otomatis time series dan pola musiman
   - Export visualisasi dalam berbagai format

### 5. **Preprocessing Data Canggih**
   - Buka tab "Preprocessing Data"
   - Pilih variabel target (otomatis untuk time series)
   - Konfigurasi penanganan missing values (mean, median, mode, advanced methods)
   - Pilih encoding untuk fitur kategorikal (one-hot, label, target)
   - Atur scaling untuk fitur numerik (Standard, MinMax, Robust, Normalizer)
   - Gunakan 8+ metode seleksi fitur dengan ensemble capabilities

### 6. **Training Model dengan Optimasi Hyperparameter**
   - Masuk ke tab "Training Model"
   - Pilih dari 15+ algoritma klasifikasi/regresi (XGBoost, LightGBM, CatBoost, Neural Networks)
   - Gunakan preset konfigurasi (Fast, High Accuracy, Small Dataset) atau custom
   - Pilih metode optimasi hyperparameter:
     - GridSearchCV untuk pencarian menyeluruh
     - RandomizedSearchCV untuk efisiensi
     - Bayesian Optimization (Optuna) untuk hasil optimal
   - Validasi parameter otomatis berbasis karakteristik data
   - Training dengan parallel processing untuk performa optimal

### 7. **Interpretasi Model dengan SHAP & LIME**
   - Gunakan tab "Interpretasi Model" untuk analisis mendalam
   - Visualisasi SHAP values untuk interpretasi model global
   - Gunakan LIME untuk interpretasi lokal per prediksi individual
   - Analisis feature importance global dan lokal
   - Dapatkan prediction explanation untuk setiap instance
   - Gunakan interpretasi untuk compliance dan audit
   - Support untuk klasifikasi, regresi, dan time series forecasting

### 8. **Time Series Forecasting & Anomaly Detection**
   - Untuk data temporal, gunakan tab "Analisis Time Series":
     - Pilih dari 6+ algoritma forecasting (ARIMA, LSTM, Prophet, VAR)
     - Otomatis parameter selection untuk ARIMA
     - Multi-horizon forecasting dengan confidence intervals
     - Dekomposisi time series dan stationarity testing
   - Gunakan tab "Deteksi Anomali" untuk:
     - Real-time anomaly detection dengan 4+ metode
     - Comparative analysis antar metode deteksi
     - Visualisasi interaktif hasil deteksi
     - Export hasil dengan flag anomali

### 9. **Prediksi Data Baru & Batch Processing**
   - Pilih tab "Prediksi Data Baru"
   - Upload file CSV untuk batch prediction atau input manual
   - Preprocessing otomatis sesuai pipeline training
   - Real-time prediction dengan model yang telah dilatih
   - Export hasil prediksi dalam multiple formats (CSV, PDF, HTML report)

### 10. **Export & Reporting Lanjutan**
   - Export model yang telah dilatih untuk digunakan kembali
   - Generate comprehensive reports dengan metrik dan visualisasi
   - Customizable report templates untuk kebutuhan spesifik
   - Multi-format export untuk berbagai keperluan (akademik, bisnis, presentasi)

## Tips dan Trik

### Untuk Optimasi Parameter yang Efektif:
- **Dataset Kecil (< 1000 sampel)**: Gunakan preset "Small Dataset" atau RandomizedSearchCV untuk menghindari overfitting
- **Dataset Besar (> 10000 sampel)**: Gunakan GridSearchCV atau Bayesian Optimization untuk hasil optimal
- **Waktu Terbatas**: Gunakan preset "Fast Training" atau RandomizedSearchCV dengan iterasi minimal
- **Akurasi Maksimal**: Gunakan preset "High Accuracy" atau Bayesian Optimization dengan iterasi banyak
- **Kolaborasi Tim**: Export konfigurasi parameter yang berhasil ke JSON dan bagikan dengan tim

### Troubleshooting & FAQ

#### Masalah Umum:

**Q: Aplikasi tidak bisa dijalankan?**
A: Pastikan Python 3.8+ terinstall dan semua dependencies terinstall dengan `pip install -r requirements.txt`

**Q: Dataset besar membuat aplikasi lambat?**
A: Gunakan fitur sampling untuk EDA, dan processing penuh hanya untuk modeling. Pertimbangkan untuk upgrade RAM ke 16GB+

**Q: Bagaimana menangani missing data yang tinggi (>30%)?**
A: Gunakan fitur **Enhanced Missing Data Handling** yang baru:
- Pilih strategi "auto" untuk handling otomatis
- Gunakan "impute" dengan time-weighted interpolation untuk time series
- Gunakan "drop" jika missing data terlalu tinggi
- Sistem akan memberikan warning dan rekomendasi otomatis

**Q: Time series pendek (<10 data points) error?**
A: Fitur baru **Short Time Series Support** secara otomatis akan:
- Gunakan interpolasi yang sesuai untuk data pendek
- Berikan warning tentang keterbatasan prediksi
- Sarankan metode yang tepat untuk dataset kecil
- Validasi parameter secara otomatis

**Q: Error "time-weighted interpolation requires DatetimeIndex"?**
A: Pastikan data time series Anda memiliki datetime column yang benar:
- Format tanggal harus konsisten
- Gunakan fungsi validasi data sebelum processing
- Sistem akan otomatis konversi jika format sesuai

**Q: Model training gagal atau error?**
A: Periksa kualitas data (missing values, outliers), pastikan target variable sesuai, dan coba gunakan preset parameter default

**Q: Captcha tidak muncul atau error?**
A: Refresh browser atau clear cache. Pastikan JavaScript enabled di browser

**Q: Time series forecasting hasilnya tidak akurat?**
A: Periksa stasioneritas data, coba dekomposisi, dan pastikan frekuensi data sudah benar

**Q: SHAP visualization error?**
A: Untuk model kompleks, gunakan subset data untuk SHAP analysis. Pastikan memory cukup untuk computation

**Q: Export PDF gagal?**
A: Install wkhtmltopdf untuk PDF export, atau gunakan format HTML sebagai alternatif

#### Error Messages dan Solusi:

**"Out of Memory" Error**: 
- Gunakan sampling untuk dataset > 1GB
- Kurangi jumlah features dengan feature selection
- Restart aplikasi dan clear cache

**"Model Not Converging" Error**:
- Kurangi complexity (max_depth, n_estimators)
- Gunakan preset "Fast Training" sebagai starting point
- Periksa data untuk multicollinearity

**"CUDA/GPU Memory" Error** (untuk Neural Networks):
- Gunakan CPU mode untuk testing
- Kurangi batch size dan network complexity
- Pertimbangkan model yang lebih sederhana

#### Performance Optimization:
- **Parallel Processing**: Aktifkan untuk training dan hyperparameter tuning
- **Caching**: Gunakan fitur caching untuk analisis yang berulang
- **Model Persistence**: Simpan dan load model untuk avoid retraining
- **Incremental Learning**: Gunakan untuk dataset yang sangat besar

#### Best Practices:
- **Data Validation**: Selalu lakukan data validation sebelum analysis
- **Version Control**: Simpan konfigurasi dan hasil untuk reproducibility
- **Documentation**: Dokumentasikan setiap step untuk future reference
- **Backup**: Backup model dan konfigurasi secara berka

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

## Kontribusi & Development

Kontribusi sangat diterima! Kami menyambut kontribusi dari komunitas untuk meningkatkan aplikasi ini.

### Area Pengembangan:
- **Machine Learning**: Penambahan algoritma ML baru dan improvement existing models
- **Deep Learning**: Integrasi PyTorch, TensorFlow untuk model kompleks
- **Natural Language Processing**: Analisis text data dan sentiment analysis
- **Computer Vision**: Image processing dan feature extraction
- **Big Data**: Support untuk dataset > 10GB dengan distributed processing
- **Cloud Integration**: AWS, GCP, Azure integration untuk scalable deployment
- **API Development**: RESTful API untuk integration dengan aplikasi lain
- **Mobile Support**: Progressive Web App (PWA) untuk mobile devices
- **Real-time Analytics**: Streaming data processing dan real-time dashboards
- **AutoML Enhancement**: Automated feature engineering dan model selection

### Cara Berkontribusi:
1. Fork repository ini
2. Buat branch untuk fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan Anda (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request dengan deskripsi yang jelas

### Code Standards:
- Ikuti PEP 8 untuk Python code style
- Tambahkan docstrings untuk semua fungsi dan classes
- Buat unit tests untuk fitur baru
- Update dokumentasi untuk perubahan signifikan
- Gunakan type hints untuk code clarity

## Lisensi & Disclaimer

### Lisensi
Proyek ini bersifat open source dan dapat digunakan secara bebas untuk keperluan:
- **Pendidikan**: Penggunaan dalam kurikulum akademik
- **Penelitian**: Penelitian ilmiah dan publikasi
- **Pengembangan**: Base untuk proyek komersial dan non-komersial
- **Komersial**: Penggunaan dalam produk dan layanan komersial

### Disclaimer
- Aplikasi ini disediakan "as-is" tanpa garansi apapun
- Hasil analisis dan prediksi bergantung pada kualitas data input
- Selalu validasi hasil dengan domain expert untuk aplikasi kritis
- Penulis tidak bertanggung jawab atas penggunaan yang tidak tepat

### Citation
Jika Anda menggunakan aplikasi ini untuk penelitian atau publikasi, mohon cite:
```
Asmeranda ML Research Platform - Advanced Machine Learning Analysis Tool
Available at: [repository-url]
```


### Community:
- 🌟 Star repository ini jika Anda merasa terbantu
- 🔄 Share dengan komunitas ML dan data science
- 🤝 Kontribusi dalam bentuk code, dokumentasi, atau testing
- 📢 Beritahu kami jika Anda menggunakan aplikasi ini untuk proyek menarik!

---
**Catatan**: Aplikasi ini terus dikembangkan. Pastikan untuk selalu memperbarui ke versi terbaru untuk mendapatkan fitur dan perbaikan terbaru.
