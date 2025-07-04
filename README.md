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
- Penskalaan fitur numerik

### 4. 🛠️ Feature Engineering & Pelatihan Model
- Pemilihan fitur dengan berbagai metode
- Pemilihan algoritma machine learning
- Pembagian data training dan testing
- Pelatihan model dengan parameter yang dapat disesuaikan
- Evaluasi performa model

### 5. 🔍 Interpretasi Model dengan SHAP
- Visualisasi nilai SHAP untuk memahami kontribusi fitur
- Analisis pengaruh fitur terhadap prediksi model

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

## Cara Menjalankan Aplikasi
1. Pastikan semua persyaratan telah terpenuhi dengan menjalankan:

pip install -r requirements.txt

2. Jalankan aplikasi dengan perintah:

streamlit run app.py

3. Aplikasi akan terbuka di browser web Anda secara otomatis (biasanya di http://localhost:8501 )

## Alur Kerja Penggunaan
1. Unggah dataset Anda di tab "Data Upload"
2. Lakukan eksplorasi data di tab "Exploratory Data Analytic"
3. Lakukan preprocessing data di tab "Preprocessing"
4. Latih dan evaluasi model di tab "Feature Engineering & Model Training"
5. Interpretasikan hasil model dengan SHAP di tab "SHAP Model Interpretation"
