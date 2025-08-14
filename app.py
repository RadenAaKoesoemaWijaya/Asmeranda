import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier, BaggingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.inspection import partial_dependence, PartialDependenceDisplay
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from kmodes.kprototypes import KPrototypes
import shap
import pickle
import os
from PIL import Image
import io
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTEENN, SMOTETomek
    IMB_AVAILABLE = True
except ImportError:
    IMB_AVAILABLE = False

# Initialize translation
TRANSLATIONS = {
    'en': {
        'app_title': 'Comprehensive Machine Learning App',
        'app_description': 'This application helps you analyze your data, preprocess it for machine learning, train models, and interpret the results using eXplainable AI.',
        # Add more translations here
    },
    'id': {
        'app_title': 'Aplikasi Machine Learning Komprehensif',
        'app_description': 'Aplikasi ini membantu Anda menganalisis data, memprosesnya untuk machine learning, melatih model, dan menginterpretasikan hasil menggunakan eXplainable AI.',
        # Add more translations here
    }
}

# Set page configuration
st.set_page_config(
    page_title="EDA & ML App",
    page_icon="📊",
    layout="wide"
)

# Language toggle button
if 'language' not in st.session_state:
    st.session_state.language = 'id'  # Default language is Indonesian

col1, col2 = st.columns([0.9, 0.1])
with col2:
    if st.button('ID' if st.session_state.language == 'en' else 'EN'):
        st.session_state.language = 'id' if st.session_state.language == 'en' else 'en'

# App title and description
st.title(f"📊 {TRANSLATIONS[st.session_state.language]['app_title']}")
st.markdown(TRANSLATIONS[st.session_state.language]['app_description'])

# Initialize session state variables
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'categorical_columns' not in st.session_state:
    st.session_state.categorical_columns = []
if 'numerical_columns' not in st.session_state:
    st.session_state.numerical_columns = []
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Data Upload", 
    "📊 Exploratory Data Analytic", 
    "🔄 Preprocessing and Feature Engineering", 
    "🛠️ Cross Validation and Model Training", 
    "🔍 SHAP Model Interpretation", 
    "🔎 LIME Model Interpretation",
    "📈 Partial Dependence Plot"
])

def adjusted_r2_score(r2, n, k):
    """Hitung Adjusted R².""" if st.session_state.language == 'id' else """Calculate Adjusted R²."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def calculate_vif(X):
    """Hitung Variance Inflation Factor (VIF) untuk setiap fitur. """ if st.session_state.language == 'id' else """Calculate VIF for each feature."""
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

def breusch_pagan_test(y_true, y_pred, X):
    """ Tampilkan Uji Heteroskedastisitas (Breusch-Pagan)""" if st.session_state.language == 'id' else """Perform Breusch-Pagan test for heteroskedasticity."""
    residuals = y_true - y_pred
    X_const = sm.add_constant(X)
    bp_test = het_breuschpagan(residuals, X_const)
    labels = ['Lagrange multiplier statistic', 'p-value', 'f-value', 'f p-value']
    return dict(zip(labels, bp_test))

# Tab 1: Data Upload
with tab1:
    st.header("Unggah Dataset Anda" if st.session_state.language == 'id' else "Upload Your Dataset")
    
    uploaded_file = st.file_uploader(
        "Pilih file CSV atau ZIP berisi folder train/test" if st.session_state.language == 'id' else "Choose a CSV file or ZIP with train/test folders",
        type=["csv", "zip"]
    )
    
    if uploaded_file is not None:
        import zipfile
        import tempfile
        import os
        if uploaded_file.name.endswith('.zip'):
            # Proses ZIP: cari file train/test dalam folder atau file langsung
            with tempfile.TemporaryDirectory() as tmpdir:
                zf = zipfile.ZipFile(uploaded_file)
                zf.extractall(tmpdir)
                # Cari file train dan test
                train_path, test_path = None, None
                train_files, test_files = [], []
                
                for root, dirs, files in os.walk(tmpdir):
                    for f in files:
                        if f.endswith('.csv'):
                            full_path = os.path.join(root, f)
                            # Cek apakah file ada di folder training atau testing
                            if 'train' in root.lower() or 'train' in f.lower():
                                train_files.append(full_path)
                            elif 'test' in root.lower() or 'test' in f.lower():
                                test_files.append(full_path)
                            # Fallback: cek nama file saja
                            elif 'train' in f.lower():
                                train_files.append(full_path)
                            elif 'test' in f.lower():
                                test_files.append(full_path)
                
                # Ambil file pertama dari masing-masing kategori
                if train_files:
                    train_path = train_files[0]
                if test_files:
                    test_path = test_files[0]
                
                if train_path and test_path:
                    try:
                        train_data = pd.read_csv(train_path)
                        test_data = pd.read_csv(test_path)
                        
                        # Gabungkan dataset training dan testing
                        combined_data = pd.concat([train_data, test_data], ignore_index=True)
                        
                        # Simpan ke session state
                        st.session_state.data = combined_data
                        st.session_state.train_data = train_data
                        st.session_state.test_data = test_data
                        
                        st.success(f"Berhasil mendeteksi dan memuat data training ({train_data.shape[0]} baris) dan testing ({test_data.shape[0]} baris) dari ZIP." if st.session_state.language == 'id' else f"Successfully loaded training ({train_data.shape[0]} rows) and testing ({test_data.shape[0]} rows) from ZIP.")
                        st.info(f"Dataset gabungan: {combined_data.shape[0]} baris dan {combined_data.shape[1]} kolom." if st.session_state.language == 'id' else f"Combined dataset: {combined_data.shape[0]} rows and {combined_data.shape[1]} columns.")
                        
                        # Tampilkan preview dataset gabungan
                        st.subheader("Preview Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Preview")
                        st.dataframe(combined_data.head())
                        
                        # Tampilkan informasi dataset terpisah
                        col1, col2 = st.columns(2)
                        with col1:
                            st.subheader("Dataset Training" if st.session_state.language == 'id' else "Training Dataset")
                            st.dataframe(train_data.head())
                        with col2:
                            st.subheader("Dataset Testing" if st.session_state.language == 'id' else "Testing Dataset")
                            st.dataframe(test_data.head())
                            
                    except Exception as e:
                        st.error(f"Error saat membaca file CSV: {e}" if st.session_state.language == 'id' else f"Error reading CSV files: {e}")
                        
                elif train_path or test_path:
                    # Jika hanya ada satu file, gunakan sebagai dataset utama
                    single_path = train_path or test_path
                    try:
                        data = pd.read_csv(single_path)
                        st.session_state.data = data
                        st.session_state.train_data = None
                        st.session_state.test_data = None
                        st.success(f"Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom." if st.session_state.language == 'id' else f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
                        st.dataframe(data.head())
                    except Exception as e:
                        st.error(f"Error saat membaca file CSV: {e}" if st.session_state.language == 'id' else f"Error reading CSV files: {e}")
                else:
                    st.error("ZIP tidak berisi file train/test CSV yang valid." if st.session_state.language == 'id' else "ZIP does not contain valid train/test CSV files.")
                    st.info("Pastikan ZIP berisi folder 'training' dan 'testing' dengan file CSV, atau file dengan nama yang mengandung 'train' dan 'test'." if st.session_state.language == 'id' else "Make sure ZIP contains 'training' and 'testing' folders with CSV files, or files with names containing 'train' and 'test'.")
        else:
            # Proses single CSV
            try:
                data = pd.read_csv(uploaded_file)
                st.session_state.data = data
                st.session_state.train_data = None
                st.session_state.test_data = None
                st.success(f"Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom." if st.session_state.language == 'id' else f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
                st.dataframe(data.head())
            except Exception as e:
                st.error(f"Error: {e}")
            
            st.subheader("Informasi Data" if st.session_state.language == 'id' else "Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Tambahkan pemilihan fitur untuk dibuang
            st.subheader("Pemilihan Fitur" if st.session_state.language == 'id' else "Feature Selection")
            all_columns = data.columns.tolist()
            
            # Gunakan session state untuk menyimpan fitur yang dipilih untuk dibuang
            if 'columns_to_drop' not in st.session_state:
                st.session_state.columns_to_drop = []
            
            columns_to_drop = st.multiselect(
                "Pilih kolom yang ingin dibuang:" if st.session_state.language == 'id' else "Select columns to drop:",
                all_columns,
                default=st.session_state.columns_to_drop
            )
            
            if columns_to_drop:
                st.warning(f"Kolom yang akan dibuang: {', '.join(columns_to_drop)}")
                
                # Update dataset dengan menghapus kolom yang dipilih
                data = data.drop(columns=columns_to_drop)
                st.session_state.data = data
                st.session_state.columns_to_drop = columns_to_drop
                
                st.success(f"Dataset telah diperbarui. Ukuran baru: {data.shape[0]} baris × {data.shape[1]} kolom")
                st.dataframe(data.head())
            
            st.subheader("Statistik Data" if st.session_state.language == 'id' else "Data Statistics")
            st.dataframe(data.describe())
            
            # Identify numerical and categorical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            st.session_state.numerical_columns = numerical_cols
            st.session_state.categorical_columns = categorical_cols
            
            st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
            st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")
        
        # Tambahkan informasi kolom untuk dataset gabungan dari ZIP
        if uploaded_file.name.endswith('.zip') and st.session_state.data is not None:
            data = st.session_state.data
            st.subheader("Informasi Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            # Tambahkan pemilihan fitur untuk dibuang
            st.subheader("Pemilihan Fitur" if st.session_state.language == 'id' else "Feature Selection")
            all_columns = data.columns.tolist()
            
            # Gunakan session state untuk menyimpan fitur yang dipilih untuk dibuang
            if 'columns_to_drop' not in st.session_state:
                st.session_state.columns_to_drop = []
            
            columns_to_drop = st.multiselect(
                "Pilih kolom yang ingin dibuang:" if st.session_state.language == 'id' else "Select columns to drop:",
                all_columns,
                default=st.session_state.columns_to_drop
            )
            
            if columns_to_drop:
                st.warning(f"Kolom yang akan dibuang: {', '.join(columns_to_drop)}")
                
                # Update dataset dengan menghapus kolom yang dipilih
                data = data.drop(columns=columns_to_drop)
                st.session_state.data = data
                st.session_state.columns_to_drop = columns_to_drop
                
                st.success(f"Dataset telah diperbarui. Ukuran baru: {data.shape[0]} baris × {data.shape[1]} kolom")
                st.dataframe(data.head())
            
            st.subheader("Statistik Dataset Gabungan" if st.session_state.language == 'id' else "Combined Dataset Statistics")
            st.dataframe(data.describe())
            
            # Identify numerical and categorical columns untuk dataset gabungan
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            st.session_state.numerical_columns = numerical_cols
            st.session_state.categorical_columns = categorical_cols
            
            st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
            st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")

# Tab 2: Exploratory Data Analysis
with tab2:
    st.header("Analisis Data Eksplorasi" if st.session_state.language == 'id' else "Exploratory Data Analysis")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Missing values analysis
        st.subheader("Analisis Nilai Hilang" if st.session_state.language == 'id' else "Missing Values Analysis")
        missing_values = data.isnull().sum()
        missing_percentage = (missing_values / len(data)) * 100
        missing_df = pd.DataFrame({
            'Missing Values': missing_values,
            'Percentage (%)': missing_percentage
        })
        st.dataframe(missing_df)
        
        # Plot missing values
        if missing_values.sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_df[missing_df['Missing Values'] > 0]['Percentage (%)'].sort_values(ascending=False).plot(kind='bar', ax=ax)
            plt.title('Persentase Nilai Hilang' if st.session_state.language == 'id' else 'Missing Values Percentage')
            plt.ylabel('Persentase (%)' if st.session_state.language == 'id' else 'Percentage (%)')
            plt.xlabel('Kolom' if st.session_state.language == 'id' else 'Columns')
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.info("Tidak ditemukan nilai yang hilang dalam dataset." if st.session_state.language == 'id' else "No missing values found in the dataset.")
        
        # Correlation analysis for numerical columns
        if len(st.session_state.numerical_columns) > 1:
            st.subheader("Analisis Korelasi" if st.session_state.language == 'id' else "Correlation Analysis")
            correlation = data[st.session_state.numerical_columns].corr()
            
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', ax=ax, fmt=".2f")
            plt.title('Matriks Korelasi' if st.session_state.language == 'id' else 'Correlation Matrix')
            st.pyplot(fig)
        
        # Distribution of numerical columns
        if len(st.session_state.numerical_columns) > 0:
            st.subheader("Distribusi Fitur Numerik" if st.session_state.language == 'id' else "Distribution of Numerical Features")
            
            selected_num_col = st.selectbox("Pilih kolom numerik untuk analisis distribusi:" if st.session_state.language == 'id' else "Select a numerical column for distribution analysis:", 
                                           st.session_state.numerical_columns)
            
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            
            # Histogram
            sns.histplot(data[selected_num_col].dropna(), kde=True, ax=ax[0])
            ax[0].set_title(f'Histogram {selected_num_col}')
            
            # Box plot
            sns.boxplot(y=data[selected_num_col].dropna(), ax=ax[1])
            ax[1].set_title(f'Boxplot {selected_num_col}')
            
            st.pyplot(fig)
        
        # Distribution of categorical columns
        if len(st.session_state.categorical_columns) > 0:
            st.subheader("Distribusi Fitur Kategorikal" if st.session_state.language == 'id' else "Distribution of Categorical Features")
            
            selected_cat_col = st.selectbox("Pilih kolom kategorikal untuk analisis distribusi:" if st.session_state.language == 'id' else "Select a categorical column for distribution analysis:", 
                                           st.session_state.categorical_columns)
            
            # Count plot
            fig, ax = plt.subplots(figsize=(12, 6))
            value_counts = data[selected_cat_col].value_counts()
            
            # If there are too many categories, show only top 20
            if len(value_counts) > 20:
                st.warning(f"Kolom ini memiliki {len(value_counts)} nilai unik. Hanya menampilkan 20 teratas." if st.session_state.language == 'id' else f"The column has {len(value_counts)} unique values. Showing only top 20.")
                value_counts = value_counts.head(20)
            
            sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
            plt.title(f'Jumlah {selected_cat_col}' if st.session_state.language == 'id' else f'Count of {selected_cat_col}')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Bivariate analysis
        st.subheader("Analisis Bivariat" if st.session_state.language == 'id' else "Bivariate Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_axis = st.selectbox("Pilih X-axis feature:" if st.session_state.language == 'id' else "Select X-axis feature:", data.columns)
        
        with col2:
            y_axis = st.selectbox("Pilih Y-axis feature:" if st.session_state.language == 'id' else "Select Y-axis feature:", [col for col in data.columns if col != x_axis])
        
        # Determine the plot type based on the data types
        x_is_numeric = data[x_axis].dtype in ['int64', 'float64']
        y_is_numeric = data[y_axis].dtype in ['int64', 'float64']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if x_is_numeric and y_is_numeric:
            # Scatter plot for numeric vs numeric
            sns.scatterplot(x=x_axis, y=y_axis, data=data, ax=ax)
            plt.title(f'Scatter plot of {x_axis} vs {y_axis}')
        elif x_is_numeric and not y_is_numeric:
            # Box plot for numeric vs categorical
            sns.boxplot(x=y_axis, y=x_axis, data=data, ax=ax)
            plt.title(f'Box plot of {x_axis} by {y_axis}')
        elif not x_is_numeric and y_is_numeric:
            # Box plot for categorical vs numeric
            sns.boxplot(x=x_axis, y=y_axis, data=data, ax=ax)
            plt.title(f'Box plot of {y_axis} by {x_axis}')
        else:
            # Count plot for categorical vs categorical
            pd.crosstab(data[x_axis], data[y_axis]).plot(kind='bar', stacked=True, ax=ax)
            plt.title(f'Stacked bar plot of {x_axis} and {y_axis}')
        
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Silakan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")

    if st.session_state.numerical_columns or st.session_state.categorical_columns:
        
        # Unsupervised Machine Learning Analysis
        st.subheader("Analisis Machine Learning Unsupervised" if st.session_state.language == 'id' else "Unsupervised Machine Learning Analysis")
        
        # Select features for clustering
        st.write("Pilih fitur untuk analisis clustering:" if st.session_state.language == 'id' else "Select features for clustering analysis:")
        
        # Combine numerical and categorical columns for selection
        all_columns = st.session_state.numerical_columns + st.session_state.categorical_columns
        selected_features = st.multiselect(
            "Pilih fitur:" if st.session_state.language == 'id' else "Select features:",
            all_columns,
            default=st.session_state.numerical_columns[:min(3, len(st.session_state.numerical_columns))]
        )
        
        if selected_features:
            # Prepare data for clustering
            clustering_data = data[selected_features].copy()
            
            # Handle categorical variables
            categorical_in_selected = [col for col in selected_features if col in st.session_state.categorical_columns]
            if categorical_in_selected:
                # Encode categorical variables
                le = LabelEncoder()
                for col in categorical_in_selected:
                    clustering_data[col] = le.fit_transform(clustering_data[col].astype(str))
            
            # Handle missing values
            clustering_data = clustering_data.dropna()
            
            if len(clustering_data) > 0:
                # Standardize the data
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(clustering_data)
                
                # Select clustering method
                clustering_method = st.selectbox(
                    "Pilih metode clustering:" if st.session_state.language == 'id' else "Select clustering method:",
                    ["K-Means", "K-Prototypes", "Hierarchical", "DBSCAN", "Spectral"]
                )
                
                if clustering_method == "K-Means":
                    # K-Means Clustering
                    max_k = min(10, len(clustering_data) - 1)
                    k_value = st.slider(
                        "Jumlah cluster (k):" if st.session_state.language == 'id' else "Number of clusters (k):",
                        2, max_k, 3
                    )
                    
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Calculate silhouette score
                    if len(set(clusters)) > 1:
                        silhouette = silhouette_score(scaled_data, clusters)
                        st.write(f"Silhouette Score: {silhouette:.3f}")
                    
                    # Add cluster labels to data
                    clustering_data['Cluster'] = clusters
                    
                    # Visualize clusters
                    if len(selected_features) >= 2:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Scatter plot of first two features
                        scatter = ax1.scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        ax1.set_xlabel(selected_features[0])
                        ax1.set_ylabel(selected_features[1])
                        ax1.set_title('Visualisasi Cluster' if st.session_state.language == 'id' else 'Cluster Visualization')
                        plt.colorbar(scatter, ax=ax1)
                        
                        # PCA visualization
                        if len(selected_features) > 2:
                            pca = PCA(n_components=2)
                            pca_data = pca.fit_transform(scaled_data)
                            scatter2 = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                            ax2.set_xlabel('PC1')
                            ax2.set_ylabel('PC2')
                            ax2.set_title('PCA - Visualisasi Cluster' if st.session_state.language == 'id' else 'PCA - Cluster Visualization')
                            plt.colorbar(scatter2, ax=ax2)
                        
                        st.pyplot(fig)
                    
                    # Show cluster statistics
                    st.write("Statistik per cluster:" if st.session_state.language == 'id' else "Cluster statistics:")
                    cluster_stats = clustering_data.groupby('Cluster').agg({
                        col: ['count', 'mean', 'std'] for col in selected_features
                    }).round(3)
                    st.dataframe(cluster_stats)
                
                elif clustering_method == "K-Prototypes":
                    # K-Prototypes Clustering for mixed data types
                    max_k = min(10, len(clustering_data) - 1)
                    k_value = st.slider(
                        "Jumlah cluster (k):" if st.session_state.language == 'id' else "Number of clusters (k):",
                        2, max_k, 3
                    )
                    
                    gamma = st.slider(
                        "Gamma (bobot kategorikal):" if st.session_state.language == 'id' else "Gamma (categorical weight):",
                        0.0, 1.0, 0.5, 0.1
                    )
                    
                    # Prepare data for K-Prototypes
                    categorical_idx = [i for i, col in enumerate(selected_features) 
                                     if col in st.session_state.categorical_columns]
                    
                    # Convert categorical columns to appropriate types
                    kproto_data = clustering_data.copy()
                    for col in categorical_in_selected:
                        kproto_data[col] = kproto_data[col].astype('category').cat.codes
                    
                    # Initialize and fit K-Prototypes
                    kproto = KPrototypes(n_clusters=k_value, init='Huang', random_state=42, gamma=gamma)
                    clusters = kproto.fit_predict(kproto_data.values, categorical=categorical_idx)
                    
                    # Calculate silhouette score (only for numerical features)
                    if len(set(clusters)) > 1 and len([col for col in selected_features 
                                                      if col in st.session_state.numerical_columns]) > 0:
                        # Use only numerical features for silhouette score
                        num_features = [col for col in selected_features 
                                      if col in st.session_state.numerical_columns]
                        if num_features:
                            num_data = clustering_data[num_features]
                            num_scaled = StandardScaler().fit_transform(num_data)
                            silhouette = silhouette_score(num_scaled, clusters)
                            st.write(f"Silhouette Score (numerical): {silhouette:.3f}")
                    
                    # Add cluster labels to data
                    clustering_data['Cluster'] = clusters
                    
                    # Visualize clusters
                    if len(selected_features) >= 2:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Scatter plot using first two features
                        scatter = ax1.scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        ax1.set_xlabel(selected_features[0])
                        ax1.set_ylabel(selected_features[1])
                        ax1.set_title('K-Prototypes Clustering' if st.session_state.language == 'id' else 'K-Prototypes Clustering')
                        plt.colorbar(scatter, ax=ax1)
                        
                        # PCA visualization for numerical features
                        num_features = [col for col in selected_features 
                                      if col in st.session_state.numerical_columns]
                        if len(num_features) >= 2:
                            pca = PCA(n_components=2)
                            pca_data = pca.fit_transform(clustering_data[num_features])
                            scatter2 = ax2.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                            ax2.set_xlabel('PC1')
                            ax2.set_ylabel('PC2')
                            ax2.set_title('PCA - K-Prototypes Clustering' if st.session_state.language == 'id' else 'PCA - K-Prototypes Clustering')
                            plt.colorbar(scatter2, ax=ax2)
                        
                        st.pyplot(fig)
                    
                    # Show cluster statistics
                    st.write("Statistik per cluster:" if st.session_state.language == 'id' else "Cluster statistics:")
                    cluster_stats = clustering_data.groupby('Cluster').agg({
                        col: ['count', 'mean', 'std'] if col in st.session_state.numerical_columns else ['count', 'nunique']
                        for col in selected_features
                    }).round(3)
                    st.dataframe(cluster_stats)

                elif clustering_method == "Hierarchical":
                    # Hierarchical Clustering
                    linkage_method = st.selectbox(
                        "Metode linkage:" if st.session_state.language == 'id' else "Linkage method:",
                        ["ward", "complete", "average", "single"]
                    )
                    n_clusters = st.slider(
                        "Jumlah cluster:" if st.session_state.language == 'id' else "Number of clusters:",
                        2, min(10, len(clustering_data) - 1), 3
                    )
                    
                    hierarchical = AgglomerativeClustering(
                        n_clusters=n_clusters, 
                        linkage=linkage_method
                    )
                    clusters = hierarchical.fit_predict(scaled_data)
                    
                    # Calculate silhouette score
                    if len(set(clusters)) > 1:
                        silhouette = silhouette_score(scaled_data, clusters)
                        st.write(f"Silhouette Score: {silhouette:.3f}")
                    
                    # Dendrogram
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Dendrogram plot
                    linkage_matrix = linkage(scaled_data[:min(100, len(scaled_data))], method=linkage_method)
                    dendrogram(linkage_matrix, ax=ax1)
                    ax1.set_title('Dendrogram' if st.session_state.language == 'id' else 'Dendrogram')
                    ax1.set_xlabel('Sample Index')
                    ax1.set_ylabel('Distance')
                    
                    # Cluster visualization
                    if len(selected_features) >= 2:
                        scatter = ax2.scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        ax2.set_xlabel(selected_features[0])
                        ax2.set_ylabel(selected_features[1])
                        ax2.set_title('Hierarchical Clustering' if st.session_state.language == 'id' else 'Hierarchical Clustering')
                        plt.colorbar(scatter, ax=ax2)
                    
                    st.pyplot(fig)
                    
                    # Add cluster labels
                    clustering_data['Cluster'] = clusters
                    st.write("Distribusi cluster:" if st.session_state.language == 'id' else "Cluster distribution:")
                    st.write(clustering_data['Cluster'].value_counts())
                
                elif clustering_method == "DBSCAN":
                    # DBSCAN Clustering
                    eps = st.slider("Eps (radius neighborhood):", 0.1, 5.0, 0.5, 0.1)
                    min_samples = st.slider("Min samples:", 1, 20, 5)
                    
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(scaled_data)
                    
                    # Calculate silhouette score
                    if len(set(clusters)) > 1 and -1 not in clusters:
                        silhouette = silhouette_score(scaled_data, clusters)
                        st.write(f"Silhouette Score: {silhouette:.3f}")
                    
                    # Count clusters (excluding noise)
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    st.write(f"Jumlah cluster: {n_clusters}")
                    st.write(f"Noise points: {(clusters == -1).sum()}")
                    
                    # Visualize DBSCAN clusters
                    if len(selected_features) >= 2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_title('DBSCAN Clustering' if st.session_state.language == 'id' else 'DBSCAN Clustering')
                        plt.colorbar(scatter, ax=ax)
                        st.pyplot(fig)
                    
                    clustering_data['Cluster'] = clusters
                
                elif clustering_method == "Spectral":
                    # Spectral Clustering
                    n_clusters = st.slider(
                        "Jumlah cluster:" if st.session_state.language == 'id' else "Number of clusters:",
                        2, min(10, len(clustering_data) - 1), 3
                    )
                    
                    spectral = SpectralClustering(
                        n_clusters=n_clusters, 
                        random_state=42,
                        affinity='nearest_neighbors'
                    )
                    clusters = spectral.fit_predict(scaled_data)
                    
                    # Calculate silhouette score
                    if len(set(clusters)) > 1:
                        silhouette = silhouette_score(scaled_data, clusters)
                        st.write(f"Silhouette Score: {silhouette:.3f}")
                    
                    # Visualize Spectral clusters
                    if len(selected_features) >= 2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        scatter = ax.scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_title('Spectral Clustering' if st.session_state.language == 'id' else 'Spectral Clustering')
                        plt.colorbar(scatter, ax=ax)
                        st.pyplot(fig)
                    
                    clustering_data['Cluster'] = clusters
                
                # Elbow Method for K-Means
                if clustering_method == "K-Means" and st.checkbox("Tampilkan Elbow Method" if st.session_state.language == 'id' else "Show Elbow Method"):
                    max_k = min(10, len(clustering_data) - 1)
                    inertias = []
                    k_range = range(1, max_k + 1)
                    
                    for k in k_range:
                        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
                        kmeans_temp.fit(scaled_data)
                        inertias.append(kmeans_temp.inertia_)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(k_range, inertias, 'bo-')
                    ax.set_xlabel('Jumlah Cluster (k)' if st.session_state.language == 'id' else 'Number of Clusters (k)')
                    ax.set_ylabel('Inertia')
                    ax.set_title('Elbow Method untuk K-Means' if st.session_state.language == 'id' else 'Elbow Method for K-Means')
                    ax.grid(True)
                    st.pyplot(fig)
                
                # Download clustered data
                if st.button("Download Data dengan Label Cluster" if st.session_state.language == 'id' else "Download Data with Cluster Labels"):
                    csv = clustering_data.to_csv(index=False)
                    st.download_button(
                        label="Download CSV" if st.session_state.language == 'id' else "Download CSV",
                        data=csv,
                        file_name=f"clustered_data_{clustering_method.lower()}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Tidak cukup data untuk clustering setelah menghapus nilai hilang." if st.session_state.language == 'id' else "Not enough data for clustering after removing missing values.")
        else:
            st.warning("Pilih minimal satu fitur untuk analisis clustering." if st.session_state.language == 'id' else "Select at least one feature for clustering analysis.")

# Tab 3: Preprocessing
with tab3:
    st.header("Pemrosesan Data Awal" if st.session_state.language == 'id' else "Data Preprocessing")
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        st.subheader("Pilih Variabel Target" if st.session_state.language == 'id' else "Select Target Variable")
        target_column = st.selectbox("Pilih kolom target untuk diprediksi:" if st.session_state.language == 'id' else "Choose the target column for prediction:", data.columns)
        st.session_state.target_column = target_column
        
        # Determine problem type
        if data[target_column].dtype in ['int64', 'float64']:
            if len(data[target_column].unique()) <= 10:
                problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", ["Classification", "Regression"], index=0)
            else:
                problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", ["Classification", "Regression"], index=1)
        else:
            problem_type = "Classification"
        
        st.session_state.problem_type = problem_type
        
        st.subheader("Atasi Nilai Hilang" if st.session_state.language == 'id' else "Handle Missing Values")
        
        # Display columns with missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            st.write("Kolom yang memiliki nilai hilang:" if st.session_state.language == 'id' else "Columns with missing values:", ", ".join(missing_cols))
            
            for col in missing_cols:
                col_type = "numerical" if data[col].dtype in ['int64', 'float64'] else "categorical"
                
                st.write(f"Handle missing values in '{col}' ({col_type}):")
                
                if col_type == "numerical":
                    method = st.radio(f"Method for {col}:", 
                                     ["Drop rows", "Mean", "Median", "Zero"], 
                                     key=f"missing_{col}")
                    
                    if method == "Drop rows":
                        data = data.dropna(subset=[col])
                    elif method == "Mean":
                        data[col] = data[col].fillna(data[col].mean())
                    elif method == "Median":
                        data[col] = data[col].fillna(data[col].median())
                    elif method == "Zero":
                        data[col] = data[col].fillna(0)
                else:
                    method = st.radio(f"Method for {col}:", 
                                     ["Drop rows", "Mode", "New category"], 
                                     key=f"missing_{col}")
                    
                    if method == "Drop rows":
                        data = data.dropna(subset=[col])
                    elif method == "Mode":
                        data[col] = data[col].fillna(data[col].mode()[0])
                    elif method == "New category":
                        data[col] = data[col].fillna("Unknown")
        else:
            st.success("Tidak ditemukan nilai yang hilang dalam dataset." if st.session_state.language == 'id' else "No missing values found in the dataset.")
        
        # Handle Outliers
        st.subheader("Atasi Data Outlier" if st.session_state.language == 'id' else "Handle Outliers")
        
        # Only for numerical columns
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if numerical_cols:
            handle_outliers = st.checkbox("Deteksi dan tangani outlier" if st.session_state.language == 'id' else "Detect and handle outliers")
            
            if handle_outliers:
                outlier_method = st.radio(
                    "Metode penanganan outlier:" if st.session_state.language == 'id' else "Outlier handling method:",
                    ["IQR (Interquartile Range)", "Z-Score", "Winsorization"]
                )
                
                if outlier_method == "IQR (Interquartile Range)":
                    for col in numerical_cols:
                        # Calculate IQR
                        Q1 = data[col].quantile(0.25)
                        Q3 = data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        
                        # Define bounds
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Count outliers
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                        outlier_count = len(outliers)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            
                            outlier_action = st.radio(
                                f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                                ["Remove", "Cap", "Keep"],
                                key=f"outlier_{col}"
                            )
                            
                            if outlier_action == "Remove":
                                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
                                st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                            elif outlier_action == "Cap":
                                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
                
                elif outlier_method == "Z-Score":
                    z_threshold = st.slider(
                        "Ambang batas Z-Score:" if st.session_state.language == 'id' else "Z-Score threshold:",
                        2.0, 4.0, 3.0, 0.1
                    )
                    
                    for col in numerical_cols:
                        # Calculate Z-scores
                        z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                        
                        # Identify outliers
                        outliers = data[z_scores > z_threshold][col]
                        outlier_count = len(outliers)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            
                            outlier_action = st.radio(
                                f"Tindakan untuk outlier di '{col}':" if st.session_state.language == 'id' else f"Action for outliers in '{col}':",
                                ["Remove", "Cap", "Keep"],
                                key=f"outlier_{col}"
                            )
                            
                            if outlier_action == "Remove":
                                data = data[z_scores <= z_threshold]
                                st.success(f"Outlier dihapus dari kolom '{col}'" if st.session_state.language == 'id' else f"Outliers removed from column '{col}'")
                            elif outlier_action == "Cap":
                                # Calculate bounds
                                mean = data[col].mean()
                                std = data[col].std()
                                lower_bound = mean - z_threshold * std
                                upper_bound = mean + z_threshold * std
                                
                                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                                st.success(f"Outlier di-cap pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers capped in column '{col}'")
                
                elif outlier_method == "Winsorization":
                    percentile = st.slider(
                        "Persentil untuk Winsorization:" if st.session_state.language == 'id' else "Percentile for Winsorization:",
                        90, 99, 95, 1
                    )
                    
                    for col in numerical_cols:
                        lower_bound = np.percentile(data[col], 100 - percentile)
                        upper_bound = np.percentile(data[col], percentile)
                        
                        # Count outliers
                        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
                        outlier_count = len(outliers)
                        
                        if outlier_count > 0:
                            st.write(f"Ditemukan {outlier_count} outlier pada kolom '{col}'" if st.session_state.language == 'id' else f"Found {outlier_count} outliers in column '{col}'")
                            data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                            st.success(f"Outlier di-winsorize pada kolom '{col}'" if st.session_state.language == 'id' else f"Outliers winsorized in column '{col}'")
                
                st.success("Penanganan outlier selesai" if st.session_state.language == 'id' else "Outlier handling completed")
        
        # Handle Duplicate Data
        st.subheader("Penanganan Data Duplikat" if st.session_state.language == 'id' else "Handle Duplicate Data")
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        
        if duplicate_count > 0:
            st.warning(f"Ditemukan {duplicate_count} baris duplikat dalam dataset" if st.session_state.language == 'id' else f"Found {duplicate_count} duplicate rows in the dataset")
            
            # Show preview of duplicate rows
            duplicate_rows = data[data.duplicated(keep=False)].sort_values(by=data.columns.tolist())
            st.write("Preview baris duplikat:" if st.session_state.language == 'id' else "Preview of duplicate rows:")
            st.dataframe(duplicate_rows.head(10))
            
            # Options for handling duplicates
            handle_duplicates = st.checkbox("Hapus data duplikat" if st.session_state.language == 'id' else "Remove duplicate data", value=True)
            
            if handle_duplicates:
                # Store original data count
                original_count = len(data)
                
                # Remove duplicate rows
                data = data.drop_duplicates()
                
                # Calculate removed duplicates
                removed_count = original_count - len(data)
                
                st.success(f"Berhasil menghapus {removed_count} baris duplikat" if st.session_state.language == 'id' else f"Successfully removed {removed_count} duplicate rows")
                st.info(f"Jumlah data: {original_count} → {len(data)}" if st.session_state.language == 'id' else f"Data count: {original_count} → {len(data)}")
        else:
            st.success("Tidak ditemukan data duplikat dalam dataset" if st.session_state.language == 'id' else "No duplicate data found in the dataset")

        # Feature selection
        st.subheader("Rekayasa Data" if st.session_state.language == 'id' else "Data Modification")

        # Penanganan imbalanced dataset untuk klasifikasi
        if problem_type == "Classification":
            st.subheader("Penanganan Imbalanced Dataset" if st.session_state.language == 'id' else "Imbalanced Dataset Handling")
            
            # Tampilkan distribusi kelas
            class_counts = data[target_column].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            class_counts.plot(kind='bar', ax=ax)
            plt.title('Distribusi Kelas' if st.session_state.language == 'id' else 'Class Distribution')
            plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
            plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
            st.pyplot(fig)
            
            # Hitung rasio imbalance
            if len(class_counts) > 1:
                imbalance_ratio = class_counts.max() / class_counts.min()
                st.info(f"Rasio imbalance: {imbalance_ratio:.2f}" if st.session_state.language == 'id' else f"Imbalance ratio: {imbalance_ratio:.2f}")
                
                # Opsi untuk menghilangkan kelas minoritas
                remove_minority = st.checkbox("Hapus kelas minoritas" if st.session_state.language == 'id' else "Remove minority classes", value=False, key="remove_minority_v1")
                
                if remove_minority:
                    # Tampilkan semua kelas dan jumlah sampelnya dalam urutan menaik
                    st.write("Pilih kelas yang ingin dihapus:" if st.session_state.language == 'id' else "Select classes to remove:")
                    
                    # Urutkan kelas berdasarkan jumlah sampel (dari yang terkecil)
                    sorted_classes = class_counts.sort_values().index.tolist()
                    class_to_remove = {}
                    
                    # Buat checkbox untuk setiap kelas
                    col1, col2 = st.columns(2)
                    with col1:
                        for i, cls in enumerate(sorted_classes[:len(sorted_classes)//2 + len(sorted_classes)%2]):
                            class_to_remove[cls] = st.checkbox(
                                f"{cls} ({class_counts[cls]} sampel)" if st.session_state.language == 'id' else f"{cls} ({class_counts[cls]} samples)", 
                                value=False,
                                key=f"remove_class_{cls}"
                            )
                    with col2:
                        for i, cls in enumerate(sorted_classes[len(sorted_classes)//2 + len(sorted_classes)%2:]):
                            class_to_remove[cls] = st.checkbox(
                                f"{cls} ({class_counts[cls]} sampel)" if st.session_state.language == 'id' else f"{cls} ({class_counts[cls]} samples)", 
                                value=False,
                                key=f"remove_class_{cls}"
                            )
                    
                    # Identifikasi kelas yang akan dihapus
                    classes_to_remove = [cls for cls, remove in class_to_remove.items() if remove]
                    
                    if classes_to_remove:
                        # Konfirmasi penghapusan
                        classes_str = ", ".join([f"'{cls}'" for cls in classes_to_remove])
                        samples_count = sum([class_counts[cls] for cls in classes_to_remove])
                        
                        st.warning(
                            f"Kelas {classes_str} dengan total {samples_count} sampel akan dihapus" if st.session_state.language == 'id' 
                            else f"Classes {classes_str} with total {samples_count} samples will be removed"
                        )
                        
                        confirm_removal = st.checkbox("Konfirmasi penghapusan" if st.session_state.language == 'id' else "Confirm removal", key="confirm_removal_v1")
                        
                        if confirm_removal:
                            # Simpan data asli
                            original_data = data.copy()
                            
                            # Hapus kelas yang dipilih
                            data = data[~data[target_column].isin(classes_to_remove)]
                            
                            # Tampilkan distribusi kelas setelah penghapusan
                            new_class_counts = data[target_column].value_counts()
                            fig, ax = plt.subplots(figsize=(10, 4))
                            new_class_counts.plot(kind='bar', ax=ax)
                            plt.title('Distribusi Kelas Setelah Penghapusan' if st.session_state.language == 'id' else 'Class Distribution After Removal')
                            plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
                            plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
                            st.pyplot(fig)
                            
                            st.success(
                                f"Kelas {classes_str} berhasil dihapus" if st.session_state.language == 'id' 
                                else f"Classes {classes_str} successfully removed"
                            )
                            
                            # Tampilkan perbandingan jumlah sampel
                            comparison_df = pd.DataFrame({
                                'Sebelum' if st.session_state.language == 'id' else 'Before': class_counts,
                                'Sesudah' if st.session_state.language == 'id' else 'After': new_class_counts
                            })
                            st.dataframe(comparison_df)
                

        # Encoding fitur kategorikal
        categorical_cols = [col for col in data.columns if col in st.session_state.categorical_columns and col != target_column]
        if categorical_cols:
            st.subheader("Lakukan Encoding" if st.session_state.language == 'id' else "Encode Categorical Features")
            
            # Tampilkan fitur kategorikal yang akan diencode
            st.write("**Fitur kategorikal yang akan diubah:**" if st.session_state.language == 'id' else "**Categorical features to be transformed:**")
            for col in categorical_cols:
                unique_values = data[col].nunique()
                st.write(f"- **{col}**: {unique_values} nilai unik" if st.session_state.language == 'id' else f"- **{col}**: {unique_values} unique values")
            
            encoding_method = st.radio("Encoding method:", ["Label Encoding", "One-Hot Encoding"])
            if encoding_method == "Label Encoding":
                encoders = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    data[col] = le.fit_transform(data[col].astype(str))
                    encoders[col] = le
                st.session_state.encoders = encoders
                st.success("Encoding label diaplikasikan pada fitur kategorikal." if st.session_state.language == 'id' else "Label encoding applied to categorical features.")
            else:  # One-Hot Encoding
                # Simpan target column
                target_series = data[target_column].copy()
                # One-hot encode data
                data = pd.get_dummies(data.drop(columns=[target_column]), columns=categorical_cols, drop_first=True)
                # Kembalikan target column
                data[target_column] = target_series
                st.success("One-hot encoding diaplikasikan pada fitur kategorikal." if st.session_state.language == 'id' else "One-hot encoding applied to categorical features.")            
            
            # Tampilkan deskripsi fitur setelah encoding
            st.subheader("Deskripsi Fitur Setelah Encoding" if st.session_state.language == 'id' else "Feature Description After Encoding")
            
            # Buat dataframe deskripsi fitur
            feature_desc = pd.DataFrame({
                'Nama Fitur' if st.session_state.language == 'id' else 'Feature Name': data.columns,
                'Tipe Data' if st.session_state.language == 'id' else 'Data Type': data.dtypes.astype(str),
                'Jumlah Non-Null' if st.session_state.language == 'id' else 'Non-Null Count': data.count(),
                'Jumlah Nilai Unik' if st.session_state.language == 'id' else 'Unique Values': data.nunique(),
                'Nilai yang Hilang' if st.session_state.language == 'id' else 'Missing Values': data.isnull().sum()
            })
            
            # Tampilkan sebagai tabel
            st.dataframe(feature_desc)
            
            # Tampilkan ringkasan statistik
            st.write("**Ringkasan Statistik:**" if st.session_state.language == 'id' else "**Statistical Summary:**")
            st.write(f"- Total fitur: {len(data.columns)}")
            st.write(f"- Total baris: {len(data)}")
            st.write(f"- Fitur numerik: {len(data.select_dtypes(include=[np.number]).columns)}")
            st.write(f"- Fitur kategorikal: {len(data.select_dtypes(include=['object', 'category']).columns)}")
 
            # Tampilkan distribusi kelas
            class_counts = data[target_column].value_counts()
            fig, ax = plt.subplots(figsize=(10, 4))
            class_counts.plot(kind='bar', ax=ax)
            plt.title('Distribusi Kelas' if st.session_state.language == 'id' else 'Class Distribution')
            plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
            plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
            st.pyplot(fig)
            
        # Update all_columns setelah encoding
        all_columns = [col for col in data.columns if col != target_column]

        # Train-test split
        st.subheader("Lakukan Train-Test Split" if st.session_state.language == 'id' else "Train-Test Split")

        test_size = st.slider("Ukuran set pengujian (persen):" if st.session_state.language == 'id' else "Test set size (%):", 10, 50, 20) / 100
        random_state = st.number_input("Status acak:" if st.session_state.language == 'id' else "Random state:", 0, 100, 42)

        # Prepare data for modeling dengan semua fitur awal
        X = data[all_columns]
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Tambahkan normalisasi setelah train test split
        st.subheader("Normalisasi Fitur" if st.session_state.language == 'id' else "Feature Normalization")

        normalization_method = st.selectbox(
            "Metode normalisasi:" if st.session_state.language == 'id' else "Normalization method:",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        )

        if normalization_method != "None":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if normalization_method == "StandardScaler":
                scaler = StandardScaler()
            elif normalization_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif normalization_method == "RobustScaler":
                scaler = RobustScaler()
            
            # Fit dan transform hanya pada fitur numerik
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
                
                st.success(f"Normalisasi {normalization_method} berhasil diterapkan")
                st.info(f"Fitur numerik yang dinormalisasi: {len(numeric_cols)} fitur")
        
        # Handle class imbalance for training data (classification only)
        if st.session_state.problem_type == "Classification" and IMB_AVAILABLE:
            st.subheader("Penanganan Ketidakseimbangan Dataset" if st.session_state.language == 'id' else "Handle Class Imbalance")
            
            # Check for class imbalance
            train_counts = pd.Series(y_train).value_counts()
            imbalance_ratio = train_counts.max() / train_counts.min()
            
            if imbalance_ratio > 1.5:  # Only show if there's significant imbalance
                st.warning(f"Terdeteksi ketidakseimbangan kelas dengan rasio {imbalance_ratio:.2f}" if st.session_state.language == 'id' else f"Detected class imbalance with ratio {imbalance_ratio:.2f}")
                
                # Imbalance handling options
                balance_method = st.selectbox(
                    "Pilih metode penyeimbangan:" if st.session_state.language == 'id' else "Select balancing method:",
                    ["Tidak ada" if st.session_state.language == 'id' else "None",
                     "Random Over Sampling",
                     "Random Under Sampling", 
                     "SMOTE",
                     "SMOTEENN",
                     "SMOTETomek"]
                )
                
                if balance_method != "Tidak ada" and balance_method != "None":
                    with st.spinner("Menerapkan penyeimbangan dataset..." if st.session_state.language == 'id' else "Applying dataset balancing..."):
                        try:
                            if balance_method == "Random Over Sampling":
                                ros = RandomOverSampler(random_state=random_state)
                                X_train_bal, y_train_bal = ros.fit_resample(X_train, y_train)
                            elif balance_method == "Random Under Sampling":
                                rus = RandomUnderSampler(random_state=random_state)
                                X_train_bal, y_train_bal = rus.fit_resample(X_train, y_train)
                            elif balance_method == "SMOTE":
                                smote = SMOTE(random_state=random_state)
                                X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
                            elif balance_method == "SMOTEENN":
                                smoteenn = SMOTEENN(random_state=random_state)
                                X_train_bal, y_train_bal = smoteenn.fit_resample(X_train, y_train)
                            elif balance_method == "SMOTETomek":
                                smotetomek = SMOTETomek(random_state=random_state)
                                X_train_bal, y_train_bal = smotetomek.fit_resample(X_train, y_train)
                            
                            # Show before/after comparison
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Distribusi Sebelum:**" if st.session_state.language == 'id' else "**Before Distribution:**")
                                st.write(pd.Series(y_train).value_counts().to_dict())
                            with col2:
                                st.write("**Distribusi Setelah:**" if st.session_state.language == 'id' else "**After Distribution:**")
                                st.write(pd.Series(y_train_bal).value_counts().to_dict())
                            
                            # Use balanced data
                            X_train, y_train = X_train_bal, y_train_bal
                            st.success(f"Dataset berhasil diseimbangkan! Ukuran training: {len(y_train)} sampel" if st.session_state.language == 'id' else f"Dataset successfully balanced! Training size: {len(y_train)} samples")
                            
                        except Exception as e:
                            st.error(f"Error saat penyeimbangan: {e}" if st.session_state.language == 'id' else f"Error during balancing: {e}")
                            st.info("Menggunakan data training asli..." if st.session_state.language == 'id' else "Using original training data...")
            else:
                st.info("Dataset seimbang, tidak perlu penanganan khusus" if st.session_state.language == 'id' else "Dataset is balanced, no special handling needed")

        # Feature selection
        st.subheader("Seleksi Fitur" if st.session_state.language == 'id' else "Feature Selection")
        
        # Pilih algoritma seleksi fitur
        feature_selection_method = st.selectbox(
            "Metode seleksi fitur:" if st.session_state.language == 'id' else "Feature selection method:",
            [
                "Manual",
                "Mutual Information", 
                "Pearson Correlation",
                "Recursive Feature Elimination (RFE)",
                "LASSO",
                "Gradient Boosting Importance",
                "Random Forest Importance",
                "Ensemble Feature Selection",
                "Multi-Stage Feature Selection",
                "Genetic Algorithm (PyGAD)"
            ]
        )

        # Gunakan data training untuk seleksi fitur
        X_train_for_selection = X_train.copy()
        y_train_for_selection = y_train.copy()
        
        # Simpan nama kolom asli untuk referensi
        all_columns_for_selection = X_train_for_selection.columns.tolist()
        selected_features = all_columns_for_selection

        # Setelah feature selection selesai, terapkan pada X_train dan X_test
        final_selected_features = selected_features
        X_train_final = X_train[final_selected_features]
        X_test_final = X_test[final_selected_features]
        
        # Update session state
        st.session_state.X_train = X_train_final
        st.session_state.X_test = X_test_final
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.processed_data = data
        
        st.success(f"Data training memiliki {X_train_final.shape[0]} sampel dan {X_train_final.shape[1]} fitur setelah seleksi" if st.session_state.language == 'id' else f"Training data has {X_train_final.shape[0]} samples and {X_train_final.shape[1]} features after selection")
        st.success(f"Data testing memiliki {X_test_final.shape[0]} sampel dan {X_test_final.shape[1]} fitur" if st.session_state.language == 'id' else f"Testing data has {X_test_final.shape[0]} samples and {X_test_final.shape[1]} features")

        # Display processed data
        st.subheader("Tampilkan Data Terproses" if st.session_state.language == 'id' else "Processed Data Preview")
        st.dataframe(X_train_final.head())

        # Update session state setelah encoding/scaling
        st.session_state.X_train = X_train_final
        st.session_state.X_test = X_test_final
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test

        # Display class distribution table for classification problems
        if st.session_state.problem_type == "Classification":
            st.subheader("Distribusi Label Target" if st.session_state.language == 'id' else "Target Label Distribution")
            
            # Create distribution table
            train_counts = pd.Series(y_train).value_counts().sort_index()
            test_counts = pd.Series(y_test).value_counts().sort_index()
            
            distribution_df = pd.DataFrame({
                'Label': train_counts.index,
                'Jumlah Data Training': train_counts.values,
                'Jumlah Data Testing': test_counts.values,
                'Total': train_counts.values + test_counts.values
            })
            
            # Add percentages
            total_samples = len(y_train) + len(y_test)
            distribution_df['Persentase Training (%)'] = (distribution_df['Jumlah Data Training'] / len(y_train) * 100).round(2)
            distribution_df['Persentase Testing (%)'] = (distribution_df['Jumlah Data Testing'] / len(y_test) * 100).round(2)
            distribution_df['Persentase Total (%)'] = (distribution_df['Total'] / total_samples * 100).round(2)
            
            st.dataframe(distribution_df)
            
            # Display summary statistics
            st.write(f"**Total sampel:** {total_samples}")
            st.write(f"**Training set:** {len(y_train)} sampel ({len(y_train)/total_samples*100:.1f}%)")
            st.write(f"**Testing set:** {len(y_test)} sampel ({len(y_test)/total_samples*100:.1f}%)")
            
            # Display class imbalance information
            if len(train_counts) > 1:
                imbalance_ratio = train_counts.max() / train_counts.min()
                st.write(f"**Rasio ketidakseimbangan kelas (training):** {imbalance_ratio:.2f}")

        if feature_selection_method == "Manual":
            selected_features = st.multiselect(
                "Pilih fitur untuk model:" if st.session_state.language == 'id' else "Select features to include in the model:",
                all_columns,
                default=all_columns
            )
        
        elif feature_selection_method == "Genetic Algorithm (PyGAD)":
            st.subheader("Genetic Algorithm Feature Selection (PyGAD)" if st.session_state.language == 'id' else "Genetic Algorithm Feature Selection (PyGAD)")
            st.info("Menggunakan algoritma genetik PyGAD untuk seleksi fitur otomatis" if st.session_state.language == 'id' else "Using PyGAD genetic algorithm for automatic feature selection")
            
            # Parameters for PyGAD
            col1, col2 = st.columns(2)
            with col1:
                ga_population_size = st.number_input(
                    "Ukuran populasi:" if st.session_state.language == 'id' else "Population size:",
                    min_value=10, max_value=200, value=50, step=5,
                    help="Jumlah kromosom dalam populasi" if st.session_state.language == 'id' else "Number of chromosomes in population"
                )
                ga_generations = st.number_input(
                    "Jumlah generasi:" if st.session_state.language == 'id' else "Number of generations:",
                    min_value=10, max_value=500, value=100, step=10,
                    help="Maksimum iterasi algoritma genetik" if st.session_state.language == 'id' else "Maximum genetic algorithm iterations"
                )
                ga_mutation_rate = st.slider(
                    "Tingkat mutasi:" if st.session_state.language == 'id' else "Mutation rate:",
                    0.01, 0.3, 0.1, 0.01,
                    help="Probabilitas mutasi gen" if st.session_state.language == 'id' else "Gene mutation probability"
                )
            
            with col2:
                ga_crossover_rate = st.slider(
                    "Tingkat crossover:" if st.session_state.language == 'id' else "Crossover rate:",
                    0.1, 0.9, 0.7, 0.1,
                    help="Probabilitas crossover antar kromosom" if st.session_state.language == 'id' else "Crossover probability between chromosomes"
                )
                ga_elite_size = st.number_input(
                    "Ukuran elit:" if st.session_state.language == 'id' else "Elite size:",
                    min_value=1, max_value=20, value=5, step=1,
                    help="Jumlah kromosom terbaik yang dilestarikan" if st.session_state.language == 'id' else "Number of best chromosomes to preserve"
                )
                target_features = st.number_input(
                    "Target jumlah fitur:" if st.session_state.language == 'id' else "Target number of features:",
                    min_value=1, max_value=len(all_columns), value=min(10, len(all_columns)), step=1
                )
            
            # Prepare data for PyGAD
            X_ga = data[all_columns].copy()
            y_ga = data[target_column].copy()
            
            # Handle categorical variables
            for col in X_ga.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_ga[col] = le.fit_transform(X_ga[col].astype(str))
            
            # Standardize features
            scaler = StandardScaler()
            X_ga_scaled = scaler.fit_transform(X_ga)
            
            if st.button("Jalankan Algoritma Genetik" if st.session_state.language == 'id' else "Run Genetic Algorithm"):
                try:
                    import pygad
                    
                    # Define fitness function
                    def fitness_func(ga_instance, solution, solution_idx):
                        # Get selected features based on binary solution
                        selected_indices = np.where(solution == 1)[0]
                        
                        if len(selected_indices) == 0:
                            return 0.0
                        
                        # Limit to target number of features
                        if len(selected_indices) > target_features:
                            # Select top features based on importance
                            if problem_type == "Regression":
                                from sklearn.ensemble import RandomForestRegressor
                                temp_model = RandomForestRegressor(n_estimators=50, random_state=42)
                                temp_model.fit(X_ga_scaled, y_ga)
                                importances = temp_model.feature_importances_
                                top_indices = np.argsort(importances)[-target_features:]
                                selected_indices = np.intersect1d(selected_indices, top_indices)
                            else:
                                from sklearn.ensemble import RandomForestClassifier
                                temp_model = RandomForestClassifier(n_estimators=50, random_state=42)
                                temp_model.fit(X_ga_scaled, y_ga)
                                importances = temp_model.feature_importances_
                                top_indices = np.argsort(importances)[-target_features:]
                                selected_indices = np.intersect1d(selected_indices, top_indices)
                        
                        if len(selected_indices) == 0:
                            return 0.0
                        
                        # Get selected features
                        X_selected = X_ga_scaled[:, selected_indices]
                        
                        # Use cross-validation to evaluate fitness
                        if problem_type == "Regression":
                            from sklearn.ensemble import RandomForestRegressor
                            model = RandomForestRegressor(n_estimators=50, random_state=42)
                            scores = cross_val_score(model, X_selected, y_ga, cv=3, 
                                                   scoring='neg_mean_squared_error')
                            fitness = -np.mean(scores)  # Negative MSE, so higher is better
                        else:
                            from sklearn.ensemble import RandomForestClassifier
                            model = RandomForestClassifier(n_estimators=50, random_state=42)
                            scores = cross_val_score(model, X_selected, y_ga, cv=3, 
                                                   scoring='accuracy')
                            fitness = np.mean(scores)
                        
                        # Penalty for too many features
                        penalty = abs(len(selected_indices) - target_features) * 0.01
                        return max(0, fitness - penalty)
                    
                    # Initialize PyGAD
                    gene_space = [0, 1]  # Binary genes
                    
                    ga_instance = pygad.GA(
                        num_generations=ga_generations,
                        num_parents_mating=ga_population_size // 2,
                        fitness_func=fitness_func,
                        sol_per_pop=ga_population_size,
                        num_genes=len(all_columns),
                        gene_space=gene_space,
                        init_range_low=0,
                        init_range_high=2,
                        parent_selection_type="tournament",
                        K_tournament=3,
                        crossover_type="single_point",
                        crossover_probability=ga_crossover_rate,
                        mutation_type="random",
                        mutation_probability=ga_mutation_rate,
                        keep_elitism=ga_elite_size,
                        random_seed=42,
                        suppress_warnings=True
                    )
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def on_generation(ga_instance):
                        generation = ga_instance.generations_completed
                        max_generations = ga_instance.num_generations
                        progress = generation / max_generations
                        progress_bar.progress(progress)
                        
                        best_fitness = ga_instance.best_solution()[1]
                        status_text.text(
                            f"Generasi {generation}/{max_generations} - Fitness terbaik: {best_fitness:.4f}" 
                            if st.session_state.language == 'id' 
                            else f"Generation {generation}/{max_generations} - Best fitness: {best_fitness:.4f}"
                        )
                    
                    ga_instance.on_generation = on_generation
                    
                    # Run genetic algorithm
                    with st.spinner("Menjalankan algoritma genetik..." if st.session_state.language == 'id' else "Running genetic algorithm..."):
                        ga_instance.run()
                    
                    # Get results
                    solution, solution_fitness, solution_idx = ga_instance.best_solution()
                    selected_indices = np.where(solution == 1)[0]
                    selected_features = [all_columns[i] for i in selected_indices]
                    
                    # Display results
                    st.success(f"Algoritma genetik selesai!" if st.session_state.language == 'id' else "Genetic algorithm completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Jumlah fitur terpilih" if st.session_state.language == 'id' else "Selected features count", 
                                len(selected_features))
                        st.metric("Fitness terbaik" if st.session_state.language == 'id' else "Best fitness", 
                                f"{solution_fitness:.4f}")
                    
                    with col2:
                        st.metric("Total fitur" if st.session_state.language == 'id' else "Total features", 
                                len(all_columns))
                        st.metric("Persentase fitur terpilih" if st.session_state.language == 'id' else "Feature selection ratio", 
                                f"{len(selected_features)/len(all_columns)*100:.1f}%")
                    
                    # Display selected features
                    st.write("**Fitur yang dipilih algoritma genetik:**" if st.session_state.language == 'id' else "**Features selected by genetic algorithm:**")
                    st.write(selected_features)
                    
                    # Feature importance visualization
                    if len(selected_features) > 0:
                        st.write("**Visualisasi seleksi fitur:**" if st.session_state.language == 'id' else "**Feature selection visualization:**")
                        
                        # Create a dataframe with selection status
                        selection_df = pd.DataFrame({
                            'Feature': all_columns,
                            'Selected': [1 if i in selected_indices else 0 for i in range(len(all_columns))]
                        })
                        
                        # Plot selection status
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Bar plot of selected vs not selected
                        selection_counts = selection_df['Selected'].value_counts()
                        colors = ['#ff9999', '#66b3ff']
                        ax1.pie(selection_counts.values, labels=['Not Selected', 'Selected'], 
                               colors=colors, autopct='%1.1f%%', startangle=90)
                        ax1.set_title('Distribusi Seleksi Fitur' if st.session_state.language == 'id' else 'Feature Selection Distribution')
                        
                        # Fitness evolution plot
                        ax2.plot(ga_instance.best_solutions_fitness, 'b-', linewidth=2)
                        ax2.set_xlabel('Generasi' if st.session_state.language == 'id' else 'Generation')
                        ax2.set_ylabel('Fitness' if st.session_state.language == 'id' else 'Fitness')
                        ax2.set_title('Evolusi Fitness Algoritma Genetik' if st.session_state.language == 'id' else 'Genetic Algorithm Fitness Evolution')
                        ax2.grid(True, alpha=0.3)
                        
                        st.pyplot(fig)
                    
                    # Clean up
                    progress_bar.empty()
                    status_text.empty()
                    
                except ImportError:
                    st.error("PyGAD tidak terinstal. Silakan install dengan: pip install pygad" if st.session_state.language == 'id' else 
                            "PyGAD not installed. Please install with: pip install pygad")
                    selected_features = all_columns

        elif feature_selection_method == "Mutual Information":
                    if problem_type == "Regression":
                        mi = mutual_info_regression(data[all_columns], data[target_column])
                    else:
                        mi = mutual_info_classif(data[all_columns], data[target_column])
                    mi_df = pd.DataFrame({"Feature": all_columns, "Mutual Information": mi})
                    mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                    
                    # Tambahan: Slider untuk ambang batas minimum
                    min_threshold = st.slider("Ambang batas minimum Mutual Information:", 0.0, 1.0, 0.25, 0.01, 
                                             help="Fitur dengan nilai Mutual Information di bawah ambang ini akan dihilangkan")
                    
                    # Filter berdasarkan ambang batas
                    filtered_df = mi_df[mi_df["Mutual Information"] >= min_threshold]
                    
                    st.dataframe(mi_df)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = mi_df.head(30)  # Tampilkan 15 fitur teratas
                    ax.barh(top_features['Feature'], top_features['Mutual Information'])
                    ax.set_xlabel('Mutual Information Score')
                    ax.set_title('Top 30 Features by Mutual Information')
                    ax.invert_yaxis()  # Fitur dengan score tertinggi di atas
                    st.pyplot(fig)
                    
                    # Pilih fitur berdasarkan ambang batas atau top N
                    use_threshold = st.checkbox("Gunakan ambang batas", value=True)
                    if use_threshold:
                        selected_features = filtered_df["Feature"].tolist()
                        st.info(f"{len(selected_features)} fitur terpilih dengan ambang batas {min_threshold}")
                    else:
                        top_n = st.slider("Top N fitur:", 1, len(all_columns), min(10, len(all_columns)))
                        selected_features = mi_df.head(top_n)["Feature"].tolist()
        elif feature_selection_method == "Pearson Correlation":
            numeric_columns = data[all_columns].select_dtypes(include=[np.number]).columns.tolist()
            if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                st.error("Target kolom harus numerik untuk Pearson Correlation.")
                corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
            else:
                corr = data[numeric_columns].corrwith(data[target_column]).abs()
            corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
            corr_df = corr_df.sort_values("Correlation", ascending=False)
            st.dataframe(corr_df)
            top_n = st.slider("Top N features:", 1, len(all_columns), min(10, len(all_columns)))
            selected_features = corr_df.head(top_n)["Feature"].tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = corr_df.head(30)
            ax.barh(top_features['Feature'], top_features['Correlation'])
            ax.set_xlabel('Absolute Correlation')
            ax.set_title('Top 30 Features by Pearson Correlation')
            ax.invert_yaxis()
            st.pyplot(fig)
        elif feature_selection_method == "Recursive Feature Elimination (RFE)":
            from sklearn.feature_selection import RFE
            X_rfe = data[all_columns].copy()
            for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
            if problem_type == "Regression":
                estimator = LinearRegression()
            else:
                estimator = LogisticRegression(max_iter=500)
            rfe = RFE(estimator, n_features_to_select=min(10, len(all_columns)))
            rfe.fit(X_rfe, data[target_column])
            rfe_df = pd.DataFrame({"Feature": all_columns, "Selected": rfe.support_})
            st.dataframe(rfe_df)
            selected_features = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
            selected_count = rfe_df['Selected'].sum()
            fig, ax = plt.subplots(figsize=(8, 6))
            value_counts = rfe_df['Selected'].value_counts()
            value_counts.plot(kind='bar', ax=ax)
             # Fix: dynamically set labels based on actual values
            labels = ['Not Selected' if val == False else 'Selected' for val in value_counts.index]
            ax.set_xticklabels(labels, rotation=0)
            ax.set_ylabel('Count')
            ax.set_title(f'RFE Selection Results ({selected_count} features selected)')
            st.pyplot(fig)

        elif feature_selection_method == "LASSO":
            from sklearn.linear_model import Lasso, LogisticRegression
            if problem_type == "Regression":
                lasso = Lasso(alpha=0.01, max_iter=1000)
            else:
                lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
            lasso.fit(data[all_columns], data[target_column])
            coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
            if coef.ndim > 1:
                coef = coef[0]
            lasso_df = pd.DataFrame({"Feature": all_columns, "Coefficient": coef})
            lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
            st.dataframe(lasso_df)
            selected_features = lasso_df["Feature"].tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = lasso_df.head(15)
            ax.barh(top_features['Feature'], abs(top_features['Coefficient']))
            ax.set_xlabel('Absolute Coefficient Value')
            ax.set_title('Top 15 Features by LASSO Coefficient')
            ax.invert_yaxis()
            st.pyplot(fig)
        elif feature_selection_method == "Gradient Boosting Importance":
            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
            if problem_type == "Regression":
                model = GradientBoostingRegressor(random_state=42)
            else:
                model = GradientBoostingClassifier(random_state=42)
            model.fit(data[all_columns], data[target_column])
            importances = model.feature_importances_
            gb_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
            gb_df = gb_df.sort_values("Importance", ascending=False)
            st.dataframe(gb_df)
            top_n = st.slider("Top N features:", 1, len(all_columns), min(10, len(all_columns)))
            selected_features = gb_df.head(top_n)["Feature"].tolist()
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = gb_df.head(30)
            ax.barh(top_features['Feature'], top_features['Importance'])
            ax.set_xlabel('Importance Score')
            ax.set_title('Top 30 Features by Gradient Boosting Importance')
            ax.invert_yaxis()
            st.pyplot(fig)
        elif feature_selection_method == "Random Forest Importance":
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    
                    # Tambahan: Input untuk jumlah pohon
                    n_estimators = st.number_input("Jumlah pohon Random Forest:", min_value=10, max_value=1000, value=100, step=10,
                                                   help="Semakin banyak pohon, semakin akurat tetapi lebih lambat")
                    
                    if problem_type == "Regression":
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    
                    model.fit(data[all_columns], data[target_column])
                    importances = model.feature_importances_
                    rf_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                    rf_df = rf_df.sort_values("Importance", ascending=False)
                    
                    # Tambahan: Slider untuk ambang batas minimum
                    min_threshold = st.slider("Ambang batas minimum Importance:", 0.0, 1.0, 0.2, 0.01,
                                             help="Fitur dengan nilai Importance di bawah ambang ini akan dihilangkan")
                    
                    # Filter berdasarkan ambang batas
                    filtered_df = rf_df[rf_df["Importance"] >= min_threshold]
                    
                    st.dataframe(rf_df)
                    
                    # Pilih fitur berdasarkan ambang batas atau top N
                    use_threshold = st.checkbox("Gunakan ambang batas", value=True)
                    if use_threshold:
                        selected_features = filtered_df["Feature"].tolist()
                        st.info(f"{len(selected_features)} fitur terpilih dengan ambang batas {min_threshold}")
                    else:
                        top_n = st.slider("Top N fitur:", 1, len(all_columns), min(10, len(all_columns)))
                        selected_features = rf_df.head(top_n)["Feature"].tolist()

                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_features = rf_df.head(30)
                    ax.barh(top_features['Feature'], top_features['Importance'])
                    ax.set_xlabel('Importance Score')
                    ax.set_title('Top 30 Features by Random Forest Importance')
                    ax.invert_yaxis()
                    st.pyplot(fig)

        elif feature_selection_method == "Ensemble Feature Selection":
            st.info("Pilih dua metode seleksi fitur untuk digabungkan." if st.session_state.language == 'id' else "Select two feature selection methods to combine.")
            method1 = st.selectbox("Metode pertama:" if st.session_state.language == 'id' else "First method:", [
                "Mutual Information",
                "Pearson Correlation",
                "Recursive Feature Elimination (RFE)",
                "LASSO",
                "Gradient Boosting Importance",
                "Random Forest Importance"
            ], key="ensemble_method1")
            method2 = st.selectbox("Metode kedua:" if st.session_state.language == 'id' else "Second method:", [
                "Random Forest Importance",
                "Mutual Information",
                "Pearson Correlation",
                "Recursive Feature Elimination (RFE)",
                "LASSO",
                "Gradient Boosting Importance"
                
            ], key="ensemble_method2")

            combine_type = st.radio("Gabungkan hasil dengan:" if st.session_state.language == 'id' else "Combine results with:", ["Intersection", "Union"], index=0)

            def get_features_by_method(method):
                if method == "Mutual Information":
                    if problem_type == "Regression":
                        mi = mutual_info_regression(data[all_columns], data[target_column])
                    else:
                        mi = mutual_info_classif(data[all_columns], data[target_column])
                    mi_df = pd.DataFrame({"Feature": all_columns, "Mutual Information": mi})
                    mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                    
                    # Tambahan: Ambang batas untuk ensemble
                    min_threshold = st.slider(f"Ambang batas minimum {method}:", 0.0, 1.0, 0.25, 0.01, 
                                            key=f"threshold_{method}")
                    filtered_df = mi_df[mi_df["Mutual Information"] >= min_threshold]
                    return set(filtered_df["Feature"].tolist())
                elif method == "Pearson Correlation":
                    corr = data[all_columns].corrwith(data[target_column]).abs()
                    corr_df = pd.DataFrame({"Feature": all_columns, "Correlation": corr})
                    corr_df = corr_df.sort_values("Correlation", ascending=False)
                    top_n = st.slider(f"Top N fitur ({method}):", 1, len(all_columns), min(10, len(all_columns)), key=f"topn_{method}")
                    return set(corr_df.head(top_n)["Feature"].tolist())
                elif method == "Recursive Feature Elimination (RFE)":
                    from sklearn.feature_selection import RFE
                    from sklearn.linear_model import LinearRegression, LogisticRegression  # Tambahkan import ini
                    if problem_type == "Regression":
                        estimator = LinearRegression()
                    else:
                        estimator = LogisticRegression(max_iter=500)
                    rfe = RFE(estimator, n_features_to_select=min(10, len(all_columns)))
                    rfe.fit(data[all_columns], data[target_column])
                    rfe_df = pd.DataFrame({"Feature": all_columns, "Selected": rfe.support_})
                    return set(rfe_df[rfe_df["Selected"]]["Feature"].tolist())
                elif method == "LASSO":
                    from sklearn.linear_model import Lasso, LogisticRegression
                    if problem_type == "Regression":
                        lasso = Lasso(alpha=0.01, max_iter=1000)
                    else:
                        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500)
                    lasso.fit(data[all_columns], data[target_column])
                    coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                    if coef.ndim > 1:
                        coef = coef[0]
                    lasso_df = pd.DataFrame({"Feature": all_columns, "Coefficient": coef})
                    lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                    return set(lasso_df["Feature"].tolist())
                elif method == "Gradient Boosting Importance":
                    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                    if problem_type == "Regression":
                        model = GradientBoostingRegressor(random_state=42)
                    else:
                        model = GradientBoostingClassifier(random_state=42)
                    model.fit(data[all_columns], data[target_column])
                    importances = model.feature_importances_
                    gb_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                    gb_df = gb_df.sort_values("Importance", ascending=False)
                    top_n = st.slider(f"Top N fitur ({method}):", 1, len(all_columns), min(10, len(all_columns)), key=f"topn_{method}")
                    return set(gb_df.head(top_n)["Feature"].tolist())
                elif method == "Random Forest Importance":
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    
                    # Tambahan: Jumlah pohon untuk ensemble
                    n_estimators = st.number_input(f"Jumlah pohon {method}:", 10, 1000, 100, 10,
                                                key=f"trees_{method}")
                    
                    if problem_type == "Regression":
                        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
                    else:
                        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
                    
                    model.fit(data[all_columns], data[target_column])
                    importances = model.feature_importances_
                    rf_df = pd.DataFrame({"Feature": all_columns, "Importance": importances})
                    rf_df = rf_df.sort_values("Importance", ascending=False)
                    
                    # Tambahan: Ambang batas untuk ensemble
                    min_threshold = st.slider(f"Ambang batas minimum {method}:", 0.0, 1.0, 0.2, 0.01,
                                            key=f"threshold_{method}")
                    filtered_df = rf_df[rf_df["Importance"] >= min_threshold]
                    return set(filtered_df["Feature"].tolist())
                else:
                    return set(all_columns)

            features1 = get_features_by_method(method1)
            features2 = get_features_by_method(method2)

            if combine_type == "Intersection":
                selected_features = list(features1 & features2)
            else:
                selected_features = list(features1 | features2)

            st.write(f"Fitur hasil gabungan: {selected_features}" if st.session_state.language == 'id' else f"Combined features: {selected_features}")

        elif feature_selection_method == "Multi-Stage Feature Selection":
            st.subheader("Multi-Stage Feature Selection" if st.session_state.language == 'id' else "Multi-Stage Feature Selection")
            st.info("Metode ini menggunakan pendekatan 3 tahap: Information Gain → Random Forest Feature Importance → RFE" if st.session_state.language == 'id' else 
                   "This method uses a 3-stage approach: Information Gain → Random Forest Feature Importance → RFE")
            
            from sklearn.feature_selection import RFE, SelectKBest
            from sklearn.ensemble import RandomForestClassifier
            
            # Persiapkan data untuk feature selection
            X_fs = data[all_columns].copy()
            for col in X_fs.select_dtypes(include=['object', 'category']).columns:
                le = LabelEncoder()
                X_fs[col] = le.fit_transform(X_fs[col].astype(str))
            
            # Tampilkan parameter untuk setiap tahap
            st.write("Tahap 1: Information Gain" if st.session_state.language == 'id' else "Stage 1: Information Gain")
            ig_percent = st.slider("Persentase fitur yang dipertahankan setelah Information Gain (%)" if st.session_state.language == 'id' else 
                                  "Percentage of features to keep after Information Gain (%)", 10, 90, 40)
            
            st.write("Tahap 2: Random Forest Feature Importance" if st.session_state.language == 'id' else "Stage 2: Random Forest Feature Importance")
            rf_percent = st.slider("Persentase fitur yang dipertahankan setelah Random Forest (%)" if st.session_state.language == 'id' else 
                                  "Percentage of features to keep after Random Forest (%)", 10, 90, 50)
            
            st.write("Tahap 3: Recursive Feature Elimination" if st.session_state.language == 'id' else "Stage 3: Recursive Feature Elimination")
            final_features = st.slider("Jumlah fitur akhir" if st.session_state.language == 'id' else "Final number of features", 
                                      1, min(20, len(all_columns)), min(10, len(all_columns)))
            
            # Tahap 1: Seleksi Fitur dengan Information Gain (SelectKBest + mutual_info_classif)
            n_features_after_ig = max(1, int(X_fs.shape[1] * ig_percent / 100))
            
            if problem_type == "Regression":
                selector_ig = SelectKBest(score_func=mutual_info_regression, k=n_features_after_ig)
            else:
                selector_ig = SelectKBest(score_func=mutual_info_classif, k=n_features_after_ig)
                
            X_train_ig = selector_ig.fit_transform(X_fs, data[target_column])
            
            # Dapatkan nama fitur yang terpilih
            selected_features_ig_mask = selector_ig.get_support()
            selected_features_ig_names = X_fs.columns[selected_features_ig_mask]
            
            # Tampilkan hasil tahap 1
            st.write(f"Fitur terpilih setelah Information Gain ({n_features_after_ig}):" if st.session_state.language == 'id' else 
                    f"Selected features after Information Gain ({n_features_after_ig}):")
            st.write(", ".join(selected_features_ig_names))
            
            # Tahap 2: Seleksi Fitur dengan Feature Importance dari Random Forest
            n_features_after_rf_fi = max(1, int(len(selected_features_ig_names) * rf_percent / 100))
            
            # Latih Random Forest pada data yang sudah difilter IG
            if problem_type == "Regression":
                rf_model_for_importance = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                rf_model_for_importance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
            rf_model_for_importance.fit(X_fs[selected_features_ig_names], data[target_column])
            
            # Dapatkan feature importance
            feature_importances_rf = pd.Series(rf_model_for_importance.feature_importances_, index=selected_features_ig_names)
            sorted_importances_rf = feature_importances_rf.sort_values(ascending=False)
            
            # Pilih fitur-fitur teratas berdasarkan importance
            top_features_rf_names = sorted_importances_rf.head(n_features_after_rf_fi).index.tolist()
            
            # Tampilkan hasil tahap 2
            st.write(f"Fitur terpilih setelah Random Forest Feature Importance ({n_features_after_rf_fi}):" if st.session_state.language == 'id' else 
                    f"Selected features after Random Forest Feature Importance ({n_features_after_rf_fi}):")
            st.write(", ".join(top_features_rf_names))
            
            # Tahap 3: Seleksi Fitur dengan Recursive Feature Elimination (RFE) + Random Forest
            n_features_final = min(final_features, len(top_features_rf_names))
            
            # Pastikan jumlah fitur akhir minimal 2 untuk RFE
            if len(top_features_rf_names) < 2:
                # Jika fitur kurang dari 2, gunakan semua fitur yang tersisa tanpa RFE
                final_selected_features_names = top_features_rf_names
                st.warning("Jumlah fitur setelah tahap 2 kurang dari 2. RFE membutuhkan minimal 2 fitur. Menggunakan semua fitur dari tahap 2." if st.session_state.language == 'id' else 
                          "Number of features after stage 2 is less than 2. RFE requires at least 2 features. Using all features from stage 2.")
            else:
                # Gunakan Random Forest Classifier/Regressor sebagai estimator untuk RFE
                if problem_type == "Regression":
                    estimator_rfe = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                else:
                    estimator_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                
                # Pastikan n_features_final minimal 2
                n_features_final = max(2, n_features_final)
                
                selector_rfe = RFE(estimator=estimator_rfe, n_features_to_select=n_features_final, step=1, verbose=0)
                
                # Lakukan RFE pada data yang sudah difilter oleh RF Feature Importance
                selector_rfe.fit(X_fs[top_features_rf_names], data[target_column])
                
                # Dapatkan nama fitur akhir yang terpilih
                selected_features_rfe_mask = selector_rfe.get_support()
                final_selected_features_names = np.array(top_features_rf_names)[selected_features_rfe_mask].tolist()
            
            # Tampilkan hasil akhir
            st.write(f"Fitur akhir terpilih setelah RFE ({n_features_final}):" if st.session_state.language == 'id' else 
                    f"Final selected features after RFE ({n_features_final}):")
            st.write(", ".join(final_selected_features_names))
            
            # Tampilkan perbandingan jumlah fitur di setiap tahap
            st.subheader("Ringkasan Seleksi Fitur" if st.session_state.language == 'id' else "Feature Selection Summary")
            summary_data = {
                "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                           "Information Gain", 
                                                                           "Random Forest", 
                                                                           "RFE"],
                "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                             n_features_after_ig, 
                                                                                             n_features_after_rf_fi, 
                                                                                             n_features_final]
            }
            st.table(pd.DataFrame(summary_data))
            
            # Set fitur yang terpilih untuk digunakan dalam model
            selected_features = final_selected_features_names    

        if not selected_features:
            st.warning("Silahkan pilih fitur terlebih dahulu." if st.session_state.language == 'id' else "Please select at least one feature.")
        else:
            # Tampilkan hasil tahap pertama
            st.success(f"Tahap 1 selesai: {len(selected_features)} fitur terpilih" if st.session_state.language == 'id' else f"Stage 1 completed: {selected_features} features selected")
            st.write(f"Fitur terpilih tahap 1: {', '.join(selected_features)}" if st.session_state.language == 'id' else f"Stage 1 selected features: {', '.join(selected_features)}")
            
            # TAHAP KEDUA FEATURE SELECTION
            st.subheader("Tahap 2: Seleksi Fitur Lanjutan" if st.session_state.language == 'id' else "Stage 2: Advanced Feature Selection")
            
            # Checkbox untuk mengaktifkan tahap kedua
            enable_second_stage = st.checkbox("Aktifkan tahap kedua seleksi fitur" if st.session_state.language == 'id' else "Enable second stage feature selection", value=False)
            
            if enable_second_stage:
                # Gunakan hasil tahap pertama sebagai input tahap kedua
                all_columns_stage2 = selected_features
                
                # Pilih algoritma seleksi fitur tahap kedua
                feature_selection_method_stage2 = st.selectbox(
                    "Metode seleksi fitur tahap 2:" if st.session_state.language == 'id' else "Feature selection method stage 2:",
                    [
                        "Manual",
                        "Mutual Information",
                        "Pearson Correlation",
                        "Recursive Feature Elimination (RFE)",
                        "LASSO",
                        "Gradient Boosting Importance",
                        "Random Forest Importance",
                        "Ensemble Feature Selection",
                        "Multi-Stage Feature Selection"
                    ],
                    key="feature_selection_stage2"
                )

                selected_features_stage2 = all_columns_stage2  # Default

                if feature_selection_method_stage2 == "Manual":
                    selected_features_stage2 = st.multiselect(
                        "Pilih fitur untuk model (tahap 2):" if st.session_state.language == 'id' else "Select features to include in the model (stage 2):",
                        all_columns_stage2,
                        default=all_columns_stage2,
                        key="manual_selection_stage2"
                    )
                elif feature_selection_method_stage2 == "Mutual Information":
                    if problem_type == "Regression":
                        mi = mutual_info_regression(data[all_columns_stage2], data[target_column])
                    else:
                        mi = mutual_info_classif(data[all_columns_stage2], data[target_column])
                    mi_df = pd.DataFrame({"Feature": all_columns_stage2, "Mutual Information": mi})
                    mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                    st.dataframe(mi_df)
                    top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_mi_stage2")
                    selected_features_stage2 = mi_df.head(top_n)["Feature"].tolist()
                elif feature_selection_method_stage2 == "Pearson Correlation":
                    numeric_columns = data[all_columns_stage2].select_dtypes(include=[np.number]).columns.tolist()
                    if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                        st.error("Target kolom harus numerik untuk Pearson Correlation.")
                        corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
                    else:
                        corr = data[numeric_columns].corrwith(data[target_column]).abs()
                    corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
                    corr_df = corr_df.sort_values("Correlation", ascending=False)
                    st.dataframe(corr_df)
                    top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_corr_stage2")
                    selected_features_stage2 = corr_df.head(top_n)["Feature"].tolist()
                elif feature_selection_method_stage2 == "Recursive Feature Elimination (RFE)":
                    from sklearn.feature_selection import RFE
                    # --- Tambahkan encoding untuk fitur kategorikal sebelum RFE ---
                    X_rfe = data[all_columns_stage2].copy()
                    for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
                    if problem_type == "Regression":
                        estimator = LinearRegression()
                    else:
                        estimator = LogisticRegression(max_iter=500)
                    n_features_rfe = st.slider("Jumlah fitur RFE (tahap 2):" if st.session_state.language == 'id' else "Number of RFE features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="rfe_features_stage2")
                    rfe = RFE(estimator, n_features_to_select=n_features_rfe)
                    rfe.fit(X_rfe, data[target_column])
                    rfe_df = pd.DataFrame({"Feature": all_columns_stage2, "Selected": rfe.support_})
                    st.dataframe(rfe_df)
                    selected_features_stage2 = rfe_df[rfe_df["Selected"]]["Feature"].tolist()
                elif feature_selection_method_stage2 == "LASSO":
                    from sklearn.linear_model import Lasso, LogisticRegression
                    alpha_lasso = st.slider("Alpha LASSO (tahap 2):" if st.session_state.language == 'id' else "LASSO Alpha (stage 2):", 0.001, 1.0, 0.01, key="alpha_lasso_stage2")
                    if problem_type == "Regression":
                        lasso = Lasso(alpha=alpha_lasso, max_iter=1000)
                    else:
                        lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, C=1/alpha_lasso)
                    lasso.fit(data[all_columns_stage2], data[target_column])
                    coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                    if coef.ndim > 1:
                        coef = coef[0]
                    lasso_df = pd.DataFrame({"Feature": all_columns_stage2, "Coefficient": coef})
                    lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                    st.dataframe(lasso_df)
                    selected_features_stage2 = lasso_df["Feature"].tolist()
                elif feature_selection_method_stage2 == "Gradient Boosting Importance":
                    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                    if problem_type == "Regression":
                        model = GradientBoostingRegressor(random_state=42)
                    else:
                        model = GradientBoostingClassifier(random_state=42)
                    model.fit(data[all_columns_stage2], data[target_column])
                    importances = model.feature_importances_
                    gb_df = pd.DataFrame({"Feature": all_columns_stage2, "Importance": importances})
                    gb_df = gb_df.sort_values("Importance", ascending=False)
                    st.dataframe(gb_df)
                    top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_gb_stage2")
                    selected_features_stage2 = gb_df.head(top_n)["Feature"].tolist()
                elif feature_selection_method_stage2 == "Random Forest Importance":
                    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                    if problem_type == "Regression":
                        model = RandomForestRegressor(random_state=42)
                    else:
                        model = RandomForestClassifier(random_state=42)
                    model.fit(data[all_columns_stage2], data[target_column])
                    importances = model.feature_importances_
                    rf_df = pd.DataFrame({"Feature": all_columns_stage2, "Importance": importances})
                    rf_df = rf_df.sort_values("Importance", ascending=False)
                    st.dataframe(rf_df)
                    top_n = st.slider("Top N features (tahap 2):" if st.session_state.language == 'id' else "Top N features (stage 2):", 1, len(all_columns_stage2), min(5, len(all_columns_stage2)), key="topn_rf_stage2")
                    selected_features_stage2 = rf_df.head(top_n)["Feature"].tolist()

                elif feature_selection_method_stage2 == "Ensemble Feature Selection":
                    st.info("Pilih dua metode seleksi fitur untuk digabungkan (tahap 2)." if st.session_state.language == 'id' else "Select two feature selection methods to combine (stage 2).")
                    method1_stage2 = st.selectbox("Metode pertama (tahap 2):" if st.session_state.language == 'id' else "First method (stage 2):", [
                        "Mutual Information",
                        "Pearson Correlation",
                        "Recursive Feature Elimination (RFE)",
                        "LASSO",
                        "Gradient Boosting Importance",
                        "Random Forest Importance"
                    ], key="ensemble_method1_stage2")
                    method2_stage2 = st.selectbox("Metode kedua (tahap 2):" if st.session_state.language == 'id' else "Second method (stage 2):", [
                        "Random Forest Importance"
                        "Mutual Information",
                        "Pearson Correlation",
                        "Recursive Feature Elimination (RFE)",
                        "LASSO",
                        "Gradient Boosting Importance"
                    ], key="ensemble_method2_stage2")

                    combine_type_stage2 = st.radio("Gabungkan hasil dengan (tahap 2):" if st.session_state.language == 'id' else "Combine results with (stage 2):", ["Intersection", "Union"], index=0, key="combine_type_stage2")

                    def get_features_by_method_stage2(method, features_list):
                        if method == "Mutual Information":
                            if problem_type == "Regression":
                                mi = mutual_info_regression(data[features_list], data[target_column])
                            else:
                                mi = mutual_info_classif(data[features_list], data[target_column])
                            mi_df = pd.DataFrame({"Feature": features_list, "Mutual Information": mi})
                            mi_df = mi_df.sort_values("Mutual Information", ascending=False)
                            top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                            return set(mi_df.head(top_n)["Feature"].tolist())
                        elif method == "Pearson Correlation":
                            numeric_columns = data[features_list].select_dtypes(include=[np.number]).columns.tolist()
                            if data[target_column].dtype not in [np.float64, np.int64, np.float32, np.int32]:
                                corr = pd.Series([np.nan]*len(numeric_columns), index=numeric_columns)
                            else:
                                corr = data[numeric_columns].corrwith(data[target_column]).abs()
                            corr_df = pd.DataFrame({"Feature": numeric_columns, "Correlation": corr})
                            corr_df = corr_df.sort_values("Correlation", ascending=False)
                            top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                            return set(corr_df.head(top_n)["Feature"].tolist())
                        elif method == "Recursive Feature Elimination (RFE)":
                            from sklearn.feature_selection import RFE
                            from sklearn.linear_model import LinearRegression, LogisticRegression
                            X_rfe = data[features_list].copy()
                            for col in X_rfe.select_dtypes(include=['object', 'category']).columns:
                                le = LabelEncoder()
                                X_rfe[col] = le.fit_transform(X_rfe[col].astype(str))
                            if problem_type == "Regression":
                                estimator = LinearRegression()
                            else:
                                estimator = LogisticRegression(max_iter=500)
                            n_features_rfe = st.slider(f"Jumlah fitur RFE ({method}, tahap 2):" if st.session_state.language == 'id' else f"Number of RFE features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"rfe_{method}_stage2")
                            rfe = RFE(estimator, n_features_to_select=n_features_rfe)
                            rfe.fit(X_rfe, data[target_column])
                            rfe_df = pd.DataFrame({"Feature": features_list, "Selected": rfe.support_})
                            return set(rfe_df[rfe_df["Selected"]]["Feature"].tolist())
                        elif method == "LASSO":
                            from sklearn.linear_model import Lasso, LogisticRegression
                            alpha_lasso = st.slider(f"Alpha LASSO ({method}, tahap 2):" if st.session_state.language == 'id' else f"LASSO Alpha ({method}, stage 2):", 0.001, 1.0, 0.01, key=f"alpha_{method}_stage2")
                            if problem_type == "Regression":
                                lasso = Lasso(alpha=alpha_lasso, max_iter=1000)
                            else:
                                lasso = LogisticRegression(penalty='l1', solver='liblinear', max_iter=500, C=1/alpha_lasso)
                            lasso.fit(data[features_list], data[target_column])
                            coef = lasso.coef_ if hasattr(lasso, "coef_") else lasso.coef_
                            if coef.ndim > 1:
                                coef = coef[0]
                            lasso_df = pd.DataFrame({"Feature": features_list, "Coefficient": coef})
                            lasso_df = lasso_df[lasso_df["Coefficient"] != 0].sort_values("Coefficient", ascending=False)
                            return set(lasso_df["Feature"].tolist())
                        elif method == "Gradient Boosting Importance":
                            from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
                            if problem_type == "Regression":
                                model = GradientBoostingRegressor(random_state=42)
                            else:
                                model = GradientBoostingClassifier(random_state=42)
                            model.fit(data[features_list], data[target_column])
                            importances = model.feature_importances_
                            gb_df = pd.DataFrame({"Feature": features_list, "Importance": importances})
                            gb_df = gb_df.sort_values("Importance", ascending=False)
                            top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                            return set(gb_df.head(top_n)["Feature"].tolist())
                        elif method == "Random Forest Importance":
                            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
                            if problem_type == "Regression":
                                model = RandomForestRegressor(random_state=42)
                            else:
                                model = RandomForestClassifier(random_state=42)
                            model.fit(data[features_list], data[target_column])
                            importances = model.feature_importances_
                            rf_df = pd.DataFrame({"Feature": features_list, "Importance": importances})
                            rf_df = rf_df.sort_values("Importance", ascending=False)
                            top_n = st.slider(f"Top N fitur ({method}, tahap 2):" if st.session_state.language == 'id' else f"Top N features ({method}, stage 2):", 1, len(features_list), min(5, len(features_list)), key=f"topn_{method}_stage2")
                            return set(rf_df.head(top_n)["Feature"].tolist())
                        else:
                            return set(features_list)

                    features1_stage2 = get_features_by_method_stage2(method1_stage2, all_columns_stage2)
                    features2_stage2 = get_features_by_method_stage2(method2_stage2, all_columns_stage2)

                    if combine_type_stage2 == "Intersection":
                        selected_features_stage2 = list(features1_stage2 & features2_stage2)
                    else:
                        selected_features_stage2 = list(features1_stage2 | features2_stage2)

                    st.write(f"Fitur hasil gabungan tahap 2: {selected_features_stage2}" if st.session_state.language == 'id' else f"Combined features stage 2: {selected_features_stage2}")

                elif feature_selection_method_stage2 == "Multi-Stage Feature Selection":
                    st.subheader("Multi-Stage Feature Selection (Tahap 2)" if st.session_state.language == 'id' else "Multi-Stage Feature Selection (Stage 2)")
                    st.info("Metode ini menggunakan pendekatan 3 tahap: Information Gain → Random Forest Feature Importance → RFE (pada hasil tahap 1)" if st.session_state.language == 'id' else 
                           "This method uses a 3-stage approach: Information Gain → Random Forest Feature Importance → RFE (on stage 1 results)")
                    
                    from sklearn.feature_selection import RFE, SelectKBest
                    from sklearn.ensemble import RandomForestClassifier
                    
                    # Persiapkan data untuk feature selection tahap 2
                    X_fs_stage2 = data[all_columns_stage2].copy()
                    for col in X_fs_stage2.select_dtypes(include=['object', 'category']).columns:
                        le = LabelEncoder()
                        X_fs_stage2[col] = le.fit_transform(X_fs_stage2[col].astype(str))
                    
                    # Tampilkan parameter untuk setiap tahap
                    st.write("Tahap 1: Information Gain (pada hasil tahap 1)" if st.session_state.language == 'id' else "Stage 1: Information Gain (on stage 1 results)")
                    ig_percent_stage2 = st.slider("Persentase fitur yang dipertahankan setelah Information Gain (%, tahap 2)" if st.session_state.language == 'id' else 
                                          "Percentage of features to keep after Information Gain (%, stage 2)", 10, 90, 40, key="ig_percent_stage2")
                    
                    st.write("Tahap 2: Random Forest Feature Importance (tahap 2)" if st.session_state.language == 'id' else "Stage 2: Random Forest Feature Importance (stage 2)")
                    rf_percent_stage2 = st.slider("Persentase fitur yang dipertahankan setelah Random Forest (%, tahap 2)" if st.session_state.language == 'id' else 
                                          "Percentage of features to keep after Random Forest (%, stage 2)", 10, 90, 50, key="rf_percent_stage2")
                    
                    st.write("Tahap 3: Recursive Feature Elimination (tahap 2)" if st.session_state.language == 'id' else "Stage 3: Recursive Feature Elimination (stage 2)")
                    final_features_stage2 = st.slider("Jumlah fitur akhir (tahap 2)" if st.session_state.language == 'id' else "Final number of features (stage 2)", 
                                              1, min(10, len(all_columns_stage2)), min(5, len(all_columns_stage2)), key="final_features_stage2")
                    
                    # Tahap 1: Seleksi Fitur dengan Information Gain (SelectKBest + mutual_info_classif)
                    n_features_after_ig_stage2 = max(1, int(X_fs_stage2.shape[1] * ig_percent_stage2 / 100))
                    
                    if problem_type == "Regression":
                        selector_ig_stage2 = SelectKBest(score_func=mutual_info_regression, k=n_features_after_ig_stage2)
                    else:
                        selector_ig_stage2 = SelectKBest(score_func=mutual_info_classif, k=n_features_after_ig_stage2)
                        
                    X_train_ig_stage2 = selector_ig_stage2.fit_transform(X_fs_stage2, data[target_column])
                    
                    # Dapatkan nama fitur yang terpilih
                    selected_features_ig_mask_stage2 = selector_ig_stage2.get_support()
                    selected_features_ig_names_stage2 = X_fs_stage2.columns[selected_features_ig_mask_stage2]
                    
                    # Tampilkan hasil tahap 1
                    st.write(f"Fitur terpilih setelah Information Gain tahap 2 ({n_features_after_ig_stage2}):" if st.session_state.language == 'id' else 
                            f"Selected features after Information Gain stage 2 ({n_features_after_ig_stage2}):")
                    st.write(", ".join(selected_features_ig_names_stage2))
                    
                    # Tahap 2: Seleksi Fitur dengan Feature Importance dari Random Forest
                    n_features_after_rf_fi_stage2 = max(1, int(len(selected_features_ig_names_stage2) * rf_percent_stage2 / 100))
                    
                    # Latih Random Forest pada data yang sudah difilter IG
                    if problem_type == "Regression":
                        rf_model_for_importance_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                    else:
                        rf_model_for_importance_stage2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        
                    rf_model_for_importance_stage2.fit(X_fs_stage2[selected_features_ig_names_stage2], data[target_column])
                    
                    # Dapatkan feature importance
                    feature_importances_rf_stage2 = pd.Series(rf_model_for_importance_stage2.feature_importances_, index=selected_features_ig_names_stage2)
                    sorted_importances_rf_stage2 = feature_importances_rf_stage2.sort_values(ascending=False)
                    
                    # Pilih fitur-fitur teratas berdasarkan importance
                    top_features_rf_names_stage2 = sorted_importances_rf_stage2.head(n_features_after_rf_fi_stage2).index.tolist()
                    
                    # Tampilkan hasil tahap 2
                    st.write(f"Fitur terpilih setelah Random Forest Feature Importance tahap 2 ({n_features_after_rf_fi_stage2}):" if st.session_state.language == 'id' else 
                            f"Selected features after Random Forest Feature Importance stage 2 ({n_features_after_rf_fi_stage2}):")
                    st.write(", ".join(top_features_rf_names_stage2))
                    
                    # Tahap 3: Seleksi Fitur dengan Recursive Feature Elimination (RFE) + Random Forest
                    n_features_final_stage2 = min(final_features_stage2, len(top_features_rf_names_stage2))
                    
                    # Pastikan jumlah fitur akhir minimal 2 untuk RFE
                    if len(top_features_rf_names_stage2) < 2:
                        # Jika fitur kurang dari 2, gunakan semua fitur yang tersisa tanpa RFE
                        final_selected_features_names_stage2 = top_features_rf_names_stage2
                        st.warning("Jumlah fitur setelah tahap 2 kurang dari 2. RFE membutuhkan minimal 2 fitur. Menggunakan semua fitur dari tahap 2." if st.session_state.language == 'id' else 
                                  "Number of features after stage 2 is less than 2. RFE requires at least 2 features. Using all features from stage 2.")
                    else:
                        # Gunakan Random Forest Classifier/Regressor sebagai estimator untuk RFE
                        if problem_type == "Regression":
                            estimator_rfe_stage2 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
                        else:
                            estimator_rfe_stage2 = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                        
                        # Pastikan n_features_final minimal 2
                        n_features_final_stage2 = max(2, n_features_final_stage2)
                        
                        selector_rfe_stage2 = RFE(estimator=estimator_rfe_stage2, n_features_to_select=n_features_final_stage2, step=1, verbose=0)
                        
                        # Lakukan RFE pada data yang sudah difilter oleh RF Feature Importance
                        selector_rfe_stage2.fit(X_fs_stage2[top_features_rf_names_stage2], data[target_column])
                        
                        # Dapatkan nama fitur akhir yang terpilih
                        selected_features_rfe_mask_stage2 = selector_rfe_stage2.get_support()
                        final_selected_features_names_stage2 = np.array(top_features_rf_names_stage2)[selected_features_rfe_mask_stage2].tolist()
                    
                    # Tampilkan hasil akhir
                    st.write(f"Fitur akhir terpilih setelah RFE tahap 2 ({n_features_final_stage2}):" if st.session_state.language == 'id' else 
                            f"Final selected features after RFE stage 2 ({n_features_final_stage2}):")
                    st.write(", ".join(final_selected_features_names_stage2))
                    
                    # Tampilkan perbandingan jumlah fitur di setiap tahap
                    st.subheader("Ringkasan Seleksi Fitur Tahap 2" if st.session_state.language == 'id' else "Feature Selection Summary Stage 2")
                    summary_data_stage2 = {
                        "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal (dari tahap 1)" if st.session_state.language == 'id' else "Initial (from stage 1)", 
                                                                               "Information Gain", 
                                                                               "Random Forest", 
                                                                               "RFE"],
                        "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns_stage2), 
                                                                                                 n_features_after_ig_stage2, 
                                                                                                 n_features_after_rf_fi_stage2, 
                                                                                                 n_features_final_stage2]
                    }
                    st.table(pd.DataFrame(summary_data_stage2))
                    
                    # Set fitur yang terpilih untuk digunakan dalam model
                    selected_features_stage2 = final_selected_features_names_stage2
                
                # Tampilkan hasil tahap kedua
                if selected_features_stage2:
                    st.success(f"Tahap 2 selesai: {len(selected_features_stage2)} fitur terpilih" if st.session_state.language == 'id' else f"Stage 2 completed: {len(selected_features_stage2)} features selected")
                    st.write(f"Fitur terpilih tahap 2: {', '.join(selected_features_stage2)}" if st.session_state.language == 'id' else f"Stage 2 selected features: {', '.join(selected_features_stage2)}")
                    
                    # Gunakan hasil tahap kedua sebagai fitur final
                    final_selected_features = selected_features_stage2
                else:
                    st.warning("Tidak ada fitur yang terpilih di tahap 2. Menggunakan hasil tahap 1." if st.session_state.language == 'id' else "No features selected in stage 2. Using stage 1 results.")
                    final_selected_features = selected_features
            else:
                # Jika tahap kedua tidak diaktifkan, gunakan hasil tahap pertama
                final_selected_features = selected_features
            
            # Tampilkan ringkasan akhir
            st.subheader("Ringkasan Seleksi Fitur Akhir" if st.session_state.language == 'id' else "Final Feature Selection Summary")
            if enable_second_stage:
                comparison_data = {
                    "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                           "Tahap 1" if st.session_state.language == 'id' else "Stage 1", 
                                                                           "Tahap 2" if st.session_state.language == 'id' else "Stage 2"],
                    "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                             len(selected_features), 
                                                                                             len(final_selected_features)]
                }
            else:
                comparison_data = {
                    "Tahap" if st.session_state.language == 'id' else "Stage": ["Awal" if st.session_state.language == 'id' else "Initial", 
                                                                           "Tahap 1" if st.session_state.language == 'id' else "Stage 1"],
                    "Jumlah Fitur" if st.session_state.language == 'id' else "Number of Features": [len(all_columns), 
                                                                                             len(selected_features)]
                }
                final_selected_features = selected_features
            
            st.table(pd.DataFrame(comparison_data))
            st.write(f"Fitur akhir yang akan digunakan: {', '.join(final_selected_features)}" if st.session_state.language == 'id' else f"Final features to be used: {', '.join(final_selected_features)}")
            
            # Prepare data for modeling dengan fitur akhir
            X = data[final_selected_features]
            y = data[target_column]
                        
            
            # Display processed data
            st.subheader("Tampilkan Data Terproses" if st.session_state.language == 'id' else "Processed Data Preview")
            st.dataframe(X.head())

            # Update session_state setelah encoding/scaling
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test

            
    else:
        st.info("Silahkan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")

# Tab 4: Feature Engineering and Model Training
with tab4:
    st.header("Pelatihan dan Evaluasi Model" if st.session_state.language == 'id' else "Model Training and Evaluation")
    
    if (st.session_state.X_train is not None and 
        st.session_state.y_train is not None and 
        st.session_state.problem_type is not None):
        
        problem_type = st.session_state.problem_type
        
        # Check if data might be time series
        is_timeseries = False
        date_columns = []
        
        # Try to identify date/time columns
        if st.session_state.data is not None:
            for col in st.session_state.data.columns:
                # Check if column name contains date-related keywords
                if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']):
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(st.session_state.data[col])
                        date_columns.append(col)
                    except:
                        pass
        
        # If date columns found, ask user if this is time series data
        if date_columns:
            is_timeseries = st.checkbox("Data ini adalah data deret waktu (time series)", value=False)
        
        # Add K-Fold Cross Validation option
        use_kfold = st.checkbox("Gunakan K-Fold Cross Validation" if st.session_state.language == 'id' else "Use K-Fold Cross Validation", value=False)
        
        if use_kfold:
            from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
            
            st.subheader("Lakukan K-Fold Cross Validation" if st.session_state.language == 'id' else "Perform K-Fold Cross Validation")
            
            n_splits = st.slider("Jumlah fold (K):" if st.session_state.language == 'id' else "Number of folds (K):", 2, 10, 5)
            cv_scoring = None
            
            if problem_type == "Classification":
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:",
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]
                )
                
                # Display fold distribution for classification
                st.write("**Distribisi Data per Fold:**" if st.session_state.language == 'id' else "**Data Distribution per Fold:**")
                
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                fold_info = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                    y_fold_train = y.iloc[train_idx]
                    y_fold_val = y.iloc[val_idx]
                    
                    fold_counts = pd.Series(y_fold_val).value_counts().sort_index()
                    fold_info.append({
                        'Fold': f'Fold {fold_idx + 1}',
                        'Training Samples': len(train_idx),
                        'Validation Samples': len(val_idx),
                        **{f'Class {label}': count for label, count in fold_counts.items()}
                    })
                
                fold_df = pd.DataFrame(fold_info)
                st.dataframe(fold_df)
            else:  # Regression
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:",
                    ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
                )
        
        if is_timeseries:
            st.subheader("Pelatihan Model Forecasting" if st.session_state.language == 'id' else "Forecasting Model Training")
            
            # Select date column
            date_column = st.selectbox("Pilih kolom tanggal/waktu:" if st.session_state.language == 'id' else "Select date column:", date_columns)
            
            # Select target column
            target_column = st.selectbox("Pilih kolom target untuk diprediksi:" if st.session_state.language == 'id' else "Select target column for prediction:", 
                                        [col for col in st.session_state.data.columns 
                                         if col != date_column and col in st.session_state.numerical_columns])
            
            # Select frequency
            freq = st.selectbox("Frekuensi data:", ["Harian (D)", "Mingguan (W)", "Bulanan (M)", "Tahunan (Y)", "Lainnya"] if st.session_state.language == 'id' else ["Daily (D)", "Weekly (W)", "Monthly (M)", "Yearly (Y)", "Other"])
            freq_map = {"Harian (D)": "D", "Mingguan (W)": "W", "Bulanan (M)": "M", "Tahunan (Y)": "Y", "Lainnya": None if st.session_state.language == 'id' else "Other"}
            selected_freq = freq_map[freq]
            
            # Number of periods to forecast
            forecast_periods = st.slider("Jumlah periode untuk prediksi ke depan:" if st.session_state.language == 'id' else "Number of periods to forecast:", 1, 100, 10)
            
            # Select forecasting model
            model_type = st.selectbox("Pilih model forecasting:" if st.session_state.language == 'id' else "Select forecasting model:", 
                                     ["ARIMA", "Exponential Smoothing", "Prophet", "Random Forest", "Gradient Boosting"])
            
            # Import required modules
            try:
                from forecasting_utils import (
                    train_arima_model, train_exponential_smoothing, 
                    train_ml_forecaster, forecast_future, 
                    evaluate_forecast_model, plot_forecast_results
                )
                from utils import prepare_timeseries_data, check_stationarity, plot_timeseries_analysis
                
                FORECASTING_MODULES_AVAILABLE = True
            except ImportError:
                st.error("Modul forecasting tidak tersedia. Pastikan file utils.py dan forecasting_utils.py ada di direktori yang sama." if st.session_state.language == 'id' else "Forecasting modules not available. Ensure utils.py and forecasting_utils.py are in the same directory.")
                FORECASTING_MODULES_AVAILABLE = False
            
            if FORECASTING_MODULES_AVAILABLE:
                # Import statsmodels conditionally
                try:
                    import statsmodels.api as sm
                    from statsmodels.tsa.arima.model import ARIMA
                    from statsmodels.tsa.holtwinters import ExponentialSmoothing
                    STATSMODELS_AVAILABLE = True
                except ImportError:
                    STATSMODELS_AVAILABLE = False
                    if model_type in ["ARIMA", "Exponential Smoothing"]:
                        st.warning("Statsmodels tidak terinstal. Silakan instal dengan 'pip install statsmodels'." if st.session_state.language == 'id' else "Statsmodels not installed. Please install with 'pip install statsmodels'.")
                
                # Import Prophet conditionally
                try:
                    from prophet import Prophet
                    PROPHET_AVAILABLE = True
                except ImportError:
                    PROPHET_AVAILABLE = False
                    if model_type == "Prophet":
                        st.warning("Prophet tidak terinstal. Silakan instal dengan 'pip install prophet'." if st.session_state.language == 'id' else "Prophet not installed. Please install with 'pip install prophet'.")
                
                # Prepare time series data
                if st.button("Proses Data Time Series" if st.session_state.language == 'id' else "Process Time Series Data"):
                    with st.spinner("Memproses data time series..." if st.session_state.language == 'id' else "Processing time series data..."):
                        # Prepare data
                        ts_data = prepare_timeseries_data(
                            st.session_state.data, 
                            date_column, 
                            target_column, 
                            freq=selected_freq
                        )
                        
                        # Check stationarity
                        stationarity_result = check_stationarity(ts_data[target_column])
                        st.write("Hasil Uji Stasioneritas:" if st.session_state.language == 'id' else "Stationarity Test Results:")
                        st.write(f"- Test Statistic: {stationarity_result['Test Statistic']:.4f}")
                        st.write(f"- p-value: {stationarity_result['p-value']:.4f}")
                        st.write(f"- Data {'stasioner' if stationarity_result['Stationary'] else 'tidak stasioner'}")
                        
                        # Plot time series analysis
                        st.write("Analisis Time Series:" if st.session_state.language == 'id' else "Time Series Analysis:")
                        fig = plot_timeseries_analysis(ts_data[target_column])
                        st.pyplot(fig)
                        
                        # Split data for training and testing
                        train_size = int(len(ts_data) * 0.8)
                        train_data = ts_data.iloc[:train_size]
                        test_data = ts_data.iloc[train_size:]
                        
                        st.write(f"Data dibagi menjadi {len(train_data)} sampel training dan {len(test_data)} sampel testing" if st.session_state.language == 'id' else f"Data split into {len(train_data)} training samples and {len(test_data)} testing samples.")
                        
                        # Train model based on selection
                        if model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                            p = st.slider("Parameter p (AR):", 0, 5, 1)
                            d = st.slider("Parameter d (differencing):", 0, 2, 1)
                            q = st.slider("Parameter q (MA):", 0, 5, 1)
                            
                            with st.spinner("Melatih model ARIMA..." if st.session_state.language == 'id' else "Training ARIMA model..."):
                                model = train_arima_model(train_data, target_column, order=(p, d, q))
                                st.session_state.model = model
                                st.success("Model ARIMA berhasil dilatih!" if st.session_state.language == 'id' else "ARIMA model trained successfully!")
                        
                        elif model_type == "Exponential Smoothing" and STATSMODELS_AVAILABLE:
                            trend = st.selectbox("Tipe trend:", ["add", "mul", None])
                            seasonal = st.selectbox("Tipe seasonal:", ["add", "mul", None])
                            seasonal_periods = st.slider("Periode seasonal:", 0, 52, 12)
                            
                            with st.spinner("Melatih model Exponential Smoothing..." if st.session_state.language == 'id' else "Training Exponential Smoothing model..."):
                                model = train_exponential_smoothing(
                                    train_data, 
                                    target_column, 
                                    trend=trend, 
                                    seasonal=seasonal, 
                                    seasonal_periods=seasonal_periods
                                )
                                st.session_state.model = model
                                st.success("Model Exponential Smoothing berhasil dilatih!" if st.session_state.language == 'id' else "Exponential Smoothing model trained successfully!")
                        
                        elif model_type == "Prophet" and PROPHET_AVAILABLE:
                            yearly_seasonality = st.selectbox("Seasonality tahunan:" if st.session_state.language == 'id' else "Yearly seasonality:", ["auto", True, False])
                            weekly_seasonality = st.selectbox("Seasonality mingguan:" if st.session_state.language == 'id' else "Weekly seasonality:", ["auto", True, False])
                            daily_seasonality = st.selectbox("Seasonality harian:" if st.session_state.language == 'id' else "Daily seasonality:", ["auto", True, False])
                            
                            # Implementasi Prophet akan dilakukan di forecasting_utils.py
                            st.info("Implementasi Prophet akan menggunakan forecasting_utils.py" if st.session_state.language == 'id' else "Prophet implementation will use forecasting_utils.py")
                        
                        elif model_type in ["Random Forest", "Gradient Boosting"]:
                            n_estimators = st.slider("Jumlah trees:" if st.session_state.language == 'id' else "Number of trees:", 10, 500, 100)
                            max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                            
                            if model_type == "Random Forest":
                                model_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'random_state': 42
                                }
                            else:  # Gradient Boosting
                                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                                model_params = {
                                    'n_estimators': n_estimators,
                                    'learning_rate': learning_rate,
                                    'max_depth': max_depth,
                                    'random_state': 42
                                }
                            
                            with st.spinner(f"Melatih model {model_type}..." if st.session_state.language == 'id' else f"Training {model_type} model..."):
                                model_info = train_ml_forecaster(
                                    st.session_state.data,
                                    date_column,
                                    target_column,
                                    model_type=model_type.lower().replace(" ", "_"),
                                    **model_params
                                )
                                st.session_state.model = model_info
                                st.success(f"Model {model_type} berhasil dilatih!" if st.session_state.language == 'id' else f"{model_type} model trained successfully!")
                        
                        # Evaluate model if available
                        if st.session_state.model is not None:
                            with st.spinner("Mengevaluasi model..." if st.session_state.language == 'id' else "Evaluating model..."):
                                try:
                                    eval_results = evaluate_forecast_model(st.session_state.model, test_data, target_column)
                                    
                                    st.write("Hasil Evaluasi Model:" if st.session_state.language == 'id' else "Model Evaluation Results:")
                                    st.write(f"- RMSE: {eval_results['RMSE']:.4f}")
                                    st.write(f"- MAE: {eval_results['MAE']:.4f}")
                                    st.write(f"- R²: {eval_results['R2']:.4f}")
                                    
                                    # Generate forecast
                                    try:
                                        forecast_data = forecast_future(st.session_state.model, periods=forecast_periods)
                                        # Perbaikan: Konversi kolom tanggal ke string jika tipe datetime64[ns] dan out of bounds
                                        if 'date' in forecast_data.columns:
                                            # Cek dan konversi semua nilai tanggal yang out of bounds ke string
                                            def safe_to_datetime(val):
                                                try:
                                                    return pd.to_datetime(val)
                                                except (pd.errors.OutOfBoundsDatetime, ValueError, OverflowError):
                                                    return str(val)
                                            forecast_data['date'] = forecast_data['date'].apply(safe_to_datetime)
                                    except Exception as e:
                                        st.error(f"Error saat membuat forecast: {str(e)}" if st.session_state.language == 'id' else f"Error generating forecast: {str(e)}")
                                        forecast_data = None

                                    # Plot results
                                    if forecast_data is not None:
                                        try:
                                            fig = plot_forecast_results(train_data, test_data, forecast_data, target_column)
                                            st.pyplot(fig)
                                        except Exception as e:
                                            st.error(f"Error saat plotting hasil forecast: {str(e)}" if st.session_state.language == 'id' else f"Error plotting forecast results: {str(e)}")

                                    # Show forecast data
                                    if forecast_data is not None:
                                        st.write("Data Hasil Forecasting:" if st.session_state.language == 'id' else "Forecast Data:")
                                        st.dataframe(forecast_data)

                                # Download forecast data
                                except Exception as e:
                                    st.error(f"Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"Error evaluating model: {str(e)}")

        else:
            # Non-time series data - Classification or Regression
            # 3D Visualization Section
            st.subheader("Visualisasi 3D Data" if st.session_state.language == 'id' else "3D Data Visualization")
            
            # Add checkbox for 3D visualization
            show_3d_viz = st.checkbox("Tampilkan visualisasi 3D PCA/t-SNE" if st.session_state.language == 'id' else "Show 3D PCA/t-SNE visualization", value=False)
            
            if show_3d_viz:
                try:
                    from sklearn.decomposition import PCA
                    from sklearn.manifold import TSNE
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Prepare data for visualization
                    X_train_viz = st.session_state.X_train.copy()
                    X_test_viz = st.session_state.X_test.copy()
                    y_train_viz = st.session_state.y_train
                    y_test_viz = st.session_state.y_test
                    
                    # Combine train and test data for consistent visualization
                    X_combined = pd.concat([X_train_viz, X_test_viz])
                    y_combined = pd.concat([y_train_viz, y_test_viz])
                    
                    # Create dataset labels
                    dataset_labels = ['Train'] * len(X_train_viz) + ['Test'] * len(X_test_viz)
                    
                    # Select visualization method
                    viz_method = st.selectbox(
                        "Pilih metode visualisasi:" if st.session_state.language == 'id' else "Select visualization method:",
                        ["PCA", "t-SNE"]
                    )
                    
                    # Parameters for t-SNE
                    if viz_method == "t-SNE":
                        perplexity = st.slider(
                            "Perplexity:" if st.session_state.language == 'id' else "Perplexity:",
                            5, 50, 30
                        )
                        learning_rate = st.slider(
                            "Learning rate:" if st.session_state.language == 'id' else "Learning rate:",
                            10, 1000, 200
                        )
                        n_iter = st.slider(
                            "Number of iterations:" if st.session_state.language == 'id' else "Number of iterations:",
                            250, 2000, 1000
                        )
                    
                    if st.button("Generate 3D Visualization" if st.session_state.language == 'id' else "Generate 3D Visualization"):
                        with st.spinner("Membuat visualisasi 3D..." if st.session_state.language == 'id' else "Creating 3D visualization..."):
                            
                            if viz_method == "PCA":
                                # Apply PCA
                                pca = PCA(n_components=3)
                                X_3d = pca.fit_transform(X_combined)
                                
                                # Calculate explained variance
                                explained_var = pca.explained_variance_ratio_
                                
                                st.write(f"**PCA Explained Variance:**")
                                st.write(f"PC1: {explained_var[0]:.2%}")
                                st.write(f"PC2: {explained_var[1]:.2%}")
                                st.write(f"PC3: {explained_var[2]:.2%}")
                                st.write(f"Total: {sum(explained_var):.2%}")
                                
                            else:  # t-SNE
                                # Apply t-SNE
                                tsne = TSNE(
                                    n_components=3,
                                    perplexity=min(perplexity, len(X_combined) - 1),
                                    learning_rate=learning_rate,
                                    n_iter=n_iter,
                                    random_state=42
                                )
                                X_3d = tsne.fit_transform(X_combined)
                                
                                st.write(f"**t-SNE Parameters:**")
                                st.write(f"Perplexity: {perplexity}")
                                st.write(f"Learning rate: {learning_rate}")
                                st.write(f"Iterations: {n_iter}")
                            
                            # Create DataFrame for visualization
                            viz_df = pd.DataFrame({
                                'X': X_3d[:, 0],
                                'Y': X_3d[:, 1],
                                'Z': X_3d[:, 2],
                                'Target': y_combined.values if hasattr(y_combined, 'values') else y_combined,
                                'Dataset': dataset_labels
                            })
                            
                            # Create color mapping for categorical data
                            if problem_type == "Classification":
                                # Create color mapping for categorical targets
                                from sklearn.preprocessing import LabelEncoder
                                le = LabelEncoder()
                                viz_df['Target_Numeric'] = le.fit_transform(viz_df['Target'])
                                
                                # Get unique classes and create color mapping
                                unique_classes = le.classes_
                                n_classes = len(unique_classes)
                                
                                # Create color palette based on number of classes
                                if n_classes <= 10:
                                    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                                else:
                                    # Use a continuous colorscale for many classes
                                    colors = None
                                    
                                # Filter data for train and test
                                train_data = viz_df[viz_df['Dataset'] == 'Train']
                                test_data = viz_df[viz_df['Dataset'] == 'Test']
                            
                            # Create 3D scatter plot
                            fig = go.Figure()
                            
                            if problem_type == "Classification":
                                # Color mapping for classification
                                if colors and n_classes <= 10:
                                    # Use discrete colors for small number of classes
                                    for i, class_name in enumerate(unique_classes):
                                        class_train = train_data[train_data['Target'] == class_name]
                                        if len(class_train) > 0:
                                            fig.add_trace(go.Scatter3d(
                                                x=class_train['X'],
                                                y=class_train['Y'],
                                                z=class_train['Z'],
                                                mode='markers',
                                                name=f'Train - {class_name}',
                                                marker=dict(
                                                    size=4,
                                                    color=colors[i % len(colors)],
                                                    opacity=0.7
                                                ),
                                                text=[f"Train - {class_name}"] * len(class_train),
                                                hovertemplate='<b>Train Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Class: %{text}<extra></extra>'
                                            ))
                                        
                                        class_test = test_data[test_data['Target'] == class_name]
                                        if len(class_test) > 0:
                                            fig.add_trace(go.Scatter3d(
                                                x=class_test['X'],
                                                y=class_test['Y'],
                                                z=class_test['Z'],
                                                mode='markers',
                                                name=f'Test - {class_name}',
                                                marker=dict(
                                                    size=4,
                                                    color=colors[i % len(colors)],
                                                    opacity=0.8,
                                                    symbol='diamond'
                                                ),
                                                text=[f"Test - {class_name}"] * len(class_test),
                                                hovertemplate='<b>Test Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Class: %{text}<extra></extra>'
                                            ))
                                else:
                                    # Use continuous colorscale for many classes
                                    fig.add_trace(go.Scatter3d(
                                        x=train_data['X'],
                                        y=train_data['Y'],
                                        z=train_data['Z'],
                                        mode='markers',
                                        name='Training Data',
                                        marker=dict(
                                            size=4,
                                            color=train_data['Target_Numeric'],
                                            colorscale='Viridis',
                                            opacity=0.7
                                        ),
                                        text=[f"Train - {t}" for t in train_data['Target']],
                                        hovertemplate='<b>Train Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Target: %{text}<extra></extra>'
                                    ))
                                    
                                    fig.add_trace(go.Scatter3d(
                                        x=test_data['X'],
                                        y=test_data['Y'],
                                        z=test_data['Z'],
                                        mode='markers',
                                        name='Testing Data',
                                        marker=dict(
                                            size=4,
                                            color=test_data['Target_Numeric'],
                                            colorscale='Plasma',
                                            opacity=0.8,
                                            symbol='diamond'
                                        ),
                                        text=[f"Test - {t}" for t in test_data['Target']],
                                        hovertemplate='<b>Test Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Target: %{text}<extra></extra>'
                                    ))
                            else:
                                # For regression, use the original approach
                                # Add train data
                                train_data = viz_df[viz_df['Dataset'] == 'Train']
                                fig.add_trace(go.Scatter3d(
                                    x=train_data['X'],
                                    y=train_data['Y'],
                                    z=train_data['Z'],
                                    mode='markers',
                                    name='Training Data',
                                    marker=dict(
                                        size=4,
                                        color=train_data['Z'],
                                        colorscale='Blues',
                                        opacity=0.7
                                    ),
                                    text=[f"Train - Target: {t}" for t in train_data['Target']],
                                    hovertemplate='<b>Train Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Target: %{text}<extra></extra>'
                                ))
                                
                                # Add test data
                                test_data = viz_df[viz_df['Dataset'] == 'Test']
                                fig.add_trace(go.Scatter3d(
                                    x=test_data['X'],
                                    y=test_data['Y'],
                                    z=test_data['Z'],
                                    mode='markers',
                                    name='Testing Data',
                                    marker=dict(
                                        size=4,
                                        color=test_data['Z'],
                                        colorscale='Reds',
                                        opacity=0.8,
                                        symbol='diamond'
                                    ),
                                    text=[f"Test - Target: {t}" for t in test_data['Target']],
                                    hovertemplate='<b>Test Data</b><br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>Target: %{text}<extra></extra>'
                                ))
                            
                            # Update layout
                            fig.update_layout(
                                title=f"3D {viz_method} Visualization - {problem_type} Data" if st.session_state.language == 'id' else f"Visualisasi 3D {viz_method} - Data {problem_type}",
                                scene=dict(
                                    xaxis_title=f"{viz_method} 1",
                                    yaxis_title=f"{viz_method} 2",
                                    zaxis_title=f"{viz_method} 3"
                                ),
                                width=800,
                                height=600,
                                margin=dict(l=0, r=0, b=0, t=40)
                            )
                            
                            # Display the plot
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add summary statistics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Training Data:**")
                                st.write(f"Samples: {len(X_train_viz)}")
                                st.write(f"Features: {X_train_viz.shape[1]}")
                            with col2:
                                st.write("**Testing Data:**")
                                st.write(f"Samples: {len(X_test_viz)}")
                                st.write(f"Features: {X_test_viz.shape[1]}")
                            
                            # Add download option
                            csv = viz_df.to_csv(index=False)
                            st.download_button(
                                label="Download 3D Coordinates CSV" if st.session_state.language == 'id' else "Download 3D Coordinates CSV",
                                data=csv,
                                file_name=f"3d_{viz_method.lower()}_coordinates.csv",
                                mime="text/csv"
                            )
                            
                except ImportError as e:
                    st.error(f"Library yang diperlukan tidak tersedia: {str(e)}. Silakan instal dengan: pip install scikit-learn plotly" if st.session_state.language == 'id' else f"Required library not available: {str(e)}. Please install with: pip install scikit-learn plotly")
                except Exception as e:
                    st.error(f"Error saat membuat visualisasi: {str(e)}" if st.session_state.language == 'id' else f"Error creating visualization: {str(e)}")
            
            st.subheader(f"Melatih Model {problem_type}" if st.session_state.language == 'id' else f"Training a {problem_type} Model")
            
            # Tambahkan opsi untuk menggunakan GridSearchCV
            use_grid_search = st.checkbox("Gunakan GridSearchCV untuk hyperparameter tuning" if st.session_state.language == 'id' else "Use GridSearchCV for hyperparameter tuning", value=False)
            
            # Model selection
            if problem_type == "Classification":
                # Define available classification models
                classification_models = ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting", "MLP (Neural Network)"]
                                   
                model_type = st.selectbox("Select a classification model:" if st.session_state.language == 'id' else "Pilih model klasifikasi:", classification_models)
                st.session_state.model_type = model_type
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of trees:" if st.session_state.language == 'id' else "Jumlah pohon:", 10, 500, 100)
                    max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                    
                    base_model = RandomForestClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        
                elif model_type == "Logistic Regression" :
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Maximum iterations:" if st.session_state.language == 'id' else "Iterasi maksimum:", 100, 1000, 100)
                    
                    base_model = LogisticRegression(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.01, C/2), C, min(10.0, C*2)],
                            'solver': ['liblinear', 'lbfgs', 'saga'],
                            'max_iter': [100, 500, 1000]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = LogisticRegression(
                            C=C,
                            max_iter=max_iter,
                            random_state=42
                        )
                        
                elif model_type == "SVM":
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                    
                    base_model = SVC(probability=True, random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                            'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                            'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = SVC(
                            C=C,
                            kernel=kernel,
                            gamma=gamma,
                            probability=True,
                            random_state=42
                        )
                        
                elif model_type == "KNN":
                    n_neighbors = st.slider("Number of neighbors (K):" if st.session_state.language == 'id' else "Jumlah tetangga (K):", 1, 20, 5)
                    weights = st.selectbox("Weight function:" if st.session_state.language == 'id' else "Fungsi bobot:", ["uniform", "distance"])
                    algorithm = st.selectbox("Algorithm:" if st.session_state.language == 'id' else "Algoritma:", ["auto", "ball_tree", "kd_tree", "brute"])
                    
                    base_model = KNeighborsClassifier()
                    
                    if use_grid_search:
                        param_grid = {
                            'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2]  # Manhattan or Euclidean distance
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm
                        )
                        
                elif model_type == "Decision Tree":
                    max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                    min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel untuk membagi:", 2, 20, 2)
                    criterion = st.selectbox("Split criterion:" if st.session_state.language == 'id' else "Kriteria membagi:", ["gini", "entropy"])
                    
                    base_model = DecisionTreeClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'criterion': ['gini', 'entropy']
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            criterion=criterion,
                            random_state=42
                        )
                        
                elif model_type == "Naive Bayes":
                    var_smoothing = st.slider("Variance smoothing:" if st.session_state.language == 'id' else "Penyesuaian varian:", 1e-10, 1e-8, 1e-9, format="%.1e")
                    
                    base_model = GaussianNB()
                    
                    if use_grid_search:
                        param_grid = {
                            'var_smoothing': [1e-10, 1e-9, 1e-8]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = GaussianNB(
                            var_smoothing=var_smoothing
                        )
                        
                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Number of boosting stages:" if st.session_state.language == 'id' else "Jumlah boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                    
                    base_model = GradientBoostingClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [3, 6, 9] if max_depth == 3 else [max(1, max_depth-3), max_depth, min(10, max_depth+3)],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                        
                elif model_type == "MLP (Neural Network)":
                    st.subheader("Konfigurasi Neural Network Klasifikasi Lengkap" if st.session_state.language == 'id' else "Complete Neural Network Classification Configuration")
                    
                    # Tambahkan penjelasan teori di bagian atas
                    with st.expander("📚 Penjelasan Teori Neural Network Klasifikasi" if st.session_state.language == 'id' else "📚 Neural Network Classification Theory Explanation"):
                        st.markdown("""
                        ### 🧠 **Fungsi Aktivasi**
                        - **ReLU**: f(x) = max(0,x) - Cocok untuk hidden layers, mengatasi vanishing gradient
                        - **Sigmoid**: f(x) = 1/(1+e^(-x)) - Cocok untuk output binary classification
                        - **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Range [-1,1], lebih stabil dari sigmoid
                        
                        ### 🏗️ **Arsitektur Jaringan**
                        - **Feedforward Neural Network (FNN)**: Informasi mengalir satu arah (input → hidden → output)
                        - **Parameter yang dikonfigurasi**: Hidden layers, neurons per layer, activation function
                        
                        ### ⚡ **Optimizer & Learning**
                        - **Adam**: Kombinasi momentum dan adaptive learning rate, efisien untuk data besar
                        - **SGD**: Stochastic Gradient Descent dengan momentum untuk konvergensi lebih stabil
                        - **L-BFGS**: Optimizer kuasi-Newton untuk dataset kecil/medium
                        
                        ### 🛡️ **Regularization & Overfitting Prevention**
                        - **L2 Regularization (alpha)**: Menghindari overfitting dengan menjaga bobot tetap kecil
                        - **Early Stopping**: Menghentikan training saat validasi tidak membaik
                        - **Dropout**: (Tidak tersedia di MLPClassifier scikit-learn)
                        
                        ### 📊 **Hyperparameter Penting**
                        - **Learning Rate**: Kontrol kecepatan pembelajaran (0.001-0.1)
                        - **Batch Size**: Jumlah sampel per update (16-512)
                        - **Epochs**: Jumlah iterasi seluruh dataset (max_iter)
                        - **Hidden Layers**: Kompleksitas model (1-5 layers)
                        """)
                    
                    # Architecture Configuration
                    st.write("**Arsitektur Jaringan Klasifikasi:**" if st.session_state.language == 'id' else "**Classification Network Architecture:**")
                    
                    # Hidden layers configuration
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 5, 2)
                    with col2:
                        neurons_per_layer = st.text_input("Neurons per layer:", "128,128")
                        try:
                            neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                            if len(neurons_list) < num_hidden_layers:
                                neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                            elif len(neurons_list) > num_hidden_layers:
                                neurons_list = neurons_list[:num_hidden_layers]
                            hidden_layer_sizes = tuple(neurons_list)
                        except:
                            hidden_layer_sizes = (128, 128)
                    with col3:
                        activation = st.selectbox("Activation function:", 
                                                ["relu", "tanh", "logistic", "identity"],
                                                help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                    
                    # Advanced parameters
                    with st.expander("Advanced Parameters"):
                        col4, col5 = st.columns(2)
                        with col4:
                            solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"])
                            
                            if solver == "adam":
                                beta_1 = st.slider("Beta 1:", 0.8, 0.999, 0.9, format="%.3f")
                                beta_2 = st.slider("Beta 2:", 0.9, 0.9999, 0.999, format="%.4f")
                                epsilon = st.slider("Epsilon:", 1e-8, 1e-3, 1e-8, format="%.1e")
                            elif solver == "sgd":
                                momentum = st.slider("Momentum:", 0.0, 0.9, 0.9)
                                power_t = st.slider("Power t:", 0.1, 0.9, 0.5)
                                
                        with col5:
                            learning_rate_init = st.slider("Initial learning rate:", 0.0001, 0.001, 0.0003, format="%.4f")
                            learning_rate = st.selectbox("Learning rate schedule:", ["constant", "invscaling", "adaptive"])
                            
                        col6, col7 = st.columns(2)
                        with col6:
                            alpha = st.slider("L2 regularization (alpha):", 0.00001, 0.1, 0.0001, format="%.5f")
                            batch_size = st.selectbox("Batch size:", ["auto", 16, 32, 64, 128, 256])
                            if batch_size == "auto":
                                actual_batch_size = min(200, len(st.session_state.X_train))
                            else:
                                actual_batch_size = batch_size
                                
                        with col7:
                            max_iter = st.slider("Maximum iterations:", 100, 2000, 200)
                            tol = st.slider("Tolerance:", 1e-6, 1e-2, 1e-4, format="%.1e")
                    
                    # Create comprehensive parameters
                    mlp_params = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'activation': activation,
                        'solver': solver,
                        'alpha': alpha,
                        'learning_rate_init': learning_rate_init,
                        'learning_rate': learning_rate,
                        'max_iter': max_iter,
                        'tol': tol,
                        'batch_size': actual_batch_size,
                        'random_state': 42
                    }
                    
                    # Add solver-specific parameters
                    if solver == "adam":
                        mlp_params.update({
                            'beta_1': beta_1,
                            'beta_2': beta_2,
                            'epsilon': epsilon
                        })
                    elif solver == "sgd":
                        mlp_params.update({
                            'momentum': momentum,
                            'power_t': power_t if learning_rate == "invscaling" else 0.5
                        })
                    
                    base_model = MLPClassifier()
                    
                    if use_grid_search:
                        param_grid = {
                            'hidden_layer_sizes': [
                                (50,), (100,), (200,),
                                (50, 50), (100, 50), (100, 100),
                                (100, 50, 25)
                            ],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam', 'sgd'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'learning_rate_init': [0.001, 0.01, 0.1],
                            'max_iter': [200, 500, 1000]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = MLPClassifier(**mlp_params)
                                           
                    if use_grid_search:
                        param_grid = {
                            'hidden_layer_sizes': [(100,), (100,50), (50,50,50)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'max_iter': [200, 500, 1000]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

            else:  # Regression
                # Regular regression models (non-time series)
                model_type = st.selectbox("Pilih model regresi:" if st.session_state.language == 'id' else "Select a regression model:", 
                                         ["Random Forest", "Linear Regression", "Gradient Boosting", "SVR", "Bagging Regressor", "Voting Regressor", "Stacking Regressor", "KNN Regressor", "MLP Regressor"])
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Jumlah pepohonan:" if st.session_state.language == 'id' else "Number of Trees:", 10, 500, 100)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                    
                    base_model = RandomForestRegressor(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            random_state=42
                        )
                        
                elif model_type == "Gradient Boosting":
                    n_estimators = st.slider("Jumlah boosting stages:" if st.session_state.language == 'id' else "Number of boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                    
                    base_model = GradientBoostingRegressor(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [2, 3, 5] if max_depth == 3 else [max(1, max_depth-1), max_depth, min(10, max_depth+2)],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                        
                elif model_type == "Linear Regression":
                    fit_intercept = st.checkbox("Fit intercept" if st.session_state.language == 'id' else "Fit intercept", value=True)
                    
                    base_model = LinearRegression()
                    
                    if use_grid_search:
                        param_grid = {
                            'fit_intercept': [True, False]
                            # 'normalize' parameter removed to avoid error
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = LinearRegression(
                            fit_intercept=fit_intercept
                        )
                        
                elif model_type == "SVR":
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                    epsilon = st.slider("Epsilon:" if st.session_state.language == 'id' else "Epsilon:", 0.01, 0.5, 0.1)

                    base_model = SVR()

                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                            'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                            'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                            'epsilon': [epsilon]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = SVR(
                            C=C,
                            kernel=kernel,
                            gamma=gamma,
                            epsilon=epsilon
                        )

                elif model_type == "Voting Regressor":
                    from sklearn.ensemble import VotingRegressor
                    from sklearn.neighbors import KNeighborsRegressor
                    # Pilih base estimators untuk VotingRegressor
                    base_estimators = []
                    if st.checkbox("Gunakan Random Forest", value=True, key="vote_rf"):
                        base_estimators.append(('rf', RandomForestRegressor(n_estimators=50, random_state=42)))
                    if st.checkbox("Gunakan Linear Regression", value=True, key="vote_lr"):
                        base_estimators.append(('lr', LinearRegression()))
                    if st.checkbox("Gunakan Gradient Boosting", value=False, key="vote_gb"):
                        base_estimators.append(('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)))
                    if st.checkbox("Gunakan KNN Regressor", value=False, key="vote_knn"):
                        base_estimators.append(('knn', KNeighborsRegressor()))
                    if len(base_estimators) < 2:
                        st.warning("Pilih minimal dua base estimator untuk Voting Regressor." if st.session_state.language == 'id' else "Select at least two base estimators for Voting Regressor.")
                        model = None
                    else:
                        model = VotingRegressor(estimators=base_estimators)
                        
                elif model_type == "Stacking Regressor":
                    # Simple stacking with 2-3 base models and a final regressor
                    base_estimators = []
                    if st.checkbox("Gunakan Random Forest (Stacking)" if st.session_state.language == 'id' else "Use Random Forest (Stacking)", value=True, key="stack_rf"):
                        base_estimators.append(('rf', RandomForestRegressor(n_estimators=50, random_state=42)))
                    if st.checkbox("Gunakan Linear Regression (Stacking)" if st.session_state.language == 'id' else "Use Linear Regression (Stacking)", value=True, key="stack_lr"):
                        base_estimators.append(('lr', LinearRegression()))
                    if st.checkbox("Gunakan Gradient Boosting (Stacking)" if st.session_state.language == 'id' else "Use Gradient Boosting (Stacking)", value=False, key="stack_gb"):
                        base_estimators.append(('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)))
                    final_estimator = st.selectbox("Final estimator:" if st.session_state.language == 'id' else "Final estimator:", ["Linear Regression", "Random Forest"], key="stack_final")
                    if final_estimator == "Linear Regression":
                        final = LinearRegression()
                    else:
                        final = RandomForestRegressor(n_estimators=20, random_state=42)
                    if len(base_estimators) < 2:
                        st.warning("Pilih minimal dua base estimator untuk Stacking Regressor." if st.session_state.language == 'id' else "Select at least two base estimators for Stacking Regressor.")
                        model = None
                    else:
                        model = StackingRegressor(
                            estimators=base_estimators,
                            final_estimator=final,
                            passthrough=True
                        )
                elif model_type == "KNN Regressor":
                    from sklearn.neighbors import KNeighborsRegressor
                    n_neighbors = st.slider("Number of neighbors (K):" if st.session_state.language == 'id' else "Jumlah tetangga (K):", 1, 20, 5)
                    weights = st.selectbox("Weight function:" if st.session_state.language == 'id' else "Fungsi bobot:", ["uniform", "distance"])
                    algorithm = st.selectbox("Algorithm:" if st.session_state.language == 'id' else "Algoritma:", ["auto", "ball_tree", "kd_tree", "brute"])
                    base_model = KNeighborsRegressor()
                    if use_grid_search:
                        param_grid = {
                            'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2]  # Manhattan or Euclidean distance
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = KNeighborsRegressor(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm
                        )
                elif model_type == "MLP Regressor":
                    st.subheader("Konfigurasi Neural Network Regresi Lengkap" if st.session_state.language == 'id' else "Complete Neural Network Regression Configuration")
                    
                    # Tambahkan penjelasan teori di bagian atas
                    with st.expander("📚 Penjelasan Teori Neural Network" if st.session_state.language == 'id' else "📚 Neural Network Theory Explanation"):
                        st.markdown("""
                        ### 🧠 **Fungsi Aktivasi**
                        - **ReLU**: f(x) = max(0,x) - Cocok untuk hidden layers, mengatasi vanishing gradient
                        - **Sigmoid**: f(x) = 1/(1+e^(-x)) - Cocok untuk output binary classification
                        - **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) - Range [-1,1], lebih stabil dari sigmoid
                        - **Identity**: f(x) = x - Untuk output regression
                        
                        ### 🏗️ **Arsitektur Jaringan**
                        - **Feedforward Neural Network (FNN)**: Informasi mengalir satu arah (input → hidden → output)
                        - **Parameter yang dikonfigurasi**: Hidden layers, neurons per layer, activation function
                        
                        ### ⚡ **Optimizer & Learning**
                        - **Adam**: Kombinasi momentum dan adaptive learning rate, efisien untuk data besar
                        - **SGD**: Stochastic Gradient Descent dengan momentum untuk konvergensi lebih stabil
                        - **L-BFGS**: Optimizer kuasi-Newton untuk dataset kecil/medium
                        
                        ### 🛡️ **Regularization & Overfitting Prevention**
                        - **L2 Regularization (alpha)**: Menghindari overfitting dengan menjaga bobot tetap kecil
                        - **Early Stopping**: Menghentikan training saat validasi tidak membaik
                        - **Dropout**: (Tidak tersedia di MLPClassifier scikit-learn)
                        
                        ### 📊 **Hyperparameter Penting**
                        - **Learning Rate**: Kontrol kecepatan pembelajaran (0.001-0.1)
                        - **Batch Size**: Jumlah sampel per update (16-512)
                        - **Epochs**: Jumlah iterasi seluruh dataset (max_iter)
                        - **Hidden Layers**: Kompleksitas model (1-5 layers)
                        """)
                    
                    # Architecture Configuration
                    st.write("**Arsitektur Jaringan Regresi:**" if st.session_state.language == 'id' else "**Regression Network Architecture:**")
                    
                    # Hidden layers configuration
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 5, 2)
                    with col2:
                        neurons_per_layer = st.text_input("Neurons per layer:", "100,50")
                        try:
                            neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                            if len(neurons_list) < num_hidden_layers:
                                neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                            elif len(neurons_list) > num_hidden_layers:
                                neurons_list = neurons_list[:num_hidden_layers]
                            hidden_layer_sizes = tuple(neurons_list)
                        except:
                            hidden_layer_sizes = (100, 50)
                    with col3:
                        activation = st.selectbox("Activation function:", 
                                                ["relu", "tanh", "logistic", "identity"],
                                                help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                    
                    # Advanced parameters
                    with st.expander("Advanced Parameters"):
                        col4, col5 = st.columns(2)
                        with col4:
                            solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"])
                            
                            if solver == "adam":
                                beta_1 = st.slider("Beta 1:", 0.8, 0.999, 0.9, format="%.3f")
                                beta_2 = st.slider("Beta 2:", 0.9, 0.9999, 0.999, format="%.4f")
                                epsilon = st.slider("Epsilon:", 1e-8, 1e-3, 1e-8, format="%.1e")
                            elif solver == "sgd":
                                momentum = st.slider("Momentum:", 0.0, 0.9, 0.9)
                                power_t = st.slider("Power t:", 0.1, 0.9, 0.5)
                                
                        with col5:
                            learning_rate_init = st.slider("Initial learning rate:", 0.0001, 0.001, 0.0003, format="%.4f")
                            learning_rate = st.selectbox("Learning rate schedule:", ["constant", "invscaling", "adaptive"])
                            
                        col6, col7 = st.columns(2)
                        with col6:
                            alpha = st.slider("L2 regularization (alpha):", 0.00001, 0.1, 0.0001, format="%.5f")
                            batch_size = st.selectbox("Batch size:", ["auto", 16, 32, 64, 128, 256])
                            if batch_size == "auto":
                                actual_batch_size = min(200, len(st.session_state.X_train))
                            else:
                                actual_batch_size = batch_size
                                
                        with col7:
                            max_iter = st.slider("Maximum iterations:", 100, 2000, 200)
                            tol = st.slider("Tolerance:", 1e-6, 1e-2, 1e-4, format="%.1e")
                    
                    # Create comprehensive parameters
                    mlp_params = {
                        'hidden_layer_sizes': hidden_layer_sizes,
                        'activation': activation,
                        'solver': solver,
                        'alpha': alpha,
                        'learning_rate_init': learning_rate_init,
                        'learning_rate': learning_rate,
                        'max_iter': max_iter,
                        'tol': tol,
                        'batch_size': actual_batch_size,
                        'random_state': 42
                    }
                    
                    # Add solver-specific parameters
                    if solver == "adam":
                        mlp_params.update({
                            'beta_1': beta_1,
                            'beta_2': beta_2,
                            'epsilon': epsilon
                        })
                    elif solver == "sgd":
                        mlp_params.update({
                            'momentum': momentum,
                            'power_t': power_t if learning_rate == "invscaling" else 0.5
                        })
                    
                    base_model = MLPRegressor()
                    
                    if use_grid_search:
                        param_grid = {
                            'hidden_layer_sizes': [
                                (50,), (100,), (200,),
                                (50, 50), (100, 50), (100, 100),
                                (100, 50, 25)
                            ],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam', 'sgd'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'learning_rate_init': [0.001, 0.01, 0.1],
                            'max_iter': [200, 500, 1000]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = MLPRegressor(**mlp_params)
                else:
                    st.error("Silahkan pilih model regresi." if st.session_state.language == 'id' else "Please select a valid regression model.")
                    model = None
            
            model_custom_name = st.text_input("Nama model (bebas, gunakan huruf/angka/underscore):" if st.session_state.language == 'id' else "Nama model (bebas, gunakan huruf/angka/underscore):", value=f"")
            st.session_state.model_type = model_type

            # Train model button
            if model is not None and st.button("Train Model"):
                with st.spinner(f"Melatih model {model_type}..." if st.session_state.language == 'id' else f"Training {model_type} model..."):
                    try:
                        start_time = time.time()
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        training_time = time.time() - start_time
                        
                        # Tambahkan validasi sebelum prediksi
                        try:
                            # Test prediksi dengan data dummy untuk memastikan model berfungsi
                            dummy_data = pd.DataFrame(np.zeros((1, len(st.session_state.X_train.columns))), 
                                                    columns=st.session_state.X_train.columns)
                            model.predict(dummy_data)  # Use local 'model' instead of st.session_state.model
                            st.success("Model siap digunakan untuk prediksi" if st.session_state.language == 'id' else "Model ready for prediction")
                        except Exception as e:
                            st.error(f"Model error: {str(e)}. Silakan latih ulang model." if st.session_state.language == 'id' else f"Model error: {str(e)}. Please retrain the model.")

                        # Jika menggunakan GridSearchCV, tampilkan parameter terbaik
                        if use_grid_search and hasattr(model, "best_params_"):
                            st.success(f"Pelatihan model selesai dalam {training_time:.2f} detik dengan GridSearchCV. Parameter terbaik: {model.best_params_}" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds with GridSearchCV!")
                            st.subheader("Parameter Terbaik" if st.session_state.language == 'id' else "Best Parameters:")
                            st.write(model.best_params_)
                            st.write(f"Skor terbaik (CV): {model.best_score_:.4f}" if st.session_state.language == 'id' else f"Best Score (CV): {model.best_score_:.4f}")

                            # Gunakan model terbaik untuk prediksi
                            y_pred = model.best_estimator_.predict(st.session_state.X_test)
                            st.session_state.model = model.best_estimator_
                        else:
                            st.success(f"Model selesai dilatih dalam {training_time:.2f} detik" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds!")
                            y_pred = model.predict(st.session_state.X_test)
                            st.session_state.model = model
                        
                        # Save model dengan nama custom
                        os.makedirs("models", exist_ok=True)
                        # Bersihkan nama agar hanya huruf/angka/underscore
                        safe_name = "".join([c if c.isalnum() or c == "_" else "_" for c in model_custom_name])
                        model_filename = f"models/{safe_name}.pkl"
                        with open(model_filename, 'wb') as f:
                            pickle.dump(st.session_state.model, f)
                        st.success(f"Model telah disimpan sebagai '{model_filename}'" if st.session_state.language == 'id' else "Model saved as '{model_filename}'")
                        
                        # Evaluasi model
                        if problem_type == "Classification":
                            accuracy = accuracy_score(st.session_state.y_test, y_pred)
                            st.write(f"Accuracy: {accuracy:.4f}")
                            
                            # Confusion Matrix
                            cm = confusion_matrix(st.session_state.y_test, y_pred)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Classification Report
                            report = classification_report(st.session_state.y_test, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.write("Label Report" if st.session_state.language == 'id' else "Classification Report:")
                            st.dataframe(report_df)
                            
                            # ROC Curve dan AUC Score
                            st.subheader("ROC Curve dan AUC Score" if st.session_state.language == 'id' else "ROC Curve and AUC Score")
                            
                            # Cek apakah model mendukung predict_proba
                            if hasattr(model, 'predict_proba'):
                                # Untuk klasifikasi biner
                                if len(np.unique(st.session_state.y_test)) == 2:
                                    y_prob = model.predict_proba(st.session_state.X_test)[:, 1]
                                    # Menangani kasus ketika y_test berisi nilai kategorikal seperti '<20', '>20'
                                    if isinstance(st.session_state.y_test.iloc[0], str):
                                        # Konversi nilai kategorikal ke numerik (0 dan 1)
                                        unique_values = sorted(np.unique(st.session_state.y_test))
                                        pos_label = unique_values[1]  # Nilai kedua sebagai pos_label
                                        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_prob, pos_label=pos_label)
                                    else:
                                        fpr, tpr, thresholds = roc_curve(st.session_state.y_test, y_prob)
                                    roc_auc = auc(fpr, tpr)
                                    
                                    # Plot ROC Curve
                                    fig, ax = plt.subplots(figsize=(10, 8))
                                    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                                    ax.plot([0, 1], [0, 1], 'k--')
                                    ax.set_xlim([0.0, 1.0])
                                    ax.set_ylim([0.0, 1.05])
                                    ax.set_xlabel('False Positive Rate')
                                    ax.set_ylabel('True Positive Rate')
                                    ax.set_title('Receiver Operating Characteristic (ROC)')
                                    ax.legend(loc="lower right")
                                    st.pyplot(fig)
                                    
                                    st.write(f"AUC Score: {roc_auc:.4f}")
                                
                                # Untuk klasifikasi multi-kelas
                                else:
                                    try:
                                        y_prob = model.predict_proba(st.session_state.X_test)
                                        
                                        # Buat label biner untuk setiap kelas
                                        y_test_bin = pd.get_dummies(st.session_state.y_test).values
                                        
                                        # Pastikan jumlah kelas dalam y_prob dan y_test_bin sama
                                        n_classes = min(y_prob.shape[1], y_test_bin.shape[1])
                                        
                                        if n_classes > 0:
                                            # One-vs-Rest ROC
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            
                                            for i in range(n_classes):
                                                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                                                roc_auc = auc(fpr, tpr)
                                                ax.plot(fpr, tpr, label=f'Class {i} (area = {roc_auc:.2f})')
                                            
                                            ax.plot([0, 1], [0, 1], 'k--')
                                            ax.set_xlim([0.0, 1.0])
                                            ax.set_ylim([0.0, 1.05])
                                            ax.set_xlabel('False Positive Rate')
                                            ax.set_ylabel('True Positive Rate')
                                            ax.set_title('Multi-class ROC Curve (One-vs-Rest)')
                                            ax.legend(loc="lower right")
                                            st.pyplot(fig)
                                            
                                            # Hitung dan tampilkan AUC Score untuk setiap kelas
                                            st.write("AUC Scores per class:")
                                            for i in range(n_classes):
                                                class_auc = roc_auc_score(y_test_bin[:, i], y_prob[:, i])
                                                st.write(f"Class {i}: {class_auc:.4f}")
                                            
                                            # Hitung rata-rata AUC (macro) hanya jika jumlah kelas sama
                                            if y_prob.shape[1] == y_test_bin.shape[1]:
                                                macro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
                                                st.write(f"Macro Average AUC: {macro_auc:.4f}")
                                            else:
                                                st.warning("Tidak dapat menghitung Macro Average AUC karena jumlah kelas berbeda antara prediksi dan aktual." 
                                                          if st.session_state.language == 'id' else 
                                                          "Cannot calculate Macro Average AUC because the number of classes differs between prediction and actual.")
                                        else:
                                            st.warning("Tidak ada kelas yang dapat digunakan untuk kurva ROC." 
                                                      if st.session_state.language == 'id' else 
                                                      "No classes available for ROC curve.")
                                    except Exception as e:
                                        st.error(f"Error saat membuat kurva ROC: {str(e)}" 
                                                if st.session_state.language == 'id' else 
                                                f"Error creating ROC curve: {str(e)}")
                                        st.warning("Pastikan data pengujian memiliki semua kelas yang ada dalam data pelatihan." 
                                                  if st.session_state.language == 'id' else 
                                                  "Make sure the test data contains all classes present in the training data.")
                                        # Tampilkan informasi tambahan untuk debugging
                                        st.write(f"Jumlah kelas unik dalam y_test: {len(np.unique(st.session_state.y_test))}")
                                        if hasattr(model, 'classes_'):
                                            st.write(f"Kelas dalam model: {model.classes_}")
                                            st.write(f"Jumlah kelas dalam model: {len(model.classes_)}")
                                        if 'y_prob' in locals():
                                            st.write(f"Dimensi y_prob: {y_prob.shape}")
                                        if 'y_test_bin' in locals():
                                            st.write(f"Dimensi y_test_bin: {y_test_bin.shape}")
                            else:
                                st.warning("Model ini tidak mendukung prediksi probabilitas, sehingga kurva ROC tidak dapat ditampilkan." 
                                          if st.session_state.language == 'id' else 
                                          "This model doesn't support probability prediction, so ROC curve cannot be displayed.")
                            
                        else:  # Regression
                            mse = mean_squared_error(st.session_state.y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(st.session_state.y_test, y_pred)
                            # Tambahan: Adjusted R²
                            n = st.session_state.X_test.shape[0]
                            k = st.session_state.X_test.shape[1]
                            adj_r2 = adjusted_r2_score(r2, n, k)
                            st.write(f"Mean Squared Error: {mse:.4f}")
                            st.write(f"Root Mean Squared Error: {rmse:.4f}")
                            st.write(f"R² Score: {r2:.4f}")
                            st.write(f"Adjusted R² Score: {adj_r2:.4f}")

                            # Tambahan: Uji Multikolinearitas (VIF) - hanya untuk Linear Regression
                            if st.session_state.model_type == "Linear Regression":
                                st.subheader("Uji Multikolinearitas (VIF)" if st.session_state.language == 'id' else "Multicollinearity Test (VIF)")
                                vif_df = calculate_vif(st.session_state.X_train)
                                st.dataframe(vif_df)

                                # Tambahan: Uji Heteroskedastisitas (Breusch-Pagan) - hanya untuk Linear Regression
                                st.subheader("Uji Heteroskedastisitas (Breusch-Pagan)" if st.session_state.language == 'id' else "Heteroskedasticity Test (Breusch-Pagan)")
                                bp_result = breusch_pagan_test(st.session_state.y_test, y_pred, st.session_state.X_test)
                                st.write(f"Lagrange multiplier statistic: {bp_result['Lagrange multiplier statistic']:.4f}")
                                st.write(f"p-value: {bp_result['p-value']:.4f}")
                                st.write(f"f-value: {bp_result['f-value']:.4f}")
                                st.write(f"f p-value: {bp_result['f p-value']:.4f}")
                                
                                # Add assumptions check for linear regression
                                st.subheader("Asumsi Regresi Linear" if st.session_state.language == 'id' else "Linear Regression Assumptions")
                                
                                # Check VIF values for multicollinearity
                                high_vif = vif_df[vif_df['VIF'] > 10]
                                if len(high_vif) > 0:
                                    st.warning(f"⚠️ Multikolinearitas terdeteksi! {len(high_vif)} fitur memiliki VIF > 10" 
                                            if st.session_state.language == 'id' else 
                                            f"⚠️ Multicollinearity detected! {len(high_vif)} features have VIF > 10")
                                    st.dataframe(high_vif)
                                else:
                                    st.success("✅ Tidak ada multikolinearitas yang signifikan (semua VIF ≤ 10)" 
                                            if st.session_state.language == 'id' else 
                                            "✅ No significant multicollinearity detected (all VIF ≤ 10)")
                                
                                # Check heteroskedasticity
                                if bp_result['p-value'] < 0.05:
                                    st.warning("⚠️ Heteroskedastisitas terdeteksi (p-value < 0.05)" 
                                            if st.session_state.language == 'id' else 
                                            "⚠️ Heteroskedasticity detected (p-value < 0.05)")
                                else:
                                    st.success("✅ Tidak ada heteroskedastisitas yang signifikan (p-value ≥ 0.05)" 
                                            if st.session_state.language == 'id' else 
                                            "✅ No significant heteroskedasticity detected (p-value ≥ 0.05)")
                            
                            # Plot actual vs predicted
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(st.session_state.y_test, y_pred, alpha=0.5)
                            ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], 
                                   [st.session_state.y_test.min(), st.session_state.y_test.max()], 
                                   'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            st.pyplot(fig)
                            
                            # Residual plot
                            residuals = st.session_state.y_test - y_pred
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_pred, residuals, alpha=0.5)
                            ax.axhline(y=0, color='r', linestyle='--')
                            plt.title('Residual Plot')
                            plt.xlabel('Predicted')
                            plt.ylabel('Residuals')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"Error during model training: {str(e)}") 

            # Tambahkan bagian untuk prediksi data baru
            if st.session_state.model is not None:
                st.subheader("Prediksi Data Baru" if st.session_state.language == 'id' else "Predict New Data")
                
                # Import library untuk PDF
                from fpdf import FPDF
                from datetime import datetime
                import json

                # Fungsi untuk membuat laporan PDF
                def create_prediction_report(input_data, predictions, model_info, problem_type):
                    pdf = FPDF()
                    pdf.add_page()
                    
                    # Header
                    pdf.set_font('Arial', 'B', 16)
                    pdf.cell(0, 10, 'Laporan Hasil Prediksi' if st.session_state.language == 'id' else 'Prediction Report', 0, 1, 'C')
                    pdf.ln(10)
                    
                    # Informasi Umum
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Informasi Umum:' if st.session_state.language == 'id' else 'General Information:', 0, 1)
                    pdf.set_font('Arial', '', 12)

                    date_format = "%Y-%m-%d %H:%M:%S"
                    pdf.cell(0, 10, f'Tanggal: {datetime.now().strftime(date_format)}' if st.session_state.language == 'id' else f'Date: {datetime.now().strftime(date_format)}', 0, 1)
                    pdf.cell(0, 10, f'Jenis Model: {type(st.session_state.model).__name__}' if st.session_state.language == 'id' else f'Model Type: {type(st.session_state.model).__name__}', 0, 1)
                    pdf.cell(0, 10, f'Metode: {problem_type}' if st.session_state.language == 'id' else f'Method: {problem_type}', 0, 1)
                    
                    # Parameter Model
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Parameter Model:' if st.session_state.language == 'id' else 'Model Parameters:', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    try:
                        for param, value in st.session_state.model.get_params().items():
                            # Konversi value ke string dan batasi panjangnya
                            value_str = str(value)
                            if len(value_str) > 50:  # Batasi panjang nilai
                                value_str = value_str[:47] + '...'
                            pdf.multi_cell(0, 10, f'{param}: {value_str}')
                    except Exception as e:
                        pdf.cell(0, 10, 'Parameter model tidak tersedia' if st.session_state.language == 'id' else 'Model parameters not available', 0, 1)
                    
                    # Metrik Evaluasi
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Metrik Evaluasi Model:' if st.session_state.language == 'id' else 'Model Evaluation Metrics:', 0, 1)
                    pdf.set_font('Arial', '', 12)
                    
                    # Tambahkan perhitungan metrik evaluasi
                    if problem_type == "Regression" and hasattr(st.session_state, 'y_test'):
                        y_pred = st.session_state.model.predict(st.session_state.X_test)
                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(st.session_state.y_test, y_pred)
                        
                        pdf.cell(0, 10, f'Mean Squared Error (MSE): {mse:.4f}', 0, 1)
                        pdf.cell(0, 10, f'Root Mean Squared Error (RMSE): {rmse:.4f}', 0, 1)
                        pdf.cell(0, 10, f'R² Score: {r2:.4f}', 0, 1)
                    else:
                        pdf.cell(0, 10, 'Metrik evaluasi tidak tersedia' if st.session_state.language == 'id' else 'Evaluation metrics not available', 0, 1)
                    
                    # Hasil Prediksi
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Hasil Prediksi:' if st.session_state.language == 'id' else 'Prediction Results:', 0, 1)
                    pdf.set_font('Arial', '', 10)
                    
                    # Tabel hasil prediksi
                    # Hitung lebar kolom yang sesuai
                    n_columns = len(input_data.columns) + 1  # +1 untuk kolom prediksi
                    col_width = min(pdf.w / n_columns - 2, 35)  # Maksimal 35 pt per kolom
                    row_height = 8
                    
                    # Header tabel
                    pdf.set_font('Arial', 'B', 10)
                    for col in input_data.columns:
                        pdf.cell(col_width, row_height, str(col)[:15], 1)
                    pdf.cell(col_width, row_height, 'Prediksi' if st.session_state.language == 'id' else 'Prediction', 1)
                    pdf.ln()
                    
                    # Isi tabel
                    pdf.set_font('Arial', '', 10)
                    for i in range(len(input_data)):
                        if i > 0 and i % 40 == 0:  # Tambah halaman baru setiap 40 baris
                            pdf.add_page()
                            # Cetak header lagi
                            pdf.set_font('Arial', 'B', 10)
                            for col in input_data.columns:
                                pdf.cell(col_width, row_height, str(col)[:15], 1)
                            pdf.cell(col_width, row_height, 'Prediksi' if st.session_state.language == 'id' else 'Prediction', 1)
                            pdf.ln()
                            pdf.set_font('Arial', '', 10)
                        
                        for col in input_data.columns:
                            value = input_data.iloc[i][col]
                            if isinstance(value, (int, float)):
                                value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
                            else:
                                value_str = str(value)
                            pdf.cell(col_width, row_height, value_str[:15], 1)
                        
                        pred_value = predictions[i] if isinstance(predictions, (list, np.ndarray)) else predictions
                        if isinstance(pred_value, (int, float)):
                            pred_str = f"{pred_value:.2f}" if isinstance(pred_value, float) else str(pred_value)
                        else:
                            pred_str = str(pred_value)
                        pdf.cell(col_width, row_height, pred_str[:15], 1)
                        pdf.ln()
                    
                    # Penanggung Jawab
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, 'Penanggung Jawab:' if st.session_state.language == 'id' else 'Responsible:', 0, 0, 1)
                    pdf.set_font('Arial', '', 12)
                    pdf.cell(0, 10, 'Nama: ____________________' if st.session_state.language == 'id' else 'Name: ____________________', 0, 1)
                    pdf.cell(0, 10, 'Jabatan: ____________________' if st.session_state.language == 'id' else 'Position: ____________________', 0, 1)
                    pdf.cell(0, 10, 'Tanggal: ____________________' if st.session_state.language == 'id' else 'Date: ____________________', 0, 1)
                    pdf.cell(0, 20, 'Tanda Tangan:'if st.session_state.language == 'id' else 'Signature:', 0, 1)
                    pdf.cell(0, 20, '_____________________', 0, 1)
                    
                    return pdf
                
                # Pilih metode input data
                input_method = st.radio("Pilih metode input data:" if st.session_state.language == 'id' else "Select input method:", ["Input Manual", "Upload CSV"])
                
                if input_method == "Input Manual":
                    # Buat form input untuk setiap fitur
                    st.write("Masukkan nilai untuk setiap fitur:" if st.session_state.language == 'id' else "Enter values for each feature:")
                    
                    input_data = {}
                    
                    for feature in st.session_state.X_train.columns:
                        # Cek apakah fitur adalah kategorikal atau numerikal
                        if feature in st.session_state.categorical_columns:
                            # Jika ada encoder untuk fitur ini, tampilkan opsi yang tersedia
                            if feature in st.session_state.encoders:
                                options = list(st.session_state.encoders[feature].classes_)
                                input_data[feature] = st.selectbox(f"{feature}:", options)
                            else:
                                input_data[feature] = st.text_input(f"{feature}:")
                        else:
                            # Untuk fitur numerikal, gunakan number_input
                            input_data[feature] = st.number_input(f"{feature}:", format="%.4f")
                    
                    if st.button("Prediksi"):
                        try:
                            # Konversi input menjadi DataFrame
                            input_df = pd.DataFrame([input_data])
                            
                            # Terapkan preprocessing yang sama seperti data training
                            # Encoding untuk fitur kategorikal
                            for col in [c for c in input_df.columns if c in st.session_state.categorical_columns]:
                                if col in st.session_state.encoders:
                                    input_df[col] = st.session_state.encoders[col].transform(input_df[col].astype(str))
                            
                            # Scaling untuk fitur numerikal
                            num_cols = [c for c in input_df.columns if c in st.session_state.numerical_columns]
                            if st.session_state.scaler is not None and num_cols:
                                input_df[num_cols] = st.session_state.scaler.transform(input_df[num_cols])

                            # Pastikan urutan kolom sama dengan saat training
                            input_df = input_df[st.session_state.X_train.columns]

                            # Lakukan prediksi
                            prediction = st.session_state.model.predict(input_df)
                            
                            # Tentukan jenis model
                            model_type = get_model_type(st.session_state.model)
                            
                            # Tampilkan hasil prediksi
                            st.subheader("Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result")
                            
                            if model_type == "Classification":
                                st.write(f"Kelas yang diprediksi: {prediction[0]}" if st.session_state.language == 'id' else f"Predicted Class: {prediction[0]}")
                                
                                # Jika model memiliki predict_proba, tampilkan probabilitas
                                if hasattr(st.session_state.model, 'predict_proba'):
                                    try:
                                        proba = st.session_state.model.predict_proba(input_df)
                                        proba_df = pd.DataFrame(proba, columns=st.session_state.model.classes_)
                                        st.write("Probabilitas untuk setiap kelas:" if st.session_state.language == 'id' else "Probabilities for each class:")
                                        st.dataframe(proba_df)
                                    except Exception as e:
                                        st.warning(f"Tidak dapat menghitung probabilitas: {str(e)}" if st.session_state.language == 'id' else f"Cannot calculate probabilities: {str(e)}")
                            else:  # Regression
                                st.write(f"Nilai yang diprediksi: {prediction[0]:.4f}" if st.session_state.language == 'id' else f"Predicted Value: {prediction[0]:.4f}")
                            
                            # Buat laporan PDF
                            try:
                                model_type = get_model_type(st.session_state.model)
                                pdf = create_prediction_report(input_df, prediction, st.session_state.model, model_type)
                                pdf_output = pdf.output(dest='S').encode('latin1')
                                st.download_button(
                                    label="Download Laporan PDF" if st.session_state.language == 'id' else "Download PDF Report",
                                    data=pdf_output,
                                    file_name=f"prediction_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                            except Exception as e:
                                st.error(f"Error saat membuat laporan PDF: {str(e)}" if st.session_state.language == 'id' else f"Error creating PDF report: {str(e)}")
                            
                        except Exception as e:
                            st.error(f"Error saat melakukan prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error during prediction: {str(e)}")
                
                else:  # Upload CSV
                    st.write("Upload file CSV dengan data yang ingin diprediksi:" if st.session_state.language == 'id' else "Upload CSV file with data to predict:")
                    uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="prediction_file")
                    
                    if uploaded_file is not None:
                        
                            # Baca file CSV
                            pred_data = pd.read_csv(uploaded_file)
                            
                            # Tampilkan preview data
                            st.write("Data Preview:" if st.session_state.language == 'id' else "Preview data:" )
                            st.dataframe(pred_data.head())
                            
                            # Periksa apakah semua fitur yang diperlukan ada
                            missing_features = [f for f in st.session_state.X_train.columns if f not in pred_data.columns]
                            
                            if missing_features:
                                st.error(f"Data tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}" if st.session_state.language == 'id' else f"Data is missing required features: {', '.join(missing_features)}")
                            
                            # Add detailed debugging information
                            st.write("**🔍 Informasi Debug Detail:**")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Fitur yang diharapkan model:**")
                                expected_features = list(st.session_state.X_train.columns)
                                st.write(expected_features)
                                
                            with col2:
                                st.write("**Fitur yang tersedia dalam data prediksi:**")
                                available_features = list(pred_data.columns)
                                st.write(available_features)
                            
                            # Show comparison table
                            comparison_df = pd.DataFrame({
                                'Expected Features': expected_features,
                                'Available in CSV': ['✅ Ada' if f in pred_data.columns else '❌ Tidak Ada' for f in expected_features]
                            })
                            st.write("**Perbandingan Fitur:**")
                            st.dataframe(comparison_df)
                            
                            # Provide guidance
                            st.info("""**Cara memperbaiki:**
                            1. Pastikan file CSV prediksi memiliki semua kolom yang digunakan saat training
                            2. Periksa nama kolom (case-sensitive)
                            3. Kolom yang hilang harus ditambahkan ke file CSV prediksi
                            4. Jika kolom tidak tersedia, pertimbangkan untuk melatih ulang model tanpa kolom tersebut""" if st.session_state.language == 'id' else 
                            """**How to fix:**
                            1. Ensure your prediction CSV has all columns used during training
                            2. Check column names (case-sensitive)
                            3. Missing columns must be added to the prediction CSV
                            4. If columns are unavailable, consider retraining the model without these columns""")
                        
                            if not st.session_state.model:
                                # Validasi fitur sebelum prediksi
                                st.write("**📊 Validasi Fitur:**")
                                
                                # Check for missing features
                                missing_features = [f for f in st.session_state.X_train.columns if f not in pred_data.columns]
                                
                                if missing_features:
                                    st.error(f"Data tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}")
                                    
                                    # Add detailed debugging information
                                    st.write("**🔍 Informasi Debug Detail:**")
                                    
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.write("**Fitur yang diharapkan model:**")
                                        expected_features = list(st.session_state.X_train.columns)
                                        st.write(expected_features)
                                        
                                    with col2:
                                        st.write("**Fitur yang tersedia dalam data prediksi:**")
                                        available_features = list(pred_data.columns)
                                        st.write(available_features)
                                    
                                    # Show comparison table
                                    comparison_df = pd.DataFrame({
                                        'Expected Features': expected_features,
                                        'Available in CSV': ['✅ Ada' if f in pred_data.columns else '❌ Tidak Ada' for f in expected_features]
                                    })
                                    st.write("**Perbandingan Fitur:**")
                                    st.dataframe(comparison_df)
                                    
                                    # Provide guidance
                                    st.info("""**Cara memperbaiki:**
                                    1. Pastikan file CSV prediksi memiliki semua kolom yang digunakan saat training
                                    2. Periksa nama kolom (case-sensitive)
                                    3. Kolom yang hilang harus ditambahkan ke file CSV prediksi
                                    4. Jika kolom tidak tersedia, pertimbangkan untuk melatih ulang model tanpa kolom tersebut""")
                                else:
                                    # Validasi tipe data
                                    type_issues = []
                                    for col in st.session_state.X_train.columns:
                                        if col in pred_data.columns:
                                            expected_dtype = st.session_state.X_train[col].dtype
                                            actual_dtype = pred_data[col].dtype
                                            if expected_dtype != actual_dtype:
                                                type_issues.append({
                                                    'Column': col,
                                                    'Expected Type': str(expected_dtype),
                                                    'Actual Type': str(actual_dtype)
                                                })

                                    if type_issues:
                                        st.warning("**Peringatan Tipe Data:** Beberapa kolom memiliki tipe data yang berbeda")
                                        st.dataframe(pd.DataFrame(type_issues))
                                        
                                        # Konversi otomatis tipe data
                                        for issue in type_issues:
                                            col = issue['Column']
                                            try:
                                                pred_data[col] = pred_data[col].astype(st.session_state.X_train[col].dtype)
                                                st.success(f"Berhasil mengkonversi {col} ke tipe data yang sesuai")
                                            except Exception as e:
                                                st.error(f"Gagal mengkonversi {col}: {str(e)}")
                                    
                                    # Lanjutkan dengan preprocessing
                                    pred_data = pred_data[st.session_state.X_train.columns]
                                    
                                    # Encoding untuk fitur kategorikal
                                    for col in [c for c in pred_data.columns if c in st.session_state.categorical_columns]:
                                        if col in st.session_state.encoders:
                                            try:
                                                pred_data[col] = st.session_state.encoders[col].transform(pred_data[col].astype(str))
                                            except ValueError as e:
                                                st.error(f"Error encoding {col}: {str(e)}")
                                                st.write(f"Nilai unik dalam data: {pred_data[col].unique()}")
                                                st.write(f"Nilai yang diharapkan encoder: {list(st.session_state.encoders[col].classes_)}")
                                    
                                    # Scaling untuk fitur numerikal
                                    num_cols = [c for c in pred_data.columns if c in st.session_state.numerical_columns]
                                    if st.session_state.scaler is not None and num_cols:
                                        pred_data[num_cols] = st.session_state.scaler.transform(pred_data[num_cols])
                                    
                                    if st.button("Prediksi Batch", key="batch_prediction_btn"):
                                        try:
                                            # Lakukan prediksi
                                            predictions = st.session_state.model.predict(pred_data)
                                            
                                            # Tentukan jenis model
                                            model_type = get_model_type(st.session_state.model)
                                            
                                            # Tambahkan hasil prediksi ke DataFrame
                                            result_df = pred_data.copy()
                                            
                                            if model_type == "Classification":
                                                result_df['Predicted_Class'] = predictions
                                                
                                                # Jika model memiliki predict_proba, tambahkan probabilitas untuk setiap kelas
                                                if hasattr(st.session_state.model, 'predict_proba'):
                                                    try:
                                                        proba = st.session_state.model.predict_proba(pred_data)
                                                        for i, class_name in enumerate(st.session_state.model.classes_):
                                                            result_df[f'Probability_{class_name}'] = proba[:, i]
                                                    except Exception as e:
                                                        st.warning(f"Tidak dapat menghitung probabilitas: {str(e)}" if st.session_state.language == 'id' else f"Cannot calculate probabilities: {str(e)}")
                                            else:  # Regression
                                                result_df['Predicted_Value'] = predictions
                                            
                                            # Tampilkan hasil
                                            st.subheader("Hasil Prediksi")
                                            st.dataframe(result_df)
                                            
                                            # Download hasil
                                            csv = result_df.to_csv(index=False)
                                            st.download_button(
                                                label="Download Hasil Prediksi (CSV)" if st.session_state.language == 'id' else "Download Prediction Results (CSV)",
                                                data=csv,
                                                file_name=f"prediction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                                mime="text/csv"
                                            )
                                            
                                        except Exception as e:
                                            st.error(f"Error saat melakukan prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error during prediction: {str(e)}")
                        
               
                # Tambahkan bagian untuk memuat model yang sudah disimpan
                st.subheader("Muat Model yang Sudah Disimpan" if st.session_state.language == 'id' else "Load Saved Model")
                
                # Cek apakah folder models ada
                if os.path.exists("models"):
                    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
                    
                    if model_files:
                        selected_model_file = st.selectbox("Pilih model yang akan dimuat:" if st.session_state.language == 'id' else "Select a model to load:", model_files)
                        
                        if st.button("Muat Model" if st.session_state.language == 'id' else "Load Model"):
                            try:
                                with open(os.path.join("models", selected_model_file), 'rb') as f:
                                    loaded_model = pickle.load(f)
                                
                                st.session_state.model = loaded_model
                                st.success(f"Model {selected_model_file} berhasil dimuat!" if st.session_state.language == 'id' else f"Model {selected_model_file} loaded successfully!")
                            except Exception as e:
                                st.error(f"Error saat memuat model: {str(e)}")
                    else:
                        st.info("Tidak ada model tersimpan di folder 'models'." if st.session_state.language == 'id' else "No saved models found in the 'models' folder.")
                else:
                    st.info("Folder 'models' belum dibuat. Latih dan simpan model terlebih dahulu." if st.session_state.language == 'id' else "Folder 'models' does not exist. Train and save models first.")
            else:
                st.info("Silakan latih model terlebih dahulu sebelum melakukan prediksi." if st.session_state.language == 'id' else "Please train a model first before making predictions.")
    else:
        st.info("Please complete the preprocessing steps in the previous tab first." if st.session_state.language == 'id' else "Please complete the preprocessing steps in the previous tab first.")

# Tab 5: SHAP Model Interpretation
with tab5:
    # Tambahkan fungsi untuk menentukan jenis model
    def get_model_type(model):
        """Menentukan jenis model (klasifikasi atau regresi) berdasarkan model yang dimuat"""
        try:
            # Cek berdasarkan jenis model
            if hasattr(model, 'predict_proba') and hasattr(model, 'classes_'):
                return 'Classification'
            elif hasattr(model, 'predict') and not hasattr(model, 'classes_'):
                return 'Regression'
            else:
                # Fallback ke problem_type dari session state
                return st.session_state.problem_type
        except:
            return st.session_state.problem_type
            
    if st.session_state.problem_type != 'Regression':
        st.info("Fitur interpretasi SHAP hanya tersedia untuk model regresi." if st.session_state.language == 'id' else "SHAP interpretation is only available for regression models.")
    else:
        st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

        if (
            st.session_state.model is not None
            and st.session_state.problem_type in ["Regression", "Classification"]
            and not ('is_timeseries' in locals() and is_timeseries)
        ):
            st.write("""
            SHAP (SHapley Additive exPlanations) adalah pendekatan teori permainan untuk menjelaskan output dari model machine learning mana pun.
            """ if st.session_state.language == 'id' else """
            SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model.
            """)
            
            # Pilih fitur untuk SHAP
            st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
            feature_names = st.session_state.X_train.columns.tolist()
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                options=feature_names,
                default=feature_names[:min(10, len(feature_names))]
            )
        
            # Jumlah sampel untuk analisis SHAP
            sample_size = st.slider(
                "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                min_value=10, max_value=min(100, len(st.session_state.X_test)), value=50
            )
            
            if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
                if not selected_features:
                    st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                else:
                    with st.spinner("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values..."):
                        # Persiapkan data untuk SHAP
                        X_sample = st.session_state.X_test[selected_features].sample(min(sample_size, len(st.session_state.X_test)), random_state=42)
                        
                        # Identifikasi fitur kategorikal dalam sampel
                        categorical_cols = [col for col in selected_features if col in st.session_state.categorical_columns]
                        
                        # Terapkan One-Hot Encoding jika ada fitur kategorikal
                        if categorical_cols:
                            st.info("Fitur kategorikal terdeteksi. Menerapkan One-Hot Encoding untuk analisis SHAP." if st.session_state.language == 'id' else 
                                "Categorical features detected. Applying One-Hot Encoding for SHAP analysis.")
                            X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=False)
                        
                        # Pastikan semua nilai dalam X_sample adalah numerik
                        for col in X_sample.columns:
                            try:
                                # Konversi ke numpy array terlebih dahulu
                                X_sample[col] = np.array(X_sample[col]).astype(float)
                            except:
                                try:
                                    # Jika gagal, gunakan factorize dan konversi ke float
                                    X_sample[col] = pd.factorize(X_sample[col])[0].astype(float)
                                except Exception as e:
                                    st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                        
                        try:
                            # Pilih explainer berdasarkan jenis model
                            model_type = type(st.session_state.model).__name__.lower()
                            
                            if any(tree_model in model_type for tree_model in ['randomforest', 'gradientboosting', 'xgb', 'lgbm', 'catboost', 'decisiontree']):                            
                                if model_type == 'gradientboostingclassifier':
                                    # Gunakan KernelExplainer untuk GradientBoostingClassifier karena TreeExplainer tidak mendukung multi-kelas
                                    background = shap.kmeans(st.session_state.X_train[selected_features].sample(min(50, len(st.session_state.X_train)), random_state=42), 5)
                                    explainer = shap.KernelExplainer(st.session_state.model.predict_proba, background)
                                    shap_values = explainer.shap_values(X_sample)
                                else:
                                    # Gunakan TreeExplainer untuk model berbasis pohon lainnya
                                    explainer = shap.TreeExplainer(st.session_state.model)
                                    shap_values = explainer.shap_values(X_sample)
                                
                                # Untuk model klasifikasi dengan output multi-kelas
                                if st.session_state.problem_type == "Classification" and isinstance(shap_values, list):
                                    st.subheader("Pilih Kelas untuk Visualisasi SHAP" if st.session_state.language == 'id' else "Select Class for SHAP Visualization")
                                    if hasattr(st.session_state.model, 'classes_'):
                                        class_names = st.session_state.model.classes_
                                        class_idx = st.selectbox(
                                            "Pilih kelas:" if st.session_state.language == 'id' else "Select class:",
                                            options=range(len(class_names)),
                                            format_func=lambda i: f"{class_names[i]}"
                                        )
                                        shap_values_selected = shap_values[class_idx]
                                        st.success(f"Menampilkan nilai SHAP untuk kelas: {class_names[class_idx]}" if st.session_state.language == 'id' else 
                                                f"Displaying SHAP values for class: {class_names[class_idx]}")
                                    else:
                                        class_idx = st.selectbox(
                                            "Pilih indeks kelas:" if st.session_state.language == 'id' else "Select class index:",
                                            options=range(len(shap_values))
                                        )
                                        shap_values_selected = shap_values[class_idx]
                                        st.success(f"Menampilkan nilai SHAP untuk indeks kelas: {class_idx}" if st.session_state.language == 'id' else 
                                                f"Displaying SHAP values for class index: {class_idx}")
                                else:
                                    shap_values_selected = shap_values
                            else:
                                # Gunakan KernelExplainer untuk model lainnya
                                background = shap.kmeans(st.session_state.X_train[selected_features].sample(min(50, len(st.session_state.X_train)), random_state=42), 5)
                                explainer = shap.KernelExplainer(st.session_state.model.predict, background)
                                shap_values_selected = explainer.shap_values(X_sample)
                            
                            # Visualisasi SHAP
                            st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                            
                            # 1. Summary Plot
                            st.write("### Summary Plot")
                            fig, ax = plt.subplots(figsize=(10, 8))
                            shap.summary_plot(shap_values_selected, X_sample, show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                            
                            # 2. Feature Importance Plot
                            st.write("### Feature Importance Plot")
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.summary_plot(shap_values_selected, X_sample, plot_type="bar", show=False)
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                            
                            # 3. Dependence Plots untuk fitur teratas
                            st.write("### Dependence Plots")
                            
                            # Hitung rata-rata nilai absolut SHAP untuk setiap fitur
                            if isinstance(shap_values_selected, list):
                                # Untuk multi-output, ambil output pertama
                                shap_arr = np.array(shap_values_selected[0], dtype=float)
                                feature_importance = np.abs(shap_arr).mean(0)
                            else:
                                shap_arr = np.array(shap_values_selected, dtype=float)
                                feature_importance = np.abs(shap_arr).mean(0)
                            
                            # Dapatkan indeks fitur terurut berdasarkan kepentingan
                            top_indices = feature_importance.argsort()[-5:][::-1]
                            
                            # Buat dependence plot untuk 5 fitur teratas
                            for idx in top_indices:
                                if idx < len(X_sample.columns):  # Pastikan indeks valid
                                    feature_name = X_sample.columns[idx]
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    shap.dependence_plot(idx, shap_values_selected, X_sample, show=False, ax=ax)
                                    plt.title(f"Dependence Plot for {feature_name}")
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.clf()
                            
                            # 4. Force Plot untuk sampel individual
                            st.write("### Force Plot untuk Sampel Individual")
                            sample_idx = st.slider(
                                "Pilih indeks sampel:" if st.session_state.language == 'id' else "Select sample index:",
                                0, len(X_sample) - 1, 0
                            )
                            
                            # Tampilkan data sampel
                            st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                            st.dataframe(X_sample.iloc[[sample_idx]])
                            
                            # Force plot
                            if isinstance(shap_values_selected, list):
                                # Untuk multi-output, ambil output pertama
                                force_plot = shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                                                        shap_values_selected[0][sample_idx, :], 
                                                        X_sample.iloc[sample_idx, :], 
                                                        matplotlib=True,
                                                        show=False)
                            else:
                                force_plot = shap.force_plot(explainer.expected_value if hasattr(explainer, 'expected_value') else 0, 
                                                        shap_values_selected[sample_idx, :], 
                                                        X_sample.iloc[sample_idx, :], 
                                                        matplotlib=True,
                                                        show=False)
                            
                            st.pyplot(force_plot)
                            
                        # 5. Waterfall Plot
                            st.write("### Waterfall Plot")
                            fig, ax = plt.subplots(figsize=(10, 8))

                            if isinstance(shap_values_selected, list):
                                # Untuk multi-output, ambil output dan expected_value untuk kelas pertama
                                shap.plots._waterfall.waterfall_legacy(
                                    explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                                    shap_values_selected[0][sample_idx, :],
                                    feature_names=X_sample.columns,
                                    show=False,
                                    max_display=10
                                )
                            else:
                                shap.plots._waterfall.waterfall_legacy(
                                    explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0],
                                    shap_values_selected[sample_idx, :],
                                    feature_names=X_sample.columns,
                                    show=False,
                                    max_display=10
                                )

                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.clf()
                            
                            # Tips untuk interpretasi
                            st.subheader("Tips untuk Interpretasi" if st.session_state.language == 'id' else "Tips for Interpretation")
                            st.info("""
                            - **Summary Plot**: Menunjukkan fitur mana yang paling penting dan bagaimana mereka mempengaruhi prediksi. Warna merah menunjukkan nilai fitur tinggi, biru menunjukkan nilai rendah.
                            - **Feature Importance**: Menampilkan fitur berdasarkan kepentingannya (rata-rata nilai absolut SHAP).
                            - **Dependence Plot**: Menunjukkan bagaimana nilai SHAP berubah berdasarkan nilai fitur, membantu mengidentifikasi interaksi.
                            - **Force Plot**: Menunjukkan kontribusi setiap fitur untuk prediksi sampel individual.
                            - **Waterfall Plot**: Menunjukkan bagaimana setiap fitur berkontribusi pada prediksi akhir dari nilai dasar.
                            
                            Jika menggunakan One-Hot Encoding, fitur kategorikal akan dipecah menjadi beberapa kolom biner.
                            """ if st.session_state.language == 'id' else """
                            - **Summary Plot**: Shows which features are most important and how they affect predictions. Red indicates high feature values, blue indicates low values.
                            - **Feature Importance**: Displays features by importance (average absolute SHAP values).
                            - **Dependence Plot**: Shows how SHAP values change based on feature values, helping identify interactions.
                            - **Force Plot**: Shows the contribution of each feature for an individual sample prediction.
                            - **Waterfall Plot**: Shows how each feature contributes to the final prediction from the base value.
                            
                            If using One-Hot Encoding, categorical features will be split into multiple binary columns.
                            """)
                            
                        except Exception as e:
                            st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
                        
            # Tambahkan dukungan untuk model forecasting
            elif (st.session_state.model is not None and 'is_timeseries' in locals() and is_timeseries):
                st.write("""
                SHAP untuk model forecasting memerlukan pendekatan khusus karena struktur data deret waktu.
                Berikut adalah interpretasi model forecasting menggunakan SHAP.
                """ if st.session_state.language == 'id' else """
                SHAP for forecasting models requires a special approach due to the time series data structure.
                Here is the interpretation of the forecasting model using SHAP.
                """)
                
                # Cek apakah model adalah model machine learning atau model statistik
                if hasattr(st.session_state, 'forecast_model_type'):
                    model_type = st.session_state.forecast_model_type
                    
                    if model_type in ['random_forest', 'gradient_boosting', 'linear_regression']:
                        # Untuk model ML, kita bisa menggunakan SHAP seperti biasa
                        st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
                        
                        if hasattr(st.session_state, 'forecast_features') and st.session_state.forecast_features:
                            feature_names = st.session_state.forecast_features
                            selected_features = st.multiselect(
                                "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                                options=feature_names,
                                default=feature_names[:min(10, len(feature_names))]
                            )
                            
                            # Jumlah sampel untuk analisis SHAP
                            sample_size = st.slider(
                                "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                                min_value=10, max_value=min(100, len(st.session_state.forecast_test_data)), value=50
                            )
                            
                            if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
                                if not selected_features:
                                    st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                                else:
                                    with st.spinner("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values..."):
                                        try:
                                            # Persiapkan data untuk SHAP
                                            X_sample = st.session_state.forecast_test_data[selected_features].sample(min(sample_size, len(st.session_state.forecast_test_data)), random_state=42)
                                            
                                            # Pastikan semua nilai dalam X_sample adalah numerik
                                            for col in X_sample.columns:
                                                try:
                                                    # Konversi ke numpy array terlebih dahulu
                                                    X_sample[col] = np.array(X_sample[col]).astype(float)
                                                except:
                                                    try:
                                                        # Jika gagal, gunakan factorize dan konversi ke float
                                                        X_sample[col] = pd.factorize(X_sample[col])[0].astype(float)
                                                    except Exception as e:
                                                        st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                                            
                                            # Pilih explainer berdasarkan jenis model
                                            if model_type in ['random_forest', 'gradient_boosting']:
                                                # Gunakan TreeExplainer untuk model berbasis pohon
                                                explainer = shap.TreeExplainer(st.session_state.model)
                                            else:
                                                # Gunakan KernelExplainer untuk model lainnya
                                                background = shap.kmeans(st.session_state.forecast_train_data[selected_features].sample(min(50, len(st.session_state.forecast_train_data)), random_state=42), 5)
                                                explainer = shap.KernelExplainer(st.session_state.model.predict, background)
                                            
                                            shap_values = explainer.shap_values(X_sample)
                                            
                                            # Visualisasi SHAP
                                            st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                                            
                                            # 1. Summary Plot
                                            st.write("### Summary Plot")
                                            fig, ax = plt.subplots(figsize=(10, 8))
                                            shap.summary_plot(shap_values, X_sample, show=False)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.clf()
                                            
                                            # 2. Feature Importance Plot
                                            st.write("### Feature Importance Plot")
                                            fig, ax = plt.subplots(figsize=(10, 6))
                                            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                                            plt.tight_layout()
                                            st.pyplot(fig)
                                            plt.clf()
                                            
                                            # Interpretasi khusus untuk forecasting
                                            st.subheader("Interpretasi untuk Model Forecasting" if st.session_state.language == 'id' else "Interpretation for Forecasting Model")
                                            st.info("""
                                            Dalam model forecasting, fitur-fitur penting biasanya meliputi:
                                            - **Lag Features**: Nilai historis dari variabel target
                                            - **Fitur Tanggal/Waktu**: Seperti hari dalam minggu, bulan, kuartal, dll.
                                            - **Fitur Rolling**: Seperti rata-rata bergerak, standar deviasi, dll.
                                            - **Fitur Eksternal**: Variabel lain yang mempengaruhi target
                                            
                                            Nilai SHAP tinggi pada lag features menunjukkan bahwa model sangat bergantung pada pola historis terbaru.
                                            """ if st.session_state.language == 'id' else """
                                            In forecasting models, important features typically include:
                                            - **Lag Features**: Historical values of the target variable
                                            - **Date/Time Features**: Such as day of week, month, quarter, etc.
                                            - **Rolling Features**: Such as moving averages, standard deviations, etc.
                                            - **External Features**: Other variables that influence the target
                                            
                                            High SHAP values on lag features indicate that the model heavily relies on recent historical patterns.
                                            """)
                                            
                                        except Exception as e:
                                            st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
                                            
                        else:
                            st.warning("Tidak dapat menemukan fitur untuk model forecasting. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                                    "Could not find features for the forecasting model. Make sure the model has been trained correctly.")
                    else:
                        # Untuk model statistik seperti ARIMA, SARIMA, dll.
                        st.info("""
                        Model statistik seperti ARIMA, SARIMA, atau Exponential Smoothing tidak mendukung interpretasi SHAP secara langsung.
                        Model-model ini didasarkan pada komponen seperti tren, musiman, dan residual, bukan pada fitur individual.
                        
                        Untuk interpretasi model statistik, pertimbangkan untuk melihat:
                        - Koefisien model (AR, MA, dll.)
                        - Dekomposisi deret waktu (tren, musiman, residual)
                        - Analisis residual
                        """ if st.session_state.language == 'id' else """
                        Statistical models like ARIMA, SARIMA, or Exponential Smoothing do not support SHAP interpretation directly.
                        These models are based on components like trend, seasonality, and residuals, not on individual features.
                        
                        For statistical model interpretation, consider looking at:
                        - Model coefficients (AR, MA, etc.)
                        - Time series decomposition (trend, seasonality, residuals)
                        - Residual analysis
                        """)
                else:
                    st.warning("Informasi model forecasting tidak lengkap. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                            "Forecasting model information is incomplete. Make sure the model has been trained correctly.")
            else:
                st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")

# Tab 6: LIME Model Interpretation
with tab6:
    if st.session_state.problem_type != 'Regression':
        st.info("Fitur interpretasi LIME hanya tersedia untuk model regresi." if st.session_state.language == 'id' else "LIME interpretation is only available for regression models.")
    else:

        if not LIME_AVAILABLE:
            st.error("LIME tidak terinstal. Silakan instal dengan 'pip install lime'." if st.session_state.language == 'id' else "LIME is not installed. Please install it with 'pip install lime'.")
        elif (
            st.session_state.model is not None
            and st.session_state.problem_type in ["Regression", "Classification"]
            and not ('is_timeseries' in locals() and is_timeseries)
        ):
            st.write("""
            LIME (Local Interpretable Model-agnostic Explanations) adalah teknik untuk menjelaskan prediksi model machine learning.
            Tidak seperti SHAP yang memberikan nilai kontribusi global, LIME fokus pada penjelasan prediksi individual dengan membuat model lokal yang dapat diinterpretasi.
            """ if st.session_state.language == 'id' else """
            LIME (Local Interpretable Model-agnostic Explanations) is a technique for explaining machine learning model predictions.
            Unlike SHAP which provides global contribution values, LIME focuses on individual prediction explanations by creating a local interpretable model.
            """)

            # Pilih fitur untuk LIME
            st.subheader("Pemilihan Fitur untuk Analisis LIME" if st.session_state.language == 'id' else "Feature Selection for LIME Analysis")
            feature_names = st.session_state.X_train.columns.tolist()
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis LIME:" if st.session_state.language == 'id' else "Select features for LIME analysis:",
                options=feature_names,
                default=feature_names[:min(10, len(feature_names))]
            )

            num_features_show = st.slider(
                "Jumlah fitur yang ditampilkan dalam penjelasan:" if st.session_state.language == 'id' else "Number of features to show in the explanation:",
                3, min(20, len(selected_features)), 5
            )

            if st.button("Generate LIME Explanations" if st.session_state.language == 'id' else "Generate LIME Explanations"):
                if not selected_features:
                    st.error("Silakan pilih setidaknya satu fitur untuk analisis LIME." if st.session_state.language == 'id' else "Please select at least one feature for LIME analysis.")
                else:
                    with st.spinner("Menghitung penjelasan LIME..." if st.session_state.language == 'id' else "Calculating LIME explanations..."):
                        X_train_selected = st.session_state.X_train[selected_features]
                        X_test_selected = st.session_state.X_test[selected_features]

                        lime_mode = "regression" if st.session_state.problem_type == "Regression" else "classification"
                        predict_fn = st.session_state.model.predict if lime_mode == "regression" else st.session_state.model.predict_proba

                        explainer = lime_tabular.LimeTabularExplainer(
                            X_train_selected.values,
                            feature_names=selected_features,
                            mode=lime_mode,
                            random_state=42
                        )

                        st.subheader("Penjelasan Prediksi Individual" if st.session_state.language == 'id' else "Individual Prediction Explanation")
                        sample_idx = st.slider(
                            "Indeks sampel:", 0, len(X_test_selected) - 1, 0,
                            key="lime_sample_idx"
                        )
                        sample = X_test_selected.iloc[sample_idx]
                        st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                        st.dataframe(pd.DataFrame([sample], columns=selected_features))

                        actual = st.session_state.y_test.iloc[sample_idx]
                        original_sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                        predicted = st.session_state.model.predict(original_sample)[0]
                        st.write(f"Nilai aktual: {actual}")
                        st.write(f"Nilai prediksi: {predicted}")

                        explanation = explainer.explain_instance(
                            sample.values,
                            predict_fn,
                            num_features=num_features_show
                        )

                        st.subheader("Visualisasi Penjelasan LIME" if st.session_state.language == 'id' else "LIME Explanation Visualization")
                        fig = plt.figure(figsize=(10, 6))
                        if st.session_state.problem_type == "Classification":
                            class_names = st.session_state.model.classes_ if hasattr(st.session_state.model, 'classes_') else None
                            if class_names is not None:
                                if predicted in class_names:
                                    label_idx = list(class_names).index(predicted)
                                else:
                                    label_idx = int(predicted)
                                explanation.as_pyplot_figure(label=label_idx)
                            else:
                                explanation.as_pyplot_figure()
                        else:
                            explanation.as_pyplot_figure()  # Untuk regresi, JANGAN beri argumen label
                        plt.tight_layout()
                        st.pyplot(fig)

                        st.subheader("Penjelasan dalam Bentuk Tabel" if st.session_state.language == 'id' else "Explanation in Table Format")
                        explanation_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Kontribusi"])
                        explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                        st.dataframe(explanation_df)

                        st.subheader("Nilai Fitur untuk Sampel yang Dijelaskan" if st.session_state.language == 'id' else "Feature Values for Explained Sample")
                        feature_values = pd.DataFrame({
                            "Feature": selected_features,
                            "Value": sample.values
                        })
                        st.dataframe(feature_values)

                        st.success("Analisis LIME selesai!" if st.session_state.language == 'id' else "LIME analysis completed successfully!")
        elif st.session_state.model is not None and st.session_state.problem_type not in ["Regression", "Classification"]:
            st.warning("LIME hanya tersedia untuk model regresi dan klasifikasi. Untuk model forecasting, fitur ini dinonaktifkan." if st.session_state.language == 'id' else "LIME is only available for regression and classification models. For forecasting models, this feature is disabled.")
        else:
            st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")

# Tab7: Model Interpretation (Partial Dependence Plot)
with tab7:
    st.header("Partial Dependence Plot (PDP) Analysis")

    if (
        st.session_state.model is not None
        and st.session_state.problem_type in ["Regression", "Classification"]
        and not ('is_timeseries' in locals() and is_timeseries)
    ):
        st.write("""
        Partial Dependence Plot (PDP) membantu memvisualisasikan hubungan antara satu atau dua fitur dan prediksi model, 
        dengan mengisolasi efek fitur tersebut dari fitur lainnya""" if st.session_state.language == 'id' else """
        Partial Dependence Plot (PDP) helps visualize the relationship between one or two features and the model predictions, 
        isolating the effect of those features from others.
        """)
        # Pilih fitur untuk PDP
        features = st.multiselect(
            "Pilih satu atau dua fitur untuk PDP:" if st.session_state.language == 'id' else "Select one or two features for PDP:",
            options=st.session_state.X_train.columns.tolist(),
            default=st.session_state.X_train.columns[:1].tolist(),
            max_selections=2
        )
        if len(features) == 0:
            st.info("Pilih minimal satu fitur." if st.session_state.language == 'id' else "Please select at least one feature.")
        elif len(features) > 2:
            st.warning("Pilih maksimal dua fitur untuk PDP." if st.session_state.language == 'id' else "Please select at most two features for PDP.")
        else:
            if st.button("Generate PDP"):
                with st.spinner("Menghitung Partial Dependence..." if st.session_state.language == 'id' else "Calculating Partial Dependence..."):
                    try:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        display = PartialDependenceDisplay.from_estimator(
                            st.session_state.model,
                            st.session_state.X_train,
                            features=features,
                            ax=ax
                        )
                        st.pyplot(fig)
                        st.success("PDP berhasil dibuat!" if st.session_state.language == 'id' else "PDP created successfully!")
                    except Exception as e:
                        st.error(f"Error saat membuat PDP: {str(e)}" if st.session_state.language == 'id' else f"Error creating PDP: {str(e)}")
    elif st.session_state.model is not None and st.session_state.problem_type not in ["Regression", "Classification"]:
        st.warning("PDP hanya tersedia untuk model regresi dan klasifikasi. Untuk model forecasting, fitur ini dinonaktifkan." if st.session_state.language == 'id' else "PDP is only available for regression and classification models. For forecasting models, this feature is disabled.")
    else:
        st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")