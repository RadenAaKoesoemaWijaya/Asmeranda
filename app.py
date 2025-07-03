import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, mutual_info_regression, mutual_info_classif
from sklearn.decomposition import PCA
import shap
import pickle
import os
from PIL import Image
import io
import time
from fpdf import FPDF


# Try to import XGBoost, but don't fail if it's not installed
try:
    import xgboost as xgb
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LIME, but don't fail if it's not installed
try:
    import lime
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# Initialize translation
TRANSLATIONS = {
    'en': {
        'app_title': 'Comprehensive Machine Learning App',
        'app_description': 'This application helps you analyze your data, preprocess it for machine learning, train models, and interpret the results using SHAP values.',
        # Add more translations here
    },
    'id': {
        'app_title': 'Aplikasi Machine Learning Komprehensif',
        'app_description': 'Aplikasi ini membantu Anda menganalisis data, memprosesnya untuk machine learning, melatih model, dan menginterpretasikan hasil menggunakan nilai SHAP.',
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
    st.session_state.language = 'en'

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

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📤 Data Upload", "📊 Exploratory Data Analytic", "🔄 Preprocessing", "🛠️ Feature Engineering & Model Training", "🔍 SHAP Model Interpretation", "🔎 LIME Model Interpretation"])

# Tab 1: Data Upload
with tab1:
    st.header("Unggah Dataset Anda" if st.session_state.language == 'id' else "Upload Your Dataset")
    
    uploaded_file = st.file_uploader("Pilih file CSV" if st.session_state.language == 'id' else "Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.success(f"Dataset berhasil dimuat dengan {data.shape[0]} baris dan {data.shape[1]} kolom." if st.session_state.language == 'id' else f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")
            
            st.subheader("Pratinjau Data" if st.session_state.language == 'id' else "Data Preview")
            st.dataframe(data.head())
            
            st.subheader("Informasi Data" if st.session_state.language == 'id' else "Data Information")
            buffer = io.StringIO()
            data.info(buf=buffer)
            st.text(buffer.getvalue())
            
            st.subheader("Statistik Data" if st.session_state.language == 'id' else "Data Statistics")
            st.dataframe(data.describe())
            
            # Identify numerical and categorical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            st.session_state.numerical_columns = numerical_cols
            st.session_state.categorical_columns = categorical_cols
            
            st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
            st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")
            
        except Exception as e:
            st.error(f"Error: {e}")

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
            x_axis = st.selectbox("Select X-axis feature:", data.columns)
        
        with col2:
            y_axis = st.selectbox("Select Y-axis feature:", [col for col in data.columns if col != x_axis])
        
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
        st.info("Please upload a dataset in the 'Data Upload' tab first.")

# Tab 3: Preprocessing
with tab3:
    st.header("Data Preprocessing")
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        st.subheader("Select Target Variable")
        target_column = st.selectbox("Choose the target column for prediction:", data.columns)
        st.session_state.target_column = target_column
        
        # Determine problem type
        if data[target_column].dtype in ['int64', 'float64']:
            if len(data[target_column].unique()) <= 10:
                problem_type = st.radio("Select problem type:", ["Classification", "Regression"], index=0)
            else:
                problem_type = st.radio("Select problem type:", ["Classification", "Regression"], index=1)
        else:
            problem_type = "Classification"
        
        st.session_state.problem_type = problem_type
        
        st.subheader("Handle Missing Values")
        
        # Display columns with missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            st.write("Columns with missing values:", ", ".join(missing_cols))
            
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
            st.success("No missing values found in the dataset.")
        
        # Handle Outliers
        st.subheader("Handle Outliers" if st.session_state.language == 'id' else "Handle Outliers")
        
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
        
        # Feature selection
        st.subheader("Feature Selection")
        
        all_columns = [col for col in data.columns if col != target_column]
        selected_features = st.multiselect("Select features to include in the model:", 
                                          all_columns, 
                                          default=all_columns)
        
        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            # Prepare data for modeling
            X = data[selected_features]
            y = data[target_column]
            
            # Encoding categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            if categorical_cols:
                st.subheader("Encode Categorical Features")
                
                encoding_method = st.radio("Encoding method:", ["Label Encoding", "One-Hot Encoding"])
                
                if encoding_method == "Label Encoding":
                    encoders = {}
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le
                    st.session_state.encoders = encoders
                    st.success("Label encoding applied to categorical features.")
                else:
                    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
                    st.success("One-hot encoding applied to categorical features.")
            
            # Scale numerical features
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            
            if numerical_cols:
                st.subheader("Scale Numerical Features")
                
                scaling = st.checkbox("Apply standard scaling to numerical features", value=True)
                
                if scaling:
                    scaler = StandardScaler()
                    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
                    st.session_state.scaler = scaler
                    st.success("Standard scaling applied to numerical features.")
            
            # Train-test split
            st.subheader("Train-Test Split")
            
            test_size = st.slider("Test set size (%):", 10, 50, 20) / 100
            random_state = st.number_input("Random state:", 0, 100, 42)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.processed_data = data
            
            st.success(f"Data split into {X_train.shape[0]} training samples and {X_test.shape[0]} testing samples.")
            
            # Display processed data
            st.subheader("Processed Data Preview")
            st.dataframe(X.head())
    else:
        st.info("Please upload a dataset in the 'Data Upload' tab first.")

# Tab 4: Feature Engineering and Model Training
with tab4:
    st.header("Model Training and Evaluation")
    
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
        use_kfold = st.checkbox("Gunakan K-Fold Cross Validation", value=False)
        
        if use_kfold:
            from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
            
            st.subheader("K-Fold Cross Validation")
            
            n_splits = st.slider("Jumlah fold (K):", 2, 10, 5)
            cv_scoring = None
            
            if problem_type == "Classification":
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:",
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]
                )
            else:  # Regression
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:",
                    ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
                )
        
        if is_timeseries:
            st.subheader("Pelatihan Model Forecasting")
            
            # Select date column
            date_column = st.selectbox("Pilih kolom tanggal/waktu:", date_columns)
            
            # Select target column
            target_column = st.selectbox("Pilih kolom target untuk diprediksi:", 
                                        [col for col in st.session_state.data.columns 
                                         if col != date_column and col in st.session_state.numerical_columns])
            
            # Select frequency
            freq = st.selectbox("Frekuensi data:", ["Harian (D)", "Mingguan (W)", "Bulanan (M)", "Tahunan (Y)", "Lainnya"])
            freq_map = {"Harian (D)": "D", "Mingguan (W)": "W", "Bulanan (M)": "M", "Tahunan (Y)": "Y", "Lainnya": None}
            selected_freq = freq_map[freq]
            
            # Number of periods to forecast
            forecast_periods = st.slider("Jumlah periode untuk prediksi ke depan:", 1, 100, 10)
            
            # Select forecasting model
            model_type = st.selectbox("Pilih model forecasting:", 
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
                st.error("Modul forecasting tidak tersedia. Pastikan file utils.py dan forecasting_utils.py ada di direktori yang sama.")
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
                        st.warning("Statsmodels tidak terinstal. Silakan instal dengan 'pip install statsmodels'.")
                
                # Import Prophet conditionally
                try:
                    from prophet import Prophet
                    PROPHET_AVAILABLE = True
                except ImportError:
                    PROPHET_AVAILABLE = False
                    if model_type == "Prophet":
                        st.warning("Prophet tidak terinstal. Silakan instal dengan 'pip install prophet'.")
                
                # Prepare time series data
                if st.button("Proses Data Time Series"):
                    with st.spinner("Memproses data time series..."):
                        # Prepare data
                        ts_data = prepare_timeseries_data(
                            st.session_state.data, 
                            date_column, 
                            target_column, 
                            freq=selected_freq
                        )
                        
                        # Check stationarity
                        stationarity_result = check_stationarity(ts_data[target_column])
                        st.write("Hasil Uji Stasioneritas:")
                        st.write(f"- Test Statistic: {stationarity_result['Test Statistic']:.4f}")
                        st.write(f"- p-value: {stationarity_result['p-value']:.4f}")
                        st.write(f"- Data {'stasioner' if stationarity_result['Stationary'] else 'tidak stasioner'}")
                        
                        # Plot time series analysis
                        st.write("Analisis Time Series:")
                        fig = plot_timeseries_analysis(ts_data[target_column])
                        st.pyplot(fig)
                        
                        # Split data for training and testing
                        train_size = int(len(ts_data) * 0.8)
                        train_data = ts_data.iloc[:train_size]
                        test_data = ts_data.iloc[train_size:]
                        
                        st.write(f"Data dibagi menjadi {len(train_data)} sampel training dan {len(test_data)} sampel testing")
                        
                        # Train model based on selection
                        if model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                            p = st.slider("Parameter p (AR):", 0, 5, 1)
                            d = st.slider("Parameter d (differencing):", 0, 2, 1)
                            q = st.slider("Parameter q (MA):", 0, 5, 1)
                            
                            with st.spinner("Melatih model ARIMA..."):
                                model = train_arima_model(train_data, target_column, order=(p, d, q))
                                st.session_state.model = model
                                st.success("Model ARIMA berhasil dilatih!")
                        
                        elif model_type == "Exponential Smoothing" and STATSMODELS_AVAILABLE:
                            trend = st.selectbox("Tipe trend:", ["add", "mul", None])
                            seasonal = st.selectbox("Tipe seasonal:", ["add", "mul", None])
                            seasonal_periods = st.slider("Periode seasonal:", 0, 52, 12)
                            
                            with st.spinner("Melatih model Exponential Smoothing..."):
                                model = train_exponential_smoothing(
                                    train_data, 
                                    target_column, 
                                    trend=trend, 
                                    seasonal=seasonal, 
                                    seasonal_periods=seasonal_periods
                                )
                                st.session_state.model = model
                                st.success("Model Exponential Smoothing berhasil dilatih!")
                        
                        elif model_type == "Prophet" and PROPHET_AVAILABLE:
                            yearly_seasonality = st.selectbox("Seasonality tahunan:", ["auto", True, False])
                            weekly_seasonality = st.selectbox("Seasonality mingguan:", ["auto", True, False])
                            daily_seasonality = st.selectbox("Seasonality harian:", ["auto", True, False])
                            
                            # Implementasi Prophet akan dilakukan di forecasting_utils.py
                            st.info("Implementasi Prophet akan menggunakan forecasting_utils.py")
                        
                        elif model_type in ["Random Forest", "Gradient Boosting"]:
                            n_estimators = st.slider("Jumlah trees:", 10, 500, 100)
                            max_depth = st.slider("Kedalaman maksimum:", 1, 50, 10)
                            
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
                            
                            with st.spinner(f"Melatih model {model_type}..."):
                                model_info = train_ml_forecaster(
                                    st.session_state.data,
                                    date_column,
                                    target_column,
                                    model_type=model_type.lower().replace(" ", "_"),
                                    **model_params
                                )
                                st.session_state.model = model_info
                                st.success(f"Model {model_type} berhasil dilatih!")
                        
                        # Evaluate model if available
                        if st.session_state.model is not None:
                            with st.spinner("Mengevaluasi model..."):
                                try:
                                    eval_results = evaluate_forecast_model(st.session_state.model, test_data, target_column)
                                    
                                    st.write("Hasil Evaluasi Model:")
                                    st.write(f"- RMSE: {eval_results['RMSE']:.4f}")
                                    st.write(f"- MAE: {eval_results['MAE']:.4f}")
                                    st.write(f"- R²: {eval_results['R2']:.4f}")
                                    
                                    # Generate forecast
                                    forecast_data = forecast_future(st.session_state.model, periods=forecast_periods)
                                    
                                    # Plot results
                                    fig = plot_forecast_results(train_data, test_data, forecast_data, target_column)
                                    st.pyplot(fig)
                                    
                                    # Show forecast data
                                    st.write("Data Hasil Forecasting:")
                                    st.dataframe(forecast_data)
                                except Exception as e:
                                    st.error(f"Error saat evaluasi model: {str(e)}")

        else:
            # Non-time series data - Classification or Regression
            st.subheader(f"Training a {problem_type} Model")
            
            # Tambahkan opsi untuk menggunakan GridSearchCV
            use_grid_search = st.checkbox("Gunakan GridSearchCV untuk hyperparameter tuning", value=False)
            
            # Model selection
            if problem_type == "Classification":
                # Define available classification models
                classification_models = ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting", "MLP (Neural Network)"]
                
                # Add XGBoost if available
                if XGBOOST_AVAILABLE:
                    classification_models.append("XGBoost")
                    
                model_type = st.selectbox("Select a classification model:", classification_models)
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Number of trees:", 10, 500, 100)
                    max_depth = st.slider("Maximum depth:", 1, 50, 10)
                    
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
                        
                elif model_type == "Logistic Regression":
                    C = st.slider("Regularization parameter (C):", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Maximum iterations:", 100, 1000, 100)
                    
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
                    C = st.slider("Regularization parameter (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):", ["scale", "auto"])
                    
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
                    n_neighbors = st.slider("Number of neighbors (K):", 1, 20, 5)
                    weights = st.selectbox("Weight function:", ["uniform", "distance"])
                    algorithm = st.selectbox("Algorithm:", ["auto", "ball_tree", "kd_tree", "brute"])
                    
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
                    max_depth = st.slider("Maximum depth:", 1, 50, 10)
                    min_samples_split = st.slider("Minimum samples to split:", 2, 20, 2)
                    criterion = st.selectbox("Split criterion:", ["gini", "entropy"])
                    
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
                    var_smoothing = st.slider("Variance smoothing:", 1e-10, 1e-8, 1e-9, format="%.1e")
                    
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
                    n_estimators = st.slider("Number of boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Maximum depth:", 1, 10, 3)
                    
                    base_model = GradientBoostingClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [2, 3, 5] if max_depth == 3 else [max(1, max_depth-1), max_depth, min(10, max_depth+2)],
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
                    hidden_layer_sizes = st.text_input("Hidden layer sizes (comma-separated):", "100,50")
                    hidden_layer_sizes = tuple(int(x) for x in hidden_layer_sizes.split(","))
                    activation = st.selectbox("Activation function:", ["relu", "tanh", "logistic"])
                    solver = st.selectbox("Solver:", ["adam", "sgd", "lbfgs"])
                    alpha = st.slider("Alpha (L2 penalty):", 0.0001, 0.01, 0.0001, format="%.4f")
                    max_iter = st.slider("Maximum iterations:", 100, 1000, 200)
                    
                    base_model = MLPClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'hidden_layer_sizes': [(100,), (100,50), (50,50,50)],
                            'activation': ['relu', 'tanh', 'logistic'],
                            'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [0.0001, 0.001, 0.01],
                            'max_iter': [200, 500, 1000]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = MLPClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            activation=activation,
                            solver=solver,
                            alpha=alpha,
                            max_iter=max_iter,
                            random_state=42
                        )
                        
                elif model_type == "XGBoost" and XGBOOST_AVAILABLE:
                    n_estimators = st.slider("Number of trees:", 10, 500, 100)
                    max_depth = st.slider("Maximum depth:", 1, 15, 6)
                    learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                    subsample = st.slider("Subsample ratio:", 0.5, 1.0, 0.8)
                    colsample_bytree = st.slider("Column sample by tree:", 0.5, 1.0, 0.8)
                    
                    base_model = XGBClassifier(random_state=42)
                    
                    if use_grid_search:
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'max_depth': [3, 6, 9] if max_depth == 6 else [max(1, max_depth-3), max_depth, min(15, max_depth+3)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'subsample': [0.6, 0.8, 1.0],
                            'colsample_bytree': [0.6, 0.8, 1.0]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    else:
                        model = XGBClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            random_state=42
                        )
                else:
                    st.error("Please select a valid classification model.")
                    model = None
                    
            else:  # Regression
                # Regular regression models (non-time series)
                model_type = st.selectbox("Pilih model regresi:", 
                                         ["Random Forest", "Linear Regression", "Gradient Boosting", "SVR"])
                
                if model_type == "Random Forest":
                    n_estimators = st.slider("Jumlah trees:", 10, 500, 100)
                    max_depth = st.slider("Kedalaman maksimum:", 1, 50, 10)
                    
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
                    n_estimators = st.slider("Jumlah boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Kedalaman maksimum:", 1, 10, 3)
                    
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
                    fit_intercept = st.checkbox("Fit intercept", value=True)
                    
                    base_model = LinearRegression()
                    
                    if use_grid_search:
                        param_grid = {
                            'fit_intercept': [True, False],
                            'normalize': [True, False]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = LinearRegression(
                            fit_intercept=fit_intercept
                        )
                        
                elif model_type == "SVR":
                    C = st.slider("Regularization parameter (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):", ["scale", "auto"])
                    epsilon = st.slider("Epsilon:", 0.01, 0.5, 0.1)
                    
                    base_model = SVR()
                    
                    if use_grid_search:
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                            'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                            'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                            'epsilon': [0.05, 0.1, 0.2] if epsilon == 0.1 else [max(0.01, epsilon/2), epsilon, min(0.5, epsilon*2)]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    else:
                        model = SVR(
                            C=C,
                            kernel=kernel,
                            gamma=gamma,
                            epsilon=epsilon
                        )
                else:
                    st.error("Please select a valid regression model.")
                    model = None
            
            # Train model button
            if model is not None and st.button("Train Model"):
                with st.spinner(f"Training {model_type} model..."):
                    try:
                        start_time = time.time()
                        model.fit(st.session_state.X_train, st.session_state.y_train)
                        training_time = time.time() - start_time
                        
                        # Jika menggunakan GridSearchCV, tampilkan parameter terbaik
                        if use_grid_search:
                            st.success(f"Model training completed in {training_time:.2f} seconds with GridSearchCV!")
                            st.subheader("Best Parameters:")
                            st.write(model.best_params_)
                            st.write(f"Best Score (CV): {model.best_score_:.4f}")
                            
                            # Gunakan model terbaik untuk prediksi
                            y_pred = model.best_estimator_.predict(st.session_state.X_test)
                            st.session_state.model = model.best_estimator_
                        else:
                            st.success(f"Model training completed in {training_time:.2f} seconds!")
                            y_pred = model.predict(st.session_state.X_test)
                            st.session_state.model = model
                        
                        # Save model
                        os.makedirs("models", exist_ok=True)
                        model_filename = f"models/{model_type.lower().replace(' ', '_')}_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
                        with open(model_filename, 'wb') as f:
                            pickle.dump(st.session_state.model, f)
                        st.success(f"Model saved as '{model_filename}'")
                        
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
                            st.write("Classification Report:")
                            st.dataframe(report_df)
                            
                        else:  # Regression
                            mse = mean_squared_error(st.session_state.y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(st.session_state.y_test, y_pred)
                            
                            st.write(f"Mean Squared Error: {mse:.4f}")
                            st.write(f"Root Mean Squared Error: {rmse:.4f}")
                            st.write(f"R² Score: {r2:.4f}")
                            
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
                        st.error(f"Error during model training: {str(e)}") 

            # Tambahkan bagian untuk prediksi data baru
            if st.session_state.model is not None:
                st.subheader("Prediksi Data Baru")          
                # Pilih metode input data
                input_method = st.radio("Pilih metode input data:", ["Input Manual", "Upload CSV"])
                
                if input_method == "Input Manual":
                    # Buat form input untuk setiap fitur
                    st.write("Masukkan nilai untuk setiap fitur:")
                    
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
                            
                            # Lakukan prediksi
                            prediction = st.session_state.model.predict(input_df)
                            
                            # Tampilkan hasil prediksi
                            st.subheader("Hasil Prediksi")
                            if problem_type == "Classification":
                                st.write(f"Kelas yang diprediksi: {prediction[0]}")
                                
                                # Jika model memiliki predict_proba, tampilkan probabilitas
                                if hasattr(st.session_state.model, 'predict_proba'):
                                    proba = st.session_state.model.predict_proba(input_df)
                                    proba_df = pd.DataFrame(proba, columns=st.session_state.model.classes_)
                                    st.write("Probabilitas untuk setiap kelas:")
                                    st.dataframe(proba_df)
                            else:
                                st.write(f"Nilai yang diprediksi: {prediction[0]:.4f}")
                            
                        except Exception as e:
                            st.error(f"Error saat melakukan prediksi: {str(e)}")
                
                else:  # Upload CSV
                    st.write("Upload file CSV dengan data yang ingin diprediksi:")
                    uploaded_file = st.file_uploader("Pilih file CSV", type="csv", key="prediction_file")
                    
                    if uploaded_file is not None:
                        try:
                            # Baca file CSV
                            pred_data = pd.read_csv(uploaded_file)
                            
                            # Tampilkan preview data
                            st.write("Preview data:")
                            st.dataframe(pred_data.head())
                            
                            # Periksa apakah semua fitur yang diperlukan ada
                            missing_features = [f for f in st.session_state.X_train.columns if f not in pred_data.columns]
                            
                            if missing_features:
                                st.error(f"Data tidak memiliki fitur yang diperlukan: {', '.join(missing_features)}")
                            else:
                                # Pilih hanya fitur yang digunakan dalam model
                                pred_data = pred_data[st.session_state.X_train.columns]
                                
                                # Preprocessing data
                                # Encoding untuk fitur kategorikal
                                for col in [c for c in pred_data.columns if c in st.session_state.categorical_columns]:
                                    if col in st.session_state.encoders:
                                        pred_data[col] = st.session_state.encoders[col].transform(pred_data[col].astype(str))
                                
                                # Scaling untuk fitur numerikal
                                num_cols = [c for c in pred_data.columns if c in st.session_state.numerical_columns]
                                if st.session_state.scaler is not None and num_cols:
                                    pred_data[num_cols] = st.session_state.scaler.transform(pred_data[num_cols])
                                
                                if st.button("Prediksi Batch"):
                                    # Lakukan prediksi
                                    predictions = st.session_state.model.predict(pred_data)
                                    
                                    # Tambahkan hasil prediksi ke DataFrame
                                    result_df = pred_data.copy()
                                    result_df['Prediction'] = predictions
                                    
                                    # Tampilkan hasil
                                    st.subheader("Hasil Prediksi")
                                    st.dataframe(result_df)
                                                                        
                                    # Opsi untuk mengunduh hasil CSV
                                    csv = result_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Hasil Prediksi (CSV)",
                                        data=csv,
                                        file_name="prediction_results.csv",
                                        mime="text/csv",
                                        key="csv_download"
                                    )
                        
                        except Exception as e:
                            st.error(f"Error saat memproses file: {str(e)}")
                            
                # Tambahkan bagian untuk memuat model yang sudah disimpan
                st.subheader("Muat Model yang Sudah Disimpan")
                
                # Cek apakah folder models ada
                if os.path.exists("models"):
                    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
                    
                    if model_files:
                        selected_model_file = st.selectbox("Pilih model yang akan dimuat:", model_files)
                        
                        if st.button("Muat Model"):
                            try:
                                with open(os.path.join("models", selected_model_file), 'rb') as f:
                                    loaded_model = pickle.load(f)
                                
                                st.session_state.model = loaded_model
                                st.success(f"Model {selected_model_file} berhasil dimuat!")
                            except Exception as e:
                                st.error(f"Error saat memuat model: {str(e)}")
                    else:
                        st.info("Tidak ada model tersimpan di folder 'models'.")
                else:
                    st.info("Folder 'models' belum dibuat. Latih dan simpan model terlebih dahulu.")
            else:
                st.info("Silakan latih model terlebih dahulu sebelum melakukan prediksi.")
    else:
        st.info("Please complete the preprocessing steps in the previous tab first.")

# Tab5: Model Interpretation (SHAP)
with tab5:
    st.header("Model Interpretation with SHAP")
    
    if st.session_state.model is not None:
        st.write("""
        SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model. 
        It connects optimal credit allocation with local explanations using the classic Shapley values from game theory and their related extensions.
        """)
        
        # 3. Implement Feature Selection Before SHAP Analysis
        if st.session_state.model is not None:
            st.write("""
            Interpretability ML
            """)
            
            # Add feature selection for SHAP analysis
            st.subheader("Feature Selection for SHAP Analysis")
            
            # If model has feature importances, use them to suggest important features
            if hasattr(st.session_state.model, 'feature_importances_'):
                # For tree-based models
                importances = st.session_state.model.feature_importances_
                feature_names = st.session_state.X_train.columns if hasattr(st.session_state.X_train, 'columns') else [f"feature_{i}" for i in range(len(importances))]
                
                # Ensure both arrays have the same length
                if len(importances) != len(feature_names):
                    # Truncate the longer array to match the shorter one
                    min_length = min(len(importances), len(feature_names))
                    importances = importances[:min_length]
                    feature_names = feature_names[:min_length]
                
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                # Get top features
                top_n = st.slider("Number of top features to include:", 5, 
                               min(30, len(st.session_state.X_train.columns)), 10)
                top_features = feature_importance.head(top_n)['Feature'].tolist()
                
                # Let user select from suggested features or choose their own
                selected_features = st.multiselect(
                    "Select features for SHAP analysis:",
                    options=st.session_state.X_train.columns.tolist(),
                    default=top_features
                )
            else:
                # If no feature_importances_, let user select all features
                selected_features = st.multiselect(
                    "Select features for SHAP analysis:",
                    options=st.session_state.X_train.columns.tolist(),
                    default=st.session_state.X_train.columns[:10].tolist()  # Default to first 10
                )
            
            if st.button("Generate SHAP Values"):
                if not selected_features:
                    st.error("Please select at least one feature for SHAP analysis.")
                else:
                    with st.spinner("Calculating SHAP values..."):
                        # Create a smaller sample with only selected features
                        sample_size = min(50, len(st.session_state.X_train))
                        X_sample = st.session_state.X_train[selected_features].sample(sample_size, random_state=42)
                        
                        # Create explainer
                        if isinstance(st.session_state.model, (RandomForestClassifier, RandomForestRegressor)):
                            # Fix for additivity error - use interventional feature perturbation only
                            # Removed check_additivity parameter which is causing the error
                            explainer = shap.TreeExplainer(
                                st.session_state.model,
                                feature_perturbation="interventional"
                            )
                        else:
                            # Convert X_sample to numpy array if it's not already
                            X_sample_values = X_sample.values if hasattr(X_sample, 'values') else np.array(X_sample)
                            
                            # Create a wrapper function for the model's predict method
                            def model_predict(X):
                                # Ensure X is in the right format for the model
                                if isinstance(X, list):
                                    X = np.array(X)
                                return st.session_state.model.predict(X)
                            
                            # Create the explainer with the wrapper function
                            explainer = shap.KernelExplainer(model_predict, X_sample_values)
                        
                        # Calculate SHAP values
                        shap_values = explainer.shap_values(X_sample)
                        
                        # Summary plot
                        st.subheader("SHAP Summary Plot")
                        st.write("This plot shows the impact of each feature on the model output.")
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        if isinstance(st.session_state.model, (RandomForestClassifier, LogisticRegression)) and not isinstance(shap_values, np.ndarray):
                            # For multi-class classification
                            class_to_show = 0  # Show SHAP values for the first class
                            shap.summary_plot(shap_values[class_to_show], X_sample, show=False)
                        else:
                            shap.summary_plot(shap_values, X_sample, show=False)
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Feature importance plot
                        st.subheader("SHAP Feature Importance")
                        st.write("This plot shows the average absolute SHAP value for each feature.")
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        if isinstance(st.session_state.model, (RandomForestClassifier, LogisticRegression)) and not isinstance(shap_values, np.ndarray):
                            shap.summary_plot(shap_values[class_to_show], X_sample, plot_type="bar", show=False)
                        else:
                            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Dependence plots
                        st.subheader("SHAP Dependence Plots")
                        st.write("These plots show how the model output varies with the value of a feature.")
                        
                        # Let user select a feature for the dependence plot
                        feature_options = X_sample.columns.tolist()
                        selected_feature = st.selectbox("Select a feature for the dependence plot:", feature_options)
                        
                        feature_idx = feature_options.index(selected_feature)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        if isinstance(st.session_state.model, (RandomForestClassifier, LogisticRegression)) and not isinstance(shap_values, np.ndarray):
                            shap.dependence_plot(feature_idx, shap_values[class_to_show], X_sample, show=False)
                        else:
                            shap.dependence_plot(feature_idx, shap_values, X_sample, show=False)
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Individual prediction explanation
                        st.subheader("Individual Prediction Explanation")
                        st.write("Select a sample from the test set to explain its prediction.")
                        
                        sample_idx = st.slider("Sample index:", 0, len(st.session_state.X_test) - 1, 0)
                        
                        # Get the selected sample
                        sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                        
                        # Calculate SHAP values for this sample
                        sample_shap_values = explainer.shap_values(sample)
                        
                        # Display the sample data
                        st.write("Sample data:")
                        st.dataframe(sample)
                        
                        # Display the actual and predicted values
                        actual = st.session_state.y_test.iloc[sample_idx]
                        predicted = st.session_state.model.predict(sample)[0]
                        
                        st.write(f"Actual value: {actual}")
                        st.write(f"Predicted value: {predicted}")
                        
                        # Force plot
                        st.write("SHAP Force Plot:")
                        st.write("This plot shows how each feature contributes to the prediction for this sample.")
                        
                        fig, ax = plt.subplots(figsize=(12, 3))
                        if isinstance(st.session_state.model, (RandomForestClassifier, LogisticRegression)) and not isinstance(sample_shap_values, np.ndarray):
                            # For multi-class classification
                            shap.force_plot(
                                explainer.expected_value[class_to_show], 
                                sample_shap_values[class_to_show], 
                                sample, 
                                matplotlib=True,
                                show=False
                            )
                        else:
                            # For regression or binary classification
                            expected_value = explainer.expected_value
                            if isinstance(expected_value, list):
                                expected_value = expected_value[0]
                            
                            shap.force_plot(
                                expected_value, 
                                sample_shap_values, 
                                sample, 
                                matplotlib=True,
                                show=False
                            )
                        st.pyplot(fig)
                        plt.clf()
                        
                        # Waterfall plot (alternative to force plot)
                        st.write("SHAP Waterfall Plot:")
                        st.write("This plot shows how each feature pushes the model output from the base value to the final prediction.")
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        if isinstance(st.session_state.model, (RandomForestClassifier, LogisticRegression)) and not isinstance(sample_shap_values, np.ndarray):
                            # For multi-class classification, use the first class
                            expected_val = explainer.expected_value[class_to_show]
                            if isinstance(expected_val, np.ndarray):
                                expected_val = float(expected_val[0])
                            
                            shap_val = sample_shap_values[class_to_show][0]
                            
                            shap.plots._waterfall.waterfall_legacy(
                                expected_val, 
                                shap_val, 
                                feature_names=X_sample.columns,
                                show=False
                            )
                        else:
                            # For regression or binary classification
                            expected_val = explainer.expected_value
                            
                            # Handle different types of expected_value
                            if isinstance(expected_val, list):
                                expected_val = expected_val[0]
                            if isinstance(expected_val, np.ndarray):
                                expected_val = float(expected_val[0])
                            
                            # Handle different types of shap_values
                            if isinstance(sample_shap_values, list):
                                shap_val = sample_shap_values[0][0]
                            else:
                                shap_val = sample_shap_values[0]
                            
                            shap.plots._waterfall.waterfall_legacy(
                                expected_val, 
                                shap_val, 
                                feature_names=X_sample.columns,
                                show=False
                            )
                        st.pyplot(fig)
                        plt.clf()
                        
                        st.success("SHAP analysis completed!")
    else:
        st.info("Please train a model in the 'Model Training' tab first.")


# Tab6: Model Interpretation (LIME)
with tab6:
    st.header("Model Interpretation with LIME")
    
    if not LIME_AVAILABLE:
        st.error("LIME tidak terinstal. Silakan instal dengan 'pip install lime'.")
    elif st.session_state.model is not None:
        st.write("""
        LIME (Local Interpretable Model-agnostic Explanations) adalah teknik untuk menjelaskan prediksi model machine learning.
        Tidak seperti SHAP yang memberikan nilai kontribusi global, LIME fokus pada penjelasan prediksi individual dengan membuat model lokal yang dapat diinterpretasi.
        """)
        
        # Feature selection for LIME analysis
        st.subheader("Pemilihan Fitur untuk Analisis LIME")
        
        # If model has feature importances, use them to suggest important features
        if hasattr(st.session_state.model, 'feature_importances_'):
            # For tree-based models
            importances = st.session_state.model.feature_importances_
            feature_names = st.session_state.X_train.columns if hasattr(st.session_state.X_train, 'columns') else [f"feature_{i}" for i in range(len(importances))]
            
            # Ensure both arrays have the same length
            if len(importances) != len(feature_names):
                # Truncate the longer array to match the shorter one
                min_length = min(len(importances), len(feature_names))
                importances = importances[:min_length]
                feature_names = feature_names[:min_length]
            
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Get top features
            top_n = st.slider("Jumlah fitur teratas yang akan digunakan:", 5, 
                           min(30, len(st.session_state.X_train.columns)), 10, key="lime_top_n")
            top_features = feature_importance.head(top_n)['Feature'].tolist()
            
            # Let user select from suggested features or choose their own
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis LIME:",
                options=st.session_state.X_train.columns.tolist(),
                default=top_features,
                key="lime_features"
            )
        else:
            # If no feature_importances_, let user select all features
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis LIME:",
                options=st.session_state.X_train.columns.tolist(),
                default=st.session_state.X_train.columns[:10].tolist(),  # Default to first 10
                key="lime_features_all"
            )
        
        # Number of features to show in the explanation
        num_features_show = st.slider("Jumlah fitur yang ditampilkan dalam penjelasan:", 3, 
                                    min(20, len(selected_features)), 5)
        
        if st.button("Generate LIME Explanations"):
            if not selected_features:
                st.error("Silakan pilih setidaknya satu fitur untuk analisis LIME.")
            else:
                with st.spinner("Menghitung penjelasan LIME..."):
                    # Create a smaller sample with only selected features for training the explainer
                    X_train_selected = st.session_state.X_train[selected_features]
                    X_test_selected = st.session_state.X_test[selected_features]
                    
                    # Determine if it's a classification or regression problem
                    mode = "classification" if st.session_state.problem_type == "Classification" else "regression"
                    
                    # Create the LIME explainer
                    explainer = lime_tabular.LimeTabularExplainer(
                        X_train_selected.values,
                        feature_names=selected_features,
                        class_names=[str(c) for c in np.unique(st.session_state.y_train)] if mode == "classification" else None,
                        mode=mode,
                        random_state=42
                    )
                    
                    # Let user select a sample from the test set to explain
                    st.subheader("Penjelasan Prediksi Individual")
                    st.write("Pilih sampel dari set pengujian untuk menjelaskan prediksinya.")
                    
                    sample_idx = st.slider("Indeks sampel:", 0, len(X_test_selected) - 1, 0, key="lime_sample_idx")
                    
                    # Get the selected sample
                    sample = X_test_selected.iloc[sample_idx]
                    
                    # Display the sample data
                    st.write("Data sampel:")
                    sample_df = pd.DataFrame([sample], columns=selected_features)
                    st.dataframe(sample_df)
                    
                    # Display the actual and predicted values
                    actual = st.session_state.y_test.iloc[sample_idx]
                    
                    # Perbaikan: Gunakan data asli dari X_test untuk prediksi
                    # Ini memastikan semua fitur yang digunakan saat training tersedia
                    original_sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                    predicted = st.session_state.model.predict(original_sample)[0]
                    
                    st.write(f"Nilai aktual: {actual}")
                    st.write(f"Nilai prediksi: {predicted}")
                    
                    # Generate LIME explanation
                    if mode == "classification":
                        # For classification, we need to use predict_proba
                        if hasattr(st.session_state.model, 'predict_proba'):
                            def predict_fn(x):
                                return st.session_state.model.predict_proba(x)
                            
                            explanation = explainer.explain_instance(
                                sample.values, 
                                predict_fn,
                                num_features=num_features_show
                            )
                        else:
                            # If model doesn't have predict_proba, use predict
                            def predict_fn(x):
                                return np.array([1-st.session_state.model.predict(x), st.session_state.model.predict(x)]).T
                            
                            explanation = explainer.explain_instance(
                                sample.values, 
                                predict_fn,
                                num_features=num_features_show
                            )
                    else:
                        # For regression
                        explanation = explainer.explain_instance(
                            sample.values, 
                            st.session_state.model.predict,
                            num_features=num_features_show
                        )
                    
                    # Plot the explanation
                    st.subheader("Visualisasi Penjelasan LIME")
                    
                    # Instead of using save_to_file, use matplotlib to display the explanation
                    fig = plt.figure(figsize=(10, 6))
                    explanation.as_pyplot_figure(plt.gca())
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display the explanation as a table
                    st.subheader("Penjelasan dalam Bentuk Tabel")
                    
                    # Get the explanation as a list of tuples
                    explanation_list = explanation.as_list()
                    
                    # Convert to DataFrame for better display
                    explanation_df = pd.DataFrame(explanation_list, columns=["Feature", "Kontribusi"])
                    explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                    
                    # Display the table
                    st.dataframe(explanation_df)
                    
                    # Display feature values for the explained instance
                    st.subheader("Nilai Fitur untuk Sampel yang Dijelaskan")
                    feature_values = pd.DataFrame({
                        "Feature": selected_features,
                        "Value": sample.values
                    })
                    st.dataframe(feature_values)
                    
                    st.success("Analisis LIME selesai!")
    else:
        st.info("Silakan latih model terlebih dahulu di tab 'Model Training'.")