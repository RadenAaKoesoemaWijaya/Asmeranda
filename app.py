import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, StratifiedKFold, LeaveOneOut, LeavePOut, KFold
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
import shap
import pickle
import os
from PIL import Image
import io
import time
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# Import optional dependencies
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

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

# Initialize translation
TRANSLATIONS = {
    'en': {
        'app_title': 'Comprehensive Machine Learning App',
        'app_description': 'This application helps you analyze your data, preprocess it for machine learning, train models, and interpret the results using eXplainable AI.',
    },
    'id': {
        'app_title': 'Aplikasi Machine Learning Komprehensif',
        'app_description': 'Aplikasi ini membantu Anda menganalisis data, memprosesnya untuk machine learning, melatih model, dan menginterpretasikan hasil menggunakan eXplainable AI.',
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
    st.session_state.language = 'id'

col1, col2 = st.columns([0.9, 0.1])
with col2:
    if st.button('ID' if st.session_state.language == 'en' else 'EN'):
        st.session_state.language = 'id' if st.session_state.language == 'en' else 'en'

# App title and description
st.title(f"📊 {TRANSLATIONS[st.session_state.language]['app_title']}")
st.markdown(TRANSLATIONS[st.session_state.language]['app_description'])

# Initialize session state variables
session_vars = [
    'data', 'processed_data', 'target_column', 'model', 'X_train', 'X_test', 
    'y_train', 'y_test', 'problem_type', 'categorical_columns', 'numerical_columns',
    'encoders', 'scaler', 'model_type', 'model_results', 'is_time_series',
    'time_column', 'forecasting_models', 'forecast_results', 'authenticated',
    'current_user', 'session_token', 'show_register', 'captcha_text'
]

for var in session_vars:
    if var not in st.session_state:
        st.session_state[var] = None if var not in ['categorical_columns', 'numerical_columns', 'encoders', 'forecasting_models'] else ([] if 'columns' in var or 'forecasting' in var else {})

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📤 Data Upload", 
    "📊 Exploratory Data Analytic", 
    "🔄 Preprocessing and Feature Engineering", 
    "🛠️ Cross Validation and Model Training", 
    "🔍 SHAP Model Interpretation", 
    "🔎 LIME Model Interpretation"
])

def adjusted_r2_score(r2, n, k):
    """Hitung Adjusted R².""" if st.session_state.language == 'id' else """Calculate Adjusted R²."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def clean_data_for_training(X, y):
    """Membersihkan data untuk training model - handle NaN dan infinite values"""
    X = X.copy()
    y = y.copy()
    
    # Handle infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    
    # Drop rows with NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean = X[mask]
    y_clean = y[mask]
    
    return X_clean, y_clean

# Tab 1: Data Upload
with tab1:
    st.header("📤 Upload Data" if st.session_state.language == 'id' else "📤 Upload Data")
    
    uploaded_file = st.file_uploader("Pilih file CSV atau Excel" if st.session_state.language == 'id' else "Choose CSV or Excel file", 
                                     type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.data = pd.read_csv(uploaded_file)
            else:
                st.session_state.data = pd.read_excel(uploaded_file)
            
            st.success("Data berhasil diupload!" if st.session_state.language == 'id' else "Data uploaded successfully!")
            st.write(f"Jumlah baris: {len(st.session_state.data)}")
            st.write(f"Jumlah kolom: {len(st.session_state.data.columns)}")
            
            st.subheader("Preview Data" if st.session_state.language == 'id' else "Preview Data")
            st.dataframe(st.session_state.data.head())
            
        except Exception as e:
            st.error(f"Error membaca file: {str(e)}" if st.session_state.language == 'id' else f"Error reading file: {str(e)}")

# Tab 2: Exploratory Data Analysis
with tab2:
    st.header("📊 Exploratory Data Analysis" if st.session_state.language == 'id' else "📊 Exploratory Data Analysis")
    
    if st.session_state.data is not None:
        data = st.session_state.data
        
        # Basic info
        st.subheader("Informasi Dasar" if st.session_state.language == 'id' else "Basic Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Jumlah Baris" if st.session_state.language == 'id' else "Number of Rows", len(data))
        with col2:
            st.metric("Jumlah Kolom" if st.session_state.language == 'id' else "Number of Columns", len(data.columns))
        with col3:
            st.metric("Missing Values" if st.session_state.language == 'id' else "Missing Values", data.isnull().sum().sum())
        
        # Data types
        st.subheader("Tipe Data" if st.session_state.language == 'id' else "Data Types")
        st.write(data.dtypes)
        
        # Statistical summary
        st.subheader("Statistik Deskriptif" if st.session_state.language == 'id' else "Descriptive Statistics")
        st.write(data.describe())
        
        # Missing values
        st.subheader("Missing Values" if st.session_state.language == 'id' else "Missing Values")
        missing_df = pd.DataFrame({
            'Kolom': data.columns,
            'Missing Count': data.isnull().sum(),
            'Missing Percentage': (data.isnull().sum() / len(data)) * 100
        })
        missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        
        if len(missing_df) > 0:
            st.write(missing_df)
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=missing_df, x='Missing Count', y='Kolom', ax=ax)
            ax.set_title("Missing Values by Column" if st.session_state.language == 'id' else "Missing Values by Column")
            st.pyplot(fig)
        else:
            st.info("Tidak ada missing values!" if st.session_state.language == 'id' else "No missing values!")

# Tab 3: Preprocessing and Feature Engineering
with tab3:
    st.header("🔄 Preprocessing dan Feature Engineering" if st.session_state.language == 'id' else "🔄 Preprocessing and Feature Engineering")
    
    if st.session_state.data is not None:
        data = st.session_state.data.copy()
        
        # Target column selection
        st.subheader("Pilih Kolom Target" if st.session_state.language == 'id' else "Select Target Column")
        target_column = st.selectbox("Kolom Target" if st.session_state.language == 'id' else "Target Column", data.columns)
        st.session_state.target_column = target_column
        
        # Problem type
        st.subheader("Jenis Masalah" if st.session_state.language == 'id' else "Problem Type")
        
        if data[target_column].dtype in ['int64', 'float64']:
            unique_ratio = len(data[target_column].unique()) / len(data)
            if unique_ratio < 0.05:
                problem_type = "Classification"
            else:
                problem_type = "Regression"
        else:
            problem_type = "Classification"
        
        st.session_state.problem_type = problem_type
        st.write(f"Jenis masalah terdeteksi: **{problem_type}**" if st.session_state.language == 'id' else f"Detected problem type: **{problem_type}**")
        
        # Feature selection
        st.subheader("Pilih Fitur" if st.session_state.language == 'id' else "Select Features")
        feature_columns = [col for col in data.columns if col != target_column]
        selected_features = st.multiselect("Fitur" if st.session_state.language == 'id' else "Features", 
                                         feature_columns, default=feature_columns)
        
        if selected_features:
            X = data[selected_features]
            y = data[target_column]
            
            # Handle categorical variables
            categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
            numerical_columns = X.select_dtypes(exclude=['object']).columns.tolist()
            
            st.session_state.categorical_columns = categorical_columns
            st.session_state.numerical_columns = numerical_columns
            
            # Encoding
            if categorical_columns:
                st.subheader("Encoding Variabel Kategorikal" if st.session_state.language == 'id' else "Encoding Categorical Variables")
                encoding_method = st.selectbox("Metode Encoding" if st.session_state.language == 'id' else "Encoding Method", 
                                             ["Label Encoding", "One-Hot Encoding"])
                
                X_processed = X.copy()
                encoders = {}
                
                if encoding_method == "Label Encoding":
                    for col in categorical_columns:
                        le = LabelEncoder()
                        X_processed[col] = le.fit_transform(X[col].astype(str))
                        encoders[col] = le
                else:
                    X_processed = pd.get_dummies(X, columns=categorical_columns)
                
                st.session_state.encoders = encoders
            else:
                X_processed = X
            
            # Scaling
            st.subheader("Scaling" if st.session_state.language == 'id' else "Scaling")
            scaling_method = st.selectbox("Metode Scaling" if st.session_state.language == 'id' else "Scaling Method", 
                                          ["StandardScaler", "MinMaxScaler", "RobustScaler", "None"])
            
            if scaling_method != "None":
                if scaling_method == "StandardScaler":
                    scaler = StandardScaler()
                elif scaling_method == "MinMaxScaler":
                    scaler = MinMaxScaler()
                else:
                    scaler = RobustScaler()
                
                X_processed[numerical_columns] = scaler.fit_transform(X_processed[numerical_columns])
                st.session_state.scaler = scaler
            
            # Train test split
            st.subheader("Pembagian Data" if st.session_state.language == 'id' else "Data Split")
            test_size = st.slider("Test Size" if st.session_state.language == 'id' else "Test Size", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State" if st.session_state.language == 'id' else "Random State", 42)
            
            X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=test_size, random_state=random_state)
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            st.success("Data berhasil diproses!" if st.session_state.language == 'id' else "Data processed successfully!")
            st.write(f"Data training: {len(X_train)} sampel" if st.session_state.language == 'id' else f"Training data: {len(X_train)} samples")
            st.write(f"Data testing: {len(X_test)} sampel" if st.session_state.language == 'id' else f"Testing data: {len(X_test)} samples")

# Tab 4: Cross Validation and Model Training
with tab4:
    st.header("🛠️ Cross Validation dan Training Model" if st.session_state.language == 'id' else "🛠️ Cross Validation and Model Training")
    
    if st.session_state.X_train is not None:
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_test = st.session_state.y_test
        problem_type = st.session_state.problem_type
        
        st.subheader("Pilih Model" if st.session_state.language == 'id' else "Select Model")
        
        if problem_type == "Classification":
            model_options = ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "Neural Network"]
        else:
            model_options = ["Random Forest", "Linear Regression", "SVM", "Neural Network"]
        
        model_type = st.selectbox("Model" if st.session_state.language == 'id' else "Model", model_options)
        st.session_state.model_type = model_type
        
        # Cross validation
        st.subheader("Cross Validation" if st.session_state.language == 'id' else "Cross Validation")
        cv_method = st.selectbox("Metode CV" if st.session_state.language == 'id' else "CV Method", 
                               ["K-Fold", "Stratified K-Fold", "Leave-One-Out", "Shuffle Split"])
        
        if cv_method == "K-Fold":
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        elif cv_method == "Stratified K-Fold":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        elif cv_method == "Leave-One-Out":
            cv = LeaveOneOut()
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
        # Hyperparameter tuning
        st.subheader("Hyperparameter Tuning" if st.session_state.language == 'id' else "Hyperparameter Tuning")
        tuning_method = st.selectbox("Metode Tuning" if st.session_state.language == 'id' else "Tuning Method", 
                                   ["None", "Grid Search", "Random Search"])
        
        if st.button("Training Model" if st.session_state.language == 'id' else "Train Model"):
            try:
                # Clean data
                X_train_clean, y_train_clean = clean_data_for_training(X_train, y_train)
                X_test_clean, y_test_clean = clean_data_for_training(X_test, y_test)
                
                if problem_type == "Classification":
                    if model_type == "Random Forest":
                        model = RandomForestClassifier(random_state=42)
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5, 10]
                        }
                    elif model_type == "Logistic Regression":
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        param_grid = {'C': [0.1, 1, 10]}
                    elif model_type == "SVM":
                        model = SVC(random_state=42)
                        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                    elif model_type == "KNN":
                        model = KNeighborsClassifier()
                        param_grid = {'n_neighbors': [3, 5, 7, 9]}
                    elif model_type == "Decision Tree":
                        model = DecisionTreeClassifier(random_state=42)
                        param_grid = {'max_depth': [None, 5, 10, 15]}
                    elif model_type == "Naive Bayes":
                        model = GaussianNB()
                        param_grid = {}
                    else:  # Neural Network
                        model = MLPClassifier(random_state=42, max_iter=1000)
                        param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)]}
                else:
                    if model_type == "Random Forest":
                        model = RandomForestRegressor(random_state=42)
                        param_grid = {
                            'n_estimators': [50, 100, 200],
                            'max_depth': [None, 10, 20],
                            'min_samples_split': [2, 5, 10]
                        }
                    elif model_type == "Linear Regression":
                        model = LinearRegression()
                        param_grid = {}
                    elif model_type == "SVM":
                        model = SVR()
                        param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
                    else:  # Neural Network
                        model = MLPRegressor(random_state=42, max_iter=1000)
                        param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)]}
                
                # Hyperparameter tuning
                if tuning_method == "Grid Search" and param_grid:
                    search = GridSearchCV(model, param_grid, cv=cv, 
                                        scoring='accuracy' if problem_type == "Classification" else 'r2')
                    search.fit(X_train_clean, y_train_clean)
                    model = search.best_estimator_
                elif tuning_method == "Random Search" and param_grid:
                    search = RandomizedSearchCV(model, param_grid, cv=cv, n_iter=10, 
                                              scoring='accuracy' if problem_type == "Classification" else 'r2')
                    search.fit(X_train_clean, y_train_clean)
                    model = search.best_estimator_
                
                # Train model
                model.fit(X_train_clean, y_train_clean)
                st.session_state.model = model
                
                # Predictions
                y_pred = model.predict(X_test_clean)
                
                # Evaluation
                if problem_type == "Classification":
                    accuracy = accuracy_score(y_test_clean, y_pred)
                    report = classification_report(y_test_clean, y_pred)
                    
                    st.success(f"Model berhasil dilatih! Accuracy: {accuracy:.4f}" if st.session_state.language == 'id' else 
                             f"Model trained successfully! Accuracy: {accuracy:.4f}")
                    st.text("Classification Report:" if st.session_state.language == 'id' else "Classification Report:")
                    st.text(report)
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test_clean, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_title("Confusion Matrix" if st.session_state.language == 'id' else "Confusion Matrix")
                    st.pyplot(fig)
                else:
                    mse = mean_squared_error(y_test_clean, y_pred)
                    r2 = r2_score(y_test_clean, y_pred)
                    adj_r2 = adjusted_r2_score(r2, len(y_test_clean), X_test_clean.shape[1])
                    
                    st.success(f"Model berhasil dilatih! R²: {r2:.4f}" if st.session_state.language == 'id' else 
                             f"Model trained successfully! R²: {r2:.4f}")
                    st.write(f"MSE: {mse:.4f}")
                    st.write(f"Adjusted R²: {adj_r2:.4f}")
                    
                    # Actual vs Predicted plot
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.scatter(y_test_clean, y_pred, alpha=0.6)
                    ax.plot([y_test_clean.min(), y_test_clean.max()], 
                           [y_test_clean.min(), y_test_clean.max()], 'r--', lw=2)
                    ax.set_xlabel("Actual" if st.session_state.language == 'id' else "Actual")
                    ax.set_ylabel("Predicted" if st.session_state.language == 'id' else "Predicted")
                    ax.set_title("Actual vs Predicted" if st.session_state.language == 'id' else "Actual vs Predicted")
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Error training model: {str(e)}" if st.session_state.language == 'id' else f"Error training model: {str(e)}")

# Tab 5: SHAP Model Interpretation
with tab5:
    st.header("🔍 SHAP Model Interpretation" if st.session_state.language == 'id' else "🔍 SHAP Model Interpretation")
    
    if st.session_state.model is not None:
        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        
        try:
            # Create SHAP explainer
            if st.session_state.problem_type == "Classification":
                explainer = shap.TreeExplainer(model) if hasattr(model, 'feature_importances_') else shap.Explainer(model)
                shap_values = explainer.shap_values(X_test)
                
                # Summary plot
                st.subheader("SHAP Summary Plot" if st.session_state.language == 'id' else "SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, show=False)
                st.pyplot(fig)
                
                # Feature importance
                st.subheader("Feature Importance" if st.session_state.language == 'id' else "Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                st.pyplot(fig)
                
            else:
                explainer = shap.Explainer(model)
                shap_values = explainer(X_test)
                
                # Summary plot
                st.subheader("SHAP Summary Plot" if st.session_state.language == 'id' else "SHAP Summary Plot")
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values.values, X_test, show=False)
                st.pyplot(fig)
                
        except Exception as e:
            st.error(f"Error creating SHAP explanation: {str(e)}" if st.session_state.language == 'id' else f"Error creating SHAP explanation: {str(e)}")

# Tab 6: LIME Model Interpretation
with tab6:
    st.header("🔎 LIME Model Interpretation" if st.session_state.language == 'id' else "🔎 LIME Model Interpretation")
    
    if st.session_state.model is not None and LIME_AVAILABLE:
        model = st.session_state.model
        X_train = st.session_state.X_train
        X_test = st.session_state.X_test
        
        try:
            # Create LIME explainer
            explainer = lime_tabular.LimeTabularExplainer(
                X_train.values,
                feature_names=X_train.columns.tolist(),
                class_names=[st.session_state.target_column],
                mode='classification' if st.session_state.problem_type == "Classification" else 'regression'
            )
            
            # Select instance to explain
            instance_idx = st.slider("Pilih indeks instance" if st.session_state.language == 'id' else "Select instance index", 
                                     0, len(X_test) - 1, 0)
            
            explanation = explainer.explain_instance(
                X_test.iloc[instance_idx].values,
                model.predict_proba if st.session_state.problem_type == "Classification" else model.predict
            )
            
            # Show explanation
            st.subheader("LIME Explanation" if st.session_state.language == 'id' else "LIME Explanation")
            components = explanation.as_list()
            
            for feature, weight in components:
                st.write(f"{feature}: {weight:.4f}")
                
        except Exception as e:
            st.error(f"Error creating LIME explanation: {str(e)}" if st.session_state.language == 'id' else f"Error creating LIME explanation: {str(e)}")
    elif not LIME_AVAILABLE:
        st.warning("LIME tidak tersedia. Install dengan: pip install lime" if st.session_state.language == 'id' else 
                 "LIME not available. Install with: pip install lime")