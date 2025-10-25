import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
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
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from kmodes.kprototypes import KPrototypes
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')
import shap
import pickle
import os
from PIL import Image
import io
import time
import json
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from auth_db import auth_db
from captcha_utils import captcha_gen, verify_captcha
from utils import (prepare_timeseries_data, check_stationarity, plot_timeseries_analysis, 
                   analyze_trend_seasonality_cycle, plot_pattern_analysis,
                   implement_shap_classification, handle_multiclass_shap,
                   implement_lime_classification, detect_model_type,
                   prepare_forecasting_data_for_interpretation,
                   interpret_forecasting_model, create_forecasting_interpretation_dashboard)
from param_presets import get_available_presets, get_preset_params, get_all_presets, save_custom_preset, load_custom_presets, export_preset_to_json, import_preset_from_json, create_preset_summary

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

if 'model_results' not in st.session_state:
    st.session_state.model_results = []

# Time series specific state variables
if 'is_time_series' not in st.session_state:
    st.session_state.is_time_series = False
if 'time_column' not in st.session_state:
    st.session_state.time_column = None
if 'forecasting_models' not in st.session_state:
    st.session_state.forecasting_models = []
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None

# Authentication state variables
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'session_token' not in st.session_state:
    st.session_state.session_token = None
if 'show_register' not in st.session_state:
    st.session_state.show_register = False
if 'captcha_text' not in st.session_state:
    st.session_state.captcha_text = ""

# Create tabs for different functionalities
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📤 Data Upload", 
    "📊 Exploratory Data Analytic", 
    "🔄 Preprocessing and Feature Engineering", 
    "🛠️ Cross Validation and Model Training", 
    "🔍 SHAP Model Interpretation", 
    "🔎 LIME Model Interpretation",
    "⚠️ Time Series Anomaly Detection"
])

def calculate_comprehensive_clustering_metrics(X, labels, method_name=""):
    """Menghitung metrik evaluasi clustering yang komprehensif""" if st.session_state.language == 'id' else """Calculate comprehensive clustering evaluation metrics"""
    metrics = {}
    
    if len(set(labels)) <= 1:
        return {"error": "Hanya satu cluster atau noise ditemukan" if st.session_state.language == 'id' else "Only one cluster or noise found"}
    
    try:
        # Internal Validation Metrics
        metrics['silhouette_score'] = silhouette_score(X, labels)
        metrics['calinski_harabasz_score'] = calinski_harabasz_score(X, labels)
        metrics['davies_bouldin_score'] = davies_bouldin_score(X, labels)
        
        # Additional metrics
        metrics['n_clusters'] = len(set(labels)) - (1 if -1 in labels else 0)
        metrics['n_noise_points'] = (labels == -1).sum() if -1 in labels else 0
        
        # Cluster size distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        metrics['cluster_sizes'] = dict(zip(unique_labels.tolist(), counts.tolist()))
        metrics['cluster_size_std'] = np.std(counts)
        metrics['cluster_size_mean'] = np.mean(counts)
        
        # Compactness (average within-cluster sum of squares)
        compactness = 0
        for label in unique_labels:
            if label != -1:  # Exclude noise points
                cluster_points = X[labels == label]
                if len(cluster_points) > 0:
                    centroid = np.mean(cluster_points, axis=0)
                    compactness += np.sum((cluster_points - centroid) ** 2)
        metrics['within_cluster_ss'] = compactness
        
        # Separation (minimum distance between cluster centroids)
        if len(unique_labels) > 2 or (len(unique_labels) == 2 and -1 not in unique_labels):
            centroids = []
            for label in unique_labels:
                if label != -1:
                    cluster_points = X[labels == label]
                    if len(cluster_points) > 0:
                        centroids.append(np.mean(cluster_points, axis=0))
            
            if len(centroids) > 1:
                centroid_distances = euclidean_distances(centroids)
                np.fill_diagonal(centroid_distances, np.inf)
                metrics['min_centroid_distance'] = np.min(centroid_distances)
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics

def find_optimal_clusters_kmeans(X, max_k=10):
    """Mencari jumlah cluster optimal untuk K-Means""" if st.session_state.language == 'id' else """Find optimal number of clusters for K-Means"""
    if len(X) < 3:
        return 2, {}
    
    max_k = min(max_k, len(X) - 1)
    metrics = {'k': [], 'inertia': [], 'silhouette': [], 'calinski_harabasz': [], 'davies_bouldin': []}
    
    for k in range(2, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            metrics['k'].append(k)
            metrics['inertia'].append(kmeans.inertia_)
            
            if len(set(labels)) > 1:
                metrics['silhouette'].append(silhouette_score(X, labels))
                metrics['calinski_harabasz'].append(calinski_harabasz_score(X, labels))
                metrics['davies_bouldin'].append(davies_bouldin_score(X, labels))
            else:
                metrics['silhouette'].append(0)
                metrics['calinski_harabasz'].append(0)
                metrics['davies_bouldin'].append(np.inf)
                
        except Exception as e:
            continue
    
    # Find optimal k using multiple criteria
    optimal_k = 2
    if metrics['silhouette']:
        # Weighted combination of metrics
        silhouette_scores = np.array(metrics['silhouette'])
        calinski_scores = np.array(metrics['calinski_harabasz'])
        davies_bouldin_scores = np.array(metrics['davies_bouldin'])
        
        # Normalize scores
        silhouette_norm = (silhouette_scores - np.min(silhouette_scores)) / (np.max(silhouette_scores) - np.min(silhouette_scores) + 1e-10)
        calinski_norm = (calinski_scores - np.min(calinski_scores)) / (np.max(calinski_scores) - np.min(calinski_scores) + 1e-10)
        davies_bouldin_norm = 1 - ((davies_bouldin_scores - np.min(davies_bouldin_scores)) / (np.max(davies_bouldin_scores) - np.min(davies_bouldin_scores) + 1e-10))
        
        # Combined score (higher is better)
        combined_score = silhouette_norm + calinski_norm + davies_bouldin_norm
        optimal_idx = np.argmax(combined_score)
        optimal_k = metrics['k'][optimal_idx]
    
    return optimal_k, metrics

def find_optimal_eps_dbscan(X, min_samples_range=range(3, 8)):
    """Mencari parameter eps optimal untuk DBSCAN""" if st.session_state.language == 'id' else """Find optimal eps parameter for DBSCAN"""
    if len(X) < 3:
        return 0.5, 3, {}
    
    # Calculate k-distance graph for eps estimation
    k = 4  # Default k for k-distance
    if len(X) > k:
        distances = np.sort(np.mean(euclidean_distances(X)[:k], axis=1))
        # Find elbow point in k-distance graph
        diffs = np.diff(distances)
        elbow_idx = np.argmax(diffs) + 1 if len(diffs) > 0 else len(distances) // 2
        suggested_eps = distances[min(elbow_idx, len(distances) - 1)]
    else:
        suggested_eps = 0.5
    
    # Test different eps and min_samples combinations
    eps_range = np.linspace(max(0.1, suggested_eps * 0.5), min(5.0, suggested_eps * 2), 10)
    
    best_score = -1
    best_params = {'eps': 0.5, 'min_samples': 3}
    results = {'eps': [], 'min_samples': [], 'n_clusters': [], 'silhouette': [], 'n_noise': []}
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            try:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X)
                
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = (labels == -1).sum()
                
                results['eps'].append(eps)
                results['min_samples'].append(min_samples)
                results['n_clusters'].append(n_clusters)
                results['n_noise'].append(n_noise)
                
                if n_clusters > 1 and n_noise < len(X) * 0.3:  # Not too much noise
                    silhouette = silhouette_score(X, labels)
                    results['silhouette'].append(silhouette)
                    
                    if silhouette > best_score:
                        best_score = silhouette
                        best_params = {'eps': eps, 'min_samples': min_samples}
                else:
                    results['silhouette'].append(0)
                    
            except Exception as e:
                continue
    
    return best_params['eps'], best_params['min_samples'], results

def analyze_cluster_stability(X, labels, n_bootstrap=10, noise_level=0.05):
    """Analisis stabilitas clustering dengan bootstrap""" if st.session_state.language == 'id' else """Analyze clustering stability with bootstrap"""
    if len(set(labels)) <= 1:
        return {"stability_score": 0, "message": "Tidak cukup cluster untuk analisis stabilitas" if st.session_state.language == 'id' else "Not enough clusters for stability analysis"}
    
    original_labels = labels.copy()
    stability_scores = []
    
    for i in range(n_bootstrap):
        # Add small random noise to data
        X_noisy = X + np.random.normal(0, noise_level * np.std(X, axis=0), X.shape)
        
        try:
            # Re-cluster with same method (simplified for K-Means)
            if len(set(original_labels)) > 1:
                kmeans = KMeans(n_clusters=len(set(original_labels)), random_state=42 + i, n_init=10)
                new_labels = kmeans.fit_predict(X_noisy)
                
                # Calculate adjusted rand score between original and new labels
                if len(set(new_labels)) > 1:
                    stability_score = adjusted_rand_score(original_labels, new_labels)
                    stability_scores.append(stability_score)
        except Exception as e:
            continue
    
    if stability_scores:
        return {
            "stability_score": np.mean(stability_scores),
            "stability_std": np.std(stability_scores),
            "n_bootstrap": len(stability_scores)
        }
    else:
        return {"stability_score": 0, "message": "Tidak dapat menghitung stabilitas" if st.session_state.language == 'id' else "Cannot calculate stability"}

def analyze_cluster_characteristics(X, labels, feature_names=None):
    """Analisis karakteristik cluster berdasarkan fitur-fitur utama""" if st.session_state.language == 'id' else """Analyze cluster characteristics based on key features"""
    
    if feature_names is None:
        feature_names = X.columns if hasattr(X, 'columns') else [f'Feature_{i}' for i in range(X.shape[1])]
    
    cluster_profiles = {}
    unique_labels = np.unique(labels)
    
    for cluster_id in unique_labels:
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue
            
        cluster_mask = labels == cluster_id
        cluster_data = X[cluster_mask] if hasattr(X, 'iloc') else X[cluster_mask]
        
        if len(cluster_data) == 0:
            continue
            
        profile = {
            'size': np.sum(cluster_mask),
            'percentage': np.sum(cluster_mask) / len(labels) * 100,
            'mean_values': np.mean(cluster_data, axis=0) if hasattr(cluster_data, 'mean') else np.mean(cluster_data),
            'std_values': np.std(cluster_data, axis=0) if hasattr(cluster_data, 'std') else np.std(cluster_data),
            'min_values': np.min(cluster_data, axis=0) if hasattr(cluster_data, 'min') else np.min(cluster_data),
            'max_values': np.max(cluster_data, axis=0) if hasattr(cluster_data, 'max') else np.max(cluster_data)
        }
        
        # Add feature-specific statistics
        if hasattr(cluster_data, 'shape') and len(cluster_data.shape) > 1:
            feature_stats = {}
            for i, feature_name in enumerate(feature_names[:cluster_data.shape[1]]):
                # Handle different types of data structures (DataFrame vs NumPy array)
                if hasattr(cluster_data, 'iloc'):
                    # For pandas DataFrame
                    if isinstance(feature_name, str) and feature_name in cluster_data.columns:
                        col_data = cluster_data[feature_name]
                    else:
                        col_data = cluster_data.iloc[:, i]
                else:
                    # For NumPy array
                    col_data = cluster_data[:, i]
                    
                feature_stats[feature_name] = {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
            profile['feature_stats'] = feature_stats
        
        cluster_profiles[f'Cluster_{cluster_id}'] = profile
    
    return cluster_profiles

def generate_cluster_report(X, labels, algorithm_name, evaluation_metrics, stability_results, cluster_profiles):
    """Hasilkan laporan komprehensif clustering""" if st.session_state.language == 'id' else """Generate comprehensive clustering report"""
    
    report = []
    report.append("="*60)
    report.append(f"CLUSTERING ANALYSIS REPORT - {algorithm_name.upper()}")
    report.append("="*60)
    
    # Dataset overview
    report.append(f"\nDATASET OVERVIEW:")
    report.append(f"Total samples: {len(labels)}")
    report.append(f"Number of features: {X.shape[1] if hasattr(X, 'shape') else len(X[0])}")
    report.append(f"Number of clusters: {len(np.unique(labels))}")
    
    # Cluster distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    report.append(f"\nCLUSTER DISTRIBUTION:")
    for label, count in zip(unique_labels, counts):
        if label == -1:
            report.append(f"  Noise points: {count} ({count/len(labels)*100:.1f}%)")
        else:
            report.append(f"  Cluster {label}: {count} samples ({count/len(labels)*100:.1f}%)")
    
    # Evaluation metrics
    report.append(f"\nEVALUATION METRICS:")
    for metric_name, value in evaluation_metrics.items():
        if value is not None:
            if isinstance(value, (int, float)):
                report.append(f"  {metric_name}: {value:.4f}")
            elif isinstance(value, dict):
                report.append(f"  {metric_name}: {str(value)}")
            else:
                report.append(f"  {metric_name}: {value}")
        else:
            report.append(f"  {metric_name}: Not available")
    
    # Stability analysis
    report.append(f"\nSTABILITY ANALYSIS:")
    if stability_results:
        report.append(f"  Average Adjusted Rand Index: {stability_results.get('avg_ari', 'N/A')}")
        report.append(f"  Average Normalized Mutual Info: {stability_results.get('avg_nmi', 'N/A')}")
        report.append(f"  Stability Score: {stability_results.get('stability_score', 'N/A')}")
    else:
        report.append("  Stability analysis not performed")
    
    # Cluster characteristics
    if cluster_profiles:
        report.append(f"\nCLUSTER CHARACTERISTICS:")
        for cluster_name, profile in cluster_profiles.items():
            report.append(f"\n  {cluster_name}:")
            report.append(f"    Size: {profile['size']} samples ({profile['percentage']:.1f}%)")
            
            if 'feature_stats' in profile:
                report.append(f"    Key features:")
                # Show top 3 most distinctive features
                feature_means = [(name, stats['mean']) for name, stats in profile['feature_stats'].items()]
                feature_means.sort(key=lambda x: abs(x[1]), reverse=True)
                
                for feature_name, mean_val in feature_means[:3]:
                    report.append(f"      {feature_name}: {mean_val:.2f}")
    
    # Recommendations
    report.append(f"\nRECOMMENDATIONS:")
    
    # Based on cluster distribution
    if len(unique_labels) == 1:
        report.append("  ⚠️  Only one cluster found - consider adjusting parameters")
    elif any(count < len(labels) * 0.05 for count in counts):
        report.append("  ⚠️  Some clusters are very small - check for outliers")
    
    # Based on silhouette score
    if 'Silhouette Score' in evaluation_metrics and evaluation_metrics['Silhouette Score'] is not None:
        silhouette = evaluation_metrics['Silhouette Score']
        if silhouette > 0.5:
            report.append("  ✅ Strong cluster separation (Silhouette > 0.5)")
        elif silhouette > 0.25:
            report.append("  ℹ️  Moderate cluster separation (0.25 < Silhouette < 0.5)")
        else:
            report.append("  ⚠️  Weak cluster separation (Silhouette < 0.25)")
    
    # Based on stability
    if stability_results and 'stability_score' in stability_results:
        stability_score = stability_results['stability_score']
        if isinstance(stability_score, (int, float)):
            if stability_score > 0.8:
                report.append("  ✅ High cluster stability")
            elif stability_score > 0.6:
                report.append("  ℹ️  Moderate cluster stability")
            else:
                report.append("  ⚠️  Low cluster stability - results may vary")
    
    report.append("\n" + "="*60)
    
    return "\n".join(report)

def load_and_predict_model(model_file, data):
    """Memuat model dari file .pkl dan melakukan prediksi pada data baru"""
    try:
        # Load the model
        model = pickle.load(model_file)
        
        # Get model type
        model_type = type(model).__name__
        
        # Prepare data for prediction
        if hasattr(model, 'feature_names_in_'):
            # If model has feature names, use them
            required_features = model.feature_names_in_
            if set(required_features).issubset(set(data.columns)):
                X = data[required_features]
            else:
                missing_features = set(required_features) - set(data.columns)
                raise ValueError(f"Missing required features: {missing_features}")
        else:
            # Use all numeric columns
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("No numeric columns found in data")
            X = data[numeric_cols]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(X)
            
            # Get prediction probabilities if available (for classification)
            probabilities = None
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X)
                except:
                    pass
            
            return {
                'success': True,
                'predictions': predictions,
                'probabilities': probabilities,
                'model_type': model_type,
                'n_samples': len(predictions),
                'features_used': X.columns.tolist() if hasattr(X, 'columns') else list(range(X.shape[1]))
            }
        else:
            raise ValueError("Model does not have predict method")
            
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def adjusted_r2_score(r2, n, k):
    """Hitung Adjusted R².""" if st.session_state.language == 'id' else """Calculate Adjusted R²."""
    return 1 - (1 - r2) * (n - 1) / (n - k - 1)

def create_optuna_study(problem_type, model_type, X_train, y_train, cv_params, custom_param_ranges=None):
    
    def objective(trial):
        if problem_type == "Classification":
            if model_type == "Random Forest":
                # Gunakan rentang parameter kustom jika tersedia, jika tidak gunakan rentang default
                if custom_param_ranges and 'n_estimators' in custom_param_ranges:
                    n_estimators_range = custom_param_ranges['n_estimators']
                    if isinstance(n_estimators_range, list) and len(n_estimators_range) == 3:
                        n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1], step=n_estimators_range[2])
                    else:
                        n_estimators = trial.suggest_int('n_estimators', 50, 300)
                else:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                
                if custom_param_ranges and 'max_depth' in custom_param_ranges:
                    max_depth_range = custom_param_ranges['max_depth']
                    if isinstance(max_depth_range, list) and len(max_depth_range) == 3:
                        max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1], step=max_depth_range[2])
                    else:
                        max_depth = trial.suggest_int('max_depth', 3, 20)
                else:
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                
                if custom_param_ranges and 'min_samples_split' in custom_param_ranges:
                    min_samples_split_range = custom_param_ranges['min_samples_split']
                    if isinstance(min_samples_split_range, list) and len(min_samples_split_range) == 3:
                        min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1], step=min_samples_split_range[2])
                    else:
                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                else:
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                
                if custom_param_ranges and 'min_samples_leaf' in custom_param_ranges:
                    min_samples_leaf_range = custom_param_ranges['min_samples_leaf']
                    if isinstance(min_samples_leaf_range, list) and len(min_samples_leaf_range) == 3:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1], step=min_samples_leaf_range[2])
                    else:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                else:
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                
                if custom_param_ranges and 'max_features' in custom_param_ranges:
                    max_features_values = custom_param_ranges['max_features']
                    if isinstance(max_features_values, list):
                        max_features = trial.suggest_categorical('max_features', max_features_values)
                    else:
                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                else:
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                if custom_param_ranges and 'bootstrap' in custom_param_ranges:
                    bootstrap_values = custom_param_ranges['bootstrap']
                    if isinstance(bootstrap_values, list):
                        bootstrap = trial.suggest_categorical('bootstrap', bootstrap_values)
                    else:
                        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                else:
                    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'bootstrap': bootstrap,
                    'random_state': 42
                }
                model = RandomForestClassifier(**params)
            elif model_type == "Logistic Regression":
                params = {
                    'C': trial.suggest_float('C', 0.01, 10.0, log=True),
                    'max_iter': trial.suggest_int('max_iter', 100, 1000),
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga', 'newton-cg']),
                    'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet']),
                    'random_state': 42
                }
                # Handle elasticnet penalty yang memerlukan l1_ratio
                if params['penalty'] == 'elasticnet':
                    params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                model = LogisticRegression(**params)
            elif model_type == "SVM":
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'coef0': trial.suggest_float('coef0', 0.0, 1.0),
                    'shrinking': trial.suggest_categorical('shrinking', [True, False]),
                    'probability': True,
                    'random_state': 42
                }
                # Handle degree untuk kernel poly
                if params['kernel'] == 'poly':
                    params['degree'] = trial.suggest_int('degree', 2, 10)
                model = SVC(**params)
            elif model_type == "KNN":
                params = {
                    'n_neighbors': trial.suggest_int('n_neighbors', 1, 20),
                    'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                    'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
                    'p': trial.suggest_int('p', 1, 5),
                    'leaf_size': trial.suggest_int('leaf_size', 10, 50),
                    'metric': trial.suggest_categorical('metric', ['minkowski', 'euclidean', 'manhattan'])
                }
                model = KNeighborsClassifier(**params)
            elif model_type == "Decision Tree":
                params = {
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
                    'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 'auto']),
                    'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
                    'ccp_alpha': trial.suggest_float('ccp_alpha', 0.0, 0.1, step=0.001),
                    'random_state': 42
                }
                model = DecisionTreeClassifier(**params)
            elif model_type == "Gradient Boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None, 'auto']),
                    'random_state': 42
                }
                model = GradientBoostingClassifier(**params)
            elif model_type == "Naive Bayes":
                params = {
                    'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-6, log=True)
                }
                model = GaussianNB(**params)
            else:
                return 0
                
            score = cross_val_score(model, X_train, y_train, cv=cv_params['cv'], scoring='accuracy').mean()
            return score
            
        else:  # Regression
            if model_type == "Random Forest":
                # Gunakan rentang parameter kustom jika tersedia, jika tidak gunakan rentang default
                if custom_param_ranges and 'n_estimators' in custom_param_ranges:
                    n_estimators_range = custom_param_ranges['n_estimators']
                    if isinstance(n_estimators_range, list) and len(n_estimators_range) == 3:
                        n_estimators = trial.suggest_int('n_estimators', n_estimators_range[0], n_estimators_range[1], step=n_estimators_range[2])
                    else:
                        n_estimators = trial.suggest_int('n_estimators', 50, 300)
                else:
                    n_estimators = trial.suggest_int('n_estimators', 50, 300)
                
                if custom_param_ranges and 'max_depth' in custom_param_ranges:
                    max_depth_range = custom_param_ranges['max_depth']
                    if isinstance(max_depth_range, list) and len(max_depth_range) == 3:
                        max_depth = trial.suggest_int('max_depth', max_depth_range[0], max_depth_range[1], step=max_depth_range[2])
                    else:
                        max_depth = trial.suggest_int('max_depth', 3, 20)
                else:
                    max_depth = trial.suggest_int('max_depth', 3, 20)
                
                if custom_param_ranges and 'min_samples_split' in custom_param_ranges:
                    min_samples_split_range = custom_param_ranges['min_samples_split']
                    if isinstance(min_samples_split_range, list) and len(min_samples_split_range) == 3:
                        min_samples_split = trial.suggest_int('min_samples_split', min_samples_split_range[0], min_samples_split_range[1], step=min_samples_split_range[2])
                    else:
                        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                else:
                    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
                
                if custom_param_ranges and 'min_samples_leaf' in custom_param_ranges:
                    min_samples_leaf_range = custom_param_ranges['min_samples_leaf']
                    if isinstance(min_samples_leaf_range, list) and len(min_samples_leaf_range) == 3:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', min_samples_leaf_range[0], min_samples_leaf_range[1], step=min_samples_leaf_range[2])
                    else:
                        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                else:
                    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
                
                if custom_param_ranges and 'max_features' in custom_param_ranges:
                    max_features_values = custom_param_ranges['max_features']
                    if isinstance(max_features_values, list):
                        max_features = trial.suggest_categorical('max_features', max_features_values)
                    else:
                        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                else:
                    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
                
                if custom_param_ranges and 'bootstrap' in custom_param_ranges:
                    bootstrap_values = custom_param_ranges['bootstrap']
                    if isinstance(bootstrap_values, list):
                        bootstrap = trial.suggest_categorical('bootstrap', bootstrap_values)
                    else:
                        bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                else:
                    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'max_features': max_features,
                    'bootstrap': bootstrap,
                    'random_state': 42
                }
                model = RandomForestRegressor(**params)
            elif model_type == "Linear Regression":
                params = {
                    'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False]),
                    'positive': trial.suggest_categorical('positive', [True, False]),
                    'copy_X': trial.suggest_categorical('copy_X', [True, False])
                }
                model = LinearRegression(**params)
            elif model_type == "Gradient Boosting":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'subsample': trial.suggest_float('subsample', 0.7, 1.0),
                    'random_state': 42
                }
                model = GradientBoostingRegressor(**params)
            elif model_type == "SVR":
                params = {
                    'C': trial.suggest_float('C', 0.1, 10.0, log=True),
                    'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly']),
                    'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                    'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
                }
                model = SVR(**params)
            else:
                return 0
                
            score = cross_val_score(model, X_train, y_train, cv=cv_params['cv'], scoring='r2').mean()
            return score
    
    return objective

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

def parse_custom_range(range_str, param_type='int'):
    """
    Parse custom parameter range string into appropriate format
    
    Args:
        range_str: String format like "10:100:10" for numeric or "sqrt,log2,None" for categorical
        param_type: 'int', 'float', or 'categorical'
    
    Returns:
        List of values or range specification
    """
    if not range_str or range_str.strip() == "":
        return None
    
    range_str = range_str.strip()
    
    # Handle categorical parameters (comma-separated values)
    if ',' in range_str:
        values = [val.strip() for val in range_str.split(',')]
        # Convert numeric strings to numbers, handle None/True/False
        converted_values = []
        for val in values:
            if val.lower() == 'none':
                converted_values.append(None)
            elif val.lower() == 'true':
                converted_values.append(True)
            elif val.lower() == 'false':
                converted_values.append(False)
            else:
                try:
                    if param_type == 'int':
                        converted_values.append(int(val))
                    elif param_type == 'float':
                        converted_values.append(float(val))
                    else:
                        converted_values.append(val)
                except ValueError:
                    converted_values.append(val)
        return converted_values
    
    # Handle numeric ranges (min:max:step format)
    elif ':' in range_str:
        parts = range_str.split(':')
        if len(parts) == 3:
            try:
                start = int(parts[0]) if param_type == 'int' else float(parts[0])
                stop = int(parts[1]) if param_type == 'int' else float(parts[1])
                step = int(parts[2]) if param_type == 'int' else float(parts[2])
                
                if param_type == 'int':
                    return list(range(start, stop + 1, step))
                else:
                    values = []
                    current = start
                    while current <= stop:
                        values.append(current)
                        current += step
                    return values
            except ValueError:
                return None
    
    # Single value
    try:
        if param_type == 'int':
            return [int(range_str)]
        elif param_type == 'float':
            return [float(range_str)]
        else:
            return [range_str]
    except ValueError:
        return None

def get_custom_param_inputs(model_name, use_custom_ranges, st_session):
    """
    Get custom parameter range inputs from user
    
    Args:
        model_name: Name of the model
        use_custom_ranges: Whether custom ranges are enabled
        st_session: Streamlit session object (for language settings)
    
    Returns:
        Dictionary of custom parameter ranges
    """
    custom_ranges = {}
    
    if not use_custom_ranges:
        return custom_ranges
    
    # Tambahkan fitur preset parameter
    available_presets = get_available_presets(model_name)
    
    if available_presets:
        st.subheader("🎯 Preset Parameter" if st_session.language == 'id' else "🎯 Parameter Presets")
        
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            selected_preset = st.selectbox(
                "Pilih preset:" if st_session.language == 'id' else "Select preset:",
                ["None"] + available_presets,
                help="Pilih preset parameter yang telah dikonfigurasi sebelumnya" if st_session.language == 'id' else "Select pre-configured parameter preset"
            )
        
        with col2:
            if st.button("Terapkan Preset" if st_session.language == 'id' else "Apply Preset"):
                if selected_preset != "None":
                    preset_params = get_preset_params(model_name, selected_preset)
                    custom_ranges.update(preset_params)
                    st.success(f"Preset '{selected_preset}' diterapkan!" if st_session.language == 'id' else f"Preset '{selected_preset}' applied!")
        
        with col3:
            if st.button("Lihat Detail" if st_session.language == 'id' else "View Details"):
                if selected_preset != "None":
                    preset_params = get_preset_params(model_name, selected_preset)
                    summary = create_preset_summary(model_name, selected_preset, preset_params)
                    st.markdown(summary)
        
        st.markdown("---")
    
    # Tambahkan fitur impor/ekspor preset
    with st.expander("💾 Impor/Ekspor Preset" if st_session.language == 'id' else "💾 Import/Export Presets"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Impor Preset" if st_session.language == 'id' else "Import Preset")
            uploaded_file = st.file_uploader(
                "Pilih file JSON preset:" if st_session.language == 'id' else "Choose preset JSON file:",
                type=['json'],
                key=f"preset_import_{model_name}"
            )
            if uploaded_file is not None:
                try:
                    imported_data = json.load(uploaded_file)
                    if st.button("Impor" if st_session.language == 'id' else "Import"):
                        if all(key in imported_data for key in ["model_type", "preset_name", "parameters"]):
                            custom_ranges.update(imported_data["parameters"])
                            st.success(f"Preset '{imported_data['preset_name']}' berhasil diimpor!" if st_session.language == 'id' else f"Preset '{imported_data['preset_name']}' successfully imported!")
                except Exception as e:
                    st.error(f"Error mengimpor preset: {str(e)}" if st_session.language == 'id' else f"Error importing preset: {str(e)}")
        
        with col2:
            st.subheader("Ekspor Preset" if st_session.language == 'id' else "Export Preset")
            if custom_ranges:
                preset_name_export = st.text_input(
                    "Nama preset untuk diekspor:" if st_session.language == 'id' else "Preset name to export:",
                    value=f"custom_{model_name.lower()}",
                    key=f"preset_export_name_{model_name}"
                )
                if st.button("Ekspor ke JSON" if st_session.language == 'id' else "Export to JSON"):
                    filename = export_preset_to_json(model_name, preset_name_export, custom_ranges)
                    if filename:
                        with open(filename, 'rb') as f:
                            st.download_button(
                                label="Unduh File JSON" if st_session.language == 'id' else "Download JSON File",
                                data=f.read(),
                                file_name=filename,
                                mime="application/json"
                            )
                        st.success(f"Preset berhasil diekspor ke {filename}!" if st_session.language == 'id' else f"Preset successfully exported to {filename}!")
            else:
                st.info("Tidak ada parameter kustom untuk diekspor" if st_session.language == 'id' else "No custom parameters to export")

def merge_custom_param_ranges(default_param_grid, custom_param_ranges):
    """
    Menggabungkan parameter grid default dengan custom parameter ranges
    
    Parameters:
    -----------
    default_param_grid : dict
        Parameter grid default dari aplikasi
    custom_param_ranges : dict
        Custom parameter ranges dari user input
        
    Returns:
    --------
    dict: Parameter grid yang sudah digabung
    """
    if custom_param_ranges is None or not custom_param_ranges:
        return default_param_grid
    
    # Salin parameter grid default
    merged_grid = default_param_grid.copy()
    
    # Override dengan custom parameter ranges
    for param, custom_range in custom_param_ranges.items():
        if custom_range is not None and len(custom_range) > 0:
            merged_grid[param] = custom_range
    
    return merged_grid

def validate_param_ranges(param_grid, X_train, model_type):
    """
    Validasi parameter ranges berdasarkan karakteristik data
    
    Parameters:
    -----------
    param_grid : dict
        Parameter grid yang akan divalidasi
    X_train : pandas.DataFrame
        Data training untuk validasi
    model_type : str
        Tipe model untuk validasi khusus
        
    Returns:
    --------
    dict: Parameter grid yang sudah divalidasi
    """
    if X_train is None or X_train.empty:
        return param_grid
    
    validated_grid = param_grid.copy()
    n_features = X_train.shape[1]
    n_samples = len(X_train)
    
    # Validasi untuk max_depth (Random Forest, Decision Tree, Gradient Boosting)
    if 'max_depth' in validated_grid and model_type in ['Random Forest', 'Decision Tree', 'Gradient Boosting']:
        max_possible_depth = int(np.log2(n_samples)) if n_samples > 1 else 1
        validated_max_depth = []
        for depth in validated_grid['max_depth']:
            if isinstance(depth, (int, float)) and depth > 0:
                if depth <= max_possible_depth:
                    validated_max_depth.append(int(depth))
                else:
                    validated_max_depth.append(max_possible_depth)
            elif depth is None:
                validated_max_depth.append(depth)
        validated_grid['max_depth'] = validated_max_depth
    
    # Validasi untuk max_features (Random Forest, Gradient Boosting)
    if 'max_features' in validated_grid and model_type in ['Random Forest', 'Gradient Boosting']:
        validated_max_features = []
        for feature in validated_grid['max_features']:
            if isinstance(feature, (int, float)) and feature > 0:
                if feature <= n_features:
                    validated_max_features.append(int(feature))
                else:
                    validated_max_features.append(n_features)
            else:
                validated_max_features.append(feature)
        validated_grid['max_features'] = validated_max_features
    
    # Validasi untuk n_neighbors (KNN)
    if 'n_neighbors' in validated_grid and model_type == 'KNN':
        max_neighbors = min(n_samples - 1, 50)  # Batasi maksimal 50 atau n_samples-1
        validated_n_neighbors = []
        for k in validated_grid['n_neighbors']:
            if isinstance(k, (int, float)) and k > 0:
                if k <= max_neighbors:
                    validated_n_neighbors.append(int(k))
                else:
                    validated_n_neighbors.append(max_neighbors)
        validated_grid['n_neighbors'] = validated_n_neighbors
    
    # Validasi untuk min_samples_split dan min_samples_leaf
    if 'min_samples_split' in validated_grid:
        min_samples = 2
        max_samples = max(2, n_samples // 10)  # Maksimal 10% dari data
        validated_min_samples = []
        for samples in validated_grid['min_samples_split']:
            if isinstance(samples, (int, float)) and samples >= 1:
                if samples <= max_samples:
                    validated_min_samples.append(int(samples))
                else:
                    validated_min_samples.append(max_samples)
        validated_grid['min_samples_split'] = validated_min_samples
    
    if 'min_samples_leaf' in validated_grid:
        max_samples = max(1, n_samples // 20)  # Maksimal 5% dari data
        validated_min_samples = []
        for samples in validated_grid['min_samples_leaf']:
            if isinstance(samples, (int, float)) and samples >= 1:
                if samples <= max_samples:
                    validated_min_samples.append(int(samples))
                else:
                    validated_min_samples.append(max_samples)
        validated_grid['min_samples_leaf'] = validated_min_samples
    
    return validated_grid
    
    # Model-specific parameter ranges
    if model_name == "Random Forest":
        st_session.subheader("Rentang Parameter Kustom - Random Forest" if st_session.language == 'id' else "Custom Parameter Ranges - Random Forest")
        
        col1, col2 = st_session.columns(2)
        with col1:
            n_estimators_range = st_session.text_input(
                "Jumlah pohon (n_estimators):" if st_session.language == 'id' else "Number of trees (n_estimators):",
                placeholder="50:300:50",
                help="Format: min:max:step" if st_session.language == 'id' else "Format: min:max:step"
            )
            max_depth_range = st_session.text_input(
                "Kedalaman maksimum (max_depth):" if st_session.language == 'id' else "Maximum depth (max_depth):",
                placeholder="3:20:1",
                help="Format: min:max:step" if st_session.language == 'id' else "Format: min:max:step"
            )
            min_samples_split_range = st_session.text_input(
                "Min samples split:" if st_session.language == 'id' else "Min samples split:",
                placeholder="2:10:1"
            )
        with col2:
            min_samples_leaf_range = st_session.text_input(
                "Min samples leaf:" if st_session.language == 'id' else "Min samples leaf:",
                placeholder="1:5:1"
            )
            max_features_range = st_session.text_input(
                "Max features:" if st_session.language == 'id' else "Max features:",
                placeholder="sqrt,log2,None",
                help="Pisahkan dengan koma" if st_session.language == 'id' else "Separate with commas"
            )
            bootstrap_range = st_session.text_input(
                "Bootstrap:" if st_session.language == 'id' else "Bootstrap:",
                placeholder="True,False"
            )
        
        # Parse ranges
        if n_estimators_range:
            custom_ranges['n_estimators'] = parse_custom_range(n_estimators_range, 'int')
        if max_depth_range:
            custom_ranges['max_depth'] = parse_custom_range(max_depth_range, 'int')
        if min_samples_split_range:
            custom_ranges['min_samples_split'] = parse_custom_range(min_samples_split_range, 'int')
        if min_samples_leaf_range:
            custom_ranges['min_samples_leaf'] = parse_custom_range(min_samples_leaf_range, 'int')
        if max_features_range:
            custom_ranges['max_features'] = parse_custom_range(max_features_range, 'categorical')
        if bootstrap_range:
            custom_ranges['bootstrap'] = parse_custom_range(bootstrap_range, 'categorical')
    
    return custom_ranges

def get_research_type_recommendations(data, problem_type=None):
    """
    Mendapatkan rekomendasi jenis-jenis penelitian yang sesuai dengan dataset
    dan referensi jurnal ilmiah menggunakan agentic AI
    """
    recommendations = []
    
    # Analisis karakteristik dataset
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Tentukan tipe masalah jika belum ditentukan
    if problem_type is None:
        if len(numerical_cols) > 0 and len(categorical_cols) > 0:
            problem_type = "classification"  # Default
        elif len(datetime_cols) > 0:
            problem_type = "time_series"
        else:
            problem_type = "regression"
    
    # Jenis-jenis penelitian berdasarkan tipe data dan domain
    
    # 1. Penelitian Klasifikasi (untuk data dengan target kategorikal)
    if problem_type == "classification":
        recommendations.append({
            'type': 'classification',
            'title': 'Penelitian Klasifikasi Medis' if st.session_state.language == 'id' else 'Medical Classification Research',
            'description': 'Mengklasifikasikan kondisi medis, diagnosis penyakit, atau prediksi outcome pasien.' if st.session_state.language == 'id' else 'Classifying medical conditions, disease diagnosis, or patient outcome prediction.',
            'research_methods': [
                '• Cross-sectional Study Design',
                '• Case-Control Study',
                '• Cohort Study (Retrospective/Prospective)',
                '• Diagnostic Accuracy Study'
            ],
            'ml_approaches': [
                '• Random Forest untuk interpretabilitas',
                '• Support Vector Machine untuk high-dimensional data',
                '• Neural Networks untuk pola kompleks',
                '• Ensemble Methods untuk akurasi tinggi'
            ],
            'journals': [
                {
                    'title': 'Journal of Medical Internet Research (JMIR)',
                    'scope': 'Digital health dan AI dalam kedokteran',
                    'impact_factor': '7.4',
                    'example_title': 'Machine Learning for Early Detection of Diabetes: A Systematic Review and Meta-Analysis',
                    'doi': '10.2196/12345'
                },
                {
                    'title': 'IEEE Journal of Biomedical and Health Informatics',
                    'scope': 'Biomedical informatics dan health AI',
                    'impact_factor': '5.8',
                    'example_title': 'Deep Learning Approaches for Medical Image Classification: A Comprehensive Review',
                    'doi': '10.1109/JBHI.2023.1234567'
                },
                {
                    'title': 'BMC Medical Informatics and Decision Making',
                    'scope': 'Medical informatics dan clinical decision support',
                    'impact_factor': '3.5',
                    'example_title': 'Comparative Analysis of Machine Learning Models for Cardiovascular Risk Prediction',
                    'doi': '10.1186/s12911-023-12345'
                }
            ]
        })
    
    # 2. Penelitian Prediksi/Regresi (untuk data dengan target numerik)
    if problem_type == "regression":
        recommendations.append({
            'type': 'regression',
            'title': 'Penelitian Prediksi Kuantitatif' if st.session_state.language == 'id' else 'Quantitative Prediction Research',
            'description': 'Memprediksi nilai kontinu seperti biomarker, durasi perawatan, atau biaya medis.' if st.session_state.language == 'id' else 'Predicting continuous values such as biomarkers, treatment duration, or medical costs.',
            'research_methods': [
                '• Longitudinal Study Design',
                '• Prospective Cohort Study',
                '• Time-to-Event Analysis (jika ada durasi)',
                '• Economic Evaluation Study'
            ],
            'ml_approaches': [
                '• Linear Regression untuk baseline',
                '• Gradient Boosting untuk non-linear patterns',
                '• Support Vector Regression untuk high-dimensional',
                '• Random Forest untuk interpretabilitas'
            ],
            'journals': [
                {
                    'title': 'Statistics in Medicine',
                    'scope': 'Statistical methods dalam medical research',
                    'impact_factor': '2.3',
                    'example_title': 'Predictive Modeling of Hospital Readmission Using Machine Learning',
                    'doi': '10.1002/sim.12345'
                },
                {
                    'title': 'Computer Methods and Programs in Biomedicine',
                    'scope': 'Computational methods dalam biomedicine',
                    'impact_factor': '4.9',
                    'example_title': 'Machine Learning Models for Predicting Blood Glucose Levels in Diabetic Patients',
                    'doi': '10.1016/j.cmpb.2023.123456'
                },
                {
                    'title': 'Artificial Intelligence in Medicine',
                    'scope': 'AI applications dalam medicine',
                    'impact_factor': '5.1',
                    'example_title': 'Comparative Study of Regression Models for Medical Outcome Prediction',
                    'doi': '10.1016/j.artmed.2023.123456'
                }
            ]
        })
    
    # 3. Penelitian Time Series (untuk data temporal)
    if len(datetime_cols) > 0 or problem_type == "time_series":
        recommendations.append({
            'type': 'time_series',
            'title': 'Penelitian Time Series dan Forecasting' if st.session_state.language == 'id' else 'Time Series and Forecasting Research',
            'description': 'Menganalisis tren temporal, memprediksi future values, atau mendeteksi anomali.' if st.session_state.language == 'id' else 'Analyzing temporal trends, predicting future values, or detecting anomalies.',
            'research_methods': [
                '• Interrupted Time Series Analysis',
                '• Seasonal Trend Analysis',
                '• Forecasting Accuracy Study',
                '• Anomaly Detection Study'
            ],
            'ml_approaches': [
                '• ARIMA/SARIMA untuk data stasioner',
                '• LSTM untuk pola jangka panjang',
                '• Prophet untuk data dengan musiman',
                '• Exponential Smoothing untuk tren sederhana'
            ],
            'journals': [
                {
                    'title': 'International Journal of Forecasting',
                    'scope': 'Forecasting theory dan applications',
                    'impact_factor': '4.2',
                    'example_title': 'Machine Learning Approaches for Healthcare Demand Forecasting',
                    'doi': '10.1016/j.ijforecast.2023.123456'
                },
                {
                    'title': 'Journal of Time Series Analysis',
                    'scope': 'Time series theory dan methodology',
                    'impact_factor': '1.8',
                    'example_title': 'Deep Learning Models for Medical Time Series Prediction',
                    'doi': '10.1111/jtsa.12345'
                },
                {
                    'title': 'PLOS ONE',
                    'scope': 'Multidisciplinary research',
                    'impact_factor': '3.7',
                    'example_title': 'Time Series Analysis of Epidemiological Data Using Machine Learning',
                    'doi': '10.1371/journal.pone.1234567'
                }
            ]
        })
    
    # 4. Penelitian Clustering/Grouping (untuk data tanpa target)
    if problem_type is None or len(data.select_dtypes(include=['int64', 'float64']).columns) > 0:
        recommendations.append({
            'type': 'clustering',
            'title': 'Penelitian Pengelompokan dan Segmentasi' if st.session_state.language == 'id' else 'Clustering and Segmentation Research',
            'description': 'Mengidentifikasi subkelompok dalam populasi, segmentasi pasien, atau pola tersembunyi.' if st.session_state.language == 'id' else 'Identifying subgroups in population, patient segmentation, or hidden patterns.',
            'research_methods': [
                '• Cross-sectional Descriptive Study',
                '• Exploratory Data Analysis',
                '• Pattern Recognition Study',
                '• Phenotype Discovery Study'
            ],
            'ml_approaches': [
                '• K-Means untuk clustering sederhana',
                '• Hierarchical Clustering untuk hierarki',
                '• DBSCAN untuk density-based clustering',
                '• Gaussian Mixture Models untuk probabilistik'
            ],
            'journals': [
                {
                    'title': 'BMC Medical Research Methodology',
                    'scope': 'Medical research methodology',
                    'impact_factor': '3.9',
                    'example_title': 'Unsupervised Learning for Patient Phenotyping in Electronic Health Records',
                    'doi': '10.1186/s12874-023-12345'
                },
                {
                    'title': 'Journal of Biomedical Informatics',
                    'scope': 'Biomedical informatics research',
                    'impact_factor': '4.5',
                    'example_title': 'Clustering Analysis for Disease Subtyping Using Clinical Data',
                    'doi': '10.1016/j.jbi.2023.123456'
                },
                {
                    'title': 'Scientific Reports',
                    'scope': 'Multidisciplinary scientific research',
                    'impact_factor': '4.6',
                    'example_title': 'Machine Learning-Based Clustering for Precision Medicine Applications',
                    'doi': '10.1038/s41598-023-12345'
                }
            ]
        })
    
    # 5. Penelitian Anomali/Outlier (untuk data dengan anomali)
    if len(numerical_cols) > 0:
        recommendations.append({
            'type': 'anomaly',
            'title': 'Penelitian Deteksi Anomali' if st.session_state.language == 'id' else 'Anomaly Detection Research',
            'description': 'Mendeteksi kejadian abnormal, fraud detection, atau early warning systems.' if st.session_state.language == 'id' else 'Detecting abnormal events, fraud detection, or early warning systems.',
            'research_methods': [
                '• Case-Control Study for Anomalies',
                '• Surveillance Study Design',
                '• Quality Control Study',
                '• Early Detection System Evaluation'
            ],
            'ml_approaches': [
                '• Isolation Forest untuk outlier detection',
                '• One-Class SVM untuk novelty detection',
                '• Autoencoders untuk reconstruction error',
                '• Statistical Methods (Z-score, IQR)'
            ],
            'journals': [
                {
                    'title': 'Expert Systems with Applications',
                    'scope': 'Expert systems dan AI applications',
                    'impact_factor': '8.5',
                    'example_title': 'Anomaly Detection in Healthcare Data Using Machine Learning: A Systematic Review',
                    'doi': '10.1016/j.eswa.2023.123456'
                },
                {
                    'title': 'Knowledge-Based Systems',
                    'scope': 'Knowledge-based AI systems',
                    'impact_factor': '8.8',
                    'example_title': 'Deep Learning Approaches for Medical Anomaly Detection',
                    'doi': '10.1016/j.knosys.2023.123456'
                },
                {
                    'title': 'Computers in Biology and Medicine',
                    'scope': 'Computational methods dalam biology dan medicine',
                    'impact_factor': '4.9',
                    'example_title': 'Machine Learning for Detection of Abnormal Clinical Patterns',
                    'doi': '10.1016/j.compbiomed.2023.123456'
                }
            ]
        })
    
    return recommendations

def recommend_research_methods(data):
    """Rekomendasikan metode penelitian berdasarkan karakteristik dataset"""
    recommendations = []
    
    # Analisis karakteristik dataset
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    
    # Rekomendasi berdasarkan ukuran dataset
    if n_rows < 100:
        recommendations.append({
            'type': 'warning',
            'title': 'Dataset Kecil' if st.session_state.language == 'id' else 'Small Dataset',
            'description': 'Gunakan Leave-One-Out Cross Validation atau k-fold dengan k yang besar untuk validasi model.' if st.session_state.language == 'id' else 'Use Leave-One-Out Cross Validation or k-fold with large k for model validation.',
            'methods': ['Leave-One-Out CV', 'Stratified K-Fold (k=5-10)', 'Simple models (Logistic/Linear Regression)']
        })
    elif 100 <= n_rows < 1000:
        recommendations.append({
            'type': 'info',
            'title': 'Dataset Sedang' if st.session_state.language == 'id' else 'Medium Dataset',
            'description': 'Gunakan Stratified K-Fold Cross Validation dengan k=5 atau 10 untuk hasil yang stabil.' if st.session_state.language == 'id' else 'Use Stratified K-Fold Cross Validation with k=5 or 10 for stable results.',
            'methods': ['Stratified K-Fold CV', 'Grid Search CV', 'Random Forest, SVM, Neural Networks']
        })
    else:
        recommendations.append({
            'type': 'success',
            'title': 'Dataset Besar' if st.session_state.language == 'id' else 'Large Dataset',
            'description': 'Dataset cukup besar untuk deep learning dan ensemble methods yang kompleks.' if st.session_state.language == 'id' else 'Dataset is large enough for complex deep learning and ensemble methods.',
            'methods': ['K-Fold CV', 'Hold-out Validation', 'Deep Learning, Gradient Boosting, XGBoost']
        })
    
    # Rekomendasi berdasarkan missing values
    if missing_percentage > 20:
        recommendations.append({
            'type': 'warning',
            'title': 'Banyak Missing Values' if st.session_state.language == 'id' else 'High Missing Values',
            'description': f'Terdapat {missing_percentage:.1f}% missing values. Pertimbangkan imputasi atau analisis sensitivitas.' if st.session_state.language == 'id' else f'There are {missing_percentage:.1f}% missing values. Consider imputation or sensitivity analysis.',
            'methods': ['Multiple Imputation', 'KNN Imputation', 'Missing Indicator Features', 'Tree-based Models']
        })
    elif missing_percentage > 5:
        recommendations.append({
            'type': 'info',
            'title': 'Missing Values Moderat' if st.session_state.language == 'id' else 'Moderate Missing Values',
            'description': f'Terdapat {missing_percentage:.1f}% missing values. Gunakan imputasi yang sesuai.' if st.session_state.language == 'id' else f'There are {missing_percentage:.1f}% missing values. Use appropriate imputation.',
            'methods': ['Mean/Median Imputation', 'KNN Imputation', 'MICE', 'Model-based Imputation']
        })
    
    # Rekomendasi berdasarkan jenis data
    if len(categorical_cols) > len(numerical_cols):
        recommendations.append({
            'type': 'info',
            'title': 'Data Dominan Kategorikal' if st.session_state.language == 'id' else 'Categorical Dominant Data',
            'description': 'Dataset memiliki lebih banyak fitur kategorikal. Gunakan encoding yang tepat.' if st.session_state.language == 'id' else 'Dataset has more categorical features. Use appropriate encoding.',
            'methods': ['One-Hot Encoding', 'Target Encoding', 'Ordinal Encoding', 'Tree-based Models']
        })
    elif len(numerical_cols) > len(categorical_cols):
        recommendations.append({
            'type': 'info',
            'title': 'Data Dominan Numerik' if st.session_state.language == 'id' else 'Numerical Dominant Data',
            'description': 'Dataset memiliki lebih banyak fitur numerik. Scaling mungkin diperlukan.' if st.session_state.language == 'id' else 'Dataset has more numerical features. Scaling may be needed.',
            'methods': ['StandardScaler', 'MinMaxScaler', 'RobustScaler', 'PCA for Dimensionality Reduction']
        })
    
    # Rekomendasi berdasarkan jumlah fitur
    if n_cols > 50:
        recommendations.append({
            'type': 'warning',
            'title': 'Dimensi Tinggi' if st.session_state.language == 'id' else 'High Dimensionality',
            'description': f'Terdapat {n_cols} fitur. Pertimbangkan reduksi dimensi untuk menghindari overfitting.' if st.session_state.language == 'id' else f'There are {n_cols} features. Consider dimensionality reduction to avoid overfitting.',
            'methods': ['PCA', 'Feature Selection (RFE, SelectKBest)', 'L1 Regularization', 'Autoencoders']
        })
    elif n_cols > 20:
        recommendations.append({
            'type': 'info',
            'title': 'Banyak Fitur' if st.session_state.language == 'id' else 'Many Features',
            'description': f'Terdapat {n_cols} fitur. Gunakan feature selection untuk meningkatkan performa.' if st.session_state.language == 'id' else f'There are {n_cols} features. Use feature selection to improve performance.',
            'methods': ['Feature Importance', 'Recursive Feature Elimination', 'Mutual Information', 'Correlation Analysis']
        })
    
    # Rekomendasi umum
    recommendations.append({
        'type': 'success',
        'title': 'Langkah Selanjutnya' if st.session_state.language == 'id' else 'Next Steps',
        'description': 'Ikuti langkah-langkah berikut untuk analisis yang komprehensif.' if st.session_state.language == 'id' else 'Follow these steps for comprehensive analysis.',
        'methods': [
            '1. Exploratory Data Analysis (EDA)',
            '2. Preprocessing & Feature Engineering',
            '3. Model Training & Cross Validation',
            '4. Model Interpretation (SHAP/LIME)',
            '5. Hyperparameter Tuning',
            '6. Final Model Evaluation'
        ]
    })
    
    return recommendations

def analyze_dataset_with_ai(data, analysis_type='comprehensive'):
    """
    Enhanced AI-powered dataset analysis using rule-based agentic system
    that provides intelligent recommendations based on dataset characteristics
    """
    recommendations = []
    
    # Basic dataset characteristics
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = data.select_dtypes(include=['datetime64']).columns.tolist()
    
    # Advanced analysis
    missing_values = data.isnull().sum().sum()
    missing_percentage = (missing_values / (n_rows * n_cols)) * 100 if n_rows * n_cols > 0 else 0
    duplicate_rows = data.duplicated().sum()
    
    # Statistical analysis
    if len(numerical_cols) > 0:
        numerical_data = data[numerical_cols]
        skewness_values = numerical_data.skew().abs().mean()
        kurtosis_values = numerical_data.kurtosis().abs().mean()
        outlier_percentage = ((numerical_data > numerical_data.quantile(0.95)).any(axis=1).sum() / n_rows) * 100
    else:
        skewness_values = 0
        kurtosis_values = 0
        outlier_percentage = 0
    
    # AI-powered analysis based on dataset patterns
    
    # 1. Dataset Quality Assessment
    quality_score = 100
    quality_issues = []
    
    if missing_percentage > 10:
        quality_score -= 20
        quality_issues.append("High missing values")
    if duplicate_rows > n_rows * 0.05:
        quality_score -= 15
        quality_issues.append("Significant duplicates")
    if outlier_percentage > 10:
        quality_score -= 10
        quality_issues.append("High outlier presence")
    if n_rows < 50:
        quality_score -= 25
        quality_issues.append("Very small dataset")
    
    recommendations.append({
        'type': 'success' if quality_score >= 80 else 'warning' if quality_score >= 60 else 'error',
        'title': 'Dataset Quality Assessment' if st.session_state.language == 'id' else 'Dataset Quality Assessment',
        'description': f"Dataset Quality Score: {quality_score:.1f}/100. {'Excellent quality' if quality_score >= 80 else 'Good quality with minor issues' if quality_score >= 60 else 'Requires significant preprocessing'}.",
        'details': quality_issues,
        'priority': 'high'
    })
    
    # 2. Optimal ML Algorithm Recommendations
    ml_recommendations = []
    
    # Rule-based algorithm selection
    if n_rows < 100:
        ml_recommendations.extend(['Naive Bayes', 'Logistic Regression', 'Decision Tree'])
    elif n_rows < 1000:
        if len(categorical_cols) > len(numerical_cols):
            ml_recommendations.extend(['Random Forest', 'XGBoost', 'LightGBM'])
        else:
            ml_recommendations.extend(['SVM', 'Random Forest', 'Neural Network'])
    else:
        if n_cols > 50:
            ml_recommendations.extend(['XGBoost', 'Deep Learning', 'Ensemble Methods'])
        else:
            ml_recommendations.extend(['Gradient Boosting', 'Neural Networks', 'Stacking Ensemble'])
    
    # Special cases
    if missing_percentage > 30:
        ml_recommendations = ['Random Forest', 'XGBoost', 'KNN Imputation + Any Model']
    
    if len(datetime_cols) > 0:
        ml_recommendations.append('Time Series Models (ARIMA, Prophet, LSTM)')
    
    recommendations.append({
        'type': 'info',
        'title': 'AI-Recommended ML Algorithms' if st.session_state.language == 'id' else 'AI-Recommended ML Algorithms',
        'description': 'Optimal algorithms based on dataset characteristics and proven ML practices.' if st.session_state.language == 'id' else 'Optimal algorithms based on dataset characteristics and proven ML practices.',
        'methods': ml_recommendations,
        'priority': 'high'
    })
    
    # 3. Feature Engineering Recommendations
    feature_recommendations = []
    
    if len(numerical_cols) > 0:
        if skewness_values > 1:
            feature_recommendations.append('Log transformation for skewed features')
        if outlier_percentage > 5:
            feature_recommendations.append('Outlier treatment (Winsorization/IQR)')
        feature_recommendations.append('StandardScaler/MinMaxScaler normalization')
    
    if len(categorical_cols) > 0:
        if len(categorical_cols) < 10:
            feature_recommendations.append('One-Hot Encoding for categorical variables')
        else:
            feature_recommendations.append('Target/Frequency Encoding for high-cardinality categoricals')
    
    if n_rows > 1000 and len(numerical_cols) > 5:
        feature_recommendations.append('Polynomial features for non-linear relationships')
        feature_recommendations.append('Feature interaction terms')
    
    recommendations.append({
        'type': 'info',
        'title': 'Smart Feature Engineering' if st.session_state.language == 'id' else 'Smart Feature Engineering',
        'description': 'Automated feature engineering suggestions based on data patterns.' if st.session_state.language == 'id' else 'Automated feature engineering suggestions based on data patterns.',
        'methods': feature_recommendations,
        'priority': 'medium'
    })
    
    # 4. Cross-Validation Strategy
    cv_recommendations = []
    
    if n_rows < 100:
        cv_recommendations.append('Leave-One-Out Cross-Validation (LOOCV)')
        cv_recommendations.append('Stratified K-Fold with k=5')
    elif n_rows < 1000:
        cv_recommendations.append('Stratified K-Fold with k=5 or 10')
        cv_recommendations.append('Repeated Stratified K-Fold (n_repeats=3)')
    else:
        cv_recommendations.append('Stratified K-Fold with k=5')
        cv_recommendations.append('Hold-out validation (80-20 split)')
        cv_recommendations.append('Time-based split for temporal data')
    
    if missing_percentage > 20:
        cv_recommendations.append('Nested cross-validation for hyperparameter tuning')
    
    recommendations.append({
        'type': 'info',
        'title': 'Optimal Cross-Validation Strategy' if st.session_state.language == 'id' else 'Optimal Cross-Validation Strategy',
        'description': 'Best validation approach based on dataset size and characteristics.' if st.session_state.language == 'id' else 'Best validation approach based on dataset size and characteristics.',
        'methods': cv_recommendations,
        'priority': 'medium'
    })
    
    # 5. Advanced Analytics Recommendations
    advanced_recommendations = []
    
    if len(numerical_cols) > 2:
        advanced_recommendations.append('Principal Component Analysis (PCA)')
        advanced_recommendations.append('Clustering analysis (K-means, DBSCAN)')
    
    if len(categorical_cols) > 1:
        advanced_recommendations.append('Association Rule Mining')
        advanced_recommendations.append('Chi-square test for independence')
    
    if n_rows > 500 and len(numerical_cols) > 3:
        advanced_recommendations.append('SHAP values for model interpretability')
        advanced_recommendations.append('Partial Dependence Plots (PDP)')
    
    if duplicate_rows > 0:
        advanced_recommendations.append('Anomaly detection for data quality')
    
    recommendations.append({
        'type': 'info',
        'title': 'Advanced Analytics' if st.session_state.language == 'id' else 'Advanced Analytics',
        'description': 'Sophisticated analysis techniques for deeper insights.' if st.session_state.language == 'id' else 'Sophisticated analysis techniques for deeper insights.',
        'methods': advanced_recommendations,
        'priority': 'low'
    })
    
    # 6. Problem-Specific Recommendations
    if analysis_type == 'classification':
        class_recommendations = []
        if len(categorical_cols) > 0:
            target_col = categorical_cols[0] if len(categorical_cols) > 0 else None
            if target_col and data[target_col].nunique() == 2:
                class_recommendations.append('Binary classification metrics (Precision, Recall, F1, AUC)')
            elif target_col:
                class_recommendations.append('Multi-class classification metrics (Accuracy, F1-macro, Cohen\'s Kappa)')
        
        recommendations.append({
            'type': 'info',
            'title': 'Classification-Specific Recommendations' if st.session_state.language == 'id' else 'Classification-Specific Recommendations',
            'description': 'Specialized techniques for classification problems.' if st.session_state.language == 'id' else 'Specialized techniques for classification problems.',
            'methods': class_recommendations,
            'priority': 'high'
        })
    
    elif analysis_type == 'regression':
        reg_recommendations = []
        if len(numerical_cols) > 0:
            reg_recommendations.append('Regression metrics (RMSE, MAE, R², Adjusted R²)')
            reg_recommendations.append('Residual analysis and diagnostics')
            reg_recommendations.append('Feature scaling for regularization methods')
        
        recommendations.append({
            'type': 'info',
            'title': 'Regression-Specific Recommendations' if st.session_state.language == 'id' else 'Regression-Specific Recommendations',
            'description': 'Specialized techniques for regression problems.' if st.session_state.language == 'id' else 'Specialized techniques for regression problems.',
            'methods': reg_recommendations,
            'priority': 'high'
        })
    
    return recommendations

def create_agentic_ai_analysis(data, analysis_type='comprehensive', language='id'):
    """
    Create an agentic AI analysis that simulates intelligent reasoning about the dataset
    """
    agent_analysis = {
        'agent_name': 'DataScience-AI-Agent',
        'analysis_timestamp': pd.Timestamp.now(),
        'dataset_summary': {},
        'intelligent_insights': [],
        'actionable_recommendations': [],
        'risk_assessment': [],
        'success_probability': 0
    }
    
    # Dataset summary
    n_rows, n_cols = data.shape
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    agent_analysis['dataset_summary'] = {
        'total_rows': n_rows,
        'total_columns': n_cols,
        'numerical_features': len(numerical_cols),
        'categorical_features': len(categorical_cols),
        'data_density': ((data.size - data.isnull().sum().sum()) / data.size) * 100,
        'complexity_score': min(100, (n_cols * 2) + (n_rows / 1000))
    }
    
    # Intelligent insights
    insights = []
    
    # Language-specific messages
    if language == 'id':
        success_msg = 'Dataset sangat cocok untuk machine learning dengan jumlah sampel dan fitur yang memadai.'
        adequate_msg = 'Dataset cukup untuk tugas ML dasar tetapi dapat diperbaiki dengan rekayasa fitur.'
        small_msg = 'Dataset mungkin terlalu kecil untuk hasil machine learning yang dapat diandalkan.'
        low_complexity_msg = 'Kompleksitas dataset rendah - model sederhana direkomendasikan.'
        moderate_complexity_msg = 'Kompleksitas sedang - metode ensemble direkomendasikan.'
        high_complexity_msg = 'Kompleksitas tinggi - teknik lanjutan diperlukan.'
    else:
        success_msg = 'Dataset is well-suited for machine learning with sufficient samples and features.'
        adequate_msg = 'Dataset is adequate for basic ML tasks but may benefit from feature engineering.'
        small_msg = 'Dataset may be too small for reliable machine learning results.'
        low_complexity_msg = 'Dataset complexity is low - simple models are recommended.'
        moderate_complexity_msg = 'Moderate complexity - ensemble methods are recommended.'
        high_complexity_msg = 'High complexity - advanced techniques required.'
    
    # Insight 1: Dataset readiness
    if n_rows >= 1000 and n_cols >= 5:
        insights.append({
            'type': 'success',
            'insight': success_msg,
            'confidence': 0.95,
            'evidence': f'{n_rows} sampel dan {n_cols} fitur memberikan kekuatan statistik yang baik.' if language == 'id' else f'{n_rows} samples and {n_cols} features provide good statistical power.'
        })
    elif n_rows >= 100 and n_cols >= 3:
        insights.append({
            'type': 'info',
            'insight': adequate_msg,
            'confidence': 0.80,
            'evidence': f'{n_rows} sampel dan {n_cols} fitur cukup untuk model sederhana.' if language == 'id' else f'{n_rows} samples and {n_cols} features are sufficient for simple models.'
        })
    else:
        insights.append({
            'type': 'warning',
            'insight': small_msg,
            'confidence': 0.90,
            'evidence': f'Hanya {n_rows} sampel dan {n_cols} fitur dapat menyebabkan overfitting.' if language == 'id' else f'Only {n_rows} samples and {n_cols} features may lead to overfitting.'
        })
    
    # Insight 2: Feature quality
    if len(numerical_cols) > 0:
        missing_pct = (data[numerical_cols].isnull().sum().sum() / data[numerical_cols].size) * 100
        if missing_pct < 5:
            insights.append({
                'type': 'success',
                'insight': 'Fitur numerik memiliki kelengkapan yang sangat baik.' if language == 'id' else 'Numerical features have excellent completeness.',
                'confidence': 0.90,
                'evidence': f'Hanya {missing_pct:.1f}% nilai yang hilang di kolom numerik.' if language == 'id' else f'Only {missing_pct:.1f}% missing values in numerical columns.'
            })
        elif missing_pct < 20:
            insights.append({
                'type': 'info',
                'insight': 'Fitur numerik memiliki kelengkapan yang dapat diterima dengan nilai hilang yang dapat dikelola.' if language == 'id' else 'Numerical features have acceptable completeness with manageable missing values.',
                'confidence': 0.85,
                'evidence': f'{missing_pct:.1f}% nilai hilang dapat ditangani secara efektif dengan imputasi.' if language == 'id' else f'{missing_pct:.1f}% missing values can be effectively handled with imputation.'
            })
    
    # Insight 3: Model complexity recommendation
    complexity_factors = []
    if n_cols > 20:
        complexity_factors.append('Dimensi tinggi' if language == 'id' else 'High dimensionality')
    if missing_pct > 15:
        complexity_factors.append('Data hilang signifikan' if language == 'id' else 'Significant missing data')
    if n_rows > 10000:
        complexity_factors.append('Ukuran dataset besar' if language == 'id' else 'Large dataset size')
    
    if len(complexity_factors) == 0:
        insights.append({
            'type': 'success',
            'insight': low_complexity_msg,
            'confidence': 0.90,
            'evidence': 'Data bersih dan terstruktur dengan baik, cocok untuk model yang dapat ditafsirkan.' if language == 'id' else 'Clean, well-structured data suitable for interpretable models.'
        })
    elif len(complexity_factors) <= 2:
        insights.append({
            'type': 'info',
            'insight': moderate_complexity_msg,
            'confidence': 0.85,
            'evidence': f'Faktor: {", ".join(complexity_factors)}' if language == 'id' else f'Factors: {", ".join(complexity_factors)}'
        })
    else:
        insights.append({
            'type': 'warning',
            'insight': high_complexity_msg,
            'confidence': 0.80,
            'evidence': f'Beberapa faktor kompleksitas: {", ".join(complexity_factors)}' if language == 'id' else f'Multiple complexity factors: {", ".join(complexity_factors)}'
        })
    
    agent_analysis['intelligent_insights'] = insights
    
    # Actionable recommendations
    recommendations = []
    
    # Language-specific recommendation messages
    if language == 'id':
        imp_action = 'Terapkan strategi imputasi lanjutan'
        scaling_action = 'Terapkan penskalaan dan transformasi fitur'
        ensemble_action = 'Gunakan metode ensemble dengan penyetelan hyperparameter'
        interpretable_action = 'Mulai dengan model yang dapat ditafsirkan'
        
        imp_rationale = f'{missing_pct:.1f}% nilai hilang memerlukan penanganan yang canggih'
        scaling_rationale = 'Beberapa fitur numerik mendapat manfaat dari standarisasi'
        ensemble_rationale = 'Dataset besar dapat mendukung model kompleks dengan validasi yang tepat'
        interpretable_rationale = 'Ukuran dataset sedang cocok untuk pendekatan yang seimbang'
        
        imp_implementation = 'Gunakan imputasi iteratif atau metode imputasi berbasis model'
        scaling_implementation = 'Gunakan StandardScaler dan pertimbangkan transformasi log untuk fitur yang miring'
        ensemble_implementation = 'Terapkan Random Forest atau XGBoost dengan optimasi Optuna'
        interpretable_implementation = 'Gunakan Regresi Logistik atau Pohon Keputusan dengan validasi silang'
        
        imp_impact = 'Tingkatkan performa model sebesar 15-25%'
        scaling_impact = 'Tingkatkan konvergensi dan performa model'
        ensemble_impact = 'Capai akurasi prediksi 85-95%'
        interpretable_impact = 'Capai akurasi prediksi 75-85% dengan dapat ditafsirkan'
    else:
        imp_action = 'Implement advanced imputation strategy'
        scaling_action = 'Apply feature scaling and transformation'
        ensemble_action = 'Use ensemble methods with hyperparameter tuning'
        interpretable_action = 'Start with interpretable models'
        
        imp_rationale = f'{missing_pct:.1f}% missing values require sophisticated handling'
        scaling_rationale = 'Multiple numerical features benefit from standardization'
        ensemble_rationale = 'Large dataset can support complex models with proper validation'
        interpretable_rationale = 'Moderate dataset size suitable for balanced approach'
        
        imp_implementation = 'Use iterative imputation or model-based imputation methods'
        scaling_implementation = 'Use StandardScaler and consider log transformations for skewed features'
        ensemble_implementation = 'Implement Random Forest or XGBoost with Optuna optimization'
        interpretable_implementation = 'Use Logistic Regression or Decision Tree with cross-validation'
        
        imp_impact = 'Improve model performance by 15-25%'
        scaling_impact = 'Improve model convergence and performance'
        ensemble_impact = 'Achieve 85-95% prediction accuracy'
        interpretable_impact = 'Achieve 75-85% prediction accuracy with interpretability'
    
    # Recommendation 1: Preprocessing strategy
    if missing_pct > 10:
        recommendations.append({
            'action': imp_action,
            'priority': 'high',
            'rationale': imp_rationale,
            'implementation': imp_implementation,
            'expected_impact': imp_impact
        })
    
    # Recommendation 2: Feature engineering
    if len(numerical_cols) > 3:
        recommendations.append({
            'action': scaling_action,
            'priority': 'medium',
            'rationale': scaling_rationale,
            'implementation': scaling_implementation,
            'expected_impact': scaling_impact
        })
    
    # Recommendation 3: Model selection
    if n_rows > 1000 and n_cols > 10:
        recommendations.append({
            'action': ensemble_action,
            'priority': 'high',
            'rationale': ensemble_rationale,
            'implementation': ensemble_implementation,
            'expected_impact': ensemble_impact
        })
    elif n_rows > 100:
        recommendations.append({
            'action': interpretable_action,
            'priority': 'medium',
            'rationale': interpretable_rationale,
            'implementation': interpretable_implementation,
            'expected_impact': interpretable_impact
        })
    
    agent_analysis['actionable_recommendations'] = recommendations
    
    # Risk assessment
    risks = []
    
    # Language-specific risk messages
    if language == 'id':
        overfitting_risk = 'Risiko overfitting tinggi'
        missing_bias_risk = 'Model bias karena pola data hilang'
        dimensionality_risk = 'Kutukan dimensionalitas'
        
        overfitting_mitigation = 'Gunakan model sederhana, regularisasi agresif, dan validasi ekstensif'
        missing_bias_mitigation = 'Analisis pola data hilang dan gunakan beberapa strategi imputasi'
        dimensionality_mitigation = 'Terapkan teknik reduksi dimensionalitas sebelum pemodelan'
    else:
        overfitting_risk = 'High overfitting risk'
        missing_bias_risk = 'Biased model due to missing data patterns'
        dimensionality_risk = 'Curse of dimensionality'
        
        overfitting_mitigation = 'Use simple models, aggressive regularization, and extensive validation'
        missing_bias_mitigation = 'Analyze missing data patterns and use multiple imputation strategies'
        dimensionality_mitigation = 'Apply dimensionality reduction techniques before modeling'
    
    if n_rows < 100:
        risks.append({
            'risk': overfitting_risk,
            'severity': 'high',
            'mitigation': overfitting_mitigation,
            'probability': 0.8
        })
    
    if missing_pct > 20:
        risks.append({
            'risk': missing_bias_risk,
            'severity': 'medium',
            'mitigation': missing_bias_mitigation,
            'probability': 0.6
        })
    
    if n_cols > 50:
        risks.append({
            'risk': dimensionality_risk,
            'severity': 'medium',
            'mitigation': dimensionality_mitigation,
            'probability': 0.7
        })
    
    agent_analysis['risk_assessment'] = risks
    
    # Success probability calculation
    success_factors = []
    if n_rows >= 200:
        success_factors.append(0.2)
    if missing_pct < 15:
        success_factors.append(0.2)
    if n_cols >= 3:
        success_factors.append(0.2)
    if len(complexity_factors) <= 1:
        success_factors.append(0.2)
    if len(risks) <= 1:
        success_factors.append(0.2)
    
    agent_analysis['success_probability'] = min(0.95, sum(success_factors))
    agent_analysis['language'] = language
    
    return agent_analysis

def verify_captcha(input_text, correct_text):
    """Verify captcha input"""
    return input_text.upper().strip() == correct_text.upper().strip()

def verify_captcha(input_text, correct_text):
    """Verify captcha input"""
    return input_text.upper().strip() == correct_text.upper().strip()

def authenticate_user(username, password):
    """Authenticate user with database"""
    user = auth_db.authenticate_user(username, password)
    if user:
        st.session_state.authenticated = True
        st.session_state.current_user = user['username']
        st.session_state.session_token = auth_db.create_session(user['username'])
        return True
    return False

def register_user(username, password, email):
    """Register new user"""
    if auth_db.create_user(username, password, email):
        return True
    return False

def logout_user():
    """Logout current user"""
    if st.session_state.session_token:
        auth_db.delete_session(st.session_state.session_token)
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.session_state.session_token = None

def show_login_page():
    """Display login page"""
    st.title("🔐 ASMERANDA" if st.session_state.language == 'id' else "🔐 Login - EDA & ML Application")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Captcha refresh button outside form
        col_captcha1, col_captcha2 = st.columns([3, 1])
        with col_captcha1:
            if 'captcha_image' not in st.session_state:
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
            st.image(st.session_state.captcha_image, width=200)
        with col_captcha2:
            if st.button("🔄 Refresh", key="refresh_captcha_login"):
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
                st.rerun()
        
        with st.form("login_form"):
            st.subheader("Login" if st.session_state.language == 'id' else "Login")
            
            username = st.text_input(
                "Username" if st.session_state.language == 'id' else "Username",
                placeholder="Enter your username"
            )
            password = st.text_input(
                "Password" if st.session_state.language == 'id' else "Password",
                type="password",
                placeholder="Enter your password"
            )
            
            captcha_input = st.text_input(
                "Enter Captcha" if st.session_state.language == 'id' else "Enter Captcha",
                placeholder="Enter the text above"
            )
            
            col_login, col_register = st.columns(2)
            with col_login:
                login_submitted = st.form_submit_button("Login", type="primary")
            with col_register:
                register_btn = st.form_submit_button("Register")
        
        if register_btn:
            st.session_state.show_register = True
            st.rerun()
        
        if login_submitted:
            if not username or not password:
                st.error("Please enter both username and password" if st.session_state.language == 'en' else "Silakan masukkan username dan password")
            elif not captcha_input:
                st.error("Please enter the captcha" if st.session_state.language == 'en' else "Silakan masukkan captcha")
            elif not verify_captcha(captcha_input, st.session_state.captcha_text):
                st.error("Invalid captcha. Please try again" if st.session_state.language == 'en' else "Captcha tidak valid. Silakan coba lagi")
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
                st.rerun()
            elif authenticate_user(username, password):
                st.success(f"Welcome {username}!" if st.session_state.language == 'en' else f"Selamat datang {username}!")
                st.rerun()
            else:
                st.error("Invalid username or password" if st.session_state.language == 'en' else "Username atau password tidak valid")
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
                st.rerun()

def show_register_page():
    """Display registration page"""
    st.title("📝 Register - EDA & ML Application" if st.session_state.language == 'id' else "📝 Register - EDA & ML Application")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Captcha refresh button outside form
        col_captcha1, col_captcha2 = st.columns([3, 1])
        with col_captcha1:
            if 'captcha_image' not in st.session_state:
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
            st.image(st.session_state.captcha_image, width=200)
        with col_captcha2:
            if st.button("🔄 Refresh", key="refresh_captcha_register"):
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
                st.rerun()
        
        with st.form("register_form"):
            st.subheader("Create New Account" if st.session_state.language == 'id' else "Create New Account")
            
            new_username = st.text_input(
                "Username" if st.session_state.language == 'id' else "Username",
                placeholder="Choose a username",
                help="Username must be at least 3 characters"
            )
            new_email = st.text_input(
                "Email (Optional)" if st.session_state.language == 'id' else "Email (Optional)",
                placeholder="your@email.com"
            )
            new_password = st.text_input(
                "Password" if st.session_state.language == 'id' else "Password",
                type="password",
                placeholder="Choose a strong password"
            )
            confirm_password = st.text_input(
                "Confirm Password" if st.session_state.language == 'id' else "Confirm Password",
                type="password",
                placeholder="Confirm your password"
            )
            
            captcha_input = st.text_input(
                "Enter Captcha" if st.session_state.language == 'id' else "Enter Captcha",
                placeholder="Enter the text above"
            )
            
            col_register, col_back = st.columns(2)
            with col_register:
                register_submitted = st.form_submit_button("Create Account", type="primary")
            with col_back:
                back_btn = st.form_submit_button("Back to Login")
        
        if back_btn:
            st.session_state.show_register = False
            st.rerun()
        
        if register_submitted:
            if not new_username or not new_password:
                st.error("Username and password are required" if st.session_state.language == 'en' else "Username dan password wajib diisi")
            elif len(new_username) < 3:
                st.error("Username must be at least 3 characters" if st.session_state.language == 'en' else "Username minimal 3 karakter")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters" if st.session_state.language == 'en' else "Password minimal 6 karakter")
            elif new_password != confirm_password:
                st.error("Passwords do not match" if st.session_state.language == 'en' else "Password tidak cocok")
            elif not captcha_input:
                st.error("Please enter the captcha" if st.session_state.language == 'en' else "Silakan masukkan captcha")
            elif not verify_captcha(captcha_input, st.session_state.captcha_text):
                st.error("Invalid captcha. Please try again" if st.session_state.language == 'en' else "Captcha tidak valid. Silakan coba lagi")
                st.session_state.captcha_image, st.session_state.captcha_text = captcha_gen.get_captcha_base64()
                st.rerun()
            elif not auth_db.is_username_available(new_username):
                st.error("Username already exists" if st.session_state.language == 'en' else "Username sudah digunakan")
            elif register_user(new_username, new_password, new_email if new_email else None):
                st.success("Account created successfully! Please login." if st.session_state.language == 'en' else "Akun berhasil dibuat! Silakan login.")
                st.session_state.show_register = False
                st.rerun()
            else:
                st.error("Failed to create account" if st.session_state.language == 'en' else "Gagal membuat akun")

# Main authentication check
if not st.session_state.authenticated:
    if st.session_state.get('show_register', False):
        show_register_page()
    else:
        show_login_page()
    st.stop()

# Main application content (after successful authentication)
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Logged in as:** {st.session_state.current_user}")
if st.sidebar.button("Logout", key="logout_btn"):
    logout_user()
    st.rerun()

# Tab 1: Data Upload
with tab1:
    st.header("Unggah Dataset Anda" if st.session_state.language == 'id' else "Upload Your Dataset")
    
    # Model Prediction Section
    st.subheader("🔮 Prediksi Data Baru" if st.session_state.language == 'id' else "🔮 Predict New Data")
    
    col_model, col_data = st.columns(2)
    
    with col_model:
        model_file = st.file_uploader(
            "Unggah model .pkl" if st.session_state.language == 'id' else "Upload .pkl model file",
            type=['pkl'],
            key="model_uploader"
        )
    
    with col_data:
        prediction_data_file = st.file_uploader(
            "Unggah data untuk prediksi (CSV)" if st.session_state.language == 'id' else "Upload data for prediction (CSV)",
            type=['csv'],
            key="prediction_data_uploader"
        )
    
    if model_file and prediction_data_file:
        try:
            # Load prediction data
            prediction_df = pd.read_csv(prediction_data_file)
            
            # Load model and make predictions
            prediction_result = load_and_predict_model(model_file, prediction_df)
            
            if prediction_result['success']:
                st.success("Prediksi berhasil!" if st.session_state.language == 'id' else "Prediction successful!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"Model: {prediction_result['model_type']}")
                    st.info(f"Jumlah sampel: {prediction_result['n_samples']}" if st.session_state.language == 'id' else f"Number of samples: {prediction_result['n_samples']}")
                    st.info(f"Fitur digunakan: {len(prediction_result['features_used'])}")
                
                with col2:
                    # Display prediction statistics
                    predictions = prediction_result['predictions']
                    if len(np.unique(predictions)) <= 10:  # Classification
                        unique, counts = np.unique(predictions, return_counts=True)
                        pred_counts = dict(zip(unique, counts))
                        st.write("**Distribusi Prediksi:**" if st.session_state.language == 'id' else "**Prediction Distribution:**")
                        for pred, count in pred_counts.items():
                            st.write(f"- {pred}: {count} ({count/len(predictions)*100:.1f}%)")
                    else:  # Regression
                        st.write("**Statistik Prediksi:**" if st.session_state.language == 'id' else "**Prediction Statistics:**")
                        st.write(f"Mean: {np.mean(predictions):.2f}")
                        st.write(f"Std: {np.std(predictions):.2f}")
                        st.write(f"Min: {np.min(predictions):.2f}")
                        st.write(f"Max: {np.max(predictions):.2f}")
                
                # Display predictions table
                st.write("**Hasil Prediksi:**" if st.session_state.language == 'id' else "**Prediction Results:**")
                result_df = prediction_df.copy()
                result_df['Prediksi'] = predictions
                
                # Add probabilities if available
                if prediction_result['probabilities'] is not None:
                    prob_df = pd.DataFrame(prediction_result['probabilities'])
                    prob_cols = [f'Prob_Kelas_{i}' for i in range(prob_df.shape[1])]
                    prob_df.columns = prob_cols
                    result_df = pd.concat([result_df, prob_df], axis=1)
                
                st.dataframe(result_df.head(100))  # Show first 100 rows
                
                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Unduh Hasil Prediksi (CSV)" if st.session_state.language == 'id' else "📥 Download Prediction Results (CSV)",
                    data=csv,
                    file_name=f"prediksi_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
            else:
                st.error(f"Gagal melakukan prediksi: {prediction_result['error']}" if st.session_state.language == 'id' else f"Prediction failed: {prediction_result['error']}")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    # Single Prediction Section
    if model_file:
        st.markdown("---")
        st.subheader("🔮 Prediksi Manual Satu Data" if st.session_state.language == 'id' else "🔮 Manual Single Prediction")
        
        # Load model to get feature information
        try:
            model = pickle.load(model_file)
            model_file.seek(0)  # Reset file pointer for future use
            
            # Get feature names if available
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
                st.info(f"Model membutuhkan {len(feature_names)} fitur: {', '.join(feature_names)}" if st.session_state.language == 'id' else f"Model requires {len(feature_names)} features: {', '.join(feature_names)}")
                
                # Create input form for each feature
                with st.form("single_prediction_form"):
                    st.write("**Masukkan nilai fitur:**" if st.session_state.language == 'id' else "**Enter feature values:**")
                    
                    input_data = {}
                    cols = st.columns(2)
                    for i, feature in enumerate(feature_names):
                        with cols[i % 2]:
                            input_data[feature] = st.number_input(
                                f"{feature}",
                                value=0.0,
                                step=0.01,
                                key=f"feature_{feature}"
                            )
                    
                    col_predict, col_clear = st.columns(2)
                    with col_predict:
                        predict_button = st.form_submit_button(
                            "🔮 Lakukan Prediksi" if st.session_state.language == 'id' else "🔮 Make Prediction",
                            type="primary"
                        )
                    with col_clear:
                        clear_button = st.form_submit_button(
                            "🗑️ Bersihkan" if st.session_state.language == 'id' else "🗑️ Clear"
                        )
                
                if predict_button:
                    try:
                        # Create DataFrame from input data
                        input_df = pd.DataFrame([input_data])
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        
                        # Display result
                        st.success("Prediksi Berhasil!" if st.session_state.language == 'id' else "Prediction Successful!")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(
                                "Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result",
                                f"{prediction:.4f}" if isinstance(prediction, (int, float)) else str(prediction)
                            )
                        
                        with col2:
                            # Show probabilities if classification
                            if hasattr(model, 'predict_proba'):
                                try:
                                    probabilities = model.predict_proba(input_df)[0]
                                    st.write("**Probabilitas Kelas:**" if st.session_state.language == 'id' else "**Class Probabilities:**")
                                    for i, prob in enumerate(probabilities):
                                        st.write(f"Kelas {i}: {prob:.4f} ({prob*100:.2f}%)" if st.session_state.language == 'id' else f"Class {i}: {prob:.4f} ({prob*100:.2f}%)")
                                except:
                                    pass
                        
                        # Show input summary
                        with st.expander("📋 Ringkasan Input" if st.session_state.language == 'id' else "📋 Input Summary"):
                            st.write("**Nilai Fitur yang Dimasukkan:**" if st.session_state.language == 'id' else "**Entered Feature Values:**")
                            for feature, value in input_data.items():
                                st.write(f"- {feature}: {value}")
                        
                    except Exception as e:
                        st.error(f"Error dalam prediksi: {str(e)}" if st.session_state.language == 'id' else f"Error in prediction: {str(e)}")
                
            else:
                st.warning("Model tidak memiliki informasi fitur. Pastikan data yang dimasukkan sesuai dengan training data." if st.session_state.language == 'id' else "Model doesn't have feature information. Make sure input data matches training data.")
                
                # Simple text area for manual input
                manual_input = st.text_area(
                    "Masukkan data (format: fitur1,nilai1;fitur2,nilai2)" if st.session_state.language == 'id' else "Enter data (format: feature1,value1;feature2,value2)",
                    placeholder="contoh: age,25;income,50000;score,85" if st.session_state.language == 'id' else "example: age,25;income,50000;score,85"
                )
                
                if st.button("🔮 Lakukan Prediksi" if st.session_state.language == 'id' else "🔮 Make Prediction"):
                    try:
                        # Parse manual input
                        data_pairs = manual_input.split(';')
                        input_dict = {}
                        for pair in data_pairs:
                            if ',' in pair:
                                feature, value = pair.split(',')
                                input_dict[feature.strip()] = float(value.strip())
                        
                        # Create DataFrame
                        input_df = pd.DataFrame([input_dict])
                        
                        # Make prediction
                        prediction = model.predict(input_df)[0]
                        
                        st.success("Prediksi Berhasil!" if st.session_state.language == 'id' else "Prediction Successful!")
                        st.metric(
                            "Hasil Prediksi" if st.session_state.language == 'id' else "Prediction Result",
                            f"{prediction:.4f}" if isinstance(prediction, (int, float)) else str(prediction)
                        )
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}. Pastikan format input benar." if st.session_state.language == 'id' else f"Error: {str(e)}. Make sure input format is correct.")
                        
        except Exception as e:
            st.error(f"Error memuat model: {str(e)}" if st.session_state.language == 'id' else f"Error loading model: {str(e)}")
    
    st.markdown("---")
    
    # Section title for new dataset training
    st.subheader("📚 Latih Dataset Baru" if st.session_state.language == 'id' else "📚 Train New Dataset")
    
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
            
            # Time Series Detection
            st.subheader("Deteksi Dataset Time Series" if st.session_state.language == 'id' else "Time Series Dataset Detection")
            
            # Check if this is a time series dataset
            is_time_series = st.checkbox(
                "Apakah ini dataset time series?" if st.session_state.language == 'id' else "Is this a time series dataset?",
                value=st.session_state.is_time_series,
                help="Centang jika dataset ini berisi data time series untuk forecasting" if st.session_state.language == 'id' else "Check if this dataset contains time series data for forecasting"
            )
            
            st.session_state.is_time_series = is_time_series
            
            if is_time_series:
                st.info("Dataset akan diproses sebagai time series untuk analisis forecasting" if st.session_state.language == 'id' else "Dataset will be processed as time series for forecasting analysis")
                
                # Select time column
                date_columns = data.select_dtypes(include=['object', 'datetime64']).columns.tolist()
                numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                
                col1, col2 = st.columns(2)
                with col1:
                    time_column = st.selectbox(
                        "Pilih kolom waktu/date:" if st.session_state.language == 'id' else "Select time/date column:",
                        [""] + date_columns + numeric_columns,
                        index=0 if st.session_state.time_column is None else ([""] + date_columns + numeric_columns).index(st.session_state.time_column)
                    )
                    st.session_state.time_column = time_column if time_column else None
                
                with col2:
                    if numeric_columns:
                        # Validasi target_column yang tersimpan di session state
                        default_index = 0
                        if st.session_state.target_column is not None:
                            available_columns = [""] + numeric_columns
                            if st.session_state.target_column in available_columns:
                                default_index = available_columns.index(st.session_state.target_column)
                            else:
                                # Reset jika kolom target tidak valid
                                st.session_state.target_column = None
                        
                        target_column = st.selectbox(
                            "Pilih kolom target untuk forecasting:" if st.session_state.language == 'id' else "Select target column for forecasting:",
                            [""] + numeric_columns,
                            index=default_index
                        )
                        st.session_state.target_column = target_column if target_column else None
                    else:
                        st.warning("Tidak ada kolom numerik untuk forecasting" if st.session_state.language == 'id' else "No numeric columns for forecasting")
                        
                if time_column and target_column:
                    # Validate time column
                    try:
                        if data[time_column].dtype == 'object':
                            data[time_column] = pd.to_datetime(data[time_column])
                        
                        # Check if time column is monotonic
                        is_monotonic = data[time_column].is_monotonic_increasing
                        
                        if not is_monotonic:
                            st.warning("Kolom waktu tidak berurutan. Data akan diurutkan berdasarkan waktu." if st.session_state.language == 'id' else "Time column is not sequential. Data will be sorted by time.")
                            data = data.sort_values(by=time_column)
                            st.session_state.data = data
                        
                        st.success(f"Dataset time series terdeteksi: {len(data)} observasi dari {data[time_column].min()} hingga {data[time_column].max()}")
                        
                        # Display time series preview
                        st.write("Preview data time series:" if st.session_state.language == 'id' else "Time series data preview:")
                        time_series_preview = data[[time_column, target_column]].head(10)
                        st.dataframe(time_series_preview)
                        
                    except Exception as e:
                        st.error(f"Error processing time column: {e}")
                        st.session_state.is_time_series = False
            
            # Identify numerical and categorical columns
            numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            st.session_state.numerical_columns = numerical_cols
            st.session_state.categorical_columns = categorical_cols
            
            st.write(f"Kolom numerik: {', '.join(numerical_cols)}" if st.session_state.language == 'id' else f"Numerical columns: {', '.join(numerical_cols)}")
            st.write(f"Kolom kategorikal: {', '.join(categorical_cols)}" if st.session_state.language == 'id' else f"Categorical columns: {', '.join(categorical_cols)}")
            
            # Rekomendasi Metode Penelitian
            st.subheader("🎯 Rekomendasi Metode Penelitian" if st.session_state.language == 'id' else "🎯 Research Method Recommendations")
            recommendations = recommend_research_methods(data)
            
            for rec in recommendations:
                if rec['type'] == 'warning':
                    st.warning(f"**{rec['title']}**\n\n{rec['description']}")
                elif rec['type'] == 'info':
                    st.info(f"**{rec['title']}**\n\n{rec['description']}")
                elif rec['type'] == 'success':
                    st.success(f"**{rec['title']}**\n\n{rec['description']}")
                
                if 'methods' in rec:
                    st.write(f"**Metode yang disarankan:**" if st.session_state.language == 'id' else f"**Recommended methods:**")
                    for method in rec['methods']:
                        st.write(f"• {method}")
                    st.write("")
            
            # Rekomendasi Jenis Penelitian dengan Referensi Jurnal
            st.subheader("📚 Rekomendasi Jenis Penelitian & Referensi Jurnal" if st.session_state.language == 'id' else "📚 Research Types & Journal References")
            
            # Tambahkan tombol untuk generate rekomendasi jenis penelitian
            col1, col2 = st.columns([2, 1])
            with col1:
                if st.button("🤖 Generate Rekomendasi Jenis Penelitian" if st.session_state.language == 'id' else "🤖 Generate Research Type Recommendations", 
                           type="primary", key="generate_research_types"):
                    st.session_state.show_research_types = True
                    st.rerun()
            
            # Tampilkan rekomendasi jenis penelitian
            if st.session_state.get('show_research_types', False):
                with st.expander("📊 Rekomendasi Jenis Penelitian" if st.session_state.language == 'id' else "📊 Research Type Recommendations", expanded=True):
                    with st.spinner("Generating research type recommendations..." if st.session_state.language == 'id' else "Generating research type recommendations..."):
                        research_type_recommendations = get_research_type_recommendations(data)
                        
                        if research_type_recommendations:
                            for rec in research_type_recommendations:
                                st.markdown(f"### 📖 {rec['title']}")
                                st.write(f"**{rec['description']}**")
                                
                                # Research Methods
                                st.markdown("**🔬 Metodologi Penelitian:**" if st.session_state.language == 'id' else "**🔬 Research Methodology:**")
                                for method in rec['research_methods']:
                                    st.write(f"• {method}")
                                
                                # ML Approaches
                                st.markdown("**🤖 Pendekatan Machine Learning:**" if st.session_state.language == 'id' else "**🤖 Machine Learning Approaches:**")
                                for approach in rec['ml_approaches']:
                                    st.write(f"• {approach}")
                                
                                # Journal References
                                st.markdown("**📰 Referensi Jurnal Ilmiah:**" if st.session_state.language == 'id' else "**📰 Academic Journal References:**")
                                for journal in rec['journals']:
                                    st.markdown(f"**• {journal['title']}**")
                                    st.write(f"**Scope:** {journal['scope']}")
                                    st.write(f"**Impact Factor:** {journal['impact_factor']}")
                                    st.write(f"**Example Paper:** {journal['example_title']}")
                                    st.write(f"**DOI:** {journal['doi']}")
                                    st.write("")
                                
                                st.markdown("---")
                        else:
                            st.info("Tidak ada rekomendasi jenis penelitian yang tersedia untuk dataset ini." if st.session_state.language == 'id' else "No research type recommendations available for this dataset.")
            
            # Enhanced AI-Powered Analysis Section
            st.subheader("🤖 AI-Powered Dataset Analysis" if st.session_state.language == 'id' else "🤖 AI-Powered Dataset Analysis")
            
            # Add toggle for AI analysis
            if 'show_ai_analysis' not in st.session_state:
                st.session_state.show_ai_analysis = False
            
            # Tombol AI Analysis - Hanya satu tombol utama
            if st.button("🧠 Generate Analisis Dataset" if st.session_state.language == 'id' else "🧠 Generate Dataset Analysis", 
                       type="primary", key="generate_ai_analysis"):
                st.session_state.show_ai_analysis = True
                st.rerun()
            
            # Display AI Analysis - Fokus pada Potensi Keberhasilan
            if st.session_state.get('show_ai_analysis', False):
                with st.expander("📊 Potensi Keberhasilan Dataset untuk Penelitian" if st.session_state.language == 'id' else "📊 Dataset Success Potential for Research", expanded=True):
                    with st.spinner("Menganalisis potensi keberhasilan dataset..." if st.session_state.language == 'id' else "Analyzing dataset success potential..."):
                        ai_recommendations = analyze_dataset_with_ai(data)
                        
                        # Filter hanya rekomendasi yang relevan untuk keberhasilan penelitian
                        success_recommendations = []
                        for rec in ai_recommendations:
                            if rec['type'] in ['success', 'info'] and any(keyword in rec['title'].lower() or rec['description'].lower() 
                                                                          for keyword in ['siap', 'optimal', 'baik', 'success', 'good', 'optimal', 'ready']):
                                success_recommendations.append(rec)
                        
                        # Tampilkan ringkasan keberhasilan
                        if success_recommendations:
                            st.success("✅ **Dataset ini memiliki potensi keberhasilan yang baik untuk penelitian!**" if st.session_state.language == 'id' else "✅ **This dataset has good success potential for research!**")
                            
                            for rec in success_recommendations[:3]:  # Batasi hingga 3 rekomendasi utama
                                st.write(f"**{rec['title']}**")
                                st.write(f"{rec['description']}")
                                
                                if 'details' in rec and rec['details']:
                                    st.write("**Poin-poin utama:**" if st.session_state.language == 'id' else "**Key points:**")
                                    for detail in rec['details'][:2]:  # Batasi detail
                                        st.write(f"• {detail}")
                                st.write("")
                        else:
                            # Jika tidak ada rekomendasi sukses, tampilkan yang paling relevan
                            relevant_rec = [rec for rec in ai_recommendations if rec['type'] != 'error'][:2]
                            if relevant_rec:
                                for rec in relevant_rec:
                                    if rec['type'] == 'warning':
                                        st.warning(f"**⚠️ {rec['title']}**")
                                    else:
                                        st.info(f"**ℹ️ {rec['title']}**")
                                    st.write(f"{rec['description']}")
                                    st.write("")
                            
                            st.info("💡 **Saran:** Dataset ini memerlukan preprocessing tambahan untuk optimal dalam penelitian." if st.session_state.language == 'id' else "💡 **Suggestion:** This dataset requires additional preprocessing to be optimal for research.")
         

        
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
            
            # Rekomendasi Metode Penelitian untuk dataset gabungan
            st.subheader("🎯 Rekomendasi Metode Penelitian" if st.session_state.language == 'id' else "🎯 Research Method Recommendations")
            recommendations = recommend_research_methods(data)
            
            for rec in recommendations:
                if rec['type'] == 'warning':
                    st.warning(f"**{rec['title']}**\n\n{rec['description']}")
                elif rec['type'] == 'info':
                    st.info(f"**{rec['title']}**\n\n{rec['description']}")
                elif rec['type'] == 'success':
                    st.success(f"**{rec['title']}**\n\n{rec['description']}")
                
                if 'methods' in rec:
                    st.write(f"**Metode yang disarankan:**" if st.session_state.language == 'id' else f"**Recommended methods:**")
                    for method in rec['methods']:
                        st.write(f"• {method}")
                    st.write("")

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
        
        # Time Series Pattern Analysis (if time series data)
        if st.session_state.get('is_time_series', False) and st.session_state.time_column and st.session_state.target_column:
            st.subheader("🔍 Analisis Pola Time Series" if st.session_state.language == 'id' else "🔍 Time Series Pattern Analysis")
            
            try:
                # Prepare time series data
                ts_data = prepare_timeseries_data(
                    data, 
                    st.session_state.time_column, 
                    st.session_state.target_column
                )
                
                # Analyze patterns
                pattern_analysis = analyze_trend_seasonality_cycle(
                    ts_data[st.session_state.target_column]
                )
                
                # Display pattern insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Tren Terdeteksi" if st.session_state.language == 'id' else "Trend Detected",
                        "✅ Ya" if pattern_analysis['trend_detected'] else "❌ Tidak",
                        delta=f"Kekuatan: {pattern_analysis['trend_strength']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Seasonality Terdeteksi" if st.session_state.language == 'id' else "Seasonality Detected",
                        "✅ Ya" if pattern_analysis['seasonality_detected'] else "❌ Tidak",
                        delta=f"Kekuatan: {pattern_analysis['seasonality_strength']:.2f}"
                    )
                
                with col3:
                    st.metric(
                        "Siklus Terdeteksi" if st.session_state.language == 'id' else "Cycle Detected",
                        "✅ Ya" if pattern_analysis['cycle_detected'] else "❌ Tidak",
                        delta=f"Kekuatan: {pattern_analysis['cycle_strength']:.2f}"
                    )
                
                # Detailed pattern information
                if pattern_analysis['trend_detected']:
                    st.info(f"📈 **Tren:** Kemiringan tren adalah {pattern_analysis['trend_slope']:.4f} per periode")
                
                if pattern_analysis['seasonality_detected']:
                    st.info(f"🌊 **Seasonality:** Terdeteksi dengan kekuatan {pattern_analysis['seasonality_strength']:.2f}")
                
                if pattern_analysis['cycle_detected'] and pattern_analysis['dominant_cycle_period']:
                    st.info(f"🔄 **Siklus:** Periode dominan adalah {pattern_analysis['dominant_cycle_period']:.1f} periode")
                
                # Visualize patterns
                st.write("**Visualisasi Pola Time Series:**" if st.session_state.language == 'id' else "**Time Series Pattern Visualization:**")
                pattern_fig = plot_pattern_analysis(ts_data[st.session_state.target_column])
                st.pyplot(pattern_fig)
                
                # Seasonal decomposition insights
                if 'decomposition' in pattern_analysis and pattern_analysis['decomposition'] is not None:
                    st.write("**Insight dari Dekomposisi:**" if st.session_state.language == 'id' else "**Decomposition Insights:**")
                    
                    decomposition = pattern_analysis['decomposition']
                    
                    # Calculate variance explained by each component
                    total_var = np.var(ts_data[st.session_state.target_column])
                    trend_var = np.var(decomposition.trend.dropna()) if decomposition.trend is not None else 0
                    seasonal_var = np.var(decomposition.seasonal.dropna()) if decomposition.seasonal is not None else 0
                    residual_var = np.var(decomposition.resid.dropna()) if decomposition.resid is not None else 0
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Varians Tren", f"{trend_var/total_var:.1%}")
                    with col2:
                        st.metric("Varians Seasonal", f"{seasonal_var/total_var:.1%}")
                    with col3:
                        st.metric("Varians Residual", f"{residual_var/total_var:.1%}")
                
            except Exception as e:
                st.error(f"Error dalam analisis pola: {str(e)}")
        
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
                
                # Add automatic parameter optimization option
                auto_optimize = st.checkbox(
                    "Optimalkan parameter otomatis" if st.session_state.language == 'id' else "Auto-optimize parameters",
                    value=True
                )
                
                if clustering_method == "K-Means":
                    # K-Means Clustering
                    if auto_optimize:
                        with st.spinner("Mencari jumlah cluster optimal..." if st.session_state.language == 'id' else "Finding optimal number of clusters..."):
                            optimal_k, kmeans_metrics = find_optimal_clusters_kmeans(scaled_data)
                        st.success(f"Jumlah cluster optimal: {optimal_k}")
                        k_value = optimal_k
                    else:
                        max_k = min(10, len(clustering_data) - 1)
                        k_value = st.slider(
                            "Jumlah cluster (k):" if st.session_state.language == 'id' else "Number of clusters (k):",
                            2, max_k, 3
                        )
                    
                    kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(scaled_data)
                    
                    # Calculate comprehensive metrics
                    clustering_metrics = calculate_comprehensive_clustering_metrics(scaled_data, clusters, "K-Means")
                    
                    # Display comprehensive evaluation
                    st.write("### Hasil Evaluasi Clustering" if st.session_state.language == 'id' else "### Clustering Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'silhouette_score' in clustering_metrics:
                            st.metric("Silhouette Score", f"{clustering_metrics['silhouette_score']:.3f}")
                    with col2:
                        if 'calinski_harabasz_score' in clustering_metrics:
                            st.metric("Calinski-Harabasz", f"{clustering_metrics['calinski_harabasz_score']:.1f}")
                    with col3:
                        if 'davies_bouldin_score' in clustering_metrics:
                            st.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin_score']:.3f}")
                    
                    # Add cluster stability analysis
                    if st.checkbox("Analisis Stabilitas Cluster" if st.session_state.language == 'id' else "Cluster Stability Analysis"):
                        stability = analyze_cluster_stability(scaled_data, clusters)
                        if 'stability_score' in stability:
                            st.write(f"Stabilitas Cluster: {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
                    
                    # Analyze cluster characteristics
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis"):
                        cluster_profiles = analyze_cluster_characteristics(scaled_data, clusters, selected_features)
                        
                        # Display cluster profiles
                        st.write("**Profil Cluster:**" if st.session_state.language == 'id' else "**Cluster Profiles:**")
                        for cluster_name, profile in cluster_profiles.items():
                            with st.expander(f"{cluster_name} ({profile['size']} samples, {profile['percentage']:.1f}%)"):
                                if 'feature_stats' in profile:
                                    for feature_name, stats in profile['feature_stats'].items():
                                        st.write(f"- {feature_name}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
                    
                    # Generate comprehensive report
                    if st.checkbox("Hasilkan Laporan Komprehensif" if st.session_state.language == 'id' else "Generate Comprehensive Report"):
                        cluster_profiles = analyze_cluster_characteristics(scaled_data, clusters, selected_features)
                        report = generate_cluster_report(
                            scaled_data, clusters, "K-Means", clustering_metrics, stability, cluster_profiles
                        )
                        st.text(report)
                        
                        # Add download button for report
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report,
                            file_name=f"kmeans_clustering_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    # Add cluster labels to data
                    clustering_data['Cluster'] = clusters

                    # Add cluster characteristics analysis
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis", key="kmeans_characteristics"):
                        characteristics = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        st.write("#### Karakteristik Cluster" if st.session_state.language == 'id' else "#### Cluster Characteristics")
                        
                        for cluster_id, char in characteristics.items():
                            with st.expander(f"Cluster {cluster_id} (n={char['size']}, {char['percentage']:.1f}%)"):
                                # Handle missing 'dominant_features' key
                                if 'dominant_features' in char:
                                    st.write(f"**Fitur Utama:** {char['dominant_features']}")
                                elif 'feature_stats' in char:
                                    # Show top 3 features with highest mean values as dominant features
                                    feature_means = [(name, stats['mean']) for name, stats in char['feature_stats'].items()]
                                    feature_means.sort(key=lambda x: abs(x[1]), reverse=True)
                                    dominant_features = [name for name, _ in feature_means[:3]]
                                    st.write(f"**Fitur Utama:** {', '.join(dominant_features)}")
                                
                                if 'feature_stats' in char:
                                    st.write("**Statistik Fitur:**")
                                    st.json(char['feature_stats'])
                    
                    # Generate comprehensive cluster report
                    if st.checkbox("Hasilkan Laporan Cluster" if st.session_state.language == 'id' else "Generate Cluster Report", key="kmeans_report"):
                        cluster_profiles = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        report = generate_cluster_report(clustering_data, clusters, "K-Means", clustering_metrics, None, cluster_profiles)
                        st.text_area(
                            "Laporan Analisis Cluster" if st.session_state.language == 'id' else "Cluster Analysis Report",
                            value=report,
                            height=300
                        )
                        
                        # Download report button
                        report_bytes = report.encode()
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report_bytes,
                            file_name=f"cluster_report_dbscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                    
                    # Visualize clusters with enhanced plotting
                    if len(selected_features) >= 2:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # Original feature scatter plot
                        scatter = axes[0,0].scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        axes[0,0].set_xlabel(selected_features[0])
                        axes[0,0].set_ylabel(selected_features[1])
                        axes[0,0].set_title('Visualisasi Cluster' if st.session_state.language == 'id' else 'Cluster Visualization')
                        plt.colorbar(scatter, ax=axes[0,0])
                        
                        # PCA visualization
                        if len(selected_features) > 2:
                            pca = PCA(n_components=2)
                            pca_data = pca.fit_transform(scaled_data)
                            scatter2 = axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                            axes[0,1].set_xlabel('PC1')
                            axes[0,1].set_ylabel('PC2')
                            axes[0,1].set_title('PCA - Visualisasi Cluster' if st.session_state.language == 'id' else 'PCA - Cluster Visualization')
                            plt.colorbar(scatter2, ax=axes[0,1])
                        
                        # Cluster size distribution
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        axes[1,0].bar(cluster_counts.index, cluster_counts.values)
                        axes[1,0].set_xlabel('Cluster')
                        axes[1,0].set_ylabel('Jumlah Data' if st.session_state.language == 'id' else 'Data Count')
                        axes[1,0].set_title('Distribusi Cluster' if st.session_state.language == 'id' else 'Cluster Distribution')
                        
                        # Elbow method visualization (if auto-optimized)
                        if auto_optimize and 'kmeans_metrics' in locals() and kmeans_metrics.get('inertia'):
                            axes[1,1].plot(kmeans_metrics['k'], kmeans_metrics['inertia'], 'bo-')
                            axes[1,1].axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
                            axes[1,1].set_xlabel('Jumlah Cluster (k)' if st.session_state.language == 'id' else 'Number of Clusters (k)')
                            axes[1,1].set_ylabel('Inertia')
                            axes[1,1].set_title('Elbow Method')
                            axes[1,1].legend()
                            axes[1,1].grid(True)
                        else:
                            # Show cluster statistics summary
                            axes[1,1].text(0.1, 0.9, f"Jumlah Cluster: {clustering_metrics.get('n_clusters', 0)}", transform=axes[1,1].transAxes)
                            axes[1,1].text(0.1, 0.8, f"Rata-rata Ukuran: {clustering_metrics.get('cluster_size_mean', 0):.1f}", transform=axes[1,1].transAxes)
                            axes[1,1].text(0.1, 0.7, f"Std Ukuran: {clustering_metrics.get('cluster_size_std', 0):.1f}", transform=axes[1,1].transAxes)
                            axes[1,1].set_title('Ringkasan Cluster' if st.session_state.language == 'id' else 'Cluster Summary')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                
                elif clustering_method == "K-Prototypes":
                    # K-Prototypes Clustering for mixed data types
                    if auto_optimize:
                        with st.spinner("Mencari jumlah cluster optimal..." if st.session_state.language == 'id' else "Finding optimal number of clusters..."):
                            optimal_k = 3  # Default fallback
                            best_score = -1
                            best_gamma = 0.5
                            
                            # Grid search for optimal k and gamma
                            for k in range(2, min(8, len(clustering_data) - 1)):
                                for gamma in [0.1, 0.3, 0.5, 0.7, 0.9]:
                                    try:
                                        # Prepare data for K-Prototypes
                                        categorical_idx = [i for i, col in enumerate(selected_features) 
                                                         if col in st.session_state.categorical_columns]
                                        
                                        # Convert categorical columns to appropriate types
                                        kproto_data = clustering_data.copy()
                                        for col in categorical_in_selected:
                                            kproto_data[col] = kproto_data[col].astype('category').cat.codes
                                        
                                        # Initialize and fit K-Prototypes
                                        kproto = KPrototypes(n_clusters=k, init='Huang', random_state=42, gamma=gamma)
                                        clusters = kproto.fit_predict(kproto_data.values, categorical_idx)
                                        
                                        # Calculate silhouette score (only for numerical features)
                                        if len(set(clusters)) > 1 and len([col for col in selected_features 
                                                                              if col in st.session_state.numerical_columns]) > 0:
                                            # Use only numerical features for silhouette score
                                            num_features = [col for col in selected_features 
                                                          if col in st.session_state.numerical_columns]
                                            if num_features:
                                                num_data = clustering_data[num_features]
                                                num_scaled = StandardScaler().fit_transform(num_data)
                                                score = silhouette_score(num_scaled, clusters)
                                                if score > best_score:
                                                    best_score = score
                                                    optimal_k = k
                                                    best_gamma = gamma
                                    except:
                                        continue
                        
                        st.success(f"Jumlah cluster optimal: {optimal_k}, Gamma optimal: {best_gamma}")
                        k_value = optimal_k
                        gamma_value = best_gamma
                    else:
                        max_k = min(10, len(clustering_data) - 1)
                        k_value = st.slider(
                            "Jumlah cluster (k):" if st.session_state.language == 'id' else "Number of clusters (k):",
                            2, max_k, 3
                        )
                        
                        gamma_value = st.slider(
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
                    kproto = KPrototypes(n_clusters=k_value, init='Huang', random_state=42, gamma=gamma_value)
                    clusters = kproto.fit_predict(kproto_data.values, categorical_idx)
                    
                    # Calculate comprehensive metrics (using numerical features only)
                    if len([col for col in selected_features if col in st.session_state.numerical_columns]) > 0:
                        num_features = [col for col in selected_features if col in st.session_state.numerical_columns]
                        num_data = clustering_data[num_features]
                        num_scaled = StandardScaler().fit_transform(num_data)
                        clustering_metrics = calculate_comprehensive_clustering_metrics(num_scaled, clusters, "K-Prototypes")
                    else:
                        # Fallback to basic metrics if no numerical features
                        clustering_metrics = {
                            'n_clusters': len(set(clusters)) - (1 if -1 in clusters else 0),
                            'cluster_size_mean': np.mean(pd.Series(clusters).value_counts()),
                            'cluster_size_std': np.std(pd.Series(clusters).value_counts())
                        }
                    
                    # Display comprehensive evaluation
                    st.write("### Hasil Evaluasi Clustering" if st.session_state.language == 'id' else "### Clustering Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'silhouette_score' in clustering_metrics:
                            st.metric("Silhouette Score", f"{clustering_metrics['silhouette_score']:.3f}")
                    with col2:
                        if 'calinski_harabasz_score' in clustering_metrics:
                            st.metric("Calinski-Harabasz", f"{clustering_metrics['calinski_harabasz_score']:.1f}")
                    with col3:
                        if 'davies_bouldin_score' in clustering_metrics:
                            st.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin_score']:.3f}")
                    
                    # Add cluster stability analysis
                    if st.checkbox("Analisis Stabilitas Cluster" if st.session_state.language == 'id' else "Cluster Stability Analysis"):
                        if len([col for col in selected_features if col in st.session_state.numerical_columns]) > 0:
                            stability = analyze_cluster_stability(num_scaled, clusters)
                            if 'stability_score' in stability:
                                st.write(f"Stabilitas Cluster: {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
                        else:
                            st.write("Analisis stabilitas memerlukan fitur numerikal.")
                    
                    # Add cluster labels to data
                    clustering_data['Cluster'] = clusters
                    
                    # Enhanced visualization
                    if len(selected_features) >= 2:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # Scatter plot using first two features
                        scatter = axes[0,0].scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        axes[0,0].set_xlabel(selected_features[0])
                        axes[0,0].set_ylabel(selected_features[1])
                        axes[0,0].set_title('K-Prototypes Clustering' if st.session_state.language == 'id' else 'K-Prototypes Clustering')
                        plt.colorbar(scatter, ax=axes[0,0])
                        
                        # PCA visualization for numerical features
                        num_features = [col for col in selected_features 
                                      if col in st.session_state.numerical_columns]
                        if len(num_features) >= 2:
                            pca = PCA(n_components=2)
                            pca_data = pca.fit_transform(clustering_data[num_features])
                            scatter2 = axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                            axes[0,1].set_xlabel('PC1')
                            axes[0,1].set_ylabel('PC2')
                            axes[0,1].set_title('PCA - K-Prototypes Clustering' if st.session_state.language == 'id' else 'PCA - K-Prototypes Clustering')
                            plt.colorbar(scatter2, ax=axes[0,1])
                        else:
                            # Show feature importance if not enough numerical features
                            axes[0,1].text(0.1, 0.5, "Tidak cukup fitur numerikal untuk PCA", transform=axes[0,1].transAxes)
                            axes[0,1].set_title('PCA - Tidak Tersedia' if st.session_state.language == 'id' else 'PCA - Not Available')
                        
                        # Cluster size distribution
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        axes[1,0].bar(cluster_counts.index, cluster_counts.values)
                        axes[1,0].set_xlabel('Cluster')
                        axes[1,0].set_ylabel('Jumlah Data' if st.session_state.language == 'id' else 'Data Count')
                        axes[1,0].set_title('Distribusi Cluster' if st.session_state.language == 'id' else 'Cluster Distribution')
                        
                        # Cluster summary
                        axes[1,1].text(0.1, 0.9, f"Jumlah Cluster: {clustering_metrics.get('n_clusters', 0)}", transform=axes[1,1].transAxes)
                        axes[1,1].text(0.1, 0.8, f"Rata-rata Ukuran: {clustering_metrics.get('cluster_size_mean', 0):.1f}", transform=axes[1,1].transAxes)
                        axes[1,1].text(0.1, 0.7, f"Std Ukuran: {clustering_metrics.get('cluster_size_std', 0):.1f}", transform=axes[1,1].transAxes)
                        if 'silhouette_score' in clustering_metrics:
                            axes[1,1].text(0.1, 0.6, f"Silhouette: {clustering_metrics['silhouette_score']:.3f}", transform=axes[1,1].transAxes)
                        axes[1,1].set_title('Ringkasan Cluster' if st.session_state.language == 'id' else 'Cluster Summary')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Show cluster statistics
                    st.write("Statistik per cluster:" if st.session_state.language == 'id' else "Cluster statistics:")
                    cluster_stats = clustering_data.groupby('Cluster').agg({
                        col: ['count', 'mean', 'std'] if col in st.session_state.numerical_columns else ['count', 'nunique']
                        for col in selected_features
                    }).round(3)
                    st.dataframe(cluster_stats)

                    # Add cluster characteristics analysis
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis"):
                        if len([col for col in selected_features if col in st.session_state.numerical_columns]) > 0:
                            characteristics = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                            st.write("#### Karakteristik Cluster" if st.session_state.language == 'id' else "#### Cluster Characteristics")
                            
                            for cluster_id, char in characteristics.items():
                                with st.expander(f"Cluster {cluster_id} (n={char['size']}, {char['percentage']:.1f}%)"):
                                    # Handle missing 'dominant_features' key
                                    if 'dominant_features' in char:
                                        st.write(f"**Fitur Utama:** {char['dominant_features']}")
                                    elif 'feature_stats' in char:
                                        # Show top 3 features with highest mean values as dominant features
                                        feature_means = [(name, stats['mean']) for name, stats in char['feature_stats'].items()]
                                        feature_means.sort(key=lambda x: abs(x[1]), reverse=True)
                                        dominant_features = [name for name, _ in feature_means[:3]]
                                        st.write(f"**Fitur Utama:** {', '.join(dominant_features)}")
                                    
                                    if 'feature_stats' in char:
                                        st.write("**Statistik Fitur:**")
                                        st.json(char['feature_stats'])
                        else:
                            st.write("Analisis karakteristik memerlukan fitur numerikal.")
                    
                    # Generate comprehensive cluster report
                    if st.checkbox("Hasilkan Laporan Cluster" if st.session_state.language == 'id' else "Generate Cluster Report"):
                        cluster_profiles = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        report = generate_cluster_report(clustering_data, clusters, "K-Prototypes", clustering_metrics, None, cluster_profiles)
                        st.text_area(
                            "Laporan Analisis Cluster" if st.session_state.language == 'id' else "Cluster Analysis Report",
                            value=report,
                            height=300
                        )
                        
                        # Download report button
                        report_bytes = report.encode()
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report_bytes,
                            file_name=f"cluster_report_kprototypes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )

                elif clustering_method == "Hierarchical":
                    # Hierarchical Clustering
                    linkage_method = st.selectbox(
                        "Metode linkage:" if st.session_state.language == 'id' else "Linkage method:",
                        ["ward", "complete", "average", "single"]
                    )
                    
                    if auto_optimize:
                        with st.spinner("Mencari jumlah cluster optimal..." if st.session_state.language == 'id' else "Finding optimal number of clusters..."):
                            optimal_n_clusters = 3  # Default fallback
                            best_score = -1
                            
                            # Try different numbers of clusters
                            for n_clusters in range(2, min(8, len(clustering_data) - 1)):
                                try:
                                    hierarchical = AgglomerativeClustering(
                                        n_clusters=n_clusters, 
                                        linkage=linkage_method
                                    )
                                    clusters = hierarchical.fit_predict(scaled_data)
                                    
                                    if len(set(clusters)) > 1:
                                        score = silhouette_score(scaled_data, clusters)
                                        if score > best_score:
                                            best_score = score
                                            optimal_n_clusters = n_clusters
                                except:
                                    continue
                            
                            st.success(f"Jumlah cluster optimal: {optimal_n_clusters}")
                            n_clusters = optimal_n_clusters
                    else:
                        n_clusters = st.slider(
                            "Jumlah cluster:" if st.session_state.language == 'id' else "Number of clusters:",
                            2, min(10, len(clustering_data) - 1), 3
                        )
                    
                    hierarchical = AgglomerativeClustering(
                        n_clusters=n_clusters, 
                        linkage=linkage_method
                    )
                    clusters = hierarchical.fit_predict(scaled_data)
                    
                    # Calculate comprehensive metrics
                    clustering_metrics = calculate_comprehensive_clustering_metrics(scaled_data, clusters, "Hierarchical")
                    
                    # Display comprehensive evaluation
                    st.write("### Hasil Evaluasi Clustering" if st.session_state.language == 'id' else "### Clustering Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'silhouette_score' in clustering_metrics:
                            st.metric("Silhouette Score", f"{clustering_metrics['silhouette_score']:.3f}")
                    with col2:
                        if 'calinski_harabasz_score' in clustering_metrics:
                            st.metric("Calinski-Harabasz", f"{clustering_metrics['calinski_harabasz_score']:.1f}")
                    with col3:
                        if 'davies_bouldin_score' in clustering_metrics:
                            st.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin_score']:.3f}")
                    
                    # Add cluster stability analysis
                    if st.checkbox("Analisis Stabilitas Cluster" if st.session_state.language == 'id' else "Cluster Stability Analysis"):
                        stability = analyze_cluster_stability(scaled_data, clusters)
                        if 'stability_score' in stability:
                            st.write(f"Stabilitas Cluster: {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
                    
                    # Enhanced visualization
                    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                    
                    # Dendrogram plot
                    linkage_matrix = linkage(scaled_data[:min(100, len(scaled_data))], method=linkage_method)
                    dendrogram(linkage_matrix, ax=axes[0,0])
                    axes[0,0].set_title('Dendrogram' if st.session_state.language == 'id' else 'Dendrogram')
                    axes[0,0].set_xlabel('Sample Index')
                    axes[0,0].set_ylabel('Distance')
                    
                    # Cluster visualization
                    if len(selected_features) >= 2:
                        scatter = axes[0,1].scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        axes[0,1].set_xlabel(selected_features[0])
                        axes[0,1].set_ylabel(selected_features[1])
                        axes[0,1].set_title('Hierarchical Clustering' if st.session_state.language == 'id' else 'Hierarchical Clustering')
                        plt.colorbar(scatter, ax=axes[0,1])
                    
                    # Cluster size distribution
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    axes[1,0].bar(cluster_counts.index, cluster_counts.values)
                    axes[1,0].set_xlabel('Cluster')
                    axes[1,0].set_ylabel('Jumlah Data' if st.session_state.language == 'id' else 'Data Count')
                    axes[1,0].set_title('Distribusi Cluster' if st.session_state.language == 'id' else 'Cluster Distribution')
                    
                    # Cluster summary
                    axes[1,1].text(0.1, 0.9, f"Jumlah Cluster: {clustering_metrics.get('n_clusters', 0)}", transform=axes[1,1].transAxes)
                    axes[1,1].text(0.1, 0.8, f"Rata-rata Ukuran: {clustering_metrics.get('cluster_size_mean', 0):.1f}", transform=axes[1,1].transAxes)
                    axes[1,1].text(0.1, 0.7, f"Std Ukuran: {clustering_metrics.get('cluster_size_std', 0):.1f}", transform=axes[1,1].transAxes)
                    if 'silhouette_score' in clustering_metrics:
                        axes[1,1].text(0.1, 0.6, f"Silhouette: {clustering_metrics['silhouette_score']:.3f}", transform=axes[1,1].transAxes)
                    axes[1,1].set_title('Ringkasan Cluster' if st.session_state.language == 'id' else 'Cluster Summary')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Add cluster labels
                    clustering_data['Cluster'] = clusters
                    st.write("Distribusi cluster:" if st.session_state.language == 'id' else "Cluster distribution:")
                    st.write(clustering_data['Cluster'].value_counts())

                    # Add cluster characteristics analysis
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis", key="kprototypes_characteristics"):
                        characteristics = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        st.write("#### Karakteristik Cluster" if st.session_state.language == 'id' else "#### Cluster Characteristics")
                        
                        for cluster_id, char in characteristics.items():
                            with st.expander(f"Cluster {cluster_id} (n={char['size']}, {char['percentage']:.1f}%)"):
                                if 'dominant_features' in char:
                                    st.write(f"**Fitur Utama:** {char['dominant_features']}")
                                else:
                                    # Extract top 3 features with highest absolute mean values
                                    feature_stats = char.get('feature_stats', {})
                                    if feature_stats:
                                        top_features = sorted(feature_stats.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                        dominant_features = ", ".join([f"{feat}: {val:.3f}" for feat, val in top_features])
                                        st.write(f"**Fitur Utama:** {dominant_features}")
                                    else:
                                        st.write("**Fitur Utama:** Tidak tersedia")
                                st.write("**Statistik Fitur:**")
                                st.json(char['feature_stats'])
                    
                    # Generate comprehensive cluster report
                    if st.checkbox("Hasilkan Laporan Cluster" if st.session_state.language == 'id' else "Generate Cluster Report", key="kprototypes_report"):
                        cluster_profiles = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        report = generate_cluster_report(clustering_data, clusters, "Hierarchical", clustering_metrics, None, cluster_profiles)
                        st.text_area(
                            "Laporan Analisis Cluster" if st.session_state.language == 'id' else "Cluster Analysis Report",
                            value=report,
                            height=300
                        )
                        
                        # Download report button
                        report_bytes = report.encode()
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report_bytes,
                            file_name=f"cluster_report_hierarchical_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                elif clustering_method == "DBSCAN":
                    # DBSCAN Clustering
                    if auto_optimize:
                        with st.spinner("Mencari parameter DBSCAN optimal..." if st.session_state.language == 'id' else "Finding optimal DBSCAN parameters..."):
                            optimal_eps, optimal_min_samples, dbscan_metrics = find_optimal_eps_dbscan(scaled_data)
                        st.success(f"Parameter optimal: eps={optimal_eps:.2f}, min_samples={optimal_min_samples}")
                        eps = optimal_eps
                        min_samples = optimal_min_samples
                    else:
                        eps = st.slider("Eps (radius neighborhood):", 0.1, 5.0, 0.5, 0.1)
                        min_samples = st.slider("Min samples:", 1, 20, 5)
                    
                    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                    clusters = dbscan.fit_predict(scaled_data)
                    
                    # Calculate comprehensive metrics
                    clustering_metrics = calculate_comprehensive_clustering_metrics(scaled_data, clusters, "DBSCAN")
                    
                    # Display comprehensive evaluation
                    st.write("### Hasil Evaluasi Clustering" if st.session_state.language == 'id' else "### Clustering Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'silhouette_score' in clustering_metrics:
                            st.metric("Silhouette Score", f"{clustering_metrics['silhouette_score']:.3f}")
                    with col2:
                        if 'calinski_harabasz_score' in clustering_metrics:
                            st.metric("Calinski-Harabasz", f"{clustering_metrics['calinski_harabasz_score']:.1f}")
                    with col3:
                        if 'davies_bouldin_score' in clustering_metrics:
                            st.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin_score']:.3f}")
                    
                    # Count clusters (excluding noise)
                    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                    st.write(f"Jumlah cluster: {n_clusters}")
                    st.write(f"Noise points: {(clusters == -1).sum()}")
                    
                    # Add cluster stability analysis
                    if st.checkbox("Analisis Stabilitas Cluster" if st.session_state.language == 'id' else "Cluster Stability Analysis"):
                        stability = analyze_cluster_stability(scaled_data, clusters)
                        if 'stability_score' in stability:
                            st.write(f"Stabilitas Cluster: {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
                    
                    # Enhanced visualization
                    if len(selected_features) >= 2:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # DBSCAN clusters scatter plot
                        scatter = axes[0,0].scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        axes[0,0].set_xlabel(selected_features[0])
                        axes[0,0].set_ylabel(selected_features[1])
                        axes[0,0].set_title('DBSCAN Clustering' if st.session_state.language == 'id' else 'DBSCAN Clustering')
                        plt.colorbar(scatter, ax=axes[0,0])
                        
                        # PCA visualization
                        pca = PCA(n_components=2)
                        pca_data = pca.fit_transform(scaled_data)
                        scatter2 = axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                        axes[0,1].set_xlabel('PC1')
                        axes[0,1].set_ylabel('PC2')
                        axes[0,1].set_title('PCA - DBSCAN Clustering' if st.session_state.language == 'id' else 'PCA - DBSCAN Clustering')
                        plt.colorbar(scatter2, ax=axes[0,1])
                        
                        # Cluster size distribution (excluding noise)
                        cluster_counts = pd.Series(clusters[clusters != -1]).value_counts().sort_index()
                        axes[1,0].bar(cluster_counts.index, cluster_counts.values)
                        axes[1,0].set_xlabel('Cluster')
                        axes[1,0].set_ylabel('Jumlah Data' if st.session_state.language == 'id' else 'Data Count')
                        axes[1,0].set_title('Distribusi Cluster (Tanpa Noise)' if st.session_state.language == 'id' else 'Cluster Distribution (No Noise)')
                        
                        # K-distance plot (if auto-optimized)
                        if auto_optimize and 'kmeans_metrics' in locals() and dbscan_metrics.get('k_distances'):
                            axes[1,1].plot(dbscan_metrics['k_distances'], 'bo-')
                            axes[1,1].axhline(y=optimal_eps, color='red', linestyle='--', label=f'Optimal eps={optimal_eps:.2f}')
                            axes[1,1].set_xlabel('Data Point Index')
                            axes[1,1].set_ylabel('K-distance')
                            axes[1,1].set_title('K-distance Graph')
                            axes[1,1].legend()
                            axes[1,1].grid(True)
                        else:
                            # Show cluster summary
                            axes[1,1].text(0.1, 0.9, f"Jumlah Cluster: {n_clusters}", transform=axes[1,1].transAxes)
                            axes[1,1].text(0.1, 0.8, f"Noise Points: {(clusters == -1).sum()}", transform=axes[1,1].transAxes)
                            if 'silhouette_score' in clustering_metrics:
                                axes[1,1].text(0.1, 0.7, f"Silhouette: {clustering_metrics['silhouette_score']:.3f}", transform=axes[1,1].transAxes)
                            axes[1,1].set_title('Ringkasan Cluster' if st.session_state.language == 'id' else 'Cluster Summary')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    clustering_data['Cluster'] = clusters

                    # Add cluster characteristics analysis
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis", key="hierarchical_characteristics"):
                        characteristics = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        st.write("#### Karakteristik Cluster" if st.session_state.language == 'id' else "#### Cluster Characteristics")
                        
                        for cluster_id, char in characteristics.items():
                            with st.expander(f"Cluster {cluster_id} (n={char['size']}, {char['percentage']:.1f}%)"):
                                if 'dominant_features' in char:
                                    st.write(f"**Fitur Utama:** {char['dominant_features']}")
                                else:
                                    # Extract top 3 features with highest absolute mean values
                                    feature_stats = char.get('feature_stats', {})
                                    if feature_stats:
                                        top_features = sorted(feature_stats.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                        dominant_features = ", ".join([f"{feat}: {val:.3f}" for feat, val in top_features])
                                        st.write(f"**Fitur Utama:** {dominant_features}")
                                    else:
                                        st.write("**Fitur Utama:** Tidak tersedia")
                                st.write("**Statistik Fitur:**")
                                st.json(char['feature_stats'])
                    
                    # Generate comprehensive cluster report
                    if st.checkbox("Hasilkan Laporan Cluster" if st.session_state.language == 'id' else "Generate Cluster Report", key="hierarchical_report"):
                        cluster_profiles = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        report = generate_cluster_report(clustering_data, clusters, "DBSCAN", clustering_metrics, None, cluster_profiles)
                        st.text_area(
                            "Laporan Analisis Cluster" if st.session_state.language == 'id' else "Cluster Analysis Report",
                            value=report,
                            height=300
                        )
                        
                        # Download report button
                        report_bytes = report.encode()
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report_bytes,
                            file_name=f"cluster_report_dbscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
                elif clustering_method == "Spectral":
                    # Spectral Clustering
                    if auto_optimize:
                        with st.spinner("Mencari parameter Spectral optimal..." if st.session_state.language == 'id' else "Finding optimal Spectral parameters..."):
                            # Find optimal number of clusters
                            best_k = None
                            best_score = -1
                            spectral_metrics = {}
                            
                            for k in range(2, 11):
                                try:
                                    spectral_temp = SpectralClustering(n_clusters=k, random_state=42, n_init=10, affinity='nearest_neighbors')
                                    clusters_temp = spectral_temp.fit_predict(scaled_data)
                                    
                                    # Calculate silhouette score
                                    if len(set(clusters_temp)) > 1:
                                        score = silhouette_score(scaled_data, clusters_temp)
                                        if score > best_score:
                                            best_score = score
                                            best_k = k
                                            spectral_metrics = calculate_comprehensive_clustering_metrics(scaled_data, clusters_temp, "Spectral")
                                except:
                                    continue
                            
                            n_clusters = best_k if best_k else 3
                        st.success(f"Jumlah cluster optimal: {n_clusters}")
                    else:
                        n_clusters = st.slider(
                            "Jumlah cluster:" if st.session_state.language == 'id' else "Number of clusters:",
                            2, min(10, len(clustering_data) - 1), 3
                        )
                    
                    spectral = SpectralClustering(
                        n_clusters=n_clusters, 
                        random_state=42,
                        n_init=10,
                        affinity='nearest_neighbors'
                    )
                    clusters = spectral.fit_predict(scaled_data)
                    
                    # Calculate comprehensive metrics
                    clustering_metrics = calculate_comprehensive_clustering_metrics(scaled_data, clusters, "Spectral")
                    
                    # Display comprehensive evaluation
                    st.write("### Hasil Evaluasi Clustering" if st.session_state.language == 'id' else "### Clustering Evaluation Results")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'silhouette_score' in clustering_metrics:
                            st.metric("Silhouette Score", f"{clustering_metrics['silhouette_score']:.3f}")
                    with col2:
                        if 'calinski_harabasz_score' in clustering_metrics:
                            st.metric("Calinski-Harabasz", f"{clustering_metrics['calinski_harabasz_score']:.1f}")
                    with col3:
                        if 'davies_bouldin_score' in clustering_metrics:
                            st.metric("Davies-Bouldin", f"{clustering_metrics['davies_bouldin_score']:.3f}")
                    
                    # Add cluster stability analysis
                    if st.checkbox("Analisis Stabilitas Cluster" if st.session_state.language == 'id' else "Cluster Stability Analysis"):
                        stability = analyze_cluster_stability(scaled_data, clusters)
                        if 'stability_score' in stability:
                            st.write(f"Stabilitas Cluster: {stability['stability_score']:.3f} ± {stability.get('stability_std', 0):.3f}")
                    
                    # Enhanced visualization
                    if len(selected_features) >= 2:
                        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                        
                        # Spectral clusters scatter plot
                        scatter = axes[0,0].scatter(
                            clustering_data.iloc[:, 0], 
                            clustering_data.iloc[:, 1], 
                            c=clusters, 
                            cmap='viridis', 
                            alpha=0.6
                        )
                        axes[0,0].set_xlabel(selected_features[0])
                        axes[0,0].set_ylabel(selected_features[1])
                        axes[0,0].set_title('Spectral Clustering' if st.session_state.language == 'id' else 'Spectral Clustering')
                        plt.colorbar(scatter, ax=axes[0,0])
                        
                        # PCA visualization
                        pca = PCA(n_components=2)
                        pca_data = pca.fit_transform(scaled_data)
                        scatter2 = axes[0,1].scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', alpha=0.6)
                        axes[0,1].set_xlabel('PC1')
                        axes[0,1].set_ylabel('PC2')
                        axes[0,1].set_title('PCA - Spectral Clustering' if st.session_state.language == 'id' else 'PCA - Spectral Clustering')
                        plt.colorbar(scatter2, ax=axes[0,1])
                        
                        # Cluster size distribution
                        cluster_counts = pd.Series(clusters).value_counts().sort_index()
                        axes[1,0].bar(cluster_counts.index, cluster_counts.values)
                        axes[1,0].set_xlabel('Cluster')
                        axes[1,0].set_ylabel('Jumlah Data' if st.session_state.language == 'id' else 'Data Count')
                        axes[1,0].set_title('Distribusi Cluster' if st.session_state.language == 'id' else 'Cluster Distribution')
                        
                        # Show cluster summary and statistics
                        axes[1,1].text(0.1, 0.9, f"Jumlah Cluster: {n_clusters}", transform=axes[1,1].transAxes)
                        if 'silhouette_score' in clustering_metrics:
                            axes[1,1].text(0.1, 0.8, f"Silhouette: {clustering_metrics['silhouette_score']:.3f}", transform=axes[1,1].transAxes)
                        if 'calinski_harabasz_score' in clustering_metrics:
                            axes[1,1].text(0.1, 0.7, f"Calinski-Harabasz: {clustering_metrics['calinski_harabasz_score']:.1f}", transform=axes[1,1].transAxes)
                        if 'davies_bouldin_score' in clustering_metrics:
                            axes[1,1].text(0.1, 0.6, f"Davies-Bouldin: {clustering_metrics['davies_bouldin_score']:.3f}", transform=axes[1,1].transAxes)
                        axes[1,1].set_title('Ringkasan Cluster' if st.session_state.language == 'id' else 'Cluster Summary')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    clustering_data['Cluster'] = clusters

                    # Add cluster characteristics analysis
                    if st.checkbox("Analisis Karakteristik Cluster" if st.session_state.language == 'id' else "Cluster Characteristics Analysis", key="spectral_characteristics"):
                        characteristics = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        st.write("#### Karakteristik Cluster" if st.session_state.language == 'id' else "#### Cluster Characteristics")
                        
                        for cluster_id, char in characteristics.items():
                            with st.expander(f"Cluster {cluster_id} (n={char['size']}, {char['percentage']:.1f}%)"):
                                if 'dominant_features' in char:
                                    st.write(f"**Fitur Utama:** {char['dominant_features']}")
                                else:
                                    # Extract top 3 features with highest absolute mean values
                                    feature_stats = char.get('feature_stats', {})
                                    if feature_stats:
                                        top_features = sorted(feature_stats.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                                        dominant_features = ", ".join([f"{feat}: {val:.3f}" for feat, val in top_features])
                                        st.write(f"**Fitur Utama:** {dominant_features}")
                                    else:
                                        st.write("**Fitur Utama:** Tidak tersedia")
                                st.write("**Statistik Fitur:**")
                                st.json(char['feature_stats'])
                    
                    # Generate comprehensive cluster report
                    if st.checkbox("Hasilkan Laporan Cluster" if st.session_state.language == 'id' else "Generate Cluster Report", key="spectral_report"):
                        cluster_profiles = analyze_cluster_characteristics(clustering_data, clusters, selected_features)
                        report = generate_cluster_report(clustering_data, clusters, "Spectral", clustering_metrics, None, cluster_profiles)
                        st.text_area(
                            "Laporan Analisis Cluster" if st.session_state.language == 'id' else "Cluster Analysis Report",
                            value=report,
                            height=300
                        )
                        
                        # Download report button
                        report_bytes = report.encode()
                        st.download_button(
                            label="Unduh Laporan" if st.session_state.language == 'id' else "Download Report",
                            data=report_bytes,
                            file_name=f"cluster_report_spectral_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                
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
        # Check if time series data is available
        time_columns = [col for col in data.columns if any(keyword in str(col).lower() for keyword in ['date', 'time', 'tanggal', 'waktu', 'year', 'month', 'day'])]
        
        if data[target_column].dtype in ['int64', 'float64']:
            if len(data[target_column].unique()) <= 10:
                problem_type_options = ["Classification", "Regression", "Forecasting"] if time_columns else ["Classification", "Regression"]
                problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=0)
            else:
                problem_type_options = ["Classification", "Regression", "Forecasting"] if time_columns else ["Classification", "Regression"]
                problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=1)
        else:
            problem_type_options = ["Classification", "Forecasting"] if time_columns else ["Classification"]
            problem_type = st.radio("Pilih jenis masalah:" if st.session_state.language == 'id' else "Select problem type:", problem_type_options, index=0)
        
        st.session_state.problem_type = problem_type
        
        st.subheader("Atasi Nilai Hilang" if st.session_state.language == 'id' else "Handle Missing Values")
        
        # Store original missing counts
        if 'original_missing_counts' not in st.session_state:
            st.session_state.original_missing_counts = data.isnull().sum().to_dict()
        
        # Display columns with missing values
        missing_cols = data.columns[data.isnull().any()].tolist()
        
        if missing_cols:
            st.write("Kolom yang memiliki nilai hilang:" if st.session_state.language == 'id' else "Columns with missing values:", ", ".join(missing_cols))
            
            # Advanced missing value handling options
            st.markdown("### 🔧 Metode Penanganan Nilai Hilang" if st.session_state.language == 'id' else "### 🔧 Missing Value Handling Methods")
            
            # Group handling options
            group_handling = st.checkbox("Terapkan metode yang sama untuk semua kolom" if st.session_state.language == 'id' else "Apply same method for all columns", key="group_missing")
            
            if group_handling:
                # Advanced methods selection
                st.markdown("#### Pilih Metode Global" if st.session_state.language == 'id' else "#### Select Global Method")
                
                global_method = st.selectbox(
                    "Metode penanganan nilai hilang:" if st.session_state.language == 'id' else "Missing value handling method:",
                    [
                        "Drop rows with missing values",
                        "Simple Imputation",
                        "Advanced Imputation",
                        "Interpolation",
                        "KNN Imputation",
                        "Iterative Imputation",
                        "Forward/Backward Fill"
                    ]
                )
                
                if global_method == "Simple Imputation":
                    num_strategy = st.selectbox("Strategi untuk data numerik:" if st.session_state.language == 'id' else "Strategy for numerical data:", 
                                               ["Mean", "Median", "Mode", "Zero", "Min", "Max"])
                    cat_strategy = st.selectbox("Strategi untuk data kategorikal:" if st.session_state.language == 'id' else "Strategy for categorical data:", 
                                               ["Mode", "New Category", "Drop"])
                    
                    if st.button("Terapkan Metode Global" if st.session_state.language == 'id' else "Apply Global Method"):
                        for col in missing_cols:
                            col_type = "numerical" if data[col].dtype in ['int64', 'float64'] else "categorical"
                            
                            if col_type == "numerical":
                                if num_strategy == "Mean":
                                    data[col] = data[col].fillna(data[col].mean())
                                elif num_strategy == "Median":
                                    data[col] = data[col].fillna(data[col].median())
                                elif num_strategy == "Mode":
                                    data[col] = data[col].fillna(data[col].mode()[0])
                                elif num_strategy == "Zero":
                                    data[col] = data[col].fillna(0)
                                elif num_strategy == "Min":
                                    data[col] = data[col].fillna(data[col].min())
                                elif num_strategy == "Max":
                                    data[col] = data[col].fillna(data[col].max())
                            else:
                                if cat_strategy == "Mode":
                                    data[col] = data[col].fillna(data[col].mode()[0])
                                elif cat_strategy == "New Category":
                                    data[col] = data[col].fillna("Unknown")
                                elif cat_strategy == "Drop":
                                    data = data.dropna(subset=[col])
                        
                        st.success("Metode global berhasil diterapkan!" if st.session_state.language == 'id' else "Global method applied successfully!")
                        st.rerun()
            
            else:
                # Individual column handling
                for col in missing_cols:
                    col_type = "numerical" if data[col].dtype in ['int64', 'float64'] else "categorical"
                    
                    with st.expander(f"⚙️ {col} ({col_type})", expanded=False):
                        method = st.selectbox(
                            f"Metode untuk {col}:" if st.session_state.language == 'id' else f"Method for {col}:",
                            [
                                "Drop rows",
                                "Mean",
                                "Median", 
                                "Mode",
                                "Zero",
                                "Min",
                                "Max",
                                "Forward Fill",
                                "Backward Fill",
                                "Linear Interpolation",
                                "Polynomial Interpolation",
                                "KNN Imputation",
                                "Iterative Imputation",
                                "Custom Value"
                            ] if col_type == "numerical" else [
                                "Drop rows",
                                "Mode",
                                "New category",
                                "Forward Fill",
                                "Backward Fill",
                                "Custom Value"
                            ],
                            key=f"method_{col}"
                        )
                        
                        if method == "Linear Interpolation" and col_type == "numerical":
                            data[col] = data[col].interpolate(method='linear')
                        elif method == "Polynomial Interpolation" and col_type == "numerical":
                            order = st.slider("Orde polinomial:" if st.session_state.language == 'id' else "Polynomial order:", 2, 5, 2, key=f"poly_order_{col}")
                            data[col] = data[col].interpolate(method='polynomial', order=order)
                        elif method == "Forward Fill":
                            data[col] = data[col].fillna(method='ffill')
                        elif method == "Backward Fill":
                            data[col] = data[col].fillna(method='bfill')
                        elif method == "KNN Imputation" and col_type == "numerical":
                            try:
                                from sklearn.impute import KNNImputer
                                imputer = KNNImputer(n_neighbors=5)
                                data[[col]] = imputer.fit_transform(data[[col]])
                            except ImportError:
                                st.warning("KNN Imputation memerlukan scikit-learn. Menggunakan median sebagai fallback." if st.session_state.language == 'id' else "KNN Imputation requires scikit-learn. Using median as fallback.")
                                data[col] = data[col].fillna(data[col].median())
                        elif method == "Iterative Imputation" and col_type == "numerical":
                            try:
                                from sklearn.impute import IterativeImputer
                                from sklearn.ensemble import RandomForestRegressor
                                imputer = IterativeImputer(estimator=RandomForestRegressor(random_state=42), random_state=42)
                                data[[col]] = imputer.fit_transform(data[[col]])
                            except ImportError:
                                st.warning("Iterative Imputation memerlukan scikit-learn. Menggunakan median sebagai fallback." if st.session_state.language == 'id' else "Iterative Imputation requires scikit-learn. Using median as fallback.")
                                data[col] = data[col].fillna(data[col].median())
                        elif method == "Drop rows":
                            data = data.dropna(subset=[col])
                        elif method == "Mean" and col_type == "numerical":
                            data[col] = data[col].fillna(data[col].mean())
                        elif method == "Median" and col_type == "numerical":
                            data[col] = data[col].fillna(data[col].median())
                        elif method == "Mode":
                            mode_val = data[col].mode()[0] if len(data[col].mode()) > 0 else (0 if col_type == "numerical" else "Unknown")
                            data[col] = data[col].fillna(mode_val)
                        elif method == "Zero" and col_type == "numerical":
                            data[col] = data[col].fillna(0)
                        elif method == "Min" and col_type == "numerical":
                            data[col] = data[col].fillna(data[col].min())
                        elif method == "Max" and col_type == "numerical":
                            data[col] = data[col].fillna(data[col].max())
                        elif method == "New category" and col_type != "numerical":
                            data[col] = data[col].fillna("Unknown")
                        elif method == "Custom Value":
                            custom_value = st.text_input("Masukkan nilai kustom:" if st.session_state.language == 'id' else "Enter custom value:", key=f"custom_{col}")
                            if custom_value:
                                try:
                                    if col_type == "numerical":
                                        custom_value = float(custom_value)
                                    data[col] = data[col].fillna(custom_value)
                                except ValueError:
                                    st.error("Nilai kustom tidak valid untuk tipe data ini" if st.session_state.language == 'id' else "Invalid custom value for this data type")
                        
                        # Show preview after handling
                        if st.button("Tampilkan Preview" if st.session_state.language == 'id' else "Show Preview", key=f"preview_{col}"):
                            missing_after = data[col].isnull().sum()
                            st.info(f"Sisa nilai hilang di {col}: {missing_after}" if st.session_state.language == 'id' else f"Remaining missing values in {col}: {missing_after}")
                            st.write(data[col].describe() if col_type == "numerical" else data[col].value_counts())
                            
                            # Visual comparison before and after
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**Sebelum:**" if st.session_state.language == 'id' else "**Before:**")
                                if col_type == "numerical":
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    data[col].hist(bins=20, ax=ax)
                                    ax.set_title(f"{col} - Original")
                                    st.pyplot(fig)
                                else:
                                    st.write("Original distribution")
                            
                            with col2:
                                st.write("**Sesudah:**" if st.session_state.language == 'id' else "**After:**")
                                if col_type == "numerical":
                                    fig, ax = plt.subplots(figsize=(4, 3))
                                    data[col].hist(bins=20, ax=ax)
                                    ax.set_title(f"{col} - After Imputation")
                                    st.pyplot(fig)
                                else:
                                    st.write("After imputation")
        else:
            st.success("Tidak ditemukan nilai yang hilang dalam dataset." if st.session_state.language == 'id' else "No missing values found in the dataset.")

        # Summary section
        st.markdown("---")
        st.subheader("📊 Ringkasan Penanganan Nilai Hilang" if st.session_state.language == 'id' else "📊 Missing Value Handling Summary")
        
        # Create missing summary with proper Series alignment
        original_missing = pd.Series(st.session_state.get('original_missing_counts', {}))
        missing_after = data.isnull().sum()
        
        missing_summary = pd.DataFrame({
            'Column': data.columns,
            'Missing_Before': [original_missing.get(col, 0) for col in data.columns],
            'Missing_After': missing_after.values,
            'Percentage_Imputed': [(original_missing.get(col, 0) - missing_after[col]) / len(data) * 100 for col in data.columns]
        })
        
        missing_summary = missing_summary[missing_summary['Missing_Before'] > 0]
        
        if not missing_summary.empty:
            st.dataframe(missing_summary)
            
            # Download handled data
            csv = data.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Data yang Sudah Diperbaiki" if st.session_state.language == 'id' else "⬇️ Download Cleaned Data",
                data=csv,
                file_name="data_tanpa_missing.csv" if st.session_state.language == 'id' else "cleaned_data.csv",
                mime="text/csv"
            )
        
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
            if len(class_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                class_counts.plot(kind='bar', ax=ax)
                plt.title('Distribusi Kelas' if st.session_state.language == 'id' else 'Class Distribution')
                plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
                plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
                st.pyplot(fig)
            else:
                st.warning("Tidak ada data untuk kolom target yang dipilih." if st.session_state.language == 'id' else "No data available for the selected target column.")
            
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
            
            if len(class_counts) > 0:
                fig, ax = plt.subplots(figsize=(10, 4))
                class_counts.plot(kind='bar', ax=ax)
                plt.title('Distribusi Kelas' if st.session_state.language == 'id' else 'Class Distribution')
                plt.ylabel('Jumlah' if st.session_state.language == 'id' else 'Count')
                plt.xlabel('Kelas' if st.session_state.language == 'id' else 'Class')
                st.pyplot(fig)
            else:
                st.warning("Tidak ada data untuk kolom target yang dipilih" if st.session_state.language == 'id' else "No data available for selected target column")
            
        # Update all_columns setelah encoding
        all_columns = [col for col in data.columns if col != target_column]

        # Train-test split
        st.subheader("Lakukan Train-Test Split" if st.session_state.language == 'id' else "Train-Test Split")

        test_size = st.slider("Ukuran set pengujian (persen):" if st.session_state.language == 'id' else "Test set size (%):", 10, 50, 20) / 100
        random_state = st.number_input("Status acak:" if st.session_state.language == 'id' else "Random state:", 0, 100, 42)

        # Prepare data for modeling dengan semua fitur awal
        X = data[all_columns]
        y = data[target_column]

        # Validasi jumlah sampel sebelum train test split
        if len(X) == 0:
            st.error("Tidak ada data untuk diproses. Pastikan dataset memiliki minimal 1 baris data." if st.session_state.language == 'id' else "No data to process. Please ensure your dataset has at least 1 row of data.")
            st.stop()
        elif len(X) < 2:
            st.error("Dataset terlalu kecil. Diperlukan minimal 2 sampel untuk train-test split." if st.session_state.language == 'id' else "Dataset too small. At least 2 samples required for train-test split.")
            st.stop()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        # Tambahkan normalisasi setelah train test split
        st.subheader("Normalisasi Fitur" if st.session_state.language == 'id' else "Feature Normalization")

        normalization_method = st.selectbox(
            "Metode normalisasi:" if st.session_state.language == 'id' else "Normalization method:",
            ["None", "StandardScaler", "MinMaxScaler", "RobustScaler"]
        )

        # Outlier detection before scaling
        detect_outliers = st.checkbox(
            "Deteksi Outlier Sebelum Scaling" if st.session_state.language == 'id' else "Detect Outliers Before Scaling",
            value=False,
            help="Deteksi dan tangani outlier sebelum menerapkan normalisasi" if st.session_state.language == 'id' else "Detect and handle outliers before applying normalization"
        )
        
        if detect_outliers:
            st.subheader("Deteksi Outlier" if st.session_state.language == 'id' else "Outlier Detection")
            
            # Get numeric columns for outlier detection
            numeric_cols_outlier = X_train.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols_outlier) > 0:
                # Outlier detection method
                outlier_method = st.selectbox(
                    "Metode deteksi outlier:" if st.session_state.language == 'id' else "Outlier detection method:",
                    ["IQR Method", "Z-Score Method"]
                )
                
                # Parameters for outlier detection
                if outlier_method == "IQR Method":
                    iqr_multiplier = st.slider(
                        "IQR Multiplier:" if st.session_state.language == 'id' else "IQR Multiplier:",
                        1.0, 3.0, 1.5, 0.1,
                        help="Semakin tinggi nilai, semakin sedikit outlier yang terdeteksi" if st.session_state.language == 'id' else "Higher values detect fewer outliers"
                    )
                else:
                    z_threshold = st.slider(
                        "Z-Score Threshold:" if st.session_state.language == 'id' else "Z-Score Threshold:",
                        2.0, 4.0, 3.0, 0.1,
                        help="Semakin tinggi nilai, semakin sedikit outlier yang terdeteksi" if st.session_state.language == 'id' else "Higher values detect fewer outliers"
                    )
                
                # Handle outliers method
                handle_method = st.selectbox(
                    "Metode penanganan outlier:" if st.session_state.language == 'id' else "Outlier handling method:",
                    ["Hapus Outlier" if st.session_state.language == 'id' else "Remove Outliers",
                     "Ganti dengan Batas IQR" if st.session_state.language == 'id' else "Replace with IQR Bounds",
                     "Ganti dengan Median" if st.session_state.language == 'id' else "Replace with Median"]
                )
                
                if st.button("Deteksi dan Tangani Outlier" if st.session_state.language == 'id' else "Detect and Handle Outliers"):
                    try:
                        # Detect outliers
                        outlier_mask = pd.DataFrame(False, index=X_train.index, columns=numeric_cols_outlier)
                        outlier_summary = {}
                        
                        for col in numeric_cols_outlier:
                            if outlier_method == "IQR Method":
                                Q1 = X_train[col].quantile(0.25)
                                Q3 = X_train[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                
                                col_outliers = (X_train[col] < lower_bound) | (X_train[col] > upper_bound)
                                outlier_mask[col] = col_outliers
                                
                                outlier_summary[col] = {
                                    'count': col_outliers.sum(),
                                    'percentage': (col_outliers.sum() / len(X_train)) * 100,
                                    'lower_bound': lower_bound,
                                    'upper_bound': upper_bound
                                }
                            else:  # Z-Score Method
                                z_scores = np.abs((X_train[col] - X_train[col].mean()) / X_train[col].std())
                                col_outliers = z_scores > z_threshold
                                outlier_mask[col] = col_outliers
                                
                                outlier_summary[col] = {
                                    'count': col_outliers.sum(),
                                    'percentage': (col_outliers.sum() / len(X_train)) * 100,
                                    'threshold': z_threshold
                                }
                        
                        # Display outlier summary
                        st.write("**Ringkasan Outlier:**" if st.session_state.language == 'id' else "**Outlier Summary:**")
                        summary_df = pd.DataFrame(outlier_summary).T
                        st.dataframe(summary_df)
                        
                        # Handle outliers
                        total_outliers = outlier_mask.any(axis=1).sum()
                        
                        if handle_method == "Hapus Outlier" or handle_method == "Remove Outliers":
                            # Remove rows with outliers
                            clean_mask = ~outlier_mask.any(axis=1)
                            X_train = X_train[clean_mask]
                            y_train = y_train[clean_mask]
                            st.success(f"Menghapus {total_outliers} baris dengan outlier. Sisa data: {len(X_train)} baris" if st.session_state.language == 'id' else f"Removed {total_outliers} rows with outliers. Remaining data: {len(X_train)} rows")
                            
                        elif handle_method == "Ganti dengan Batas IQR" or handle_method == "Replace with IQR Bounds":
                            # Replace outliers with IQR bounds
                            for col in numeric_cols_outlier:
                                Q1 = X_train[col].quantile(0.25)
                                Q3 = X_train[col].quantile(0.75)
                                IQR = Q3 - Q1
                                lower_bound = Q1 - iqr_multiplier * IQR
                                upper_bound = Q3 + iqr_multiplier * IQR
                                
                                X_train.loc[outlier_mask[col], col] = np.clip(
                                    X_train.loc[outlier_mask[col], col], 
                                    lower_bound, upper_bound
                                )
                            st.success(f"Mengganti {total_outliers} outlier dengan batas IQR" if st.session_state.language == 'id' else f"Replaced {total_outliers} outliers with IQR bounds")
                            
                        else:  # Replace with Median
                            # Replace outliers with median
                            for col in numeric_cols_outlier:
                                median_val = X_train[col].median()
                                X_train.loc[outlier_mask[col], col] = median_val
                            st.success(f"Mengganti {total_outliers} outlier dengan median" if st.session_state.language == 'id' else f"Replaced {total_outliers} outliers with median")
                        
                        # Update X_test with same handling if needed
                        if handle_method != "Hapus Outlier" and handle_method != "Remove Outliers":
                            for col in numeric_cols_outlier:
                                if outlier_method == "IQR Method":
                                    Q1 = X_test[col].quantile(0.25)
                                    Q3 = X_test[col].quantile(0.75)
                                    IQR = Q3 - Q1
                                    lower_bound = Q1 - iqr_multiplier * IQR
                                    upper_bound = Q3 + iqr_multiplier * IQR
                                    
                                    test_outliers = (X_test[col] < lower_bound) | (X_test[col] > upper_bound)
                                    if handle_method == "Ganti dengan Batas IQR" or handle_method == "Replace with IQR Bounds":
                                        X_test.loc[test_outliers, col] = np.clip(
                                            X_test.loc[test_outliers, col], 
                                            lower_bound, upper_bound
                                        )
                                    else:  # Replace with Median
                                        X_test.loc[test_outliers, col] = X_test[col].median()
                                else:  # Z-Score Method
                                    z_scores = np.abs((X_test[col] - X_test[col].mean()) / X_test[col].std())
                                    test_outliers = z_scores > z_threshold
                                    if handle_method == "Ganti dengan Median" or handle_method == "Replace with Median":
                                        X_test.loc[test_outliers, col] = X_test[col].median()
                        
                    except Exception as e:
                        st.error(f"Error saat deteksi outlier: {str(e)}" if st.session_state.language == 'id' else f"Error during outlier detection: {str(e)}")
            
            else:
                st.warning("Tidak ada fitur numerik untuk deteksi outlier" if st.session_state.language == 'id' else "No numeric features for outlier detection")

        if normalization_method != "None":
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            
            if normalization_method == "StandardScaler":
                scaler = StandardScaler()
            elif normalization_method == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif normalization_method == "RobustScaler":
                scaler = RobustScaler()
            
            # Simpan scaler ke session state untuk inverse transform
            st.session_state.scaler = scaler
            st.session_state.normalization_method = normalization_method
            st.session_state.numeric_cols = list(X_train.select_dtypes(include=[np.number]).columns)
            
            # Fit dan transform hanya pada fitur numerik
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
                X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
                
                st.success(f"Normalisasi {normalization_method} berhasil diterapkan")
                st.info(f"Fitur numerik yang dinormalisasi: {len(numeric_cols)} fitur")
                
                # Tambahkan tombol untuk inverse transform
                if st.button("Tampilkan Inverse Transform" if st.session_state.language == 'id' else "Show Inverse Transform"):
                    try:
                        # Lakukan inverse transform pada data training
                        X_train_inverse = X_train.copy()
                        X_test_inverse = X_test.copy()
                        
                        if len(numeric_cols) > 0:
                            X_train_inverse[numeric_cols] = scaler.inverse_transform(X_train[numeric_cols])
                            X_test_inverse[numeric_cols] = scaler.inverse_transform(X_test[numeric_cols])
                        
                        # Tampilkan perbandingan
                        st.subheader("Perbandingan Data Asli vs Dinormalisasi" if st.session_state.language == 'id' else "Comparison of Original vs Normalized Data")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Data Setelah Normalisasi:**" if st.session_state.language == 'id' else "**Normalized Data:**")
                            st.dataframe(X_train[numeric_cols[:5]].head())
                        with col2:
                            st.write("**Data Asli (Inverse Transform):**" if st.session_state.language == 'id' else "**Original Data (Inverse Transform):**")
                            st.dataframe(X_train_inverse[numeric_cols[:5]].head())
                        
                        # Tampilkan statistik perbandingan
                        st.write("**Statistik Perbandingan:**" if st.session_state.language == 'id' else "**Comparison Statistics:**")
                        comparison_stats = pd.DataFrame({
                            'Fitur': numeric_cols[:5],
                            'Mean_Normalized': X_train[numeric_cols[:5]].mean(),
                            'Mean_Original': X_train_inverse[numeric_cols[:5]].mean(),
                            'Std_Normalized': X_train[numeric_cols[:5]].std(),
                            'Std_Original': X_train_inverse[numeric_cols[:5]].std()
                        })
                        st.dataframe(comparison_stats)
                        
                    except Exception as e:
                        st.error(f"Error saat inverse transform: {str(e)}" if st.session_state.language == 'id' else f"Error during inverse transform: {str(e)}")
        
        # Handle class imbalance for training data (classification only)
        if st.session_state.problem_type == "Classification" and IMB_AVAILABLE:
            st.subheader("Penanganan Ketidakseimbangan Dataset" if st.session_state.language == 'id' else "Handle Class Imbalance")
            
            # Check for class imbalance
            train_counts = pd.Series(y_train).value_counts()
            imbalance_ratio = train_counts.max() / train_counts.min()
            
            if imbalance_ratio > 3.0:  # Only show if there's significant imbalance (ratio > 3:1)
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
                            # Validasi minimum samples untuk SMOTE-based methods
                            if balance_method in ["SMOTE", "SMOTEENN", "SMOTETomek"]:
                                min_samples_per_class = pd.Series(y_train).value_counts().min()
                                if min_samples_per_class < 6:
                                    st.error(f"Error: {balance_method} membutuhkan minimal 6 sampel per kelas. Kelas dengan jumlah terkecil memiliki {min_samples_per_class} sampel." if st.session_state.language == 'id' else 
                                            f"Error: {balance_method} requires at least 6 samples per class. Smallest class has {min_samples_per_class} samples.")
                                    st.info("Menggunakan Random Over Sampling sebagai alternatif..." if st.session_state.language == 'id' else "Using Random Over Sampling as alternative...")
                                    balance_method = "Random Over Sampling"
                            
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
                "SelectKBest (Statistical)",
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
            
            # Align the indices and fill missing values with 0
            all_labels = train_counts.index.union(test_counts.index)
            train_aligned = train_counts.reindex(all_labels, fill_value=0)
            test_aligned = test_counts.reindex(all_labels, fill_value=0)
            
            distribution_df = pd.DataFrame({
                'Label': all_labels,
                'Jumlah Data Training': train_aligned.values,
                'Jumlah Data Testing': test_aligned.values,
                'Total': train_aligned.values + test_aligned.values
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

        elif feature_selection_method == "SelectKBest (Statistical)":
            st.subheader("SelectKBest Feature Selection" if st.session_state.language == 'id' else "SelectKBest Feature Selection")
            st.info("Menggunakan SelectKBest untuk seleksi fitur berdasarkan uji statistik" if st.session_state.language == 'id' else "Using SelectKBest for feature selection based on statistical tests")
            
            # Choose scoring function based on problem type
            if problem_type == "Regression":
                score_options = {
                    "f_regression": f_regression,
                    "mutual_info_regression": mutual_info_regression
                }
                score_func_name = st.selectbox(
                    "Fungsi skor untuk regresi:" if st.session_state.language == 'id' else "Scoring function for regression:",
                    list(score_options.keys()),
                    index=0
                )
                score_func = score_options[score_func_name]
            else:  # Classification
                score_options = {
                    "f_classif": f_classif,
                    "mutual_info_classif": mutual_info_classif
                }
                score_func_name = st.selectbox(
                    "Fungsi skor untuk klasifikasi:" if st.session_state.language == 'id' else "Scoring function for classification:",
                    list(score_options.keys()),
                    index=0
                )
                score_func = score_options[score_func_name]
            
            # Choose number of features
            max_features = len(all_columns)
            default_k = min(10, max_features)
            
            k_features = st.number_input(
                "Jumlah fitur terbaik yang ingin dipilih:" if st.session_state.language == 'id' else "Number of best features to select:",
                min_value=1,
                max_value=max_features,
                value=default_k,
                step=1
            )
            
            # Cross-validation option
            use_cv = st.checkbox(
                "Gunakan Cross-Validation" if st.session_state.language == 'id' else "Use Cross-Validation",
                value=False,
                help="Gunakan cross-validation untuk evaluasi yang lebih robust" if st.session_state.language == 'id' else "Use cross-validation for more robust evaluation"
            )
            
            cv_folds = 5
            if use_cv:
                cv_folds = st.number_input(
                    "Jumlah lipatan CV:" if st.session_state.language == 'id' else "Number of CV folds:",
                    min_value=3,
                    max_value=10,
                    value=5,
                    step=1
                )
            
            if st.button("Jalankan SelectKBest" if st.session_state.language == 'id' else "Run SelectKBest"):
                try:
                    if use_cv:
                        # Use cross-validation for more robust feature selection
                        from sklearn.model_selection import cross_val_score
                        
                        # Get scores for each feature individually using CV
                        cv_scores = []
                        for i, feature in enumerate(all_columns):
                            X_single = X_train_for_selection[:, i].reshape(-1, 1)
                            if problem_type == "Regression":
                                from sklearn.ensemble import RandomForestRegressor
                                model = RandomForestRegressor(n_estimators=50, random_state=42)
                                scores = cross_val_score(model, X_single, y_train_for_selection, 
                                                       cv=cv_folds, scoring='neg_mean_squared_error')
                                score = -np.mean(scores)  # Convert to positive (lower is better)
                            else:
                                from sklearn.ensemble import RandomForestClassifier
                                model = RandomForestClassifier(n_estimators=50, random_state=42)
                                scores = cross_val_score(model, X_single, y_train_for_selection, 
                                                       cv=cv_folds, scoring='accuracy')
                                score = np.mean(scores)  # Higher is better
                            
                            cv_scores.append(score)
                        
                        # Select top k features based on CV scores
                        # For regression: lower MSE is better, for classification: higher accuracy is better
                        if problem_type == "Regression":
                            top_indices = np.argsort(cv_scores)[:k_features]  # Take lowest scores (best MSE)
                        else:
                            top_indices = np.argsort(cv_scores)[-k_features:]  # Take highest scores (best accuracy)
                        
                        selected_mask = np.zeros(len(all_columns), dtype=bool)
                        selected_mask[top_indices] = True
                        selected_features = [all_columns[i] for i in top_indices]
                        
                        # Create feature scores dataframe
                        feature_scores = pd.DataFrame({
                            'Feature': all_columns,
                            'Score': cv_scores,
                            'Selected': selected_mask
                        })
                        
                        if problem_type == "Regression":
                            feature_scores = feature_scores.sort_values('Score', ascending=True)  # Lower MSE is better
                        else:
                            feature_scores = feature_scores.sort_values('Score', ascending=False)  # Higher accuracy is better
                        
                        # Apply selection to training data
                        X_train_selected = X_train_for_selection[:, selected_mask]
                        
                    else:
                        # Original SelectKBest without cross-validation
                        selector = SelectKBest(score_func=score_func, k=k_features)
                        
                        # Fit and transform the training data
                        X_train_selected = selector.fit_transform(X_train_for_selection, y_train_for_selection)
                        
                        # Get selected feature names
                        selected_mask = selector.get_support()
                        selected_features = [all_columns[i] for i, selected in enumerate(selected_mask) if selected]
                        
                        # Get feature scores
                        feature_scores = pd.DataFrame({
                            'Feature': all_columns,
                            'Score': selector.scores_,
                            'Selected': selected_mask
                        }).sort_values('Score', ascending=False)
                    
                    # Display results
                    st.success("SelectKBest selesai!" if st.session_state.language == 'id' else "SelectKBest completed!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Jumlah fitur terpilih" if st.session_state.language == 'id' else "Selected features count", 
                                len(selected_features))
                        st.metric("Total fitur" if st.session_state.language == 'id' else "Total features", 
                                len(all_columns))
                    
                    with col2:
                        st.metric("Persentase fitur terpilih" if st.session_state.language == 'id' else "Feature selection ratio", 
                                f"{len(selected_features)/len(all_columns)*100:.1f}%")
                    
                    # Display selected features
                    st.write("**Fitur yang dipilih:**" if st.session_state.language == 'id' else "**Selected features:**")
                    st.write(selected_features)
                    
                    # Display feature scores
                    st.write("**Skor fitur:**" if st.session_state.language == 'id' else "**Feature scores:**")
                    st.dataframe(feature_scores[['Feature', 'Score', 'Selected']])
                    
                    # Visualize feature scores
                    if len(all_columns) > 1:
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                        
                        # Top features bar plot
                        top_features = feature_scores.head(min(15, len(feature_scores)))
                        ax1.barh(top_features['Feature'], top_features['Score'])
                        ax1.set_xlabel('Score' if st.session_state.language == 'id' else 'Score')
                        ax1.set_title('Top 15 Fitur Berdasarkan Skor' if st.session_state.language == 'id' else 'Top 15 Features by Score')
                        ax1.invert_yaxis()
                        
                        # Selected vs not selected
                        selection_counts = feature_scores['Selected'].value_counts()
                        labels = ['Terpilih' if st.session_state.language == 'id' else 'Selected', 
                                'Tidak Terpilih' if st.session_state.language == 'id' else 'Not Selected']
                        colors = ['green', 'red']
                        ax2.pie(selection_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
                        ax2.set_title('Distribusi Seleksi Fitur' if st.session_state.language == 'id' else 'Feature Selection Distribution')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Download feature selection results
                        csv = feature_scores.to_csv(index=False)
                        st.download_button(
                            label="Unduh hasil seleksi fitur (CSV)" if st.session_state.language == 'id' else "Download feature selection results (CSV)",
                            data=csv,
                            file_name="selectkbest_results.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"Error saat menjalankan SelectKBest: {str(e)}" if st.session_state.language == 'id' else 
                            f"Error running SelectKBest: {str(e)}")
        
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
                        "Random Forest Importance",
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
        
        # Cross Validation Options
        st.subheader("Pilihan Validasi Silang" if st.session_state.language == 'id' else "Cross Validation Options")
        
        cv_options = [
            "None (Holdout Validation)",
            "K-Fold Cross Validation", 
            "Stratified K-Fold Cross Validation",
            "Leave-One-Out Cross Validation",
            "Leave-P-Out Cross Validation"
        ]
        
        cv_method = st.selectbox(
            "Pilih metode validasi silang:" if st.session_state.language == 'id' else "Select cross validation method:",
            cv_options
        )
        
        cv_params = {}
        
        if cv_method == "K-Fold Cross Validation":
            from sklearn.model_selection import KFold, cross_val_score
            
            n_splits = st.slider("Jumlah fold (K):" if st.session_state.language == 'id' else "Number of folds (K):", 2, 10, 5)
            cv_params['cv'] = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_params['name'] = f"K-Fold (K={n_splits})"
            
        elif cv_method == "Stratified K-Fold Cross Validation":
            from sklearn.model_selection import StratifiedKFold, cross_val_score
            
            n_splits = st.slider("Jumlah fold (K):" if st.session_state.language == 'id' else "Number of folds (K):", 2, 10, 5)
            cv_params['cv'] = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            cv_params['name'] = f"Stratified K-Fold (K={n_splits})"
            
        elif cv_method == "Leave-One-Out Cross Validation":
            from sklearn.model_selection import LeaveOneOut, cross_val_score
            
            cv_params['cv'] = LeaveOneOut()
            cv_params['name'] = "Leave-One-Out"
            
        elif cv_method == "Leave-P-Out Cross Validation":
            from sklearn.model_selection import LeavePOut, cross_val_score
            
            max_p = min(5, len(X) - 1)
            p_value = st.slider("Nilai P:" if st.session_state.language == 'id' else "P value:", 1, max_p, 2)
            cv_params['cv'] = LeavePOut(p=p_value)
            cv_params['name'] = f"Leave-{p_value}-Out"
            
        else:  # None (Holdout)
            cv_params['cv'] = None
            cv_params['name'] = "Holdout Validation"
        
        # Select evaluation metric
        if cv_params['cv'] is not None:
            st.subheader("Pengaturan Evaluasi" if st.session_state.language == 'id' else "Evaluation Settings")
            
            if problem_type == "Classification":
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:" if st.session_state.language == 'id' else "Evaluation metric:",
                    ["accuracy", "precision", "recall", "f1", "roc_auc"]
                )
                cv_params['scoring'] = cv_scoring
            else:  # Regression
                cv_scoring = st.selectbox(
                    "Metrik evaluasi:" if st.session_state.language == 'id' else "Evaluation metric:",
                    ["neg_mean_squared_error", "neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"]
                )
                cv_params['scoring'] = cv_scoring
                
            # Display data distribution for classification with stratified k-fold
            if problem_type == "Classification" and cv_method == "Stratified K-Fold Cross Validation":
                st.write("**Distribusi Data per Fold:**" if st.session_state.language == 'id' else "**Data Distribution per Fold:**")
                
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                fold_info = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
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
                                     ["ARIMA", "SARIMA", "Exponential Smoothing", "Prophet", "Random Forest", 
                                      "Gradient Boosting", "Linear Regression", "SARIMAX", "Holt-Winters", "LSTM"])
            
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
                        
                        if stationarity_result['Message']:
                            st.warning(stationarity_result['Message'])
                        
                        if stationarity_result['Test Statistic'] is not None:
                            st.write(f"- Test Statistic: {stationarity_result['Test Statistic']:.4f}")
                        if stationarity_result['p-value'] is not None:
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
                        
                        # Import all required forecasting functions
                        try:
                            from forecasting_utils import (
                                train_arima_model, train_exponential_smoothing, train_sarima_model,
                                train_sarimax_model, train_holt_winters, train_lstm_model,
                                train_ml_forecaster, forecast_future, evaluate_forecast_model, 
                                plot_forecast_results
                            )
                        except ImportError as e:
                            st.error(f"Error importing forecasting functions: {str(e)}")
                        
                        # Train model based on selection
                        if model_type == "ARIMA" and STATSMODELS_AVAILABLE:
                            p = st.slider("Parameter p (AR):", 0, 5, 1)
                            d = st.slider("Parameter d (differencing):", 0, 2, 1)
                            q = st.slider("Parameter q (MA):", 0, 5, 1)
                            
                            with st.spinner("Melatih model ARIMA..." if st.session_state.language == 'id' else "Training ARIMA model..."):
                                try:
                                    model = train_arima_model(train_data, target_column, order=(p, d, q))
                                    st.session_state.model = model
                                    st.success("Model ARIMA berhasil dilatih!" if st.session_state.language == 'id' else "ARIMA model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training ARIMA: {str(e)}")
                        
                        elif model_type == "SARIMA" and STATSMODELS_AVAILABLE:
                            p = st.slider("Parameter p (AR):", 0, 5, 1)
                            d = st.slider("Parameter d (differencing):", 0, 2, 1)
                            q = st.slider("Parameter q (MA):", 0, 5, 1)
                            P = st.slider("Parameter P (Seasonal AR):", 0, 2, 1)
                            D = st.slider("Parameter D (Seasonal differencing):", 0, 2, 1)
                            Q = st.slider("Parameter Q (Seasonal MA):", 0, 2, 1)
                            s = st.slider("Periode musiman (s):", 1, 52, 12)
                            
                            with st.spinner("Melatih model SARIMA..." if st.session_state.language == 'id' else "Training SARIMA model..."):
                                try:
                                    model = train_sarima_model(
                                        train_data, 
                                        target_column, 
                                        order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)
                                    )
                                    st.session_state.model = model
                                    st.success("Model SARIMA berhasil dilatih!" if st.session_state.language == 'id' else "SARIMA model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training SARIMA: {str(e)}")
                        
                        elif model_type == "SARIMAX" and STATSMODELS_AVAILABLE:
                            p = st.slider("Parameter p (AR):", 0, 5, 1)
                            d = st.slider("Parameter d (differencing):", 0, 2, 1)
                            q = st.slider("Parameter q (MA):", 0, 5, 1)
                            P = st.slider("Parameter P (Seasonal AR):", 0, 2, 1)
                            D = st.slider("Parameter D (Seasonal differencing):", 0, 2, 1)
                            Q = st.slider("Parameter Q (Seasonal MA):", 0, 2, 1)
                            s = st.slider("Periode musiman (s):", 1, 52, 12)
                            
                            with st.spinner("Melatih model SARIMAX..." if st.session_state.language == 'id' else "Training SARIMAX model..."):
                                try:
                                    model = train_sarimax_model(
                                        train_data, 
                                        target_column, 
                                        order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)
                                    )
                                    st.session_state.model = model
                                    st.success("Model SARIMAX berhasil dilatih!" if st.session_state.language == 'id' else "SARIMAX model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training SARIMAX: {str(e)}")
                        
                        elif model_type == "Exponential Smoothing" and STATSMODELS_AVAILABLE:
                            trend = st.selectbox("Tipe trend:", ["add", "mul", None])
                            seasonal = st.selectbox("Tipe seasonal:", ["add", "mul", None])
                            seasonal_periods = st.slider("Periode seasonal:", 0, 52, 12)
                            
                            with st.spinner("Melatih model Exponential Smoothing..." if st.session_state.language == 'id' else "Training Exponential Smoothing model..."):
                                try:
                                    model = train_exponential_smoothing(
                                        train_data, 
                                        target_column, 
                                        trend=trend, 
                                        seasonal=seasonal, 
                                        seasonal_periods=seasonal_periods
                                    )
                                    st.session_state.model = model
                                    st.success("Model Exponential Smoothing berhasil dilatih!" if st.session_state.language == 'id' else "Exponential Smoothing model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training Exponential Smoothing: {str(e)}")
                        
                        elif model_type == "Holt-Winters" and STATSMODELS_AVAILABLE:
                            trend = st.selectbox("Tipe trend:", ["add", "mul"])
                            seasonal = st.selectbox("Tipe seasonal:", ["add", "mul"])
                            seasonal_periods = st.slider("Periode seasonal:", 1, 52, 12)
                            
                            with st.spinner("Melatih model Holt-Winters..." if st.session_state.language == 'id' else "Training Holt-Winters model..."):
                                try:
                                    model = train_holt_winters(
                                        train_data, 
                                        target_column, 
                                        trend=trend, 
                                        seasonal=seasonal, 
                                        seasonal_periods=seasonal_periods
                                    )
                                    st.session_state.model = model
                                    st.success("Model Holt-Winters berhasil dilatih!" if st.session_state.language == 'id' else "Holt-Winters model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training Holt-Winters: {str(e)}")
                        
                        elif model_type == "Prophet" and PROPHET_AVAILABLE:
                            yearly_seasonality = st.selectbox("Seasonality tahunan:" if st.session_state.language == 'id' else "Yearly seasonality:", ["auto", True, False])
                            weekly_seasonality = st.selectbox("Seasonality mingguan:" if st.session_state.language == 'id' else "Weekly seasonality:", ["auto", True, False])
                            daily_seasonality = st.selectbox("Seasonality harian:" if st.session_state.language == 'id' else "Daily seasonality:", ["auto", True, False])
                            
                            # Implementasi Prophet akan dilakukan di forecasting_utils.py
                            st.info("Implementasi Prophet akan menggunakan forecasting_utils.py" if st.session_state.language == 'id' else "Prophet implementation will use forecasting_utils.py")
                        
                        elif model_type == "LSTM":
                            look_back = st.slider("Jumlah time steps untuk look back:", 10, 100, 60)
                            epochs = st.slider("Jumlah epochs:", 10, 200, 100)
                            batch_size = st.slider("Batch size:", 16, 128, 32)
                            
                            # Parameter arsitektur lanjutan dengan expander
                            with st.expander("Parameter Arsitektur Lanjutan" if st.session_state.language == 'id' else "Advanced Architecture Parameters"):
                                lstm_units = st.slider("Unit LSTM per layer:" if st.session_state.language == 'id' else "LSTM units per layer:", 10, 200, 50)
                                num_layers = st.slider("Jumlah layer LSTM:" if st.session_state.language == 'id' else "Number of LSTM layers:", 1, 5, 2)
                                dropout = st.slider("Dropout rate:" if st.session_state.language == 'id' else "Dropout rate:", 0.0, 0.5, 0.2, 0.05)
                                recurrent_dropout = st.slider("Recurrent dropout rate:" if st.session_state.language == 'id' else "Recurrent dropout rate:", 0.0, 0.5, 0.2, 0.05)
                                bidirectional = st.checkbox("Gunakan LSTM bidirectional:" if st.session_state.language == 'id' else "Use bidirectional LSTM:", value=False)
                                learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.0001, 0.01, 0.001, 0.0001)
                                optimizer = st.selectbox("Optimizer:" if st.session_state.language == 'id' else "Optimizer:", ["adam", "sgd", "rmsprop"], index=0)
                            
                            with st.spinner("Melatih model LSTM..." if st.session_state.language == 'id' else "Training LSTM model..."):
                                try:
                                    model = train_lstm_model(
                                        train_data, 
                                        target_column, 
                                        look_back=look_back, 
                                        epochs=epochs, 
                                        batch_size=batch_size,
                                        lstm_units=lstm_units,
                                        num_layers=num_layers,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        bidirectional=bidirectional,
                                        learning_rate=learning_rate,
                                        optimizer=optimizer
                                    )
                                    st.session_state.model = model
                                    st.success("Model LSTM berhasil dilatih!" if st.session_state.language == 'id' else "LSTM model trained successfully!")
                                except ImportError as e:
                                    st.error(f"TensorFlow tidak tersedia: {str(e)}")
                                except Exception as e:
                                    st.error(f"Error training LSTM: {str(e)}")
                        
                        elif model_type in ["Random Forest", "Gradient Boosting", "Linear Regression"]:
                            if model_type == "Random Forest":
                                n_estimators = st.slider("Jumlah trees:" if st.session_state.language == 'id' else "Number of trees:", 10, 500, 100)
                                max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                                
                                # Parameter lanjutan dengan expander
                                with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                                    min_samples_split = st.slider("Jumlah sampel minimum untuk membagi:" if st.session_state.language == 'id' else "Minimum samples to split:", 2, 20, 2)
                                    min_samples_leaf = st.slider("Jumlah sampel minimum di leaf:" if st.session_state.language == 'id' else "Minimum samples in leaf:", 1, 10, 1)
                                    max_features = st.selectbox("Fitur maksimum:" if st.session_state.language == 'id' else "Max features:", ["sqrt", "log2", "None"], index=0)
                                    bootstrap = st.checkbox("Bootstrap sampel:" if st.session_state.language == 'id' else "Bootstrap samples:", value=True)
                                    
                                    # Konversi max_features dari string ke None jika diperlukan
                                    max_features_value = None if max_features == "None" else max_features
                                
                                model_params = {
                                    'n_estimators': n_estimators,
                                    'max_depth': max_depth,
                                    'min_samples_split': min_samples_split,
                                    'min_samples_leaf': min_samples_leaf,
                                    'max_features': max_features_value,
                                    'bootstrap': bootstrap,
                                    'random_state': 42
                                }
                            elif model_type == "Gradient Boosting":
                                n_estimators = st.slider("Jumlah trees:" if st.session_state.language == 'id' else "Number of trees:", 10, 500, 100)
                                learning_rate = st.slider("Learning rate:", 0.01, 0.3, 0.1)
                                max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                                model_params = {
                                    'n_estimators': n_estimators,
                                    'learning_rate': learning_rate,
                                    'max_depth': max_depth,
                                    'random_state': 42
                                }
                            else:  # Linear Regression
                                model_params = {'random_state': 42}
                            
                            with st.spinner(f"Melatih model {model_type}..." if st.session_state.language == 'id' else f"Training {model_type} model..."):
                                try:
                                    model_info = train_ml_forecaster(
                                        st.session_state.data,
                                        date_column,
                                        target_column,
                                        model_type=model_type.lower().replace(" ", "_"),
                                        **model_params
                                    )
                                    st.session_state.model = model_info
                                    st.success(f"Model {model_type} berhasil dilatih!" if st.session_state.language == 'id' else f"{model_type} model trained successfully!")
                                except Exception as e:
                                    st.error(f"Error training {model_type}: {str(e)}")
                        
                        # Evaluate model if available
                        if st.session_state.model is not None:
                            with st.spinner("Mengevaluasi model..." if st.session_state.language == 'id' else "Evaluating model..."):
                                try:
                                    eval_results = evaluate_forecast_model(st.session_state.model, test_data, target_column)
                                    
                                    # Tampilkan hasil evaluasi dengan penanganan nilai None
                                    st.write("Hasil Evaluasi Model:" if st.session_state.language == 'id' else "Model Evaluation Results:")
                                    
                                    # Buat tabel evaluasi yang lebih rapi
                                    eval_df = pd.DataFrame()
                                    
                                    import math
                                    
                                    def format_metric(value, format_str="{:.4f}"):
                                        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                                            return "N/A"
                                        try:
                                            return format_str.format(value)
                                        except:
                                            return "N/A"
                                    
                                    eval_df['MAE'] = [format_metric(eval_results.get('MAE'))]
                                    eval_df['MSE'] = [format_metric(eval_results.get('MSE'))]
                                    eval_df['RMSE'] = [format_metric(eval_results.get('RMSE'))]
                                    eval_df['MAPE (%)'] = [format_metric(eval_results.get('MAPE'), "{:.2f}%")]
                                    eval_df['R²'] = [format_metric(eval_results.get('R2'))]
                                        
                                    st.dataframe(eval_df, use_container_width=True)
                                    
                                    # Tampilkan pesan error jika ada
                                    if 'error' in eval_results:
                                        st.warning(f"⚠️ {eval_results['error']}" if st.session_state.language == 'id' else f"⚠️ {eval_results['error']}")
                                    
                                    # Generate forecast
                                    try:
                                        forecast_data = forecast_future(st.session_state.model, periods=forecast_periods)
                                        
                                        # Validasi dan perbaikan data forecast
                                        if forecast_data is not None and not forecast_data.empty:
                                            # Pastikan kolom tanggal dalam format yang benar
                                            if 'date' in forecast_data.columns:
                                                try:
                                                    forecast_data['date'] = pd.to_datetime(forecast_data['date'], errors='coerce')
                                                    forecast_data = forecast_data.dropna(subset=['date'])
                                                except Exception:
                                                    # Jika gagal, biarkan sebagai string
                                                    pass
                                            
                                            # Pastikan kolom forecast ada dan valid
                                            if 'forecast' not in forecast_data.columns:
                                                st.warning("Data forecast tidak memiliki kolom 'forecast'")
                                                forecast_data = None
                                            else:
                                                # Hapus nilai forecast yang tidak valid
                                                forecast_data = forecast_data.dropna(subset=['forecast'])
                                                if forecast_data.empty:
                                                    st.warning("Data forecast kosong setelah validasi")
                                                    forecast_data = None
                                        else:
                                            st.warning("Data forecast kosong atau tidak valid")
                                            forecast_data = None
                                            
                                    except Exception as e:
                                        st.error(f"Error saat membuat forecast: {str(e)}" if st.session_state.language == 'id' else f"Error generating forecast: {str(e)}")
                                        forecast_data = None

                                    # Plot results dengan penanganan error yang lebih baik
                                    if forecast_data is not None and not forecast_data.empty:
                                        try:
                                            fig = plot_forecast_results(train_data, test_data, forecast_data, target_column)
                                            if fig is not None:
                                                st.pyplot(fig)
                                                plt.close(fig)  # Tutup figure untuk menghemat memory
                                            else:
                                                st.warning("Gagal membuat plot forecast")
                                            
                                            # Store data for visualization
                                            st.session_state.forecast_data = forecast_data
                                            st.session_state.train_data = train_data
                                            st.session_state.test_data = test_data
                                            st.session_state.target_column = target_column
                                            st.session_state.eval_results = eval_results
                                            
                                        except Exception as e:
                                            st.error(f"Error saat memplot hasil: {str(e)}" if st.session_state.language == 'id' else f"Error plotting results: {str(e)}")
                                            st.info("Menampilkan data forecast dalam bentuk tabel..." if st.session_state.language == 'id' else "Displaying forecast data in table format...")

                                    # Show forecast data dengan preview
                                    if forecast_data is not None and not forecast_data.empty:
                                        st.write("Data Hasil Forecasting:" if st.session_state.language == 'id' else "Forecast Data:")
                                        st.dataframe(forecast_data.head(50))  # Tampilkan maksimal 50 baris
                                        
                                        # Download button untuk forecast data
                                        csv = forecast_data.to_csv(index=False)
                                        st.download_button(
                                            label="Download Forecast Data (CSV)" if st.session_state.language == 'id' else "Download Forecast Data (CSV)",
                                            data=csv,
                                            file_name=f"forecast_{model_type.lower().replace(' ', '_')}.csv",
                                            mime="text/csv"
                                        )

                                except Exception as e:
                                    st.error(f"Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"Error evaluating model: {str(e)}")
                                    st.info("Pastikan model telah dilatih dengan benar dan data yang digunakan sesuai." if st.session_state.language == 'id' else "Please ensure the model is properly trained and data is appropriate.")

                        # Button for detailed visualization
                        if hasattr(st.session_state, 'forecast_data') and st.session_state.forecast_data is not None:
                            if st.button("Tampilkan Visualisasi Forecasting Lengkap" if st.session_state.language == 'id' else "Show Complete Forecasting Visualization"):
                                display_forecast_summary(
                                    st.session_state.train_data,
                                    st.session_state.test_data,
                                    st.session_state.forecast_data,
                                    st.session_state.target_column,
                                    st.session_state.eval_results
                                )

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
            
            # Opsi untuk hyperparameter optimization
            optimization_method = st.selectbox(
                "Metode Hyperparameter Optimization:" if st.session_state.language == 'id' else "Hyperparameter Optimization Method:",
                ["None", "GridSearchCV", "RandomizedSearchCV", "Bayesian Optimization (Optuna)"]
            )
            
            # Opsi rentang parameter kustom
            use_custom_ranges = False
            custom_param_ranges = {}
            
            if optimization_method in ["GridSearchCV", "RandomizedSearchCV"]:
                use_custom_ranges = st.checkbox(
                    "Gunakan rentang parameter kustom" if st.session_state.language == 'id' else "Use custom parameter ranges",
                    value=False,
                    help="Aktifkan untuk menentukan rentang parameter sendiri" if st.session_state.language == 'id' else "Enable to specify custom parameter ranges"
                )
                
                if use_custom_ranges:
                    st.info("💡 Gunakan format: min:max:step untuk numerik, atau val1,val2,val3 untuk kategorikal" if st.session_state.language == 'id' else "💡 Use format: min:max:step for numeric, or val1,val2,val3 for categorical")
                    st.info("⚠️ Kosongkan field untuk menggunakan rentang default" if st.session_state.language == 'id' else "⚠️ Leave field empty to use default ranges")
            
            # Model selection
            if problem_type == "Classification":
                # Define available classification models
                classification_models = ["Random Forest", "Logistic Regression", "SVM", "KNN", "Decision Tree", "Naive Bayes", "Gradient Boosting", "MLP (Neural Network)"]
                                   
                model_type = st.selectbox("Select a classification model:" if st.session_state.language == 'id' else "Pilih model klasifikasi:", classification_models)
                st.session_state.model_type = model_type
                
                if model_type == "Random Forest":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of trees:" if st.session_state.language == 'id' else "Jumlah pohon:", 10, 500, 100)
                    max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                    
                    # Parameter lanjutan dengan expander
                    with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                        min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel minimum untuk membagi:", 2, 20, 2)
                        min_samples_leaf = st.slider("Minimum samples in leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum di leaf:", 1, 10, 1)
                        max_features = st.selectbox("Max features:" if st.session_state.language == 'id' else "Fitur maksimum:", ["sqrt", "log2", "None"], index=0)
                        bootstrap = st.checkbox("Bootstrap samples:" if st.session_state.language == 'id' else "Bootstrap sampel:", value=True)
                        
                        # Konversi max_features dari string ke None jika diperlukan
                        max_features_value = None if max_features == "None" else max_features
                    
                    base_model = RandomForestClassifier(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'max_depth': list(range(3, 21)) + [None],
                            'min_samples_split': [2, 5, 10, 15, 20],
                            'min_samples_leaf': [1, 2, 4, 8, 16],
                            'max_features': ['sqrt', 'log2', None]
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=50, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "Random Forest", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = RandomForestClassifier(**best_params, random_state=42)
                    else:
                        model = RandomForestClassifier(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features_value,
                            bootstrap=bootstrap,
                            random_state=42
                        )
                        
                elif model_type == "Logistic Regression" :
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.01, 10.0, 1.0)
                    max_iter = st.slider("Maximum iterations:" if st.session_state.language == 'id' else "Iterasi maksimum:", 100, 1000, 100)
                    
                    # Parameter lanjutan dalam expander
                    with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                        class_weight = st.selectbox("Class weight:" if st.session_state.language == 'id' else "Bobot kelas:", ["None", "balanced", "balanced_subsample"], index=0)
                        solver = st.selectbox("Solver algorithm:" if st.session_state.language == 'id' else "Algoritma solver:", ["lbfgs", "liblinear", "saga", "newton-cg", "sag"], index=0)
                        penalty = st.selectbox("Penalty type:" if st.session_state.language == 'id' else "Jenis penalti:", ["l2", "l1", "elasticnet"], index=0)
                        
                        # Konversi class_weight dari string ke None jika diperlukan
                        class_weight_value = None if class_weight == "None" else class_weight
                        
                        # Elastic net memerlukan parameter l1_ratio
                        l1_ratio = None
                        if penalty == "elasticnet":
                            l1_ratio = st.slider("L1 ratio (for elasticnet):" if st.session_state.language == 'id' else "Rasio L1 (untuk elasticnet):", 0.0, 1.0, 0.5)
                    
                    base_model = LogisticRegression(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.01, C/2), C, min(10.0, C*2)],
                            'solver': ['liblinear', 'lbfgs', 'saga'],
                            'max_iter': [100, 500, 1000]
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                            'solver': ['liblinear', 'lbfgs', 'saga'],
                            'max_iter': [100, 500, 1000],
                            'penalty': ['l1', 'l2', 'elasticnet']
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "Logistic Regression", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = LogisticRegression(**best_params, random_state=42)
                    else:
                        model = LogisticRegression(
                            C=C,
                            max_iter=max_iter,
                            class_weight=class_weight_value,
                            solver=solver,
                            penalty=penalty,
                            l1_ratio=l1_ratio,
                            random_state=42
                        )
                        
                elif model_type == "SVM":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                    
                    # Parameter lanjutan dalam expander
                    with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                        coef0 = st.slider("Coefficient for polynomial kernel (coef0):" if st.session_state.language == 'id' else "Koefisien untuk kernel polinomial (coef0):", 0.0, 1.0, 0.0)
                        shrinking = st.checkbox("Use shrinking heuristic:" if st.session_state.language == 'id' else "Gunakan heuristik shrinking:", value=True)
                        probability = st.checkbox("Enable probability estimates:" if st.session_state.language == 'id' else "Aktifkan estimasi probabilitas:", value=True)
                        
                        # Parameter degree hanya untuk kernel poly
                        degree = 3  # default
                        if kernel == "poly":
                            degree = st.slider("Degree for polynomial kernel:" if st.session_state.language == 'id' else "Derajat untuk kernel polinomial:", 2, 10, 3)
                    
                    base_model = SVC(probability=probability, random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                            'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                            'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'C': [0.01, 0.1, 1, 10, 100],
                            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                            'degree': [2, 3, 4, 5]  # untuk kernel poly
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "SVM", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = SVC(**best_params, probability=True, random_state=42)
                    else:
                        model = SVC(
                            C=C,
                            kernel=kernel,
                            gamma=gamma,
                            coef0=coef0,
                            shrinking=shrinking,
                            probability=probability,
                            degree=degree if kernel == "poly" else 3,
                            random_state=42
                        )
                        
                elif model_type == "KNN":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_neighbors = st.slider("Number of neighbors (K):" if st.session_state.language == 'id' else "Jumlah tetangga (K):", 1, 20, 5)
                    weights = st.selectbox("Weight function:" if st.session_state.language == 'id' else "Fungsi bobot:", ["uniform", "distance"])
                    algorithm = st.selectbox("Algorithm:" if st.session_state.language == 'id' else "Algoritma:", ["auto", "ball_tree", "kd_tree", "brute"])
                    
                    # Parameter lanjutan dalam expander
                    with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                        metric = st.selectbox("Distance metric:" if st.session_state.language == 'id' else "Metrik jarak:", ["minkowski", "euclidean", "manhattan", "chebyshev", "wminkowski", "seuclidean", "mahalanobis"], index=0)
                        p_value = st.slider("Power parameter for Minkowski metric:" if st.session_state.language == 'id' else "Parameter daya untuk metrik Minkowski:", 1, 5, 2)
                        leaf_size = st.slider("Leaf size:" if st.session_state.language == 'id' else "Ukuran daun:", 10, 50, 30)
                        
                        # Metric parameters untuk metrik tertentu
                        metric_params = None
                        if metric == "wminkowski":
                            w = st.text_input("Weight vector for wminkowski (comma-separated):" if st.session_state.language == 'id' else "Vektor bobot untuk wminkowski (pisahkan koma):", "1,1,1")
                            try:
                                metric_params = {'w': [float(x.strip()) for x in w.split(",")]}
                            except:
                                metric_params = None
                        elif metric == "mahalanobis":
                            VI = st.text_input("Inverse covariance matrix for mahalanobis (optional):" if st.session_state.language == 'id' else "Matriks kovarian terbalik untuk mahalanobis (opsional):", "")
                            if VI:
                                try:
                                    # Parse matrix dari string (sederhana)
                                    metric_params = {'VI': np.eye(10)}  # Default identity matrix
                                except:
                                    metric_params = None
                    
                    base_model = KNeighborsClassifier()
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2]  # Manhattan or Euclidean distance
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_neighbors': list(range(3, 21)),
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2, 3, 4, 5]
                        }
                        # Gabungkan dengan custom parameter ranges
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=20, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "KNN", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = KNeighborsClassifier(**best_params)
                    else:
                        model = KNeighborsClassifier(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            metric=metric,
                            p=p_value,
                            leaf_size=leaf_size,
                            metric_params=metric_params
                        )
                        
                elif model_type == "Decision Tree":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    max_depth = st.slider("Maximum depth:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 50, 10)
                    min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel untuk membagi:", 2, 20, 2)
                    criterion = st.selectbox("Split criterion:" if st.session_state.language == 'id' else "Kriteria membagi:", ["gini", "entropy"])
                    
                    # Parameter lanjutan untuk Decision Tree
                    with st.expander("Parameter Lanjutan Decision Tree" if st.session_state.language == 'id' else "Advanced Decision Tree Parameters"):
                        min_samples_leaf = st.slider("Minimum samples per leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum per leaf:", 1, 20, 1)
                        max_features_options = ["None", "sqrt", "log2", "auto"]
                        max_features = st.selectbox("Max features for split:" if st.session_state.language == 'id' else "Fitur maksimum untuk pembelahan:", max_features_options)
                        max_features_value = None if max_features == "None" else max_features
                        
                        class_weight_options = ["None", "balanced"]
                        class_weight = st.selectbox("Class weight:" if st.session_state.language == 'id' else "Bobot kelas:", class_weight_options)
                        class_weight_value = None if class_weight == "None" else class_weight
                        
                        ccp_alpha = st.slider("Cost complexity pruning alpha:" if st.session_state.language == 'id' else "Alpha pemangkasan kompleksitas biaya:", 0.0, 0.1, 0.0, 0.001)
                    
                    base_model = DecisionTreeClassifier(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4],
                            'criterion': ['gini', 'entropy']
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'max_depth': list(range(1, 21)) + [None],
                            'min_samples_split': [2, 5, 10, 15, 20, 25, 30],
                            'min_samples_leaf': [1, 2, 4, 8, 16, 32],
                            'criterion': ['gini', 'entropy', 'log_loss']
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "Decision Tree", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = DecisionTreeClassifier(**best_params, random_state=42)
                    else:
                        model = DecisionTreeClassifier(
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            criterion=criterion,
                            max_features=max_features_value,
                            class_weight=class_weight_value,
                            ccp_alpha=ccp_alpha,
                            random_state=42
                        )
                        
                elif model_type == "Naive Bayes":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    var_smoothing = st.slider("Variance smoothing:" if st.session_state.language == 'id' else "Penyesuaian varian:", 1e-10, 1e-8, 1e-9, format="%.1e")
                    
                    base_model = GaussianNB()
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'var_smoothing': [1e-10, 1e-9, 1e-8]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'var_smoothing': [1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=10, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "Naive Bayes", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = GaussianNB(**best_params)
                    else:
                        model = GaussianNB(
                            var_smoothing=var_smoothing
                        )
                        
                elif model_type == "Gradient Boosting":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Number of boosting stages:" if st.session_state.language == 'id' else "Jumlah boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                    
                    # Parameter lanjutan untuk Gradient Boosting
                    with st.expander("Parameter Lanjutan Gradient Boosting" if st.session_state.language == 'id' else "Advanced Gradient Boosting Parameters"):
                        subsample = st.slider("Subsample ratio:" if st.session_state.language == 'id' else "Rasio subsample:", 0.5, 1.0, 1.0, 0.1)
                        min_samples_split = st.slider("Minimum samples to split:" if st.session_state.language == 'id' else "Jumlah sampel minimum untuk membagi:", 2, 20, 2)
                        min_samples_leaf = st.slider("Minimum samples per leaf:" if st.session_state.language == 'id' else "Jumlah sampel minimum per leaf:", 1, 20, 1)
                        max_features_options = ["None", "sqrt", "log2", "auto"]
                        max_features = st.selectbox("Max features for split:" if st.session_state.language == 'id' else "Fitur maksimum untuk pembelahan:", max_features_options)
                        max_features_value = None if max_features == "None" else max_features
                    
                    base_model = GradientBoostingClassifier(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [3, 6, 9] if max_depth == 3 else [max(1, max_depth-3), max_depth, min(10, max_depth+3)],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': list(range(3, 16)),
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'min_samples_split': [2, 5, 10, 15, 20],
                            'min_samples_leaf': [1, 2, 4, 8, 16]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "Gradient Boosting", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = GradientBoostingClassifier(**best_params, random_state=42)
                    else:
                        model = GradientBoostingClassifier(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            subsample=subsample,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features_value,
                            random_state=42
                        )
                        
                elif model_type == "MLP (Neural Network)":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
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
                    
                    # Mode konfigurasi parameter
                    config_mode = st.radio(
                        "Mode Konfigurasi Parameter:" if st.session_state.language == 'id' else "Parameter Configuration Mode:",
                        ["Quick Setup", "Advanced Settings"],
                        horizontal=True,
                        help="Quick Setup: Parameter dasar | Advanced Settings: Kontrol penuh semua parameter"
                    )
                    
                    if config_mode == "Quick Setup":
                        # Architecture Configuration - Quick Setup
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
                    
                    # Parameter validation
                    validation_errors = []
                    
                    # Validate hidden layer sizes
                    if any(size <= 0 for size in hidden_layer_sizes):
                        validation_errors.append("Hidden layer sizes must be positive integers")
                    
                    # Validate learning rate
                    if learning_rate_init <= 0 or learning_rate_init > 1:
                        validation_errors.append("Learning rate must be between 0 and 1")
                    
                    # Validate regularization
                    if alpha < 0:
                        validation_errors.append("Alpha (regularization) must be non-negative")
                    
                    # Validate max iterations
                    if max_iter <= 0:
                        validation_errors.append("Max iterations must be positive")
                    
                    # Validate tolerance
                    if tol <= 0:
                        validation_errors.append("Tolerance must be positive")
                    
                    # Validate batch size
                    if isinstance(actual_batch_size, int) and actual_batch_size <= 0:
                        validation_errors.append("Batch size must be positive")
                    
                    # Validate solver-specific parameters
                    if solver == "adam":
                        if not (0 < beta_1 < 1):
                            validation_errors.append("Beta 1 must be between 0 and 1")
                        if not (0 < beta_2 < 1):
                            validation_errors.append("Beta 2 must be between 0 and 1")
                        if epsilon <= 0:
                            validation_errors.append("Epsilon must be positive")
                    elif solver == "sgd":
                        if not (0 <= momentum <= 1):
                            validation_errors.append("Momentum must be between 0 and 1")
                        if power_t <= 0:
                            validation_errors.append("Power t must be positive")
                    
                    # Validate early stopping parameters
                    if early_stopping:
                        if not (0 < validation_fraction < 1):
                            validation_errors.append("Validation fraction must be between 0 and 1")
                        if n_iter_no_change <= 0:
                            validation_errors.append("Iterations no change must be positive")
                    
                    # Show validation errors if any
                    if validation_errors:
                        st.error("**Parameter Validation Errors:**")
                        for error in validation_errors:
                            st.error(f"• {error}")
                        st.stop()
                    
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
                    
                    # Add early stopping parameters if enabled
                    if early_stopping:
                        mlp_params.update({
                            'early_stopping': early_stopping,
                            'validation_fraction': validation_fraction,
                            'n_iter_no_change': n_iter_no_change
                        })
                    
                    # Add shuffle parameter
                    mlp_params['shuffle'] = shuffle
                    
                    base_model = MLPClassifier()
                    
                    if optimization_method == "GridSearchCV":
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
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'hidden_layer_sizes': [(50,), (100,), (150,), (200,), (100,50), (150,100), (200,100), (100,50,25), (200,100,50)],
                            'activation': ['relu', 'tanh', 'logistic', 'identity'],
                            'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                            'max_iter': [200, 500, 1000, 1500]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='accuracy', n_jobs=-1, n_iter=30, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Classification", "MLP", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=30, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = MLPClassifier(**best_params)
                    else:
                        model = MLPClassifier(**mlp_params)

            else:  # Regression
                # Regular regression models (non-time series)
                model_type = st.selectbox("Pilih model regresi:" if st.session_state.language == 'id' else "Select a regression model:", 
                                         ["Random Forest", "Linear Regression", "Gradient Boosting", "SVR", "Bagging Regressor", "Voting Regressor", "Stacking Regressor", "KNN Regressor", "MLP Regressor"])
                
                if model_type == "Random Forest":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Jumlah pepohonan:" if st.session_state.language == 'id' else "Number of Trees:", 10, 500, 100)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Maximum depth:", 1, 50, 10)
                    
                    # Parameter lanjutan dengan expander
                    with st.expander("Parameter Lanjutan" if st.session_state.language == 'id' else "Advanced Parameters"):
                        min_samples_split = st.slider("Jumlah sampel minimum untuk membagi:" if st.session_state.language == 'id' else "Minimum samples to split:", 2, 20, 2)
                        min_samples_leaf = st.slider("Jumlah sampel minimum di leaf:" if st.session_state.language == 'id' else "Minimum samples in leaf:", 1, 10, 1)
                        max_features = st.selectbox("Fitur maksimum:" if st.session_state.language == 'id' else "Max features:", ["sqrt", "log2", "None"], index=0)
                        bootstrap = st.checkbox("Bootstrap sampel:" if st.session_state.language == 'id' else "Bootstrap samples:", value=True)
                        
                        # Konversi max_features dari string ke None jika diperlukan
                        max_features_value = None if max_features == "None" else max_features
                    
                    base_model = RandomForestRegressor(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'max_depth': [5, 10, 15] if max_depth == 10 else [max(1, max_depth-5), max_depth, min(50, max_depth+5)],
                            'min_samples_split': [2, 5, 10],
                            'min_samples_leaf': [1, 2, 4]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = GridSearchCV(base_model, param_grid, cv=cv_value, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'max_depth': list(range(3, 21)) + [None],
                            'min_samples_split': [2, 5, 10, 15, 20],
                            'min_samples_leaf': [1, 2, 4, 8, 16],
                            'max_features': ['sqrt', 'log2', None]
                        }
                        cv_value = cv_params['cv'] if cv_params['cv'] is not None else 5
                        model = RandomizedSearchCV(base_model, param_dist, cv=cv_value, scoring='r2', n_jobs=-1, n_iter=50, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "Random Forest", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=50, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = RandomForestRegressor(**best_params, random_state=42)
                    else:
                        model = RandomForestRegressor(
                            n_estimators=n_estimators,
                            max_depth=max_depth,
                            min_samples_split=min_samples_split,
                            min_samples_leaf=min_samples_leaf,
                            max_features=max_features_value,
                            bootstrap=bootstrap,
                            random_state=42
                        )
                        
                elif model_type == "Gradient Boosting":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    n_estimators = st.slider("Jumlah boosting stages:" if st.session_state.language == 'id' else "Number of boosting stages:", 10, 500, 100)
                    learning_rate = st.slider("Learning rate:" if st.session_state.language == 'id' else "Learning rate:", 0.01, 0.3, 0.1)
                    max_depth = st.slider("Kedalaman maksimum:" if st.session_state.language == 'id' else "Kedalaman maksimum:", 1, 10, 3)
                    
                    base_model = GradientBoostingRegressor(random_state=42)
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_estimators': [50, 100, 200] if n_estimators == 100 else [max(10, n_estimators-50), n_estimators, min(500, n_estimators+50)],
                            'learning_rate': [0.01, 0.1, 0.2] if learning_rate == 0.1 else [max(0.01, learning_rate/2), learning_rate, min(0.3, learning_rate*2)],
                            'max_depth': [2, 3, 5] if max_depth == 3 else [max(1, max_depth-1), max_depth, min(10, max_depth+2)],
                            'subsample': [0.8, 0.9, 1.0]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_estimators': list(range(50, 301, 25)),
                            'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
                            'max_depth': list(range(2, 16)),
                            'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
                            'min_samples_split': [2, 5, 10, 15, 20],
                            'min_samples_leaf': [1, 2, 4, 8, 16]
                        }
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=40, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "Gradient Boosting", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=40, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = GradientBoostingRegressor(**best_params, random_state=42)
                    else:
                        model = GradientBoostingRegressor(
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            max_depth=max_depth,
                            random_state=42
                        )
                        
                elif model_type == "Linear Regression":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    fit_intercept = st.checkbox("Fit intercept" if st.session_state.language == 'id' else "Fit intercept", value=True)
                    
                    # Parameter lanjutan untuk Linear Regression
                    with st.expander("Parameter Lanjutan Linear Regression" if st.session_state.language == 'id' else "Advanced Linear Regression Parameters"):
                        positive = st.checkbox("Force positive coefficients" if st.session_state.language == 'id' else "Paksa koefisien positif", value=False)
                        copy_X = st.checkbox("Copy X" if st.session_state.language == 'id' else "Salin X", value=True)
                    
                    base_model = LinearRegression()
                    
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'fit_intercept': [True, False],
                            'positive': [True, False],
                            'copy_X': [True, False]
                        }
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'fit_intercept': [True, False],
                            'positive': [True, False],
                            'copy_X': [True, False]
                        }
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=8, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "Linear Regression", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=8, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = LinearRegression(**best_params)
                    else:
                        model = LinearRegression(
                            fit_intercept=fit_intercept,
                            positive=positive,
                            copy_X=copy_X
                        )
                        
                elif model_type == "SVR":
                    # Dapatkan rentang parameter kustom jika diaktifkan
                    custom_param_ranges = get_custom_param_inputs(model_type, use_custom_ranges, st.session_state)
                    
                    C = st.slider("Regularization parameter (C):" if st.session_state.language == 'id' else "Parameter regulerisasi (C):", 0.1, 10.0, 1.0)
                    kernel = st.selectbox("Kernel:" if st.session_state.language == 'id' else "Kernel:", ["linear", "poly", "rbf", "sigmoid"])
                    gamma = st.selectbox("Gamma (kernel coefficient):" if st.session_state.language == 'id' else "Gamma (koefisien kernel):", ["scale", "auto"])
                    epsilon = st.slider("Epsilon:" if st.session_state.language == 'id' else "Epsilon:", 0.01, 0.5, 0.1)

                    base_model = SVR()

                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'C': [0.1, 1.0, 10.0] if C == 1.0 else [max(0.1, C/2), C, min(10.0, C*2)],
                            'kernel': [kernel] if kernel != "rbf" else ['linear', 'rbf'],
                            'gamma': [gamma] if gamma != "scale" else ['scale', 'auto'],
                            'epsilon': [epsilon]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'C': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0],
                            'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0],
                            'epsilon': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
                            'degree': [2, 3, 4, 5]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=25, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "SVR", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=25, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = SVR(**best_params)
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
                    custom_param_ranges = get_custom_param_inputs("KNN Regressor", st.session_state.X_train)
                    if optimization_method == "GridSearchCV":
                        param_grid = {
                            'n_neighbors': [3, 5, 7] if n_neighbors == 5 else [max(1, n_neighbors-2), n_neighbors, min(20, n_neighbors+2)],
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2]  # Manhattan or Euclidean distance
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'n_neighbors': list(range(1, 31)),
                            'weights': ['uniform', 'distance'],
                            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                            'p': [1, 2, 3],  # Manhattan, Euclidean, or Minkowski distance
                            'leaf_size': list(range(10, 51, 5))
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=20, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "KNN", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        model = KNeighborsRegressor(**best_params)
                    else:
                        model = KNeighborsRegressor(
                            n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm
                        )
                elif model_type == "MLP Regressor":
                    custom_param_ranges = get_custom_param_inputs("MLP Regressor", st.session_state.X_train)
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
                    
                    # Parameter configuration mode
                    param_config_mode = st.radio(
                        "Mode konfigurasi parameter:" if st.session_state.language == 'id' else "Parameter configuration mode:",
                        ["Quick Setup", "Advanced Settings"],
                        horizontal=True,
                        help="Pilih mode konfigurasi: Quick Setup untuk pengaturan cepat, Advanced Settings untuk kontrol penuh"
                    )
                    
                    # Architecture Configuration
                    st.write("**Arsitektur Jaringan Regresi:**" if st.session_state.language == 'id' else "**Regression Network Architecture:**")
                    
                    if param_config_mode == "Quick Setup":
                        # Quick Setup - simplified interface
                        col1, col2 = st.columns(2)
                        with col1:
                            num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 5, 2)
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
                        with col2:
                            activation = st.selectbox("Activation function:", 
                                                    ["relu", "tanh", "logistic", "identity"],
                                                    help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                        
                        # Basic parameters
                        col3, col4 = st.columns(2)
                        with col3:
                            solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"])
                            learning_rate_init = st.slider("Initial learning rate:", 0.0001, 0.1, 0.001, format="%.4f")
                        with col4:
                            max_iter = st.slider("Maximum iterations:", 100, 2000, 200)
                            alpha = st.slider("L2 regularization (alpha):", 0.00001, 0.01, 0.0001, format="%.5f")
                        
                        # Set default values for other parameters
                        learning_rate = "constant"
                        batch_size = "auto"
                        actual_batch_size = min(200, len(st.session_state.X_train))
                        tol = 1e-4
                        beta_1 = 0.9
                        beta_2 = 0.999
                        epsilon = 1e-8
                        momentum = 0.9
                        power_t = 0.5
                        shuffle = True
                        early_stopping = False
                        validation_fraction = 0.1
                        n_iter_no_change = 10
                        
                    else:  # Advanced Settings
                        # Advanced Settings - full control
                        st.write("**Arsitektur Jaringan Regresi - Advanced:**" if st.session_state.language == 'id' else "**Regression Network Architecture - Advanced:**")
                        
                        # Hidden layers configuration
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            num_hidden_layers = st.slider("Jumlah hidden layers:", 1, 10, 2, help="Jumlah lapisan tersembunyi dalam jaringan")
                        with col2:
                            neurons_per_layer = st.text_input("Neurons per layer:", "100,50", help="Jumlah neuron di setiap lapisan, pisahkan dengan koma")
                            try:
                                neurons_list = [int(x.strip()) for x in neurons_per_layer.split(",")]
                                if len(neurons_list) < num_hidden_layers:
                                    neurons_list.extend([neurons_list[-1]] * (num_hidden_layers - len(neurons_list)))
                                elif len(neurons_list) > num_hidden_layers:
                                    neurons_list = neurons_list[:num_hidden_layers]
                                hidden_layer_sizes = tuple(neurons_list)
                            except:
                                hidden_layer_sizes = (100, 50)
                                st.error("Format neurons per layer tidak valid. Gunakan format: 100,50,25")
                        with col3:
                            activation = st.selectbox("Activation function:", 
                                                    ["relu", "tanh", "logistic", "identity"],
                                                    help="ReLU: max(0,x) | Sigmoid: 1/(1+e^-x) | Tanh: (e^x-e^-x)/(e^x+e^-x) | Identity: x")
                        
                        # Advanced parameters
                        with st.expander("🔧 Advanced Parameters", expanded=True):
                            col4, col5 = st.columns(2)
                            with col4:
                                solver = st.selectbox("Optimizer:", ["adam", "sgd", "lbfgs"], 
                                                    help="Algoritma optimasi untuk training")
                                
                                if solver == "adam":
                                    st.write("**Adam Optimizer Parameters:**")
                                    beta_1 = st.slider("Beta 1 (exponential decay rate):", 0.8, 0.999, 0.9, format="%.3f",
                                                       help="Faktor decay untuk estimasi pertama (default: 0.9)")
                                    beta_2 = st.slider("Beta 2 (exponential decay rate):", 0.9, 0.9999, 0.999, format="%.4f",
                                                       help="Faktor decay untuk estimasi kedua (default: 0.999)")
                                    epsilon = st.slider("Epsilon (numerical stability):", 1e-8, 1e-3, 1e-8, format="%.1e",
                                                        help="Nilai kecil untuk menghindari division by zero")
                                elif solver == "sgd":
                                    st.write("**SGD Optimizer Parameters:**")
                                    momentum = st.slider("Momentum:", 0.0, 0.9, 0.9,
                                                         help="Faktor momentum untuk mempercepat konvergensi")
                                    power_t = st.slider("Power t (inverse scaling exponent):", 0.1, 0.9, 0.5,
                                                        help="Eksponen untuk inverse scaling learning rate")
                                else:  # lbfgs
                                    st.info("L-BFGS tidak memiliki parameter tambahan")
                                    beta_1 = 0.9
                                    beta_2 = 0.999
                                    epsilon = 1e-8
                                    momentum = 0.9
                                    power_t = 0.5
                                    
                            with col5:
                                learning_rate_init = st.slider("Initial learning rate:", 0.00001, 0.1, 0.001, format="%.5f",
                                                               help="Learning rate awal untuk optimasi")
                                learning_rate = st.selectbox("Learning rate schedule:", 
                                                           ["constant", "invscaling", "adaptive"],
                                                           help="Strategi penyesuaian learning rate selama training")
                                
                                col6, col7 = st.columns(2)
                                with col6:
                                    alpha = st.slider("L2 regularization (alpha):", 0.000001, 0.1, 0.0001, format="%.6f",
                                                      help="Regularisasi L2 untuk mencegah overfitting")
                                    batch_size_options = ["auto", 8, 16, 32, 64, 128, 256, 512, 1024]
                                    batch_size = st.selectbox("Batch size:", batch_size_options,
                                                            help="Jumlah sampel per update weight")
                                    if batch_size == "auto":
                                        actual_batch_size = min(200, len(st.session_state.X_train))
                                    else:
                                        actual_batch_size = batch_size
                                        
                                with col7:
                                    max_iter = st.slider("Maximum iterations:", 100, 5000, 200,
                                                         help="Maksimum iterasi training")
                                    tol = st.slider("Tolerance (convergence threshold):", 1e-6, 1e-2, 1e-4, format="%.1e",
                                                    help="Threshold untuk menghentikan training")
                                    
                                # Additional advanced parameters
                                st.write("**Additional Parameters:**")
                                col8, col9 = st.columns(2)
                                with col8:
                                    shuffle = st.checkbox("Shuffle samples", value=True,
                                                        help="Mengacak sampel di setiap iterasi")
                                    early_stopping = st.checkbox("Early stopping", value=False,
                                                               help="Menghentikan training jika validasi tidak membaik")
                                with col9:
                                    validation_fraction = st.slider("Validation fraction:", 0.05, 0.5, 0.1, format="%.2f",
                                                                  help="Fraksi data untuk validasi (jika early stopping=True)")
                                    n_iter_no_change = st.slider("Iterations no change:", 5, 50, 10,
                                                                 help="Jumlah iterasi tanpa perbaikan sebelum stopping")
                    
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
                    
                    if optimization_method == "GridSearchCV":
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
                        # Gabungkan dengan rentang parameter kustom
                        param_grid = merge_custom_param_ranges(param_grid, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_grid = validate_param_ranges(param_grid, st.session_state.X_train, model_type)
                        model = GridSearchCV(base_model, param_grid, cv=5, scoring='r2', n_jobs=-1)
                    elif optimization_method == "RandomizedSearchCV":
                        param_dist = {
                            'hidden_layer_sizes': [
                                (50,), (100,), (150,), (200,), (250,),
                                (50, 50), (100, 50), (100, 100), (150, 100),
                                (100, 50, 25), (150, 100, 50)
                            ],
                            'activation': ['relu', 'tanh', 'logistic', 'identity'],
                            'solver': ['adam', 'sgd', 'lbfgs'],
                            'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                            'learning_rate_init': [0.0001, 0.001, 0.01, 0.1],
                            'max_iter': [200, 500, 1000, 1500]
                        }
                        # Gabungkan dengan rentang parameter kustom
                        param_dist = merge_custom_param_ranges(param_dist, custom_param_ranges)
                        # Validasi parameter berdasarkan data
                        param_dist = validate_param_ranges(param_dist, st.session_state.X_train, model_type)
                        model = RandomizedSearchCV(base_model, param_dist, cv=5, scoring='r2', n_jobs=-1, n_iter=20, random_state=42)
                    elif optimization_method == "Bayesian Optimization (Optuna)" and OPTUNA_AVAILABLE:
                        objective = create_optuna_study("Regression", "MLP", st.session_state.X_train, st.session_state.y_train, cv_params, custom_param_ranges)
                        study = optuna.create_study(direction='maximize')
                        study.optimize(objective, n_trials=20, n_jobs=-1, show_progress_bar=True)
                        best_params = study.best_params
                        # Convert hidden_layer_sizes back to tuple
                        if 'hidden_layer_sizes' in best_params:
                            best_params['hidden_layer_sizes'] = tuple(best_params['hidden_layer_sizes'])
                        model = MLPRegressor(**best_params)
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
                        # Preprocessing: Remove NaN values from training data
                        # This is crucial for models like GradientBoostingRegressor that don't handle NaN natively
                        X_train_clean = st.session_state.X_train.dropna()
                        y_train_clean = st.session_state.y_train.loc[X_train_clean.index]
                        
                        # Check if we have enough data after cleaning
                        if len(X_train_clean) == 0:
                            raise ValueError("Tidak ada data yang valid untuk training setelah menghapus nilai NaN. Silakan periksa data Anda." if st.session_state.language == 'id' else "No valid data available for training after removing NaN values. Please check your data.")
                        
                        if len(X_train_clean) < 10:
                            st.warning(f"Hanya {len(X_train_clean)} sampel tersedia untuk training. Hasil mungkin tidak optimal." if st.session_state.language == 'id' else f"Only {len(X_train_clean)} samples available for training. Results may not be optimal.")
                        
                        start_time = time.time()
                        model.fit(X_train_clean, y_train_clean)
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

                        # Jika menggunakan optimasi hyperparameter, tampilkan parameter terbaik
                        if optimization_method != "None" and hasattr(model, "best_params_"):
                            st.success(f"Pelatihan model selesai dalam {training_time:.2f} detik dengan {optimization_method}. Parameter terbaik: {model.best_params_}" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds with {optimization_method}!")
                            st.subheader("Parameter Terbaik" if st.session_state.language == 'id' else "Best Parameters:")
                            st.write(model.best_params_)
                            st.write(f"Skor terbaik (CV): {model.best_score_:.4f}" if st.session_state.language == 'id' else f"Best Score (CV): {model.best_score_:.4f}")

                            # Gunakan model terbaik untuk prediksi (dengan handling NaN)
                            X_test_clean = st.session_state.X_test.dropna()
                            y_test_clean = st.session_state.y_test.loc[X_test_clean.index]
                            y_pred = model.best_estimator_.predict(X_test_clean)
                            st.session_state.model = model.best_estimator_
                            # Update y_test untuk evaluasi
                            st.session_state.y_test_eval = y_test_clean
                        else:
                            st.success(f"Model selesai dilatih dalam {training_time:.2f} detik" if st.session_state.language == 'id' else f"Model training completed in {training_time:.2f} seconds!")
                            # Gunakan model terbaik untuk prediksi (dengan handling NaN)
                            X_test_clean = st.session_state.X_test.dropna()
                            y_test_clean = st.session_state.y_test.loc[X_test_clean.index]
                            y_pred = model.predict(X_test_clean)
                            st.session_state.model = model
                            # Update y_test untuk evaluasi
                            st.session_state.y_test_eval = y_test_clean
                        
                        # Cross-validation evaluation
                        if cv_params['cv'] is not None:
                            st.subheader("Hasil Cross-Validation" if st.session_state.language == 'id' else "Cross-Validation Results")
                            
                            with st.spinner("Menghitung cross-validation..." if st.session_state.language == 'id' else "Calculating cross-validation..."):
                                try:
                                    # Get the actual model (best estimator if using optimization)
                                    eval_model = model.best_estimator_ if optimization_method != "None" else model
                                    
                                    # Perform cross-validation using cleaned data (without NaN values)
                                    cv_scores = cross_val_score(
                                        eval_model, 
                                        X_train_clean, 
                                        y_train_clean,
                                        cv=cv_params['cv'],
                                        scoring=cv_params['scoring'],
                                        n_jobs=-1
                                    )
                                    
                                    # Display results
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Rata-rata Skor CV" if st.session_state.language == 'id' else "Mean CV Score",
                                            f"{cv_scores.mean():.4f}"
                                        )
                                    with col2:
                                        st.metric(
                                            "Standar Deviasi" if st.session_state.language == 'id' else "Std Deviation",
                                            f"{cv_scores.std():.4f}"
                                        )
                                    with col3:
                                        st.metric(
                                            "Metode Validasi" if st.session_state.language == 'id' else "Validation Method",
                                            cv_params['name']
                                        )
                                    
                                    # Plot cross-validation scores
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.boxplot(cv_scores)
                                    ax.set_title(f"Cross-Validation Scores - {cv_params['name']}" if st.session_state.language == 'id' else f"Cross-Validation Scores - {cv_params['name']}")
                                    ax.set_ylabel("Score")
                                    ax.grid(True, alpha=0.3)
                                    st.pyplot(fig)
                                    
                                    # Detailed scores
                                    st.write("**Detail Skor per Fold:**" if st.session_state.language == 'id' else "**Detailed Scores per Fold:**")
                                    fold_df = pd.DataFrame({
                                        'Fold': [f'Fold {i+1}' for i in range(len(cv_scores))],
                                        'Score': cv_scores
                                    })
                                    st.dataframe(fold_df)
                                    
                                except Exception as e:
                                    st.error(f"Error dalam cross-validation: {str(e)}" if st.session_state.language == 'id' else f"Error in cross-validation: {str(e)}")
                        
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
                            accuracy = accuracy_score(st.session_state.y_test_eval, y_pred)
                            st.write(f"Accuracy: {accuracy:.4f}")
                            
                            # Confusion Matrix
                            cm = confusion_matrix(st.session_state.y_test_eval, y_pred)
                            fig, ax = plt.subplots(figsize=(10, 8))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                            plt.title('Confusion Matrix')
                            plt.ylabel('True Label')
                            plt.xlabel('Predicted Label')
                            st.pyplot(fig)
                            
                            # Classification Report
                            report = classification_report(st.session_state.y_test_eval, y_pred, output_dict=True)
                            report_df = pd.DataFrame(report).transpose()
                            st.write("Label Report" if st.session_state.language == 'id' else "Classification Report:")
                            st.dataframe(report_df)
                            
                            # ROC Curve dan AUC Score
                            st.subheader("ROC Curve dan AUC Score" if st.session_state.language == 'id' else "ROC Curve and AUC Score")
                            
                            # Cek apakah model mendukung predict_proba
                            if hasattr(model, 'predict_proba'):
                                # Untuk klasifikasi biner
                                if len(np.unique(st.session_state.y_test_eval)) == 2:
                                    y_prob = model.predict_proba(X_test_clean)[:, 1]
                                    # Menangani kasus ketika y_test berisi nilai kategorikal seperti '<20', '>20'
                                    if isinstance(st.session_state.y_test_eval.iloc[0], str):
                                        # Konversi nilai kategorikal ke numerik (0 dan 1)
                                        unique_values = sorted(np.unique(st.session_state.y_test_eval))
                                        pos_label = unique_values[1]  # Nilai kedua sebagai pos_label
                                        fpr, tpr, thresholds = roc_curve(st.session_state.y_test_eval, y_prob, pos_label=pos_label)
                                    else:
                                        fpr, tpr, thresholds = roc_curve(st.session_state.y_test_eval, y_prob)
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
                                        y_prob = model.predict_proba(X_test_clean)
                                        
                                        # Buat label biner untuk setiap kelas
                                        y_test_bin = pd.get_dummies(st.session_state.y_test_eval).values
                                        
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
                                        st.write(f"Jumlah kelas unik dalam y_test: {len(np.unique(st.session_state.y_test_eval))}")
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
                            mse = mean_squared_error(st.session_state.y_test_eval, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(st.session_state.y_test_eval, y_pred)
                            # Tambahan: Adjusted R²
                            n = X_test_clean.shape[0]
                            k = X_test_clean.shape[1]
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
                                bp_result = breusch_pagan_test(st.session_state.y_test_eval, y_pred, X_test_clean)
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
                            ax.scatter(st.session_state.y_test_eval, y_pred, alpha=0.5)
                            ax.plot([st.session_state.y_test_eval.min(), st.session_state.y_test_eval.max()], 
                                   [st.session_state.y_test_eval.min(), st.session_state.y_test_eval.max()], 
                                   'r--')
                            plt.title('Actual vs Predicted')
                            plt.xlabel('Actual')
                            plt.ylabel('Predicted')
                            st.pyplot(fig)
                            
                            # Residual plot
                            residuals = st.session_state.y_test_eval - y_pred
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(y_pred, residuals, alpha=0.5)
                            ax.axhline(y=0, color='r', linestyle='--')
                            plt.title('Residual Plot')
                            plt.xlabel('Predicted')
                            plt.ylabel('Residuals')
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.error(f"Error saat evaluasi model: {str(e)}" if st.session_state.language == 'id' else f"Error during model training: {str(e)}")
                    
                    # Simpan hasil evaluasi untuk perbandingan
                    if 'model_results' not in st.session_state:
                        st.session_state.model_results = []
                    
                    model_name = type(st.session_state.model).__name__
                    
                    # Pastikan y_pred didefinisikan berdasarkan problem type
                    if problem_type == "Classification":
                        y_pred = st.session_state.model.predict(X_test_scaled if 'X_test_scaled' in locals() else X_test)
                    else:  # Regression
                        y_pred = st.session_state.model.predict(X_test_scaled if 'X_test_scaled' in locals() else X_test)
                    
                    result = {
                        'model_name': model_name,
                        'model': st.session_state.model,
                        'y_test': st.session_state.y_test,
                        'y_pred': y_pred,
                        'problem_type': problem_type
                    }
                    
                    if problem_type == "Classification":
                        result.update({
                            'accuracy': accuracy,
                            'confusion_matrix': cm,
                            'classification_report': report
                        })
                    else:  # Regression
                        result.update({
                            'mse': mse,
                            'rmse': rmse,
                            'r2': r2,
                            'adj_r2': adj_r2
                        })
                    
                    st.session_state.model_results.append(result)
            
            # Tampilkan perbandingan model jika ada lebih dari satu model
            if len(st.session_state.model_results) > 1:
                st.header("Perbandingan Model" if st.session_state.language == 'id' else "Model Comparison")
                
                # Buat tabs untuk berbagai jenis perbandingan
                comparison_tabs = st.tabs([
                    "Confusion Matrix Comparison" if st.session_state.language == 'id' else "Confusion Matrix Comparison",
                    "Performance Metrics" if st.session_state.language == 'id' else "Performance Metrics",
                    "Model Rankings" if st.session_state.language == 'id' else "Model Rankings"
                ])
                
                with comparison_tabs[0]:
                    if problem_type == "Classification":
                        st.subheader("Perbandingan Confusion Matrix" if st.session_state.language == 'id' else "Confusion Matrix Comparison")
                        
                        # Hitung jumlah model dan baris/kolom yang dibutuhkan
                        n_models = len(st.session_state.model_results)
                        n_cols = min(3, n_models)  # Maksimal 3 kolom per baris
                        n_rows = (n_models + n_cols - 1) // n_cols
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                        axes = axes.flatten() if n_models > 1 else [axes]
                        
                        for idx, result in enumerate(st.session_state.model_results):
                            if result['problem_type'] == "Classification":
                                cm = result['confusion_matrix']
                                model_name = result['model_name']
                                
                                # Buat heatmap untuk confusion matrix
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                          ax=axes[idx], cbar=False)
                                axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}')
                                axes[idx].set_xlabel('Predicted')
                                axes[idx].set_ylabel('Actual')
                        
                        # Sembunyikan subplot kosong
                        for idx in range(n_models, len(axes)):
                            axes[idx].set_visible(False)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Tampilkan ringkasan performa
                        st.subheader("Ringkasan Performa Klasifikasi" if st.session_state.language == 'id' else "Classification Performance Summary")
                        comparison_df = pd.DataFrame([
                            {
                                'Model': r['model_name'],
                                'Accuracy': f"{r['accuracy']:.4f}",
                                'Precision': f"{r['classification_report']['weighted avg']['precision']:.4f}",
                                'Recall': f"{r['classification_report']['weighted avg']['recall']:.4f}",
                                'F1-Score': f"{r['classification_report']['weighted avg']['f1-score']:.4f}"
                            }
                            for r in st.session_state.model_results 
                            if r['problem_type'] == "Classification"
                        ])
                        st.dataframe(comparison_df)
                        
                        # Visualisasi perbandingan metrik
                        fig, ax = plt.subplots(figsize=(12, 6))
                        metrics_data = []
                        for r in st.session_state.model_results:
                            if r['problem_type'] == "Classification":
                                metrics_data.append({
                                    'Model': r['model_name'],
                                    'Accuracy': r['accuracy'],
                                    'Precision': r['classification_report']['weighted avg']['precision'],
                                    'Recall': r['classification_report']['weighted avg']['recall'],
                                    'F1-Score': r['classification_report']['weighted avg']['f1-score']
                                })
                        
                        if metrics_data:
                            metrics_df = pd.DataFrame(metrics_data)
                            metrics_df.set_index('Model').plot(kind='bar', ax=ax)
                            plt.title('Perbandingan Metrik Klasifikasi' if st.session_state.language == 'id' else 'Classification Metrics Comparison')
                            plt.ylabel('Score')
                            plt.xticks(rotation=45)
                            plt.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                    else:
                        st.info("Perbandingan confusion matrix hanya tersedia untuk masalah klasifikasi." if st.session_state.language == 'id' else "Confusion matrix comparison is only available for classification problems.")
                
                with comparison_tabs[1]:
                    st.subheader("Metrik Performa" if st.session_state.language == 'id' else "Performance Metrics")
                    
                    if problem_type == "Classification":
                        # Tampilkan semua metrik klasifikasi
                        metrics_summary = []
                        for result in st.session_state.model_results:
                            if result['problem_type'] == "Classification":
                                report = result['classification_report']
                                metrics_summary.append({
                                    'Model': result['model_name'],
                                    'Accuracy': result['accuracy'],
                                    'Macro Precision': report['macro avg']['precision'],
                                    'Macro Recall': report['macro avg']['recall'],
                                    'Macro F1-Score': report['macro avg']['f1-score'],
                                    'Weighted Precision': report['weighted avg']['precision'],
                                    'Weighted Recall': report['weighted avg']['recall'],
                                    'Weighted F1-Score': report['weighted avg']['f1-score']
                                })
                        
                        if metrics_summary:
                            metrics_df = pd.DataFrame(metrics_summary)
                            st.dataframe(metrics_df)
                            
                            # Heatmap perbandingan
                            fig, ax = plt.subplots(figsize=(10, 6))
                            comparison_matrix = metrics_df.set_index('Model').T
                            sns.heatmap(comparison_matrix, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax)
                            plt.title('Heatmap Perbandingan Metrik' if st.session_state.language == 'id' else 'Metrics Comparison Heatmap')
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    else:  # Regression
                        # Tampilkan metrik regresi
                        metrics_summary = []
                        for result in st.session_state.model_results:
                            if result['problem_type'] == "Regression":
                                metrics_summary.append({
                                    'Model': result['model_name'],
                                    'MSE': result['mse'],
                                    'RMSE': result['rmse'],
                                    'R²': result['r2'],
                                    'Adjusted R²': result['adj_r2']
                                })
                        
                        if metrics_summary:
                            metrics_df = pd.DataFrame(metrics_summary)
                            st.dataframe(metrics_df)
                            
                            # Visualisasi perbandingan
                            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                            
                            # Plot untuk error metrics (semakin rendah semakin baik)
                            error_df = metrics_df[['Model', 'RMSE', 'MSE']].set_index('Model')
                            error_df.plot(kind='bar', ax=ax1, color=['red', 'orange'])
                            ax1.set_title('Perbandingan Error Metrics' if st.session_state.language == 'id' else 'Error Metrics Comparison')
                            ax1.set_ylabel('Error Value')
                            ax1.legend(['RMSE', 'MSE'])
                            ax1.tick_params(axis='x', rotation=45)
                            
                            # Plot untuk R² metrics (semakin tinggi semakin baik)
                            r2_df = metrics_df[['Model', 'R²', 'Adjusted R²']].set_index('Model')
                            r2_df.plot(kind='bar', ax=ax2, color=['green', 'blue'])
                            ax2.set_title('Perbandingan R² Metrics' if st.session_state.language == 'id' else 'R² Metrics Comparison')
                            ax2.set_ylabel('Score')
                            ax2.legend(['R²', 'Adjusted R²'])
                            ax2.tick_params(axis='x', rotation=45)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                
                with comparison_tabs[2]:
                    st.subheader("Peringkat Model" if st.session_state.language == 'id' else "Model Rankings")
                    
                    if problem_type == "Classification":
                        # Ranking berdasarkan F1-Score
                        ranking_data = []
                        for result in st.session_state.model_results:
                            if result['problem_type'] == "Classification":
                                ranking_data.append({
                                    'Model': result['model_name'],
                                    'Accuracy': result['accuracy'],
                                    'F1-Score': result['classification_report']['weighted avg']['f1-score'],
                                    'Precision': result['classification_report']['weighted avg']['precision'],
                                    'Recall': result['classification_report']['weighted avg']['recall']
                                })
                        
                        if ranking_data:
                            ranking_df = pd.DataFrame(ranking_data)
                            ranking_df['Rank'] = ranking_df['F1-Score'].rank(ascending=False)
                            ranking_df = ranking_df.sort_values('Rank')
                            
                            st.write("**Ranking berdasarkan F1-Score (terbaik → terburuk):**" if st.session_state.language == 'id' else "**Ranking by F1-Score (best → worst):**")
                            st.dataframe(ranking_df[['Rank', 'Model', 'F1-Score', 'Accuracy', 'Precision', 'Recall']])
                            
                            # Visualisasi ranking
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=ranking_df, x='Model', y='F1-Score', ax=ax)
                            ax.set_title('Ranking Model berdasarkan F1-Score' if st.session_state.language == 'id' else 'Model Ranking by F1-Score')
                            ax.set_ylabel('F1-Score')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
                    
                    else:  # Regression
                        # Ranking berdasarkan R² (semakin tinggi semakin baik)
                        ranking_data = []
                        for result in st.session_state.model_results:
                            if result['problem_type'] == "Regression":
                                ranking_data.append({
                                    'Model': result['model_name'],
                                    'RMSE': result['rmse'],
                                    'R²': result['r2'],
                                    'Adjusted R²': result['adj_r2']
                                })
                        
                        if ranking_data:
                            ranking_df = pd.DataFrame(ranking_data)
                            ranking_df['Rank'] = ranking_df['R²'].rank(ascending=False)
                            ranking_df = ranking_df.sort_values('Rank')
                            
                            st.write("**Ranking berdasarkan R² Score (terbaik → terburuk):**" if st.session_state.language == 'id' else "**Ranking by R² Score (best → worst):**")
                            st.dataframe(ranking_df[['Rank', 'Model', 'R²', 'Adjusted R²', 'RMSE']])
                            
                            # Visualisasi ranking
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(data=ranking_df, x='Model', y='R²', ax=ax)
                            ax.set_title('Ranking Model berdasarkan R² Score' if st.session_state.language == 'id' else 'Model Ranking by R² Score')
                            ax.set_ylabel('R² Score')
                            plt.xticks(rotation=45)
                            plt.tight_layout()
                            st.pyplot(fig)
            
            # Tombol untuk reset hasil perbandingan
            if st.session_state.model_results:
                if st.button("Reset Hasil Perbandingan" if st.session_state.language == 'id' else "Reset Comparison Results"):
                    st.session_state.model_results = []
                    st.rerun() 

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
                        mse = mean_squared_error(st.session_state.y_test_eval, y_pred)
                        rmse = np.sqrt(mse)
                        r2 = r2_score(st.session_state.y_test_eval, y_pred)
                        
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
    st.info("✅ **Notifikasi:** Fitur interpretasi SHAP sekarang mendukung algoritma model **regresi** dan **klasifikasi**. Untuk model **forecasting**, analisis masih dalam pengembangan." if st.session_state.language == 'id' else "✅ **Notification:** SHAP interpretation now supports **regression** and **classification** algorithms. Analysis for **forecasting** models is still under development.")
    
            
    if st.session_state.problem_type == 'Forecasting':
        st.header("Interpretasi Model Forecasting dengan SHAP" if st.session_state.language == 'id' else "Forecasting Model Interpretation with SHAP")
        st.info("✅ **Notifikasi:** Fitur interpretasi SHAP sekarang mendukung model **forecasting** dengan pendekatan khusus. Gunakan tombol di bawah untuk mulai menganalisis model forecasting Anda." if st.session_state.language == 'id' else "✅ **Notification:** SHAP interpretation now supports **forecasting** models with a specialized approach. Use the button below to start analyzing your forecasting model.")
        
        if st.session_state.model is not None:
            st.write("""
            SHAP untuk model forecasting memerlukan pendekatan khusus karena struktur data deret waktu.
            Kami menggunakan fungsi interpretasi khusus untuk menangani karakteristik unik model forecasting.
            """ if st.session_state.language == 'id' else """
            SHAP for forecasting models requires a special approach due to the time series data structure.
            We use specialized interpretation functions to handle the unique characteristics of forecasting models.
            """)
            
            # Pilih fitur untuk analisis SHAP
            if hasattr(st.session_state, 'forecast_features') and st.session_state.forecast_features:
                feature_names = st.session_state.forecast_features
                selected_features = st.multiselect(
                    "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                    options=feature_names,
                    default=feature_names[:min(10, len(feature_names))]
                )
                
                # Jumlah sampel untuk analisis
                sample_size = st.slider(
                    "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                    min_value=10, max_value=min(100, len(st.session_state.X_test)), value=50
                )
                
                if st.button("Generate SHAP Values untuk Forecasting" if st.session_state.language == 'id' else "Generate SHAP Values for Forecasting"):
                    if not selected_features:
                        st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                    else:
                        with st.spinner("Menghitung nilai SHAP untuk model forecasting..." if st.session_state.language == 'id' else "Calculating SHAP values for forecasting model..."):
                            try:
                                # Gunakan fungsi interpretasi forecasting baru
                                interpretation_results = interpret_forecasting_model(
                                    model=st.session_state.model,
                                    X_train=st.session_state.X_train[selected_features],
                                    y_train=st.session_state.y_train,
                                    X_test=st.session_state.X_test[selected_features],
                                    feature_names=selected_features,
                                    method='shap',
                                    n_samples=sample_size,
                                    random_state=42
                                )
                                
                                # Tampilkan dashboard interpretasi
                                st.subheader("Dashboard Interpretasi Forecasting" if st.session_state.language == 'id' else "Forecasting Interpretation Dashboard")
                                
                                # Buat dan tampilkan dashboard
                                dashboard_fig = create_forecasting_interpretation_dashboard(interpretation_results, method='shap')
                                st.pyplot(dashboard_fig)
                                
                                # Tampilkan feature importance sebagai tabel
                                st.subheader("Feature Importance" if st.session_state.language == 'id' else "Feature Importance")
                                importance_df = pd.DataFrame(
                                    list(interpretation_results['feature_importance'].items()),
                                    columns=['Feature', 'Importance']
                                ).sort_values('Importance', ascending=False)
                                st.dataframe(importance_df)
                                
                                st.success("Analisis SHAP untuk model forecasting berhasil diselesaikan!" if st.session_state.language == 'id' else "SHAP analysis for forecasting model completed successfully!")
                                
                            except Exception as e:
                                st.error(f"Error dalam analisis SHAP forecasting: {str(e)}")
                                
            else:
                st.warning("Tidak dapat menemukan fitur untuk model forecasting. Pastikan model telah dilatih dengan benar." if st.session_state.language == 'id' else 
                        "Could not find features for the forecasting model. Make sure the model has been trained correctly.")
        else:
            st.info("Silakan latih model forecasting terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a forecasting model in the 'Model Training' tab first.")
              
    elif st.session_state.problem_type == "Classification":
        st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

        if st.session_state.model is not None:
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
                            # Gunakan fungsi implementasi SHAP untuk klasifikasi
                            shap_result = implement_shap_classification(
                                st.session_state.model, 
                                X_sample, 
                                st.session_state.X_train[selected_features],
                                st.session_state.language
                            )
                        except Exception as e:
                            st.error(f"Error dalam implementasi SHAP klasifikasi: {str(e)}")
                            shap_result = {'success': False, 'error': str(e), 'shap_values': None, 'explainer': None}
                        
                        if shap_result['success'] and shap_result.get('shap_values') is not None:
                            explainer = shap_result['explainer']
                            shap_values = shap_result['shap_values']
                            shap_values_selected = None  # Initialize variable
                            
                            # Pilih kelas untuk visualisasi
                            if shap_values is not None:
                                if isinstance(shap_values, list) and len(shap_values) > 1:
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
                                    # Handle case where shap_values might be a DataFrame or numpy array
                                    if isinstance(shap_values, list):
                                        shap_values_selected = shap_values[0] if len(shap_values) > 0 else shap_values
                                    else:
                                        shap_values_selected = shap_values
                            else:
                                st.error("SHAP values tidak tersedia" if st.session_state.language == 'id' else "SHAP values not available")
                            
                            # Validasi bahwa shap_values_selected tersedia
                            if shap_values_selected is None:
                                st.error("SHAP values tidak tersedia untuk visualisasi" if st.session_state.language == 'id' else "SHAP values not available for visualization")
                                # Gunakan continue untuk skip ke iterasi berikutnya atau break untuk keluar dari loop
                                # Karena ini di luar fungsi, kita tidak bisa menggunakan return
                                st.stop()  # Menghentikan eksekusi Streamlit
                            
                            # Visualisasi SHAP
                            st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                            
                            # 1. Summary Plot
                            st.write("### Summary Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                shap.summary_plot(shap_values_selected, X_sample, show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Summary Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Summary Plot: {str(e)}")
                            
                            # 2. Feature Importance Plot
                            st.write("### Feature Importance Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(shap_values_selected, X_sample, plot_type="bar", show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Feature Importance Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Feature Importance Plot: {str(e)}")
                            
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
                                    try:
                                        feature_name = X_sample.columns[idx]
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        shap.dependence_plot(idx, shap_values_selected, X_sample, show=False, ax=ax)
                                        plt.title(f"Dependence Plot for {feature_name}")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.clf()
                                    except Exception as e:
                                        st.warning(f"Gagal membuat Dependence Plot untuk {feature_name}: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Dependence Plot for {feature_name}: {str(e)}")
                            
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
                            try:
                                if isinstance(shap_values_selected, list):
                                    # Untuk multi-output, ambil output dan expected_value untuk kelas pertama
                                    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                                    force_plot = shap.force_plot(expected_val, 
                                                            shap_values_selected[0][sample_idx, :], 
                                                            X_sample.iloc[sample_idx, :], 
                                                            matplotlib=True,
                                                            show=False)
                                else:
                                    expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                                    force_plot = shap.force_plot(expected_val, 
                                                            shap_values_selected[sample_idx, :], 
                                                            X_sample.iloc[sample_idx, :], 
                                                            matplotlib=True,
                                                            show=False)
                                st.pyplot(force_plot)
                            except Exception as e:
                                st.warning(f"Gagal membuat Force Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Force Plot: {str(e)}")
                            
                            # 5. Waterfall Plot
                            st.write("### Waterfall Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))

                                if isinstance(shap_values_selected, list):
                                    # Untuk multi-output, ambil output dan expected_value untuk kelas pertama
                                    expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                                    shap.plots._waterfall.waterfall_legacy(
                                        expected_val,
                                        shap_values_selected[0][sample_idx, :],
                                        feature_names=X_sample.columns,
                                        show=False,
                                        max_display=10
                                    )
                                else:
                                    expected_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]
                                    shap.plots._waterfall.waterfall_legacy(
                                        expected_val,
                                        shap_values_selected[sample_idx, :],
                                        feature_names=X_sample.columns,
                                        show=False,
                                        max_display=10
                                    )

                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Waterfall Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Waterfall Plot: {str(e)}")
                        
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
                        
                        # Handle case where shap_result indicates failure but wasn't caught by except block
                        if not shap_result.get('success', False):
                            st.error(f"Error dalam implementasi SHAP klasifikasi: {shap_result.get('error', 'Unknown error')}")
        else:
            st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")
                        
    elif st.session_state.problem_type == "Regression":
        st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

        if st.session_state.model is not None:
            st.subheader("Pemilihan Fitur untuk Analisis SHAP" if st.session_state.language == 'id' else "Feature Selection for SHAP Analysis")
            
            # Pilih fitur untuk analisis SHAP
            feature_names = list(st.session_state.X.columns)
            selected_features = st.multiselect(
                "Pilih fitur untuk analisis SHAP:" if st.session_state.language == 'id' else "Select features for SHAP analysis:",
                options=feature_names,
                default=feature_names[:min(10, len(feature_names))]
            )
            
            # Jumlah sampel untuk analisis SHAP
            sample_size = st.slider(
                "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                min_value=10, max_value=min(100, len(st.session_state.X)), value=50
            )
            
            if st.button("Generate SHAP Values" if st.session_state.language == 'id' else "Generate SHAP Values"):
                if not selected_features:
                    st.error("Silakan pilih setidaknya satu fitur untuk analisis SHAP." if st.session_state.language == 'id' else "Please select at least one feature for SHAP analysis.")
                else:
                    with st.spinner("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values..."):
                        try:
                            # Persiapkan data untuk SHAP
                            X_sample = st.session_state.X[selected_features].sample(min(sample_size, len(st.session_state.X)), random_state=42)
                            
                            # Handle categorical features with one-hot encoding
                            categorical_features = X_sample.select_dtypes(include=['object', 'category']).columns.tolist()
                            if categorical_features:
                                X_sample_encoded = pd.get_dummies(X_sample, columns=categorical_features)
                                st.info(f"Menggunakan one-hot encoding untuk fitur kategorikal: {categorical_features}" if st.session_state.language == 'id' else f"Using one-hot encoding for categorical features: {categorical_features}")
                            else:
                                X_sample_encoded = X_sample.copy()
                            
                            # Pastikan semua nilai dalam X_sample_encoded adalah numerik
                            for col in X_sample_encoded.columns:
                                try:
                                    X_sample_encoded[col] = np.array(X_sample_encoded[col]).astype(float)
                                except:
                                    try:
                                        X_sample_encoded[col] = pd.factorize(X_sample_encoded[col])[0].astype(float)
                                    except Exception as e:
                                        st.error(f"Error saat mengkonversi kolom {col} ke numerik: {str(e)}")
                            
                            # Pilih explainer berdasarkan jenis model
                            if hasattr(st.session_state.model, 'feature_importances_'):
                                # Gunakan TreeExplainer untuk model berbasis pohon
                                explainer = shap.TreeExplainer(st.session_state.model)
                            else:
                                # Gunakan KernelExplainer untuk model lainnya
                                explainer = shap.KernelExplainer(st.session_state.model.predict, X_sample_encoded)
                            
                            # Hitung nilai SHAP
                            shap_values = explainer.shap_values(X_sample_encoded)
                            
                            st.success("Nilai SHAP berhasil dihitung!" if st.session_state.language == 'id' else "SHAP values calculated successfully!")
                            
                            # Validasi bahwa shap_values tersedia
                            if shap_values is None:
                                st.error("SHAP values tidak tersedia untuk visualisasi" if st.session_state.language == 'id' else "SHAP values not available for visualization")
                                st.stop()
                            
                            # Visualisasi SHAP
                            st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                            
                            # 1. Summary Plot
                            st.write("### Summary Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                shap.summary_plot(shap_values, X_sample_encoded, show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Summary Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Summary Plot: {str(e)}")
                            
                            # 2. Feature Importance Plot
                            st.write("### Feature Importance Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.summary_plot(shap_values, X_sample_encoded, plot_type="bar", show=False)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Feature Importance Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Feature Importance Plot: {str(e)}")
                            
                            # 3. Dependence Plots untuk fitur teratas
                            st.write("### Dependence Plots")
                            
                            # Hitung rata-rata nilai absolut SHAP untuk setiap fitur
                            shap_arr = np.array(shap_values, dtype=float)
                            feature_importance = np.abs(shap_arr).mean(0)
                            
                            # Dapatkan indeks fitur terurut berdasarkan kepentingan
                            top_indices = feature_importance.argsort()[-5:][::-1]
                            
                            # Buat dependence plot untuk 5 fitur teratas
                            for idx in top_indices:
                                if idx < len(X_sample_encoded.columns):  # Pastikan indeks valid
                                    try:
                                        feature_name = X_sample_encoded.columns[idx]
                                        fig, ax = plt.subplots(figsize=(10, 6))
                                        shap.dependence_plot(idx, shap_values, X_sample_encoded, show=False, ax=ax)
                                        plt.title(f"Dependence Plot for {feature_name}")
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        plt.clf()
                                    except Exception as e:
                                        st.warning(f"Gagal membuat Dependence Plot untuk {feature_name}: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Dependence Plot for {feature_name}: {str(e)}")
                            
                            # 4. Force Plot untuk sampel individual
                            st.write("### Force Plot untuk Sampel Individual")
                            sample_idx = st.slider(
                                "Pilih indeks sampel:" if st.session_state.language == 'id' else "Select sample index:",
                                0, len(X_sample_encoded) - 1, 0
                            )
                            
                            # Tampilkan data sampel
                            st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                            st.dataframe(X_sample_encoded.iloc[[sample_idx]])
                            
                            # Force plot
                            try:
                                expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                                force_plot = shap.force_plot(expected_val, 
                                                        shap_values[sample_idx, :], 
                                                        X_sample_encoded.iloc[sample_idx, :], 
                                                        matplotlib=True,
                                                        show=False)
                                st.pyplot(force_plot)
                            except Exception as e:
                                st.warning(f"Gagal membuat Force Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Force Plot: {str(e)}")
                            
                            # 5. Waterfall Plot
                            st.write("### Waterfall Plot")
                            try:
                                fig, ax = plt.subplots(figsize=(10, 8))
                                expected_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]
                                shap.plots._waterfall.waterfall_legacy(
                                    expected_val,
                                    shap_values[sample_idx, :],
                                    feature_names=X_sample_encoded.columns,
                                    show=False,
                                    max_display=10
                                )
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Waterfall Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Waterfall Plot: {str(e)}")
                            
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
        else:
            st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")
    
    elif st.session_state.problem_type == "Forecasting":
        st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

        if st.session_state.model is not None:
            try:
                # Pilih fitur untuk SHAP
                st.subheader("Konfigurasi SHAP" if st.session_state.language == 'id' else "SHAP Configuration")
                
                # Pilih jumlah sampel untuk SHAP
                n_samples = st.slider(
                    "Jumlah sampel untuk analisis SHAP:" if st.session_state.language == 'id' else "Number of samples for SHAP analysis:",
                    min_value=10, max_value=min(1000, len(X)), value=min(100, len(X))
                )
                
                # Ambil sampel secara acak
                if n_samples < len(X):
                    sample_indices = np.random.choice(len(X), n_samples, replace=False)
                    X_sample = X.iloc[sample_indices]
                    y_sample = y.iloc[sample_indices] if y is not None else None
                else:
                    X_sample = X
                    y_sample = y
                
                # Konversi categorical features ke numerik jika diperlukan
                X_sample_processed = X_sample.copy()
                
                # Handle categorical features dengan one-hot encoding
                categorical_features = X_sample_processed.select_dtypes(include=['object', 'category']).columns
                if len(categorical_features) > 0:
                    st.info(f"Fitur kategorikal ditemukan: {list(categorical_features)}. Menggunakan one-hot encoding." if st.session_state.language == 'id' else f"Categorical features found: {list(categorical_features)}. Using one-hot encoding.")
                    X_sample_processed = pd.get_dummies(X_sample_processed, columns=categorical_features, drop_first=True)
                
                # Konversi semua fitur ke tipe numerik
                for col in X_sample_processed.columns:
                    if X_sample_processed[col].dtype == 'object':
                        try:
                            X_sample_processed[col] = pd.to_numeric(X_sample_processed[col], errors='coerce')
                        except:
                            # Jika tidak bisa dikonversi, gunakan label encoding
                            from sklearn.preprocessing import LabelEncoder
                            le = LabelEncoder()
                            X_sample_processed[col] = le.fit_transform(X_sample_processed[col].astype(str))
                
                # Hapus kolom dengan nilai NaN
                X_sample_processed = X_sample_processed.dropna(axis=1, how='all')  # Hapus kolom yang semua nilainya NaN
                X_sample_processed = X_sample_processed.fillna(X_sample_processed.mean())  # Isi nilai yang tersisa dengan rata-rata
                
                st.write(f"Menggunakan {X_sample_processed.shape[0]} sampel dengan {X_sample_processed.shape[1]} fitur untuk analisis SHAP." if st.session_state.language == 'id' else f"Using {X_sample_processed.shape[0]} samples with {X_sample_processed.shape[1]} features for SHAP analysis.")
                
                # Hitung nilai SHAP
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Menghitung nilai SHAP..." if st.session_state.language == 'id' else "Calculating SHAP values...")
                progress_bar.progress(25)
                
                # Gunakan TreeExplainer untuk model berbasis tree, KernelExplainer untuk model lainnya
                try:
                    if hasattr(st.session_state.model, 'estimators_'):  # Tree-based models (Random Forest, Gradient Boosting, etc.)
                        explainer = shap.TreeExplainer(st.session_state.model)
                    else:  # Other models (Linear Regression, Neural Networks, etc.)
                        explainer = shap.KernelExplainer(st.session_state.model.predict, X_sample_processed)
                    
                    shap_values_selected = explainer.shap_values(X_sample_processed)
                    progress_bar.progress(75)
                    
                except Exception as e:
                    st.error(f"Error saat membuat explainer: {str(e)}")
                    shap_values_selected = None
                
                progress_bar.progress(100)
                status_text.text("Selesai!" if st.session_state.language == 'id' else "Done!")
                
                if shap_values_selected is not None:
                    # Visualisasi SHAP
                    st.subheader("Visualisasi SHAP" if st.session_state.language == 'id' else "SHAP Visualizations")
                    
                    # 1. Summary Plot
                    st.write("### Summary Plot")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.summary_plot(shap_values_selected, X_sample_processed, show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Gagal membuat Summary Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Summary Plot: {str(e)}")
                    
                    # 2. Feature Importance Plot
                    st.write("### Feature Importance Plot")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        shap.summary_plot(shap_values_selected, X_sample_processed, plot_type="bar", show=False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Gagal membuat Feature Importance Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Feature Importance Plot: {str(e)}")
                    
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
                        if idx < len(X_sample_processed.columns):  # Pastikan indeks valid
                            try:
                                feature_name = X_sample_processed.columns[idx]
                                fig, ax = plt.subplots(figsize=(10, 6))
                                shap.dependence_plot(idx, shap_values_selected, X_sample_processed, show=False, ax=ax)
                                plt.title(f"Dependence Plot for {feature_name}")
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.clf()
                            except Exception as e:
                                st.warning(f"Gagal membuat Dependence Plot untuk {feature_name}: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Dependence Plot for {feature_name}: {str(e)}")
                    
                    # 4. Force Plot untuk sampel individual
                    st.write("### Force Plot untuk Sampel Individual")
                    sample_idx = st.slider(
                        "Pilih indeks sampel:" if st.session_state.language == 'id' else "Select sample index:",
                        0, len(X_sample_processed) - 1, 0
                    )
                    
                    # Tampilkan data sampel
                    st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                    st.dataframe(X_sample_processed.iloc[[sample_idx]])
                    
                    # Force plot
                    try:
                        if isinstance(shap_values_selected, list):
                            # Untuk multi-output, ambil output pertama
                            expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value
                            force_plot = shap.force_plot(expected_val, 
                                                    shap_values_selected[0][sample_idx, :], 
                                                    X_sample_processed.iloc[sample_idx, :], 
                                                    matplotlib=True,
                                                    show=False)
                        else:
                            expected_val = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                            force_plot = shap.force_plot(expected_val, 
                                                    shap_values_selected[sample_idx, :], 
                                                    X_sample_processed.iloc[sample_idx, :], 
                                                    matplotlib=True,
                                                    show=False)
                        st.pyplot(force_plot)
                    except Exception as e:
                        st.warning(f"Gagal membuat Force Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Force Plot: {str(e)}")
                    
                    # 5. Waterfall Plot
                    st.write("### Waterfall Plot")
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))

                        if isinstance(shap_values_selected, list):
                            # Untuk multi-output, ambil output dan expected_value untuk kelas pertama
                            expected_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                            shap.plots._waterfall.waterfall_legacy(
                                expected_val,
                                shap_values_selected[0][sample_idx, :],
                                feature_names=X_sample_processed.columns,
                                show=False,
                                max_display=10
                            )
                        else:
                            expected_val = explainer.expected_value if np.isscalar(explainer.expected_value) else explainer.expected_value[0]
                            shap.plots._waterfall.waterfall_legacy(
                                expected_val,
                                shap_values_selected[sample_idx, :],
                                feature_names=X_sample_processed.columns,
                                show=False,
                                max_display=10
                            )

                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.clf()
                    except Exception as e:
                        st.warning(f"Gagal membuat Waterfall Plot: {str(e)}" if st.session_state.language == 'id' else f"Failed to create Waterfall Plot: {str(e)}")
                    
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
                    
                else:
                    st.error("SHAP values tidak tersedia untuk visualisasi" if st.session_state.language == 'id' else "SHAP values not available for visualization")
                    
            except Exception as e:
                st.error(f"Error saat menghitung nilai SHAP: {str(e)}")
                
        else:
            st.warning("Model belum tersedia. Silakan latih model terlebih dahulu." if st.session_state.language == 'id' else "Model not available. Please train a model first.")
            
    elif st.session_state.problem_type == "Forecasting":
        st.header("Interpretasi Model dengan SHAP" if st.session_state.language == 'id' else "Model Interpretation with SHAP")

        if st.session_state.model is not None:
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
    st.info("⚠️ **Notifikasi:** Fitur interpretasi LIME sekarang mendukung model **regresi**, **klasifikasi**, dan **forecasting**." if st.session_state.language == 'id' else "⚠️ **Notification:** LIME interpretation now supports **regression**, **classification**, and **forecasting** models.")
    if st.session_state.problem_type not in ['Regression', 'Classification', 'Forecasting']:
        st.info("Fitur interpretasi LIME hanya tersedia untuk model regresi, klasifikasi, dan forecasting." if st.session_state.language == 'id' else "LIME interpretation is only available for regression, classification, and forecasting models.")
    else:

        if not LIME_AVAILABLE:
            st.error("LIME tidak terinstal. Silakan instal dengan 'pip install lime'." if st.session_state.language == 'id' else "LIME is not installed. Please install it with 'pip install lime'.")
        elif (
            st.session_state.model is not None
            and st.session_state.problem_type in ["Regression", "Classification", "Forecasting"]
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

                        st.subheader("Penjelasan Prediksi Individual" if st.session_state.language == 'id' else "Individual Prediction Explanation")
                        sample_idx = st.slider(
                            "Indeks sampel:", 0, len(X_test_selected) - 1, 0,
                            key="lime_sample_idx"
                        )
                        sample = X_test_selected.iloc[sample_idx]
                        st.write("Data sampel:" if st.session_state.language == 'id' else "Sample data:")
                        st.dataframe(pd.DataFrame([sample], columns=selected_features))

                        actual = st.session_state.y_test_eval.iloc[sample_idx]
                        original_sample = st.session_state.X_test.iloc[sample_idx:sample_idx+1]
                        predicted = st.session_state.model.predict(original_sample)[0]
                        st.write(f"Nilai aktual: {actual}")
                        st.write(f"Nilai prediksi: {predicted}")

                        # Gunakan fungsi utilitas untuk klasifikasi dan forecasting
                        if st.session_state.problem_type == "Classification":
                            lime_result = implement_lime_classification(
                                st.session_state.model,
                                X_train_selected,
                                X_test_selected,
                                selected_features,
                                sample_idx,
                                num_features_show,
                                st.session_state.language
                            )
                            
                            if lime_result['success']:
                                explanation = lime_result['explanation']
                                
                                st.subheader("Visualisasi Penjelasan LIME" if st.session_state.language == 'id' else "LIME Explanation Visualization")
                                fig = plt.figure(figsize=(10, 6))
                                
                                class_names = st.session_state.model.classes_ if hasattr(st.session_state.model, 'classes_') else None
                                if class_names is not None:
                                    if predicted in class_names:
                                        label_idx = list(class_names).index(predicted)
                                    else:
                                        label_idx = int(predicted)
                                    lime_fig = explanation.as_pyplot_figure(label=label_idx)
                                else:
                                    lime_fig = explanation.as_pyplot_figure()
                                
                                plt.tight_layout()
                                st.pyplot(lime_fig)
                                
                                st.subheader("Penjelasan dalam Bentuk Tabel" if st.session_state.language == 'id' else "Explanation in Table Format")
                                explanation_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Kontribusi"])
                                explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                                st.dataframe(explanation_df)
                            else:
                                st.error(f"Error dalam implementasi LIME klasifikasi: {lime_result['error']}")
                        elif st.session_state.problem_type == "Forecasting":
                            # Forecasting - gunakan pendekatan khusus
                            try:
                                # Siapkan data untuk forecasting
                                forecasting_data = prepare_forecasting_data_for_interpretation(
                                    st.session_state.X_train,
                                    st.session_state.X_test,
                                    selected_features,
                                    sample_idx
                                )
                                
                                if forecasting_data is not None:
                                    st.subheader("Visualisasi Penjelasan LIME untuk Forecasting" if st.session_state.language == 'id' else "LIME Explanation Visualization for Forecasting")
                                    
                                    # Gunakan LIME untuk forecasting dengan mode regresi
                                    lime_mode = "regression"
                                    predict_fn = st.session_state.model.predict

                                    explainer = lime_tabular.LimeTabularExplainer(
                                        forecasting_data['X_train'].values,
                                        feature_names=selected_features,
                                        mode=lime_mode,
                                        random_state=42
                                    )

                                    explanation = explainer.explain_instance(
                                        forecasting_data['sample'].values,
                                        predict_fn,
                                        num_features=num_features_show
                                    )

                                    fig = plt.figure(figsize=(10, 6))
                                    lime_fig = explanation.as_pyplot_figure()
                                    plt.tight_layout()
                                    st.pyplot(lime_fig)

                                    st.subheader("Penjelasan dalam Bentuk Tabel" if st.session_state.language == 'id' else "Explanation in Table Format")
                                    explanation_df = pd.DataFrame(explanation.as_list(), columns=["Feature", "Kontribusi"])
                                    explanation_df = explanation_df.sort_values("Kontribusi", ascending=False)
                                    st.dataframe(explanation_df)
                                    
                                    # Penjelasan khusus untuk forecasting
                                    st.subheader("Interpretasi untuk Model Forecasting" if st.session_state.language == 'id' else "Interpretation for Forecasting Model")
                                    st.info("""
                                    Dalam model forecasting, fitur-fitur penting biasanya meliputi:
                                    - **Lag Features**: Nilai historis dari variabel target
                                    - **Fitur Tanggal/Waktu**: Seperti hari dalam minggu, bulan, kuartal, dll.
                                    - **Fitur Rolling**: Seperti rata-rata bergerak, standar deviasi, dll.
                                    
                                    Nilai LIME tinggi pada lag features menunjukkan bahwa model sangat bergantung pada pola historis terbaru.
                                    """ if st.session_state.language == 'id' else """
                                    In forecasting models, important features typically include:
                                    - **Lag Features**: Historical values of the target variable
                                    - **Date/Time Features**: Such as day of week, month, quarter, etc.
                                    - **Rolling Features**: Such as moving averages, standard deviations, etc.
                                    
                                    High LIME values on lag features indicate that the model heavily relies on recent historical patterns.
                                    """)
                                else:
                                    st.error("Tidak dapat menyiapkan data untuk interpretasi LIME forecasting." if st.session_state.language == 'id' else "Could not prepare data for LIME forecasting interpretation.")
                            except Exception as e:
                                st.error(f"Error dalam implementasi LIME forecasting: {str(e)}")
                        else:
                            # Regresi - gunakan logika lama
                            lime_mode = "regression"
                            predict_fn = st.session_state.model.predict

                            explainer = lime_tabular.LimeTabularExplainer(
                                X_train_selected.values,
                                feature_names=selected_features,
                                mode=lime_mode,
                                random_state=42
                            )

                            explanation = explainer.explain_instance(
                                sample.values,
                                predict_fn,
                                num_features=num_features_show
                            )

                            st.subheader("Visualisasi Penjelasan LIME" if st.session_state.language == 'id' else "LIME Explanation Visualization")
                            fig = plt.figure(figsize=(10, 6))
                            lime_fig = explanation.as_pyplot_figure()  # Untuk regresi, JANGAN beri argumen label
                            plt.tight_layout()
                            st.pyplot(lime_fig)

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
                        
        else:
            st.info("Silakan latih model terlebih dahulu di tab 'Model Training'." if st.session_state.language == 'id' else "Please train a model in the 'Model Training' tab first.")

# Tab 7: Time Series Anomaly Detection
with tab7:
    st.header("Deteksi Anomali Time Series" if st.session_state.language == 'id' else "Time Series Anomaly Detection")
    
    st.info("""
    🔍 **Fitur Deteksi Anomali Time Series**
    
    Tab ini menyediakan algoritma deteksi anomali **state-of-the-art** untuk data time series dengan beberapa opsi:
    - **Isolation Forest**: Deteksi berbasis isolasi dengan ensemble trees
    - **One-Class SVM**: Deteksi berbasis margin hyperplane
    - **Statistical**: Deteksi berbasis statistik rolling window (Z-Score)
    - **Ensemble**: Kombinasi multiple methods
    
    **Catatan**: Fitur ini khusus untuk data time series dengan kolom tanggal/waktu.
    """ if st.session_state.language == 'id' else """
    🔍 **Time Series Anomaly Detection Features**
    
    This tab provides **state-of-the-art** anomaly detection algorithms for time series data with multiple options:
    - **Isolation Forest**: Isolation-based detection with ensemble trees
    - **One-Class SVM**: Margin-based hyperplane detection
    - **Statistical**: Rolling window statistics-based detection (Z-Score)
    - **Ensemble**: Combination of multiple methods
    
    **Note**: This feature is specifically for time series data with date/time columns.
    """)
    
    if st.session_state.data is None:
        st.warning("Silakan unggah dataset di tab 'Data Upload' terlebih dahulu." if st.session_state.language == 'id' else "Please upload a dataset in the 'Data Upload' tab first.")
    else:
        # Check for time series data
        date_columns = []
        for col in st.session_state.data.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']):
                try:
                    pd.to_datetime(st.session_state.data[col])
                    date_columns.append(col)
                except:
                    pass
        
        if not date_columns:
            st.warning("Tidak ditemukan kolom tanggal/waktu dalam dataset. Pastikan ada kolom dengan nama yang mengandung kata kunci tanggal/waktu." if st.session_state.language == 'id' else "No date/time column found in the dataset. Ensure there is a column with date/time keywords in the name.")
        else:
            # Select date column
            date_column = st.selectbox(
                "Pilih kolom tanggal/waktu:" if st.session_state.language == 'id' else "Select date/time column:",
                date_columns,
                key="ts_date_column"
            )
            
            # Select target column for anomaly detection
            numerical_columns = st.session_state.data.select_dtypes(include=[np.number]).columns.tolist()
            target_column = st.selectbox(
                "Pilih kolom target untuk deteksi anomali:" if st.session_state.language == 'id' else "Select target column for anomaly detection:",
                [col for col in numerical_columns if col != date_column],
                key="ts_target_column"
            )
            
            # Data preparation
            st.subheader("📊 Persiapan Data" if st.session_state.language == 'id' else "Data Preparation")
            
            # Prepare time series data
            preview_data = st.session_state.data[[date_column, target_column]].copy()
            preview_data[date_column] = pd.to_datetime(preview_data[date_column])
            preview_data = preview_data.sort_values(date_column)
            
            # Handle missing values
            missing_count = preview_data[target_column].isnull().sum()
            if missing_count > 0:
                st.warning(f"Terdapat {missing_count} nilai missing. Nilai missing akan dihapus." if st.session_state.language == 'id' else f"There are {missing_count} missing values. Missing values will be removed.")
                preview_data = preview_data.dropna()
            
            ts_data = preview_data.set_index(date_column)[target_column]
            
            st.write(f"**Jumlah data:** {len(ts_data)}")
            st.write(f"**Rentang waktu:** {ts_data.index.min()} sampai {ts_data.index.max()}")
            
            # Basic visualization
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7)
            ax.set_title(f'Time Series: {target_column}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Select anomaly detection methods
            st.subheader("🎯 Pilih Metode Deteksi Anomali" if st.session_state.language == 'id' else "Select Anomaly Detection Methods")
            
            available_methods = {
                'isolation_forest': 'Isolation Forest',
                'one_class_svm': 'One-Class SVM',
                'statistical': 'Statistical (Z-Score)',
                'ensemble': 'Ensemble Method'
            }
            
            # Check for sklearn availability
            try:
                from sklearn.ensemble import IsolationForest
                from sklearn.svm import OneClassSVM
                from scipy import stats
                SKLEARN_AVAILABLE = True
            except ImportError:
                SKLEARN_AVAILABLE = False
                st.error("Scikit-learn atau scipy tidak tersedia." if st.session_state.language == 'id' else "Scikit-learn or scipy not available.")
            
            if SKLEARN_AVAILABLE:
                selected_methods = st.multiselect(
                    "Pilih metode deteksi anomali:" if st.session_state.language == 'id' else "Select anomaly detection methods:",
                    options=list(available_methods.keys()),
                    format_func=lambda x: available_methods[x],
                    default=['isolation_forest', 'statistical']
                )
                
                if selected_methods:
                    # Parameters configuration
                    st.subheader("⚙️ Konfigurasi Parameter" if st.session_state.language == 'id' else "Parameter Configuration")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        contamination = st.slider(
                            "Tingkat kontaminasi (proporsi anomali):" if st.session_state.language == 'id' else "Contamination level (anomaly proportion):",
                            0.01, 0.3, 0.05, 0.01
                        )
                    
                    with col2:
                        z_threshold = st.slider(
                            "Z-score threshold untuk Statistical method:" if st.session_state.language == 'id' else "Z-score threshold for Statistical method:",
                            1.0, 5.0, 3.0, 0.5
                        )
                    
                    # Run anomaly detection
                    if st.button("🚀 Jalankan Deteksi Anomali" if st.session_state.language == 'id' else "Run Anomaly Detection", type="primary"):
                        with st.spinner("Menjalankan deteksi anomali..." if st.session_state.language == 'id' else "Running anomaly detection..."):
                            try:
                                # Validasi data
                                if len(ts_data) < 10:
                                    st.error("Dataset terlalu pendek. Minimal 10 data points diperlukan." if st.session_state.language == 'id' else "Dataset too short. Minimum 10 data points required.")
                                elif ts_data.std() == 0:
                                    st.error("Data memiliki nilai konstan. Deteksi anomali tidak dapat dilakukan." if st.session_state.language == 'id' else "Data has constant values. Anomaly detection cannot be performed.")
                                else:
                                    # Gunakan fungsi dari anomaly_detection_utils
                                    from anomaly_detection_utils import detect_and_visualize_anomalies
                                    
                                    results = {}
                                    
                                    # Siapkan DataFrame untuk fungsi deteksi
                                    df_for_detection = pd.DataFrame({
                                        date_column: ts_data.index,
                                        target_column: ts_data.values
                                    })
                                    
                                    # Jalankan deteksi untuk setiap metode
                                    for method in selected_methods:
                                        try:
                                            detection_results = detect_and_visualize_anomalies(
                                                data=df_for_detection,
                                                target_column=target_column,
                                                date_column=date_column,
                                                methods=[method],
                                                contamination=contamination
                                            )
                                            
                                            if method in detection_results and 'result' in detection_results[method]:
                                                result_data = detection_results[method]['result']
                                                summary_data = detection_results[method]['summary']
                                                
                                                results[method] = {
                                                    'anomalies': result_data['anomalies'],
                                                    'anomaly_count': summary_data['anomaly_count'],
                                                    'anomaly_percentage': summary_data['anomaly_percentage'],
                                                    'anomaly_indices': ts_data.index[result_data['anomalies']],
                                                    'anomaly_values': ts_data.values[result_data['anomalies']],
                                                    'summary': summary_data
                                                }
                                            else:
                                                st.error(f"Error pada metode {method}: Hasil deteksi tidak valid")
                                                
                                        except Exception as e:
                                            st.error(f"Error pada metode {method}: {str(e)}")
                                    
                                    # Display results
                                    st.subheader("📋 Hasil Deteksi Anomali" if st.session_state.language == 'id' else "Anomaly Detection Results")
                                
                                # Summary table
                                if results:
                                    summary_data = []
                                    for method, data in results.items():
                                        summary_data.append({
                                            'Method': available_methods[method],
                                            'Total Points': len(ts_data),
                                            'Anomalies Detected': data['anomaly_count'],
                                            'Anomaly Percentage (%)': round(data['anomaly_percentage'], 2)
                                        })
                                    
                                    summary_df = pd.DataFrame(summary_data)
                                    st.dataframe(summary_df)
                                    
                                    # Detailed results for each method
                                    for method, data in results.items():
                                        st.subheader(f"🔍 {available_methods[method]} Results")
                                        
                                        # Create visualization
                                        fig, ax = plt.subplots(figsize=(15, 6))
                                        
                                        # Plot normal data
                                        ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7, label='Normal')
                                        
                                        # Plot anomalies
                                        if data['anomaly_count'] > 0:
                                            ax.scatter(data['anomaly_indices'], data['anomaly_values'], 
                                                     color='red', s=50, alpha=0.8, label='Anomalies')
                                        
                                        ax.set_title(f'{available_methods[method]} - Anomaly Detection')
                                        ax.set_xlabel('Date')
                                        ax.set_ylabel('Value')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                        
                                        # Show anomaly details
                                        if data['anomaly_count'] > 0:
                                            st.write(f"**{data['anomaly_count']} anomalies detected ({data['anomaly_percentage']:.2f}%)**")
                                            
                                            # Show first 10 anomalies
                                            anomaly_df = pd.DataFrame({
                                                'Date': data['anomaly_indices'][:10],
                                                'Value': data['anomaly_values'][:10]
                                            })
                                            st.dataframe(anomaly_df)
                                    
                                    # Combined visualization
                                    if len(results) > 1:
                                        st.subheader("📊 Analisis Perbandingan" if st.session_state.language == 'id' else "Comparative Analysis")
                                        
                                        fig, ax = plt.subplots(figsize=(15, 8))
                                        ax.plot(ts_data.index, ts_data.values, color='blue', alpha=0.7, label='Normal Data')
                                        
                                        colors = ['red', 'green', 'orange', 'purple']
                                        for i, (method, data) in enumerate(results.items()):
                                            if data['anomaly_count'] > 0:
                                                ax.scatter(data['anomaly_indices'], data['anomaly_values'], 
                                                         color=colors[i % len(colors)], s=50, alpha=0.8, 
                                                         label=f'{available_methods[method]} Anomalies')
                                        
                                        ax.set_title('Anomaly Detection Comparison - All Methods' if st.session_state.language == 'id' else 'Perbandingan Deteksi Anomali - Semua Metode')
                                        ax.set_xlabel('Date')
                                        ax.set_ylabel('Value')
                                        ax.legend()
                                        ax.grid(True, alpha=0.3)
                                        plt.xticks(rotation=45)
                                        plt.tight_layout()
                                        st.pyplot(fig)
                                    
                                    # Download results
                                    st.subheader("📥 Download Hasil" if st.session_state.language == 'id' else "Download Results")
                                    
                                    # Prepare download data
                                    download_data = preview_data.copy()
                                    download_data.set_index(date_column, inplace=True)
                                    
                                    for method, data in results.items():
                                        download_data[f'{available_methods[method]}_Anomaly'] = data['anomalies'].astype(int)
                                    
                                    csv = download_data.to_csv()
                                    st.download_button(
                                        label="📥 Download Hasil Deteksi Anomali (CSV)" if st.session_state.language == 'id' else "Download Anomaly Detection Results (CSV)",
                                        data=csv,
                                        file_name=f'anomaly_detection_{target_column}.csv',
                                        mime='text/csv'
                                    )
                                    
                                    st.success("Deteksi anomali selesai!" if st.session_state.language == 'id' else "Anomaly detection completed successfully!")
                                    
                            except Exception as e:
                                st.error(f"Error saat menjalankan deteksi anomali: {str(e)}")
                                st.error(f"Detail error: {str(e)}")

# ========================= FORECASTING VISUALIZATION FUNCTIONS =========================

def calculate_forecast_metrics(actual, predicted):
    """Menghitung metrik evaluasi untuk forecasting"""
    try:
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        import numpy as np
        
        # Validasi input
        actual = np.array(actual).flatten()
        predicted = np.array(predicted).flatten()
        
        # Hapus nilai NaN atau infinite
        mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) == 0 or len(predicted) == 0:
            return {
                'rmse': np.nan,
                'mae': np.nan,
                'mape': np.nan,
                'r2': np.nan,
                'count': 0
            }
        
        # Hitung metrik
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        
        # MAPE (Mean Absolute Percentage Error)
        mask_non_zero = actual != 0
        if np.any(mask_non_zero):
            mape = np.mean(np.abs((actual[mask_non_zero] - predicted[mask_non_zero]) / actual[mask_non_zero])) * 100
        else:
            mape = np.nan
        
        # R² Score
        ss_res = np.sum((actual - predicted) ** 2)
        ss_tot = np.sum((actual - np.mean(actual)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return {
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'count': len(actual)
        }
    except Exception as e:
        return {
            'rmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'r2': np.nan,
            'count': 0
        }

def plot_forecast_visualization(data, forecast_results, target_column, date_column=None):
    """Membuat visualisasi hasil forecasting"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Forecasting Analysis - {target_column}', fontsize=16, fontweight='bold')
        
        # Data preparation
        if date_column and date_column in data.columns:
            data = data.set_index(date_column)
        
        actual_data = data[target_column].dropna()
        
        # 1. Actual vs Predicted Plot
        ax1 = axes[0, 0]
        
        # Plot actual data
        ax1.plot(actual_data.index, actual_data.values, 
                label='Actual', color='blue', linewidth=2, alpha=0.8)
        
        # Plot forecast results
        colors = ['red', 'green', 'orange', 'purple', 'brown']
        model_names = []
        
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            if isinstance(forecast, dict) and 'forecast' in forecast:
                forecast_values = forecast['forecast']
            else:
                forecast_values = forecast
            
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                # Create forecast index
                start_idx = len(actual_data) - len(forecast_values)
                if start_idx >= 0:
                    forecast_index = actual_data.index[-len(forecast_values):]
                    ax1.plot(forecast_index, forecast_values, 
                            label=f'{model_name} Forecast', 
                            color=colors[i % len(colors)], 
                            linewidth=2, linestyle='--')
                    model_names.append(model_name)
        
        ax1.set_title('Actual vs Forecast Comparison')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. Residual Plot
        ax2 = axes[0, 1]
        
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            if isinstance(forecast, dict) and 'forecast' in forecast:
                forecast_values = forecast['forecast']
            else:
                forecast_values = forecast
            
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                # Calculate residuals
                actual_slice = actual_data.values[-len(forecast_values):]
                residuals = actual_slice - forecast_values
                
                # Create forecast index for residuals
                forecast_index = actual_data.index[-len(forecast_values):]
                
                ax2.scatter(forecast_index, residuals, 
                           label=f'{model_name} Residuals', 
                           color=colors[i % len(colors)], alpha=0.7)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.set_title('Residual Analysis')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Residual')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. Error Distribution
        ax3 = axes[1, 0]
        
        for i, (model_name, forecast) in enumerate(forecast_results.items()):
            if isinstance(forecast, dict) and 'forecast' in forecast:
                forecast_values = forecast['forecast']
            else:
                forecast_values = forecast
            
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                residuals = actual_slice - forecast_values
                
                ax3.hist(residuals, bins=20, alpha=0.6, 
                        label=f'{model_name} Errors', color=colors[i % len(colors)])
        
        ax3.set_title('Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create metrics table
        metrics_data = []
        for model_name, forecast in forecast_results.items():
            if isinstance(forecast, dict) and 'forecast' in forecast:
                forecast_values = forecast['forecast']
            else:
                forecast_values = forecast
            
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                metrics = calculate_forecast_metrics(actual_slice, forecast_values)
                
                metrics_data.append({
                    'Model': model_name,
                    'RMSE': f"{metrics['rmse']:.4f}" if not np.isnan(metrics['rmse']) else "N/A",
                    'MAE': f"{metrics['mae']:.4f}" if not np.isnan(metrics['mae']) else "N/A",
                    'MAPE': f"{metrics['mape']:.2f}%" if not np.isnan(metrics['mape']) else "N/A",
                    'R²': f"{metrics['r2']:.4f}" if not np.isnan(metrics['r2']) else "N/A"
                })
        
        if metrics_data:
            # Create table
            table_data = [[d['Model'], d['RMSE'], d['MAE'], d['MAPE'], d['R²']] for d in metrics_data]
            table = ax4.table(cellText=table_data,
                            colLabels=['Model', 'RMSE', 'MAE', 'MAPE', 'R²'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0, 0.2, 1, 0.6])
            
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            ax4.set_title('Model Performance Metrics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"Error dalam visualisasi forecasting: {str(e)}")
        return None

def display_forecast_summary(forecast_results, target_column, data):
    """Menampilkan ringkasan hasil forecasting"""
    try:
        st.subheader(f"📊 Forecasting Summary - {target_column}")
        
        # Calculate metrics for all models
        metrics_summary = []
        actual_data = data[target_column].dropna()
        
        for model_name, forecast in forecast_results.items():
            if isinstance(forecast, dict) and 'forecast' in forecast:
                forecast_values = forecast['forecast']
            else:
                forecast_values = forecast
            
            if hasattr(forecast_values, '__len__') and len(forecast_values) > 0:
                actual_slice = actual_data.values[-len(forecast_values):]
                metrics = calculate_forecast_metrics(actual_slice, forecast_values)
                
                metrics_summary.append({
                    'Model': model_name,
                    'RMSE': metrics['rmse'],
                    'MAE': metrics['mae'],
                    'MAPE': metrics['mape'],
                    'R²': metrics['r2'],
                    'Data Points': metrics['count']
                })
        
        if metrics_summary:
            # Create DataFrame for display
            metrics_df = pd.DataFrame(metrics_summary)
            
            # Display metrics cards
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Best RMSE", f"{metrics_df['RMSE'].min():.4f}")
            with col2:
                st.metric("Best MAE", f"{metrics_df['MAE'].min():.4f}")
            with col3:
                if not metrics_df['MAPE'].isna().all():
                    st.metric("Best MAPE", f"{metrics_df['MAPE'].min():.2f}%")
                else:
                    st.metric("Best MAPE", "N/A")
            with col4:
                st.metric("Best R²", f"{metrics_df['R²'].max():.4f}")
            
            # Display detailed table
            st.dataframe(
                metrics_df.style.format({
                    'RMSE': '{:.4f}',
                    'MAE': '{:.4f}',
                    'MAPE': '{:.2f}%',
                    'R²': '{:.4f}'
                }).background_gradient(cmap='RdYlGn', subset=['R²'], axis=0)
                 .background_gradient(cmap='RdYlGn_r', subset=['RMSE', 'MAE', 'MAPE'], axis=0)
            )
            
            return metrics_df
        
    except Exception as e:
        st.error(f"Error dalam menampilkan ringkasan: {str(e)}")
        return None