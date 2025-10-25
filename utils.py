import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
import warnings
warnings.filterwarnings('ignore')

def is_timeseries(data, column_name):
    """
    Mendeteksi apakah kolom tertentu dalam dataframe merupakan data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
    column_name : str
        Nama kolom yang akan diperiksa
        
    Returns:
    --------
    bool
        True jika kolom tersebut kemungkinan merupakan data timeseries, False jika tidak
    """
    # Cek apakah kolom tersebut dapat dikonversi ke datetime
    try:
        pd.to_datetime(data[column_name])
        return True
    except:
        pass
    
    # Cek apakah nama kolom mengandung kata kunci terkait waktu
    time_keywords = ['date', 'time', 'year', 'month', 'day', 'tanggal', 'waktu', 'tahun', 'bulan', 'hari']
    if any(keyword in column_name.lower() for keyword in time_keywords):
        return True
    
    return False

def detect_timeseries_columns(data):
    """
    Mendeteksi kolom-kolom yang kemungkinan merupakan data timeseries dalam dataframe
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
        
    Returns:
    --------
    list
        Daftar nama kolom yang kemungkinan merupakan data timeseries
    """
    timeseries_columns = []
    
    for col in data.columns:
        if is_timeseries(data, col):
            timeseries_columns.append(col)
    
    return timeseries_columns

def prepare_timeseries_data(data, date_column, target_column, freq=None):
    """
    Menyiapkan data timeseries untuk analisis dan forecasting
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame yang berisi data
    date_column : str
        Nama kolom tanggal/waktu
    target_column : str
        Nama kolom target yang akan diprediksi
    freq : str, optional
        Frekuensi data (D untuk harian, W untuk mingguan, M untuk bulanan, dll.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame yang telah disiapkan untuk analisis timeseries
    """
    # Buat salinan data
    ts_data = data.copy()
    
    # Konversi kolom tanggal ke datetime
    ts_data[date_column] = pd.to_datetime(ts_data[date_column])
    
    # Urutkan data berdasarkan tanggal
    ts_data = ts_data.sort_values(by=date_column)
    
    # Set tanggal sebagai index
    ts_data = ts_data.set_index(date_column)
    
    # Resampling data jika frekuensi ditentukan
    if freq is not None:
        ts_data = ts_data.resample(freq)[target_column].mean().to_frame()
    
    return ts_data

def check_stationarity(timeseries):
    """
    Memeriksa stasioneritas data timeseries menggunakan uji Augmented Dickey-Fuller
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan diperiksa
        
    Returns:
    --------
    dict
        Hasil uji stasioneritas
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    # Cek apakah data konstan (semua nilai sama)
    if len(clean_timeseries.unique()) <= 1:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,  # Data konstan dianggap stasioner
            'Message': 'Data konstan (semua nilai sama) - tidak dapat dilakukan uji stasioneritas'
        }
    
    # Cek apakah data terlalu pendek
    if len(clean_timeseries) < 3:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,
            'Message': 'Data terlalu pendek untuk uji stasioneritas'
        }
    
    try:
        # Uji Augmented Dickey-Fuller
        result = adfuller(clean_timeseries)
        
        # Siapkan output
        output = {
            'Test Statistic': result[0],
            'p-value': result[1],
            'Lags Used': result[2],
            'Number of Observations Used': result[3],
            'Critical Values': result[4],
            'Stationary': result[1] <= 0.05,
            'Message': None
        }
        
        return output
        
    except Exception as e:
        return {
            'Test Statistic': None,
            'p-value': None,
            'Lags Used': None,
            'Number of Observations Used': len(clean_timeseries),
            'Critical Values': None,
            'Stationary': True,
            'Message': f'Error dalam uji stasioneritas: {str(e)}'
        }

def plot_timeseries_analysis(timeseries, figsize=(15, 12)):
    """
    Membuat plot analisis timeseries (data asli, ACF, PACF)
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi plot analisis
    """
    # Hapus nilai missing dan cek jumlah data
    clean_timeseries = timeseries.dropna()
    
    # Cek apakah data terlalu sedikit untuk ACF/PACF
    if len(clean_timeseries) < 2:
        # Hanya plot data asli jika terlalu sedikit data
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(clean_timeseries)
        ax.set_title('Data Timeseries (Data terlalu sedikit untuk ACF/PACF)')
        plt.tight_layout()
        return fig

def interpret_forecasting_model(model, X_train, y_train, X_test, feature_names=None, 
                               method='shap', n_samples=100, random_state=42):
    """
    Fungsi untuk interpretasi model forecasting menggunakan SHAP atau LIME
    
    Parameters:
    -----------
    model : sklearn model
        Model forecasting yang telah dilatih
    X_train : pandas.DataFrame atau numpy.array
        Data training
    y_train : pandas.Series atau numpy.array
        Target training
    X_test : pandas.DataFrame atau numpy.array
        Data testing untuk interpretasi
    feature_names : list, optional
        Nama fitur
    method : str, default='shap'
        Metode interpretasi ('shap' atau 'lime')
    n_samples : int, default=100
        Jumlah sampel untuk interpretasi
    random_state : int, default=42
        Random state untuk reproduktivitas
        
    Returns:
    --------
    dict
        Hasil interpretasi yang berisi:
        - shap_values atau lime_explanations
        - feature_importance
        - summary_plot
    """
    
    # Validasi input
    if method not in ['shap', 'lime']:
        raise ValueError("Method harus 'shap' atau 'lime'")
    
    # Konversi ke numpy array jika perlu
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Siapkan nama fitur
    if feature_names is None:
        if hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        else:
            feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Pilih sampel untuk interpretasi
    if len(X_test) > n_samples:
        np.random.seed(random_state)
        indices = np.random.choice(len(X_test), n_samples, replace=False)
        X_sample = X_test[indices]
    else:
        X_sample = X_test.copy()
    
    results = {}
    
    if method == 'shap':
        try:
            import shap
            
            # Buat explainer berdasarkan tipe model
            if hasattr(model, 'tree_'):  # Tree-based models
                explainer = shap.TreeExplainer(model)
            else:  # Other models
                # Untuk forecasting, kita gunakan KernelExplainer dengan background data
                background = shap.sample(X_train, min(100, len(X_train)))
                explainer = shap.KernelExplainer(model.predict, background)
            
            # Hitung SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Simpan hasil
            results['shap_values'] = shap_values
            results['explainer'] = explainer
            results['X_sample'] = X_sample
            results['feature_names'] = feature_names
            
            # Hitung feature importance (rata-rata absolute SHAP values)
            if len(shap_values.shape) == 3:  # Multi-output
                feature_importance = np.mean(np.abs(shap_values), axis=(0, 2))
            elif len(shap_values.shape) == 2:  # Single output
                feature_importance = np.mean(np.abs(shap_values), axis=0)
            else:  # 1D array
                feature_importance = np.abs(shap_values)
            
            results['feature_importance'] = dict(zip(feature_names, feature_importance))
            
            # Buat summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
            results['summary_plot'] = plt.gcf()
            plt.close()
            
        except ImportError:
            raise ImportError("SHAP tidak terinstal. Silakan install dengan: pip install shap")
        except Exception as e:
            raise Exception(f"Error dalam menghitung SHAP values: {str(e)}")
    
    elif method == 'lime':
        try:
            import lime
            import lime.lime_tabular
            
            # Buat LIME explainer
            explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=feature_names,
                mode='regression',
                random_state=random_state
            )
            
            # Generate explanations untuk beberapa sampel
            lime_explanations = []
            n_lime_samples = min(10, len(X_sample))  # Batasi untuk efisiensi
            
            for i in range(n_lime_samples):
                explanation = explainer.explain_instance(
                    X_sample[i], 
                    model.predict,
                    num_features=len(feature_names)
                )
                lime_explanations.append(explanation)
            
            # Simpan hasil
            results['lime_explanations'] = lime_explanations
            results['explainer'] = explainer
            results['X_sample'] = X_sample[:n_lime_samples]
            results['feature_names'] = feature_names
            
            # Ekstrak feature importance dari LIME
            feature_importance = {}
            for exp in lime_explanations:
                for feature, weight in exp.as_list():
                    # Ekstrak nama fitur dari string LIME
                    feature_name = feature.split('=')[0].strip()
                    if feature_name in feature_importance:
                        feature_importance[feature_name] += abs(weight)
                    else:
                        feature_importance[feature_name] = abs(weight)
            
            # Rata-ratakan importance
            for feature in feature_importance:
                feature_importance[feature] /= len(lime_explanations)
            
            results['feature_importance'] = feature_importance
            
        except ImportError:
            raise ImportError("LIME tidak terinstal. Silakan install dengan: pip install lime")
        except Exception as e:
            raise Exception(f"Error dalam menghitung LIME explanations: {str(e)}")
    
    return results

def create_forecasting_interpretation_dashboard(interpretation_results, method='shap'):
    """
    Membuat dashboard visualisasi untuk hasil interpretasi forecasting
    
    Parameters:
    -----------
    interpretation_results : dict
        Hasil dari interpret_forecasting_model()
    method : str, default='shap'
        Metode interpretasi yang digunakan
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi dashboard interpretasi
    """
    
    if method == 'shap':
        return create_shap_forecasting_dashboard(interpretation_results)
    elif method == 'lime':
        return create_lime_forecasting_dashboard(interpretation_results)
    else:
        raise ValueError("Method harus 'shap' atau 'lime'")

def create_shap_forecasting_dashboard(results):
    """
    Membuat dashboard SHAP untuk forecasting
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature Importance
    ax1 = plt.subplot(2, 2, 1)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    ax1.barh(features, importance)
    ax1.set_xlabel('Mean |SHAP Value|')
    ax1.set_title('Feature Importance (Forecasting)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: SHAP Summary Plot (jika tersedia)
    ax2 = plt.subplot(2, 2, 2)
    if 'summary_plot' in results:
        # Tampilkan summary plot yang sudah dibuat
        ax2.text(0.5, 0.5, 'Summary plot tersedia di objek results', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('SHAP Summary Plot')
    else:
        # Buat summary plot sederhana
        shap_values = results['shap_values']
        X_sample = results['X_sample']
        feature_names = results['feature_names']
        
        # Plot rata-rata absolute SHAP values
        if len(shap_values.shape) == 2:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values))
        
        ax2.bar(range(len(feature_names)), mean_shap)
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45)
        ax2.set_ylabel('Mean |SHAP Value|')
        ax2.set_title('SHAP Feature Impact')
    
    # Plot 3: SHAP Values Distribution
    ax3 = plt.subplot(2, 2, 3)
    shap_values = results['shap_values']
    
    if len(shap_values.shape) == 2:
        # Flatten SHAP values untuk distribusi
        flat_shap = shap_values.flatten()
        ax3.hist(flat_shap, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('SHAP Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of SHAP Values')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Distribusi SHAP tidak tersedia\nuntuk bentuk data ini', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # Plot 4: Sample Interpretation
    ax4 = plt.subplot(2, 2, 4)
    X_sample = results['X_sample']
    feature_names = results['feature_names']
    
    # Tampilkan interpretasi untuk sampel pertama
    if len(X_sample) > 0:
        sample_idx = 0
        sample_features = X_sample[sample_idx]
        
        # Buat bar plot untuk nilai fitur sampel
        ax4.bar(range(len(feature_names)), sample_features)
        ax4.set_xticks(range(len(feature_names)))
        ax4.set_xticklabels(feature_names, rotation=45)
        ax4.set_ylabel('Feature Value')
        ax4.set_title(f'Sample Features (Index {sample_idx})')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_lime_forecasting_dashboard(results):
    """
    Membuat dashboard LIME untuk forecasting
    """
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Feature Importance
    ax1 = plt.subplot(2, 2, 1)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]
    
    ax1.barh(features, importance)
    ax1.set_xlabel('Mean |LIME Weight|')
    ax1.set_title('Feature Importance (LIME)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: LIME Explanation untuk sampel pertama
    ax2 = plt.subplot(2, 2, 2)
    lime_explanations = results['lime_explanations']
    
    if lime_explanations:
        first_exp = lime_explanations[0]
        features, weights = zip(*first_exp.as_list())
        
        # Buat horizontal bar plot
        y_pos = np.arange(len(features))
        colors = ['red' if w < 0 else 'green' for w in weights]
        
        ax2.barh(y_pos, weights, color=colors, alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(features)
        ax2.set_xlabel('LIME Weight')
        ax2.set_title('LIME Explanation (Sample 0)')
        ax2.grid(True, alpha=0.3)
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: Distribusi bobot LIME
    ax3 = plt.subplot(2, 2, 3)
    all_weights = []
    for exp in lime_explanations:
        for _, weight in exp.as_list():
            all_weights.append(weight)
    
    ax3.hist(all_weights, bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('LIME Weight')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of LIME Weights')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Plot 4: Rata-rata absolute importance per fitur
    ax4 = plt.subplot(2, 2, 4)
    feature_importance = results['feature_importance']
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())
    
    ax4.bar(range(len(features)), importance)
    ax4.set_xticks(range(len(features)))
    ax4.set_xticklabels(features, rotation=45)
    ax4.set_ylabel('Mean |LIME Weight|')
    ax4.set_title('Average Feature Importance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def analyze_trend_seasonality_cycle(timeseries, period=None):
    """
    Menganalisis tren, seasonality, dan siklus dalam data timeseries
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    period : int, optional
        Periode seasonal (jika diketahui)
        
    Returns:
    --------
    dict
        Hasil analisis tren, seasonality, dan siklus
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    if len(clean_timeseries) < 4:
        return {
            'trend_detected': False,
            'seasonality_detected': False,
            'cycle_detected': False,
            'trend_strength': 0,
            'seasonality_strength': 0,
            'cycle_strength': 0,
            'message': 'Data terlalu sedikit untuk analisis tren, seasonality, dan siklus'
        }
    
    try:
        # Gunakan periode default jika tidak ditentukan
        if period is None:
            # Estimasi periode berdasarkan panjang data
            period = min(12, len(clean_timeseries) // 4)
            if period < 2:
                period = 2
        
        # Decompose time series
        decomposition = seasonal_decompose(clean_timeseries, model='additive', period=period)
        
        # Hitung kekuatan tren, seasonality, dan siklus
        trend_strength = np.var(decomposition.trend.dropna()) / np.var(clean_timeseries)
        seasonal_strength = np.var(decomposition.seasonal.dropna()) / np.var(clean_timeseries)
        residual_strength = np.var(decomposition.resid.dropna()) / np.var(clean_timeseries)
        
        # Deteksi tren menggunakan regresi linear
        x = np.arange(len(clean_timeseries))
        slope, _ = np.polyfit(x, clean_timeseries.values, 1)
        trend_detected = abs(slope) > 0.01 * np.std(clean_timeseries)
        
        # Deteksi seasonality berdasarkan kekuatan seasonal
        seasonality_detected = seasonal_strength > 0.1
        
        # Deteksi siklus menggunakan periodogram
        frequencies, power_spectrum = periodogram(clean_timeseries)
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1  # Skip frequency 0
        dominant_period = 1 / frequencies[dominant_freq_idx] if frequencies[dominant_freq_idx] > 0 else None
        
        # Deteksi siklus berdasarkan periode dominan yang signifikan
        cycle_detected = False
        if dominant_period is not None and dominant_period > period:
            max_power = power_spectrum[dominant_freq_idx]
            avg_power = np.mean(power_spectrum)
            cycle_detected = max_power > 3 * avg_power  # Threshold untuk deteksi siklus
        
        return {
            'trend_detected': trend_detected,
            'seasonality_detected': seasonality_detected,
            'cycle_detected': cycle_detected,
            'trend_strength': float(trend_strength),
            'seasonality_strength': float(seasonality_strength),
            'cycle_strength': float(residual_strength),
            'trend_slope': float(slope),
            'dominant_cycle_period': float(dominant_period) if dominant_period else None,
            'decomposition': decomposition,
            'message': None
        }
        
    except Exception as e:
        return {
            'trend_detected': False,
            'seasonality_detected': False,
            'cycle_detected': False,
            'trend_strength': 0,
            'seasonality_strength': 0,
            'cycle_strength': 0,
            'message': f'Error dalam analisis: {str(e)}'
        }

def plot_pattern_analysis(timeseries, period=None, figsize=(20, 16)):
    """
    Membuat visualisasi komprehensif untuk analisis pola timeseries
    
    Parameters:
    -----------
    timeseries : pandas.Series
        Data timeseries yang akan dianalisis
    period : int, optional
        Periode seasonal
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi analisis pola lengkap
    """
    # Hapus nilai missing
    clean_timeseries = timeseries.dropna()
    
    if len(clean_timeseries) < 4:
        fig, ax = plt.subplots(1, 1, figsize=(15, 4))
        ax.plot(clean_timeseries)
        ax.set_title('Data Timeseries (Data terlalu sedikit untuk analisis pola)')
        plt.tight_layout()
        return fig
    
    # Gunakan periode default jika tidak ditentukan
    if period is None:
        period = min(12, len(clean_timeseries) // 4)
        if period < 2:
            period = 2
    
    # Buat figure dengan subplots
    fig = plt.figure(figsize=figsize)
    
    # Subplot 1: Original data with trend line
    ax1 = plt.subplot(4, 2, 1)
    ax1.plot(clean_timeseries.index, clean_timeseries.values, label='Original Data', alpha=0.7)
    
    # Add trend line
    x_numeric = np.arange(len(clean_timeseries))
    z = np.polyfit(x_numeric, clean_timeseries.values, 1)
    p = np.poly1d(z)
    ax1.plot(clean_timeseries.index, p(x_numeric), "r--", label=f'Trend (slope: {z[0]:.4f})', linewidth=2)
    ax1.set_title('Original Data with Trend Line')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Seasonal decomposition
    try:
        decomposition = seasonal_decompose(clean_timeseries, model='additive', period=period)
        
        ax2 = plt.subplot(4, 2, 2)
        ax2.plot(decomposition.trend.dropna().index, decomposition.trend.dropna().values, color='red')
        ax2.set_title('Trend Component')
        ax2.grid(True, alpha=0.3)
        
        ax3 = plt.subplot(4, 2, 3)
        ax3.plot(decomposition.seasonal.dropna().index, decomposition.seasonal.dropna().values, color='green')
        ax3.set_title('Seasonal Component')
        ax3.grid(True, alpha=0.3)
        
        ax4 = plt.subplot(4, 2, 4)
        ax4.plot(decomposition.resid.dropna().index, decomposition.resid.dropna().values, color='orange')
        ax4.set_title('Residual Component')
        ax4.grid(True, alpha=0.3)
        
    except Exception as e:
        ax2 = plt.subplot(4, 2, 2)
        ax2.text(0.5, 0.5, f'Error in decomposition:\n{str(e)}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes)
        ax2.set_title('Decomposition Error')
    
    # Subplot 5: Seasonal subseries plot
    ax5 = plt.subplot(4, 2, 5)
    if period <= len(clean_timeseries):
        seasonal_data = []
        for i in range(period):
            seasonal_values = clean_timeseries.iloc[i::period]
            if len(seasonal_values) > 0:
                seasonal_data.append(seasonal_values.values)
        
        if seasonal_data:
            ax5.boxplot(seasonal_data, labels=[f'P{i+1}' for i in range(len(seasonal_data))])
            ax5.set_title(f'Seasonal Pattern (Period={period})')
            ax5.set_xlabel('Seasonal Period')
            ax5.set_ylabel('Value')
    
    # Subplot 6: Year-over-year growth (if applicable)
    ax6 = plt.subplot(4, 2, 6)
    if len(clean_timeseries) >= 365:  # At least one year of data
        try:
            # Calculate year-over-year growth
            yoy_growth = clean_timeseries.pct_change(periods=365).dropna() * 100
            ax6.plot(yoy_growth.index, yoy_growth.values, color='purple')
            ax6.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            ax6.set_title('Year-over-Year Growth Rate (%)')
            ax6.grid(True, alpha=0.3)
        except:
            ax6.text(0.5, 0.5, 'YoY growth not applicable', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax6.transAxes)
    
    # Subplot 7: Moving averages
    ax7 = plt.subplot(4, 2, 7)
    window_short = min(7, len(clean_timeseries) // 4)
    window_long = min(30, len(clean_timeseries) // 2)
    
    if window_short >= 2:
        short_ma = clean_timeseries.rolling(window=window_short).mean()
        ax7.plot(clean_timeseries.index, clean_timeseries.values, alpha=0.5, label='Original')
        ax7.plot(short_ma.index, short_ma.values, label=f'MA-{window_short}', linewidth=2)
        
        if window_long > window_short:
            long_ma = clean_timeseries.rolling(window=window_long).mean()
            ax7.plot(long_ma.index, long_ma.values, label=f'MA-{window_long}', linewidth=2)
        
        ax7.set_title('Moving Averages')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # Subplot 8: Distribution by time period
    ax8 = plt.subplot(4, 2, 8)
    if hasattr(clean_timeseries.index, 'month'):
        # Group by month
        monthly_data = clean_timeseries.groupby(clean_timeseries.index.month)
        monthly_means = monthly_data.mean()
        monthly_stds = monthly_data.std()
        
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        ax8.bar(range(1, 13), [monthly_means.get(i, 0) for i in range(1, 13)], 
                yerr=[monthly_stds.get(i, 0) for i in range(1, 13)],
                capsize=5)
        ax8.set_xticks(range(1, 13))
        ax8.set_xticklabels(months)
        ax8.set_title('Seasonal Distribution by Month')
        ax8.set_ylabel('Mean Value')
    
    plt.tight_layout()
    return fig

def create_simplified_timeseries_plot(clean_timeseries, figsize=(12, 8)):
    """
    Create simplified timeseries plot for small datasets
    """
    if len(clean_timeseries) < 5:
        # Plot data asli dan ACF saja jika data cukup untuk ACF tapi tidak untuk PACF
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot data asli
        axes[0].plot(clean_timeseries)
        axes[0].set_title('Data Timeseries')
        
        # Plot ACF
        try:
            plot_acf(clean_timeseries, ax=axes[1])
            axes[1].set_title('Autocorrelation Function (ACF)')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error plotting ACF: {str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ACF - Error')
        
        plt.tight_layout()
        return fig

def create_full_timeseries_plot(clean_timeseries, figsize=(12, 8)):
    """
    Create full timeseries plot with ACF and PACF
    """
    if len(clean_timeseries) >= 5:
        # Plot lengkap jika data cukup
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # Plot data asli
        axes[0].plot(clean_timeseries)
        axes[0].set_title('Data Timeseries')
        
        # Plot ACF
        try:
            plot_acf(clean_timeseries, ax=axes[1])
            axes[1].set_title('Autocorrelation Function (ACF)')
        except Exception as e:
            axes[1].text(0.5, 0.5, f'Error plotting ACF: {str(e)}', 
                        ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_title('ACF - Error')
        
        # Plot PACF
        try:
            plot_pacf(clean_timeseries, ax=axes[2])
            axes[2].set_title('Partial Autocorrelation Function (PACF)')
        except Exception as e:
            axes[2].text(0.5, 0.5, f'Error plotting PACF: {str(e)}', 
                        ha='center', va='center', transform=axes[2].transAxes)
            axes[2].set_title('PACF - Error')
        
        plt.tight_layout()
        return fig

def create_features_from_date(df, date_column):
    """
    Membuat fitur-fitur dari kolom tanggal (tahun, bulan, hari, dll.)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    date_column : str
        Nama kolom tanggal
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur tambahan dari tanggal
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Konversi kolom tanggal ke datetime jika belum
    if not pd.api.types.is_datetime64_any_dtype(df_new[date_column]):
        df_new[date_column] = pd.to_datetime(df_new[date_column])
    
    # Ekstrak fitur dari tanggal
    df_new['year'] = df_new[date_column].dt.year
    df_new['month'] = df_new[date_column].dt.month
    df_new['day'] = df_new[date_column].dt.day
    df_new['dayofweek'] = df_new[date_column].dt.dayofweek
    df_new['quarter'] = df_new[date_column].dt.quarter
    df_new['is_month_start'] = df_new[date_column].dt.is_month_start.astype(int)
    df_new['is_month_end'] = df_new[date_column].dt.is_month_end.astype(int)
    
    return df_new

def check_model_compatibility(model, method='shap', language='id'):
    """
    Mengecek kompatibilitas model dengan metode interpretasi SHAP atau LIME
    
    Parameters:
    -----------
    model : object
        Model yang akan dicek kompatibilitasnya
    method : str, optional
        Metode interpretasi ('shap' atau 'lime')
    language : str, optional
        Bahasa untuk pesan error ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi:
        - 'compatible': Boolean apakah model kompatibel
        - 'message': Pesan penjelasan
        - 'suggestion': Saran alternatif
        - 'error_type': Tipe error jika tidak kompatibel
    """
    
    # Pesan error dalam berbagai bahasa
    messages = {
        'id': {
            'unsupported_model': "Model ini belum didukung untuk interpretasi {method}.",
            'ensemble_not_supported': "Model ensemble kompleks belum didukung untuk interpretasi {method}.",
            'neural_network': "Model neural network memerlukan pendekatan khusus untuk interpretasi.",
            'deep_learning': "Model deep learning memerlukan library khusus seperti DeepSHAP atau Integrated Gradients.",
            'clustering': "Model clustering bukan model prediktif dan tidak dapat diinterpretasi dengan {method}.",
            'dimensionality_reduction': "Model reduksi dimensi bukan model prediktif dan tidak dapat diinterpretasi dengan {method}.",
            'suggestion': "\n\nSaran:\n",
            'alternatives': "- Gunakan model yang lebih sederhana (Random Forest, XGBoost, LightGBM)\n",
            'lime_alternative': "- Cobalah metode LIME jika SHAP tidak didukung\n",
            'shap_alternative': "- Cobalah metode SHAP jika LIME tidak didukung\n",
            'library_suggestion': "- Install library tambahan: {library}\n",
            'expert_help': "- Konsultasikan dengan tim data science untuk pendekatan interpretasi yang sesuai"
        },
        'en': {
            'unsupported_model': "This model is not yet supported for {method} interpretation.",
            'ensemble_not_supported': "Complex ensemble models are not yet supported for {method} interpretation.",
            'neural_network': "Neural network models require special approaches for interpretation.",
            'deep_learning': "Deep learning models require specialized libraries like DeepSHAP or Integrated Gradients.",
            'clustering': "Clustering models are not predictive models and cannot be interpreted with {method}.",
            'dimensionality_reduction': "Dimensionality reduction models are not predictive models and cannot be interpreted with {method}.",
            'suggestion': "\n\nSuggestions:\n",
            'alternatives': "- Use simpler models (Random Forest, XGBoost, LightGBM)\n",
            'lime_alternative': "- Try LIME method if SHAP is not supported\n",
            'shap_alternative': "- Try SHAP method if LIME is not supported\n",
            'library_suggestion': "- Install additional library: {library}\n",
            'expert_help': "- Consult with data science team for appropriate interpretation approach"
        }
    }
    
    lang = messages.get(language, messages['id'])
    
    try:
        # Cek apakah model memiliki method yang diperlukan
        if not hasattr(model, 'predict'):
            return {
                'compatible': False,
                'message': f"{lang['unsupported_model'].format(method=method.upper())} Model tidak memiliki method 'predict'.",
                'suggestion': f"{lang['suggestion']}{lang['alternatives']}{lang['expert_help']}",
                'error_type': 'missing_predict_method'
            }
        
        # Dapatkan tipe model
        model_type = type(model).__name__.lower()
        module_name = type(model).__module__.lower()
        
        # Model yang jelas tidak didukung
        unsupported_models = [
            'kmeans', 'dbscan', 'hierarchical', 'agglomerative',  # Clustering
            'pca', 'tsne', 'umap', 'lda', 'nmf',  # Dimensionality reduction
            'isolationforest', 'oneclasssvm', 'localoutlierfactor'  # Anomaly detection
        ]
        
        for unsupported in unsupported_models:
            if unsupported in model_type or unsupported in module_name:
                if 'cluster' in unsupported:
                    error_msg = lang['clustering']
                elif unsupported in ['pca', 'tsne', 'umap', 'lda', 'nmf']:
                    error_msg = lang['dimensionality_reduction']
                else:
                    error_msg = lang['unsupported_model'].format(method=method.upper())
                    
                return {
                    'compatible': False,
                    'message': error_msg,
                    'suggestion': f"{lang['suggestion']}{lang['alternatives']}{lang['expert_help']}",
                    'error_type': 'unsupported_model_type'
                }
        
        # Cek untuk neural networks dan deep learning
        neural_networks = ['mlp', 'neural', 'dense', 'lstm', 'gru', 'cnn', 'rnn']
        if any(nn in model_type for nn in neural_networks) or 'keras' in module_name or 'torch' in module_name:
            if method == 'shap':
                return {
                    'compatible': False,
                    'message': lang['deep_learning'] if 'deep' in model_type else lang['neural_network'],
                    'suggestion': f"{lang['suggestion']}{lang['library_suggestion'].format(library='shap[deep] atau deepexplain')}{lang['expert_help']}",
                    'error_type': 'neural_network'
                }
            else:  # LIME
                return {
                    'compatible': True,
                    'message': f"Model neural network dapat diinterpretasi dengan LIME, namun hasil mungkin terbatas.",
                    'suggestion': "- Pastikan data input dalam format yang tepat\n- Perhatikan interpretasi secara hati-hati",
                    'error_type': None
                }
        
        # Cek untuk ensemble kompleks
        complex_ensembles = ['voting', 'stacking', 'blending']
        if any(ensemble in model_type for ensemble in complex_ensembles):
            return {
                'compatible': False,
                'message': lang['ensemble_not_supported'].format(method=method.upper()),
                'suggestion': f"{lang['suggestion']}{lang['alternatives']}{lang['expert_help']}",
                'error_type': 'complex_ensemble'
            }
        
        # Model yang umumnya didukung
        supported_models = [
            'randomforest', 'xgb', 'lgbm', 'catboost',  # Tree-based
            'linear', 'logistic', 'ridge', 'lasso', 'elastic',  # Linear
            'svm', 'svc', 'svr',  # SVM
            'knn', 'nearest',  # KNN
            'decisiontree', 'extratree'  # Tree models
        ]
        
        # Cek apakah model termasuk dalam kategori yang didukung
        is_supported = any(supported in model_type for supported in supported_models)
        
        if is_supported:
            # Berikan saran berdasarkan metode
            suggestions = []
            if method == 'shap' and ('tree' in model_type or 'forest' in model_type or 'xgb' in model_type or 'lgbm' in model_type):
                suggestions.append("- Gunakan TreeExplainer untuk hasil yang lebih cepat dan akurat")
            elif method == 'lime':
                suggestions.append("- Pastikan data dalam format yang dapat diinterpretasi oleh LIME")
            
            return {
                'compatible': True,
                'message': f"Model {model_type} didukung untuk interpretasi {method.upper()}.",
                'suggestion': '\n'.join(suggestions) if suggestions else "",
                'error_type': None
            }
        
        # Default: model tidak dikenali
        return {
            'compatible': False,
            'message': f"{lang['unsupported_model'].format(method=method.upper())} Tipe model: {model_type}",
            'suggestion': f"{lang['suggestion']}{lang['alternatives']}{lang['shap_alternative'] if method == 'lime' else lang['lime_alternative']}{lang['expert_help']}",
            'error_type': 'unknown_model_type'
        }
        
    except Exception as e:
        return {
            'compatible': False,
            'message': f"Error saat mengecek kompatibilitas model: {str(e)}",
            'suggestion': f"{lang['suggestion']}{lang['expert_help']}",
            'error_type': 'compatibility_check_error'
        }

def get_model_interpretation_recommendations(model, language='id'):
    """
    Memberikan rekomendasi metode interpretasi yang sesuai untuk model
    
    Parameters:
    -----------
    model : object
        Model yang akan direkomendasikan metode interpretasinya
    language : str, optional
        Bahasa untuk pesan ('id' atau 'en')
        
    Returns:
    --------
    dict
        Dictionary berisi rekomendasi interpretasi
    """
    
    lang = language if language in ['id', 'en'] else 'id'
    
    # Dapatkan tipe model
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    
    # Rekomendasi berdasarkan tipe model
    recommendations = {
        'tree_based': {
            'primary': 'SHAP (TreeExplainer)',
            'secondary': 'LIME',
            'reason': 'Model berbasis pohon sangat cocok dengan SHAP TreeExplainer karena cepat dan akurat',
            'alternatives': ['Feature Importance', 'Permutation Importance']
        },
        'linear': {
            'primary': 'SHAP (LinearExplainer)',
            'secondary': 'LIME',
            'reason': 'Model linear memiliki interpretasi yang sangat baik dengan SHAP LinearExplainer',
            'alternatives': ['Coefficients', 'Feature Importance']
        },
        'neural_network': {
            'primary': 'LIME',
            'secondary': 'SHAP (DeepSHAP)',
            'reason': 'Neural network lebih cocok dengan LIME untuk interpretasi lokal',
            'alternatives': ['Integrated Gradients', 'Grad-CAM']
        },
        'ensemble': {
            'primary': 'SHAP',
            'secondary': 'LIME',
            'reason': 'Ensemble model dapat diinterpretasi dengan SHAP untuk hasil global yang baik',
            'alternatives': ['Feature Importance', 'Permutation Importance']
        },
        'unsupported': {
            'primary': 'Tidak tersedia',
            'secondary': 'Tidak tersedia',
            'reason': 'Model ini tidak cocok untuk interpretasi otomatis',
            'alternatives': ['Analisis manual', 'Visualisasi data']
        }
    }
    
    # Kategorikan model
    if any(tree in model_type for tree in ['randomforest', 'xgb', 'lgbm', 'catboost', 'decisiontree', 'extratree']):
        category = 'tree_based'
    elif any(linear in model_type for linear in ['linear', 'logistic', 'ridge', 'lasso', 'elastic']):
        category = 'linear'
    elif any(nn in model_type for nn in ['mlp', 'neural', 'dense', 'lstm', 'gru']) or 'keras' in module_name or 'torch' in module_name:
        category = 'neural_network'
    elif any(ensemble in model_type for ensemble in ['voting', 'bagging', 'boosting']):
        category = 'ensemble'
    elif any(unsupported in model_type for unsupported in ['kmeans', 'pca', 'tsne']):
        category = 'unsupported'
    else:
        category = 'ensemble'  # Default ke ensemble untuk model yang tidak dikenal
    
    rec = recommendations[category]
    
    if lang == 'id':
        return {
            'primary_method': rec['primary'],
            'secondary_method': rec['secondary'],
            'reason': rec['reason'],
            'alternatives': rec['alternatives'],
            'explanation': f"Model Anda tergolong {category.replace('_', ' ')}. "
                          f"Metode yang paling direkomendasikan adalah {rec['primary']}. "
                          f"Alasan: {rec['reason']}"
        }
    else:
        return {
            'primary_method': rec['primary'],
            'secondary_method': rec['secondary'],
            'reason': rec['reason'],
            'alternatives': rec['alternatives'],
            'explanation': f"Your model is categorized as {category.replace('_', ' ')}. "
                          f"The most recommended method is {rec['primary']}. "
                          f"Reason: {rec['reason']}"
        }

def implement_shap_classification(model, X_sample, problem_type='binary', class_names=None, feature_names=None):
    """
    Implementasi SHAP untuk model klasifikasi dengan penanganan multi-class
    
    Parameters:
    -----------
    model : sklearn model
        Model yang telah dilatih
    X_sample : pandas.DataFrame
        Sample data untuk interpretasi
    problem_type : str, optional
        'binary' atau 'multiclass'
    class_names : list, optional
        Nama kelas untuk klasifikasi
    feature_names : list, optional
        Nama fitur
        
    Returns:
    --------
    dict
        Dictionary berisi SHAP values dan informasi interpretasi
    """
    import shap
    
    result = {
        'shap_values': None,
        'expected_value': None,
        'explainer': None,
        'problem_type': problem_type,
        'class_names': class_names,
        'feature_names': feature_names or X_sample.columns.tolist()
    }
    
    try:
        # Tentukan jenis explainer berdasarkan model
        if hasattr(model, 'tree_'):  # Tree-based models
            explainer = shap.TreeExplainer(model)
        else:  # General models
            explainer = shap.KernelExplainer(model.predict_proba, X_sample)
        
        # Hitung SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Penanganan untuk berbagai jenis output
        if problem_type == 'binary':
            # Binary classification: SHAP values untuk kelas positif
            if isinstance(shap_values, list) and len(shap_values) == 2:
                result['shap_values'] = shap_values[1]  # Kelas positif
                result['expected_value'] = explainer.expected_value[1] if hasattr(explainer, 'expected_value') else None
            else:
                result['shap_values'] = shap_values
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
                
        elif problem_type == 'multiclass':
            # Multi-class classification
            if isinstance(shap_values, list):
                result['shap_values'] = shap_values  # List of SHAP values untuk setiap kelas
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
            else:
                result['shap_values'] = shap_values
                result['expected_value'] = explainer.expected_value if hasattr(explainer, 'expected_value') else None
        
        result['explainer'] = explainer
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        
    return result

def handle_multiclass_shap(shap_values, predicted_class=None, method='individual'):
    """
    Penanganan SHAP values untuk multi-class classification
    
    Parameters:
    -----------
    shap_values : list or array
        SHAP values dari model multi-class
    predicted_class : int, optional
        Kelas yang diprediksi untuk focus
    method : str, optional
        'individual' (fokus pada kelas tertentu) atau 'average' (rata-rata semua kelas)
        
    Returns:
    --------
    dict
        Dictionary berisi processed SHAP values
    """
    result = {}
    
    if isinstance(shap_values, list):
        n_classes = len(shap_values)
        
        if method == 'individual' and predicted_class is not None:
            # Fokus pada kelas yang diprediksi
            result['shap_values_focused'] = shap_values[predicted_class]
            result['class_focused'] = predicted_class
            result['method'] = 'individual'
            
        elif method == 'average':
            # Rata-rata absolute SHAP values untuk semua kelas
            avg_shap = np.mean([np.abs(values) for values in shap_values], axis=0)
            result['shap_values_average'] = avg_shap
            result['method'] = 'average'
            result['n_classes'] = n_classes
            
        else:
            # Simpan semua SHAP values
            result['shap_values_all'] = shap_values
            result['n_classes'] = n_classes
            result['method'] = 'all'
            
    else:
        result['shap_values'] = shap_values
        result['method'] = 'single'
        
    return result

def implement_lime_classification(model, X_sample, y_sample=None, problem_type='binary', 
                                class_names=None, feature_names=None, num_features=10):
    """
    Implementasi LIME untuk model klasifikasi
    
    Parameters:
    -----------
    model : sklearn model
        Model yang telah dilatih
    X_sample : pandas.DataFrame
        Sample data untuk interpretasi
    y_sample : array, optional
        Target untuk sample data
    problem_type : str, optional
        'binary' atau 'multiclass'
    class_names : list, optional
        Nama kelas untuk klasifikasi
    feature_names : list, optional
        Nama fitur
    num_features : int, optional
        Jumlah fitur terpenting yang akan ditampilkan
        
    Returns:
    --------
    dict
        Dictionary berisi LIME explanations dan informasi interpretasi
    """
    try:
        import lime
        import lime.lime_tabular
        
        result = {
            'explanations': [],
            'problem_type': problem_type,
            'class_names': class_names,
            'feature_names': feature_names or X_sample.columns.tolist(),
            'num_features': num_features
        }
        
        # Buat LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample.values,
            feature_names=result['feature_names'],
            class_names=class_names,
            mode='classification'
        )
        
        # Fungsi prediksi untuk LIME
        if problem_type == 'binary':
            predict_fn = lambda x: model.predict_proba(x)[:, 1]  # Probabilitas kelas positif
        else:
            predict_fn = model.predict_proba  # Semua probabilitas untuk multi-class
        
        # Generate explanations untuk beberapa sample
        n_samples = min(5, len(X_sample))  # Maksimal 5 sample
        
        for i in range(n_samples):
            explanation = explainer.explain_instance(
                X_sample.iloc[i].values,
                predict_fn,
                num_features=num_features,
                top_labels=1 if problem_type == 'binary' else len(class_names)
            )
            
            result['explanations'].append({
                'sample_index': i,
                'predicted_class': model.predict(X_sample.iloc[i:i+1])[0],
                'predicted_proba': model.predict_proba(X_sample.iloc[i:i+1])[0],
                'explanation': explanation
            })
        
        result['explainer'] = explainer
        result['predict_fn'] = predict_fn
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        result['success'] = False
        
    return result

def detect_model_type(model):
    """
    Mendeteksi jenis model (classification, regression, atau forecasting)
    
    Parameters:
    -----------
    model : sklearn/statsmodels model
        Model yang akan dideteksi
        
    Returns:
    --------
    str
        'classification', 'regression', atau 'forecasting'
    """
    # Cek apakah model memiliki classes_ (indikator classification)
    if hasattr(model, 'classes_'):
        return 'classification'
    
    # Cek apakah model memiliki predict_proba (indikator classification)
    if hasattr(model, 'predict_proba'):
        return 'classification'
    
    # Cek model forecasting (ARIMA, SARIMA, dsb)
    model_name = type(model).__name__.lower()
    forecasting_models = ['arima', 'sarima', 'sarimax', 'holtwinters', 'exponentialsmoothing', 'var', 'vecm']
    if any(fm in model_name for fm in forecasting_models):
        return 'forecasting'
    
    # Default ke regression
    return 'regression'

def prepare_forecasting_data_for_interpretation(X_train, X_test, selected_features, sample_idx):
    """
    Mempersiapkan data forecasting untuk interpretasi SHAP/LIME
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Data training
    X_test : pandas.DataFrame
        Data testing  
    selected_features : list
        Daftar fitur yang dipilih
    sample_idx : int
        Indeks sampel yang akan dijelaskan
        
    Returns:
    --------
    dict
        Dictionary berisi data yang siap untuk interpretasi
    """
    try:
        # Filter data berdasarkan fitur yang dipilih
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
        
        # Pastikan sample_idx valid
        if sample_idx >= len(X_test_selected):
            sample_idx = len(X_test_selected) - 1
        
        # Ambil sampel yang akan dijelaskan
        sample = X_test_selected.iloc[sample_idx]
        
        result = {
            'X_train': X_train_selected,
            'X_test': X_test_selected,
            'sample': sample,
            'sample_idx': sample_idx,
            'success': True
        }
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def validate_data_for_ml(df, target_column=None, feature_columns=None, 
                        max_missing_ratio=0.5, min_samples=10, 
                        handle_missing='auto', verbose=True):
    """
    Validasi dan membersihkan data untuk machine learning dengan pendekatan yang konsisten
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang akan divalidasi
    target_column : str, optional
        Nama kolom target (jika ada)
    feature_columns : list, optional
        Daftar kolom fitur yang akan digunakan (jika None, gunakan semua kolom numerik)
    max_missing_ratio : float, optional
        Rasio missing value maksimum yang diizinkan per kolom (default: 0.5)
    min_samples : int, optional
        Jumlah minimum sampel yang dibutuhkan (default: 10)
    handle_missing : str, optional
        Metode penanganan missing value ('auto', 'drop', 'impute', 'none')
    verbose : bool, optional
        Apakah akan menampilkan informasi proses validasi
        
    Returns:
    --------
    dict
        Dictionary berisi:
        - 'data': DataFrame yang sudah divalidasi dan dibersihkan
        - 'target_column': Nama kolom target (jika ada)
        - 'feature_columns': Daftar kolom fitur yang digunakan
        - 'missing_handled': Informasi bagaimana missing value ditangani
        - 'warnings': Daftar peringatan jika ada
        - 'errors': Daftar error jika ada
    """
    
    result = {
        'data': None,
        'target_column': target_column,
        'feature_columns': feature_columns,
        'missing_handled': {},
        'warnings': [],
        'errors': []
    }
    
    # Validasi input
    if df is None or df.empty:
        result['errors'].append("DataFrame kosong atau None")
        return result
    
    if len(df) < min_samples:
        result['errors'].append(f"Jumlah sampel ({len(df)}) kurang dari minimum ({min_samples})")
        return result
    
    # Buat salinan data
    df_clean = df.copy()
    
    # Identifikasi kolom fitur jika tidak ditentukan
    if feature_columns is None:
        # Gunakan semua kolom numerik kecuali target column
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_column and target_column in numeric_columns:
            feature_columns = [col for col in numeric_columns if col != target_column]
        else:
            feature_columns = numeric_columns
    
    result['feature_columns'] = feature_columns
    
    # Validasi target column jika ada
    if target_column and target_column not in df_clean.columns:
        result['errors'].append(f"Kolom target '{target_column}' tidak ditemukan")
        return result
    
    # Validasi feature columns
    missing_features = [col for col in feature_columns if col not in df_clean.columns]
    if missing_features:
        result['errors'].append(f"Kolom fitur tidak ditemukan: {missing_features}")
        return result
    
    # Analisis missing values
    columns_to_check = feature_columns.copy()
    if target_column:
        columns_to_check.append(target_column)
    
    missing_analysis = df_clean[columns_to_check].isnull().sum()
    missing_ratio = missing_analysis / len(df_clean)
    
    if verbose:
        print("Analisis Missing Values:")
        for col in columns_to_check:
            ratio = missing_ratio[col]
            count = missing_analysis[col]
            print(f"  {col}: {count} missing ({ratio:.2%})")
    
    # Identifikasi kolom dengan terlalu banyak missing values
    high_missing_cols = missing_ratio[missing_ratio > max_missing_ratio].index.tolist()
    
    if high_missing_cols:
        result['warnings'].append(f"Kolom dengan missing > {max_missing_ratio:.0%}: {high_missing_cols}")
        
        if handle_missing == 'auto':
            # Drop kolom dengan terlalu banyak missing
            df_clean = df_clean.drop(columns=high_missing_cols)
            result['missing_handled']['dropped_columns'] = high_missing_cols
            
            # Update feature columns
            feature_columns = [col for col in feature_columns if col not in high_missing_cols]
            result['feature_columns'] = feature_columns
            
            if verbose:
                print(f"Menghapus kolom dengan terlalu banyak missing: {high_missing_cols}")
    
    # Tangani missing values pada kolom yang tersisa
    if handle_missing == 'auto':
        # Gunakan strategi yang sesuai untuk setiap tipe data
        for col in feature_columns:
            if df_clean[col].dtype in ['float64', 'int64']:
                # Untuk data numerik, gunakan median
                if df_clean[col].isnull().any():
                    median_val = df_clean[col].median()
                    df_clean[col] = df_clean[col].fillna(median_val)
                    result['missing_handled'][col] = f'filled_with_median({median_val:.2f})'
                    
            elif df_clean[col].dtype == 'object':
                # Untuk data kategorikal, gunakan mode
                if df_clean[col].isnull().any():
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    result['missing_handled'][col] = f'filled_with_mode({mode_val})'
        
        # Tangani missing values pada target column
        if target_column and df_clean[target_column].isnull().any():
            if df_clean[target_column].dtype in ['float64', 'int64']:
                median_val = df_clean[target_column].median()
                df_clean[target_column] = df_clean[target_column].fillna(median_val)
                result['missing_handled'][target_column] = f'filled_with_median({median_val:.2f})'
            else:
                mode_val = df_clean[target_column].mode().iloc[0] if not df_clean[target_column].mode().empty else 'unknown'
                df_clean[target_column] = df_clean[target_column].fillna(mode_val)
                result['missing_handled'][target_column] = f'filled_with_mode({mode_val})'
                
    elif handle_missing == 'drop':
        # Hapus baris dengan missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna(subset=columns_to_check)
        dropped_rows = initial_rows - len(df_clean)
        
        if dropped_rows > 0:
            result['missing_handled']['dropped_rows'] = dropped_rows
            if verbose:
                print(f"Menghapus {dropped_rows} baris dengan missing values")
                
    elif handle_missing == 'impute':
        # Gunakan imputasi yang lebih canggih (mirip dengan yang ada di app.py)
        for col in columns_to_check:
            if df_clean[col].isnull().any():
                if df_clean[col].dtype in ['float64', 'int64']:
                    # Gunakan interpolasi untuk data time series jika memungkinkan
                    try:
                        df_clean[col] = df_clean[col].interpolate(method='linear')
                        df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                        result['missing_handled'][col] = 'interpolated_and_filled'
                    except:
                        # Fallback ke median
                        median_val = df_clean[col].median()
                        df_clean[col] = df_clean[col].fillna(median_val)
                        result['missing_handled'][col] = f'filled_with_median({median_val:.2f})'
                else:
                    mode_val = df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else 'unknown'
                    df_clean[col] = df_clean[col].fillna(mode_val)
                    result['missing_handled'][col] = f'filled_with_mode({mode_val})'
    
    # Validasi final
    final_missing = df_clean[columns_to_check].isnull().sum().sum()
    if final_missing > 0:
        result['warnings'].append(f"Masih ada {final_missing} missing values setelah cleaning")
        if verbose:
            print(f"Peringatan: Masih ada {final_missing} missing values")
    
    # Validasi ukuran data akhir
    if len(df_clean) < min_samples:
        result['errors'].append(f"Jumlah sampel akhir ({len(df_clean)}) kurang dari minimum ({min_samples})")
        return result
    
    # Hasil akhir
    result['data'] = df_clean
    
    if verbose:
        print(f"\nValidasi selesai:")
        print(f"  Jumlah sampel awal: {len(df)}")
        print(f"  Jumlah sampel akhir: {len(df_clean)}")
        print(f"  Kolom fitur: {len(feature_columns)}")
        if target_column:
            print(f"  Kolom target: {target_column}")
        if result['missing_handled']:
            print(f"  Missing values ditangani: {result['missing_handled']}")
        if result['warnings']:
            print(f"  Peringatan: {result['warnings']}")
    
    return result

def create_lag_features(df, target_column, lag_periods=[1, 2, 3, 7, 14, 30], fill_method='ffill'):
    """
    Membuat fitur lag dari kolom target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    target_column : str
        Nama kolom target
    lag_periods : list, optional
        Daftar periode lag yang akan dibuat
    fill_method : str, optional
        Metode untuk mengisi NaN yang dihasilkan dari lag (ffill, bfill, zero, mean, median)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur lag tambahan
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Buat fitur lag
    for lag in lag_periods:
        df_new[f'{target_column}_lag_{lag}'] = df_new[target_column].shift(lag)
    
    # Tangani NaN yang dihasilkan dari operasi lag
    lag_columns = [f'{target_column}_lag_{lag}' for lag in lag_periods]
    
    if fill_method == 'ffill':
        df_new[lag_columns] = df_new[lag_columns].fillna(method='ffill')
    elif fill_method == 'bfill':
        df_new[lag_columns] = df_new[lag_columns].fillna(method='bfill')
    elif fill_method == 'zero':
        df_new[lag_columns] = df_new[lag_columns].fillna(0)
    elif fill_method == 'mean':
        for col in lag_columns:
            df_new[col] = df_new[col].fillna(df_new[col].mean())
    elif fill_method == 'median':
        for col in lag_columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    else:
        # Default: gunakan forward fill dan backward fill
        df_new[lag_columns] = df_new[lag_columns].fillna(method='ffill').fillna(method='bfill')
    
    return df_new

def create_rolling_features(df, target_column, windows=[7, 14, 30], fill_method='ffill', min_periods=None):
    """
    Membuat fitur rolling (rata-rata, standar deviasi, min, max) dari kolom target
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame yang berisi data
    target_column : str
        Nama kolom target
    windows : list, optional
        Daftar ukuran window untuk rolling
    fill_method : str, optional
        Metode untuk mengisi NaN yang dihasilkan dari rolling (ffill, bfill, zero, mean, median)
    min_periods : int, optional
        Jumlah minimum observasi yang dibutuhkan untuk menghitung rolling statistics
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur rolling tambahan
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Set min_periods jika tidak ditentukan (setengah dari window size)
    if min_periods is None:
        min_periods = 1  # Minimal 1 observasi untuk menghitung
    
    # Buat fitur rolling
    for window in windows:
        df_new[f'{target_column}_rolling_mean_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).mean()
        df_new[f'{target_column}_rolling_std_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).std()
        df_new[f'{target_column}_rolling_min_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).min()
        df_new[f'{target_column}_rolling_max_{window}'] = df_new[target_column].rolling(window=window, min_periods=min_periods).max()
    
    # Tangani NaN yang dihasilkan dari operasi rolling
    rolling_columns = []
    for window in windows:
        rolling_columns.extend([
            f'{target_column}_rolling_mean_{window}',
            f'{target_column}_rolling_std_{window}',
            f'{target_column}_rolling_min_{window}',
            f'{target_column}_rolling_max_{window}'
        ])
    
    if fill_method == 'ffill':
        df_new[rolling_columns] = df_new[rolling_columns].fillna(method='ffill')
    elif fill_method == 'bfill':
        df_new[rolling_columns] = df_new[rolling_columns].fillna(method='bfill')
    elif fill_method == 'zero':
        df_new[rolling_columns] = df_new[rolling_columns].fillna(0)
    elif fill_method == 'mean':
        for col in rolling_columns:
            df_new[col] = df_new[col].fillna(df_new[col].mean())
    elif fill_method == 'median':
        for col in rolling_columns:
            df_new[col] = df_new[col].fillna(df_new[col].median())
    else:
        # Default: gunakan forward fill dan backward fill
        df_new[rolling_columns] = df_new[rolling_columns].fillna(method='ffill').fillna(method='bfill')
    
    return df_new