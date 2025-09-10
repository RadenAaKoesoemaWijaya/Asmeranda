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
        # Handle different aggregation methods based on frequency
        if freq.upper() in ['D', 'W', 'M', 'Y']:
            # For daily, weekly, monthly, yearly - use mean for continuous data
            ts_data = ts_data.resample(freq.upper())[target_column].mean().to_frame()
        elif freq.upper() == 'H':
            # For hourly - use mean
            ts_data = ts_data.resample('H')[target_column].mean().to_frame()
        else:
            # Default resampling
            ts_data = ts_data.resample(freq)[target_column].mean().to_frame()
        
        # Handle missing values after resampling
        if ts_data[target_column].isnull().sum() > 0:
            # Forward fill for missing values, then backward fill
            ts_data[target_column] = ts_data[target_column].fillna(method='ffill').fillna(method='bfill')
            
            # If still missing, use interpolation
            if ts_data[target_column].isnull().sum() > 0:
                ts_data[target_column] = ts_data[target_column].interpolate(method='linear')
    
    # Remove any remaining NaN values
    ts_data = ts_data.dropna()
    
    # Ensure we have enough data points
    if len(ts_data) < 3:
        raise ValueError(f"Data terlalu sedikit setelah resampling ({len(ts_data)} data points). Pastikan data mencukupi untuk frekuensi {freq}")
    
    return ts_data
    
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

def create_lag_features(df, target_column, lag_periods=[1, 2, 3, 7, 14, 30]):
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
    
    return df_new

def create_rolling_features(df, target_column, windows=[7, 14, 30]):
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
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan fitur-fitur rolling tambahan
    """
    # Buat salinan data
    df_new = df.copy()
    
    # Buat fitur rolling
    for window in windows:
        df_new[f'{target_column}_rolling_mean_{window}'] = df_new[target_column].rolling(window=window).mean()
        df_new[f'{target_column}_rolling_std_{window}'] = df_new[target_column].rolling(window=window).std()
        df_new[f'{target_column}_rolling_min_{window}'] = df_new[target_column].rolling(window=window).min()
        df_new[f'{target_column}_rolling_max_{window}'] = df_new[target_column].rolling(window=window).max()
    
    return df_new