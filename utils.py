import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    # Uji Augmented Dickey-Fuller
    result = adfuller(timeseries.dropna())
    
    # Siapkan output
    output = {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations Used': result[3],
        'Critical Values': result[4],
        'Stationary': result[1] <= 0.05
    }
    
    return output

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
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Plot data asli
    axes[0].plot(timeseries)
    axes[0].set_title('Data Timeseries')
    
    # Plot ACF
    plot_acf(timeseries.dropna(), ax=axes[1])
    axes[1].set_title('Autocorrelation Function (ACF)')
    
    # Plot PACF
    plot_pacf(timeseries.dropna(), ax=axes[2])
    axes[2].set_title('Partial Autocorrelation Function (PACF)')
    
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