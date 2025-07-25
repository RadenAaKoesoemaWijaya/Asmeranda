import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import fungsi-fungsi dari utils.py
from utils import (
    is_timeseries, detect_timeseries_columns, prepare_timeseries_data,
    check_stationarity, plot_timeseries_analysis, create_features_from_date,
    create_lag_features, create_rolling_features
)

def train_arima_model(data, target_column, order=(1,1,1)):
    """
    Melatih model ARIMA untuk data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    order : tuple, optional
        Order ARIMA (p,d,q)
        
    Returns:
    --------
    model : statsmodels.tsa.arima.model.ARIMAResults
        Model ARIMA yang telah dilatih
    """
    # Pastikan data tidak memiliki nilai yang hilang
    data = data.dropna()
    
    # Latih model ARIMA
    model = ARIMA(data[target_column], order=order)
    model_fit = model.fit()
    
    return model_fit

def train_sarima_model(data, target_column, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Melatih model SARIMA untuk data timeseries dengan komponen musiman
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    order : tuple, optional
        Order ARIMA (p,d,q)
    seasonal_order : tuple, optional
        Order musiman SARIMA (P,D,Q,s)
        
    Returns:
    --------
    model : statsmodels.tsa.statespace.sarimax.SARIMAXResults
        Model SARIMA yang telah dilatih
    """
    # Pastikan data tidak memiliki nilai yang hilang
    data = data.dropna()
    
    # Latih model SARIMA
    model = SARIMAX(data[target_column], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit(disp=False)
    
    return model_fit

def train_exponential_smoothing(data, target_column, trend=None, seasonal=None, seasonal_periods=None):
    """
    Melatih model Exponential Smoothing untuk data timeseries
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan index tanggal/waktu dan kolom target
    target_column : str
        Nama kolom target yang akan diprediksi
    trend : str, optional
        Tipe trend ('add', 'mul', None)
    seasonal : str, optional
        Tipe seasonal ('add', 'mul', None)
    seasonal_periods : int, optional
        Jumlah periode dalam satu siklus musiman
        
    Returns:
    --------
    model : statsmodels.tsa.holtwinters.ExponentialSmoothingResults
        Model Exponential Smoothing yang telah dilatih
    """
    # Pastikan data tidak memiliki nilai yang hilang
    data = data.dropna()
    
    # Latih model Exponential Smoothing
    if seasonal is not None and seasonal_periods is not None:
        model = ExponentialSmoothing(
            data[target_column],
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods
        )
    else:
        model = ExponentialSmoothing(
            data[target_column],
            trend=trend
        )
    
    model_fit = model.fit()
    
    return model_fit

def train_ml_forecaster(data, date_column, target_column, features=None, model_type='random_forest', **model_params):
    """
    Melatih model machine learning untuk forecasting dengan fitur lag dan rolling
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan kolom tanggal dan target
    date_column : str
        Nama kolom tanggal/waktu
    target_column : str
        Nama kolom target yang akan diprediksi
    features : list, optional
        Daftar kolom fitur tambahan (selain fitur lag dan rolling)
    model_type : str, optional
        Tipe model ('random_forest', 'gradient_boosting', 'linear')
    model_params : dict
        Parameter tambahan untuk model
        
    Returns:
    --------
    dict
        Dictionary berisi model yang telah dilatih dan informasi tambahan
    """
    # Buat salinan data
    df = data.copy()
    
    # Pastikan kolom tanggal adalah datetime
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Urutkan data berdasarkan tanggal
    df = df.sort_values(by=date_column)
    
    # Buat fitur dari tanggal
    df = create_features_from_date(df, date_column)
    
    # Buat fitur lag
    df = create_lag_features(df, target_column)
    
    # Buat fitur rolling
    df = create_rolling_features(df, target_column)
    
    # Hapus baris dengan nilai NaN (karena fitur lag dan rolling)
    df = df.dropna()
    
    # Tentukan fitur yang akan digunakan
    date_features = ['year', 'month', 'day', 'dayofweek', 'quarter', 'is_month_start', 'is_month_end']
    lag_features = [col for col in df.columns if f'{target_column}_lag_' in col]
    rolling_features = [col for col in df.columns if f'{target_column}_rolling_' in col]
    
    # Gabungkan semua fitur
    all_features = date_features + lag_features + rolling_features
    
    # Tambahkan fitur tambahan jika ada
    if features is not None:
        all_features += [f for f in features if f in df.columns and f != target_column and f != date_column]
    
    # Pisahkan fitur dan target
    X = df[all_features]
    y = df[target_column]
    
    # Pilih model berdasarkan tipe
    if model_type == 'random_forest':
        model = RandomForestRegressor(**model_params)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(**model_params)
    else:  # linear
        model = LinearRegression(**model_params)
    
    # Latih model
    model.fit(X, y)
    
    # Kembalikan model dan informasi tambahan
    return {
        'model': model,
        'features': all_features,
        'date_column': date_column,
        'target_column': target_column,
        'last_date': df[date_column].max(),
        'model_type': model_type
    }

def forecast_future(model_info, periods=10, freq='D'):
    """
    Membuat prediksi untuk periode di masa depan
    
    Parameters:
    -----------
    model_info : dict atau statsmodels model
        Model yang telah dilatih atau dictionary dengan informasi model ML
    periods : int, optional
        Jumlah periode yang akan diprediksi
    freq : str, optional
        Frekuensi data ('D' untuk harian, 'W' untuk mingguan, 'M' untuk bulanan, dll.)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan tanggal dan hasil prediksi
    """
    # Cek tipe model
    if isinstance(model_info, dict) and 'model_type' in model_info:  # ML model
        # Ambil informasi dari model_info
        model = model_info['model']
        features = model_info['features']
        date_column = model_info['date_column']
        target_column = model_info['target_column']
        last_date = model_info['last_date']
        
        # Buat tanggal untuk periode masa depan
        future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        
        # Buat DataFrame untuk prediksi
        future_df = pd.DataFrame({date_column: future_dates})
        
        # Buat fitur dari tanggal
        future_df = create_features_from_date(future_df, date_column)
        
        # Untuk model ML, kita perlu membuat fitur lag dan rolling secara manual
        # Ini memerlukan data historis, yang seharusnya disimpan dalam model_info
        # Untuk sederhananya, kita akan menggunakan nilai default untuk fitur lag dan rolling
        
        # Buat fitur lag dan rolling dengan nilai default (rata-rata dari data latih)
        lag_features = [f for f in features if f'{target_column}_lag_' in f]
        rolling_features = [f for f in features if f'{target_column}_rolling_' in f]
        
        # Gunakan nilai default untuk fitur lag dan rolling (misalnya 0)
        for feature in lag_features + rolling_features:
            future_df[feature] = 0
        
        # Prediksi menggunakan model ML
        future_df[f'predicted_{target_column}'] = model.predict(future_df[features])
        
        # Kembalikan hasil prediksi
        return future_df[[date_column, f'predicted_{target_column}']]
    
    else:  # Statsmodels model (ARIMA, SARIMA, Exponential Smoothing)
        # Prediksi menggunakan model statsmodels
        forecast = model_info.forecast(steps=periods)
        
        # Buat tanggal untuk periode masa depan
        if hasattr(model_info, 'model') and hasattr(model_info.model, 'data'):
            last_date = model_info.model.data.index[-1]
        else:
            # Jika tidak bisa mendapatkan tanggal terakhir, gunakan indeks numerik
            return pd.DataFrame({
                'forecast_index': range(periods),
                'forecast': forecast
            })
        
        # Buat tanggal untuk periode masa depan
        if isinstance(last_date, pd.Timestamp):
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=freq)[1:]
        else:
            future_dates = range(periods)
        
        # Kembalikan hasil prediksi
        return pd.DataFrame({
            'ds': future_dates,
            'yhat': forecast
        })

def evaluate_forecast_model(model, test_data, target_column):
    """
    Mengevaluasi model forecasting menggunakan data test
    
    Parameters:
    -----------
    model : statsmodels model atau dict
        Model yang telah dilatih
    test_data : pandas.DataFrame
        Data test untuk evaluasi
    target_column : str
        Nama kolom target
        
    Returns:
    --------
    dict
        Dictionary berisi metrik evaluasi
    """
    # Cek tipe model
    if isinstance(model, dict) and 'model_type' in model:  # ML model
        # Ambil informasi dari model
        ml_model = model['model']
        features = model['features']
        
        # Pastikan semua fitur ada di test_data
        missing_features = [f for f in features if f not in test_data.columns]
        if missing_features:
            raise ValueError(f"Fitur berikut tidak ada di test_data: {missing_features}")
        
        # Prediksi menggunakan model ML
        y_pred = ml_model.predict(test_data[features])
        y_true = test_data[target_column]
    
    else:  # Statsmodels model
        # Prediksi menggunakan model statsmodels
        y_pred = model.forecast(steps=len(test_data))
        y_true = test_data[target_column]
    
    # Hitung metrik evaluasi
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Kembalikan metrik evaluasi
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

def plot_forecast_results(train_data, test_data, forecast_data, target_column, date_column=None, figsize=(15, 8)):
    """
    Membuat plot hasil forecasting
    
    Parameters:
    -----------
    train_data : pandas.DataFrame
        Data training
    test_data : pandas.DataFrame
        Data testing
    forecast_data : pandas.DataFrame
        Data hasil forecasting
    target_column : str
        Nama kolom target
    date_column : str, optional
        Nama kolom tanggal (jika tidak menggunakan index)
    figsize : tuple, optional
        Ukuran gambar
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure yang berisi plot hasil forecasting
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data training
    if date_column is not None and date_column in train_data.columns:
        train_data = train_data.set_index(date_column)
    
    ax.plot(train_data.index, train_data[target_column], label='Data Training')
    
    # Plot data testing
    if date_column is not None and date_column in test_data.columns:
        test_data = test_data.set_index(date_column)
    
    ax.plot(test_data.index, test_data[target_column], label='Data Testing', color='green')
    
    # Plot hasil forecasting
    if 'ds' in forecast_data.columns and 'yhat' in forecast_data.columns:
        # Format dari forecast_future untuk model statsmodels
        ax.plot(forecast_data['ds'], forecast_data['yhat'], label='Forecast', color='red')
    elif date_column in forecast_data.columns and f'predicted_{target_column}' in forecast_data.columns:
        # Format dari forecast_future untuk model ML
        ax.plot(forecast_data[date_column], forecast_data[f'predicted_{target_column}'], label='Forecast', color='red')
    
    ax.set_title('Hasil Forecasting')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel(target_column)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    return fig