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
import math
import warnings
warnings.filterwarnings('ignore')

# Utility untuk deteksi frekuensi data
def detect_frequency(data, date_column=None):
    """
    Mendeteksi frekuensi data timeseries secara otomatis
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame dengan kolom tanggal atau index datetime
    date_column : str, optional
        Nama kolom tanggal jika bukan index
        
    Returns:
    --------
    str
        Frekuensi terdeteksi ('D', 'W', 'M', 'Q', 'Y', atau 'H')
    """
    if date_column is not None:
        dates = pd.to_datetime(data[date_column])
    else:
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        else:
            return 'D'  # Default fallback
    
    # Hitung selisih antar tanggal
    diffs = dates.diff().dropna()
    
    # Hitung median selisih dalam hari
    median_diff_days = diffs.dt.total_seconds().median() / (24 * 3600)
    
    # Deteksi frekuensi berdasarkan median selisih
    if median_diff_days < 0.05:  # Kurang dari ~1.2 jam
        return 'H'  # Hourly
    elif median_diff_days < 1.5:  # Sekitar 1 hari
        return 'D'  # Daily
    elif median_diff_days < 3.5:  # Sekitar 1 minggu
        return 'W'  # Weekly
    elif median_diff_days < 15:  # Sekitar 1 bulan
        return 'M'  # Monthly
    elif median_diff_days < 45:  # Sekitar 1 kuartal
        return 'Q'  # Quarterly
    else:
        return 'Y'  # Yearly

def get_frequency_info(freq):
    """
    Mendapatkan informasi detail tentang frekuensi
    
    Parameters:
    -----------
    freq : str
        Kode frekuensi
        
    Returns:
    --------
    dict
        Informasi frekuensi (nama, periode dalam setahun, dll)
    """
    freq_map = {
        'H': {'name': 'Hourly', 'periods_per_year': 8760, 'period_name': 'hour'},
        'D': {'name': 'Daily', 'periods_per_year': 365, 'period_name': 'day'},
        'W': {'name': 'Weekly', 'periods_per_year': 52, 'period_name': 'week'},
        'M': {'name': 'Monthly', 'periods_per_year': 12, 'period_name': 'month'},
        'Q': {'name': 'Quarterly', 'periods_per_year': 4, 'period_name': 'quarter'},
        'Y': {'name': 'Yearly', 'periods_per_year': 1, 'period_name': 'year'}
    }
    return freq_map.get(freq, freq_map['D'])

def adjust_seasonal_periods(freq, seasonal_periods=None):
    """
    Menyesuaikan periode musiman berdasarkan frekuensi data
    
    Parameters:
    -----------
    freq : str
        Frekuensi data
    seasonal_periods : int, optional
        Periode musiman yang ingin digunakan
        
    Returns:
    --------
    int
        Periode musiman yang disesuaikan
    """
    if seasonal_periods is not None:
        return seasonal_periods
    
    # Default seasonal periods berdasarkan frekuensi
    seasonal_map = {
        'H': 24,      # 24 jam dalam sehari
        'D': 7,       # 7 hari dalam seminggu
        'W': 52,      # 52 minggu dalam setahun
        'M': 12,      # 12 bulan dalam setahun
        'Q': 4,       # 4 kuartal dalam setahun
        'Y': 1        # Tidak ada musiman untuk data tahunan
    }
    
    return seasonal_map.get(freq, 12)

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
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model ARIMA")
    
    if len(data) < 5:
        raise ValueError(f"Data terlalu sedikit untuk training model ARIMA. Minimal 5 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model ARIMA
    try:
        model = ARIMA(data[target_column], order=order)
        model_fit = model.fit()
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model ARIMA: {str(e)}")

def calculate_forecast_metrics(actual, predicted):
    """
    Menghitung berbagai metrik evaluasi untuk forecasting
    
    Parameters:
    -----------
    actual : array-like
        Nilai aktual
    predicted : array-like
        Nilai prediksi
        
    Returns:
    --------
    dict
        Dictionary berisi MAE, MSE, RMSE, dan MAPE
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Hapus nilai NaN atau infinite
    mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'MAPE': None,
            'R2': None
        }
    
    # Hitung metrik
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = math.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    # Hindari pembagian dengan nol
    mask_nonzero = actual != 0
    if np.sum(mask_nonzero) > 0:
        mape = np.mean(np.abs((actual[mask_nonzero] - predicted[mask_nonzero]) / actual[mask_nonzero])) * 100
    else:
        mape = None
    
    # R-squared
    r2 = r2_score(actual, predicted)
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'R2': r2
    }

def evaluate_forecast_model(model, test_data, target_column, date_column=None):
    """
    Mengevaluasi model forecasting dengan berbagai metrik
    
    Parameters:
    -----------
    model : object
        Model yang telah dilatih
    test_data : pandas.DataFrame
        Data testing
    target_column : str
        Nama kolom target
    date_column : str, optional
        Nama kolom tanggal (untuk model ML)
        
    Returns:
    --------
    dict
        Hasil evaluasi model
    """
    try:
        # Validasi input data
        if test_data is None or test_data.empty:
            raise ValueError("Data testing kosong")
            
        # Validasi kolom target
        if target_column not in test_data.columns:
            raise ValueError(f"Kolom '{target_column}' tidak ditemukan dalam data testing")
        
        # Validasi timestamp untuk mencegah out of bounds
        test_data_valid = test_data.copy()
        
        # Cek dan validasi index datetime
        if hasattr(test_data_valid, 'index') and isinstance(test_data_valid.index, pd.DatetimeIndex):
            try:
                # Batasi tanggal agar tidak terlalu jauh di masa depan/lampau
                current_year = pd.Timestamp.now().year
                min_year = current_year - 50  # Maksimum 50 tahun ke belakang
                max_year = current_year + 10  # Maksimum 10 tahun ke depan
                
                # Filter tanggal yang valid
                mask = (test_data_valid.index.year >= min_year) & (test_data_valid.index.year <= max_year)
                test_data_valid = test_data_valid[mask]
                
                if test_data_valid.empty:
                    raise ValueError("Data testing mengandung tanggal yang tidak valid (di luar batas yang diizinkan)")
                    
            except Exception as e:
                if "Out of bounds" in str(e) or "timestamp" in str(e).lower():
                    # Fallback ke data numerik jika datetime bermasalah
                    test_data_valid = test_data_valid.reset_index(drop=True)
        
        # Dapatkan prediksi untuk data test dengan handling yang aman
        if hasattr(model, 'predict') and date_column is not None:
            # Untuk model ML (Random Forest, Gradient Boosting, dll)
            try:
                # Buat fitur untuk data testing
                from utils import create_features_from_date, create_lag_features, create_rolling_features
                
                df = test_data_valid.copy()
                df[date_column] = pd.to_datetime(df[date_column])
                df = df.sort_values(by=date_column)
                
                # Buat fitur
                df = create_features_from_date(df, date_column)
                df = create_lag_features(df, target_column)
                df = create_rolling_features(df, target_column)
                
                # Hapus NaN
                df = df.dropna()
                
                # Dapatkan fitur yang digunakan
                features = [col for col in df.columns if col != target_column and col != date_column]
                
                if len(df) > 0:
                    X_test = df[features]
                    y_actual = df[target_column]
                    y_pred = model.predict(X_test)
                    
                    return calculate_forecast_metrics(y_actual, y_pred)
                else:
                    return {'error': 'Tidak cukup data untuk evaluasi'}
                    
            except Exception as e:
                return {'error': f'Error dalam evaluasi model ML: {str(e)}'}
        
        elif hasattr(model, 'forecast'):
            # Untuk model ARIMA, SARIMA, Exponential Smoothing
            try:
                # Dapatkan prediksi untuk periode testing
                steps = len(test_data_valid)
                forecast_result = model.forecast(steps=steps)
                
                if hasattr(forecast_result, 'values'):
                    y_pred = forecast_result.values
                else:
                    y_pred = np.array(forecast_result)
                
                y_actual = test_data_valid[target_column].values
                
                return calculate_forecast_metrics(y_actual, y_pred)
                
            except Exception as e:
                return {'error': f'Error dalam evaluasi model: {str(e)}'}
        
        else:
            return {'error': 'Tipe model tidak dikenali'}
            
    except Exception as e:
        # Error handling terakhir
        error_msg = str(e)
        if "Out of bounds" in error_msg or "timestamp" in error_msg:
            error_msg = "Error: Timestamp out of bounds. Silakan periksa tanggal dalam data Anda."
        
        return {
            'error': error_msg,
            'MAE': None,
            'MSE': None,
            'RMSE': None,
            'MAPE': None,
            'R2': None
        }

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
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model SARIMA")
    
    if len(data) < 5:
        raise ValueError(f"Data terlalu sedikit untuk training model SARIMA. Minimal 5 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model SARIMA
    try:
        model = SARIMAX(data[target_column], order=order, seasonal_order=seasonal_order)
        model_fit = model.fit(disp=False)
        return model_fit
    except Exception as e:
        raise ValueError(f"Gagal melatih model SARIMA: {str(e)}")

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
    
    # Validasi apakah ada data yang cukup untuk training
    if len(data) == 0:
        raise ValueError("Tidak ada data yang valid untuk training model Exponential Smoothing")
    
    if len(data) < 5:
        raise ValueError(f"Data terlalu sedikit untuk training model Exponential Smoothing. Minimal 5 data, tersedia: {len(data)}")
    
    # Validasi apakah kolom target memiliki nilai yang valid
    if data[target_column].isnull().all():
        raise ValueError(f"Kolom target '{target_column}' tidak memiliki nilai yang valid")
    
    # Latih model Exponential Smoothing
    try:
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
    except Exception as e:
        raise ValueError(f"Gagal melatih model Exponential Smoothing: {str(e)}")

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

def forecast_future(model_info, periods=10, freq=None, max_years_ahead=2, data_frequency=None):
    """
    Membuat prediksi untuk periode di masa depan dengan support berbagai frekuensi data
    
    Parameters:
    -----------
    model_info : dict atau statsmodels model
        Model yang telah dilatih atau dictionary dengan informasi model ML
    periods : int, optional
        Jumlah periode yang akan diprediksi
    freq : str, optional
        Frekuensi data ('D' untuk harian, 'W' untuk mingguan, 'M' untuk bulanan, dll.)
        Jika None, akan otomatis dideteksi dari data
    max_years_ahead : int, optional
        Batas maksimum tahun ke depan untuk prediksi (default: 2 tahun)
    data_frequency : str, optional
        Frekuensi data yang terdeteksi secara otomatis
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame dengan tanggal dan hasil prediksi yang sesuai dengan frekuensi data
    """
    # Deteksi frekuensi jika tidak ditentukan
    if freq is None:
        if data_frequency is not None:
            freq = data_frequency
        else:
            freq = 'D'  # Default ke harian
    
    # Validasi dan map frekuensi yang valid
    freq_map = {
        'harian': 'D', 'daily': 'D', 'D': 'D',
        'mingguan': 'W', 'weekly': 'W', 'W': 'W',
        'bulanan': 'M', 'monthly': 'M', 'M': 'M',
        'kuartal': 'Q', 'quarterly': 'Q', 'Q': 'Q',
        'tahunan': 'Y', 'yearly': 'Y', 'Y': 'Y'
    }
    
    # Gunakan frekuensi yang valid
    safe_freq = freq_map.get(str(freq).lower(), freq) if str(freq).lower() in freq_map else freq
    
    # Cek tipe model
    if isinstance(model_info, dict) and 'model_type' in model_info:  # ML model
        # Ambil informasi dari model_info
        model = model_info['model']
        features = model_info['features']
        date_column = model_info['date_column']
        target_column = model_info['target_column']
        last_date = model_info['last_date']
        
        # Validasi batas tanggal prediksi
        max_date = last_date + pd.DateOffset(years=max_years_ahead)
        
        # Hitung periode yang valid dengan frekuensi yang sesuai
        try:
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq=safe_freq)[1:]
            valid_dates = future_dates[future_dates <= max_date]
        except ValueError:
            # Fallback ke frekuensi harian jika error
            future_dates = pd.date_range(start=last_date, periods=periods+1, freq='D')[1:]
            valid_dates = future_dates[future_dates <= max_date]
        
        if len(valid_dates) < periods:
            print(f"Peringatan: Prediksi dibatasi hingga {max_years_ahead} tahun ke depan ({len(valid_dates)} dari {periods} periode)")
            
        future_dates = valid_dates
        
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
        
        # Buat tanggal untuk periode masa depan dengan validasi timestamp
        try:
            if hasattr(model_info, 'model') and hasattr(model_info.model, 'data'):
                # Handle both regular DataFrame and PandasData object
                data_obj = model_info.model.data
                
                # Try different ways to get the index
                last_date = None
                
                # Method 1: Direct index access
                if hasattr(data_obj, 'index') and len(data_obj.index) > 0:
                    try:
                        last_date = data_obj.index[-1]
                    except (IndexError, TypeError):
                        pass
                
                # Method 2: Access through orig_endog for PandasData
                if last_date is None and hasattr(data_obj, 'orig_endog'):
                    orig_endog = data_obj.orig_endog
                    if hasattr(orig_endog, 'index') and len(orig_endog.index) > 0:
                        try:
                            last_date = orig_endog.index[-1]
                        except (IndexError, TypeError):
                            pass
                
                # Method 3: Access through _index attribute
                if last_date is None and hasattr(data_obj, '_index') and len(data_obj._index) > 0:
                    try:
                        last_date = data_obj._index[-1]
                    except (IndexError, TypeError):
                        pass
                
                # If still no date, use numerical index
                if last_date is None:
                    return pd.DataFrame({
                        'forecast_index': range(periods),
                        'forecast': forecast
                    })
                    
            else:
                # Fallback to numerical index
                return pd.DataFrame({
                    'forecast_index': range(periods),
                    'forecast': forecast
                })
        except Exception:
            # Fallback to numerical index if any error occurs
            return pd.DataFrame({
                'forecast_index': range(periods),
                'forecast': forecast
            })
        
        # Validasi dan batasi tanggal prediksi dengan multiple safety checks
        if isinstance(last_date, pd.Timestamp):
            try:
                # Cek apakah tanggal terakhir sudah reasonable
                current_year = pd.Timestamp.now().year
                last_year = last_date.year
                
                # Batasi range tahun yang reasonable (50 tahun ke belakang, 10 tahun ke depan)
                if last_year < current_year - 50 or last_year > current_year + 10:
                    print(f"Warning: Tanggal terakhir ({last_date}) di luar range yang diizinkan")
                    # Gunakan fallback ke indeks numerik
                    return pd.DataFrame({
                        'forecast_index': range(min(periods, 365)),
                        'forecast': forecast[:min(periods, 365)]
                    })
                
                # Batasi prediksi maksimum 2 tahun ke depan (lebih konservatif)
                max_prediction_date = last_date + pd.DateOffset(years=2)
                
                # Gunakan frekuensi yang aman dengan mapping yang lebih baik
                safe_freq = safe_freq if safe_freq in ['D', 'W', 'M', 'Q', 'Y', 'H'] else 'D'
                
                try:
                    future_dates = pd.date_range(start=last_date, periods=periods+1, freq=safe_freq)[1:]
                except (OverflowError, OSError, ValueError) as e:
                    # Jika error dengan frekuensi tertentu, gunakan frekuensi harian
                    future_dates = pd.date_range(start=last_date, periods=min(periods+1, 730))[1:]  # Maksimum 2 tahun
                
                # Filter tanggal yang tidak out of bounds dengan validasi yang ketat
                valid_dates = []
                for date in future_dates:
                    try:
                        # Validasi timestamp bounds (32-bit limits)
                        timestamp = date.timestamp()
                        if -2147483648 <= timestamp <= 2147483647:  # 32-bit timestamp limits
                            # Validasi tahun
                            if current_year - 50 <= date.year <= current_year + 10:
                                valid_dates.append(date)
                            else:
                                break
                        else:
                            break
                    except (OverflowError, OSError, ValueError):
                        # Tanggal out of bounds, hentikan prediksi
                        break
                
                if len(valid_dates) == 0:
                    # Jika semua tanggal invalid, gunakan indeks numerik
                    return pd.DataFrame({
                        'forecast_index': range(min(periods, 365)),  # Batasi maksimum 1 tahun
                        'forecast': forecast[:min(periods, 365)]
                    })
                
                future_dates = pd.DatetimeIndex(valid_dates)
                
            except (OverflowError, OSError, ValueError) as e:
                # Tangani error timestamp dengan fallback ke indeks numerik
                print(f"Warning: Timestamp error - {str(e)}")
                return pd.DataFrame({
                    'forecast_index': range(min(periods, 365)),
                    'forecast': forecast[:min(periods, 365)]
                })
        else:
            # Gunakan indeks numerik jika bukan timestamp
            future_dates = range(min(periods, 365))
        
        # Kembalikan hasil prediksi dengan handling yang aman
        try:
            if isinstance(future_dates, pd.DatetimeIndex):
                return pd.DataFrame({
                    'ds': future_dates,
                    'yhat': forecast[:len(future_dates)]
                })
            else:
                return pd.DataFrame({
                    'forecast_index': list(future_dates),
                    'forecast': forecast[:len(list(future_dates))]
                })
        except Exception as e:
            # Fallback terakhir
            return pd.DataFrame({
                'forecast_index': range(min(periods, 100)),
                'forecast': forecast[:min(periods, 100)]
            })

def validate_timestamp_data(data, target_column=None):
    """
    Validasi data untuk memastikan tidak ada timestamp yang out of bounds
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Data yang akan divalidasi
    target_column : str, optional
        Nama kolom target untuk validasi tambahan
        
    Returns:
    --------
    pandas.DataFrame
        Data yang sudah divalidasi dan difilter
    """
    if data is None or data.empty:
        return data
    
    data_valid = data.copy()
    
    # Validasi index datetime
    if hasattr(data_valid, 'index') and isinstance(data_valid.index, pd.DatetimeIndex):
        try:
            # Batasi range tanggal yang reasonable
            current_year = pd.Timestamp.now().year
            min_year = current_year - 50
            max_year = current_year + 10
            
            # Filter berdasarkan tahun
            mask = (data_valid.index.year >= min_year) & (data_valid.index.year <= max_year)
            data_valid = data_valid[mask]
            
            # Validasi individual timestamps
            valid_indices = []
            for idx in data_valid.index:
                try:
                    idx.timestamp()
                    valid_indices.append(idx)
                except (OverflowError, OSError, ValueError):
                    continue
            
            if valid_indices:
                data_valid = data_valid.loc[valid_indices]
            else:
                # Fallback ke index numerik
                data_valid = data_valid.reset_index(drop=True)
                
        except Exception as e:
            # Jika error, gunakan index numerik
            data_valid = data_valid.reset_index(drop=True)
    
    # Validasi kolom tanggal jika ada
    date_columns = data_valid.select_dtypes(include=['datetime64']).columns
    for col in date_columns:
        try:
            data_valid[col] = pd.to_datetime(data_valid[col])
            # Filter tanggal yang reasonable
            mask = (data_valid[col].dt.year >= current_year - 50) & (data_valid[col].dt.year <= current_year + 10)
            data_valid = data_valid[mask]
        except Exception:
            # Jika error, skip kolom ini
            continue
    
    return data_valid

def evaluate_forecast_model(model, test_data, target_column):
    """
    Mengevaluasi model forecasting menggunakan data test dengan validasi timestamp
    
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
        Dictionary berisi metrik evaluasi atau error message
    """
    try:
        # Validasi input
        if test_data is None or test_data.empty:
            return {'error': 'Data testing kosong atau tidak valid'}
            
        if target_column not in test_data.columns:
            return {'error': f"Kolom '{target_column}' tidak ditemukan dalam data testing"}
        
        # Validasi timestamp data
        test_data_valid = validate_timestamp_data(test_data, target_column)
        
        if test_data_valid.empty:
            return {'error': 'Data testing tidak valid setelah validasi timestamp'}
        
        # Cek tipe model
        if isinstance(model, dict) and 'model_type' in model:  # ML model
            ml_model = model['model']
            features = model['features']
            
            # Pastikan semua fitur ada
            missing_features = [f for f in features if f not in test_data_valid.columns]
            if missing_features:
                return {'error': f"Fitur tidak ditemukan: {missing_features}"}
            
            # Prediksi
            y_pred = ml_model.predict(test_data_valid[features])
            y_true = test_data_valid[target_column]
            
        else:  # Statsmodels model
            # Prediksi dengan handling yang aman
            try:
                y_pred = model.forecast(steps=len(test_data_valid))
                y_true = test_data_valid[target_column]
            except Exception as e:
                return {'error': f"Error saat forecasting: {str(e)}"}
        
        # Hitung metrik dengan validasi
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Hapus nilai yang tidak valid
        mask = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isinf(y_true) | np.isinf(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        if len(y_true) == 0:
            return {'error': 'Tidak ada data yang valid untuk evaluasi'}
        
        # Hitung metrik
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # MAPE dengan handling division by zero
        mask_nonzero = y_true != 0
        if np.sum(mask_nonzero) > 0:
            mape = np.mean(np.abs((y_true[mask_nonzero] - y_pred[mask_nonzero]) / y_true[mask_nonzero])) * 100
        else:
            mape = None
        
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R2': r2,
            'n_samples': len(y_true)
        }
        
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['out of bounds', 'timestamp', 'overflow', 'datetime']):
            return {'error': 'Error timestamp: Tanggal di luar batas yang diizinkan'}
        else:
            return {'error': f"Error evaluasi model: {str(e)}"}

def plot_forecast_results(train_data, test_data, forecast_data, target_column, date_column=None, figsize=(15, 8), frequency=None):
    fig, ax = plt.subplots(figsize=figsize)
    
    # Handle data preparation
    if date_column and date_column in train_data.columns:
        train_data = train_data.set_index(date_column)
    if date_column and date_column in test_data.columns:
        test_data = test_data.set_index(date_column)
    
    # Plot data
    ax.plot(train_data.index, train_data[target_column], label='Training', linewidth=2)
    ax.plot(test_data.index, test_data[target_column], label='Testing', color='green', linewidth=2)
    
    # Plot forecast
    if 'ds' in forecast_data.columns and 'yhat' in forecast_data.columns:
        ax.plot(forecast_data['ds'], forecast_data['yhat'], label='Forecast', color='red', linewidth=2, linestyle='--')
    elif date_column in forecast_data.columns and f'predicted_{target_column}' in forecast_data.columns:
        ax.plot(forecast_data[date_column], forecast_data[f'predicted_{target_column}'], 
               label='Forecast', color='red', linewidth=2, linestyle='--')
    
    # Format based on frequency
    if frequency == 'D':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
    elif frequency == 'M':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
    elif frequency == 'Y':
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    ax.set_title(f'Forecasting Results ({frequency or "Auto"})')
    ax.set_xlabel('Date')
    ax.set_ylabel(target_column)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig