"""
State-of-the-art Time Series Anomaly Detection Utilities
This module provides multiple advanced algorithms for detecting anomalies in time series data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.tsa.arima.model import ARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


class TimeSeriesAnomalyDetector:
    """Comprehensive time series anomaly detection with multiple state-of-the-art algorithms"""
    
    def __init__(self):
        self.results = {}
        self.models = {}
        
    def isolation_forest_detection(self, data, contamination=0.1, random_state=42):
        """
        Isolation Forest for time series anomaly detection
        
        Parameters:
        - data: pandas Series with datetime index
        - contamination: proportion of outliers in the data set
        - random_state: random seed for reproducibility
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Prepare features: use rolling statistics
        features = pd.DataFrame()
        features['value'] = data.values
        features['rolling_mean'] = data.rolling(window=5).mean().fillna(method='bfill')
        features['rolling_std'] = data.rolling(window=5).std().fillna(method='bfill')
        features['diff'] = data.diff().fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
        anomalies = iso_forest.fit_predict(features_scaled)
        
        # Convert to boolean array (True = anomaly)
        anomaly_mask = anomalies == -1
        
        self.models['isolation_forest'] = {
            'model': iso_forest,
            'scaler': scaler,
            'features': features,
            'anomalies': anomaly_mask
        }
        
        return {
            'anomalies': anomaly_mask,
            'anomaly_scores': iso_forest.decision_function(features_scaled),
            'model': iso_forest
        }
    
    def one_class_svm_detection(self, data, nu=0.1, kernel='rbf'):
        """
        One-Class SVM for time series anomaly detection
        
        Parameters:
        - data: pandas Series with datetime index
        - nu: upper bound on the fraction of training errors and lower bound on fraction of support vectors
        - kernel: kernel type for SVM
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Prepare features
        features = pd.DataFrame()
        features['value'] = data.values
        features['rolling_mean'] = data.rolling(window=5).mean().fillna(method='bfill')
        features['rolling_std'] = data.rolling(window=5).std().fillna(method='bfill')
        features['diff'] = data.diff().fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Fit One-Class SVM
        svm = OneClassSVM(nu=nu, kernel=kernel)
        anomalies = svm.fit_predict(features_scaled)
        
        # Convert to boolean array
        anomaly_mask = anomalies == -1
        
        self.models['one_class_svm'] = {
            'model': svm,
            'scaler': scaler,
            'features': features,
            'anomalies': anomaly_mask
        }
        
        return {
            'anomalies': anomaly_mask,
            'anomaly_scores': svm.decision_function(features_scaled),
            'model': svm
        }
    
    def lstm_autoencoder_detection(self, data, sequence_length=10, epochs=50, contamination=0.1):
        """
        LSTM Autoencoder for time series anomaly detection
        
        Parameters:
        - data: pandas Series with datetime index
        - sequence_length: length of sequences for LSTM
        - epochs: number of training epochs
        - contamination: proportion of outliers
        """
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow not available. Install with: pip install tensorflow")
            
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Prepare sequences
        values = data.values.reshape(-1, 1)
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        
        def create_sequences(values, seq_length):
            sequences = []
            for i in range(len(values) - seq_length + 1):
                sequences.append(values[i:i+seq_length])
            return np.array(sequences)
        
        sequences = create_sequences(values_scaled, sequence_length)
        
        # Split data
        train_size = int(0.8 * len(sequences))
        train_sequences = sequences[:train_size]
        
        # Build LSTM Autoencoder
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1), return_sequences=True),
            LSTM(25, activation='relu', return_sequences=False),
            LSTM(25, activation='relu', return_sequences=True),
            LSTM(50, activation='relu', return_sequences=True),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        
        # Train model
        model.fit(train_sequences, train_sequences[:, -1:, :], 
                 epochs=epochs, batch_size=32, verbose=0)
        
        # Calculate reconstruction errors
        predictions = model.predict(sequences)
        reconstruction_errors = np.mean(np.square(predictions - sequences[:, -1:, :]), axis=(1, 2))
        
        # Pad errors to match original data length
        padded_errors = np.concatenate([np.zeros(sequence_length-1), reconstruction_errors])
        
        # Determine threshold
        threshold = np.percentile(padded_errors, (1 - contamination) * 100)
        anomaly_mask = padded_errors > threshold
        
        self.models['lstm_autoencoder'] = {
            'model': model,
            'scaler': scaler,
            'reconstruction_errors': padded_errors,
            'threshold': threshold,
            'anomalies': anomaly_mask
        }
        
        return {
            'anomalies': anomaly_mask,
            'reconstruction_errors': padded_errors,
            'threshold': threshold,
            'model': model
        }
    
    def prophet_based_detection(self, data, contamination=0.1, changepoint_prior_scale=0.05):
        """
        Prophet-based anomaly detection for time series
        
        Parameters:
        - data: pandas Series with datetime index
        - contamination: proportion of outliers
        - changepoint_prior_scale: flexibility of trend changes
        """
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
            
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Prepare data for Prophet
        df = pd.DataFrame({
            'ds': data.index,
            'y': data.values
        })
        
        # Fit Prophet model
        model = Prophet(changepoint_prior_scale=changepoint_prior_scale)
        model.fit(df)
        
        # Make predictions
        future = model.make_future_dataframe(periods=0)
        forecast = model.predict(future)
        
        # Calculate residuals
        residuals = np.abs(data.values - forecast['yhat'].values)
        
        # Determine threshold
        threshold = np.percentile(residuals, (1 - contamination) * 100)
        anomaly_mask = residuals > threshold
        
        self.models['prophet'] = {
            'model': model,
            'forecast': forecast,
            'residuals': residuals,
            'threshold': threshold,
            'anomalies': anomaly_mask
        }
        
        return {
            'anomalies': anomaly_mask,
            'residuals': residuals,
            'threshold': threshold,
            'forecast': forecast,
            'model': model
        }
    
    def statistical_detection(self, data, window_size=5, threshold=3):
        """
        Statistical anomaly detection using rolling statistics
        
        Parameters:
        - data: pandas Series with datetime index
        - window_size: size of rolling window
        - threshold: number of standard deviations for anomaly detection
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window_size).mean()
        rolling_std = data.rolling(window=window_size).std()
        
        # Calculate z-scores
        z_scores = np.abs((data - rolling_mean) / rolling_std)
        
        # Detect anomalies
        anomaly_mask = z_scores > threshold
        
        self.models['statistical'] = {
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'z_scores': z_scores,
            'threshold': threshold,
            'anomalies': anomaly_mask
        }
        
        return {
            'anomalies': anomaly_mask,
            'z_scores': z_scores,
            'threshold': threshold
        }
    
    def ensemble_detection(self, data, methods=['isolation_forest', 'statistical'], contamination=0.1, z_threshold=3.0):
        """
        Ensemble anomaly detection combining multiple methods
        
        Parameters:
        - data: pandas Series with datetime index
        - methods: list of detection methods to use
        - contamination: proportion of outliers
        - z_threshold: Z-score threshold for statistical method
        """
        results = {}
        
        for method in methods:
            if method == 'isolation_forest':
                result = self.isolation_forest_detection(data, contamination)
            elif method == 'one_class_svm':
                result = self.one_class_svm_detection(data, nu=contamination)
            elif method == 'statistical':
                result = self.statistical_detection(data, threshold=z_threshold)
            elif method == 'lstm_autoencoder' and TF_AVAILABLE:
                try:
                    result = self.lstm_autoencoder_detection(data, contamination=contamination)
                except Exception as e:
                    print(f"LSTM Autoencoder error: {e}")
                    continue
            elif method == 'prophet' and PROPHET_AVAILABLE:
                try:
                    result = self.prophet_based_detection(data, contamination=contamination)
                except Exception as e:
                    print(f"Prophet error: {e}")
                    continue
            
            results[method] = result['anomalies']
        
        # Combine results using majority voting
        combined_anomalies = np.zeros(len(data), dtype=bool)
        for method, anomalies in results.items():
            combined_anomalies |= anomalies
        
        self.models['ensemble'] = {
            'individual_results': results,
            'anomalies': combined_anomalies
        }
        
        return {
            'anomalies': combined_anomalies,
            'individual_results': results
        }
    
    def plot_anomaly_detection(self, data, method, figsize=(15, 8)):
        """
        Create visualization for anomaly detection results
        
        Parameters:
        - data: pandas Series with datetime index
        - method: detection method used
        - figsize: figure size for plotting
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Plot 1: Time series with anomalies
        axes[0].plot(data.index, data.values, color='blue', alpha=0.7, label='Normal')
        
        if method in self.models and 'anomalies' in self.models[method]:
            anomalies = self.models[method]['anomalies']
            if np.any(anomalies):
                axes[0].scatter(data.index[anomalies], data.values[anomalies], 
                              color='red', s=50, zorder=5, label='Anomalies')
        
        axes[0].set_title(f'Anomaly Detection Results - {method.upper()}')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Value')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Anomaly scores or indicators
        if method == 'isolation_forest' and method in self.models:
            scores = self.models[method]['model'].decision_function(
                self.models[method]['scaler'].transform(self.models[method]['features'])
            )
            axes[1].plot(data.index, scores, color='green', alpha=0.7)
            axes[1].axhline(y=np.percentile(scores, 10), color='red', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Anomaly Score')
            
        elif method == 'statistical' and method in self.models:
            z_scores = self.models[method]['z_scores']
            axes[1].plot(data.index, z_scores, color='green', alpha=0.7)
            axes[1].axhline(y=self.models[method]['threshold'], color='red', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Z-Score')
            
        elif method == 'lstm_autoencoder' and method in self.models:
            errors = self.models[method]['reconstruction_errors']
            axes[1].plot(data.index, errors, color='green', alpha=0.7)
            axes[1].axhline(y=self.models[method]['threshold'], color='red', linestyle='--', alpha=0.5)
            axes[1].set_ylabel('Reconstruction Error')
        
        axes[1].set_xlabel('Date')
        axes[1].set_title('Anomaly Indicators')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def get_anomaly_summary(self, data, method):
        """
        Get summary statistics for detected anomalies
        
        Parameters:
        - data: pandas Series with datetime index
        - method: detection method used
        """
        if not isinstance(data, pd.Series):
            data = pd.Series(data)
            
        if method not in self.models or 'anomalies' not in self.models[method]:
            return None
            
        anomalies = self.models[method]['anomalies']
        total_points = len(data)
        anomaly_count = np.sum(anomalies)
        anomaly_percentage = (anomaly_count / total_points) * 100
        
        summary = {
            'total_points': total_points,
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'anomaly_values': data[anomalies].values if anomaly_count > 0 else [],
            'anomaly_dates': data.index[anomalies].tolist() if anomaly_count > 0 else []
        }
        
        return summary


def detect_and_visualize_anomalies(data, target_column, date_column=None, methods=None, contamination=0.1, z_threshold=3.0):
    """
    Complete pipeline for anomaly detection in time series data
    
    Parameters:
    - data: pandas DataFrame
    - target_column: column name for time series values
    - date_column: column name for datetime (optional)
    - methods: list of detection methods to use
    - contamination: proportion of outliers
    - z_threshold: Z-score threshold for statistical method
    
    Returns:
    - Dictionary with results for each method
    """
    if methods is None:
        methods = ['isolation_forest', 'statistical']
    
    # Validasi input
    if data is None or len(data) == 0:
        raise ValueError("Data tidak boleh kosong")
    
    if target_column not in data.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam data")
    
    # Prepare data
    if date_column and date_column in data.columns:
        ts_data = data.set_index(date_column)[target_column]
    else:
        ts_data = data[target_column]
    
    # Validasi data time series
    if len(ts_data) < 10:
        raise ValueError("Dataset terlalu pendek. Minimal 10 data points diperlukan")
    
    if ts_data.std() == 0:
        raise ValueError("Data memiliki nilai konstan. Deteksi anomali tidak dapat dilakukan")
    
    # Handle missing values
    if ts_data.isnull().any():
        ts_data = ts_data.dropna()
    
    # Initialize detector
    detector = TimeSeriesAnomalyDetector()
    results = {}
    
    # Run detection for each method
    for method in methods:
        try:
            if method == 'isolation_forest':
                result = detector.isolation_forest_detection(ts_data, contamination=contamination)
            elif method == 'one_class_svm':
                result = detector.one_class_svm_detection(ts_data)
            elif method == 'statistical':
                result = detector.statistical_detection(ts_data)
            elif method == 'lstm_autoencoder':
                result = detector.lstm_autoencoder_detection(ts_data, contamination=contamination)
            elif method == 'prophet':
                result = detector.prophet_based_detection(ts_data, contamination=contamination)
            elif method == 'ensemble':
                result = detector.ensemble_detection(ts_data, contamination=contamination)
            
            results[method] = {
                'detector': detector,
                'result': result,
                'summary': detector.get_anomaly_summary(ts_data, method)
            }
            
        except Exception as e:
            results[method] = {
                'error': str(e),
                'detector': detector,
                'result': None,
                'summary': None
            }
    
    return results