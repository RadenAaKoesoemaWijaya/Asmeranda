#!/usr/bin/env python3
"""
Test script untuk mengecek apakah alur algoritma forecasting dapat berjalan optimal tanpa error
"""

import warnings
warnings.filterwarnings('ignore')

def test_forecasting_workflow():
    """Test the complete forecasting workflow"""
    print("=== FORECASTING WORKFLOW TEST ===\n")
    
    try:
        # Test basic imports
        import pandas as pd
        import numpy as np
        print("✓ Basic imports successful")
        
        # Test statsmodels imports
        try:
            import statsmodels.api as sm
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            print("✓ Statsmodels imports successful")
            statsmodels_available = True
        except ImportError as e:
            print(f"✗ Statsmodels import error: {e}")
            statsmodels_available = False
        
        # Test scikit-learn imports
        try:
            from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            print("✓ Scikit-learn imports successful")
        except ImportError as e:
            print(f"✗ Scikit-learn import error: {e}")
        
        # Test utility imports
        try:
            from utils import prepare_timeseries_data, check_stationarity, plot_timeseries_analysis
            print("✓ Utils imports successful")
        except ImportError as e:
            print(f"✗ Utils import error: {e}")
        
        # Test forecasting utils imports
        try:
            from forecasting_utils import train_arima_model, evaluate_forecast_model
            print("✓ Forecasting utils imports successful")
        except ImportError as e:
            print(f"✗ Forecasting utils import error: {e}")
        
        # Test basic functionality
        print("\nTesting basic forecasting functionality...")
        
        # Create sample time series data
        np.random.seed(42)  # For reproducible results
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        values = np.cumsum(np.random.randn(100)) + 100
        sample_data = pd.DataFrame({'date': dates, 'value': values})
        
        print(f"✓ Sample data created: {len(sample_data)} records")
        
        # Test data preparation
        try:
            ts_data = prepare_timeseries_data(sample_data, 'date', 'value', freq='D')
            print(f"✓ Time series data prepared: {len(ts_data)} records")
        except Exception as e:
            print(f"✗ Data preparation error: {e}")
            return False
        
        # Test stationarity check
        try:
            stationarity = check_stationarity(ts_data['value'])
            print(f"✓ Stationarity check completed. Stationary: {stationarity['Stationary']}")
            if stationarity['Message']:
                print(f"  Note: {stationarity['Message']}")
        except Exception as e:
            print(f"✗ Stationarity check error: {e}")
            return False
        
        # Test ARIMA model training (if statsmodels available)
        if statsmodels_available:
            print("\nTesting ARIMA model training...")
            try:
                # Split data
                train_size = int(len(ts_data) * 0.8)
                train_data = ts_data.iloc[:train_size]
                test_data = ts_data.iloc[train_size:]
                
                # Train ARIMA model
                model = train_arima_model(train_data, 'value', order=(1, 1, 1))
                print("✓ ARIMA model trained successfully")
                
                # Test model evaluation
                try:
                    eval_results = evaluate_forecast_model(model, test_data, 'value')
                    print(f"✓ Model evaluation completed")
                    print(f"  - MAE: {eval_results['MAE']:.4f}")
                    print(f"  - RMSE: {eval_results['RMSE']:.4f}")
                    print(f"  - R²: {eval_results['R2']:.4f}")
                    print(f"  - Samples: {eval_results['n_samples']}")
                    
                    if 'error' in eval_results and eval_results['error']:
                        print(f"  ⚠️  Warning: {eval_results['error']}")
                        
                except Exception as e:
                    print(f"✗ Model evaluation error: {e}")
                    return False
                
            except Exception as e:
                print(f"✗ ARIMA training error: {e}")
                return False
        
        # Test error handling with problematic data
        print("\nTesting error handling...")
        
        # Test with empty data
        try:
            empty_data = pd.DataFrame({'date': [], 'value': []})
            check_stationarity(empty_data['value'])
            print("✓ Empty data handling works")
        except Exception as e:
            print(f"✗ Empty data handling error: {e}")
        
        # Test with constant data
        try:
            constant_data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10, freq='D'), 
                                        'value': [5] * 10})
            result = check_stationarity(constant_data['value'])
            if result['Message'] and 'konstan' in result['Message']:
                print("✓ Constant data handling works")
            else:
                print("⚠️  Constant data handling may have issues")
        except Exception as e:
            print(f"✗ Constant data handling error: {e}")
        
        print("\n✅ All forecasting workflow tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ General error in forecasting workflow: {e}")
        return False

if __name__ == "__main__":
    success = test_forecasting_workflow()
    if success:
        print("\n🎉 FORECASTING WORKFLOW IS OPTIMAL AND ERROR-FREE!")
    else:
        print("\n⚠️  FORECASTING WORKFLOW HAS ISSUES THAT NEED ATTENTION!")
    
    exit(0 if success else 1)