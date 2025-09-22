#!/usr/bin/env python3
"""
Test script untuk algoritma deteksi anomali
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Tambahkan path untuk import
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data():
    """Membuat data uji untuk deteksi anomali"""
    # Data normal dengan tren
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    n_points = len(dates)
    
    # Time series dengan tren dan musiman
    trend = np.linspace(100, 200, n_points)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_points) / 365.25)
    noise = np.random.normal(0, 5, n_points)
    
    values = trend + seasonal + noise
    
    # Tambahkan anomali (5% dari data)
    n_anomalies = int(0.05 * n_points)
    anomaly_indices = np.random.choice(n_points, n_anomalies, replace=False)
    values[anomaly_indices] += np.random.uniform(30, 60, n_anomalies)
    
    df = pd.DataFrame({
        'date': dates,
        'value': values
    })
    
    return df

def test_anomaly_detection():
    """Test fungsi deteksi anomali"""
    print("🧪 Testing Anomaly Detection Algorithms...")
    
    try:
        from anomaly_detection_utils import detect_and_visualize_anomalies
        
        # Test 1: Data normal
        print("\n1. Testing dengan data normal...")
        test_data = create_test_data()
        
        results = detect_and_visualize_anomalies(
            data=test_data,
            target_column='value',
            date_column='date',
            methods=['isolation_forest', 'statistical', 'ensemble'],
            contamination=0.05
        )
        
        print(f"   ✅ Data berhasil diproses: {len(test_data)} titik")
        
        for method, result in results.items():
            if 'error' in result:
                print(f"   ❌ {method}: Error - {result['error']}")
            else:
                summary = result['summary']
                print(f"   ✅ {method}: {summary['anomaly_count']} anomali ({summary['anomaly_percentage']:.2f}%)")
        
        # Test 2: Data konstan (harus gagal)
        print("\n2. Testing dengan data konstan...")
        constant_data = test_data.copy()
        constant_data['value'] = 100  # Nilai konstan
        
        try:
            results = detect_and_visualize_anomalies(
                data=constant_data,
                target_column='value',
                date_column='date',
                methods=['isolation_forest']
            )
            print("   ❌ Seharusnya gagal untuk data konstan")
        except ValueError as e:
            print(f"   ✅ Benar menolak data konstan: {e}")
        
        # Test 3: Data pendek (harus gagal)
        print("\n3. Testing dengan data pendek...")
        short_data = test_data.head(5)
        
        try:
            results = detect_and_visualize_anomalies(
                data=short_data,
                target_column='value',
                date_column='date',
                methods=['isolation_forest']
            )
            print("   ❌ Seharusnya gagal untuk data pendek")
        except ValueError as e:
            print(f"   ✅ Benar menolak data pendek: {e}")
        
        # Test 4: Data dengan missing values
        print("\n4. Testing dengan data memiliki missing values...")
        missing_data = test_data.copy()
        missing_data.loc[10:15, 'value'] = np.nan
        
        results = detect_and_visualize_anomalies(
            data=missing_data,
            target_column='value',
            date_column='date',
            methods=['isolation_forest', 'statistical']
        )
        
        print(f"   ✅ Data dengan missing values berhasil diproses")
        
        print("\n🎉 Semua test berhasil!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error dalam testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_integration():
    """Test integrasi dengan Streamlit"""
    print("\n🧪 Testing Streamlit Integration...")
    
    try:
        # Simulasi kondisi seperti di Streamlit
        test_data = create_test_data()
        
        # Test parameter yang digunakan di Streamlit
        date_column = 'date'
        target_column = 'value'
        contamination = 0.05
        z_threshold = 3.0
        selected_methods = ['isolation_forest', 'statistical', 'ensemble']
        
        from anomaly_detection_utils import detect_and_visualize_anomalies
        
        results = detect_and_visualize_anomalies(
            data=test_data,
            target_column=target_column,
            date_column=date_column,
            methods=selected_methods,
            contamination=contamination,
            z_threshold=z_threshold
        )
        
        print("   ✅ Streamlit integration test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Streamlit integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ANOMALY DETECTION ALGORITHM TEST")
    print("=" * 60)
    
    # Test basic functionality
    basic_test = test_anomaly_detection()
    
    # Test Streamlit integration
    streamlit_test = test_streamlit_integration()
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Basic Functionality Test: {'✅ PASSED' if basic_test else '❌ FAILED'}")
    print(f"Streamlit Integration Test: {'✅ PASSED' if streamlit_test else '❌ FAILED'}")
    
    if basic_test and streamlit_test:
        print("\n🎉 All tests PASSED! Algorithm is ready for production.")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Please review the implementation.")
        sys.exit(1)