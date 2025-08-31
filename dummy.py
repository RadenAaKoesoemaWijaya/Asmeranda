# Contoh pembuatan dataset dummy time series untuk forecasting kasus DBD
import pandas as pd
import numpy as np

np.random.seed(42)

# Buat tanggal bulanan selama 1000 bulan
dates = pd.date_range(start='2000-01-01', periods=1000, freq='M')

# 7 parameter cuaca & suhu (misal: suhu rata-rata, kelembapan, curah hujan, kecepatan angin, tekanan udara, radiasi matahari, jumlah hari hujan)
data = {
    'date': dates,
    'avg_temperature': np.random.normal(28, 2, 1000),      # Suhu rata-rata (°C)
    'humidity': np.random.uniform(60, 95, 1000),           # Kelembapan (%)
    'rainfall': np.random.gamma(2, 50, 1000),              # Curah hujan (mm)
    'wind_speed': np.random.uniform(1, 10, 1000),          # Kecepatan angin (m/s)
    'pressure': np.random.normal(1010, 5, 1000),           # Tekanan udara (hPa)
    'solar_radiation': np.random.uniform(100, 300, 1000),  # Radiasi matahari (W/m2)
    'rainy_days': np.random.randint(5, 25, 1000),          # Jumlah hari hujan per bulan
}

df = pd.DataFrame(data)

# Buat target: kasus DBD, dipengaruhi oleh beberapa parameter + noise
df['dengue_cases'] = (
    10
    + 2 * df['rainfall']
    + 0.5 * df['humidity']
    - 1.5 * df['avg_temperature']
    + 0.2 * df['rainy_days']
    + np.random.normal(0, 10, 1000)
).astype(int)

# Pastikan tidak ada kasus negatif
df['dengue_cases'] = df['dengue_cases'].clip(lower=0)

# Simpan ke CSV (opsional)
df.to_csv('dummy_dbd_timeseries.csv', index=False)
