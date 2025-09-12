import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt

# Set seed untuk reproduktifitas
np.random.seed(42)
random.seed(42)

# Generate 3000 data points (132 bulan x 23 wilayah/kabupaten)
start_date = datetime(2010, 1, 1)
end_date = datetime(2020, 12, 31)

# Generate monthly data for 11 years
dates = pd.date_range(start=start_date, end=end_date, freq='M')

# Create dataset with realistic patterns for Papua region
data = []

for date in dates:
    # Generate multiple districts per month to reach 3000 rows
    for district in range(23):  # 23 districts/regencies in Papua
        # Seasonal patterns (higher cases in wet season: Nov-Apr)
        month = date.month
        is_wet_season = month in [11, 12, 1, 2, 3, 4]
        
        # Base values with realistic ranges for Papua districts
        base_population = np.random.normal(35000, 12000)  # 20k-50k per district
        population = max(15000, int(base_population))
        
        # Weather patterns with district variations
        if is_wet_season:
            temperature = np.random.normal(27.8, 1.8)  # Warmer in wet season
            humidity = np.random.normal(87, 6)  # Higher humidity
            rainfall = np.random.normal(450, 120)  # Heavy rainfall (mm/month)
        else:
            temperature = np.random.normal(26.2, 1.6)
            humidity = np.random.normal(76, 6)
            rainfall = np.random.normal(180, 45)
        
        # Healthcare indicators with improvement over time
        year_factor = (date.year - 2010) / 10  # 0 to 1 over 10 years
        
        # ITN distribution (increasing over time)
        base_itn = 80 + (year_factor * 280)
        itn = int(np.random.normal(base_itn, base_itn * 0.25))
        itn_per_1000 = (itn / population) * 1000
        
        # Healthcare access indicators
        anc = max(0, np.random.normal(65 + year_factor * 25, 12))  # ANC coverage
        idl = max(0, np.random.normal(55 + year_factor * 30, 18))  # IDL coverage
        
        # Socioeconomic factors
        gini_index = max(0.32, min(0.58, np.random.normal(0.48 - year_factor * 0.08, 0.06)))
        
        # Screening and treatment (increasing capacity)
        base_screening_rate = 0.04 + (year_factor * 0.03)  # 4-7% monthly
        screening_cases = int(np.random.poisson(population * base_screening_rate))
        initial_treatment = int(screening_cases * np.random.uniform(0.82, 0.96))
        
        # Malaria cases with seasonal and trend patterns
        base_incidence = 0.018 - (year_factor * 0.008)  # Decreasing from 1.8% to 1.0%
        seasonal_multiplier = 1.6 if is_wet_season else 0.65
        
        # District-specific risk factors
        risk_multiplier = np.random.uniform(0.7, 1.4)  # Some districts higher risk
        
        cases = int(np.random.poisson(population * base_incidence * seasonal_multiplier * risk_multiplier))
        
        # Deaths with improving CFR
        base_cfr = 2.8 - (year_factor * 1.6)  # CFR decreasing from 2.8 to 1.2 per 1000
        deaths = int(np.random.binomial(cases, max(0.0005, base_cfr / 1000)))
        
        # Calculate rates
        cfr_per_1000 = (deaths / max(cases, 1)) * 1000
        mortality_rate_per_100000 = (deaths / population) * 100000
        
        # Add outbreak events (rare)
        if np.random.random() < 0.03:  # 3% chance of outbreak per district-month
            outbreak_multiplier = np.random.uniform(1.8, 4.2)
            cases = int(cases * outbreak_multiplier)
            deaths = int(deaths * min(outbreak_multiplier * 0.8, 2.5))
        
        # Create row
        row = {
            'bulan': date.strftime('%Y-%m'),
            'cfr_per_1000': round(cfr_per_1000, 2),
            'mortality_rate_per_100000': round(mortality_rate_per_100000, 2),
            'itn': max(0, itn),
            'itn_per_1000': round(max(0, itn_per_1000), 2),
            'jumlah_penduduk': population,
            'jumlah_skrining_malaria': max(0, screening_cases),
            'jumlah_tatalaksana_awal': max(0, initial_treatment),
            'anc': round(max(0, anc), 1),
            'idl': round(max(0, idl), 1),
            'indeks_gini': round(gini_index, 3),
            'suhu': round(temperature, 1),
            'kelembapan_udara': round(humidity, 1),
            'curah_hujan': round(rainfall, 1),
            'kasus_malaria': max(0, cases),
            'kematian': max(0, deaths)
        }
        
        data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Ensure we have exactly 3000 rows
target_rows = 3000
if len(df) > target_rows:
    df = df.iloc[:target_rows]
elif len(df) < target_rows:
    # Add more data with slight variations
    additional_needed = target_rows - len(df)
    sample_df = df.sample(additional_needed, replace=True)
    
    # Add noise to duplicated data
    for col in ['jumlah_penduduk', 'kasus_malaria', 'kematian', 'itn', 'jumlah_skrining_malaria']:
        noise = np.random.uniform(0.9, 1.1, len(sample_df))
        sample_df[col] = (sample_df[col] * noise).astype(int)
    
    df = pd.concat([df, sample_df], ignore_index=True)

# Recalculate derived columns
df['cfr_per_1000'] = (df['kematian'] / df['kasus_malaria'].clip(lower=1)) * 1000
df['mortality_rate_per_100000'] = (df['kematian'] / df['jumlah_penduduk']) * 100000
df['itn_per_1000'] = (df['itn'] / df['jumlah_penduduk']) * 1000

# Round numeric columns
df = df.round({
    'cfr_per_1000': 2, 
    'mortality_rate_per_100000': 2, 
    'itn_per_1000': 2, 
    'indeks_gini': 3, 
    'suhu': 1, 
    'kelembapan_udara': 1, 
    'curah_hujan': 1,
    'anc': 1, 
    'idl': 1
})

# Ensure positive values
df['kasus_malaria'] = df['kasus_malaria'].clip(lower=0)
df['kematian'] = df['kematian'].clip(lower=0)

# Save to CSV
df.to_csv('malaria_forecasting_dataset.csv', index=False)

# Display summary
print("Dataset Malaria Forecasting Papua 2010-2020")
print("=" * 50)
print(f"Total data: {len(df)} baris")
print(f"Periode: {df['bulan'].min()} sampai {df['bulan'].max()}")
print(f"Wilayah: 23 kabupaten/kota di Papua")
print("\nStatistik Ringkas:")
print(df[['kasus_malaria', 'kematian', 'jumlah_penduduk', 'itn', 'suhu', 'curah_hujan']].describe())

# Create comprehensive visualization
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: Cases over time
axes[0, 0].plot(pd.to_datetime(df['bulan']).dt.to_period('M').dt.to_timestamp(), df['kasus_malaria'])
axes[0, 0].set_title('Kasus Malaria per Bulan (2010-2020)')
axes[0, 0].set_xlabel('Tahun')
axes[0, 0].set_ylabel('Kasus')
axes[0, 0].tick_params(axis='x', rotation=45)

# Plot 2: Deaths over time
axes[0, 1].plot(pd.to_datetime(df['bulan']).dt.to_period('M').dt.to_timestamp(), df['kematian'])
axes[0, 1].set_title('Kematian Akibat Malaria per Bulan')
axes[0, 1].set_xlabel('Tahun')
axes[0, 1].set_ylabel('Kematian')
axes[0, 1].tick_params(axis='x', rotation=45)

# Plot 3: ITN distribution
axes[0, 2].scatter(df['itn_per_1000'], df['kasus_malaria'], alpha=0.5)
axes[0, 2].set_title('ITN per 1000 Penduduk vs Kasus Malaria')
axes[0, 2].set_xlabel('ITN per 1000 Penduduk')
axes[0, 2].set_ylabel('Kasus Malaria')

# Plot 4: Temperature vs Cases
axes[1, 0].scatter(df['suhu'], df['kasus_malaria'], alpha=0.5)
axes[1, 0].set_title('Hubungan Suhu vs Kasus Malaria')
axes[1, 0].set_xlabel('Suhu (°C)')
axes[1, 0].set_ylabel('Kasus Malaria')

# Plot 5: Rainfall vs Cases
axes[1, 1].scatter(df['curah_hujan'], df['kasus_malaria'], alpha=0.5)
axes[1, 1].set_title('Hubungan Curah Hujan vs Kasus Malaria')
axes[1, 1].set_xlabel('Curah Hujan (mm/bulan)')
axes[1, 1].set_ylabel('Kasus Malaria')

# Plot 6: CFR trend
axes[1, 2].plot(pd.to_datetime(df['bulan']).dt.to_period('M').dt.to_timestamp(), df['cfr_per_1000'])
axes[1, 2].set_title('Tren CFR per 1000 Kasus')
axes[1, 2].set_xlabel('Tahun')
axes[1, 2].set_ylabel('CFR per 1000')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('malaria_forecasting_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nDataset berhasil dibuat: malaria_forecasting_dataset.csv")
print("Visualisasi tersimpan: malaria_forecasting_analysis.png")
print("\nPreview 10 baris pertama:")
print(df.head(10))