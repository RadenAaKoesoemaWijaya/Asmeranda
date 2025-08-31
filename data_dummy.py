import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed untuk reproducibility
np.random.seed(42)
random.seed(42)

# Fungsi untuk generate data
def generate_heart_attack_dataset(n_samples=8000):
    data = []
    
    for i in range(n_samples):
        # Demographics
        age = np.random.randint(18, 85)
        gender = np.random.choice(['Male', 'Female'], p=[0.55, 0.45])
        
        # Risk factors based on scientific literature
        # Blood pressure (systolic/diastolic)
        if age < 40:
            systolic_bp = np.random.normal(120, 15)
        elif age < 60:
            systolic_bp = np.random.normal(130, 18)
        else:
            systolic_bp = np.random.normal(140, 20)
        
        systolic_bp = max(90, min(200, systolic_bp))
        diastolic_bp = max(60, min(120, systolic_bp * np.random.normal(0.65, 0.1)))
        
        # Cholesterol levels (mg/dL)
        total_cholesterol = np.random.normal(200, 40)
        if gender == 'Male':
            hdl_cholesterol = np.random.normal(45, 12)
        else:
            hdl_cholesterol = np.random.normal(55, 12)
        
        total_cholesterol = max(120, min(350, total_cholesterol))
        hdl_cholesterol = max(20, min(100, hdl_cholesterol))
        ldl_cholesterol = max(50, min(250, total_cholesterol - hdl_cholesterol - 40))
        
        # Blood sugar (mg/dL)
        if np.random.random() < 0.15:  # 15% diabetes prevalence
            fasting_bs = np.random.normal(140, 30)
        else:
            fasting_bs = np.random.normal(95, 15)
        fasting_bs = max(70, min(300, fasting_bs))
        
        # BMI calculation
        height = np.random.normal(170 if gender == 'Male' else 160, 10)
        if age < 30:
            weight_factor = 25 if gender == 'Male' else 22
        elif age < 50:
            weight_factor = 27 if gender == 'Male' else 24
        else:
            weight_factor = 28 if gender == 'Male' else 26
            
        weight = np.random.normal(weight_factor * (height/100)**2, 10)
        bmi = weight / (height/100)**2
        bmi = max(15, min(50, bmi))
        
        # Smoking status
        smoking_status = np.random.choice(['Never', 'Former', 'Current'], p=[0.55, 0.25, 0.20])
        
        # Alcohol consumption
        alcohol_consumption = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], 
                                               p=[0.30, 0.40, 0.20, 0.10])
        
        # Physical activity
        physical_activity = np.random.choice(['Low', 'Moderate', 'High'], p=[0.35, 0.45, 0.20])
        
        # Family history
        family_history = np.random.choice(['No', 'Yes'], p=[0.70, 0.30])
        
        # Diet quality (0-10 scale)
        diet_quality = np.random.normal(6, 2)
        diet_quality = max(1, min(10, diet_quality))
        
        # Stress level (0-10 scale)
        stress_level = np.random.normal(5, 2.5)
        stress_level = max(0, min(10, stress_level))
        
        # Sleep quality (hours per night)
        sleep_hours = np.random.normal(7, 1.5)
        sleep_hours = max(4, min(12, sleep_hours))
        
        # Additional clinical markers
        # Troponin levels (ng/mL) - typically elevated in heart damage
        troponin = np.random.exponential(0.01)
        troponin = min(10, troponin)
        
        # Ejection fraction (%)
        ejection_fraction = np.random.normal(60, 8)
        ejection_fraction = max(30, min(75, ejection_fraction))
        
        # Calculate risk score based on literature-based risk factors
        risk_score = 0
        
        # Age factor
        if age >= 65:
            risk_score += 3
        elif age >= 45:
            risk_score += 2
        else:
            risk_score += 1
            
        # Blood pressure factor
        if systolic_bp >= 140 or diastolic_bp >= 90:
            risk_score += 3
        elif systolic_bp >= 130 or diastolic_bp >= 80:
            risk_score += 2
        else:
            risk_score += 1
            
        # Cholesterol factor
        if total_cholesterol >= 240 or ldl_cholesterol >= 160:
            risk_score += 3
        elif total_cholesterol >= 200 or ldl_cholesterol >= 130:
            risk_score += 2
        else:
            risk_score += 1
            
        # HDL protective factor
        if hdl_cholesterol < 40:
            risk_score += 2
        elif hdl_cholesterol < 50:
            risk_score += 1
            
        # Diabetes factor
        if fasting_bs >= 126:
            risk_score += 3
        elif fasting_bs >= 100:
            risk_score += 1
            
        # BMI factor
        if bmi >= 30:
            risk_score += 3
        elif bmi >= 25:
            risk_score += 2
        else:
            risk_score += 1
            
        # Smoking factor
        if smoking_status == 'Current':
            risk_score += 3
        elif smoking_status == 'Former':
            risk_score += 1
            
        # Family history
        if family_history == 'Yes':
            risk_score += 2
            
        # Physical activity
        if physical_activity == 'Low':
            risk_score += 2
        elif physical_activity == 'Moderate':
            risk_score += 1
            
        # Alcohol factor
        if alcohol_consumption == 'Heavy':
            risk_score += 2
        elif alcohol_consumption == 'Moderate':
            risk_score += 1
            
        # Stress and sleep
        if stress_level >= 7:
            risk_score += 1
        if sleep_hours < 6:
            risk_score += 1
            
        # Determine heart attack risk based on risk score
        if risk_score >= 15:
            risk_level = 'High Risk'
            heart_attack = np.random.choice([0, 1], p=[0.3, 0.7])  # 70% probability
        elif risk_score >= 10:
            risk_level = 'Medium Risk'
            heart_attack = np.random.choice([0, 1], p=[0.7, 0.3])  # 30% probability
        else:
            risk_level = 'Low Risk'
            heart_attack = np.random.choice([0, 1], p=[0.9, 0.1])  # 10% probability
            
        # Add some randomness for realistic variation
        if np.random.random() < 0.05:  # 5% random flip
            heart_attack = 1 - heart_attack
            
        # Create record
        record = {
            'patient_id': f'P{str(i+1).zfill(4)}',
            'age': age,
            'gender': gender,
            'systolic_bp': round(systolic_bp, 1),
            'diastolic_bp': round(diastolic_bp, 1),
            'total_cholesterol': round(total_cholesterol, 1),
            'hdl_cholesterol': round(hdl_cholesterol, 1),
            'ldl_cholesterol': round(ldl_cholesterol, 1),
            'fasting_bs': round(fasting_bs, 1),
            'bmi': round(bmi, 1),
            'smoking_status': smoking_status,
            'alcohol_consumption': alcohol_consumption,
            'physical_activity': physical_activity,
            'family_history': family_history,
            'diet_quality': round(diet_quality, 1),
            'stress_level': round(stress_level, 1),
            'sleep_hours': round(sleep_hours, 1),
            'troponin': round(troponin, 3),
            'ejection_fraction': round(ejection_fraction, 1),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'heart_attack': heart_attack  # 0 = No, 1 = Yes
        }
        
        data.append(record)
    
    return pd.DataFrame(data)

# Generate dataset
print("Generating heart attack risk dataset...")
df = generate_heart_attack_dataset(8000)

# Display basic statistics
print(f"Dataset shape: {df.shape}")
print(f"\nTarget variable distribution:")
print(df['heart_attack'].value_counts())
print(f"\nRisk level distribution:")
print(df['risk_level'].value_counts())

# Save to CSV
df.to_csv('heart_attack_risk_dataset_8000.csv', index=False)
print("\nDataset saved as 'heart_attack_risk_dataset_8000.csv'")

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Display summary statistics
print("\nSummary statistics:")
print(df.describe())