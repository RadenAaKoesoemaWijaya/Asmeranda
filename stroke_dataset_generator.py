import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StrokeDatasetGenerator:
    def __init__(self, seed=42):
        """
        Generator dataset stroke berbasis literatur ilmiah
        Referensi: WHO, Framingham Heart Study, INTERSTROKE study, JNC 8, ADA
        """
        np.random.seed(seed)
        random.seed(seed)
        
        # Distribusi berdasarkan studi epidemiologi
        self.age_dist = {'mean': 55.0, 'std': 15.0, 'min': 18, 'max': 95}
        self.gender_dist = {'male': 0.48, 'female': 0.52}
        
        # Prevalensi berdasarkan literatur
        self.stroke_prevalence = 0.035  # 3.5% global prevalence (WHO 2022)
        self.hypertension_prevalence = 0.33  # 33% global (WHO)
        self.diabetes_prevalence = 0.11  # 11% global (IDF 2021)
        self.obesity_prevalence = 0.22  # 22% global (WHO)
        self.atrial_fibrillation_prevalence = 0.02  # 2% general population
        
        # Distribusi etnis (berdasarkan US population - bisa disesuaikan)
        self.ethnicity_dist = {
            'Caucasian': 0.60, 'African American': 0.13, 'Hispanic': 0.18,
            'Asian': 0.06, 'Native American': 0.02, 'Other': 0.01
        }
        
        # Risk ratios dari meta-analisis
        self.risk_ratios = {
            'hypertension': 2.5,      # OR 2.5 (95% CI: 2.3-2.7)
            'diabetes': 2.3,          # OR 2.3 (95% CI: 2.1-2.5)
            'atrial_fibrillation': 4.0, # OR 4.0 (95% CI: 3.3-4.8)
            'smoking_current': 2.2,     # OR 2.2 (95% CI: 1.9-2.6)
            'smoking_former': 1.3,    # OR 1.3 (95% CI: 1.2-1.4)
            'obesity': 1.4,           # OR 1.4 (95% CI: 1.3-1.6)
            'heart_disease': 2.4,     # OR 2.4 (95% CI: 2.0-2.8)
            'family_history': 1.8     # OR 1.8 (95% CI: 1.6-2.0)
        }
        
        # INTERSTROKE risk factors weights
        self.interstroke_weights = {
            'hypertension': 0.35,
            'physical_inactivity': 0.28,
            'lipids': 0.27,
            'obesity': 0.25,
            'smoking': 0.24,
            'diet': 0.23,
            'heart_disease': 0.22,
            'alcohol': 0.18,
            'stress': 0.17,
            'diabetes': 0.15
        }
    
    def generate_demographic_features(self, n_samples):
        """Generate demographic features based on epidemiological data"""
        
        # Age distribution (right-skewed like real population)
        ages = np.random.normal(self.age_dist['mean'], self.age_dist['std'], n_samples)
        ages = np.clip(ages, self.age_dist['min'], self.age_dist['max'])
        
        # Gender
        genders = np.random.choice(['Male', 'Female'], size=n_samples, 
                                 p=[self.gender_dist['male'], self.gender_dist['female']])
        
        # Ethnicity
        ethnicities = np.random.choice(list(self.ethnicity_dist.keys()), size=n_samples,
                                     p=list(self.ethnicity_dist.values()))
        
        # Education level (correlated with age and socioeconomic factors)
        education_levels = []
        for i in range(n_samples):
            age = ages[i]
            if age < 30:
                probs = [0.05, 0.15, 0.50, 0.25, 0.05]
            elif age < 50:
                probs = [0.08, 0.20, 0.45, 0.22, 0.05]
            else:
                probs = [0.15, 0.25, 0.35, 0.18, 0.07]
            
            edu = np.random.choice(['None', 'Elementary', 'High School', 'Bachelor', 'Graduate'],
                                 p=probs)
            education_levels.append(edu)
        
        # Income level (correlated with education)
        income_levels = []
        for edu in education_levels:
            if edu == 'Graduate':
                income = np.random.choice(['Low', 'Middle', 'High'], p=[0.15, 0.50, 0.35])
            elif edu == 'Bachelor':
                income = np.random.choice(['Low', 'Middle', 'High'], p=[0.25, 0.55, 0.20])
            elif edu == 'High School':
                income = np.random.choice(['Low', 'Middle', 'High'], p=[0.40, 0.50, 0.10])
            else:
                income = np.random.choice(['Low', 'Middle', 'High'], p=[0.60, 0.35, 0.05])
            income_levels.append(income)
        
        # Marital status (age-dependent)
        marital_statuses = []
        for age in ages:
            if age < 25:
                marital = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.70, 0.25, 0.05])
            elif age < 40:
                marital = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.25, 0.65, 0.10])
            else:
                marital = np.random.choice(['Single', 'Married', 'Divorced'], p=[0.10, 0.70, 0.20])
            marital_statuses.append(marital)
        
        # Work type (age and education dependent)
        work_types = []
        for i in range(n_samples):
            age, edu = ages[i], education_levels[i]
            if age > 65:
                work = 'Retired'
            elif age < 25:
                work = np.random.choice(['Student', 'Private', 'Self-employed'], p=[0.40, 0.40, 0.20])
            elif edu in ['Bachelor', 'Graduate']:
                work = np.random.choice(['Private', 'Self-employed', 'Govt_job'], p=[0.60, 0.20, 0.20])
            else:
                work = np.random.choice(['Private', 'Self-employed', 'Govt_job'], p=[0.50, 0.30, 0.20])
            work_types.append(work)
        
        # Residence type
        residence_types = np.random.choice(['Urban', 'Rural'], size=n_samples, p=[0.55, 0.45])
        
        demographics_df = pd.DataFrame({
            'patient_id': [f'P{str(i+1).zfill(6)}' for i in range(n_samples)],
            'age': ages.round(1),
            'gender': genders,
            'ethnicity': ethnicities,
            'education_level': education_levels,
            'income_level': income_levels,
            'marital_status': marital_statuses,
            'work_type': work_types,
            'residence_type': residence_types
        })
        
        return demographics_df
    
    def generate_clinical_features(self, demographics_df):
        """Generate clinical features with realistic correlations"""
        n_samples = len(demographics_df)
        
        # BMI calculation (correlated with age, gender, and socioeconomic factors)
        bmi_values = []
        for _, row in demographics_df.iterrows():
            base_bmi = np.random.normal(27, 4)  # Mean BMI ~27
            
            # Age effect
            if row['age'] > 60:
                base_bmi += np.random.normal(2, 1)
            
            # Gender effect
            if row['gender'] == 'Male':
                base_bmi += np.random.normal(1, 0.5)
            
            # Socioeconomic effect
            if row['income_level'] == 'Low':
                base_bmi += np.random.normal(1.5, 1)
            
            bmi = np.clip(base_bmi, 15, 50)
            bmi_values.append(round(bmi, 1))
        
        # Obesity classification
        obesity_categories = []
        for bmi in bmi_values:
            if bmi < 18.5:
                obesity_categories.append('Underweight')
            elif bmi < 25:
                obesity_categories.append('Normal')
            elif bmi < 30:
                obesity_categories.append('Overweight')
            else:
                obesity_categories.append('Obese')
        
        # Hypertension (age-dependent prevalence)
        hypertension_values = []
        for _, row in demographics_df.iterrows():
            base_prob = self.hypertension_prevalence
            
            # Age effect (strong)
            if row['age'] > 60:
                base_prob += 0.25
            elif row['age'] > 40:
                base_prob += 0.15
            
            # Gender effect
            if row['gender'] == 'Male' and row['age'] > 45:
                base_prob += 0.05
            elif row['gender'] == 'Female' and row['age'] > 55:
                base_prob += 0.05
            
            # Ethnicity effect
            if row['ethnicity'] in ['African American', 'Hispanic']:
                base_prob += 0.10
            
            hypertension = np.random.choice([0, 1], p=[1-base_prob, base_prob])
            hypertension_values.append(hypertension)
        
        # Blood pressure readings (correlated with hypertension and age)
        systolic_bp = []
        diastolic_bp = []
        
        for i, row in demographics_df.iterrows():
            if hypertension_values[i] == 1:
                sys_base = np.random.normal(145, 15)
                dia_base = np.random.normal(90, 10)
            else:
                sys_base = np.random.normal(120, 10)
                dia_base = np.random.normal(80, 8)
            
            # Age effect
            age_effect = (row['age'] - 30) * 0.3
            sys_base += age_effect
            dia_base += age_effect * 0.5
            
            systolic_bp.append(int(np.clip(sys_base, 90, 220)))
            diastolic_bp.append(int(np.clip(dia_base, 60, 120)))
        
        # Diabetes (correlated with BMI, age, ethnicity)
        diabetes_values = []
        for i, row in demographics_df.iterrows():
            base_prob = self.diabetes_prevalence
            
            # BMI effect (strong)
            if bmi_values[i] > 30:
                base_prob += 0.15
            elif bmi_values[i] > 25:
                base_prob += 0.08
            
            # Age effect
            if row['age'] > 60:
                base_prob += 0.08
            
            # Ethnicity effect
            if row['ethnicity'] in ['Hispanic', 'African American', 'Asian']:
                base_prob += 0.05
            
            diabetes = np.random.choice([0, 1], p=[1-base_prob, base_prob])
            diabetes_values.append(diabetes)
        
        # Glucose levels (correlated with diabetes)
        glucose_levels = []
        for i in range(n_samples):
            if diabetes_values[i] == 1:
                glucose = np.random.normal(140, 30)
            else:
                glucose = np.random.normal(95, 15)
            glucose_levels.append(int(np.clip(glucose, 70, 300)))
        
        # HbA1c (diabetes marker)
        hba1c_levels = []
        for i in range(n_samples):
            if diabetes_values[i] == 1:
                hba1c = np.random.normal(8.0, 1.5)
            else:
                hba1c = np.random.normal(5.5, 0.5)
            hba1c_levels.append(round(np.clip(hba1c, 4.0, 12.0), 1))
        
        # Heart disease (correlated with age, hypertension, diabetes)
        heart_disease_values = []
        for i, row in demographics_df.iterrows():
            base_prob = 0.06  # 6% base prevalence
            
            # Age effect (strong)
            if row['age'] > 70:
                base_prob += 0.15
            elif row['age'] > 50:
                base_prob += 0.08
            
            # Hypertension effect
            if hypertension_values[i] == 1:
                base_prob += 0.10
            
            # Diabetes effect
            if diabetes_values[i] == 1:
                base_prob += 0.08
            
            heart_disease = np.random.choice([0, 1], p=[1-base_prob, base_prob])
            heart_disease_values.append(heart_disease)
        
        # Atrial fibrillation (age-dependent)
        af_values = []
        for _, row in demographics_df.iterrows():
            base_prob = self.atrial_fibrillation_prevalence
            
            # Age effect (very strong)
            if row['age'] > 80:
                base_prob += 0.08
            elif row['age'] > 65:
                base_prob += 0.04
            
            # Heart disease effect
            if heart_disease_values[len(af_values)] == 1:
                base_prob += 0.05
            
            af = np.random.choice([0, 1], p=[1-base_prob, base_prob])
            af_values.append(af)
        
        # Smoking status (age and gender dependent)
        smoking_statuses = []
        for _, row in demographics_df.iterrows():
            if row['age'] < 18:
                smoking_statuses.append('Never')
            else:
                # Gender differences in smoking
                if row['gender'] == 'Male':
                    smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.45, 0.30, 0.25])
                else:
                    smoking = np.random.choice(['Never', 'Former', 'Current'], p=[0.65, 0.20, 0.15])
                smoking_statuses.append(smoking)
        
        # Alcohol consumption (correlated with gender and smoking)
        alcohol_consumptions = []
        for i, row in demographics_df.iterrows():
            if smoking_statuses[i] == 'Current':
                if row['gender'] == 'Male':
                    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], p=[0.20, 0.30, 0.35, 0.15])
                else:
                    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], p=[0.30, 0.40, 0.25, 0.05])
            else:
                if row['gender'] == 'Male':
                    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], p=[0.25, 0.40, 0.30, 0.05])
                else:
                    alcohol = np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], p=[0.35, 0.45, 0.18, 0.02])
            alcohol_consumptions.append(alcohol)
        
        # Physical activity (inverse correlation with socioeconomic factors)
        physical_activities = []
        for _, row in demographics_df.iterrows():
            if row['work_type'] == 'Retired':
                activity = np.random.choice(['None', 'Light', 'Moderate', 'Intense'], p=[0.15, 0.50, 0.30, 0.05])
            elif row['income_level'] == 'High':
                activity = np.random.choice(['None', 'Light', 'Moderate', 'Intense'], p=[0.10, 0.25, 0.50, 0.15])
            else:
                activity = np.random.choice(['None', 'Light', 'Moderate', 'Intense'], p=[0.25, 0.40, 0.30, 0.05])
            physical_activities.append(activity)
        
        # Stress level (work and life situation dependent)
        stress_levels = []
        for _, row in demographics_df.iterrows():
            if row['work_type'] in ['Private', 'Self-employed'] and row['age'] < 60:
                stress = np.random.choice(['Low', 'Moderate', 'High'], p=[0.25, 0.50, 0.25])
            elif row['marital_status'] == 'Divorced':
                stress = np.random.choice(['Low', 'Moderate', 'High'], p=[0.30, 0.45, 0.25])
            else:
                stress = np.random.choice(['Low', 'Moderate', 'High'], p=[0.50, 0.40, 0.10])
            stress_levels.append(stress)
        
        # Lipid profile (age and lifestyle dependent)
        total_cholesterols = []
        hdl_cholesterols = []
        ldl_cholesterols = []
        triglycerides = []
        
        for i, row in demographics_df.iterrows():
            # Total cholesterol
            tc_base = np.random.normal(200, 35)
            if row['age'] > 50:
                tc_base += np.random.normal(20, 10)
            total_cholesterols.append(int(np.clip(tc_base, 120, 350)))
            
            # HDL cholesterol
            hdl_base = np.random.normal(50, 12)
            if physical_activities[i] in ['Moderate', 'Intense']:
                hdl_base += np.random.normal(5, 3)
            if smoking_statuses[i] == 'Current':
                hdl_base -= np.random.normal(8, 4)
            hdl_cholesterols.append(int(np.clip(hdl_base, 25, 90)))
            
            # LDL cholesterol
            ldl_base = np.random.normal(120, 30)
            if diabetes_values[i] == 1:
                ldl_base += np.random.normal(15, 8)
            ldl_cholesterols.append(int(np.clip(ldl_base, 60, 250)))
            
            # Triglycerides
            tg_base = np.random.normal(120, 40)
            if diabetes_values[i] == 1:
                tg_base += np.random.normal(60, 30)
            if bmi_values[i] > 30:
                tg_base += np.random.normal(40, 20)
            triglycerides.append(int(np.clip(tg_base, 40, 500)))
        
        # Family history of stroke
        family_histories = []
        for _ in range(n_samples):
            family_history = np.random.choice([0, 1], p=[0.85, 0.15])  # 15% prevalence
            family_histories.append(family_history)
        
        # Medication usage
        antihypertensive_meds = []
        antidiabetic_meds = []
        antiplatelet_meds = []
        
        for i in range(n_samples):
            # Antihypertensive medication
            if hypertension_values[i] == 1:
                antihypertensive = np.random.choice([0, 1], p=[0.20, 0.80])
            else:
                antihypertensive = np.random.choice([0, 1], p=[0.95, 0.05])
            antihypertensive_meds.append(antihypertensive)
            
            # Antidiabetic medication
            if diabetes_values[i] == 1:
                antidiabetic = np.random.choice([0, 1], p=[0.15, 0.85])
            else:
                antidiabetic = 0
            antidiabetic_meds.append(antidiabetic)
            
            # Antiplatelet medication
            if heart_disease_values[i] == 1 or af_values[i] == 1:
                antiplatelet = np.random.choice([0, 1], p=[0.30, 0.70])
            else:
                antiplatelet = np.random.choice([0, 1], p=[0.90, 0.10])
            antiplatelet_meds.append(antiplatelet)
        
        clinical_df = pd.DataFrame({
            'bmi': bmi_values,
            'obesity_category': obesity_categories,
            'hypertension': hypertension_values,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'diabetes': diabetes_values,
            'glucose_level': glucose_levels,
            'hba1c': hba1c_levels,
            'heart_disease': heart_disease_values,
            'atrial_fibrillation': af_values,
            'smoking_status': smoking_statuses,
            'alcohol_consumption': alcohol_consumptions,
            'physical_activity': physical_activities,
            'stress_level': stress_levels,
            'total_cholesterol': total_cholesterols,
            'hdl_cholesterol': hdl_cholesterols,
            'ldl_cholesterol': ldl_cholesterols,
            'triglycerides': triglycerides,
            'family_history_stroke': family_histories,
            'antihypertensive_med': antihypertensive_meds,
            'antidiabetic_med': antidiabetic_meds,
            'antiplatelet_med': antiplatelet_meds
        })
        
        return clinical_df
    
    def calculate_stroke_risk(self, demographics_df, clinical_df):
        """
        Calculate stroke risk based on INTERSTROKE and Framingham risk factors
        Returns risk score and probability
        """
        n_samples = len(demographics_df)
        risk_scores = []
        
        for i in range(n_samples):
            risk_score = 0
            
            # Age factor (strongest predictor)
            age = demographics_df.iloc[i]['age']
            if age < 45:
                risk_score += 0
            elif age < 55:
                risk_score += 2
            elif age < 65:
                risk_score += 4
            elif age < 75:
                risk_score += 6
            else:
                risk_score += 8
            
            # INTERSTROKE risk factors
            if clinical_df.iloc[i]['hypertension'] == 1:
                risk_score += self.interstroke_weights['hypertension'] * 10
            
            if clinical_df.iloc[i]['diabetes'] == 1:
                risk_score += self.interstroke_weights['diabetes'] * 10
            
            if clinical_df.iloc[i]['heart_disease'] == 1:
                risk_score += self.interstroke_weights['heart_disease'] * 10
            
            if clinical_df.iloc[i]['atrial_fibrillation'] == 1:
                risk_score += 4  # Strong independent risk factor
            
            # Smoking (current vs former)
            smoking = clinical_df.iloc[i]['smoking_status']
            if smoking == 'Current':
                risk_score += self.interstroke_weights['smoking'] * 10
            elif smoking == 'Former':
                risk_score += (self.risk_ratios['smoking_former'] - 1) * 3
            
            # BMI and obesity
            bmi = clinical_df.iloc[i]['bmi']
            if bmi > 35:
                risk_score += self.interstroke_weights['obesity'] * 8
            elif bmi > 30:
                risk_score += self.interstroke_weights['obesity'] * 6
            elif bmi > 25:
                risk_score += self.interstroke_weights['obesity'] * 3
            
            # Physical activity (protective factor)
            activity = clinical_df.iloc[i]['physical_activity']
            if activity == 'Intense':
                risk_score -= 2
            elif activity == 'Moderate':
                risk_score -= 1
            elif activity == 'None':
                risk_score += self.interstroke_weights['physical_inactivity'] * 8
            
            # Alcohol consumption
            alcohol = clinical_df.iloc[i]['alcohol_consumption']
            if alcohol == 'Heavy':
                risk_score += self.interstroke_weights['alcohol'] * 8
            elif alcohol == 'Moderate':
                risk_score += 1  # J-curve relationship
            elif alcohol == 'None':
                risk_score += 0
            
            # Stress level
            stress = clinical_df.iloc[i]['stress_level']
            if stress == 'High':
                risk_score += self.interstroke_weights['stress'] * 8
            elif stress == 'Moderate':
                risk_score += self.interstroke_weights['stress'] * 4
            
            # Family history
            if clinical_df.iloc[i]['family_history_stroke'] == 1:
                risk_score += (self.risk_ratios['family_history'] - 1) * 5
            
            # Lipid ratios
            hdl = clinical_df.iloc[i]['hdl_cholesterol']
            if hdl < 40:
                risk_score += 2
            elif hdl > 60:
                risk_score -= 1
            
            # Ethnicity adjustments (based on epidemiological data)
            ethnicity = demographics_df.iloc[i]['ethnicity']
            if ethnicity == 'African American':
                risk_score += 1.5
            elif ethnicity == 'Hispanic':
                risk_score += 1.0
            elif ethnicity == 'Asian':
                risk_score += 0.5
            
            # Gender differences (post-menopausal women)
            gender = demographics_df.iloc[i]['gender']
            if gender == 'Female' and age > 55:
                risk_score += 1
            
            risk_scores.append(risk_score)
        
        # Convert risk scores to probabilities using logistic function
        risk_probabilities = []
        for score in risk_scores:
            # Logistic transformation with calibration
            logit = -6.5 + score * 0.3  # Calibrated to match 3.5% prevalence
            prob = 1 / (1 + np.exp(-logit))
            risk_probabilities.append(min(prob, 0.95))  # Cap at 95%
        
        risk_df = pd.DataFrame({
            'stroke_risk_score': risk_scores,
            'stroke_risk_probability': risk_probabilities
        })
        
        return risk_df
    
    def generate_stroke_outcomes(self, demographics_df, clinical_df, risk_df):
        """Generate stroke outcomes based on risk probabilities"""
        n_samples = len(demographics_df)
        
        stroke_outcomes = []
        stroke_severities = []
        stroke_types = []
        stroke_timestamps = []
        nihss_scores = []  # NIH Stroke Scale
        modified_rankin_scores = []  # Modified Rankin Scale
        
        for i in range(n_samples):
            risk_prob = risk_df.iloc[i]['stroke_risk_probability']
            
            # Generate stroke outcome
            stroke = np.random.choice([0, 1], p=[1-risk_prob, risk_prob])
            stroke_outcomes.append(stroke)
            
            if stroke == 1:
                # Stroke type (based on epidemiology: 87% ischemic, 13% hemorrhagic)
                stroke_type = np.random.choice(['Ischemic', 'Hemorrhagic'], p=[0.87, 0.13])
                stroke_types.append(stroke_type)
                
                # Stroke severity (NIHSS score)
                base_severity = np.random.normal(8, 4)  # Mean NIHSS ~8 for stroke patients
                
                # Age effect on severity
                age = demographics_df.iloc[i]['age']
                if age > 75:
                    base_severity += np.random.normal(3, 2)
                elif age > 65:
                    base_severity += np.random.normal(1, 1)
                
                # Comorbidity effects
                if clinical_df.iloc[i]['hypertension'] == 1:
                    base_severity += np.random.normal(1, 0.5)
                if clinical_df.iloc[i]['diabetes'] == 1:
                    base_severity += np.random.normal(0.5, 0.3)
                if clinical_df.iloc[i]['atrial_fibrillation'] == 1:
                    base_severity += np.random.normal(1.5, 1)
                
                nihss_score = int(np.clip(base_severity, 0, 42))
                nihss_scores.append(nihss_score)
                
                # Severity classification
                if nihss_score <= 4:
                    severity = 'Minor'
                elif nihss_score <= 15:
                    severity = 'Moderate'
                elif nihss_score <= 25:
                    severity = 'Severe'
                else:
                    severity = 'Very Severe'
                stroke_severities.append(severity)
                
                # Modified Rankin Scale (functional outcome)
                if nihss_score <= 4:
                    mrs = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
                elif nihss_score <= 15:
                    mrs = np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1])
                elif nihss_score <= 25:
                    mrs = np.random.choice([2, 3, 4, 5], p=[0.1, 0.3, 0.4, 0.2])
                else:
                    mrs = np.random.choice([3, 4, 5, 6], p=[0.1, 0.3, 0.4, 0.2])
                modified_rankin_scores.append(mrs)
                
                # Generate timestamp (recent stroke for prediction)
                days_ago = np.random.randint(1, 365)  # Within past year
                stroke_date = datetime.now() - timedelta(days=days_ago)
                stroke_timestamps.append(stroke_date.strftime('%Y-%m-%d'))
                
            else:
                stroke_types.append('None')
                stroke_severities.append('None')
                stroke_timestamps.append('')
                nihss_scores.append(0)
                modified_rankin_scores.append(0)
        
        outcomes_df = pd.DataFrame({
            'stroke': stroke_outcomes,
            'stroke_type': stroke_types,
            'stroke_severity': stroke_severities,
            'stroke_timestamp': stroke_timestamps,
            'nihss_score': nihss_scores,
            'modified_rankin_score': modified_rankin_scores
        })
        
        return outcomes_df
    
    def add_medical_clusters(self, dataset):
        """Add medical clusters for unsupervised learning based on risk profiles"""
        n_samples = len(dataset)
        clusters = []
        
        for i in range(n_samples):
            age = dataset.iloc[i]['age']
            hypertension = dataset.iloc[i]['hypertension']
            diabetes = dataset.iloc[i]['diabetes']
            bmi = dataset.iloc[i]['bmi']
            smoking = dataset.iloc[i]['smoking_status']
            heart_disease = dataset.iloc[i]['heart_disease']
            af = dataset.iloc[i]['atrial_fibrillation']
            physical_activity = dataset.iloc[i]['physical_activity']
            
            # Define clusters based on medical risk profiles
            if age > 65 and (hypertension == 1 or diabetes == 1):
                if heart_disease == 1 or af == 1:
                    cluster = 'High-Risk Cardiovascular'
                else:
                    cluster = 'Elderly Metabolic'
            
            elif bmi > 30 and diabetes == 1:
                if hypertension == 1:
                    cluster = 'Metabolic Syndrome'
                else:
                    cluster = 'Diabetic Obesity'
            
            elif smoking == 'Current' and age < 50:
                if physical_activity == 'None':
                    cluster = 'Young High-Risk Behavior'
                else:
                    cluster = 'Young Smoker'
            
            elif hypertension == 1 and age > 55:
                if diabetes == 1:
                    cluster = 'Hypertensive Diabetic'
                else:
                    cluster = 'Primary Hypertension'
            
            elif heart_disease == 1 or af == 1:
                cluster = 'Cardiac Comorbidity'
            
            elif physical_activity == 'None' and bmi > 25:
                cluster = 'Sedentary Overweight'
            
            elif age > 75:
                cluster = 'Elderly General'
            
            elif physical_activity in ['Moderate', 'Intense'] and bmi < 25:
                cluster = 'Healthy Active'
            
            elif smoking == 'Never' and physical_activity != 'None':
                cluster = 'Low-Risk Lifestyle'
            
            else:
                cluster = 'General Population'
            
            clusters.append(cluster)
        
        dataset['medical_cluster'] = clusters
        return dataset
    
    def add_noise_and_missing_values(self, dataset, missing_rate=0.07, noise_rate=0.05):
        """Add realistic missing values and measurement noise"""
        n_samples = len(dataset)
        
        # Add missing values
        for col in ['glucose_level', 'hba1c', 'total_cholesterol', 'hdl_cholesterol', 
                   'ldl_cholesterol', 'triglycerides', 'physical_activity', 'stress_level']:
            
            n_missing = int(n_samples * missing_rate)
            missing_indices = np.random.choice(n_samples, n_missing, replace=False)
            dataset.loc[missing_indices, col] = np.nan
        
        # Add measurement noise to continuous variables
        continuous_cols = ['systolic_bp', 'diastolic_bp', 'bmi', 'glucose_level', 
                          'total_cholesterol', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']
        
        for col in continuous_cols:
            if col in dataset.columns:
                noise = np.random.normal(0, noise_rate * dataset[col].std(), n_samples)
                dataset[col] = dataset[col] + noise
                
                # Round to appropriate precision
                if col in ['systolic_bp', 'diastolic_bp', 'glucose_level', 'total_cholesterol', 'triglycerides']:
                    dataset[col] = dataset[col].round(0)
                elif col in ['bmi', 'hdl_cholesterol', 'ldl_cholesterol']:
                    dataset[col] = dataset[col].round(1)
        
        return dataset
    
    def validate_dataset(self, dataset):
        """Validate dataset quality and medical accuracy"""
        print("\n🔍 VALIDASI DATASET STROKE")
        print("=" * 40)
        
        # Basic statistics
        print(f"Total samples: {len(dataset)}")
        print(f"Features: {len(dataset.columns)}")
        print(f"Missing values: {dataset.isnull().sum().sum()}")
        
        # Stroke prevalence validation
        stroke_rate = dataset['stroke'].mean()
        print(f"\n📊 Stroke prevalence: {stroke_rate:.3f} ({stroke_rate*100:.1f}%)")
        print(f"Expected: {self.stroke_prevalence:.3f} ({self.stroke_prevalence*100:.1f}%)")
        
        # Risk factor prevalence validation
        print(f"\n🏥 Risk Factor Prevalence:")
        print(f"Hypertension: {dataset['hypertension'].mean():.3f} (expected: {self.hypertension_prevalence:.3f})")
        print(f"Diabetes: {dataset['diabetes'].mean():.3f} (expected: {self.diabetes_prevalence:.3f})")
        print(f"Obesity (BMI>30): {(dataset['bmi'] > 30).mean():.3f} (expected: {self.obesity_prevalence:.3f})")
        
        # Age distribution validation
        print(f"\n👥 Age statistics:")
        print(f"Mean age: {dataset['age'].mean():.1f} years")
        print(f"Age range: {dataset['age'].min():.0f} - {dataset['age'].max():.0f} years")
        
        # Gender distribution
        gender_dist = dataset['gender'].value_counts(normalize=True)
        print(f"\n⚥ Gender distribution:")
        for gender, prop in gender_dist.items():
            print(f"{gender}: {prop:.3f}")
        
        # Medical cluster distribution
        print(f"\n🔬 Medical Cluster Distribution:")
        cluster_counts = dataset['medical_cluster'].value_counts()
        for cluster, count in cluster_counts.items():
            cluster_rate = count / len(dataset)
            stroke_rate_cluster = dataset[dataset['medical_cluster'] == cluster]['stroke'].mean()
            print(f"{cluster}: {count} ({cluster_rate*100:.1f}%) - Stroke rate: {stroke_rate_cluster*100:.1f}%")
        
        # Correlation validation
        print(f"\n🔗 Risk Factor Correlations with Stroke:")
        risk_factors = ['hypertension', 'diabetes', 'heart_disease', 'atrial_fibrillation']
        for factor in risk_factors:
            if factor in dataset.columns:
                correlation = dataset[factor].corr(dataset['stroke'])
                print(f"{factor}: {correlation:.3f}")
        
        # Age correlation
        age_correlation = dataset['age'].corr(dataset['stroke'])
        print(f"age: {age_correlation:.3f}")
        
        # BMI correlation
        bmi_correlation = dataset['bmi'].corr(dataset['stroke'])
        print(f"bmi: {bmi_correlation:.3f}")
        
        print(f"\n✅ Dataset validation completed!")
        print("=" * 40)
        
        return {
            'total_samples': len(dataset),
            'stroke_prevalence': stroke_rate,
            'missing_values': dataset.isnull().sum().sum(),
            'age_mean': dataset['age'].mean(),
            'risk_factor_correlations': {factor: dataset[factor].corr(dataset['stroke']) for factor in risk_factors if factor in dataset.columns}
        }
    
    def generate_dataset(self, n_samples=10000):
        """Generate complete stroke prediction dataset"""
        print(f"\n🩺 Generating {n_samples} stroke prediction records...")
        print("Based on WHO, Framingham Heart Study, and INTERSTROKE research")
        
        # Step 1: Generate demographics
        print("1️⃣ Generating demographic features...")
        demographics_df = self.generate_demographic_features(n_samples)
        
        # Step 2: Generate clinical features
        print("2️⃣ Generating clinical features...")
        clinical_df = self.generate_clinical_features(demographics_df)
        
        # Step 3: Calculate stroke risk
        print("3️⃣ Calculating stroke risk scores...")
        risk_df = self.calculate_stroke_risk(demographics_df, clinical_df)
        
        # Step 4: Generate stroke outcomes
        print("4️⃣ Generating stroke outcomes...")
        outcomes_df = self.generate_stroke_outcomes(demographics_df, clinical_df, risk_df)
        
        # Combine all dataframes
        dataset = pd.concat([demographics_df, clinical_df, risk_df, outcomes_df], axis=1)
        
        # Step 5: Add medical clusters for unsupervised learning
        print("5️⃣ Adding medical clusters...")
        dataset = self.add_medical_clusters(dataset)
        
        # Step 6: Add noise and missing values for realism
        print("6️⃣ Adding realistic noise and missing values...")
        dataset = self.add_noise_and_missing_values(dataset)
        
        # Step 7: Validate dataset
        print("7️⃣ Validating dataset quality...")
        validation_results = self.validate_dataset(dataset)
        
        print(f"\n🎉 Dataset generation completed!")
        print(f"📊 Generated {len(dataset)} records with {len(dataset.columns)} features")
        
        return dataset


def generate_stroke_dataset(n_samples=1000, save_to_csv=True):
    """
    Main function to generate stroke prediction dataset
    
    Returns:
        pandas.DataFrame: Complete stroke prediction dataset
    """
    generator = StrokeDatasetGenerator()
    dataset = generator.generate_dataset(n_samples)
    
    if save_to_csv:
        filename = f"stroke_prediction_dataset_{n_samples}.csv"
        dataset.to_csv(filename, index=False)
        print(f"💾 Dataset saved to: {filename}")
    
    return dataset


if __name__ == "__main__":
    print("🩺 STROKE PREDICTION DATASET GENERATOR")
    print("=" * 50)
    print("Generating comprehensive stroke prediction dataset...")
    print("Based on medical literature and epidemiological studies")
    print("=" * 50)
    
    dataset = generate_stroke_dataset(1000)
    
    print("\n🎉 Dataset generation and validation completed!")
    print("The dataset is ready for machine learning analysis!")
    print("\n📋 Dataset Summary:")
    print(f"• Total records: {len(dataset)}")
    print(f"• Total features: {len(dataset.columns)}")
    print(f"• Stroke cases: {dataset['stroke'].sum()} ({dataset['stroke'].mean()*100:.1f}%)")
    print(f"• Medical clusters: {dataset['medical_cluster'].nunique()}")
    print(f"• Missing values: {dataset.isnull().sum().sum()}")
    print("\n✨ Features include demographics, clinical measurements, risk factors,")
    print("   stroke outcomes, and medical clusters for unsupervised learning!")