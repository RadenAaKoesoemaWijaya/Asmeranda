import streamlit as st
import pandas as pd
from typing import Dict, List, Optional

class WorkflowValidator:
    """Comprehensive workflow validation with detailed error reporting"""
    
    def __init__(self):
        self.validation_rules = {
            'upload_to_eda': {
                'required': ['data'],
                'checks': [
                    self._check_data_not_empty,
                    self._check_minimum_rows,
                    self._check_minimum_columns
                ]
            },
            'eda_to_preprocessing': {
                'required': ['data', 'numerical_columns', 'categorical_columns'],
                'checks': [
                    self._check_column_consistency,
                    self._check_target_column_validity,
                    self._check_data_quality
                ]
            },
            'preprocessing_to_training': {
                'required': ['X_train', 'X_test', 'y_train', 'y_test', 'problem_type'],
                'checks': [
                    self._check_train_test_split,
                    self._check_feature_target_consistency,
                    self._check_problem_type_validity
                ]
            },
            'training_to_interpretation': {
                'required': ['model_results'],
                'checks': [
                    self._check_model_results_validity,
                    self._check_model_availability_for_interpretation
                ]
            }
        }
    
    def validate_workflow_transition(self, from_step: str, to_step: str) -> Dict:
        """Validate workflow transition between steps"""
        transition_key = f"{from_step}_to_{to_step}"
        
        if transition_key not in self.validation_rules:
            return {
                'valid': False,
                'errors': [f"Unknown workflow transition: {transition_key}"],
                'warnings': [],
                'recommendations': [],
                'missing': {}
            }
        
        rule = self.validation_rules[transition_key]
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': [],
            'missing': {}
        }
        
        # Check required session state variables
        for required_key in rule['required']:
            if not self._check_session_state_exists(required_key):
                result['errors'].append(f"Missing required data: {required_key}")
                result['missing'][required_key] = f"Missing required data: {required_key}"
                result['valid'] = False
        
        # Run specific validation checks
        for check_func in rule['checks']:
            try:
                check_result = check_func()
                if not check_result['valid']:
                    result['valid'] = False
                    result['errors'].extend(check_result.get('errors', []))
                result['warnings'].extend(check_result.get('warnings', []))
                result['recommendations'].extend(check_result.get('recommendations', []))
            except Exception as e:
                result['valid'] = False
                result['errors'].append(f"Validation check failed: {str(e)}")
        
        return result
    
    def _check_session_state_exists(self, key: str) -> bool:
        """Check if a session state key exists and is not None/empty"""
        if key not in st.session_state:
            return False
        
        value = st.session_state[key]
        
        if value is None:
            return False
        
        if isinstance(value, (list, dict, str)) and len(value) == 0:
            return False
        
        return True
    
    def _check_data_not_empty(self) -> Dict:
        """Check if data is not empty"""
        data = st.session_state.get('data')
        
        if data is None or len(data) == 0:
            return {
                'valid': False,
                'errors': ['Dataset is empty or not loaded'],
                'warnings': [],
                'recommendations': ['Please upload a valid dataset']
            }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_minimum_rows(self) -> Dict:
        """Check minimum row requirement"""
        data = st.session_state.get('data')
        
        if data is not None and len(data) < 10:
            return {
                'valid': False,
                'errors': ['Dataset has insufficient rows (minimum 10 required)'],
                'warnings': [],
                'recommendations': ['Upload a dataset with at least 10 rows']
            }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_minimum_columns(self) -> Dict:
        """Check minimum column requirement"""
        data = st.session_state.get('data')
        
        if data is not None and len(data.columns) < 2:
            return {
                'valid': False,
                'errors': ['Dataset needs at least 2 columns (1 feature + 1 target)'],
                'warnings': [],
                'recommendations': ['Upload a dataset with multiple columns']
            }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_column_consistency(self) -> Dict:
        """Check consistency between data and column lists"""
        data = st.session_state.get('data')
        numerical_cols = st.session_state.get('numerical_columns', [])
        categorical_cols = st.session_state.get('categorical_columns', [])
        
        errors = []
        warnings = []
        
        if data is not None:
            # Check if columns exist in data
            missing_numerical = [col for col in numerical_cols if col not in data.columns]
            missing_categorical = [col for col in categorical_cols if col not in data.columns]
            
            if missing_numerical:
                errors.append(f"Numerical columns not found in data: {missing_numerical}")
            
            if missing_categorical:
                errors.append(f"Categorical columns not found in data: {missing_categorical}")
            
            # Check for overlap
            overlap = set(numerical_cols) & set(categorical_cols)
            if overlap:
                warnings.append(f"Columns appear in both numerical and categorical lists: {overlap}")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'recommendations': []
        }
    
    def _check_target_column_validity(self) -> Dict:
        """Check if target column is valid"""
        data = st.session_state.get('data')
        target_column = st.session_state.get('target_column')
        
        if data is not None and target_column:
            if target_column not in data.columns:
                return {
                    'valid': False,
                    'errors': [f"Target column '{target_column}' not found in dataset"],
                    'warnings': [],
                    'recommendations': ['Select a valid target column']
                }
            
            # Check target column characteristics
            target_data = data[target_column]
            null_percentage = target_data.isnull().sum() / len(target_data)
            
            if null_percentage > 0.5:
                return {
                    'valid': False,
                    'errors': [f"Target column has too many missing values ({null_percentage:.1%})"],
                    'warnings': [],
                    'recommendations': ['Handle missing values in target column or choose different target']
                }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_data_quality(self) -> Dict:
        """Check overall data quality"""
        data = st.session_state.get('data')
        
        if data is None:
            return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
        
        warnings = []
        recommendations = []
        
        # Check for high missing value percentage
        missing_pct = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if missing_pct > 0.3:
            warnings.append(f"High percentage of missing values ({missing_pct:.1%})")
            recommendations.append("Consider imputation or removal of columns with many missing values")
        
        # Check for duplicate rows
        duplicate_pct = data.duplicated().sum() / len(data)
        if duplicate_pct > 0.1:
            warnings.append(f"High percentage of duplicate rows ({duplicate_pct:.1%})")
            recommendations.append("Consider removing duplicate rows")
        
        return {
            'valid': True,
            'errors': [],
            'warnings': warnings,
            'recommendations': recommendations
        }
    
    def _check_train_test_split(self) -> Dict:
        """Check train-test split validity"""
        X_train = st.session_state.get('X_train')
        X_test = st.session_state.get('X_test')
        y_train = st.session_state.get('y_train')
        y_test = st.session_state.get('y_test')
        
        errors = []
        
        # Check if all components exist
        components = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
        
        for name, component in components.items():
            if component is None:
                errors.append(f"Missing {name}")
            elif hasattr(component, '__len__') and len(component) == 0:
                errors.append(f"Empty {name}")
        
        # Check consistency
        if X_train is not None and y_train is not None:
            if len(X_train) != len(y_train):
                errors.append("X_train and y_train have different lengths")
        
        if X_test is not None and y_test is not None:
            if len(X_test) != len(y_test):
                errors.append("X_test and y_test have different lengths")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': [],
            'recommendations': []
        }
    
    def _check_feature_target_consistency(self) -> Dict:
        """Check consistency between features and target"""
        X_train = st.session_state.get('X_train')
        X_test = st.session_state.get('X_test')
        
        errors = []
        
        if X_train is not None and X_test is not None:
            if hasattr(X_train, 'columns') and hasattr(X_test, 'columns'):
                if list(X_train.columns) != list(X_test.columns):
                    errors.append("X_train and X_test have different feature columns")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': [],
            'recommendations': []
        }
    
    def _check_problem_type_validity(self) -> Dict:
        """Check if problem type is valid"""
        problem_type = st.session_state.get('problem_type')
        y_train = st.session_state.get('y_train')
        
        if problem_type is None:
            return {
                'valid': False,
                'errors': ['Problem type not specified'],
                'warnings': [],
                'recommendations': ['Select a problem type (Classification/Regression/Forecasting)']
            }
        
        if y_train is not None and problem_type in ['Classification', 'Regression']:
            unique_values = pd.Series(y_train).nunique()
            
            if problem_type == 'Classification' and unique_values < 2:
                return {
                    'valid': False,
                    'errors': ['Classification requires at least 2 unique target values'],
                    'warnings': [],
                    'recommendations': ['Check target column or change problem type']
                }
            
            if problem_type == 'Regression' and unique_values < 10:
                return {
                    'valid': False,
                    'errors': ['Regression requires continuous target values'],
                    'warnings': [],
                    'recommendations': ['Ensure target column is continuous or change problem type']
                }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_model_results_validity(self) -> Dict:
        """Check if model results are valid"""
        model_results = st.session_state.get('model_results')
        
        if model_results is None or len(model_results) == 0:
            return {
                'valid': False,
                'errors': ['No trained models found'],
                'warnings': [],
                'recommendations': ['Train at least one model before proceeding to interpretation']
            }
        
        return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
    
    def _check_model_availability_for_interpretation(self) -> Dict:
        """Check if models are available for interpretation"""
        model_results = st.session_state.get('model_results')
        
        if model_results is None:
            return {'valid': True, 'errors': [], 'warnings': [], 'recommendations': []}
        
        warnings = []
        
        # Check for models that support interpretation
        interpretable_models = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'Linear Regression']
        
        available_interpretable = False
        for result in model_results:
            if result.get('model_type') in interpretable_models:
                available_interpretable = True
                break
        
        if not available_interpretable:
            warnings.append("No interpretable models found. Consider training Random Forest, Gradient Boosting, or Linear models")
        
        return {
            'valid': True,
            'errors': [],
            'warnings': warnings,
            'recommendations': ['Train interpretable models for better SHAP/LIME analysis']
        }

    def validate_data_readiness(self, data: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]) -> List[Dict]:
        """Validate data readiness for machine learning workflows"""
        results = []
        
        # 1. Dataset size check
        if len(data) >= 100:
            results.append({'status': 'success', 'message': f"Dataset has sufficient rows ({len(data)}) for training."})
        elif len(data) >= 20:
            results.append({'status': 'warning', 'message': f"Dataset is relatively small ({len(data)} rows). Consider cross-validation."})
        else:
            results.append({'status': 'error', 'message': f"Dataset is very small ({len(data)} rows). Machine learning might not be effective."})
            
        # 2. Column count check
        if len(data.columns) >= 3:
            results.append({'status': 'success', 'message': f"Dataset has {len(data.columns)} columns, sufficient for feature engineering."})
        else:
            results.append({'status': 'warning', 'message': "Dataset has very few columns. Limit feature engineering potential."})
            
        # 3. Missing values check
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct == 0:
            results.append({'status': 'success', 'message': "No missing values detected in the dataset."})
        elif missing_pct < 0.1:
            results.append({'status': 'success', 'message': f"Low percentage of missing values ({missing_pct:.1%}). Easy to handle."})
        elif missing_pct < 0.3:
            results.append({'status': 'warning', 'message': f"Moderate percentage of missing values ({missing_pct:.1%}). Imputation recommended."})
        else:
            results.append({'status': 'error', 'message': f"High percentage of missing values ({missing_pct:.1%}). Data cleaning is critical."})
            
        # 4. Target column potential check
        if len(numerical_cols) > 0:
            results.append({'status': 'success', 'message': "Numeric columns detected, suitable for regression or time series."})
        
        if len(categorical_cols) > 0:
            results.append({'status': 'success', 'message': "Categorical columns detected, suitable for classification."})
            
        return results

    def check_ml_readiness(self, validation_results: List[Dict]) -> Dict:
        """Check if data is ready for ML workflows based on validation results"""
        errors = [r for r in validation_results if r['status'] == 'error']
        warnings = [r for r in validation_results if r['status'] == 'warning']
        
        available_workflows = ["Exploratory Data Analysis"]
        
        if not errors:
            available_workflows.extend(["Preprocessing", "Model Training", "Model Interpretation"])
            
        ready = len(errors) == 0
        
        if ready:
            if not warnings:
                message = "Dataset is highly optimal for machine learning."
            else:
                message = "Dataset is ready for machine learning with some considerations."
        else:
            message = "Dataset requires significant cleaning before machine learning."
            
        recommendations = []
        for r in errors + warnings:
            if 'imputation' in r['message'].lower() or 'missing' in r['message'].lower():
                recommendations.append("Apply missing value imputation in Preprocessing tab.")
            if 'small' in r['message'].lower():
                recommendations.append("Use robust models like Random Forest or Cross-Validation.")
            if 'cleaning' in r['message'].lower():
                recommendations.append("Review data quality in EDA tab.")
                
        return {
            'ready': ready,
            'message': message,
            'available_workflows': available_workflows,
            'recommendations': recommendations
        }

    def validate_eda_completeness(self, data: pd.DataFrame, numerical_cols: List[str], categorical_cols: List[str]) -> List[Dict]:
        """Validate EDA completeness for transition to ML"""
        results = []
        
        # 1. Missing values check
        missing_count = data.isnull().sum().sum()
        if missing_count == 0:
            results.append({'status': 'success', 'message': "Dataset is clean (no missing values)."})
        else:
            results.append({'status': 'warning', 'message': f"Dataset has {missing_count} missing values. Preprocessing recommended."})
            
        # 2. Correlation check
        if len(numerical_cols) > 1:
            results.append({'status': 'success', 'message': f"Correlation analysis possible for {len(numerical_cols)} numerical features."})
        else:
            results.append({'status': 'warning', 'message': "Too few numerical features for correlation analysis."})
            
        # 3. Distribution check
        if len(numerical_cols) > 0:
            results.append({'status': 'success', 'message': "Distribution analysis available for numerical features."})
            
        # 4. Outlier potential
        if len(numerical_cols) > 0:
            results.append({'status': 'info', 'message': "Consider checking for outliers in numerical features before training."})
            
        return results

    def check_eda_readiness(self, eda_validation: List[Dict]) -> Dict:
        """Check if EDA is ready for ML transition"""
        warnings = [r for r in eda_validation if r['status'] == 'warning']
        
        ready = True
        message = "Exploratory Data Analysis is sufficient for basic ML transition."
        available_transitions = ["Preprocessing", "Feature Engineering"]
        
        recommendations = []
        for r in warnings:
            if 'missing' in r['message'].lower():
                recommendations.append("Handle missing values in the Preprocessing tab.")
            if 'correlation' in r['message'].lower():
                recommendations.append("Add more numerical features if possible for better insight.")
                
        return {
            'ready': ready,
            'message': message,
            'available_transitions': available_transitions,
            'recommendations': recommendations
        }

    def validate_ml_training_readiness(self, X_train: pd.DataFrame, y_train: pd.Series, problem_type: str, model_type: str) -> List[Dict]:
        """Validate ML training readiness for a specific model"""
        results = []
        
        # 1. Data existence check
        if X_train is None or y_train is None:
            results.append({'status': 'error', 'message': "Training data (X or y) is missing."})
            return results
            
        # 2. Row count check
        if len(X_train) < 10:
            results.append({'status': 'error', 'message': f"Insufficient data for training ({len(X_train)} samples)."})
        elif len(X_train) < 50:
            results.append({'status': 'warning', 'message': f"Small dataset ({len(X_train)} samples). Results might be unstable."})
        else:
            results.append({'status': 'success', 'message': f"Training data has {len(X_train)} samples."})
            
        # 3. Target type check
        if problem_type == 'Classification':
            unique_targets = len(np.unique(y_train.dropna()))
            if unique_targets < 2:
                results.append({'status': 'error', 'message': "Classification requires at least 2 unique target classes."})
            else:
                results.append({'status': 'success', 'message': f"Found {unique_targets} classes for classification."})
        
        # 4. Model-specific checks
        if 'Linear' in model_type or 'Logistic' in model_type:
            results.append({'status': 'info', 'message': f"{model_type} assumes feature scaling. Check if scaling was applied."})
            
        if 'Gradient' in model_type and X_train.isnull().sum().sum() > 0:
            results.append({'status': 'warning', 'message': f"{model_type} might be sensitive to missing values."})
            
        return results