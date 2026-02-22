import streamlit as st
import pandas as pd
import pickle
from typing import Optional, Dict, Any

class SessionStateManager:
    """Centralized session state management with validation and persistence"""
    
    def __init__(self):
        self.required_keys = {
            'data': None,
            'processed_data': None,
            'target_column': None,
            'problem_type': None,
            'X_train': None,
            'X_test': None,
            'y_train': None,
            'y_test': None,
            'numerical_columns': [],
            'categorical_columns': [],
            'encoders': {},
            'scaler': None,
            'model_results': [],
            'is_time_series': False,
            'time_column': None,
            'clustering_results': None,
            'eda_insights': {}
        }
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize all required session state variables with validation"""
        for key, default_value in self.required_keys.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def validate_data_flow(self, from_tab: str, to_tab: str) -> Dict[str, Any]:
        """Validate data availability between workflow steps"""
        validation_rules = {
            ('upload', 'eda'): ['data'],
            ('eda', 'preprocessing'): ['data', 'numerical_columns', 'categorical_columns'],
            ('preprocessing', 'training'): ['X_train', 'X_test', 'y_train', 'y_test', 'problem_type'],
            ('training', 'interpretation'): ['model_results']
        }
        
        missing = []
        required_keys = validation_rules.get((from_tab, to_tab), [])
        
        for key in required_keys:
            if st.session_state.get(key) is None:
                if key in ['numerical_columns', 'categorical_columns'] and not st.session_state.get(key):
                    missing.append(key)
                elif key not in ['numerical_columns', 'categorical_columns']:
                    missing.append(key)
        
        return {
            'valid': len(missing) == 0,
            'missing': missing,
            'message': f"Missing required data: {', '.join(missing)}" if missing else "Valid"
        }
    
    def save_workflow_state(self, tab_name: str):
        """Save current workflow state for recovery"""
        state_snapshot = {}
        for key in self.required_keys:
            if key in st.session_state:
                # Handle non-serializable objects
                try:
                    pickle.dumps(st.session_state[key])
                    state_snapshot[key] = st.session_state[key]
                except (pickle.PicklingError, TypeError):
                    # Convert to serializable format
                    if hasattr(st.session_state[key], 'to_dict'):
                        state_snapshot[key] = st.session_state[key].to_dict()
                    else:
                        state_snapshot[key] = str(st.session_state[key])
        
        st.session_state[f'workflow_state_{tab_name}'] = state_snapshot
    
    def clear_workflow_data(self, keep_basic: bool = True):
        """Clear workflow data while preserving basic settings"""
        basic_keys = ['language', 'authenticated', 'current_username', 'user_email']
        
        keys_to_clear = [key for key in self.required_keys.keys() if key not in basic_keys]
        
        for key in keys_to_clear:
            if key in st.session_state:
                if keep_basic and key in basic_keys:
                    continue
                st.session_state[key] = self.required_keys[key]