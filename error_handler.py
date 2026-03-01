import streamlit as st
import traceback
import logging
import pandas as pd
from typing import Dict, Any, Optional, List

class ErrorHandler:
    """Centralized error handling with user-friendly messages"""
    
    def __init__(self, language: str = 'id'):
        self.language = language
        self.error_messages = self._load_error_messages()
        self.setup_logging()
    
    def _load_error_messages(self) -> Dict:
        """Load localized error messages"""
        return {
            'id': {
                'data_empty': "Dataset kosong atau tidak valid. Silakan unggah dataset yang valid.",
                'column_not_found': "Kolom '{column}' tidak ditemukan dalam dataset.",
                'model_training_failed': "Pelatihan model gagal: {error}. Silakan periksa parameter atau data.",
                'prediction_failed': "Prediksi gagal: {error}. Pastikan model telah dilatih dengan benar.",
                'invalid_parameter': "Parameter '{param}' tidak valid: {value}",
                'memory_error': "Memori tidak cukup untuk operasi ini. Coba dengan dataset yang lebih kecil.",
                'timeout_error': "Operasi terlalu lama. Coba dengan parameter yang lebih sederhana.",
                'unknown_error': "Terjadi kesalahan tidak terduga: {error}. Silakan coba lagi atau hubungi admin."
            },
            'en': {
                'data_empty': "Dataset is empty or invalid. Please upload a valid dataset.",
                'column_not_found': "Column '{column}' not found in dataset.",
                'model_training_failed': "Model training failed: {error}. Please check parameters or data.",
                'prediction_failed': "Prediction failed: {error}. Ensure model is properly trained.",
                'invalid_parameter': "Parameter '{param}' is invalid: {value}",
                'memory_error': "Insufficient memory for this operation. Try with a smaller dataset.",
                'timeout_error': "Operation took too long. Try with simpler parameters.",
                'unknown_error': "Unexpected error occurred: {error}. Please try again or contact admin."
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('app_errors.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def handle_error(self, error: Exception, context: str = "", user_message: Optional[str] = None) -> Dict[str, Any]:
        """Handle errors with proper logging and user messages"""
        
        # Log the error
        self.logger.error(f"Error in {context}: {str(error)}")
        self.logger.error(traceback.format_exc())
        
        # Determine error type
        error_type = self._classify_error(error)
        
        # Get user message
        if user_message is None:
            user_message = self._get_user_message(error_type, str(error))
        
        # Create detailed error info
        error_info = {
            'type': error_type,
            'message': user_message,
            'technical_details': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
            'timestamp': pd.Timestamp.now(),
            'suggestions': self._get_suggestions(error_type)
        }
        
        # Display to user
        self._display_error_to_user(error_info)
        
        return error_info
    
    def _classify_error(self, error: Exception) -> str:
        """Classify error type for appropriate handling"""
        error_str = str(error).lower()
        
        # First check by exception type
        if isinstance(error, MemoryError):
            return 'memory_error'
        elif isinstance(error, KeyError):
            return 'column_not_found'
        elif isinstance(error, ValueError):
            return 'invalid_parameter'
        elif isinstance(error, TimeoutError):
            return 'timeout_error'
        
        # Then check by string content as fallback
        if 'memory' in error_str or 'ram' in error_str:
            return 'memory_error'
        elif 'timeout' in error_str or 'time' in error_str:
            return 'timeout_error'
        elif 'column' in error_str or 'key' in error_str:
            return 'column_not_found'
        elif 'empty' in error_str or 'none' in error_str:
            return 'data_empty'
        elif 'parameter' in error_str or 'argument' in error_str:
            return 'invalid_parameter'
        elif 'model' in error_str or 'training' in error_str:
            return 'model_training_failed'
        elif 'prediction' in error_str or 'predict' in error_str:
            return 'prediction_failed'
        else:
            return 'unknown_error'
    
    def _get_user_message(self, error_type: str, error_details: str) -> str:
        """Get user-friendly error message"""
        messages = self.error_messages.get(self.language, self.error_messages['en'])
        
        template = messages.get(error_type, messages['unknown_error'])
        
        try:
            return template.format(error=error_details)
        except:
            return template
    
    def _get_suggestions(self, error_type: str) -> List[str]:
        """Get suggestions for fixing the error"""
        suggestions = {
            'data_empty': [
                "Upload a valid dataset",
                "Check if the file was uploaded correctly",
                "Verify the file format (CSV, Excel, etc.)"
            ],
            'column_not_found': [
                "Check column names for typos",
                "Verify the column exists in your dataset",
                "Refresh the dataset if you made changes"
            ],
            'model_training_failed': [
                "Check your data quality and preprocessing",
                "Verify model parameters are appropriate",
                "Try a different model type",
                "Ensure you have enough training data"
            ],
            'prediction_failed': [
                "Ensure your model is trained successfully",
                "Check if input data matches training data format",
                "Verify model compatibility with prediction task"
            ],
            'memory_error': [
                "Use a smaller dataset or sample",
                "Reduce model complexity",
                "Close other applications to free memory",
                "Consider cloud deployment for large datasets"
            ],
            'timeout_error': [
                "Reduce dataset size",
                "Use simpler model parameters",
                "Disable complex features like cross-validation",
                "Use fewer features for training"
            ]
        }
        
        return suggestions.get(error_type, ["Please check your input data and parameters"])
    
    def _display_error_to_user(self, error_info: Dict[str, Any]):
        """Display error information to user in a friendly way"""
        # Main error message
        st.error(f"❌ {error_info['message']}")
        
        # Suggestions
        if error_info['suggestions']:
            with st.expander("💡 Saran Perbaikan" if self.language == 'id' else "💡 Suggestions"):
                for suggestion in error_info['suggestions']:
                    st.write(f"• {suggestion}")
        
        # Technical details (collapsible)
        # Add a unique key to the checkbox to avoid StreamlitDuplicateElementId
        checkbox_key = f"tech_details_{error_info['context']}_{hash(error_info['technical_details'])}"
        if st.checkbox("Tampilkan detail teknis" if self.language == 'id' else "Show technical details", key=checkbox_key):
            st.code(error_info['technical_details'])
            st.caption(f"Context: {error_info['context']}")
            st.caption(f"Time: {error_info['timestamp']}")
    
    def validate_input(self, value: Any, expected_type: type, param_name: str, min_value: Optional[Any] = None, max_value: Optional[Any] = None) -> Dict[str, Any]:
        """Validate input parameters with detailed feedback"""
        try:
            # Type validation
            if not isinstance(value, expected_type):
                try:
                    value = expected_type(value)
                except (ValueError, TypeError):
                    return {
                        'valid': False,
                        'error': f"Parameter '{param_name}' must be of type {expected_type.__name__}",
                        'value': value
                    }
            
            # Range validation
            if min_value is not None and value < min_value:
                return {
                    'valid': False,
                    'error': f"Parameter '{param_name}' must be >= {min_value}",
                    'value': value
                }
            
            if max_value is not None and value > max_value:
                return {
                    'valid': False,
                    'error': f"Parameter '{param_name}' must be <= {max_value}",
                    'value': value
                }
            
            return {
                'valid': True,
                'value': value,
                'error': None
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f"Validation failed for parameter '{param_name}': {str(e)}",
                'value': value
            }