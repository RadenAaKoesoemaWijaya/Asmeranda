import pytest
import pandas as pd
import numpy as np
import streamlit as st
from session_manager import SessionStateManager
from workflow_validator import WorkflowValidator
from data_type_detector import DataTypeDetector
from error_handler import ErrorHandler

class TestWorkflowValidation:
    """Comprehensive tests for workflow validation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample test data"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
    
    @pytest.fixture
    def session_manager(self):
        """Create session manager instance"""
        return SessionStateManager()
    
    @pytest.fixture
    def workflow_validator(self):
        """Create workflow validator instance"""
        return WorkflowValidator()
    
    @pytest.fixture
    def data_type_detector(self):
        """Create data type detector instance"""
        return DataTypeDetector()
    
    @pytest.fixture
    def error_handler(self):
        """Create error handler instance"""
        return ErrorHandler(language='id')
    
    def test_session_state_initialization(self, session_manager):
        """Test session state initialization"""
        # Check all required keys are initialized
        for key in session_manager.required_keys:
            assert key in st.session_state
            assert st.session_state[key] == session_manager.required_keys[key]
    
    def test_data_type_detection(self, data_type_detector, sample_data):
        """Test data type detection accuracy"""
        results = data_type_detector.detect_column_types(sample_data)
        
        # Check numerical column detection
        assert results['feature1']['detected_type'] == 'numeric'
        assert results['feature1']['confidence'] > 0.8
        
        # Check categorical column detection
        assert results['category']['detected_type'] == 'object'
        assert results['category']['confidence'] > 0.8
    
    def test_workflow_validation_upload_to_eda(self, workflow_validator, session_manager, sample_data):
        """Test workflow validation from upload to EDA"""
        # Setup session state
        st.session_state.data = sample_data
        
        # Validate transition
        result = workflow_validator.validate_workflow_transition('upload', 'eda')
        
        assert result['valid'] is True
        assert len(result['errors']) == 0
    
    def test_workflow_validation_missing_data(self, workflow_validator):
        """Test workflow validation with missing data"""
        # Clear session state
        st.session_state.data = None
        
        # Validate transition
        result = workflow_validator.validate_workflow_transition('upload', 'eda')
        
        assert result['valid'] is False
        assert 'data' in result['missing']
    
    def test_error_handling_classification(self, error_handler):
        """Test error handling and classification"""
        # Test different error types
        test_errors = [
            (MemoryError("Out of memory"), 'memory_error'),
            (KeyError("Column not found"), 'column_not_found'),
            (ValueError("Invalid parameter"), 'invalid_parameter')
        ]
        
        for error, expected_type in test_errors:
            result = error_handler.handle_error(error, "test_context")
            assert result['type'] == expected_type
            assert 'message' in result
            assert 'suggestions' in result
    
    def test_input_validation(self, error_handler):
        """Test input parameter validation"""
        # Test valid input
        result = error_handler.validate_input(10, int, "test_param", 0, 100)
        assert result['valid'] is True
        assert result['value'] == 10
        
        # Test invalid type
        result = error_handler.validate_input("invalid", int, "test_param")
        assert result['valid'] is False
        assert 'error' in result
        
        # Test out of range
        result = error_handler.validate_input(150, int, "test_param", 0, 100)
        assert result['valid'] is False
        assert 'error' in result
    
    def test_clustering_workflow_integration(self, session_manager, sample_data):
        """Test clustering workflow integration"""
        # Setup data
        st.session_state.data = sample_data
        st.session_state.numerical_columns = ['feature1', 'feature2']
        st.session_state.categorical_columns = ['category']
        
        # Validate clustering can proceed
        validation = session_manager.validate_data_flow('eda', 'preprocessing')
        assert validation['valid'] is True
    
    def test_time_series_detection(self, data_type_detector):
        """Test time series column detection"""
        # Create time series data
        df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=100),
            'value': np.random.randn(100)
        })
        
        results = data_type_detector.detect_column_types(df)
        assert results['date']['detected_type'] == 'datetime'
        assert results['date']['confidence'] > 0.9

def run_workflow_tests():
    """Run comprehensive workflow tests"""
    print("🧪 Running comprehensive workflow tests...")
    
    # Initialize components
    session_manager = SessionStateManager()
    workflow_validator = WorkflowValidator()
    data_type_detector = DataTypeDetector()
    error_handler = ErrorHandler(language='id')
    
    # Test 1: Session State Management
    print("\n📋 Testing Session State Management...")
    try:
        # Check initialization
        for key in session_manager.required_keys:
            assert key in st.session_state
            print(f"✅ {key} initialized correctly")
        print("✅ Session state management test passed")
    except Exception as e:
        print(f"❌ Session state test failed: {e}")
    
    # Test 2: Data Type Detection
    print("\n🔍 Testing Data Type Detection...")
    try:
        # Create test data
        test_data = pd.DataFrame({
            'numeric_col': np.random.randn(50),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 50),
            'datetime_col': pd.date_range('2023-01-01', periods=50)
        })
        
        results = data_type_detector.detect_column_types(test_data)
        
        # Validate results
        assert results['numeric_col']['detected_type'] == 'numeric'
        assert results['categorical_col']['detected_type'] == 'object'
        assert results['datetime_col']['detected_type'] == 'datetime'
        
        print("✅ Data type detection test passed")
    except Exception as e:
        print(f"❌ Data type detection test failed: {e}")
    
    # Test 3: Workflow Validation
    print("\n🔄 Testing Workflow Validation...")
    try:
        # Setup test data
        st.session_state.data = test_data
        st.session_state.numerical_columns = ['numeric_col']
        st.session_state.categorical_columns = ['categorical_col']
        
        # Test transitions
        upload_to_eda = workflow_validator.validate_workflow_transition('upload', 'eda')
        assert upload_to_eda['valid'] is True
        print("✅ Upload to EDA validation passed")
        
        eda_to_preprocessing = workflow_validator.validate_workflow_transition('eda', 'preprocessing')
        assert eda_to_preprocessing['valid'] is True
        print("✅ EDA to Preprocessing validation passed")
        
        print("✅ Workflow validation tests passed")
    except Exception as e:
        print(f"❌ Workflow validation test failed: {e}")
    
    # Test 4: Error Handling
    print("\n⚠️ Testing Error Handling...")
    try:
        # Test error classification
        test_errors = [
            MemoryError("Out of memory"),
            KeyError("Column not found"),
            ValueError("Invalid parameter")
        ]
        
        for error in test_errors:
            result = error_handler.handle_error(error, "test_context")
            assert 'type' in result
            assert 'message' in result
            assert 'suggestions' in result
            print(f"✅ Error handling for {type(error).__name__} passed")
        
        print("✅ Error handling tests passed")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test 5: Input Validation
    print("\n🔢 Testing Input Validation...")
    try:
        # Test valid input
        result = error_handler.validate_input(10, int, "test_param", 0, 100)
        assert result['valid'] is True
        print("✅ Valid input validation passed")
        
        # Test invalid input
        result = error_handler.validate_input("invalid", int, "test_param")
        assert result['valid'] is False
        print("✅ Invalid input validation passed")
        
        print("✅ Input validation tests passed")
    except Exception as e:
        print(f"❌ Input validation test failed: {e}")
    
    print("\n🎉 All workflow tests completed!")

if __name__ == "__main__":
    # Run tests
    run_workflow_tests()
    
    # Optional: Run pytest if available
    try:
        import pytest
        print("\n🧪 Running pytest tests...")
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\n📋 Pytest not available, skipping pytest tests")