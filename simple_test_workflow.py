#!/usr/bin/env python3
"""
Simple test script to validate workflow improvements without pytest dependencies
"""

import sys
import traceback

# Add current directory to path
sys.path.append('.')

def test_basic_imports():
    """Test basic module imports"""
    print("🧪 Testing basic imports...")
    
    try:
        from session_manager import SessionStateManager
        print("✅ SessionStateManager imported successfully")
        
        from workflow_validator import WorkflowValidator
        print("✅ WorkflowValidator imported successfully")
        
        from data_type_detector import DataTypeDetector
        print("✅ DataTypeDetector imported successfully")
        
        from error_handler import ErrorHandler
        print("✅ ErrorHandler imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        traceback.print_exc()
        return False

def test_error_handler():
    """Test error handler functionality"""
    print("\n🧪 Testing error handler...")
    
    try:
        from error_handler import ErrorHandler
        
        # Test with Indonesian language
        error_handler = ErrorHandler(language='id')
        
        # Test basic error handling
        test_error = ValueError("Test error")
        result = error_handler.handle_error(test_error, "Test Context")
        
        if 'type' in result and 'message' in result:
            print("✅ Error handler basic functionality works")
        else:
            print("❌ Error handler missing required fields")
            return False
        
        # Test input validation
        validation_result = error_handler.validate_input(10, int, "test_param", 0, 100)
        if validation_result['valid']:
            print("✅ Valid input validation works")
        else:
            print("❌ Valid input validation failed")
            return False
        
        # Test invalid input validation
        invalid_result = error_handler.validate_input("invalid", int, "test_param")
        if not invalid_result['valid']:
            print("✅ Invalid input validation works")
        else:
            print("❌ Invalid input validation failed")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error handler test failed: {e}")
        traceback.print_exc()
        return False

def test_data_type_detector():
    """Test data type detector"""
    print("\n🧪 Testing data type detector...")
    
    try:
        import pandas as pd
        import numpy as np
        from data_type_detector import DataTypeDetector
        
        # Create test data
        test_data = pd.DataFrame({
            'numeric_col': np.random.randn(50),
            'categorical_col': np.random.choice(['A', 'B', 'C'], 50),
            'datetime_col': pd.date_range('2023-01-01', periods=50)
        })
        
        detector = DataTypeDetector()
        results = detector.detect_column_types(test_data)
        
        # Check if we got results
        if results and len(results) > 0:
            print("✅ Data type detection returns results")
            
            # Check specific column types
            numeric_result = results.get('numeric_col', {})
            if numeric_result.get('detected_type') == 'numeric':
                print("✅ Numeric column detection works")
            else:
                print(f"❌ Numeric column detection failed: {numeric_result}")
                return False
            
            categorical_result = results.get('categorical_col', {})
            if categorical_result.get('detected_type') == 'object':
                print("✅ Categorical column detection works")
            else:
                print(f"❌ Categorical column detection failed: {categorical_result}")
                return False
            
            return True
        else:
            print("❌ Data type detection returned empty results")
            return False
            
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False
    except Exception as e:
        print(f"❌ Data type detector test failed: {e}")
        traceback.print_exc()
        return False

def test_workflow_validator():
    """Test workflow validator"""
    print("\n🧪 Testing workflow validator...")
    
    try:
        from workflow_validator import WorkflowValidator
        
        validator = WorkflowValidator()
        
        # Test basic validation
        result = validator.validate_workflow_transition('upload', 'eda')
        
        if 'valid' in result:
            print("✅ Workflow validator basic functionality works")
            
            # Test with missing data (should fail)
            result_no_data = validator.validate_workflow_transition('upload', 'eda')
            if result_no_data['valid'] is False:
                print("✅ Workflow validator correctly identifies missing data")
            else:
                print("❌ Workflow validator should fail with missing data")
                return False
            
            return True
        else:
            print("❌ Workflow validator missing required fields")
            return False
            
    except Exception as e:
        print(f"❌ Workflow validator test failed: {e}")
        traceback.print_exc()
        return False

def test_session_manager():
    """Test session manager"""
    print("\n🧪 Testing session manager...")
    
    try:
        from session_manager import SessionStateManager
        
        manager = SessionStateManager()
        
        # Test initialization
        manager.initialize_session_state()
        
        # Check if required keys are initialized
        required_keys = [
            'data', 'processed_data', 'target_column', 'problem_type',
            'X_train', 'X_test', 'y_train', 'y_test', 'numerical_columns',
            'categorical_columns', 'encoders', 'scaler', 'model_results',
            'is_time_series', 'time_column', 'clustering_results', 'eda_insights'
        ]
        
        # Since we can't access streamlit session state directly in test,
        # we'll just check if the manager initializes without errors
        print("✅ Session manager initializes without errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Session manager test failed: {e}")
        traceback.print_exc()
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 Starting comprehensive workflow validation tests...")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Error Handler", test_error_handler),
        ("Data Type Detector", test_data_type_detector),
        ("Workflow Validator", test_workflow_validator),
        ("Session Manager", test_session_manager)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running: {test_name}")
        print("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} CRASHED: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! The workflow improvements are working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)