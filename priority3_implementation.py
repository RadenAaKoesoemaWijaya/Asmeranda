"""
Priority 3 Implementation for SHAP/LIME Performance Optimization
This module contains the complete implementation ready for integration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from priority3_functions import (
    InterpretationCache, 
    optimized_shap_for_large_dataset, 
    optimized_lime_for_large_dataset,
    create_interactive_shap_plot, 
    create_interactive_lime_plot,
    interpretation_cache
)


def add_priority3_optimization_to_shap_tab():
    """
    Add Priority 3 optimization options to SHAP tab
    Call this function in the SHAP tab section
    """
    st.subheader("⚡ Opsi Optimasi (Priority 3)" if st.session_state.language == 'id' else "⚡ Optimization Options (Priority 3)")
    
    use_optimization = st.checkbox(
        "Gunakan optimasi untuk dataset besar" if st.session_state.language == 'id' else "Use optimization for large datasets",
        value=len(st.session_state.X_test) > 1000 if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else False
    )
    
    optimization_config = {
        'use_optimization': use_optimization,
        'use_cache': False,
        'use_interactive': False,
        'max_samples': 100,
        'background_samples': 100
    }
    
    if use_optimization:
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_config['max_samples'] = st.slider(
                "Maksimal sampel (optimasi):" if st.session_state.language == 'id' else "Max samples (optimization):",
                min_value=100, 
                max_value=min(2000, len(st.session_state.X_test) if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else 1000), 
                value=min(1000, len(st.session_state.X_test) if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else 500)
            )
            
            optimization_config['use_cache'] = st.checkbox(
                "Gunakan cache untuk hasil" if st.session_state.language == 'id' else "Use cache for results",
                value=True
            )
        
        with col2:
            optimization_config['background_samples'] = st.slider(
                "Background samples:" if st.session_state.language == 'id' else "Background samples:",
                min_value=50, max_value=500, value=100
            )
            
            optimization_config['use_interactive'] = st.checkbox(
                "Visualisasi interaktif" if st.session_state.language == 'id' else "Interactive visualization",
                value=True
            )
    
    return optimization_config


def add_priority3_optimization_to_lime_tab():
    """
    Add Priority 3 optimization options to LIME tab
    Call this function in the LIME tab section
    """
    st.subheader("⚡ Opsi Optimasi (Priority 3)" if st.session_state.language == 'id' else "⚡ Optimization Options (Priority 3)")
    
    use_optimization = st.checkbox(
        "Gunakan optimasi untuk dataset besar" if st.session_state.language == 'id' else "Use optimization for large datasets",
        value=len(st.session_state.X_test) > 1000 if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else False
    )
    
    optimization_config = {
        'use_optimization': use_optimization,
        'use_cache': False,
        'use_interactive': False,
        'max_samples': 100,
        'num_features': 10
    }
    
    if use_optimization:
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_config['max_samples'] = st.slider(
                "Maksimal sampel (optimasi):" if st.session_state.language == 'id' else "Max samples (optimization):",
                min_value=50, 
                max_value=min(500, len(st.session_state.X_test) if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else 200), 
                value=min(100, len(st.session_state.X_test) if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None else 50)
            )
            
            optimization_config['use_cache'] = st.checkbox(
                "Gunakan cache untuk hasil" if st.session_state.language == 'id' else "Use cache for results",
                value=True
            )
        
        with col2:
            optimization_config['num_features'] = st.slider(
                "Jumlah fitur LIME:" if st.session_state.language == 'id' else "Number of LIME features:",
                min_value=5, max_value=20, value=10
            )
            
            optimization_config['use_interactive'] = st.checkbox(
                "Visualisasi interaktif" if st.session_state.language == 'id' else "Interactive visualization",
                value=True
            )
    
    return optimization_config


def run_optimized_shap(model, X_data, optimization_config, language='id'):
    """
    Run optimized SHAP analysis based on configuration
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data : pandas.DataFrame
        Data for SHAP analysis
    optimization_config : dict
        Optimization configuration
    language : str
        Language for messages
        
    Returns:
    --------
    dict
        SHAP results
    """
    if optimization_config['use_optimization']:
        cache_to_use = interpretation_cache if optimization_config['use_cache'] else None
        
        shap_result = optimized_shap_for_large_dataset(
            model=model,
            X_data=X_data,
            max_samples=optimization_config['max_samples'],
            background_samples=optimization_config['background_samples'],
            cache=cache_to_use,
            random_state=42
        )
        
        if shap_result['success']:
            # Display optimization information
            display_shap_optimization_info(shap_result, language)
            
            # Display interactive plots if enabled
            if optimization_config['use_interactive']:
                display_interactive_shap_plots(shap_result, language)
        
        return shap_result
    else:
        # Use standard SHAP implementation
        from utils import implement_shap_classification
        return implement_shap_classification(
            model=model,
            X_sample=X_data,
            language=language
        )


def run_optimized_lime(model, X_data, y_data, optimization_config, language='id'):
    """
    Run optimized LIME analysis based on configuration
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_data : pandas.DataFrame
        Data for LIME analysis
    y_data : array
        Target data
    optimization_config : dict
        Optimization configuration
    language : str
        Language for messages
        
    Returns:
    --------
    dict
        LIME results
    """
    if optimization_config['use_optimization']:
        cache_to_use = interpretation_cache if optimization_config['use_cache'] else None
        
        lime_result = optimized_lime_for_large_dataset(
            model=model,
            X_data=X_data,
            max_samples=optimization_config['max_samples'],
            num_features=optimization_config['num_features'],
            cache=cache_to_use,
            random_state=42
        )
        
        if lime_result['success']:
            # Display optimization information
            display_lime_optimization_info(lime_result, language)
            
            # Display interactive plots if enabled
            if optimization_config['use_interactive']:
                display_interactive_lime_plots(lime_result, language)
        
        return lime_result
    else:
        # Use standard LIME implementation
        from utils import implement_lime_classification
        return implement_lime_classification(
            model=model,
            X_sample=X_data,
            y_sample=y_data,
            problem_type='classification' if hasattr(model, 'classes_') else 'regression',
            num_features=optimization_config['num_features']
        )


def display_shap_optimization_info(shap_result, language='id'):
    """
    Display SHAP optimization information
    
    Parameters:
    -----------
    shap_result : dict
        SHAP results from optimized function
    language : str
        Language for messages
    """
    st.subheader("📊 Informasi Optimasi" if language == 'id' else "📊 Optimization Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Sampel Asli" if language == 'id' else "Original Samples",
            shap_result['sample_info']['original_size']
        )
        st.metric(
            "Sampel Digunakan" if language == 'id' else "Samples Used",
            shap_result['sample_info']['sampled_size']
        )
    
    with col2:
        st.metric(
            "Waktu Komputasi" if language == 'id' else "Computation Time",
            f"{shap_result['optimization_info']['computation_time']:.2f}s"
        )
        st.metric(
            "Samples/s" if language == 'id' else "Samples/s",
            f"{shap_result['optimization_info']['samples_per_second']:.1f}"
        )
    
    with col3:
        if shap_result.get('from_cache'):
            st.success("✅ Hasil dari cache" if language == 'id' else "✅ Result from cache")
        else:
            st.info("🔄 Hasil baru dihitung" if language == 'id' else "🔄 New result computed")
    
    # Display explainer type
    st.info(f"🔧 Explainer: {shap_result['optimization_info']['explainer_type']}")


def display_lime_optimization_info(lime_result, language='id'):
    """
    Display LIME optimization information
    
    Parameters:
    -----------
    lime_result : dict
        LIME results from optimized function
    language : str
        Language for messages
    """
    st.subheader("📊 Informasi Optimasi" if language == 'id' else "📊 Optimization Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Sampel Asli" if language == 'id' else "Original Samples",
            lime_result['sample_info']['original_size']
        )
        st.metric(
            "Sampel Digunakan" if language == 'id' else "Samples Used",
            lime_result['sample_info']['sampled_size']
        )
    
    with col2:
        st.metric(
            "Berhasil" if language == 'id' else "Successful",
            lime_result.get('n_successful', 0)
        )
        st.metric(
            "Gagal" if language == 'id' else "Failed",
            lime_result.get('n_failed', 0)
        )
    
    with col3:
        st.metric(
            "Waktu Komputasi" if language == 'id' else "Computation Time",
            f"{lime_result['optimization_info']['computation_time']:.2f}s"
        )
        if lime_result.get('from_cache'):
            st.success("✅ Hasil dari cache" if language == 'id' else "✅ Result from cache")
        else:
            st.info("🔄 Hasil baru dihitung" if language == 'id' else "🔄 New result computed")


def display_interactive_shap_plots(shap_result, language='id'):
    """
    Display interactive SHAP plots
    
    Parameters:
    -----------
    shap_result : dict
        SHAP results
    language : str
        Language for messages
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        st.subheader("📈 Visualisasi Interaktif" if language == 'id' else "📈 Interactive Visualization")
        
        # Create interactive plots
        interactive_plots = create_interactive_shap_plot(
            shap_result['shap_values'],
            shap_result['X_sample'],
            feature_names=shap_result['X_sample'].columns.tolist()
        )
        
        if interactive_plots['success']:
            if 'importance' in interactive_plots:
                st.plotly_chart(interactive_plots['importance'], use_container_width=True)
            
            if 'scatter' in interactive_plots:
                st.plotly_chart(interactive_plots['scatter'], use_container_width=True)
        else:
            st.warning("Gagal membuat visualisasi interaktif" if language == 'id' else "Failed to create interactive visualization")
            
    except ImportError:
        st.warning("Plotly tidak tersedia. Install dengan: pip install plotly" if language == 'id' else "Plotly not available. Install with: pip install plotly")


def display_interactive_lime_plots(lime_result, language='id'):
    """
    Display interactive LIME plots
    
    Parameters:
    -----------
    lime_result : dict
        LIME results
    language : str
        Language for messages
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        st.subheader("📈 Visualisasi Interaktif" if language == 'id' else "📈 Interactive Visualization")
        
        # Create interactive plots
        interactive_plots = create_interactive_lime_plot(
            lime_result['explanations']
        )
        
        if interactive_plots['success']:
            if 'importance' in interactive_plots:
                st.plotly_chart(interactive_plots['importance'], use_container_width=True)
            
            # Sample selection for detailed view
            if 'sample_options' in interactive_plots and 'explanations' in interactive_plots:
                selected_sample = st.selectbox(
                    "Pilih sampel:" if language == 'id' else "Select sample:",
                    options=range(len(interactive_plots['sample_options'])),
                    format_func=lambda i: interactive_plots['sample_options'][i]
                )
                
                # Display detailed explanation for selected sample
                if selected_sample < len(interactive_plots['explanations']):
                    exp = interactive_plots['explanations'][selected_sample]
                    if exp.get('explanation'):
                        exp_list = exp['explanation'].as_list()
                        exp_df = pd.DataFrame(exp_list, columns=['Feature', 'Contribution'])
                        exp_df = exp_df.sort_values('Contribution', ascending=False, key=abs)
                        
                        st.write(f"**Penjelasan untuk {interactive_plots['sample_options'][selected_sample]}:**")
                        st.dataframe(exp_df)
        else:
            st.warning("Gagal membuat visualisasi interaktif" if language == 'id' else "Failed to create interactive visualization")
            
    except ImportError:
        st.warning("Plotly tidak tersedia. Install dengan: pip install plotly" if language == 'id' else "Plotly not available. Install with: pip install plotly")


def add_performance_monitoring_tab():
    """
    Add performance monitoring section
    """
    st.subheader("📊 Monitoring Performa" if st.session_state.language == 'id' else "📊 Performance Monitoring")
    
    # Display performance statistics
    from priority3_functions import get_interpretation_performance_stats
    stats = get_interpretation_performance_stats()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**SHAP Performance:**" if st.session_state.language == 'id' else "**SHAP Performance:**")
        for key, value in stats['shap'].items():
            st.write(f"• {key}: {value}")
    
    with col2:
        st.write("**LIME Performance:**" if st.session_state.language == 'id' else "**LIME Performance:**")
        for key, value in stats['lime'].items():
            st.write(f"• {key}: {value}")
    
    st.write("**Optimization Tips:**" if st.session_state.language == 'id' else "**Optimization Tips:**")
    for tip in stats['optimization_tips']:
        st.write(f"• {tip}")
    
    # Cache management
    st.subheader("🗂️ Manajemen Cache" if st.session_state.language == 'id' else "🗂️ Cache Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear Cache" if st.session_state.language == 'id' else "Clear Cache"):
            interpretation_cache.clear_cache()
            st.success("Cache berhasil dibersihkan!" if st.session_state.language == 'id' else "Cache cleared successfully!")
    
    with col2:
        # Display cache info
        try:
            import os
            cache_dir = "interpretation_cache"
            if os.path.exists(cache_dir):
                cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
                st.metric(
                    "Cache Files" if st.session_state.language == 'id' else "Cache Files",
                    len(cache_files)
                )
        except:
            pass


# Example usage in app.py:
"""
# In SHAP tab:
optimization_config = add_priority3_optimization_to_shap_tab()

if st.button("Generate SHAP Values"):
    shap_result = run_optimized_shap(
        model=st.session_state.model,
        X_data=st.session_state.X_test[selected_features],
        optimization_config=optimization_config,
        language=st.session_state.language
    )

# In LIME tab:
optimization_config = add_priority3_optimization_to_lime_tab()

if st.button("Generate LIME Explanations"):
    lime_result = run_optimized_lime(
        model=st.session_state.model,
        X_data=st.session_state.X_test[selected_features],
        y_data=st.session_state.y_train,
        optimization_config=optimization_config,
        language=st.session_state.language
    )

# Add performance monitoring:
add_performance_monitoring_tab()
"""
