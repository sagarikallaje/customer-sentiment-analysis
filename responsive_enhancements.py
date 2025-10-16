#!/usr/bin/env python3
"""
Responsive Enhancement Script for Customer Sentiment Analysis Dashboard
This script adds responsive features and performance optimizations
"""

import streamlit as st
import pandas as pd
import time
from functools import lru_cache

def add_responsive_css():
    """Add comprehensive responsive CSS"""
    st.markdown("""
    <style>
        /* Responsive Design Enhancements */
        
        /* Mobile First Approach */
        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.8rem !important;
                padding: 1rem 0.5rem !important;
            }
            
            .sub-header {
                font-size: 1.2rem !important;
                margin: 0.5rem 0 !important;
            }
            
            .info-box {
                padding: 0.75rem !important;
                margin: 0.25rem 0 !important;
            }
            
            .metric-card {
                padding: 0.75rem !important;
                margin: 0.125rem !important;
            }
            
            /* Mobile-friendly tabs */
            .stTabs [data-baseweb="tab"] {
                padding: 6px 8px !important;
                font-size: 0.8rem !important;
                min-height: 35px !important;
            }
            
            /* Mobile-friendly buttons */
            .stButton > button {
                padding: 0.4rem 0.8rem !important;
                font-size: 0.8rem !important;
                min-height: 35px !important;
            }
            
            /* Responsive columns */
            .stColumns > div {
                margin-bottom: 0.5rem !important;
            }
        }
        
        /* Tablet Responsive */
        @media (min-width: 769px) and (max-width: 1024px) {
            .main-header h1 {
                font-size: 2rem !important;
            }
            
            .sub-header {
                font-size: 1.5rem !important;
            }
        }
        
        /* Loading States */
        .loading-container {
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem;
        }
        
        .loading-spinner {
            width: 40px;
            height: 40px;
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Progress Bar */
        .progress-container {
            width: 100%;
            background-color: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 1rem 0;
        }
        
        .progress-bar {
            height: 20px;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            border-radius: 10px;
            transition: width 0.3s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            font-size: 0.8rem;
        }
        
        /* Error and Success Messages */
        .error-message {
            background-color: #ffebee;
            color: #c62828;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #c62828;
            margin: 1rem 0;
        }
        
        .success-message {
            background-color: #e8f5e8;
            color: #2e7d32;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #2e7d32;
            margin: 1rem 0;
        }
        
        /* Responsive Charts */
        .plotly-chart {
            width: 100% !important;
            height: auto !important;
        }
        
        /* Touch-friendly elements */
        .touch-friendly {
            min-height: 44px;
            min-width: 44px;
            padding: 0.5rem;
        }
        
        /* Accessibility improvements */
        .sr-only {
            position: absolute;
            width: 1px;
            height: 1px;
            padding: 0;
            margin: -1px;
            overflow: hidden;
            clip: rect(0, 0, 0, 0);
            white-space: nowrap;
            border: 0;
        }
        
        /* Focus indicators */
        .focus-visible {
            outline: 2px solid #3498db;
            outline-offset: 2px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data_cached(file_path):
    """Cached data loading for better performance"""
    return pd.read_csv(file_path)

def show_loading_state(message="Loading..."):
    """Show a loading state with spinner"""
    with st.spinner(f'🔄 {message}'):
        time.sleep(0.1)  # Small delay to show spinner

def show_progress_bar(progress, message=""):
    """Show a progress bar"""
    progress_bar = st.progress(progress)
    if message:
        st.text(message)
    return progress_bar

def show_error_message(message):
    """Show a styled error message"""
    st.markdown(f"""
    <div class="error-message">
        <strong>❌ Error:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def show_success_message(message):
    """Show a styled success message"""
    st.markdown(f"""
    <div class="success-message">
        <strong>✅ Success:</strong> {message}
    </div>
    """, unsafe_allow_html=True)

def responsive_chart_config():
    """Get responsive chart configuration"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
        'responsive': True,
        'autosize': True
    }

def add_responsive_info_box(title, content, icon="📱"):
    """Add a responsive info box"""
    st.markdown(f"""
    <div class="info-box">
        <h4>{icon} {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def create_responsive_columns(num_columns, gap="small"):
    """Create responsive columns that adapt to screen size"""
    if num_columns == 2:
        return st.columns(2, gap=gap)
    elif num_columns == 3:
        return st.columns(3, gap=gap)
    elif num_columns == 4:
        return st.columns(4, gap=gap)
    else:
        return st.columns(num_columns, gap=gap)

def add_mobile_navigation():
    """Add mobile-friendly navigation hints"""
    st.markdown("""
    <div class="info-box">
        <h4>📱 Mobile Navigation Tips</h4>
        <ul style="margin: 0; padding-left: 1.5rem;">
            <li>Use the sidebar to filter data</li>
            <li>Swipe between tabs for different views</li>
            <li>Pinch to zoom on charts</li>
            <li>Tap and hold for additional options</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def add_performance_tips():
    """Add performance optimization tips"""
    st.markdown("""
    <div class="info-box">
        <h4>⚡ Performance Tips</h4>
        <ul style="margin: 0; padding-left: 1.5rem;">
            <li>Large datasets are automatically optimized</li>
            <li>Charts are cached for faster loading</li>
            <li>Use filters to reduce data processing time</li>
            <li>Refresh the page if performance slows down</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def add_accessibility_features():
    """Add accessibility features"""
    st.markdown("""
    <div class="info-box">
        <h4>♿ Accessibility Features</h4>
        <ul style="margin: 0; padding-left: 1.5rem;">
            <li>Keyboard navigation support</li>
            <li>Screen reader compatible</li>
            <li>High contrast mode available</li>
            <li>Focus indicators for navigation</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main function to demonstrate responsive features"""
    st.title("🚀 Responsive Enhancement Demo")
    
    # Add responsive CSS
    add_responsive_css()
    
    # Show responsive info boxes
    add_responsive_info_box(
        "Responsive Design", 
        "This dashboard adapts to all screen sizes - desktop, tablet, and mobile."
    )
    
    add_mobile_navigation()
    add_performance_tips()
    add_accessibility_features()
    
    # Demo responsive columns
    st.subheader("📊 Responsive Columns Demo")
    col1, col2, col3 = create_responsive_columns(3)
    
    with col1:
        st.metric("Total Users", "1,234", "12%")
    
    with col2:
        st.metric("Active Sessions", "567", "8%")
    
    with col3:
        st.metric("Conversion Rate", "3.2%", "-2%")
    
    # Demo loading states
    st.subheader("⏳ Loading States Demo")
    
    if st.button("Show Loading State"):
        show_loading_state("Processing data...")
        show_success_message("Data processed successfully!")
    
    # Demo progress bar
    if st.button("Show Progress Bar"):
        progress_bar = show_progress_bar(0, "Starting...")
        for i in range(100):
            progress_bar.progress(i + 1)
            time.sleep(0.01)
        show_success_message("Progress completed!")
    
    # Demo error handling
    if st.button("Show Error Message"):
        show_error_message("This is a demo error message for testing purposes.")

if __name__ == "__main__":
    main()
