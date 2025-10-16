"""
Interactive Customer Sentiment Analysis Dashboard
Professional Streamlit web app with modern UI and comprehensive features
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
import io
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from src.preprocessing import TextPreprocessor
from src.model import SentimentModel, ModelTrainer
from src.visualizations import SentimentVisualizer

# Page configuration
st.set_page_config(
    page_title="Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Customer Sentiment Analysis Dashboard v2.0"
    }
)

# Configure file uploader settings
# Note: File upload limits are configured in .streamlit/config.toml

# Custom CSS
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ff7f0e;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
        --border-radius: 12px;
        --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s ease;
    }
    
    /* Main Container */
    .main-container {
        background: linear-gradient(rgba(0, 0, 0, 0.4), rgba(0, 0, 0, 0.4)), 
                    url('https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?ixlib=rb-4.0.3&auto=format&fit=crop&w=2070&q=80');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        min-height: 100vh;
        padding: 2rem;
    }
    
    /* Header Styles */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.8);
        background: linear-gradient(45deg, #fff, #f0f0f0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .sub-header {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255, 255, 255, 0.9);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        text-shadow: 1px 1px 4px rgba(0,0,0,0.7);
    }
    
    /* Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--box-shadow);
        margin: 0.5rem 0;
        border-left: 4px solid var(--primary-color);
        transition: var(--transition);
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card h4 {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        font-weight: 500;
        color: #6c757d;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card h2 {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--dark-color);
        margin: 0;
    }
    
    .positive-metric {
        border-left-color: var(--success-color);
    }
    
    .negative-metric {
        border-left-color: var(--danger-color);
    }
    
    .accuracy-metric {
        border-left-color: var(--warning-color);
    }
    
    .info-metric {
        border-left-color: var(--info-color);
    }
    
    /* Sidebar Styles */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: var(--border-radius);
        margin: 1rem;
        padding: 1rem;
        box-shadow: var(--box-shadow);
    }
    
    .sidebar .sidebar-content h1 {
        font-family: 'Inter', sans-serif;
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .sidebar .sidebar-content h2 {
        font-family: 'Inter', sans-serif;
        color: var(--dark-color);
        font-weight: 500;
        font-size: 1.1rem;
    }
    
    /* Button Styles */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(45deg, var(--primary-color), #1565c0);
        color: white;
        border-radius: var(--border-radius);
        border: none;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: var(--transition);
        box-shadow: var(--box-shadow);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(31, 119, 180, 0.4);
        background: linear-gradient(45deg, #1565c0, var(--primary-color));
    }
    
    /* Selectbox Styles */
    .stSelectbox > div > div {
        background-color: white;
        border-radius: var(--border-radius);
        border: 2px solid #e9ecef;
        transition: var(--transition);
    }
    
    .stSelectbox > div > div:hover {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    /* Slider Styles */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, var(--danger-color), var(--warning-color), var(--success-color));
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        padding: 0.5rem;
        border-radius: var(--border-radius);
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding-left: 24px;
        padding-right: 24px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        border-radius: var(--border-radius);
        transition: var(--transition);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.9);
        color: var(--primary-color);
        font-weight: 600;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--danger-color), var(--warning-color), var(--success-color));
        border-radius: var(--border-radius);
    }
    
    /* Text Area */
    .stTextArea > div > div > textarea {
        border-radius: var(--border-radius);
        border: 2px solid #e9ecef;
        transition: var(--transition);
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgba(31, 119, 180, 0.1);
    }
    
    /* File Uploader */
    .stFileUploader > div {
        border-radius: var(--border-radius);
        border: 2px dashed var(--primary-color);
        background: rgba(31, 119, 180, 0.05);
        transition: var(--transition);
    }
    
    .stFileUploader > div:hover {
        background: rgba(31, 119, 180, 0.1);
        border-color: #1565c0;
    }
    
    /* Alert Styles */
    .stAlert {
        border-radius: var(--border-radius);
        border: none;
        box-shadow: var(--box-shadow);
    }
    
    /* Success Alert */
    .stAlert[data-testid="stAlert"] .stAlert-success {
        background: linear-gradient(45deg, rgba(44, 160, 44, 0.1), rgba(44, 160, 44, 0.05));
        border-left: 4px solid var(--success-color);
    }
    
    /* Error Alert */
    .stAlert[data-testid="stAlert"] .stAlert-error {
        background: linear-gradient(45deg, rgba(214, 39, 40, 0.1), rgba(214, 39, 40, 0.05));
        border-left: 4px solid var(--danger-color);
    }
    
    /* Warning Alert */
    .stAlert[data-testid="stAlert"] .stAlert-warning {
        background: linear-gradient(45deg, rgba(255, 127, 14, 0.1), rgba(255, 127, 14, 0.05));
        border-left: 4px solid var(--warning-color);
    }
    
    /* Info Alert */
    .stAlert[data-testid="stAlert"] .stAlert-info {
        background: linear-gradient(45deg, rgba(23, 162, 184, 0.1), rgba(23, 162, 184, 0.05));
        border-left: 4px solid var(--info-color);
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--primary-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #1565c0;
    }
    
    /* Animation Classes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        
        .metric-card h2 {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)
    
    # Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = SentimentVisualizer()

def load_data():
    """Load and preprocess the data"""
    try:
        # Try to load from data directory first
        data_paths = ["data/comprehensive_reviews.csv", "data/amazon_reviews.csv"]
        df = None
        
        for data_path in data_paths:
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
                st.info(f"Loaded dataset from {data_path}")
                break
        
        if df is None:
            # Create sample data if no file exists
            st.warning("No data file found. Creating sample data...")
            df = create_sample_data()
        
        # Initialize preprocessor
        preprocessor = TextPreprocessor()
        
        # Preprocess data
        processed_df = preprocessor.get_processed_data(df)
        
        st.session_state.data = df
        st.session_state.processed_data = processed_df
        st.session_state.preprocessor = preprocessor
        st.session_state.data_loaded = True
        
        return True
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def create_sample_data():
    """Create sample customer review data"""
    np.random.seed(42)
    
    # Sample positive and negative reviews
    positive_reviews = [
        "Absolutely love this product! Great quality and fast shipping.",
        "Excellent value for money. Highly recommend to everyone.",
        "Perfect! Exactly what I was looking for. Will buy again.",
        "Amazing product! Works perfectly and looks great.",
        "Outstanding quality! Exceeded my expectations completely.",
        "Fantastic purchase! Great customer service too.",
        "Love it! Perfect fit and excellent material quality.",
        "Wonderful product! Fast delivery and great packaging.",
        "Excellent! Works exactly as described. Very satisfied.",
        "Amazing quality! Great price and perfect functionality."
    ]
    
    negative_reviews = [
        "Terrible product! Poor quality and not as described.",
        "Waste of money. Broke after just one week of use.",
        "Disappointed with this purchase. Quality is very poor.",
        "Not worth the price. Cheaply made and doesn't work well.",
        "Horrible experience! Product arrived damaged and broken.",
        "Very disappointed. Poor quality and bad customer service.",
        "Regret buying this. Doesn't work as advertised at all.",
        "Terrible quality! Broke immediately after opening.",
        "Waste of time and money. Product is completely useless.",
        "Poor quality! Not worth the money spent on this."
    ]
    
    categories = ["Electronics", "Books", "Clothing", "Home & Garden", "Sports & Outdoors"]
    
    reviews = []
    for i in range(500):
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.3])
        
        if rating >= 4:
            review_text = np.random.choice(positive_reviews)
        else:
            review_text = np.random.choice(negative_reviews)
        
        reviews.append({
            'reviewText': review_text,
            'overall': rating,
            'category': np.random.choice(categories)
        })
    
    return pd.DataFrame(reviews)

def train_model():
    """Train the sentiment analysis model"""
    try:
        if not st.session_state.data_loaded:
            st.error("Please load data first!")
            return False
        
        # Initialize model trainer
        trainer = ModelTrainer(st.session_state.preprocessor, 'logistic_regression')
        
        # Train model
        with st.spinner("Training model..."):
            results = trainer.train_model(st.session_state.processed_data)
        
        st.session_state.model = trainer.get_model()
        st.session_state.training_results = results
        st.session_state.model_trained = True
        
        return True
        
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False

def create_enhanced_sidebar():
    """Create an enhanced user-friendly sidebar with additional features"""
    
    # Enhanced sidebar header with better styling
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0; border-bottom: 2px solid #3498db; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 1rem;">
        <h2 style="color: white; margin: 0; font-size: 1.5rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">🎛️ Control Panel</h2>
        <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Customize your analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats Section
    st.sidebar.markdown("### 📊 Quick Stats")
    if st.session_state.data_loaded:
        total_reviews = len(st.session_state.processed_data)
        st.sidebar.metric("Total Reviews", f"{total_reviews:,}")
        
        # Find rating column for quick stats
        rating_column = None
        possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
        
        for col in possible_rating_columns:
            if col in st.session_state.processed_data.columns:
                rating_column = col
                break
        
        if rating_column:
            avg_rating = st.session_state.processed_data[rating_column].mean()
            st.sidebar.metric("Avg Rating", f"{avg_rating:.1f} ⭐")
    
    st.sidebar.markdown("---")
    
    # Data Upload Section with enhanced features
    st.sidebar.markdown("### 📁 Data Management")
    
    # Sample Data Generation Option
    st.sidebar.markdown("#### 🎲 Generate Sample Data")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        sample_size = st.sidebar.selectbox(
            "Sample Size",
            [100, 500, 1000, 5000, 10000],
            index=2,  # Default to 1000
            help="Number of sample reviews to generate",
            key="sample_size"
        )
    
    with col2:
        if st.sidebar.button("🎲 Generate", help="Create sample e-commerce review data"):
            with st.sidebar.spinner('Generating sample data...'):
                try:
                    # Import the sample data generator
                    from create_sample_data import generate_sample_data
                    
                    # Generate sample data
                    sample_data = generate_sample_data(sample_size)
                    
                    # Store in session state
                    st.session_state.data = sample_data
                    st.session_state.data_loaded = True
                    st.session_state.processed_data = st.session_state.preprocessor.preprocess_data(sample_data)
                    
                    st.sidebar.success(f"✅ Generated {sample_size:,} sample reviews!")
                    st.rerun()
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error generating sample data: {str(e)}")
    
    st.sidebar.markdown("---")
    
    # File uploader with better styling
    st.sidebar.markdown("#### 📤 Upload Your Data")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload your customer review data in CSV format",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Show file info
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024  # MB
            st.sidebar.success(f"📄 File loaded: {uploaded_file.name}")
            st.sidebar.info(f"📏 Size: {file_size:.2f} MB")
            
            # Load and process data
            with st.sidebar.spinner('Processing data...'):
                new_data = pd.read_csv(uploaded_file)
                st.session_state.data = new_data
                st.session_state.data_loaded = True
                st.session_state.processed_data = st.session_state.preprocessor.preprocess_data(new_data)
                st.sidebar.success("✅ Data processed successfully!")
                st.rerun()
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
    
    # Quick Actions Section
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Refresh", help="Reload the current dataset"):
            st.session_state.data_loaded = False
            st.rerun()
    
    with col2:
        if st.button("📊 Reset Filters", help="Clear all applied filters"):
            st.session_state.filtered_data = st.session_state.processed_data
            st.rerun()
    
    # Export Options
    st.sidebar.markdown("### 📤 Export Options")
    
    if st.sidebar.button("📥 Download Current Data", help="Download filtered data as CSV"):
        if 'filtered_data' in st.session_state and len(st.session_state.filtered_data) > 0:
            csv = st.session_state.filtered_data.to_csv(index=False)
            st.sidebar.download_button(
                label="💾 Download CSV",
                data=csv,
                file_name=f"filtered_sentiment_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        else:
            st.sidebar.warning("No data to export")
    
    st.sidebar.markdown("---")
    
    # Enhanced Data Filters Section
    st.sidebar.markdown("### 🔍 Data Filters")
    
    # Smart Filter Presets
    st.sidebar.markdown("#### 🎯 Quick Filters")
    
    filter_preset = st.sidebar.selectbox(
        "Choose Filter Preset",
        ["All Data", "High Ratings (4-5)", "Low Ratings (1-2)", "Recent Reviews", "Long Reviews", "Short Reviews"],
        help="Apply common filter combinations quickly"
    )
    
    # Apply preset filters
    filtered_data = st.session_state.processed_data.copy()
    
    if filter_preset == "High Ratings (4-5)":
        rating_column = None
        possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
        
        for col in possible_rating_columns:
            if col in filtered_data.columns:
                rating_column = col
                break
        
        if rating_column:
            filtered_data = filtered_data[filtered_data[rating_column] >= 4]
    
    elif filter_preset == "Low Ratings (1-2)":
        rating_column = None
        possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
        
        for col in possible_rating_columns:
            if col in filtered_data.columns:
                rating_column = col
                break
        
        if rating_column:
            filtered_data = filtered_data[filtered_data[rating_column] <= 2]
    
    elif filter_preset == "Long Reviews":
        text_column = None
        possible_text_columns = ['reviewText', 'Review_Text', 'text', 'review_text', 'content', 'review', 'comment']
        
        for col in possible_text_columns:
            if col in filtered_data.columns:
                text_column = col
                break
        
        if text_column:
            filtered_data = filtered_data[filtered_data[text_column].str.len() > 200]
    
    elif filter_preset == "Short Reviews":
        text_column = None
        possible_text_columns = ['reviewText', 'Review_Text', 'text', 'review_text', 'content', 'review', 'comment']
        
        for col in possible_text_columns:
            if col in filtered_data.columns:
                text_column = col
                break
        
        if text_column:
            filtered_data = filtered_data[filtered_data[text_column].str.len() < 100]
    
    # Advanced Filters Section
    st.sidebar.markdown("#### ⚙️ Advanced Filters")
    
    # Category filter with better styling
    category_column = None
    possible_category_columns = ['Category', 'category', 'cat', 'product_category']
    
    for col in possible_category_columns:
        if col in filtered_data.columns:
            category_column = col
            break
    
    if category_column:
        categories = ['All Categories'] + sorted(list(filtered_data[category_column].unique()))
        selected_category = st.sidebar.selectbox(
            "🏷️ Product Category", 
            categories,
            help="Filter reviews by product category",
            key="category_filter"
        )
        
        if selected_category != 'All Categories':
            filtered_data = filtered_data[filtered_data[category_column] == selected_category]
    
    # Enhanced Rating filter with visual slider
    st.sidebar.markdown("#### ⭐ Rating Range")
    
    rating_column = None
    possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
    
    for col in possible_rating_columns:
        if col in filtered_data.columns:
            rating_column = col
            break
    
    if rating_column:
        min_rating, max_rating = st.sidebar.slider(
            "Select Rating Range",
            min_value=1.0,
            max_value=5.0,
            value=(1.0, 5.0),
            step=0.5,
            help="Drag sliders to filter by rating range",
            key="rating_slider"
        )
        
        if min_rating > max_rating:
            min_rating, max_rating = max_rating, min_rating
        
        filtered_data = filtered_data[
            (filtered_data[rating_column] >= min_rating) &
            (filtered_data[rating_column] <= max_rating)
        ]
    
    # Text Length Filter
    st.sidebar.markdown("#### 📝 Review Length")
    
    text_column = None
    possible_text_columns = ['reviewText', 'Review_Text', 'text', 'review_text', 'content', 'review', 'comment']
    
    for col in possible_text_columns:
        if col in filtered_data.columns:
            text_column = col
            break
    
    if text_column:
        min_length, max_length = st.sidebar.slider(
            "Review Length (characters)",
            min_value=0,
            max_value=1000,
            value=(0, 1000),
            step=50,
            help="Filter reviews by text length",
            key="length_slider"
        )
        
        filtered_data = filtered_data[
            (filtered_data[text_column].str.len() >= min_length) &
            (filtered_data[text_column].str.len() <= max_length)
        ]
    
    # Search Filter
    st.sidebar.markdown("#### 🔍 Text Search")
    
    search_term = st.sidebar.text_input(
        "Search in Reviews",
        placeholder="Enter keywords...",
        help="Search for specific words or phrases in reviews",
        key="search_input"
    )
    
    if search_term:
        if text_column:
            filtered_data = filtered_data[
                filtered_data[text_column].str.contains(search_term, case=False, na=False)
            ]
    
    # Filter Summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Filter Summary")
    
    original_count = len(st.session_state.processed_data)
    filtered_count = len(filtered_data)
    
    st.sidebar.metric("Original Reviews", f"{original_count:,}")
    st.sidebar.metric("Filtered Reviews", f"{filtered_count:,}")
    
    if original_count > 0:
        filter_percentage = (filtered_count / original_count) * 100
        st.sidebar.metric("Filter Applied", f"{filter_percentage:.1f}%")
    
    # Data Preview Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👀 Data Preview")
    
    if st.session_state.data_loaded and len(filtered_data) > 0:
        with st.sidebar.expander("📊 Sample Data Preview"):
            # Show first few rows
            preview_data = filtered_data.head(3)
            
            # Display key columns only
            display_cols = []
            for col in preview_data.columns:
                if any(keyword in col.lower() for keyword in ['review', 'rating', 'category', 'text']):
                    display_cols.append(col)
            
            if display_cols:
                st.dataframe(preview_data[display_cols], use_container_width=True)
            else:
                st.dataframe(preview_data.iloc[:, :3], use_container_width=True)
            
            st.caption(f"Showing 3 of {len(filtered_data):,} total reviews")
    
    # Tips and Help Section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💡 Tips & Help")
    
    with st.sidebar.expander("📚 How to Use Filters"):
        st.markdown("""
        **Quick Filters**: Use presets for common scenarios
        
        **Advanced Filters**: 
        - Category: Filter by product type
        - Rating: Use sliders for range selection
        - Length: Filter by review length
        - Search: Find specific keywords
        
        **Export**: Download filtered data anytime
        """)
    
    with st.sidebar.expander("🎲 Sample Data Features"):
        st.markdown("""
        **Generated Sample Data Includes:**
        - Product reviews with ratings
        - Customer demographics
        - Product categories
        - Review dates and metadata
        - Sentiment labels
        
        **Perfect for testing and demos!**
        """)
    
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        **No Data Showing?**
        - Check if filters are too restrictive
        - Try resetting filters
        - Verify data format
        
        **Slow Performance?**
        - Reduce data size with filters
        - Use quick filters for faster results
        
        **Need Sample Data?**
        - Use the "Generate Sample Data" option
        - Choose from 100 to 10,000 reviews
        """)
    
    # Store filtered data in session state
    st.session_state.filtered_data = filtered_data
    
    return filtered_data

def create_sidebar():
    """Create sidebar with modern, accessible controls"""
    
    # Main header with better styling
    st.sidebar.markdown("""
    <div style="background: linear-gradient(45deg, #1f77b4, #1565c0); 
                padding: 1rem; border-radius: 12px; margin-bottom: 1rem; text-align: center;">
        <h2 style="color: white; margin: 0; font-size: 1.2rem;">Control Panel</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Data upload section
    st.sidebar.markdown("### Data Upload")
    st.sidebar.markdown("---")
    
    # File uploader for CSV files with extended limits
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV Dataset",
        type=['csv'],
        help="Upload your own dataset or Kaggle dataset (CSV format). Supports files up to 200MB.",
        key="data_uploader",
        accept_multiple_files=False
    )
    
    # Show file size information
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        st.sidebar.info(f"File size: {file_size_mb:.2f} MB")
        
        if file_size_mb > 200:
            st.sidebar.error("File too large! Maximum size is 200MB.")
            uploaded_file = None
        elif file_size_mb > 50:
            st.sidebar.warning("Large file detected. Processing may take longer...")
    
    if uploaded_file is not None:
        try:
            # Show progress for large files
            if uploaded_file.size > 10 * 1024 * 1024:  # > 10MB
                progress_bar = st.sidebar.progress(0)
                status_text = st.sidebar.empty()
                status_text.text("Reading CSV file...")
            
            # Read the uploaded file with progress and chunking for large files
            if uploaded_file.size > 50 * 1024 * 1024:  # > 50MB
                # Use chunked reading for very large files
                chunk_size = 10000
                chunks = []
                total_rows = 0
                
                for chunk in pd.read_csv(uploaded_file, chunksize=chunk_size):
                    chunks.append(chunk)
                    total_rows += len(chunk)
                    
                    if uploaded_file.size > 10 * 1024 * 1024:
                        progress = min(50, (total_rows / 100000) * 50)  # Estimate progress
                        progress_bar.progress(int(progress))
                
                new_data = pd.concat(chunks, ignore_index=True)
            else:
                new_data = pd.read_csv(uploaded_file)
            
            if uploaded_file.size > 10 * 1024 * 1024:
                progress_bar.progress(50)
                status_text.text("Validating dataset...")
            
            # Validate the dataset with flexible column detection
            # Try to find review text and rating columns with different possible names
            review_text_columns = ['Review_Text', 'reviewText', 'text', 'review_text', 'content', 'review', 'comment']
            rating_columns = ['Rating', 'overall', 'rating', 'score', 'stars', 'star_rating']
            
            actual_review_column = None
            actual_rating_column = None
            
            # Find review text column
            for col in review_text_columns:
                if col in new_data.columns:
                    actual_review_column = col
                    break
            
            # Find rating column
            for col in rating_columns:
                if col in new_data.columns:
                    actual_rating_column = col
                    break
            
            if actual_review_column and actual_rating_column:
                # Rename columns to standard names for processing
                new_data = new_data.rename(columns={
                    actual_review_column: 'Review_Text',
                    actual_rating_column: 'Rating'
                })
                
                st.sidebar.success(f"Dataset loaded! {len(new_data):,} reviews")
                st.sidebar.info(f"Detected columns: '{actual_review_column}' → 'Review_Text', '{actual_rating_column}' → 'Rating'")
                
                if uploaded_file.size > 10 * 1024 * 1024:
                    progress_bar.progress(75)
                    status_text.text("Preprocessing data...")
                
                # Update session state with new data
                # For very large datasets, optimize memory usage
                if len(new_data) > 100000:  # > 100k rows
                    st.sidebar.info("Large dataset detected. Optimizing memory usage...")
                    # Convert object columns to category to save memory
                    for col in new_data.select_dtypes(include=['object']).columns:
                        if col not in ['Review_Text', 'Review_Title', 'Features', 'Aspect_Sentiments']:
                            new_data[col] = new_data[col].astype('category')
                
                preprocessor = TextPreprocessor()
                processed_new_data = preprocessor.get_processed_data(new_data)
                
                if uploaded_file.size > 10 * 1024 * 1024:
                    progress_bar.progress(100)
                    status_text.text("Complete!")
                
                st.session_state.data = new_data
                st.session_state.processed_data = processed_new_data
                st.session_state.preprocessor = preprocessor
                st.session_state.data_loaded = True
                st.session_state.model_trained = False  # Reset model when new data is loaded
                
                st.sidebar.info("Dataset updated! You may need to retrain the model.")
                
                # Show dataset summary
                with st.sidebar.expander("Dataset Summary", expanded=False):
                    st.write(f"**Total Reviews:** {len(new_data):,}")
                    st.write(f"**Columns:** {len(new_data.columns)}")
                    st.write(f"**File Size:** {file_size_mb:.2f} MB")
                    
                    if 'Rating' in new_data.columns:
                        rating_dist = new_data['Rating'].value_counts().sort_index()
                        st.write("**Rating Distribution:**")
                        for rating, count in rating_dist.items():
                            st.write(f"  {rating} stars: {count:,} ({count/len(new_data)*100:.1f}%)")
                
            else:
                missing_info = []
                if not actual_review_column:
                    missing_info.append("review text")
                if not actual_rating_column:
                    missing_info.append("rating")
                
                st.sidebar.error(f"Could not find {', '.join(missing_info)} column(s)")
                st.sidebar.info("**Supported column names:**")
                st.sidebar.info(f"**Review Text:** {', '.join(review_text_columns)}")
                st.sidebar.info(f"**Rating:** {', '.join(rating_columns)}")
                st.sidebar.info("**Available columns in your file:**")
                st.sidebar.info(f"{list(new_data.columns)}")
                
        except Exception as e:
            st.sidebar.error(f"Error loading file: {str(e)}")
            st.sidebar.info("Please check that your file is a valid CSV format.")
    
    # Sample data button
    if st.sidebar.button("Generate Sample Data", use_container_width=True):
        st.sidebar.info("Generating sample data...")
        sample_data = create_sample_data()
        
        preprocessor = TextPreprocessor()
        processed_sample_data = preprocessor.get_processed_data(sample_data)
        
        st.session_state.data = sample_data
        st.session_state.processed_data = processed_sample_data
        st.session_state.preprocessor = preprocessor
        st.session_state.data_loaded = True
        st.session_state.model_trained = False
        
        st.sidebar.success("Sample data generated!")
        st.rerun()
    
    st.sidebar.markdown("---")
    
    # Data filters section with better organization
    st.sidebar.markdown("### Data Filters")
    st.sidebar.markdown("---")
    
    # Category filter with better styling
    category_column = None
    possible_category_columns = ['Category', 'category', 'cat', 'product_category']
    
    for col in possible_category_columns:
        if col in st.session_state.processed_data.columns:
            category_column = col
            break
    
    if category_column:
        categories = ['All Categories'] + list(st.session_state.processed_data[category_column].unique())
        selected_category = st.sidebar.selectbox(
            "Product Category", 
            categories,
            help="Filter reviews by product category",
            key="category_filter"
        )
        
        if selected_category != 'All Categories':
            filtered_data = st.session_state.processed_data[
                st.session_state.processed_data[category_column] == selected_category
            ]
        else:
            filtered_data = st.session_state.processed_data
    else:
        filtered_data = st.session_state.processed_data
    
    # Rating filter with better layout
    st.sidebar.markdown("**Rating Range**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        min_rating = st.slider("Min Rating", 1, 5, 1, help="Minimum star rating", key="min_rating")
    with col2:
        max_rating = st.slider("Max Rating", 1, 5, 5, help="Maximum star rating", key="max_rating")
    
    # Ensure min <= max
    if min_rating > max_rating:
        min_rating, max_rating = max_rating, min_rating
    
    # Find the correct rating column name
    rating_column = None
    possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
    
    for col in possible_rating_columns:
        if col in filtered_data.columns:
            rating_column = col
            break
    
    if rating_column:
        filtered_data = filtered_data[
            (filtered_data[rating_column] >= min_rating) & 
            (filtered_data[rating_column] <= max_rating)
        ]
    
    # Date range filter (if available)
    if 'reviewDate' in filtered_data.columns:
        st.sidebar.markdown("**Date Range**")
        try:
            filtered_data['reviewDate'] = pd.to_datetime(filtered_data['reviewDate'])
            min_date = filtered_data['reviewDate'].min().date()
            max_date = filtered_data['reviewDate'].max().date()
            
            date_range = st.sidebar.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                max_value=max_date,
                help="Filter reviews by date range",
                key="date_range"
            )
            
            if len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['reviewDate'].dt.date >= date_range[0]) &
                    (filtered_data['reviewDate'].dt.date <= date_range[1])
                ]
        except:
            pass
    
    # Add separator
    st.sidebar.markdown("---")
    
    # Model controls section with better organization
    st.sidebar.markdown("### Model Controls")
    
    # Model type selection with descriptions
    model_descriptions = {
        "logistic_regression": "Logistic Regression - Fast, interpretable",
        "random_forest": "Random Forest - Robust, handles non-linearity", 
        "svm": "Support Vector Machine - Good for high dimensions",
        "naive_bayes": "Naive Bayes - Fast, good baseline"
    }
    
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["logistic_regression", "random_forest", "svm", "naive_bayes"],
        help="Choose the machine learning algorithm",
        key="model_type"
    )
    
    # Show model description
    st.sidebar.info(f"**{model_descriptions[model_type]}**")
    
    # Training button with better styling
    st.sidebar.markdown("---")
    if st.sidebar.button("Train Model", type="primary", use_container_width=True, key="train_button"):
        with st.sidebar:
            with st.spinner("Training model..."):
                if train_model():
                    st.success("Model trained successfully!")
                else:
                    st.error("Failed to train model!")
    
    # Model status with better visual indicators
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Status")
    
    if st.session_state.model_trained:
        st.sidebar.success("Model Ready")
        
        # Show quick metrics in expandable section
        with st.sidebar.expander("Performance Metrics", expanded=True):
            if st.session_state.training_results:
                accuracy = st.session_state.training_results['evaluation']['accuracy']
                precision = st.session_state.training_results['evaluation']['classification_report']['weighted avg']['precision']
                recall = st.session_state.training_results['evaluation']['classification_report']['weighted avg']['recall']
                f1_score = st.session_state.training_results['evaluation']['classification_report']['weighted avg']['f1-score']
                
                st.metric("Accuracy", f"{accuracy:.3f}")
                st.metric("Precision", f"{precision:.3f}")
                st.metric("Recall", f"{recall:.3f}")
                st.metric("F1-Score", f"{f1_score:.3f}")
    else:
        st.sidebar.warning("Model Not Trained")
        st.sidebar.info("Click 'Train Model' to start training")
    
    # Advanced settings with better organization
    st.sidebar.markdown("---")
    with st.sidebar.expander("Advanced Settings", expanded=False):
        st.markdown("**Training Parameters**")
        max_features = st.slider("Max Features", 1000, 10000, 5000, help="Maximum number of text features")
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, help="Proportion of data for testing")
        random_state = st.number_input("Random State", 0, 100, 42, help="Seed for reproducibility")
        
        st.markdown("**Display Options**")
        show_debug = st.checkbox("Show Debug Info", help="Display additional debugging information")
        auto_refresh = st.checkbox("Auto Refresh", value=True, help="Automatically refresh when data changes")
    
    # Quick actions section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Reset Filters", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("View Data", use_container_width=True):
            st.session_state.show_data = True
    
    # Data summary
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Data Summary")
    st.sidebar.info(f"**Total Reviews:** {len(filtered_data):,}")
    if len(filtered_data) > 0:
        positive_count = len(filtered_data[filtered_data['sentiment'] == 'Positive'])
        negative_count = len(filtered_data[filtered_data['sentiment'] == 'Negative'])
        st.sidebar.info(f"**Positive:** {positive_count:,} ({positive_count/len(filtered_data)*100:.1f}%)")
        st.sidebar.info(f"**Negative:** {negative_count:,} ({negative_count/len(filtered_data)*100:.1f}%)")
    
    return filtered_data

def create_dashboard(data):
    """Create dashboard with modern UI"""
    st.markdown('<h2 class="sub-header fade-in-up">Analytics Dashboard</h2>', unsafe_allow_html=True)
    
    # Check if data is empty
    if len(data) == 0:
        st.warning("⚠️ No data available with current filters. Please adjust your filter settings.")
        return
    
    # Dataset summary with enhanced metrics
    summary = st.session_state.visualizer.get_dataset_summary(data)
    
    # Enhanced metrics cards with animations
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card info-metric fade-in-up">
            <h4>Total Reviews</h4>
            <h2>{summary['total_reviews']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card positive-metric fade-in-up">
            <h4>Positive Reviews</h4>
            <h2>{summary['positive_percentage']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card negative-metric fade-in-up">
            <h4>Negative Reviews</h4>
            <h2>{summary['negative_percentage']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = summary['avg_rating']
        st.markdown(f"""
        <div class="metric-card accuracy-metric fade-in-up">
            <h4>Average Rating</h4>
            <h2>{avg_rating:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance metrics (if trained)
    if st.session_state.model_trained:
        st.markdown('<h3 class="sub-header fade-in-up">Model Performance</h3>', unsafe_allow_html=True)
        
        metrics = st.session_state.training_results['evaluation']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}", delta=f"{metrics['accuracy']-0.5:.3f}")
        
        with col2:
            precision = metrics['classification_report']['weighted avg']['precision']
            st.metric("Precision", f"{precision:.3f}", delta=f"{precision-0.5:.3f}")
        
        with col3:
            recall = metrics['classification_report']['weighted avg']['recall']
            st.metric("Recall", f"{recall:.3f}", delta=f"{recall-0.5:.3f}")
        
        with col4:
            f1_score = metrics['classification_report']['weighted avg']['f1-score']
            st.metric("F1-Score", f"{f1_score:.3f}", delta=f"{f1_score-0.5:.3f}")
    
    # Charts section
    st.markdown('<h3 class="sub-header fade-in-up">Interactive Visualizations</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            st.session_state.visualizer.create_sentiment_pie_chart(data),
            use_container_width=True,
            key="sentiment_pie_chart",
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
    
    with col2:
        st.plotly_chart(
            st.session_state.visualizer.create_rating_distribution_chart(data),
            use_container_width=True,
            key="rating_distribution_chart",
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToRemove': ['pan2d', 'lasso2d', 'select2d'],
                'responsive': True
            }
        )
    
    # Additional insights
    st.markdown('<h3 class="sub-header fade-in-up">Key Insights</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Find the correct rating column name
        rating_column = None
        possible_rating_columns = ['overall', 'Rating', 'rating', 'score', 'stars', 'star_rating']
        
        for col in possible_rating_columns:
            if col in data.columns:
                rating_column = col
                break
        
        if rating_column:
            st.info(f"**Most Common Rating**: {data[rating_column].mode().iloc[0]} stars")
    
    with col2:
        # Find the correct category column name
        category_column = None
        possible_category_columns = ['Category', 'category', 'cat', 'product_category']
        
        for col in possible_category_columns:
            if col in data.columns:
                category_column = col
                break
        
        if category_column:
            top_category = data[category_column].mode().iloc[0]
            st.info(f"**Top Category**: {top_category}")
    
    with col3:
        # Find the correct review text column name
        review_text_column = None
        possible_text_columns = ['reviewText', 'Review_Text', 'text', 'review_text', 'content', 'review', 'comment']
        
        for col in possible_text_columns:
            if col in data.columns:
                review_text_column = col
                break
        
        if review_text_column:
            avg_text_length = data[review_text_column].str.len().mean()
            st.info(f"**Avg Review Length**: {avg_text_length:.0f} characters")
        else:
            st.info("**Avg Review Length**: Not available")
    
    # Download section
    st.markdown('<h3 class="sub-header fade-in-up">Export Data</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data (CSV)",
            data=csv,
            file_name=f"filtered_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        with col2:
            if st.button("Download All Charts", use_container_width=True):
                st.info("Chart download functionality would be implemented here")
        
        with col3:
            if st.button("Generate Report", use_container_width=True):
                st.info("Report generation functionality would be implemented here")

def create_prediction_tab():
    """Create prediction tab with modern UI"""
    st.markdown('<h2 class="sub-header fade-in-up">Sentiment Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Please train the model first using the sidebar controls!")
        return
    
    # Single review prediction with enhanced UI
    st.markdown("### Single Review Analysis")
    
    # Text input with styling
    review_text = st.text_area(
        "Enter a review to analyze:",
        placeholder="Type your review here... (e.g., 'This product is absolutely amazing! Great quality and fast shipping.')",
        height=120,
        help="Enter any product review text to analyze its sentiment"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        if st.button("Analyze Random Review", use_container_width=True):
            # Get a random review from the dataset
            random_review = st.session_state.processed_data.sample(1)['reviewText'].iloc[0]
            st.session_state.random_review = random_review
            st.rerun()
    
    # Show random review if selected
    if hasattr(st.session_state, 'random_review'):
        st.text_area("Random Review:", value=st.session_state.random_review, height=80, disabled=True)
        review_text = st.session_state.random_review
    
    if analyze_btn and review_text.strip():
        try:
            # Make prediction
            prediction = st.session_state.model.predict_single_review(
                review_text, st.session_state.preprocessor
            )
            
            # Enhanced results display
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment = prediction['prediction']
                confidence = prediction['confidence']
                
                if sentiment == 'Positive':
                    st.success(f"**{sentiment}** (Confidence: {confidence:.3f})")
                else:
                    st.error(f"**{sentiment}** (Confidence: {confidence:.3f})")
            
            with col2:
                st.markdown("**Probability Distribution:**")
                for class_name, prob in prediction['probabilities'].items():
                    st.write(f"{class_name}: {prob:.3f}")
            
            # Enhanced confidence visualization
            st.markdown("**Confidence Level:**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Additional insights
            st.markdown("### Analysis Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if confidence > 0.8:
                    st.success("High Confidence Prediction")
                elif confidence > 0.6:
                    st.warning("Medium Confidence Prediction")
                else:
                    st.error("Low Confidence Prediction")
            
            with col2:
                text_length = len(review_text.split())
                st.info(f"Review Length: {text_length} words")
            
            with col3:
                if sentiment == 'Positive':
                    st.success("Positive Sentiment Detected")
                else:
                    st.error("Negative Sentiment Detected")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction with UI
    st.markdown("### Batch Analysis")
    
    uploaded_file = st.file_uploader(
        "Upload a CSV file with reviews:",
        type=['csv'],
        help="CSV should have a 'reviewText' column. Other columns are optional.",
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            
            if 'reviewText' in batch_df.columns:
                st.success(f"File uploaded successfully! {len(batch_df)} reviews found.")
                
                # Preview data
                with st.expander("Preview Data"):
                    st.dataframe(batch_df.head(), use_container_width=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("Analyze All Reviews", type="primary", use_container_width=True):
                        with st.spinner("Processing reviews..."):
                            predictions = []
                            for text in batch_df['reviewText']:
                                pred = st.session_state.model.predict_single_review(
                                    text, st.session_state.preprocessor
                                )
                                predictions.append(pred['prediction'])
                            
                            batch_df['predicted_sentiment'] = predictions
                            
                            st.success("Predictions completed!")
                            
                            # Show results
                            st.dataframe(batch_df, use_container_width=True)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                with col2:
                    # Show statistics
                    if len(batch_df) > 0:
                        st.info(f"**File Statistics:**")
                        st.write(f"- Total Reviews: {len(batch_df)}")
                        st.write(f"- Columns: {list(batch_df.columns)}")
                        st.write(f"- File Size: {uploaded_file.size / 1024:.1f} KB")
            else:
                st.error("CSV file must contain a 'reviewText' column!")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def create_visualizations_tab(data):
    """Create visualizations tab"""
    st.markdown('<h2 class="sub-header fade-in-up">Data Visualizations</h2>', unsafe_allow_html=True)
    
    if len(data) == 0:
        st.warning("No data available with current filters.")
        return
    
    # Sentiment vs Rating scatter plot
    st.plotly_chart(
        st.session_state.visualizer.create_sentiment_vs_rating_scatter(data),
        use_container_width=True,
        key="sentiment_vs_rating_scatter"
    )
    
    # Category analysis
    category_chart = st.session_state.visualizer.create_category_analysis(data)
    if category_chart.data:  # Check if chart has data
        st.plotly_chart(
            category_chart,
            use_container_width=True,
            key="category_analysis_chart"
        )
    
    # Model performance charts
    if st.session_state.model_trained:
        st.markdown('<h3 class="sub-header fade-in-up">Model Performance Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                st.session_state.visualizer.create_model_performance_chart(
                    st.session_state.training_results['evaluation']
                ),
                use_container_width=True,
                key="model_performance_chart"
            )
        
        with col2:
            # Confusion matrix
            y_true = np.array(st.session_state.training_results['evaluation']['predictions'])
            y_pred = np.array(st.session_state.training_results['evaluation']['predictions'])
            st.plotly_chart(
                st.session_state.visualizer.create_confusion_matrix_heatmap(y_true, y_pred),
                use_container_width=True,
                key="confusion_matrix_heatmap"
            )
    
    # Top words charts
    st.markdown('<h3 class="sub-header fade-in-up">Most Frequent Words Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            st.session_state.visualizer.create_top_words_chart(data, 'Positive'),
            use_container_width=True,
            key="top_words_positive_chart"
        )
    
    with col2:
        st.plotly_chart(
            st.session_state.visualizer.create_top_words_chart(data, 'Negative'),
            use_container_width=True,
            key="top_words_negative_chart"
        )

def create_word_clouds_tab(data):
    """Create word clouds tab"""
    st.markdown('<h2 class="sub-header fade-in-up">Word Cloud Analysis</h2>', unsafe_allow_html=True)
    
    if len(data) == 0:
        st.warning("No data available with current filters.")
        return
    
    st.info("Word cloud functionality is temporarily disabled due to technical issues. This feature will be restored in a future update.")
    
    # Show alternative analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="sub-header fade-in-up">Positive Reviews</h4>', unsafe_allow_html=True)
        positive_data = data[data['sentiment'] == 'Positive'] if 'sentiment' in data.columns else data[data['Rating'] >= 4]
        st.info(f"Found {len(positive_data)} positive reviews")
        
        if len(positive_data) > 0:
            # Show top words for positive reviews
            st.plotly_chart(
                st.session_state.visualizer.create_top_words_chart(positive_data, 'Positive'),
                use_container_width=True,
                key="word_cloud_positive_chart"
            )
    
    with col2:
        st.markdown('<h4 class="sub-header fade-in-up">Negative Reviews</h4>', unsafe_allow_html=True)
        negative_data = data[data['sentiment'] == 'Negative'] if 'sentiment' in data.columns else data[data['Rating'] <= 2]
        st.info(f"Found {len(negative_data)} negative reviews")
        
        if len(negative_data) > 0:
            # Show top words for negative reviews
            st.plotly_chart(
                st.session_state.visualizer.create_top_words_chart(negative_data, 'Negative'),
                use_container_width=True,
                key="word_cloud_negative_chart"
            )

def main():
    """Main application function"""
    
    # Header with gradient background and responsive design
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.markdown('<h1>Customer Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Quick Demo Section
    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; margin: 1rem 0; text-align: center;">
            <h3 style="color: white; margin: 0 0 1rem 0;">🚀 Get Started Quickly!</h3>
            <p style="color: rgba(255,255,255,0.9); margin: 0;">Generate sample data or upload your own CSV file using the sidebar controls</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🎲 Generate 1000 Sample Reviews", help="Create sample data for testing"):
                with st.spinner('Generating sample data...'):
                    try:
                        from create_sample_data import generate_sample_data
                        sample_data = generate_sample_data(1000)
                        st.session_state.data = sample_data
                        st.session_state.data_loaded = True
                        st.session_state.processed_data = st.session_state.preprocessor.preprocess_data(sample_data)
                        st.success("✅ Sample data generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("📊 Generate 5000 Sample Reviews", help="Create larger sample dataset"):
                with st.spinner('Generating sample data...'):
                    try:
                        from create_sample_data import generate_sample_data
                        sample_data = generate_sample_data(5000)
                        st.session_state.data = sample_data
                        st.session_state.data_loaded = True
                        st.session_state.processed_data = st.session_state.preprocessor.preprocess_data(sample_data)
                        st.success("✅ Sample data generated successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        
        with col3:
            st.info("💡 **Tip**: Use the sidebar to upload your own CSV files or generate custom sample data!")
    
    # Load data if not already loaded with progress tracking
    if not st.session_state.data_loaded:
        with st.spinner('🔄 Loading data and initializing dashboard...'):
            if load_data():
                st.success("✅ Data loaded successfully!")
            else:
                st.error("❌ Failed to load data!")
                return
    
    # Create enhanced sidebar
    filtered_data = create_enhanced_sidebar()
    
    # Main content tabs with styling
    tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Prediction", "Visualizations", "Word Clouds"])
    
    with tab1:
        create_dashboard(filtered_data)
    
    with tab2:
        create_prediction_tab()
    
    with tab3:
        create_visualizations_tab(filtered_data)
    
    with tab4:
        create_word_clouds_tab(filtered_data)

if __name__ == "__main__":
    main()