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
    page_title="📊 Customer Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "Customer Sentiment Analysis Dashboard v2.0"
    }
)

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
                    url('ecommerce-bg.jpg');
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
        data_path = "data/amazon_reviews.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            # Create sample data if file doesn't exist
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

def create_sidebar():
    """Create sidebar with modern controls"""
    st.sidebar.markdown("## 🎛️ Control Panel")
    
    # Data filters section
    st.sidebar.markdown("### 📊 Data Filters")
    
    # Category filter
    if 'category' in st.session_state.processed_data.columns:
        categories = ['All Categories'] + list(st.session_state.processed_data['category'].unique())
        selected_category = st.sidebar.selectbox(
            "🏷️ Product Category", 
            categories,
            help="Filter reviews by product category"
        )
        
        if selected_category != 'All Categories':
            filtered_data = st.session_state.processed_data[
                st.session_state.processed_data['category'] == selected_category
            ]
        else:
            filtered_data = st.session_state.processed_data
    else:
        filtered_data = st.session_state.processed_data
    
    # Rating filter with slider
    st.sidebar.markdown("⭐ **Rating Range**")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        min_rating = st.slider("Min Rating", 1, 5, 1, help="Minimum star rating")
    with col2:
        max_rating = st.slider("Max Rating", 1, 5, 5, help="Maximum star rating")
    
    # Ensure min <= max
    if min_rating > max_rating:
        min_rating, max_rating = max_rating, min_rating
    
    filtered_data = filtered_data[
        (filtered_data['overall'] >= min_rating) & 
        (filtered_data['overall'] <= max_rating)
    ]
    
    # Date range filter (if available)
    if 'reviewDate' in filtered_data.columns:
        st.sidebar.markdown("📅 **Date Range**")
        try:
            filtered_data['reviewDate'] = pd.to_datetime(filtered_data['reviewDate'])
            min_date = filtered_data['reviewDate'].min().date()
            max_date = filtered_data['reviewDate'].max().date()
            
            date_range = st.sidebar.date_input(
                        "Select Date Range",
                        value=(min_date, max_date),
                        min_value=min_date,
                max_value=max_date,
                help="Filter reviews by date range"
            )
            
            if len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['reviewDate'].dt.date >= date_range[0]) &
                    (filtered_data['reviewDate'].dt.date <= date_range[1])
                ]
        except:
            pass
    
    # Model controls section
    st.sidebar.markdown("### 🤖 Model Controls")
    
    # Model type selection
    model_type = st.sidebar.selectbox(
        "🧠 Model Type",
        ["logistic_regression", "random_forest", "svm", "naive_bayes"],
        help="Choose the machine learning algorithm"
    )
    
    # Training button with progress
    if st.sidebar.button("🚀 Train Model", type="primary", use_container_width=True):
        if train_model():
            st.sidebar.success("✅ Model trained successfully!")
        else:
            st.sidebar.error("❌ Failed to train model!")
    
    # Model status indicator
    if st.session_state.model_trained:
        st.sidebar.markdown("### 📈 Model Status")
        st.sidebar.success("🟢 Model Ready")
        
        # Show quick metrics
        if st.session_state.training_results:
            accuracy = st.session_state.training_results['evaluation']['accuracy']
            st.sidebar.metric("Accuracy", f"{accuracy:.3f}")
        else:
            st.sidebar.warning("🟡 Model Not Trained")
    
    # Settings
    with st.sidebar.expander("⚙️ Settings"):
        max_features = st.slider("Max Features", 1000, 10000, 5000)
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Random State", 0, 100, 42)
    
    return filtered_data

def create_dashboard(data):
    """Create dashboard with modern UI"""
    st.markdown('<h2 class="sub-header fade-in-up">📈 Analytics Dashboard</h2>', unsafe_allow_html=True)
    
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
            <h4>📊 Total Reviews</h4>
            <h2>{summary['total_reviews']:,}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card positive-metric fade-in-up">
            <h4>😊 Positive Reviews</h4>
            <h2>{summary['positive_percentage']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card negative-metric fade-in-up">
            <h4>😞 Negative Reviews</h4>
            <h2>{summary['negative_percentage']:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_rating = summary['avg_rating']
        st.markdown(f"""
        <div class="metric-card accuracy-metric fade-in-up">
            <h4>⭐ Average Rating</h4>
            <h2>{avg_rating:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Model performance metrics (if trained)
    if st.session_state.model_trained:
        st.markdown('<h3 class="sub-header fade-in-up">🤖 Model Performance</h3>', unsafe_allow_html=True)
        
        metrics = st.session_state.training_results['evaluation']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Accuracy", f"{metrics['accuracy']:.3f}", delta=f"{metrics['accuracy']-0.5:.3f}")
        
        with col2:
            precision = metrics['classification_report']['weighted avg']['precision']
            st.metric("🎯 Precision", f"{precision:.3f}", delta=f"{precision-0.5:.3f}")
        
        with col3:
            recall = metrics['classification_report']['weighted avg']['recall']
            st.metric("🎯 Recall", f"{recall:.3f}", delta=f"{recall-0.5:.3f}")
        
        with col4:
            f1_score = metrics['classification_report']['weighted avg']['f1-score']
            st.metric("🎯 F1-Score", f"{f1_score:.3f}", delta=f"{f1_score-0.5:.3f}")
    
    # Charts section
    st.markdown('<h3 class="sub-header fade-in-up">📊 Interactive Visualizations</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            st.session_state.visualizer.create_sentiment_pie_chart(data),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            st.session_state.visualizer.create_rating_distribution_chart(data),
            use_container_width=True
        )
    
    # Additional insights
    st.markdown('<h3 class="sub-header fade-in-up">💡 Key Insights</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"📈 **Most Common Rating**: {data['overall'].mode().iloc[0]} stars")
    
    with col2:
        if 'category' in data.columns:
            top_category = data['category'].mode().iloc[0]
            st.info(f"🏷️ **Top Category**: {top_category}")
    
    with col3:
        avg_text_length = data['reviewText'].str.len().mean()
        st.info(f"📝 **Avg Review Length**: {avg_text_length:.0f} characters")
    
    # Download section
    st.markdown('<h3 class="sub-header fade-in-up">📥 Export Data</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        csv = data.to_csv(index=False)
        st.download_button(
            label="📊 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"filtered_reviews_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        with col2:
            if st.button("📈 Download All Charts", use_container_width=True):
                st.info("Chart download functionality would be implemented here")
        
        with col3:
            if st.button("📋 Generate Report", use_container_width=True):
                st.info("Report generation functionality would be implemented here")

def create_prediction_tab():
    """Create prediction tab with modern UI"""
    st.markdown('<h2 class="sub-header fade-in-up">🔮 Sentiment Prediction</h2>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first using the sidebar controls!")
        return
    
    # Single review prediction with enhanced UI
    st.markdown("### 📝 Single Review Analysis")
    
    # Text input with styling
    review_text = st.text_area(
        "Enter a review to analyze:",
        placeholder="Type your review here... (e.g., 'This product is absolutely amazing! Great quality and fast shipping.')",
        height=120,
        help="Enter any product review text to analyze its sentiment"
    )
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        analyze_btn = st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🎲 Analyze Random Review", use_container_width=True):
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
                    st.success(f"✅ **{sentiment}** (Confidence: {confidence:.3f})")
                else:
                    st.error(f"❌ **{sentiment}** (Confidence: {confidence:.3f})")
            
            with col2:
                st.markdown("**📊 Probability Distribution:**")
                for class_name, prob in prediction['probabilities'].items():
                    st.write(f"{class_name}: {prob:.3f}")
            
            # Enhanced confidence visualization
            st.markdown("**🎯 Confidence Level:**")
            st.progress(confidence)
            st.caption(f"Confidence: {confidence:.1%}")
            
            # Additional insights
            st.markdown("### 💡 Analysis Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if confidence > 0.8:
                    st.success("🟢 High Confidence Prediction")
                elif confidence > 0.6:
                    st.warning("🟡 Medium Confidence Prediction")
                else:
                    st.error("🔴 Low Confidence Prediction")
            
            with col2:
                text_length = len(review_text.split())
                st.info(f"📝 Review Length: {text_length} words")
            
            with col3:
                if sentiment == 'Positive':
                    st.success("😊 Positive Sentiment Detected")
                else:
                    st.error("😞 Negative Sentiment Detected")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Batch prediction with UI
    st.markdown("### 📁 Batch Analysis")
    
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
                st.success(f"✅ File uploaded successfully! {len(batch_df)} reviews found.")
                
                # Preview data
                with st.expander("📋 Preview Data"):
                    st.dataframe(batch_df.head(), use_container_width=True)
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if st.button("🚀 Analyze All Reviews", type="primary", use_container_width=True):
                        with st.spinner("Processing reviews..."):
                            predictions = []
                            for text in batch_df['reviewText']:
                                pred = st.session_state.model.predict_single_review(
                                    text, st.session_state.preprocessor
                                )
                                predictions.append(pred['prediction'])
                            
                            batch_df['predicted_sentiment'] = predictions
                            
                            st.success("✅ Predictions completed!")
                            
                            # Show results
                            st.dataframe(batch_df, use_container_width=True)
                            
                            # Download results
                            csv = batch_df.to_csv(index=False)
                            st.download_button(
                                label="📊 Download Predictions (CSV)",
                                data=csv,
                                file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                
                with col2:
                    # Show statistics
                    if len(batch_df) > 0:
                        st.info(f"📊 **File Statistics:**")
                        st.write(f"- Total Reviews: {len(batch_df)}")
                        st.write(f"- Columns: {list(batch_df.columns)}")
                        st.write(f"- File Size: {uploaded_file.size / 1024:.1f} KB")
            else:
                st.error("❌ CSV file must contain a 'reviewText' column!")
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

def create_visualizations_tab(data):
    """Create visualizations tab"""
    st.markdown('<h2 class="sub-header fade-in-up">📊 Data Visualizations</h2>', unsafe_allow_html=True)
    
    if len(data) == 0:
        st.warning("⚠️ No data available with current filters.")
        return
    
    # Sentiment vs Rating scatter plot
    st.plotly_chart(
        st.session_state.visualizer.create_sentiment_vs_rating_scatter(data),
        use_container_width=True
    )
    
    # Category analysis
    if 'category' in data.columns:
        st.plotly_chart(
            st.session_state.visualizer.create_category_analysis(data),
            use_container_width=True
        )
    
    # Model performance charts
    if st.session_state.model_trained:
        st.markdown('<h3 class="sub-header fade-in-up">🤖 Model Performance Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(
                st.session_state.visualizer.create_model_performance_chart(
                    st.session_state.training_results['evaluation']
                ),
                use_container_width=True
            )
        
        with col2:
            # Confusion matrix
            y_true = np.array(st.session_state.training_results['evaluation']['predictions'])
            y_pred = np.array(st.session_state.training_results['evaluation']['predictions'])
            st.plotly_chart(
                st.session_state.visualizer.create_confusion_matrix_heatmap(y_true, y_pred),
                use_container_width=True
            )
    
    # Top words charts
    st.markdown('<h3 class="sub-header fade-in-up">📝 Most Frequent Words Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.plotly_chart(
            st.session_state.visualizer.create_top_words_chart(data, 'Positive'),
            use_container_width=True
        )
    
    with col2:
        st.plotly_chart(
            st.session_state.visualizer.create_top_words_chart(data, 'Negative'),
            use_container_width=True
        )

def create_word_clouds_tab(data):
    """Create word clouds tab"""
    st.markdown('<h2 class="sub-header fade-in-up">☁️ Word Cloud Analysis</h2>', unsafe_allow_html=True)
    
    if len(data) == 0:
        st.warning("⚠️ No data available with current filters.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<h4 class="sub-header fade-in-up">😊 Positive Reviews</h4>', unsafe_allow_html=True)
        positive_wc = st.session_state.visualizer.create_word_cloud(data, 'Positive')
        if positive_wc:
            st.image(f"data:image/png;base64,{positive_wc}", use_container_width=True)
        else:
            st.info("No positive reviews found in the filtered data.")
    
    with col2:
        st.markdown('<h4 class="sub-header fade-in-up">😞 Negative Reviews</h4>', unsafe_allow_html=True)
        negative_wc = st.session_state.visualizer.create_word_cloud(data, 'Negative')
        if negative_wc:
            st.image(f"data:image/png;base64,{negative_wc}", use_container_width=True)
        else:
            st.info("No negative reviews found in the filtered data.")

def main():
    """Main application function"""
    
    # Header with gradient background
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="main-header">📊 Customer Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Professional Machine Learning Web Application with Modern UI</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data if not already loaded
    if not st.session_state.data_loaded:
        if load_data():
            st.success("✅ Data loaded successfully!")
        else:
            st.error("❌ Failed to load data!")
            return
    
    # Create sidebar
    filtered_data = create_sidebar()
    
    # Main content tabs with styling
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Dashboard", "🔮 Prediction", "📊 Visualizations", "☁️ Word Clouds"])
    
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