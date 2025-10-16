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
    
    # File uploader with better styling
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
    
    with st.sidebar.expander("🔧 Troubleshooting"):
        st.markdown("""
        **No Data Showing?**
        - Check if filters are too restrictive
        - Try resetting filters
        - Verify data format
        
        **Slow Performance?**
        - Reduce data size with filters
        - Use quick filters for faster results
        """)
    
    # Store filtered data in session state
    st.session_state.filtered_data = filtered_data
    
    return filtered_data
