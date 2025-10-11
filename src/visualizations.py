"""
Visualization Module for Sentiment Analysis
Creates comprehensive visualizations for the Streamlit app
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set style for matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SentimentVisualizer:
    """
    A comprehensive visualization class for sentiment analysis results
    """
    
    def __init__(self):
        """
        Initialize the visualizer
        """
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#FFD700'    # Gold
        }
    
    def create_sentiment_pie_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a pie chart for sentiment distribution
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment column
            
        Returns:
            plotly.graph_objects.Figure: Pie chart figure
        """
        sentiment_counts = df['sentiment'].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            marker_colors=['#2E8B57', '#DC143C'],
            textinfo='label+percent+value',
            textfont_size=12,
            hole=0.3
        )])
        
        fig.update_layout(
            title="Sentiment Distribution",
            title_x=0.5,
            font=dict(size=14),
            showlegend=True,
            height=400
        )
        
        return fig
    
    def create_rating_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart for rating distribution
        
        Args:
            df (pd.DataFrame): DataFrame with overall rating column
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        rating_counts = df['overall'].value_counts().sort_index()
        
        fig = go.Figure(data=[go.Bar(
            x=rating_counts.index,
            y=rating_counts.values,
            marker_color=['#DC143C', '#FF6347', '#FFD700', '#90EE90', '#2E8B57'],
            text=rating_counts.values,
            textposition='auto',
        )])
        
        fig.update_layout(
            title="Rating Distribution",
            title_x=0.5,
            xaxis_title="Rating (Stars)",
            yaxis_title="Number of Reviews",
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def create_confusion_matrix_heatmap(self, y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
        """
        Create a confusion matrix heatmap
        
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            
        Returns:
            plotly.graph_objects.Figure: Heatmap figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Negative', 'Positive'],
            y=['Negative', 'Positive'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Confusion Matrix",
            title_x=0.5,
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def create_word_cloud(self, df: pd.DataFrame, sentiment: str, 
                         text_column: str = 'reviewText_processed') -> str:
        """
        Create word cloud for specific sentiment
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            sentiment (str): Sentiment to filter ('Positive' or 'Negative')
            text_column (str): Name of the text column
            
        Returns:
            str: Base64 encoded image string
        """
        import base64
        import io
        
        # Filter data by sentiment
        sentiment_data = df[df['sentiment'] == sentiment]
        
        if len(sentiment_data) == 0:
            return None
        
        # Combine all text
        text = ' '.join(sentiment_data[text_column].astype(str))
        
        if not text.strip():
            return None
        
        try:
            # Create word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color='white',
                colormap='viridis' if sentiment == 'Positive' else 'Reds',
                max_words=100,
                relative_scaling=0.5,
                random_state=42
            ).generate(text)
            
            # Convert to base64 string
            img_buffer = io.BytesIO()
            
            # Create figure and axis properly
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            # Display the word cloud
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"Word Cloud - {sentiment} Reviews", fontsize=16, fontweight='bold', pad=20)
            
            # Save to buffer
            plt.tight_layout()
            plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            img_buffer.seek(0)
            img_str = base64.b64encode(img_buffer.read()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"Error creating word cloud: {e}")
            return None
    
    def create_top_words_chart(self, df: pd.DataFrame, sentiment: str, 
                              text_column: str = 'reviewText_processed',
                              top_n: int = 20) -> go.Figure:
        """
        Create a horizontal bar chart for top words
        
        Args:
            df (pd.DataFrame): DataFrame with text data
            sentiment (str): Sentiment to filter
            text_column (str): Name of the text column
            top_n (int): Number of top words to show
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        # Filter data by sentiment
        sentiment_data = df[df['sentiment'] == sentiment]
        
        if len(sentiment_data) == 0:
            return go.Figure()
        
        # Combine all text and split into words
        all_text = ' '.join(sentiment_data[text_column].astype(str))
        words = all_text.split()
        
        # Count word frequencies
        word_counts = Counter(words)
        top_words = word_counts.most_common(top_n)
        
        if not top_words:
            return go.Figure()
        
        words, counts = zip(*top_words)
        color = '#2E8B57' if sentiment == 'Positive' else '#DC143C'
        
        fig = go.Figure(data=[go.Bar(
            x=list(counts),
            y=list(words),
            orientation='h',
            marker_color=color,
            text=list(counts),
            textposition='auto',
        )])
        
        fig.update_layout(
            title=f"Top {top_n} Words - {sentiment} Reviews",
            title_x=0.5,
            xaxis_title="Frequency",
            yaxis_title="Words",
            font=dict(size=12),
            height=500
        )
        
        return fig
    
    def create_model_performance_chart(self, results: dict) -> go.Figure:
        """
        Create a bar chart for model performance metrics
        
        Args:
            results (dict): Model evaluation results
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        metrics = ['Precision', 'Recall', 'F1-Score']
        values = [
            results['classification_report']['weighted avg']['precision'],
            results['classification_report']['weighted avg']['recall'],
            results['classification_report']['weighted avg']['f1-score']
        ]
        
        fig = go.Figure(data=[go.Bar(
            x=metrics,
            y=values,
            marker_color=['#3498db', '#e74c3c', '#2ecc71'],
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        )])
        
        fig.update_layout(
            title="Model Performance Metrics",
            title_x=0.5,
            yaxis_title="Score",
            font=dict(size=12),
            height=400
        )
        
        # Add accuracy line
        accuracy = results['accuracy']
        fig.add_hline(y=accuracy, line_dash="dash", line_color="red",
                     annotation_text=f"Accuracy: {accuracy:.3f}")
        
        return fig
    
    def create_sentiment_vs_rating_scatter(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a scatter plot showing sentiment vs rating
        
        Args:
            df (pd.DataFrame): DataFrame with sentiment and rating columns
            
        Returns:
            plotly.graph_objects.Figure: Scatter plot figure
        """
        sentiment_numeric = df['sentiment'].map({'Negative': 0, 'Positive': 1})
        
        fig = go.Figure(data=[go.Scatter(
            x=df['overall'],
            y=sentiment_numeric,
            mode='markers',
            marker=dict(
                color=sentiment_numeric,
                colorscale='RdYlGn',
                size=8,
                opacity=0.7
            ),
            text=df['sentiment'],
            hovertemplate='Rating: %{x}<br>Sentiment: %{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Sentiment vs Rating",
            title_x=0.5,
            xaxis_title="Rating (Stars)",
            yaxis_title="Sentiment (0=Negative, 1=Positive)",
            font=dict(size=12),
            height=400
        )
        
        return fig
    
    def create_category_analysis(self, df: pd.DataFrame) -> go.Figure:
        """
        Create a bar chart for sentiment by category
        
        Args:
            df (pd.DataFrame): DataFrame with category and sentiment columns
            
        Returns:
            plotly.graph_objects.Figure: Bar chart figure
        """
        if 'category' not in df.columns:
            return go.Figure()
        
        # Create cross-tabulation
        cross_tab = pd.crosstab(df['category'], df['sentiment'])
        
        fig = go.Figure()
        
        for sentiment in cross_tab.columns:
            fig.add_trace(go.Bar(
                name=sentiment,
                x=cross_tab.index,
                y=cross_tab[sentiment],
                marker_color=self.colors.get(sentiment.lower(), '#3498db')
            ))
        
        fig.update_layout(
            title="Sentiment Distribution by Category",
            title_x=0.5,
            xaxis_title="Category",
            yaxis_title="Number of Reviews",
            font=dict(size=12),
            height=500,
            barmode='group'
        )
        
        return fig
    
    def get_dataset_summary(self, df: pd.DataFrame) -> dict:
        """
        Get a summary of the dataset
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            dict: Dataset summary
        """
        total_reviews = len(df)
        
        if total_reviews == 0:
            return {
                'total_reviews': 0,
                'positive_reviews': 0,
                'negative_reviews': 0,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'avg_rating': 0,
                'categories': 0
            }
        
        positive_reviews = len(df[df['sentiment'] == 'Positive'])
        negative_reviews = len(df[df['sentiment'] == 'Negative'])
        
        summary = {
            'total_reviews': total_reviews,
            'positive_reviews': positive_reviews,
            'negative_reviews': negative_reviews,
            'positive_percentage': (positive_reviews / total_reviews) * 100,
            'negative_percentage': (negative_reviews / total_reviews) * 100,
            'avg_rating': df['overall'].mean() if 'overall' in df.columns else 0,
            'categories': df['category'].nunique() if 'category' in df.columns else 0
        }
        
        return summary
