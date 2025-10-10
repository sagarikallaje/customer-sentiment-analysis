"""
Data Preprocessing Module for Sentiment Analysis
Handles text cleaning, preprocessing, and feature engineering for the Streamlit app
"""

import pandas as pd
import numpy as np
import re
import string
from typing import Tuple, List
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class TextPreprocessor:
    """
    A comprehensive text preprocessing class for sentiment analysis
    """
    
    def __init__(self):
        """
        Initialize the TextPreprocessor
        """
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text string
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove punctuation and special characters
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with stopwords removed
        """
        if not text:
            return ""
            
        words = word_tokenize(text)
        filtered_words = [word for word in words if word not in self.stop_words]
        return ' '.join(filtered_words)
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'reviewText') -> pd.DataFrame:
        """
        Preprocess entire dataframe
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the text column
            
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Remove missing values
        initial_count = len(processed_df)
        processed_df = processed_df.dropna(subset=[text_column])
        removed_count = initial_count - len(processed_df)
        
        # Clean text
        processed_df[f'{text_column}_cleaned'] = processed_df[text_column].apply(self.clean_text)
        
        # Remove stopwords
        processed_df[f'{text_column}_processed'] = processed_df[f'{text_column}_cleaned'].apply(self.remove_stopwords)
        
        # Remove empty reviews after preprocessing
        processed_df = processed_df[processed_df[f'{text_column}_processed'].str.len() > 0]
        
        return processed_df
    
    def create_sentiment_labels(self, df: pd.DataFrame, rating_column: str = 'overall') -> pd.DataFrame:
        """
        Convert ratings to sentiment labels
        
        Args:
            df (pd.DataFrame): Input dataframe
            rating_column (str): Name of the rating column
            
        Returns:
            pd.DataFrame: Dataframe with sentiment labels
        """
        # Create sentiment labels: 1-3 = Negative, 4-5 = Positive
        df['sentiment'] = df[rating_column].apply(
            lambda x: 'Negative' if x <= 3 else 'Positive'
        )
        
        return df
    
    def vectorize_text(self, df: pd.DataFrame, text_column: str = 'reviewText_processed', 
                      max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorize text data using TF-IDF
        
        Args:
            df (pd.DataFrame): Input dataframe
            text_column (str): Name of the processed text column
            max_features (int): Maximum number of features
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix and target labels
        """
        # Prepare text data
        texts = df[text_column].tolist()
        labels = df['sentiment'].tolist()
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95  # Ignore terms that appear in more than 95% of documents
        )
        
        # Fit and transform
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        return X, y
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        return X_train, X_test, y_train, y_test
    
    def transform_new_text(self, texts: List[str]) -> np.ndarray:
        """
        Transform new text data using fitted vectorizer
        
        Args:
            texts (List[str]): List of new texts to transform
            
        Returns:
            np.ndarray: Transformed feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Please run vectorize_text first.")
        
        # Clean and preprocess new texts
        cleaned_texts = []
        for text in texts:
            cleaned = self.clean_text(text)
            cleaned = self.remove_stopwords(cleaned)
            cleaned_texts.append(cleaned)
        
        # Transform using fitted vectorizer
        return self.vectorizer.transform(cleaned_texts)
    
    def get_processed_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get the complete processed dataset
        
        Args:
            df (pd.DataFrame): Original dataframe
            
        Returns:
            pd.DataFrame: Fully processed dataframe
        """
        # Preprocess text
        processed_df = self.preprocess_dataframe(df)
        
        # Create sentiment labels
        processed_df = self.create_sentiment_labels(processed_df)
        
        return processed_df
