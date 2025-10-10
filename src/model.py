"""
Machine Learning Model Module for Sentiment Analysis
Handles model training, evaluation, and prediction for the Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')


class SentimentModel:
    """
    A comprehensive sentiment analysis model class for the Streamlit app
    """
    
    def __init__(self, model_type: str = 'logistic_regression'):
        """
        Initialize the sentiment model
        
        Args:
            model_type (str): Type of model to use ('logistic_regression', 'random_forest', 'svm', 'naive_bayes')
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        self.classes_ = None
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """
        Initialize the machine learning model
        """
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                random_state=42,
                max_iter=1000,
                C=1.0,
                solver='liblinear'
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='linear',
                random_state=42,
                C=1.0,
                probability=True
            )
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB(alpha=1.0)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Train the sentiment analysis model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            
        Returns:
            dict: Training results and metrics
        """
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        self.classes_ = self.model.classes_
        
        # Calculate training accuracy
        train_predictions = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        results = {
            'model_type': self.model_type,
            'train_accuracy': train_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'classes': self.classes_.tolist(),
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0]
        }
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        Evaluate the model on test data
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Get classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        return results
    
    def predict_single_review(self, review_text: str, preprocessor) -> dict:
        """
        Predict sentiment for a single review
        
        Args:
            review_text (str): Review text to analyze
            preprocessor: Fitted preprocessor object
            
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Transform the text
        X = preprocessor.transform_new_text([review_text])
        
        # Make prediction
        prediction = self.predict(X)[0]
        probabilities = self.predict_proba(X)[0]
        
        # Create probability dictionary
        prob_dict = {class_name: prob for class_name, prob in zip(self.classes_, probabilities)}
        
        return {
            'text': review_text,
            'prediction': prediction,
            'probabilities': prob_dict,
            'confidence': max(probabilities)
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath (str): Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'classes': self.classes_,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk
        
        Args:
            filepath (str): Path to the saved model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.classes_ = model_data['classes']
        self.is_trained = model_data['is_trained']
    
    def get_model_info(self) -> dict:
        """
        Get model information
        
        Returns:
            dict: Model information
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'classes': self.classes_.tolist() if self.classes_ is not None else None
        }


class ModelTrainer:
    """
    A class to handle model training pipeline
    """
    
    def __init__(self, preprocessor, model_type: str = 'logistic_regression'):
        """
        Initialize the model trainer
        
        Args:
            preprocessor: TextPreprocessor object
            model_type (str): Type of model to train
        """
        self.preprocessor = preprocessor
        self.model = SentimentModel(model_type)
        self.training_results = None
        self.evaluation_results = None
    
    def train_model(self, df: pd.DataFrame) -> dict:
        """
        Complete model training pipeline
        
        Args:
            df (pd.DataFrame): Processed dataframe
            
        Returns:
            dict: Training and evaluation results
        """
        # Vectorize text
        X, y = self.preprocessor.vectorize_text(df)
        
        # Split data
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y)
        
        # Train model
        self.training_results = self.model.train(X_train, y_train)
        
        # Evaluate model
        self.evaluation_results = self.model.evaluate(X_test, y_test)
        
        return {
            'training': self.training_results,
            'evaluation': self.evaluation_results
        }
    
    def get_model(self) -> SentimentModel:
        """
        Get the trained model
        
        Returns:
            SentimentModel: Trained model
        """
        return self.model
    
    def predict_review(self, review_text: str) -> dict:
        """
        Predict sentiment for a review
        
        Args:
            review_text (str): Review text
            
        Returns:
            dict: Prediction results
        """
        return self.model.predict_single_review(review_text, self.preprocessor)
    
    def get_metrics_summary(self) -> dict:
        """
        Get a summary of model metrics
        
        Returns:
            dict: Metrics summary
        """
        if not self.evaluation_results:
            return {}
        
        return {
            'accuracy': self.evaluation_results['accuracy'],
            'precision': self.evaluation_results['classification_report']['weighted avg']['precision'],
            'recall': self.evaluation_results['classification_report']['weighted avg']['recall'],
            'f1_score': self.evaluation_results['classification_report']['weighted avg']['f1-score']
        }
