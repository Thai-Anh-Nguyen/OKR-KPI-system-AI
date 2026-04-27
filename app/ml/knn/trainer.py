"""KNN Model Training Utilities"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class KNNModelTrainer:
    """Train and save KNN risk model"""
    
    @staticmethod
    def train_and_save(
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_cols: list,
        output_dir: str = "./models",
        n_neighbors: int = 5,
        metric: str = "euclidean"
    ):
        """
        Train KNN model and save artifacts
        
        Args:
            X_train: Training features
            y_train: Training labels
            feature_cols: List of feature column names
            output_dir: Directory to save model
            n_neighbors: Number of neighbors for KNN
            metric: Distance metric
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        
        # Train KNN
        model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            metric=metric,
            weights='distance'
        )
        model.fit(X_scaled, y_train)
        
        # Create pipeline-like wrapper that includes scaler
        class ModelWithScaler:
            def __init__(self, scaler, knn):
                self.scaler = scaler
                self.knn = knn
            
            def predict(self, X):
                X_scaled = self.scaler.transform(X)
                return self.knn.predict(X_scaled)
            
            def score(self, X, y):
                X_scaled = self.scaler.transform(X)
                return self.knn.score(X_scaled, y)
        
        final_model = ModelWithScaler(scaler, model)
        
        # Save artifacts
        model_path = os.path.join(output_dir, 'knn_risk_model.pkl')
        cols_path = os.path.join(output_dir, 'feature_columns.pkl')
        
        joblib.dump(final_model, model_path)
        joblib.dump(feature_cols, cols_path)
        
        logger.info(f"Model saved to {model_path}")
        return final_model
