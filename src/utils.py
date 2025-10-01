from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ScaleOnlyScaler(BaseEstimator, TransformerMixin):
    """
    Custom scaler that only applies scaling (division by standard deviation) 
    without centering (no mean subtraction).
    """
    def __init__(self):
        self.scale_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """Fit the scaler by computing standard deviations."""
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.scale_ = np.std(X, axis=0)
        # Avoid division by zero
        self.scale_[self.scale_ == 0] = 1.0
        return self
    
    def transform(self, X):
        """Transform data by dividing by standard deviations."""
        X = np.asarray(X)
        return X / self.scale_
    
    def inverse_transform(self, X):
        """Inverse transform by multiplying by standard deviations."""
        X = np.asarray(X)
        return X * self.scale_