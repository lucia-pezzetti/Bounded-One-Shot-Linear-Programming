from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler
import numpy as np

class RobustScaleOnlyScaler(BaseEstimator, TransformerMixin):
    """
    Robust scaler that only applies scaling (division by IQR) 
    without centering (no median subtraction).
    Uses RobustScaler with with_centering=False for better outlier robustness.
    """
    def __init__(self):
        self.scaler_ = RobustScaler(with_centering=False, with_scaling=True)
        self.scale_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y=None):
        """Fit the scaler by computing IQR-based scaling."""
        X = np.asarray(X)
        self.n_features_in_ = X.shape[1]
        self.scaler_.fit(X)
        self.scale_ = self.scaler_.scale_
        return self
    
    def transform(self, X):
        """Transform data by dividing by IQR-based scale."""
        X = np.asarray(X)
        return self.scaler_.transform(X)
    
    def inverse_transform(self, X):
        """Inverse transform by multiplying by IQR-based scale."""
        X = np.asarray(X)
        return X * self.scale_


class RobustBlockScaleOnlyScaler(BaseEstimator, TransformerMixin):
    """
    Robust scaler that scales state and action blocks separately (no centering), 
    using per-dimension IQR-based scaling.
    Keeps the same (x,u) concatenated API: fit/transform/inverse_transform expect
    arrays shaped (n_samples, dx+du).
    """
    def __init__(self, dx: int, du: int):
        self.dx = dx
        self.du = du
        self.scaler_x_ = RobustScaler(with_centering=False, with_scaling=True)
        self.scaler_u_ = RobustScaler(with_centering=False, with_scaling=True)
        self.scale_x_ = None
        self.scale_u_ = None
        self.n_features_in_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        assert X.shape[1] == self.dx + self.du, "X must have dx + du columns"
        self.n_features_in_ = X.shape[1]
        X_block = X[:, :self.dx]
        U_block = X[:, self.dx:]
        
        # Fit robust scalers on each block separately
        self.scaler_x_.fit(X_block)
        self.scaler_u_.fit(U_block)
        
        self.scale_x_ = self.scaler_x_.scale_
        self.scale_u_ = self.scaler_u_.scale_
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_block = self.scaler_x_.transform(X[:, :self.dx])
        U_block = self.scaler_u_.transform(X[:, self.dx:])
        return np.concatenate([X_block, U_block], axis=1)

    def inverse_transform(self, X):
        X = np.asarray(X)
        X_block = X[:, :self.dx] * self.scale_x_
        U_block = X[:, self.dx:] * self.scale_u_
        return np.concatenate([X_block, U_block], axis=1)