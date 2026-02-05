"""
Simple polynomial features generator that excludes u^2 terms.

This module provides a simple polynomial feature generator that:
- Generates polynomial features from state-action pairs [x, u]
- Excludes terms where any control variable has degree >= 2 (i.e., u^2 terms)
- No scaling is applied
"""

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin


class FilteredPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Polynomial features generator that excludes u^2 terms.
    
    This class wraps sklearn's PolynomialFeatures and filters out terms
    where control variables have degree >= 2 (i.e., u^2 terms).
    """
    
    def __init__(self, degree=2, include_bias=False, dx=None, du=None):
        """
        Initialize the filtered polynomial features generator.
        
        Args:
            degree: Polynomial degree
            include_bias: Whether to include bias term
            dx: Number of state dimensions (required)
            du: Number of action dimensions (required)
        """
        if dx is None or du is None:
            raise ValueError("dx and du must be specified")
        
        self.degree = degree
        self.include_bias = include_bias
        self.dx = dx
        self.du = du
        
        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.feature_mask_ = None
        self.n_features_out_ = None
        self.n_output_features_ = None
        self.powers_ = None
    
    def fit(self, X, y=None):
        """Fit the polynomial features generator."""
        self.poly.fit(X)
        
        # Get powers for each feature
        powers = self.poly.powers_  # shape: (n_features, n_inputs)
        
        # Vectorized filtering: check if any control variable has degree >= 2
        # Control variables are indices [dx:dx+du]
        control_powers = powers[:, self.dx:self.dx+self.du]  # shape: (n_features, du)
        
        # Create mask: keep features where no control variable has degree >= 2
        self.feature_mask_ = ~np.any(control_powers >= 2, axis=1)
        
        self.n_features_out_ = np.sum(self.feature_mask_)
        self.n_output_features_ = self.n_features_out_
        # Filter powers to match filtered features
        self.powers_ = self.poly.powers_[self.feature_mask_]
        
        return self
    
    def transform(self, X):
        """Transform data to polynomial features, excluding u^2 terms."""
        if self.feature_mask_ is None:
            raise ValueError("Must fit before transform")
        
        X_poly = self.poly.transform(X)
        return X_poly[:, self.feature_mask_]
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """Get feature names for the filtered features."""
        if self.feature_mask_ is None:
            raise ValueError("Must fit before getting feature names")
        
        all_names = self.poly.get_feature_names_out(input_features)
        return all_names[self.feature_mask_]
    
    def get_feature_names(self, input_features=None):
        """Alias for get_feature_names_out for sklearn compatibility."""
        return self.get_feature_names_out(input_features)


class PolynomialFeatureScaler:
    """
    Simple polynomial feature scaler that generates polynomial features excluding u^2 terms.
    
    This is a compatibility wrapper that matches the interface used in the codebase.
    It generates polynomial features without any scaling.
    """
    
    def __init__(self, degree=2, scaling_method='none', dx=None, du=None, exclude_u_squared=False):
        """
        Initialize the polynomial feature scaler.
        
        Args:
            degree: Polynomial degree
            scaling_method: Ignored (no scaling is applied)
            dx: Number of state dimensions (required)
            du: Number of action dimensions (required)
            exclude_u_squared: Whether to exclude u^2 terms (default: True, always excludes)
        """
        if dx is None or du is None:
            raise ValueError("dx and du must be specified")
        
        self.degree = degree
        self.dx = dx
        self.du = du
        
        # Always use filtered polynomial features (excludes u^2)
        self.poly = FilteredPolynomialFeatures(
            degree=degree,
            include_bias=False,
            dx=dx,
            du=du
        )
        self.feature_names_ = None
        self.is_fitted_ = False
    
    def fit_transform(self, X):
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are actions
            
        Returns:
            Polynomial features (no scaling applied)
        """
        X_poly = self.poly.fit_transform(X)
        
        # Get feature names
        if hasattr(self.poly, 'get_feature_names_out'):
            self.feature_names_ = self.poly.get_feature_names_out()
        elif hasattr(self.poly, 'get_feature_names'):
            self.feature_names_ = self.poly.get_feature_names()
        else:
            self.feature_names_ = None
        
        self.is_fitted_ = True
        return X_poly
    
    def transform(self, X):
        """
        Transform new data using the fitted scaler.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are actions
            
        Returns:
            Polynomial features (no scaling applied)
        """
        if not self.is_fitted_:
            raise ValueError("Scaler must be fitted before transform")
        
        return self.poly.transform(X)
    
    def get_feature_names(self):
        """Get the names of the features."""
        if self.feature_names_ is None:
            return None
        return self.feature_names_


class StateOnlyPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Polynomial features generator that applies polynomial features only to state variables,
    leaving control variables unchanged.
    
    This class:
    - Takes input [x, u] where x has dx dimensions and u has du dimensions
    - Generates polynomial features of specified degree from x only
    - Concatenates the polynomial features with u unchanged
    - Result: [poly_features(x, degree=d), u]
    """
    
    def __init__(self, degree=2, include_bias=False, dx=None, du=None):
        """
        Initialize the state-only polynomial features generator.
        
        Args:
            degree: Polynomial degree for state features
            include_bias: Whether to include bias term in polynomial features
            dx: Number of state dimensions (required)
            du: Number of control dimensions (required)
        """
        if dx is None or du is None:
            raise ValueError("dx and du must be specified")
        
        self.degree = degree
        self.include_bias = include_bias
        self.dx = dx
        self.du = du
        
        # Polynomial feature generator for states only
        self.poly_x = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.n_features_out_ = None
        self.n_output_features_ = None
        self.powers_ = None
    
    def fit(self, X, y=None):
        """
        Fit the polynomial features generator.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are controls
        """
        # Extract state variables only
        X_state = X[:, :self.dx]
        
        # Fit polynomial features on states
        self.poly_x.fit(X_state)
        
        # Compute output dimensions
        n_poly_features = self.poly_x.n_output_features_
        self.n_features_out_ = n_poly_features + self.du
        self.n_output_features_ = self.n_features_out_
        
        # Store powers for state features (for compatibility/analysis)
        # Powers will be shape (n_poly_features, dx)
        self.powers_ = self.poly_x.powers_
        
        return self
    
    def transform(self, X):
        """
        Transform data to polynomial features.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are controls
        
        Returns:
            Transformed data of shape (n_samples, n_poly_features + du) where:
            - First n_poly_features columns are polynomial features of x
            - Last du columns are the original u values
        """
        if self.powers_ is None:
            raise ValueError("Must fit before transform")
        
        # Extract state and control variables
        X_state = X[:, :self.dx]
        X_control = X[:, self.dx:self.dx+self.du]
        
        # Transform states to polynomial features
        X_poly = self.poly_x.transform(X_state)
        
        # Concatenate polynomial features with unchanged controls
        return np.concatenate([X_poly, X_control], axis=1)
    
    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        """
        Get feature names for the output features.
        
        Args:
            input_features: Optional list of input feature names [x1, x2, ..., x_dx, u1, u2, ..., u_du]
        
        Returns:
            List of output feature names
        """
        if self.powers_ is None:
            raise ValueError("Must fit before getting feature names")
        
        # Get polynomial feature names for states
        if input_features is not None:
            state_features = input_features[:self.dx]
            control_features = input_features[self.dx:self.dx+self.du]
        else:
            state_features = [f'x{i}' for i in range(1, self.dx+1)]
            control_features = [f'u{i}' for i in range(1, self.du+1)]
        
        poly_names = self.poly_x.get_feature_names_out(state_features)
        
        # Concatenate with control feature names
        return list(poly_names) + list(control_features)
    
    def get_feature_names(self, input_features=None):
        """Alias for get_feature_names_out for sklearn compatibility."""
        return self.get_feature_names_out(input_features)
