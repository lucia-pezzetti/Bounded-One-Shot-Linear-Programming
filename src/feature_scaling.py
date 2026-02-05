"""
Feature scaling utilities for polynomial features.

This module provides robust scaling methods to handle the extreme
scaling differences in higher-degree polynomial features.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

class FilteredPolynomialFeatures(BaseEstimator, TransformerMixin):
    """
    Polynomial features generator that can exclude specific terms.
    
    This class wraps sklearn's PolynomialFeatures and filters out terms
    where control variables have degree >= 2 (i.e., u^2 terms).
    """
    
    def __init__(self, degree=2, include_bias=False, dx=None, du=None, exclude_u_squared=True):
        """
        Initialize the filtered polynomial features generator.
        
        Args:
            degree: Polynomial degree
            include_bias: Whether to include bias term
            dx: Number of state dimensions (required if exclude_u_squared=True)
            du: Number of action dimensions (required if exclude_u_squared=True)
            exclude_u_squared: Whether to exclude terms where any control variable has degree >= 2
        """
        self.degree = degree
        self.include_bias = include_bias
        self.dx = dx
        self.du = du
        self.exclude_u_squared = exclude_u_squared
        
        if exclude_u_squared and (dx is None or du is None):
            raise ValueError("dx and du must be specified when exclude_u_squared=True")
        
        self.poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        self.feature_mask_ = None
        self.n_features_out_ = None
        self.n_output_features_ = None
        self.powers_ = None
        
    def fit(self, X, y=None):
        """Fit the polynomial features generator."""
        self.poly.fit(X)
        
        if self.exclude_u_squared:
            # Get powers for each feature
            powers = self.poly.powers_  # shape: (n_features, n_inputs)
            
            # Vectorized filtering: check if any control variable has degree >= 2
            # Control variables are indices [dx:dx+du]
            control_powers = powers[:, self.dx:self.dx+self.du]  # shape: (n_features, du)
            
            # Create mask: keep features where no control variable has degree >= 2
            # Using vectorized operation: check if any control power >= 2 for each feature
            self.feature_mask_ = ~np.any(control_powers >= 2, axis=1)
            
            self.n_features_out_ = np.sum(self.feature_mask_)
            # Filter powers to match filtered features
            self.powers_ = self.poly.powers_[self.feature_mask_]
        else:
            # Keep all features
            self.feature_mask_ = np.ones(len(self.poly.powers_), dtype=bool)
            self.n_features_out_ = len(self.poly.powers_)
            # Use original powers
            self.powers_ = self.poly.powers_
        
        # Add sklearn-compatible attribute name
        self.n_output_features_ = self.n_features_out_
        
        return self
    
    def transform(self, X):
        """Transform data to polynomial features, excluding filtered terms."""
        X_poly = self.poly.transform(X)
        
        if self.feature_mask_ is None:
            raise ValueError("Must fit before transform")
        
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
    Robust feature scaler specifically designed for polynomial features.
    
    This scaler handles the extreme scaling differences that occur
    with higher-degree polynomial features by applying separate
    scaling strategies to states and actions for better numerical stability.
    """
    
    def __init__(self, degree=2, scaling_method='robust', dx=None, du=None, exclude_u_squared=False):
        """
        Initialize the polynomial feature scaler.
        
        Args:
            degree: Polynomial degree
            scaling_method: 'robust', 'standard', or 'minmax'
            dx: Number of state dimensions (if None, assumes single scaler)
            du: Number of action dimensions (if None, assumes single scaler)
            exclude_u_squared: Whether to exclude u^2 terms from polynomial features
        """
        self.degree = degree
        self.scaling_method = scaling_method
        self.dx = dx
        self.du = du
        self.exclude_u_squared = exclude_u_squared
        self.use_separate_scaling = (scaling_method != 'none' and dx is not None and du is not None)
        
        # Initialize scalers
        if scaling_method == 'none':
            self.scaler = None
            self.scaler_x = None
            self.scaler_u = None
        elif self.use_separate_scaling:
            # Use separate scalers for states and actions
            if scaling_method == 'robust':
                self.scaler_x = RobustScaler()
                self.scaler_u = RobustScaler()
            elif scaling_method == 'standard':
                # For states: use scaling without centering to preserve zero equilibrium
                # For actions: can use centering since control offset doesn't affect equilibrium
                self.scaler_x = StandardScaler(with_mean=False)  # Don't center states
                self.scaler_u = StandardScaler()  # Can center actions
            elif scaling_method == 'minmax':
                self.scaler_x = MinMaxScaler()
                self.scaler_u = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
            self.scaler = None  # Not used when separate scaling
        else:
            # Use single scaler for all features
            if scaling_method == 'robust':
                self.scaler = RobustScaler()
            elif scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown scaling method: {scaling_method}")
            self.scaler_x = None
            self.scaler_u = None
        
        # Use filtered polynomial features if exclude_u_squared is True
        if self.exclude_u_squared:
            if dx is None or du is None:
                raise ValueError("dx and du must be specified when exclude_u_squared=True")
            self.poly = FilteredPolynomialFeatures(
                degree=degree, 
                include_bias=False, 
                dx=dx, 
                du=du, 
                exclude_u_squared=True
            )
        else:
            self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def fit_transform(self, X):
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Input data of shape (n_samples, n_features) where first dx columns are states,
               last du columns are actions (if using separate scaling)
            
        Returns:
            Scaled polynomial features
        """
        if self.use_separate_scaling:
            # Separate states and actions for scaling
            X_states = X[:, :self.dx]
            X_actions = X[:, self.dx:]
            
            # Scale states and actions separately
            if self.scaler_x is not None:
                X_states_scaled = self.scaler_x.fit_transform(X_states)
            else:
                X_states_scaled = X_states
                
            if self.scaler_u is not None:
                X_actions_scaled = self.scaler_u.fit_transform(X_actions)
            else:
                X_actions_scaled = X_actions
            
            # Recombine scaled states and actions
            X_scaled = np.concatenate([X_states_scaled, X_actions_scaled], axis=1)
        else:
            # Use original data for single scaler approach
            X_scaled = X
        
        # Generate polynomial features from scaled data
        X_poly = self.poly.fit_transform(X_scaled)
        # Get feature names (handle both PolynomialFeatures and FilteredPolynomialFeatures)
        if hasattr(self.poly, 'get_feature_names_out'):
            self.feature_names_ = self.poly.get_feature_names_out()
        elif hasattr(self.poly, 'get_feature_names'):
            self.feature_names_ = self.poly.get_feature_names()
        else:
            self.feature_names_ = None
        
        # Apply final scaling to polynomial features if using single scaler
        if not self.use_separate_scaling and self.scaler is not None:
            X_poly_scaled = self.scaler.fit_transform(X_poly)
        else:
            X_poly_scaled = X_poly
        
        self.is_fitted_ = True
        return X_poly_scaled
    
    def transform(self, X):
        """
        Transform new data using the fitted scaler.
        
        Args:
            X: Input data of shape (n_samples, n_features) where first dx columns are states,
               last du columns are actions (if using separate scaling)
            
        Returns:
            Scaled polynomial features
        """
        if not self.is_fitted_:
            raise ValueError("Scaler must be fitted before transform")
        
        if self.use_separate_scaling:
            # Separate states and actions for scaling
            X_states = X[:, :self.dx]
            X_actions = X[:, self.dx:]
            
            # Scale states and actions separately
            if self.scaler_x is not None:
                X_states_scaled = self.scaler_x.transform(X_states)
            else:
                X_states_scaled = X_states
                
            if self.scaler_u is not None:
                X_actions_scaled = self.scaler_u.transform(X_actions)
            else:
                X_actions_scaled = X_actions
            
            # Recombine scaled states and actions
            X_scaled = np.concatenate([X_states_scaled, X_actions_scaled], axis=1)
        else:
            # Use original data for single scaler approach
            X_scaled = X
        
        # Generate polynomial features from scaled data
        X_poly = self.poly.transform(X_scaled)
        
        # Apply final scaling to polynomial features if using single scaler
        if not self.use_separate_scaling and self.scaler is not None:
            X_poly_scaled = self.scaler.transform(X_poly)
        else:
            X_poly_scaled = X_poly
        
        return X_poly_scaled
    
    def get_feature_names(self):
        """Get the names of the selected features."""
        if self.feature_names_ is None:
            return None
        
        return self.feature_names_
    
    def get_scaling_info(self):
        """Get information about the scaling applied."""
        if not self.is_fitted_:
            return None
        
        info = {
            'scaling_method': self.scaling_method,
            'use_separate_scaling': self.use_separate_scaling,
            'n_features_original': len(self.feature_names_) if self.feature_names_ is not None else 0,
        }
        
        if self.use_separate_scaling:
            # Information for separate state/action scalers
            info['scaler_params'] = {
                'state_scaler': {
                    'scale_': self.scaler_x.scale_ if hasattr(self.scaler_x, 'scale_') else None,
                    'mean_': self.scaler_x.mean_ if hasattr(self.scaler_x, 'mean_') else None,
                    'center_': self.scaler_x.center_ if hasattr(self.scaler_x, 'center_') else None,
                } if self.scaler_x is not None else None,
                'action_scaler': {
                    'scale_': self.scaler_u.scale_ if hasattr(self.scaler_u, 'scale_') else None,
                    'mean_': self.scaler_u.mean_ if hasattr(self.scaler_u, 'mean_') else None,
                    'center_': self.scaler_u.center_ if hasattr(self.scaler_u, 'center_') else None,
                } if self.scaler_u is not None else None,
            }
        else:
            # Information for single scaler
            info['scaler_params'] = {
                'scale_': self.scaler.scale_ if hasattr(self.scaler, 'scale_') else None,
                'mean_': self.scaler.mean_ if hasattr(self.scaler, 'mean_') else None,
                'center_': self.scaler.center_ if hasattr(self.scaler, 'center_') else None,
            }
        
        return info
    
    def get_scaling_factors(self):
        """
        Get the scaling factors for states and actions separately.
        
        Returns:
            Dictionary with 'state_scale' and 'action_scale' if using separate scaling,
            or None if using single scaler
        """
        if not self.is_fitted_:
            return None
            
        if self.use_separate_scaling:
            return {
                'state_scale': self.scaler_x.scale_ if self.scaler_x is not None else None,
                'action_scale': self.scaler_u.scale_ if self.scaler_u is not None else None,
            }
        else:
            return None


class PolynomialStateFeatureScaler:
    """
    Feature scaler that generates polynomial features (degree 2) from states only,
    then concatenates them with unchanged inputs.
    
    This scaler:
    1. Takes state-input pairs [x, u] where x has dx dimensions and u has du dimensions
    2. Generates polynomial features of degree 2 from x only
    3. Concatenates the polynomial features with u unchanged
    4. Result: [poly_features(x, degree=2), u]
    """
    
    def __init__(self, degree=2, scaling_method='robust', dx=None, du=None):
        """
        Initialize the polynomial state feature scaler.
        
        Args:
            degree: Polynomial degree for state features (default: 2)
            scaling_method: 'robust', 'standard', 'minmax', or 'none'
            dx: Number of state dimensions (required)
            du: Number of input dimensions (required)
        """
        if dx is None or du is None:
            raise ValueError("dx and du must be specified")
        
        self.degree = degree
        self.scaling_method = scaling_method
        self.dx = dx
        self.du = du
        
        # Initialize scaler for states only
        if scaling_method == 'none':
            self.scaler_x = None
        elif scaling_method == 'robust':
            self.scaler_x = RobustScaler()
        elif scaling_method == 'standard':
            # Use scaling without centering to preserve zero equilibrium
            self.scaler_x = StandardScaler(with_mean=False)
        elif scaling_method == 'minmax':
            self.scaler_x = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")
        
        # Polynomial feature generator for states only
        self.poly_x = PolynomialFeatures(degree=degree, include_bias=False)
        self.feature_names_ = None
        self.is_fitted_ = False
        
    def fit_transform(self, X):
        """
        Fit the scaler and transform the data.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are inputs
            
        Returns:
            Concatenated features: [polynomial_features(states, degree=2), inputs]
        """
        if X.shape[1] != self.dx + self.du:
            raise ValueError(f"Expected {self.dx + self.du} features, got {X.shape[1]}")
        
        # Separate states and inputs
        X_states = X[:, :self.dx]
        X_inputs = X[:, self.dx:]
        
        # Scale states if scaler is provided
        if self.scaler_x is not None:
            X_states_scaled = self.scaler_x.fit_transform(X_states)
        else:
            X_states_scaled = X_states
        
        # Generate polynomial features from states only
        X_poly_states = self.poly_x.fit_transform(X_states_scaled)
        
        # Get feature names for polynomial state features
        if hasattr(self.poly_x, 'get_feature_names_out'):
            poly_state_names = self.poly_x.get_feature_names_out()
        elif hasattr(self.poly_x, 'get_feature_names'):
            poly_state_names = self.poly_x.get_feature_names()
        else:
            poly_state_names = None
        
        # Create feature names: polynomial state features + input features
        if poly_state_names is not None:
            input_names = [f'u{i}' for i in range(self.du)]
            self.feature_names_ = list(poly_state_names) + input_names
        else:
            self.feature_names_ = None
        
        # Concatenate polynomial state features with unchanged inputs
        X_output = np.concatenate([X_poly_states, X_inputs], axis=1)
        
        self.is_fitted_ = True
        return X_output
    
    def transform(self, X):
        """
        Transform new data using the fitted scaler.
        
        Args:
            X: Input data of shape (n_samples, dx+du) where first dx columns are states,
               last du columns are inputs
            
        Returns:
            Concatenated features: [polynomial_features(states, degree=2), inputs]
        """
        if not self.is_fitted_:
            raise ValueError("Scaler must be fitted before transform")
        
        if X.shape[1] != self.dx + self.du:
            raise ValueError(f"Expected {self.dx + self.du} features, got {X.shape[1]}")
        
        # Separate states and inputs
        X_states = X[:, :self.dx]
        X_inputs = X[:, self.dx:]
        
        # Scale states if scaler is provided
        if self.scaler_x is not None:
            X_states_scaled = self.scaler_x.transform(X_states)
        else:
            X_states_scaled = X_states
        
        # Generate polynomial features from states only
        X_poly_states = self.poly_x.transform(X_states_scaled)
        
        # Concatenate polynomial state features with unchanged inputs
        X_output = np.concatenate([X_poly_states, X_inputs], axis=1)
        
        return X_output
    
    def get_feature_names(self):
        """Get the names of the features."""
        if self.feature_names_ is None:
            return None
        return self.feature_names_
    
    def get_scaling_info(self):
        """Get information about the scaling applied."""
        if not self.is_fitted_:
            return None
        
        info = {
            'scaling_method': self.scaling_method,
            'degree': self.degree,
            'dx': self.dx,
            'du': self.du,
            'n_poly_features': self.poly_x.n_output_features_ if hasattr(self.poly_x, 'n_output_features_') else None,
            'n_total_features': self.poly_x.n_output_features_ + self.du if hasattr(self.poly_x, 'n_output_features_') else None,
        }
        
        if self.scaler_x is not None:
            info['scaler_params'] = {
                'state_scaler': {
                    'scale_': self.scaler_x.scale_ if hasattr(self.scaler_x, 'scale_') else None,
                    'mean_': self.scaler_x.mean_ if hasattr(self.scaler_x, 'mean_') else None,
                    'center_': self.scaler_x.center_ if hasattr(self.scaler_x, 'center_') else None,
                }
            }
        else:
            info['scaler_params'] = None
        
        return info
    
    def get_scaling_factors(self):
        """
        Get the scaling factors for states.
        
        Returns:
            Dictionary with 'state_scale' or None if no scaling
        """
        if not self.is_fitted_:
            return None
        
        if self.scaler_x is not None:
            return {
                'state_scale': self.scaler_x.scale_ if hasattr(self.scaler_x, 'scale_') else None,
            }
        else:
            return None
            