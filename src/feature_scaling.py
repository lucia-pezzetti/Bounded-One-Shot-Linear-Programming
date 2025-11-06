"""
Feature scaling utilities for polynomial features.

This module provides robust scaling methods to handle the extreme
scaling differences in higher-degree polynomial features.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
import warnings

class PolynomialFeatureScaler:
    """
    Robust feature scaler specifically designed for polynomial features.
    
    This scaler handles the extreme scaling differences that occur
    with higher-degree polynomial features by applying separate
    scaling strategies to states and actions for better numerical stability.
    """
    
    def __init__(self, degree=2, scaling_method='robust', dx=None, du=None):
        """
        Initialize the polynomial feature scaler.
        
        Args:
            degree: Polynomial degree
            scaling_method: 'robust', 'standard', or 'minmax'
            dx: Number of state dimensions (if None, assumes single scaler)
            du: Number of action dimensions (if None, assumes single scaler)
        """
        self.degree = degree
        self.scaling_method = scaling_method
        self.dx = dx
        self.du = du
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
        self.feature_names_ = self.poly.get_feature_names_out()
        
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
            