#!/usr/bin/env python3
"""
Utility functions for statistical analysis.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress
import logging
import traceback # For detailed error logging if needed

class StatsAnalyzer:
    """Class for statistical analysis utilities."""

    @staticmethod
    def calculate_rolling_mean(data_series, window_size=5, center=True, min_periods=1):
        """
        Calculates the rolling mean of a time series.

        Args:
            data_series (xr.DataArray or pd.Series or np.ndarray): The time series.
            window_size (int): The size of the rolling window.
            center (bool): If True, the window is centered.
            min_periods (int): Minimum number of observations in window required to have a value.

        Returns:
            Same type as input: The smoothed time series.
        """
        if data_series is None:
            return None
        
        try:
            if isinstance(data_series, xr.DataArray):
                # Assuming the time dimension is named 'time' or 'season_year'
                time_dim = next((d for d in ['time', 'season_year'] if d in data_series.dims), None)
                if not time_dim:
                    logging.error("calculate_rolling_mean: Could not find a time dimension ('time' or 'season_year') in xarray.DataArray.")
                    return data_series # Return original if no time dim
                
                smoothed_series = data_series.rolling({time_dim: window_size}, center=center, min_periods=min_periods).mean()
                if hasattr(data_series, 'attrs'):
                    smoothed_series.attrs = data_series.attrs.copy()
                    smoothed_series.attrs['smoothing'] = f'{window_size}-unit rolling mean'
                return smoothed_series
            elif isinstance(data_series, pd.Series):
                return data_series.rolling(window=window_size, center=center, min_periods=min_periods).mean()
            elif isinstance(data_series, np.ndarray):
                if data_series.ndim != 1:
                    logging.error("calculate_rolling_mean: NumPy array must be 1D for rolling mean.")
                    return data_series # Or raise error
                # Simple numpy rolling mean (less robust than pandas for NaNs etc.)
                # For a more robust numpy version, one might use convolve or a loop similar to your original.
                # Using pandas for simplicity and robustness here, even for numpy array input.
                s = pd.Series(data_series)
                return s.rolling(window=window_size, center=center, min_periods=min_periods).mean().to_numpy()
            else:
                logging.warning(f"calculate_rolling_mean: Unsupported data type {type(data_series)}.")
                return data_series
        except Exception as e:
            logging.error(f"Error in calculate_rolling_mean: {e}")
            return data_series # Return original on error

    @staticmethod
    def calculate_regression(x_values, y_values):
        """
        Calculate linear regression between two 1D arrays.
        Handles NaN values by removing corresponding pairs.
        Ensures inputs are 1D, have matching lengths, sufficient valid points, and variance.

        Args:
            x_values (array-like): Independent variable values.
            y_values (array-like): Dependent variable values.

        Returns:
            tuple: (slope, intercept, r_value, p_value, stderr)
                   Returns (np.nan, np.nan, np.nan, np.nan, np.nan) if regression cannot be performed.
        """
        try:
            x_np = np.asarray(x_values).squeeze() # Remove single-dimensional entries
            y_np = np.asarray(y_values).squeeze()

            # Ensure inputs are 1D
            if x_np.ndim == 0: x_np = np.array([x_np]) # Convert scalar to 1D array
            if y_np.ndim == 0: y_np = np.array([y_np])
            
            if x_np.ndim != 1 or y_np.ndim != 1:
                logging.warning(f"calculate_regression: Inputs are not 1D after squeeze. x_shape: {x_np.shape}, y_shape: {y_np.shape}")
                return np.nan, np.nan, np.nan, np.nan, np.nan

            if len(x_np) != len(y_np):
                logging.warning(f"calculate_regression: Input arrays have different lengths. x_len: {len(x_np)}, y_len: {len(y_np)}")
                return np.nan, np.nan, np.nan, np.nan, np.nan

            # Create a mask for finite values (non-NaN, non-infinite)
            valid_mask = np.isfinite(x_np) & np.isfinite(y_np)
            n_valid_points = np.sum(valid_mask)

            if n_valid_points < 3: # Need at least 3 points for meaningful regression
                # logging.debug(f"calculate_regression: Not enough valid data points ({n_valid_points}) for regression.")
                return np.nan, np.nan, np.nan, np.nan, np.nan

            x_masked = x_np[valid_mask]
            y_masked = y_np[valid_mask]

            # Check for variance in masked data
            if np.var(x_masked) < 1e-10 or np.var(y_masked) < 1e-10:
                # Handle cases with zero variance (e.g., all x values are the same)
                # logging.debug("calculate_regression: Zero variance in x or y after masking.")
                # Slope is 0 if x has variance but y doesn't (horizontal line)
                # Slope is undefined (NaN) if x has no variance
                slope = 0.0 if np.var(x_masked) >= 1e-10 else np.nan
                intercept = np.mean(y_masked) if np.var(x_masked) >= 1e-10 else (y_masked[0] if n_valid_points > 0 else np.nan)
                r_value = 0.0 # Or NaN, depending on definition for zero variance
                p_value = np.nan # P-value is tricky here
                stderr = np.nan
                return slope, intercept, r_value, p_value, stderr

            # Perform linear regression
            slope, intercept, r_value, p_value, stderr = linregress(x_masked, y_masked)
            
            # Check for NaN results from linregress (can happen in edge cases)
            if any(np.isnan(val) for val in [slope, intercept, r_value, p_value, stderr]):
                # logging.warning("calculate_regression: linregress returned NaN values.")
                return np.nan, np.nan, np.nan, np.nan, np.nan
                
            return slope, intercept, r_value, p_value, stderr

        except ValueError as ve: # Handles errors from linregress itself
            logging.warning(f"calculate_regression: ValueError during linregress: {ve}")
            return np.nan, np.nan, np.nan, np.nan, np.nan
        except Exception as e:
            logging.error(f"calculate_regression: An unexpected error occurred: {e}")
            logging.error(traceback.format_exc())
            return np.nan, np.nan, np.nan, np.nan, np.nan

    @staticmethod
    def normalize_series(data_series):
        """
        Normalize a series (subtract mean, divide by standard deviation).
        Handles xr.DataArray, pd.Series, or np.ndarray.
        """
        if data_series is None:
            return None

        try:
            if isinstance(data_series, xr.DataArray):
                valid_values = data_series.where(np.isfinite(data_series))
                mean = valid_values.mean(skipna=True)
                std = valid_values.std(skipna=True)
                if std == 0 or np.isnan(std): # Avoid division by zero or NaN std
                    return data_series - mean if mean.notnull() else data_series
                normalized = (data_series - mean) / std
                if hasattr(data_series, 'attrs'):
                    normalized.attrs = data_series.attrs.copy()
                    normalized.attrs['normalization'] = 'z-score'
                return normalized
            else: # Convert pandas Series or numpy array for consistent processing
                if isinstance(data_series, pd.Series):
                    values = data_series.to_numpy()
                elif isinstance(data_series, np.ndarray):
                    values = data_series.copy()
                else:
                    logging.warning(f"normalize_series: Unsupported data type {type(data_series)}.")
                    return data_series

                valid_mask = np.isfinite(values)
                if not np.any(valid_mask): # All NaN or empty
                    return data_series
                
                mean_val = np.mean(values[valid_mask])
                std_val = np.std(values[valid_mask])

                if std_val == 0 or np.isnan(std_val):
                    return values - mean_val # Return centered values if std is zero/NaN
                
                normalized_values = (values - mean_val) / std_val
                
                if isinstance(data_series, pd.Series):
                    return pd.Series(normalized_values, index=data_series.index, name=data_series.name)
                return normalized_values # For numpy array input

        except Exception as e:
            logging.error(f"Error in normalize_series: {e}")
            return data_series # Return original on error