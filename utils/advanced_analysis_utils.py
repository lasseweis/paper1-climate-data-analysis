#!/usr/bin/env python3
"""
Utility functions for advanced climate analysis, including jet index calculations,
regression analysis, and correlation studies.
"""
import logging
import numpy as np
import xarray as xr
import pandas as pd # For correlation matrix, if used there
from scipy.stats import linregress, pearsonr, spearmanr # Behalten, falls direkt von scipy.stats genutzt

# Absolute Importe für andere Utility-Module (da 'utils' im sys.path ist)
from config_setup import Config
from data_utils import DataProcessor # Korrekter Klassenname ist DataProcessor
from stats_utils import StatsAnalyzer 
from jet_utils import JetStreamAnalyzer # Korrekter Klassenname ist JetStreamAnalyzer

class AdvancedAnalyzer:
    """
    Provides methods for advanced analysis of climate data, including
    jet stream characteristics, regression maps, and correlation analyses.
    """

    @staticmethod
    def calculate_regression_maps(reanalysis_data_dict_of_dicts: dict, jet_indices_data_all: dict, dataset_id_str: str):
        """
        Calculates regression maps of U850 anomalies onto box-averaged climate
        variable anomalies (PR, TAS) for specified seasons. This is for creating
        spatial maps of regression slopes.

        Args:
            reanalysis_data_dict_of_dicts (dict): A dictionary where keys are dataset IDs (e.g., "ERA5")
                                                 and values are dictionaries of loaded xarray Datasets/DataArrays
                                                 (e.g., {'ua850': ua_data, 'pr': pr_data, 'tas': tas_data}).
            jet_indices_data_all (dict): A dictionary containing all jet indices, keyed by strings
                                        like 'ERA5_jet_speed_winter_raw'. Not directly used for box index
                                        regression maps but kept for consistency with other calls if needed.
            dataset_id_str (str): The specific dataset ID (e.g., "ERA5") from reanalysis_data_dict_of_dicts
                                  to process.

        Returns:
            dict: A dictionary structured by season, then by variable type,
                  containing slopes, p-values, and coordinate meshes.
                  Example: {'Winter': {'slopes_pr': ..., 'p_values_pr': ..., 'lons': ..., 'lats': ...}, ...}
        """
        logging.info(f"Calculating regression maps for {dataset_id_str} (U850 vs Box Climate Indices)...")
        
        reanalysis_data_specific_dataset = reanalysis_data_dict_of_dicts.get(dataset_id_str)
        if not reanalysis_data_specific_dataset:
            logging.error(f"Data for dataset '{dataset_id_str}' not found in reanalysis_data_dict_of_dicts. Cannot calculate regression maps.")
            return {}

        ua850_full_field = reanalysis_data_specific_dataset.get('ua850')
        pr_full_field = reanalysis_data_specific_dataset.get('pr')
        tas_full_field = reanalysis_data_specific_dataset.get('tas')

        if ua850_full_field is None:
            logging.error(f"U850 data not found for {dataset_id_str}. Cannot calculate regression maps.")
            return {}

        seasonal_results = {}
        seasons_to_process = ['Winter', 'Summer'] 
        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        detrend_active = Config.DETREND_REANALYSIS_DATA_FOR_REGRESSION_MAPS 

        logging.debug(f"Regression maps for {dataset_id_str}: Detrending set to {detrend_active}.")
        logging.debug(f"Regression maps for {dataset_id_str}: Using box {box_coords} for climate indices.")

        for season in seasons_to_process:
            logging.info(f"  Processing regression maps for {season} - {dataset_id_str}")
            seasonal_results[season] = {}
            
            try:
                ua850_season_field = DataProcessor.select_season(ua850_full_field, season, use_datetime_season=Config.USE_DATETIME_SEASON_FOR_REANALYSIS) # DataProcessor verwenden
                if ua850_season_field is None or ua850_season_field.size == 0:
                    logging.warning(f"U850 field data for {season} ({dataset_id_str}) is empty or None after season selection. Skipping.")
                    continue
            except Exception as e:
                logging.error(f"Error selecting season {season} for U850 field ({dataset_id_str}): {e}", exc_info=True)
                continue
            
            ua850_seasonal_mean_for_contours = ua850_season_field.mean(dim='time', skipna=True)
            seasonal_results[season]['ua850_mean_for_contours'] = ua850_seasonal_mean_for_contours

            for var_type, var_full_field_data, unit_label in [('pr', pr_full_field, '% change'), ('tas', tas_full_field, '°C change')]:
                logging.info(f"    Calculating U850 vs {var_type.upper()} Box Index for {season} - {dataset_id_str}")
                if var_full_field_data is None:
                    logging.warning(f"{var_type.upper()} data not found for {dataset_id_str}. Skipping this variable for {season} maps.")
                    seasonal_results[season][f'slopes_{var_type}'] = None
                    seasonal_results[season][f'p_values_{var_type}'] = None
                    continue
                
                try:
                    var_season_field = DataProcessor.select_season(var_full_field_data, season, use_datetime_season=Config.USE_DATETIME_SEASON_FOR_REANALYSIS) # DataProcessor verwenden
                    if var_season_field is None or var_season_field.size == 0:
                        logging.warning(f"{var_type.upper()} field data for {season} ({dataset_id_str}) is empty or None after season selection. Skipping.")
                        seasonal_results[season][f'slopes_{var_type}'] = None
                        seasonal_results[season][f'p_values_{var_type}'] = None
                        continue
                except Exception as e:
                    logging.error(f"Error selecting season {season} for {var_type.upper()} field ({dataset_id_str}): {e}", exc_info=True)
                    seasonal_results[season][f'slopes_{var_type}'] = None
                    seasonal_results[season][f'p_values_{var_type}'] = None
                    continue

                try:
                    box_mean_climate_index = DataProcessor.calculate_spatial_mean(var_season_field, box_coords) # DataProcessor verwenden
                    if box_mean_climate_index is None or box_mean_climate_index.size == 0:
                        logging.warning(f"Box mean index for {var_type.upper()} ({season}, {dataset_id_str}) is empty or None. Skipping map.")
                        seasonal_results[season][f'slopes_{var_type}'] = None
                        seasonal_results[season][f'p_values_{var_type}'] = None
                        continue
                except Exception as e:
                    logging.error(f"Error calculating box mean for {var_type.upper()} index ({season}, {dataset_id_str}): {e}", exc_info=True)
                    seasonal_results[season][f'slopes_{var_type}'] = None
                    seasonal_results[season][f'p_values_{var_type}'] = None
                    continue

                try:
                    ua850_field_aligned, box_index_aligned = xr.align(ua850_season_field, box_mean_climate_index, join='inner')
                    if ua850_field_aligned.size == 0 or box_index_aligned.size == 0:
                        logging.warning(f"Alignment failed or resulted in empty series for U850 field vs {var_type.upper()} box index ({season}, {dataset_id_str}). Skipping map.")
                        seasonal_results[season][f'slopes_{var_type}'] = None
                        seasonal_results[season][f'p_values_{var_type}'] = None
                        continue
                except Exception as e:
                    logging.error(f"Error aligning U850 field and {var_type.upper()} box index ({season}, {dataset_id_str}): {e}", exc_info=True)
                    seasonal_results[season][f'slopes_{var_type}'] = None
                    seasonal_results[season][f'p_values_{var_type}'] = None
                    continue
                
                predictand_series_for_regression = ua850_field_aligned
                predictor_series_for_regression = box_index_aligned

                if detrend_active:
                    try:
                        logging.debug(f"    Detrending aligned U850 field for {season}, {var_type.upper()} map...")
                        predictand_series_for_regression = StatsAnalyzer.detrend_data(ua850_field_aligned, dim='time')
                        logging.debug(f"    Detrending aligned {var_type.upper()} box index for {season} map...")
                        predictor_series_for_regression = StatsAnalyzer.detrend_data(box_index_aligned, dim='time')
                    except Exception as e:
                        logging.error(f"Error detrending data for regression map ({season}, {var_type.upper()}, {dataset_id_str}): {e}", exc_info=True)
                        seasonal_results[season][f'slopes_{var_type}'] = None
                        seasonal_results[season][f'p_values_{var_type}'] = None
                        continue
                
                if var_type == 'pr':
                    try:
                        clim_mean_pr_box = box_index_aligned.mean(dim='time', skipna=True) 
                        if clim_mean_pr_box == 0 or np.isnan(clim_mean_pr_box):
                            logging.warning(f"Climatological mean PR for box index ({season}, {dataset_id_str}) is zero or NaN. Cannot convert to % change. Skipping map.")
                            seasonal_results[season][f'slopes_{var_type}'] = None
                            seasonal_results[season][f'p_values_{var_type}'] = None
                            continue
                        
                        if detrend_active: 
                            predictor_series_for_regression = (predictor_series_for_regression / clim_mean_pr_box) * 100
                        else: 
                            predictor_series_for_regression = ((predictor_series_for_regression - clim_mean_pr_box) / clim_mean_pr_box) * 100
                        logging.info(f"    Converted PR box index to percentage change for {season} - {dataset_id_str}")
                    except Exception as e:
                        logging.error(f"Error converting PR box index to percentage change ({season}, {dataset_id_str}): {e}", exc_info=True)
                        seasonal_results[season][f'slopes_{var_type}'] = None
                        seasonal_results[season][f'p_values_{var_type}'] = None
                        continue

                try:
                    logging.info(f"    Performing grid regression: U850 field vs {var_type.upper()} box index ({unit_label}) for {season} - {dataset_id_str}")
                    slopes_grid, p_values_grid, _, _ = StatsAnalyzer.perform_linear_regression_on_grid(
                        predictand_series_for_regression, 
                        predictor_series_for_regression   
                    )
                    seasonal_results[season][f'slopes_{var_type}'] = slopes_grid
                    seasonal_results[season][f'p_values_{var_type}'] = p_values_grid
                    if 'longitude' in slopes_grid.coords and 'latitude' in slopes_grid.coords:
                         seasonal_results[season]['lons'] = slopes_grid.longitude.values
                         seasonal_results[season]['lats'] = slopes_grid.latitude.values
                    elif 'lon' in slopes_grid.coords and 'lat' in slopes_grid.coords: 
                         seasonal_results[season]['lons'] = slopes_grid.lon.values
                         seasonal_results[season]['lats'] = slopes_grid.lat.values
                    else: 
                        logging.warning(f"Could not determine lon/lat coordinate names from slopes_grid for {var_type} {season}. Plotting might fail.")
                        if len(slopes_grid.dims) >= 2: 
                            lon_dim_name = slopes_grid.dims[-1]
                            lat_dim_name = slopes_grid.dims[-2]
                            if hasattr(slopes_grid,lon_dim_name) and hasattr(slopes_grid,lat_dim_name):
                                seasonal_results[season]['lons'] = slopes_grid[lon_dim_name].values
                                seasonal_results[season]['lats'] = slopes_grid[lat_dim_name].values
                except Exception as e:
                    logging.error(f"Error during grid regression for U850 vs {var_type.upper()} box index ({season}, {dataset_id_str}): {e}", exc_info=True)
                    seasonal_results[season][f'slopes_{var_type}'] = None
                    seasonal_results[season][f'p_values_{var_type}'] = None
        
        logging.info(f"Finished calculating regression maps for {dataset_id_str}.")
        return seasonal_results

    @staticmethod
    def calculate_beta_obs_slopes_for_era5(reanalysis_data_dict: dict, dataset_id: str):
        """
        Calculates observational constraint beta_obs slopes (single values) from reanalysis data.
        This involves regressing U850 box anomalies onto box precipitation/temperature box anomalies.
        """
        logging.info(f"Calculating single value beta_obs_slopes (U850 Box vs Climate Box Indices) for {dataset_id}...")
        beta_obs_results = {}

        ua850_data = reanalysis_data_dict.get('ua850')
        pr_data = reanalysis_data_dict.get('pr')
        tas_data = reanalysis_data_dict.get('tas')

        if ua850_data is None:
            logging.error(f"U850 data not found in reanalysis_data_dict for {dataset_id} for beta_obs_slopes. Cannot calculate.")
            return beta_obs_results
        else:
            logging.debug(f"U850 data for beta_obs_slopes loaded for {dataset_id}. Shape: {ua850_data.dims if hasattr(ua850_data, 'dims') else 'N/A'}, Coords: {list(ua850_data.coords.keys()) if hasattr(ua850_data, 'coords') else 'N/A'}")
            if hasattr(ua850_data, 'time') and 'time' in ua850_data.coords:
                 logging.debug(f"U850 time range: {ua850_data.time.min().values} to {ua850_data.time.max().values}")

        regression_pairs = {
            'Winter': {'var_data': pr_data, 'var_name': 'PR', 'var_unit_for_slope': '% precip change'},
            'Summer': {'var_data': tas_data, 'var_name': 'TAS', 'var_unit_for_slope': 'temp change in C'}
        }

        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        logging.debug(f"Using analysis box coordinates for beta_obs_slopes: {box_coords}")
        detrend_data = Config.DETREND_REANALYSIS_DATA 
        logging.debug(f"Detrending for reanalysis data (for beta_obs_slopes) set to: {detrend_data}")

        for season, details in regression_pairs.items():
            climate_var_data_orig = details['var_data'] 
            var_name = details['var_name']
            logging.info(f"--- Processing beta_obs_slope for: {season} - {var_name} ---")

            if climate_var_data_orig is None:
                logging.warning(f"{var_name} data not found for {season} in {dataset_id} for beta_obs_slopes. Skipping U850 vs {var_name} regression.")
                continue
            else:
                logging.debug(f"{var_name} data for {season} loaded. Shape: {climate_var_data_orig.dims if hasattr(climate_var_data_orig, 'dims') else 'N/A'}, Coords: {list(climate_var_data_orig.coords.keys()) if hasattr(climate_var_data_orig, 'coords') else 'N/A'}")
                if hasattr(climate_var_data_orig, 'time') and 'time' in climate_var_data_orig.coords:
                    logging.debug(f"{var_name} time range: {climate_var_data_orig.time.min().values} to {climate_var_data_orig.time.max().values}")

            try:
                logging.debug(f"Selecting season {season} for U850 for beta_obs_slope...")
                ua850_season_field = DataProcessor.select_season(ua850_data, season, use_datetime_season=Config.USE_DATETIME_SEASON_FOR_REANALYSIS) # DataProcessor verwenden
                logging.debug(f"U850 {season} field data shape: {ua850_season_field.dims if hasattr(ua850_season_field, 'dims') else 'N/A'}, Num NaNs: {np.isnan(ua850_season_field.data).sum() if hasattr(ua850_season_field, 'data') and hasattr(ua850_season_field.data, 'sum') else 'N/A'}")
                if not hasattr(ua850_season_field, 'size') or ua850_season_field.size == 0:
                    logging.warning(f"U850 field data for {season} is empty or invalid after season selection. Skipping beta_obs_slope.")
                    continue

                logging.debug(f"Selecting season {season} for {var_name} for beta_obs_slope...")
                climate_var_season_field = DataProcessor.select_season(climate_var_data_orig, season, use_datetime_season=Config.USE_DATETIME_SEASON_FOR_REANALYSIS) # DataProcessor verwenden
                logging.debug(f"{var_name} {season} field data shape: {climate_var_season_field.dims if hasattr(climate_var_season_field, 'dims') else 'N/A'}, Num NaNs: {np.isnan(climate_var_season_field.data).sum() if hasattr(climate_var_season_field, 'data') and hasattr(climate_var_season_field.data, 'sum') else 'N/A'}")
                if not hasattr(climate_var_season_field, 'size') or climate_var_season_field.size == 0:
                    logging.warning(f"{var_name} field data for {season} is empty or invalid after season selection. Skipping beta_obs_slope.")
                    continue
            except Exception as e:
                logging.error(f"Error selecting season {season} for beta_obs_slope ({dataset_id}, var {var_name}): {e}", exc_info=True)
                continue

            try:
                logging.debug(f"Calculating box mean for U850 ({season}, {dataset_id}) for beta_obs_slope...")
                ua850_box_mean_season = DataProcessor.calculate_spatial_mean(ua850_season_field, box_coords) # DataProcessor verwenden
                if ua850_box_mean_season is None or (hasattr(ua850_box_mean_season, 'size') and ua850_box_mean_season.size == 0):
                    logging.warning(f"Box mean for U850 ({season}, {dataset_id}) for beta_obs_slope is empty or None. Skipping.")
                    continue
                logging.debug(f"Box mean U850 ({season}) shape: {ua850_box_mean_season.dims if hasattr(ua850_box_mean_season, 'dims') else 'N/A'}, Num NaNs: {np.isnan(ua850_box_mean_season.data).sum() if hasattr(ua850_box_mean_season, 'data') and hasattr(ua850_box_mean_season.data, 'sum') else 'N/A'}")

                logging.debug(f"Calculating box mean for {var_name} ({season}, {dataset_id}) for beta_obs_slope...")
                box_mean_climate_var = DataProcessor.calculate_spatial_mean(climate_var_season_field, box_coords) # DataProcessor verwenden
                if box_mean_climate_var is None or (hasattr(box_mean_climate_var, 'size') and box_mean_climate_var.size == 0):
                    logging.warning(f"Box mean for {var_name} ({season}, {dataset_id}) for beta_obs_slope is empty or None. Skipping.")
                    continue
                logging.debug(f"Box mean {var_name} ({season}) shape: {box_mean_climate_var.dims if hasattr(box_mean_climate_var, 'dims') else 'N/A'}, Num NaNs: {np.isnan(box_mean_climate_var.data).sum() if hasattr(box_mean_climate_var, 'data') and hasattr(box_mean_climate_var.data, 'sum') else 'N/A'}")

            except Exception as e:
                logging.error(f"Error calculating box mean for beta_obs_slope ({season}, {dataset_id}, var {var_name}): {e}", exc_info=True)
                continue
            
            try:
                logging.debug(f"Aligning time series for U850 box mean and {var_name} box mean ({season}, {dataset_id})...")
                ua850_box_aligned, box_mean_climate_var_aligned = xr.align(ua850_box_mean_season, box_mean_climate_var, join='inner')
                if ua850_box_aligned.size == 0 or box_mean_climate_var_aligned.size == 0:
                    logging.warning(f"Time alignment of box means resulted in empty series for U850 vs {var_name} ({season}, {dataset_id}). Skipping beta_obs_slope.")
                    continue
                logging.debug(f"Aligned U850 box mean ({season}) size: {ua850_box_aligned.size}, Aligned {var_name} box mean ({season}) size: {box_mean_climate_var_aligned.size}")
            except Exception as e:
                logging.error(f"Error aligning box mean time series for beta_obs_slope ({season}, {dataset_id}, var {var_name}): {e}", exc_info=True)
                continue
            
            box_mean_climate_var_aligned_for_clim_pr = box_mean_climate_var_aligned.copy(deep=True)

            if detrend_data:
                try:
                    logging.debug(f"Detrending U850 box aligned ({season})...")
                    ua850_box_aligned = StatsAnalyzer.detrend_data(ua850_box_aligned, dim='time')
                    logging.debug(f"Detrended U850 box aligned ({season}) Num NaNs: {np.isnan(ua850_box_aligned.data).sum() if hasattr(ua850_box_aligned, 'data') and hasattr(ua850_box_aligned.data, 'sum') else 'N/A'}")

                    logging.debug(f"Detrending {var_name} box aligned ({season})...")
                    box_mean_climate_var_aligned = StatsAnalyzer.detrend_data(box_mean_climate_var_aligned, dim='time')
                    logging.debug(f"Detrended {var_name} box aligned ({season}) Num NaNs: {np.isnan(box_mean_climate_var_aligned.data).sum() if hasattr(box_mean_climate_var_aligned, 'data') and hasattr(box_mean_climate_var_aligned.data, 'sum') else 'N/A'}")
                    
                    if (hasattr(ua850_box_aligned, 'data') and np.all(np.isnan(ua850_box_aligned.data))) or \
                       (hasattr(box_mean_climate_var_aligned, 'data') and np.all(np.isnan(box_mean_climate_var_aligned.data))):
                        logging.warning(f"Box-averaged data became all NaNs after detrending for {var_name} or U850 ({season}, {dataset_id}). Skipping beta_obs_slope.")
                        continue
                except Exception as e:
                    logging.error(f"Error detrending box-averaged data for beta_obs_slope ({season}, {dataset_id}, var {var_name}): {e}", exc_info=True)
                    continue
            
            predictor_series = None
            if var_name == 'PR':
                try:
                    logging.debug(f"Calculating PR percentage change for beta_obs_slope ({season}, {dataset_id})...")
                    clim_mean_pr = box_mean_climate_var_aligned_for_clim_pr.mean(dim='time', skipna=True)
                    logging.debug(f"  Climatological mean PR for beta_obs_slope ({season}): {clim_mean_pr.item() if hasattr(clim_mean_pr, 'item') and clim_mean_pr.size > 0 else 'N/A or Empty'}")

                    if not hasattr(clim_mean_pr, 'item') or clim_mean_pr.size == 0 or clim_mean_pr.item() == 0 or np.isnan(clim_mean_pr.item()):
                        logging.warning(f"Climatological mean PR for beta_obs_slope ({season}, {dataset_id}) is zero, NaN, or empty ({clim_mean_pr.item() if hasattr(clim_mean_pr, 'item') else 'N/A'}). Cannot calculate percentage change. Skipping.")
                        continue
                    
                    clim_mean_pr_val = clim_mean_pr.item()

                    if detrend_data: 
                        logging.debug(f"  PR data for beta_obs_slope was detrended. Predictor is (detrended_anomaly / clim_mean_pr) * 100.")
                        predictor_series = (box_mean_climate_var_aligned / clim_mean_pr_val) * 100
                    else: 
                        logging.debug(f"  PR data for beta_obs_slope NOT detrended. Predictor is ((value - clim_mean_pr) / clim_mean_pr) * 100.")
                        predictor_series = ((box_mean_climate_var_aligned - clim_mean_pr_val) / clim_mean_pr_val) * 100
                    
                    if hasattr(predictor_series, 'data') and np.any(np.isinf(predictor_series.data)):
                        num_inf = np.isinf(predictor_series.data).sum()
                        logging.warning(f"Predictor series (PR % change for beta_obs_slope) for {season} contains Inf values. Num Inf: {num_inf}. Converting Infs to NaNs.")
                        predictor_series = predictor_series.where(np.isfinite(predictor_series)) 
                except Exception as e:
                    logging.error(f"Error processing PR percentage change for beta_obs_slope ({season}, {dataset_id}): {e}", exc_info=True)
                    continue
            else: 
                predictor_series = box_mean_climate_var_aligned 
                logging.debug(f"  Predictor series (TAS for beta_obs_slope) for {season} - Num NaNs: {np.isnan(predictor_series.data).sum() if hasattr(predictor_series, 'data') and hasattr(predictor_series.data, 'sum') else 'N/A'}")

            if predictor_series is None or (hasattr(predictor_series, 'data') and np.all(np.isnan(predictor_series.data))):
                logging.warning(f"Predictor series for {var_name} ({season}, {dataset_id}) for beta_obs_slope is None or all NaNs. Skipping.")
                continue

            try:
                logging.debug(f"Performing linear regression for beta_obs_slope: U850_box_aligned vs {var_name}_predictor ({season}, {dataset_id})...")
                
                if np.all(np.isnan(predictor_series.data)) or np.all(np.isnan(ua850_box_aligned.data)):
                    logging.warning(f"Cannot perform regression for beta_obs_slope ({season}, {dataset_id}): one or both series are all NaNs before regression call. Skipping.")
                    continue
                if predictor_series.size < 2 or ua850_box_aligned.size < 2 : 
                     logging.warning(f"Not enough data points for regression for beta_obs_slope ({season}, {dataset_id}): X_size={predictor_series.size}, Y_size={ua850_box_aligned.size}. Skipping.")
                     continue
                if predictor_series.size > 1 and np.nanvar(predictor_series.data) < 1e-9: 
                    logging.warning(f"Predictor series for {var_name} ({season}, {dataset_id}) for beta_obs_slope has near-zero variance ({np.nanvar(predictor_series.data)}). Skipping regression.")
                    continue

                slope, intercept, r_value, p_value, std_err = StatsAnalyzer.perform_linear_regression(
                    predictor_series,    
                    ua850_box_aligned    
                )
                if np.isnan(slope):
                    logging.warning(f"Regression slope for beta_obs_slope is NaN for U850 vs {var_name} ({season}, {dataset_id}). Skipping.")
                    continue

                result_key = f"{season}_U850_vs_{var_name}" 
                beta_obs_results[result_key] = slope
                logging.info(f"  SUCCESS (beta_obs_slope): {dataset_id} {season} U850_box vs {var_name}_box: slope = {slope:.3f} (p={p_value:.3f}, r={r_value:.3f})")

            except Exception as e:
                logging.error(f"Error during linear regression for beta_obs_slope ({season}, {dataset_id}, var {var_name}): {e}", exc_info=True)
                continue 
        
        logging.info(f"--- Finished processing beta_obs_slopes for all pairs for {dataset_id} ---")
        if not beta_obs_results: 
             logging.warning(f"Beta_obs_slopes calculation for {dataset_id} resulted in NO VALID SLOPES. Returning empty dictionary.")
        else:
            logging.info(f"Final beta_obs_slopes for {dataset_id}: {beta_obs_results}")

        return beta_obs_results

    @staticmethod
    def analyze_jet_indices(jet_data_input: dict, dataset_name: str):
        """
        Analyzes calculated jet indices (e.g., from ClimateAnalysis module output).
        Focuses on detrending the raw jet indices.
        """
        logging.info(f"Performing advanced analysis (detrending) on jet indices for {dataset_name}...")
        jet_analysis_results = jet_data_input.copy() 

        if not jet_data_input:
            logging.warning(f"No jet data provided for {dataset_name} to AdvancedAnalyzer.analyze_jet_indices. Returning empty results for advanced jet analysis.")
            return {} 

        detrend_jet = Config.DETREND_JET_INDICES 

        for key, jet_index_da in jet_data_input.items():
            if not isinstance(jet_index_da, xr.DataArray):
                logging.warning(f"Item '{key}' in jet_data_input for {dataset_name} is not an xarray.DataArray. Skipping.")
                continue
            
            if jet_index_da.size == 0:
                logging.warning(f"Jet index DataArray '{key}' for {dataset_name} is empty. Skipping detrending.")
                detrended_key = key.replace("_raw", "_detrended") if "_raw" in key else f"{key}_detrended"
                jet_analysis_results[detrended_key] = jet_index_da 
                continue

            logging.debug(f"  Analyzing jet index: {key}")
            
            if "_raw" not in key: 
                 raw_key_explicit = f"{key}_raw_from_analyze_jet" 
                 if raw_key_explicit not in jet_analysis_results: 
                    jet_analysis_results[raw_key_explicit] = jet_index_da.copy(deep=True)

            detrended_jet_index_da = jet_index_da.copy(deep=True) 
            if detrend_jet:
                try:
                    logging.debug(f"    Detrending {key}...")
                    # Sicherstellen, dass die Dimension für Detrending korrekt ist (oft 'time' oder 'season_year')
                    time_dim = next((d for d in ['time', 'season_year'] if d in jet_index_da.dims), None)
                    if not time_dim:
                        logging.error(f"    Could not find a suitable time dimension in {key} for detrending. Skipping detrend.")
                    else:
                        detrended_jet_index_da = StatsAnalyzer.detrend_data(jet_index_da, dim=time_dim)
                        logging.debug(f"    Detrending successful for {key}.")
                except Exception as e:
                    logging.error(f"    Error detrending jet index {key} for {dataset_name}: {e}. Using raw data instead for detrended version.", exc_info=True)
                    detrended_jet_index_da = jet_index_da.copy(deep=True) 
            
            detrended_key = key.replace("_raw", "_detrended") if "_raw" in key else f"{key}_detrended"
            jet_analysis_results[detrended_key] = detrended_jet_index_da
            logging.debug(f"  Stored detrended jet index as: {detrended_key}")

        logging.info(f"Finished advanced jet index analysis for {dataset_name}.")
        return jet_analysis_results


    @staticmethod
    def analyze_correlations(dataset_id: str, 
                             processed_data_for_dataset: dict, 
                             jet_data_for_dataset: dict, 
                             discharge_flow_data: dict = None):
        """
        Analyzes correlations between jet indices and other climate variables.
        """
        logging.info(f"Analyzing correlations for {dataset_id}...")
        correlation_results = {'pr': {}, 'tas': {}, 'discharge': {}, 'flow': {}} 

        if not processed_data_for_dataset and not discharge_flow_data:
            logging.warning(f"No climate data (processed_data or discharge_flow_data) provided for correlations with {dataset_id} jet. Skipping.")
            return correlation_results
        if not jet_data_for_dataset:
            logging.warning(f"No jet data provided for {dataset_id} for correlations. Skipping.")
            return correlation_results

        seasons_to_analyze = ['Winter', 'Summer']
        box_coords = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        detrend_climate_vars_for_corr = Config.DETREND_CLIMATE_VARS_FOR_CORRELATIONS 

        for season_l in [s.lower() for s in seasons_to_analyze]: 
            season_u = season_l.capitalize() 
            correlation_results['pr'][season_l] = {}
            correlation_results['tas'][season_l] = {}

            jet_speed_key = f"{dataset_id}_jet_speed_{season_l}_detrended"
            jet_lat_key = f"{dataset_id}_jet_lat_{season_l}_detrended"
            
            jet_speed_series = jet_data_for_dataset.get(jet_speed_key)
            jet_lat_series = jet_data_for_dataset.get(jet_lat_key)

            for var_type, var_data_full_field in [('pr', processed_data_for_dataset.get('pr')), 
                                                  ('tas', processed_data_for_dataset.get('tas'))]:
                if var_data_full_field is None:
                    logging.debug(f"  {var_type.upper()} data not available for {dataset_id} for {season_u} correlations. Skipping.")
                    continue

                try:
                    var_season_field = DataProcessor.select_season(var_data_full_field, season_u, use_datetime_season=Config.USE_DATETIME_SEASON_FOR_REANALYSIS) # DataProcessor verwenden
                    if var_season_field.size == 0: continue
                    
                    box_mean_var = DataProcessor.calculate_spatial_mean(var_season_field, box_coords) # DataProcessor verwenden
                    if box_mean_var.size == 0: continue

                    climate_var_for_corr = box_mean_var
                    if detrend_climate_vars_for_corr:
                        time_dim_clim = next((d for d in ['time', 'season_year'] if d in climate_var_for_corr.dims), 'time')
                        climate_var_for_corr = StatsAnalyzer.detrend_data(box_mean_var, dim=time_dim_clim)
                    
                    if jet_speed_series is not None and jet_speed_series.size > 0:
                        aligned_jet_speed, aligned_var = xr.align(jet_speed_series, climate_var_for_corr, join='inner')
                        if aligned_jet_speed.size > 1 and aligned_var.size > 1: 
                            r, p = StatsAnalyzer.perform_correlation(aligned_jet_speed, aligned_var, Config.CORRELATION_METHOD)
                            correlation_results[var_type][season_l]['speed'] = {
                                'r_value': r, 'p_value': p, 'n_eff': aligned_jet_speed.size, 
                                'variable1_name': f"{dataset_id} Jet Speed ({season_u}, detrended)",
                                'variable2_name': f"{dataset_id} Box {var_type.upper()} ({season_u}, {'detrended' if detrend_climate_vars_for_corr else 'raw'})",
                                'values1_on_common_years': aligned_jet_speed.data, 
                                'values2_on_common_years': aligned_var.data,
                            }
                            logging.debug(f"  Corr ({Config.CORRELATION_METHOD}) {dataset_id} Jet Speed vs Box {var_type.upper()} ({season_u}): r={r:.2f}, p={p:.3f}")

                    if jet_lat_series is not None and jet_lat_series.size > 0:
                        aligned_jet_lat, aligned_var = xr.align(jet_lat_series, climate_var_for_corr, join='inner')
                        if aligned_jet_lat.size > 1 and aligned_var.size > 1:
                            r, p = StatsAnalyzer.perform_correlation(aligned_jet_lat, aligned_var, Config.CORRELATION_METHOD)
                            correlation_results[var_type][season_l]['lat'] = {
                                'r_value': r, 'p_value': p, 'n_eff': aligned_jet_lat.size,
                                'variable1_name': f"{dataset_id} Jet Latitude ({season_u}, detrended)",
                                'variable2_name': f"{dataset_id} Box {var_type.upper()} ({season_u}, {'detrended' if detrend_climate_vars_for_corr else 'raw'})",
                                'values1_on_common_years': aligned_jet_lat.data,
                                'values2_on_common_years': aligned_var.data,
                            }
                            logging.debug(f"  Corr ({Config.CORRELATION_METHOD}) {dataset_id} Jet Latitude vs Box {var_type.upper()} ({season_u}): r={r:.2f}, p={p:.3f}")
                except Exception as e:
                    logging.error(f"Error during correlation analysis for {var_type.upper()} with jet in {season_u} for {dataset_id}: {e}", exc_info=True)

            if discharge_flow_data:
                correlation_results['discharge'][season_l] = {} 
                correlation_results['flow'][season_l] = {}

                for hydro_key, hydro_series_raw in discharge_flow_data.items():
                    if season_l not in hydro_key.lower(): 
                        continue
                    
                    hydro_var_name_pretty = hydro_key.replace(f"_{season_l}", "").replace("_", " ") 

                    hydro_series_for_corr = hydro_series_raw
                    if Config.DETREND_HYDRO_VARS_FOR_CORRELATIONS: 
                        try:
                            time_dim_hydro = next((d for d in ['time', 'season_year'] if d in hydro_series_raw.dims), 'time')
                            hydro_series_for_corr = StatsAnalyzer.detrend_data(hydro_series_raw, dim=time_dim_hydro) 
                        except Exception as e_hydro_detrend:
                            logging.error(f"Could not detrend hydro series {hydro_key}: {e_hydro_detrend}. Using raw for correlation.")
                            hydro_series_for_corr = hydro_series_raw 

                    category = 'discharge' if 'discharge' in hydro_key.lower() else 'flow'
                    
                    if jet_speed_series is not None and jet_speed_series.size > 0:
                        aligned_jet_speed, aligned_hydro = xr.align(jet_speed_series, hydro_series_for_corr, join='inner')
                        if aligned_jet_speed.size > 1 and aligned_hydro.size > 1:
                            r, p = StatsAnalyzer.perform_correlation(aligned_jet_speed, aligned_hydro, Config.CORRELATION_METHOD)
                            corr_label = f"jet_speed_vs_{hydro_key.replace(f'_{season_l}','')}" 
                            correlation_results[category][season_l][corr_label] = {
                                'r_value': r, 'p_value': p, 'n_eff': aligned_jet_speed.size,
                                'variable1_name': f"{dataset_id} Jet Speed ({season_u}, detrended)",
                                'variable2_name': f"{hydro_var_name_pretty} ({season_u}, {'detrended' if Config.DETREND_HYDRO_VARS_FOR_CORRELATIONS else 'raw'})",
                                'values1_on_common_years': aligned_jet_speed.data,
                                'values2_on_common_years': aligned_hydro.data,
                            }
                            logging.debug(f"  Corr ({Config.CORRELATION_METHOD}) {dataset_id} Jet Speed vs {hydro_var_name_pretty} ({season_u}): r={r:.2f}, p={p:.3f}")
                    
                    if jet_lat_series is not None and jet_lat_series.size > 0:
                        aligned_jet_lat, aligned_hydro = xr.align(jet_lat_series, hydro_series_for_corr, join='inner')
                        if aligned_jet_lat.size > 1 and aligned_hydro.size > 1:
                            r, p = StatsAnalyzer.perform_correlation(aligned_jet_lat, aligned_hydro, Config.CORRELATION_METHOD)
                            corr_label = f"jet_lat_vs_{hydro_key.replace(f'_{season_l}','')}"
                            correlation_results[category][season_l][corr_label] = {
                                'r_value': r, 'p_value': p, 'n_eff': aligned_jet_lat.size,
                                'variable1_name': f"{dataset_id} Jet Latitude ({season_u}, detrended)",
                                'variable2_name': f"{hydro_var_name_pretty} ({season_u}, {'detrended' if Config.DETREND_HYDRO_VARS_FOR_CORRELATIONS else 'raw'})",
                                'values1_on_common_years': aligned_jet_lat.data,
                                'values2_on_common_years': aligned_hydro.data,
                            }
                            logging.debug(f"  Corr ({Config.CORRELATION_METHOD}) {dataset_id} Jet Latitude vs {hydro_var_name_pretty} ({season_u}): r={r:.2f}, p={p:.3f}")

        logging.info(f"Finished correlation analysis for {dataset_id}.")
        return correlation_results

    @staticmethod
    def calculate_cmip6_regression_maps(cmip6_model_data_loaded: dict, historical_period: tuple):
        """
        Calculates regression maps for CMIP6 multi-model mean (MMM) historical data.
        """
        logging.info(f"Calculating CMIP6 MMM Historical Regression Maps for period: {historical_period}...")
        
        mmm_data = {}
        variables = ['ua850', 'pr', 'tas']
        start_year_hist, end_year_hist = historical_period

        for var in variables:
            model_arrays_for_var = []
            for model_name, model_vars in cmip6_model_data_loaded.items():
                if var in model_vars:
                    try:
                        var_da_hist_period = DataProcessor.select_time_period(model_vars[var], start_year_hist, end_year_hist) # DataProcessor verwenden
                        if var_da_hist_period is not None and var_da_hist_period.size > 0:
                            model_arrays_for_var.append(var_da_hist_period)
                        else:
                            logging.debug(f"  No data for var '{var}' in model '{model_name}' for period {historical_period}.")
                    except Exception as e_hist_select:
                        logging.warning(f"  Error selecting historical period for var '{var}' in model '{model_name}': {e_hist_select}")
                else:
                    logging.debug(f"  Variable '{var}' not found in loaded data for model '{model_name}'.")
            
            if not model_arrays_for_var:
                logging.warning(f"No model data found for variable '{var}' in the historical period {historical_period} to compute MMM. Skipping this variable.")
                mmm_data[var] = None
                continue
            
            try:
                logging.info(f"  Calculating MMM for '{var}' using {len(model_arrays_for_var)} models for period {historical_period}...")
                mmm_var_da = DataProcessor.calculate_multi_model_mean(model_arrays_for_var) # DataProcessor verwenden
                if mmm_var_da is not None and mmm_var_da.size > 0:
                    mmm_data[var] = mmm_var_da
                    logging.info(f"  MMM for '{var}' calculated. Shape: {mmm_var_da.dims if hasattr(mmm_var_da, 'dims') else 'N/A'}")
                else:
                    logging.warning(f"  MMM calculation for '{var}' resulted in empty or None. Skipping this variable.")
                    mmm_data[var] = None
            except Exception as e_mmm:
                logging.error(f"  Error calculating MMM for variable '{var}': {e_mmm}", exc_info=True)
                mmm_data[var] = None

        if not mmm_data.get('ua850'):
            logging.error("MMM for U850 could not be calculated. Halting CMIP6 MMM regression maps.")
            return {}

        cmip6_mmm_data_for_regression_fn = {"CMIP6_MMM_Hist": mmm_data} 

        cmip6_mmm_regression_results = AdvancedAnalyzer.calculate_regression_maps(
            cmip6_mmm_data_for_regression_fn, 
            jet_indices_data_all={}, 
            dataset_id_str="CMIP6_MMM_Hist"
        )
        
        logging.info("Finished calculating CMIP6 MMM Historical Regression Maps.")
        return cmip6_mmm_regression_results

    @staticmethod
    def calculate_historical_slopes_comparison(beta_obs_slopes_reanalysis: dict,
                                             cmip6_model_data_loaded: dict,
                                             jet_data_reanalysis: dict, 
                                             historical_period_for_cmip6: tuple):
        """
        Compares beta_obs slopes from reanalysis with slopes calculated from individual CMIP6 models.
        """
        logging.info(f"Calculating CMIP6 historical slopes for comparison with reanalysis beta_obs over period {historical_period_for_cmip6}...")
        cmip6_historical_slopes_all_models = {}

        if not beta_obs_slopes_reanalysis:
            logging.warning("Reanalysis beta_obs_slopes not provided. Comparison will only show CMIP6 slopes.")
        else:
            cmip6_historical_slopes_all_models[Config.DATASET_ERA5] = beta_obs_slopes_reanalysis

        start_year_cmip, end_year_cmip = historical_period_for_cmip6

        for model_name, model_vars_data in cmip6_model_data_loaded.items():
            logging.info(f"  Processing historical slopes for CMIP6 model: {model_name}")
            
            model_data_for_slope_calc = {}
            valid_model_for_slopes = True
            for var_key in ['ua850', 'pr', 'tas']:
                if var_key in model_vars_data:
                    try:
                        var_da_hist = DataProcessor.select_time_period(model_vars_data[var_key], start_year_cmip, end_year_cmip) # DataProcessor verwenden
                        if var_da_hist is not None and var_da_hist.size > 0:
                            model_data_for_slope_calc[var_key] = var_da_hist
                        else:
                            logging.warning(f"    Variable {var_key} for model {model_name} has no data in period {historical_period_for_cmip6}. Cannot calculate slopes for this model.")
                            valid_model_for_slopes = False; break
                    except Exception as e_hist_sel_cmip:
                        logging.error(f"    Error selecting historical period for {var_key} in {model_name}: {e_hist_sel_cmip}. Skipping model.", exc_info=True)
                        valid_model_for_slopes = False; break
                else:
                    logging.warning(f"    Variable {var_key} not found for model {model_name}. Cannot calculate historical slopes for this model.")
                    valid_model_for_slopes = False; break
            
            if not valid_model_for_slopes:
                cmip6_historical_slopes_all_models[model_name] = {} 
                continue

            try:
                model_specific_slopes = AdvancedAnalyzer.calculate_beta_obs_slopes_for_era5( # Wiederverwendung der Logik
                    model_data_for_slope_calc, 
                    model_name 
                )
                cmip6_historical_slopes_all_models[model_name] = model_specific_slopes
                if model_specific_slopes:
                    logging.info(f"    Successfully calculated historical slopes for {model_name}: {model_specific_slopes}")
                else:
                    logging.warning(f"    Calculation of historical slopes for {model_name} resulted in empty dictionary.")
            except Exception as e_model_slope:
                logging.error(f"    Error calculating historical slopes for model {model_name}: {e_model_slope}", exc_info=True)
                cmip6_historical_slopes_all_models[model_name] = {}

        logging.info("Finished calculating CMIP6 historical slopes for comparison.")
        return cmip6_historical_slopes_all_models