#!/usr/bin/env python3
"""
Advanced analysis methods for climate data, including CMIP6 specific analyses,
regression maps, and jet impact assessments.
"""
import numpy as np
import xarray as xr
import pandas as pd # Though not directly used in all methods, often useful with xarray
import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.interpolate import griddata # For regridding in difference plots

# Relative imports for utility modules
from config_setup import Config
from data_utils import DataProcessor # Assuming DataProcessor is in data_utils.py
from stats_utils import StatsAnalyzer # Assuming StatsAnalyzer is in stats_utils.py
from jet_utils import JetStreamAnalyzer # Assuming JetStreamAnalyzer is in jet_utils.py


class AdvancedAnalyzer:
    """Advanced analysis methods for climate data."""

    @staticmethod
    def _calculate_regression_grid_point(index_series_np, weather_variable_timeseries_np):
        """
        Helper to calculate regression for a single grid point.
        Both inputs are 1D NumPy arrays of the same length (already time-matched).
        """
        # StatsAnalyzer.calculate_regression handles NaN and length checks internally
        slope, _, _, p_value, _ = StatsAnalyzer.calculate_regression(
            index_series_np, weather_variable_timeseries_np # X=index, Y=weather_var
        )
        return slope, p_value

    @staticmethod
    def _parallel_regression_chunk(index_series_np, weather_variable_chunk_np,
                                   lat_indices_in_chunk, lon_indices_global,
                                   global_lat_dim_name, global_lon_dim_name):
        """
        Helper for parallel regression calculation on a chunk of spatial data.
        Args:
            index_series_np (np.ndarray): 1D array of the index (e.g., PR box mean).
            weather_variable_chunk_np (np.ndarray): Chunk of the weather variable data,
                                                    expected shape (n_time, n_lat_chunk, n_lon_global)
                                                    or (n_time, n_lat_chunk) if lon is squeezed.
            lat_indices_in_chunk (list): List of original latitude indices this chunk corresponds to.
            lon_indices_global (list): List of all original longitude indices.
            global_lat_dim_name (str): Name of latitude dimension in original full DataArray.
            global_lon_dim_name (str): Name of longitude dimension in original full DataArray.

        Returns:
            tuple: (slopes_chunk, p_values_chunk, lat_indices_in_chunk, lon_indices_global)
                   slopes_chunk and p_values_chunk have shape (len(lat_indices_in_chunk), len(lon_indices_global))
        """
        num_lats_chunk = len(lat_indices_in_chunk)
        num_lons_global = len(lon_indices_global)

        slopes_chunk = np.full((num_lats_chunk, num_lons_global), np.nan)
        p_values_chunk = np.full((num_lats_chunk, num_lons_global), np.nan)

        # weather_variable_chunk_np should have time as the first dimension after .transpose()
        # Expected shape: (time, lat_chunk, lon_global)
        if weather_variable_chunk_np.shape[0] != len(index_series_np):
            logging.error(f"[_parallel_regression_chunk] Time dimension mismatch: "
                          f"Index len {len(index_series_np)}, "
                          f"Weather var chunk time dim {weather_variable_chunk_np.shape[0]}")
            return slopes_chunk, p_values_chunk, lat_indices_in_chunk, lon_indices_global
        
        for i_lat_chunk, _ in enumerate(lat_indices_in_chunk): # iter over lat dim of the chunk
            for j_lon_global, _ in enumerate(lon_indices_global): # iter over global lon dim
                try:
                    point_timeseries = weather_variable_chunk_np[:, i_lat_chunk, j_lon_global]
                    slope, p_val = AdvancedAnalyzer._calculate_regression_grid_point(
                        index_series_np, point_timeseries
                    )
                    slopes_chunk[i_lat_chunk, j_lon_global] = slope
                    p_values_chunk[i_lat_chunk, j_lon_global] = p_val
                except IndexError:
                    logging.error(f"IndexError at chunk_lat_idx {i_lat_chunk}, global_lon_idx {j_lon_global}. "
                                  f"Weather var chunk shape: {weather_variable_chunk_np.shape}")
                except Exception as e_point:
                    logging.error(f"Error at chunk_lat_idx {i_lat_chunk}, global_lon_idx {j_lon_global}: {e_point}")

        return slopes_chunk, p_values_chunk, lat_indices_in_chunk, lon_indices_global


    @staticmethod
    def calculate_regression_maps(processed_datasets_dict, dataset_key_to_process):
        """
        Calculate regression maps (e.g., U850 onto PR/TAS box indices) for a specific dataset.
        Args:
            processed_datasets_dict (dict): Dictionary containing processed DataArrays from DataProcessor.
                                           Keys like 'ERA5_pr_box_mean', '20CRv3_ua850_seasonal'.
            dataset_key_to_process (str): "ERA5" or "20CRv3".
        Returns:
            dict: Regression results for Winter and Summer.
                  e.g., {'Winter': {'slopes_pr':..., 'p_values_pr':..., ...}, 'Summer':{...}}
        """
        if dataset_key_to_process not in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.warning(f"Invalid dataset key for regression maps: {dataset_key_to_process}")
            return {}

        logging.info(f"Calculating U850 vs Box Index regression maps for {dataset_key_to_process}...")

        try:
            # Retrieve necessary processed data for the specified dataset
            pr_box_mean_full_season = processed_datasets_dict.get(f'{dataset_key_to_process}_pr_box_mean')
            tas_box_mean_full_season = processed_datasets_dict.get(f'{dataset_key_to_process}_tas_box_mean')
            ua850_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_ua850_seasonal')

            if pr_box_mean_full_season is None or tas_box_mean_full_season is None or ua850_full_seasonal is None:
                logging.error(f"Missing input data for regression maps ({dataset_key_to_process}). "
                              f"Needs pr_box_mean, tas_box_mean, ua850_seasonal.")
                return {}

            # Detrend data before seasonal filtering and regression
            logging.debug(f"Detrending inputs for {dataset_key_to_process} regression maps...")
            pr_box_mean_detrended = DataProcessor.detrend_data(pr_box_mean_full_season)
            tas_box_mean_detrended = DataProcessor.detrend_data(tas_box_mean_full_season)
            ua850_seasonal_detrended = DataProcessor.detrend_data(ua850_full_seasonal) # For regression
            # Original ua850_seasonal is used for mean contours, no detrending for that.

            all_season_results = {}
            seasons_to_analyze = ['Winter', 'Summer']

            for season in seasons_to_analyze:
                logging.info(f"  - Processing {season} regression maps for {dataset_key_to_process}...")
                # Filter detrended indices for the current season
                pr_index_season = DataProcessor.filter_by_season(pr_box_mean_detrended, season)
                tas_index_season = DataProcessor.filter_by_season(tas_box_mean_detrended, season)
                # Filter detrended U850 field for the current season
                ua850_field_season_detrended = DataProcessor.filter_by_season(ua850_seasonal_detrended, season)
                # Filter original U850 for mean contours
                ua850_field_season_original = DataProcessor.filter_by_season(ua850_full_seasonal, season)


                if not all(da is not None and da.size > 0 for da in [pr_index_season, tas_index_season, ua850_field_season_detrended]):
                    logging.warning(f"Skipping {season} for {dataset_key_to_process}: Missing or empty seasonal data after filtering/detrending.")
                    continue

                # --- Regression for PR vs U850 ---
                # Find common years between PR index and U850 field
                common_years_pr_ua = np.intersect1d(pr_index_season.season_year.values,
                                                    ua850_field_season_detrended.season_year.values)
                if len(common_years_pr_ua) < 3:
                    logging.warning(f"  Not enough common years for PR-U850 regression in {season} ({len(common_years_pr_ua)}).")
                    slopes_pr, p_values_pr = np.nan, np.nan # Placeholder
                else:
                    pr_index_common = pr_index_season.sel(season_year=common_years_pr_ua).data # Numpy array
                    ua850_common_pr = ua850_field_season_detrended.sel(season_year=common_years_pr_ua)
                    
                    # Prepare for parallel processing
                    lat_dim_name = next((d for d in ua850_common_pr.dims if 'lat' in d.lower()), 'lat')
                    lon_dim_name = next((d for d in ua850_common_pr.dims if 'lon' in d.lower()), 'lon')
                    time_dim_name = 'season_year' # Should be this after seasonal processing

                    # Transpose ua850_common_pr to have time as the first dimension for easier chunk processing
                    ua850_time_first_pr = ua850_common_pr.transpose(time_dim_name, lat_dim_name, lon_dim_name).data

                    num_lats = ua850_common_pr.sizes[lat_dim_name]
                    num_lons = ua850_common_pr.sizes[lon_dim_name]
                    slopes_pr = np.full((num_lats, num_lons), np.nan)
                    p_values_pr = np.full((num_lats, num_lons), np.nan)
                    
                    lat_indices_global = list(range(num_lats))
                    lon_indices_global = list(range(num_lons))

                    # Parallel execution
                    with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
                        futures = []
                        # Chunking by latitude for parallelism
                        # Determine a reasonable chunk size
                        num_chunks = Config.N_PROCESSES * 2 # Example: Aim for twice as many chunks as workers
                        chunk_size = max(1, num_lats // num_chunks)

                        for i_start_lat in range(0, num_lats, chunk_size):
                            lat_chunk_indices = lat_indices_global[i_start_lat : min(i_start_lat + chunk_size, num_lats)]
                            if not lat_chunk_indices: continue
                            
                            # Select the corresponding slice of ua850_time_first_pr data for this chunk
                            # weather_variable_chunk_np shape (n_time, n_lat_chunk, n_lon_global)
                            ua850_data_chunk = ua850_time_first_pr[:, lat_chunk_indices[0]:lat_chunk_indices[-1]+1, :]

                            futures.append(executor.submit(AdvancedAnalyzer._parallel_regression_chunk,
                                                           pr_index_common, ua850_data_chunk,
                                                           lat_chunk_indices, lon_indices_global, # Pass original global indices
                                                           lat_dim_name, lon_dim_name))
                        for future in as_completed(futures):
                            try:
                                s_chunk, p_chunk, lats_processed, lons_processed = future.result()
                                # Place results from chunk into the global arrays
                                # lats_processed are the original global latitude indices for this chunk
                                for i_chunk, lat_idx_global in enumerate(lats_processed):
                                    slopes_pr[lat_idx_global, :] = s_chunk[i_chunk, :]
                                    p_values_pr[lat_idx_global, :] = p_chunk[i_chunk, :]
                            except Exception as e_future:
                                logging.error(f"Error processing a PR-U850 regression chunk: {e_future}")


                # --- Regression for TAS vs U850 (similar logic) ---
                common_years_tas_ua = np.intersect1d(tas_index_season.season_year.values,
                                                     ua850_field_season_detrended.season_year.values)
                if len(common_years_tas_ua) < 3:
                    logging.warning(f"  Not enough common years for TAS-U850 regression in {season} ({len(common_years_tas_ua)}).")
                    slopes_tas, p_values_tas = np.nan, np.nan # Placeholder
                else:
                    tas_index_common = tas_index_season.sel(season_year=common_years_tas_ua).data
                    ua850_common_tas = ua850_field_season_detrended.sel(season_year=common_years_tas_ua)
                    ua850_time_first_tas = ua850_common_tas.transpose(time_dim_name, lat_dim_name, lon_dim_name).data

                    num_lats_tas = ua850_common_tas.sizes[lat_dim_name] # Should be same as for PR
                    num_lons_tas = ua850_common_tas.sizes[lon_dim_name]
                    slopes_tas = np.full((num_lats_tas, num_lons_tas), np.nan)
                    p_values_tas = np.full((num_lats_tas, num_lons_tas), np.nan)
                    
                    lat_indices_global_tas = list(range(num_lats_tas))
                    lon_indices_global_tas = list(range(num_lons_tas))


                    with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
                        futures_tas = []
                        num_chunks_tas = Config.N_PROCESSES * 2
                        chunk_size_tas = max(1, num_lats_tas // num_chunks_tas)

                        for i_start_lat in range(0, num_lats_tas, chunk_size_tas):
                            lat_chunk_indices_tas = lat_indices_global_tas[i_start_lat : min(i_start_lat + chunk_size_tas, num_lats_tas)]
                            if not lat_chunk_indices_tas: continue
                            ua850_data_chunk_tas = ua850_time_first_tas[:, lat_chunk_indices_tas[0]:lat_chunk_indices_tas[-1]+1, :]
                            futures_tas.append(executor.submit(AdvancedAnalyzer._parallel_regression_chunk,
                                                               tas_index_common, ua850_data_chunk_tas,
                                                               lat_chunk_indices_tas, lon_indices_global_tas,
                                                               lat_dim_name, lon_dim_name))
                        for future in as_completed(futures_tas):
                            try:
                                s_chunk, p_chunk, lats_processed, _ = future.result()
                                for i_chunk, lat_idx_global in enumerate(lats_processed):
                                    slopes_tas[lat_idx_global, :] = s_chunk[i_chunk, :]
                                    p_values_tas[lat_idx_global, :] = p_chunk[i_chunk, :]
                            except Exception as e_future:
                                logging.error(f"Error processing a TAS-U850 regression chunk: {e_future}")

                # Calculate seasonal mean of original U850 for contours
                ua850_mean_contour_val = np.nan
                if ua850_field_season_original is not None and ua850_field_season_original.size > 0:
                    try:
                        ua850_mean_da = ua850_field_season_original.mean(dim='season_year', skipna=True)
                        # Ensure 2D for plotting
                        if ua850_mean_da.ndim > 2:
                            non_spatial_dims = [d for d in ua850_mean_da.dims if d not in [lat_dim_name, lon_dim_name]]
                            dims_to_squeeze = [d for d in non_spatial_dims if ua850_mean_da.sizes.get(d, 0) == 1]
                            if dims_to_squeeze:
                                ua850_mean_da = ua850_mean_da.squeeze(dim=dims_to_squeeze, drop=True)
                        if ua850_mean_da.ndim == 2:
                            ua850_mean_contour_val = ua850_mean_da.data # Get numpy array
                        else:
                            logging.warning(f"U850 mean for contours is not 2D after squeeze ({ua850_mean_da.dims}). Contours might be affected.")
                            ua850_mean_contour_val = ua850_mean_da.data # Try anyway
                    except Exception as e_mean_contour:
                        logging.warning(f"Could not calculate U850 mean for contours ({season}): {e_mean_contour}")

                # Get coordinates for plotting (from one of the common U850 arrays)
                coords_source = ua850_common_pr if 'ua850_common_pr' in locals() and ua850_common_pr is not None else ua850_field_season_detrended
                lons_plot = coords_source[lon_dim_name].values
                lats_plot = coords_source[lat_dim_name].values
                # Meshgrid if 1D
                if lons_plot.ndim == 1 and lats_plot.ndim == 1:
                    lons_plot, lats_plot = np.meshgrid(lons_plot, lats_plot)

                all_season_results[season] = {
                    'slopes_pr': slopes_pr, 'p_values_pr': p_values_pr,
                    'slopes_tas': slopes_tas, 'p_values_tas': p_values_tas,
                    'ua850_mean_for_contours': ua850_mean_contour_val,
                    'lons': lons_plot, 'lats': lats_plot,
                    'common_years_pr_ua': common_years_pr_ua if 'common_years_pr_ua' in locals() else [],
                    'common_years_tas_ua': common_years_tas_ua if 'common_years_tas_ua' in locals() else []
                }
            return all_season_results
        except Exception as e:
            logging.error(f"General error in calculate_regression_maps for {dataset_key_to_process}: {e}")
            logging.error(traceback.format_exc())
            return {}


    @staticmethod
    def calculate_jet_impact_maps(processed_datasets_dict, jet_data_dict, dataset_key_to_process, season_to_analyze):
        """
        Calculate regression maps showing the impact of jet indices on PR and TAS for a given season and dataset.
        Args:
            processed_datasets_dict (dict): From ClimateAnalysis.process_..._data()
            jet_data_dict (dict): From AdvancedAnalyzer.analyze_jet_indices()
            dataset_key_to_process (str): "ERA5" or "20CRv3"
            season_to_analyze (str): "Winter" or "Summer"
        Returns:
            dict: {season: {impact_key: {'slopes':..., 'p_values':..., 'lons':..., 'lats':...}}}
        """
        logging.info(f"Calculating jet impact maps for {dataset_key_to_process} ({season_to_analyze})...")

        if dataset_key_to_process not in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.warning(f"Invalid dataset key for jet impact maps: {dataset_key_to_process}")
            return {}
        if season_to_analyze not in ['Winter', 'Summer']:
            logging.warning(f"Invalid season for jet impact maps: {season_to_analyze}")
            return {}

        try:
            # Get full seasonal PR and TAS data (these are typically anomalies already)
            tas_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_tas_seasonal')
            pr_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_pr_seasonal')

            if tas_full_seasonal is None or pr_full_seasonal is None:
                logging.error(f"Missing tas_seasonal or pr_seasonal for {dataset_key_to_process} to calculate jet impacts.")
                return {}

            # Filter PR/TAS for the specific season and detrend
            tas_season_filtered = DataProcessor.filter_by_season(tas_full_seasonal, season_to_analyze)
            pr_season_filtered = DataProcessor.filter_by_season(pr_full_seasonal, season_to_analyze)
            
            tas_season_detrended = DataProcessor.detrend_data(tas_season_filtered)
            pr_season_detrended = DataProcessor.detrend_data(pr_season_filtered)

            impact_maps_for_season = {season_to_analyze: {}}
            season_lower = season_to_analyze.lower()

            # Define which jet data bundles from jet_data_dict to use
            # The keys in jet_data_dict are like 'ERA5_winter_speed_tas_data', '20CRv3_summer_lat_pr_data'
            # We need the 'jet' DataArray and 'years' from these bundles.
            jet_analyses_config = [
                {'var_to_impact': tas_season_detrended, 'var_name': 'tas', 'jet_type': 'speed',
                 'jet_bundle_key_suffix': f'_{season_lower}_speed_tas_data'}, # Suffix for jet_data_dict key
                {'var_to_impact': pr_season_detrended, 'var_name': 'pr', 'jet_type': 'speed',
                 'jet_bundle_key_suffix': f'_{season_lower}_speed_pr_data'},
                {'var_to_impact': tas_season_detrended, 'var_name': 'tas', 'jet_type': 'lat',
                 'jet_bundle_key_suffix': f'_{season_lower}_lat_tas_data'},
                {'var_to_impact': pr_season_detrended, 'var_name': 'pr', 'jet_type': 'lat',
                 'jet_bundle_key_suffix': f'_{season_lower}_lat_pr_data'}
            ]

            for analysis_item in jet_analyses_config:
                var_field_detrended = analysis_item['var_to_impact'] # This is the 2D+time field (e.g., TAS over Europe)
                var_name_str = analysis_item['var_name']
                jet_type_str = analysis_item['jet_type']
                
                jet_bundle_full_key = f"{dataset_key_to_process}{analysis_item['jet_bundle_key_suffix']}"
                jet_data_bundle = jet_data_dict.get(jet_bundle_full_key)

                output_key = f'jet_{jet_type_str}_{var_name_str}' # e.g., 'jet_speed_tas'
                logging.info(f"  Processing impact: {dataset_key_to_process} {season_to_analyze} - {var_name_str} vs Jet {jet_type_str} (using bundle: {jet_bundle_full_key})")

                if var_field_detrended is None or var_field_detrended.size == 0:
                    logging.warning(f"    Skipping {output_key}: Variable field '{var_name_str}' for season '{season_to_analyze}' is missing or empty after detrending.")
                    continue
                if jet_data_bundle is None or 'jet' not in jet_data_bundle or 'years' not in jet_data_bundle:
                    logging.warning(f"    Skipping {output_key}: Jet data bundle '{jet_bundle_full_key}' is missing or incomplete.")
                    continue
                
                jet_index_ts = jet_data_bundle['jet'] # This is the 1D detrended jet index time series
                jet_index_years = jet_data_bundle['years'] # Numpy array of years for the jet index

                if jet_index_ts is None or jet_index_ts.size == 0:
                    logging.warning(f"    Skipping {output_key}: Jet index timeseries from bundle '{jet_bundle_full_key}' is missing or empty.")
                    continue

                # Find common years
                common_years = np.intersect1d(var_field_detrended.season_year.values, jet_index_years)
                if len(common_years) < 5: # Need a few years for regression
                    logging.warning(f"    Skipping {output_key}: Not enough common years ({len(common_years)}) between variable field and jet index.")
                    continue

                var_field_common = var_field_detrended.sel(season_year=common_years)
                jet_index_common = jet_index_ts.sel(season_year=common_years).data # Get as numpy array

                # Parallel regression calculation (similar to U850 vs Box Index)
                lat_dim_name = next((d for d in var_field_common.dims if 'lat' in d.lower()), 'lat')
                lon_dim_name = next((d for d in var_field_common.dims if 'lon' in d.lower()), 'lon')
                time_dim_name = 'season_year'
                
                var_field_time_first = var_field_common.transpose(time_dim_name, lat_dim_name, lon_dim_name).data

                num_lats = var_field_common.sizes[lat_dim_name]
                num_lons = var_field_common.sizes[lon_dim_name]
                slopes_map = np.full((num_lats, num_lons), np.nan)
                p_values_map = np.full((num_lats, num_lons), np.nan)

                lat_indices_global = list(range(num_lats))
                lon_indices_global = list(range(num_lons))

                with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
                    futures = []
                    num_chunks = Config.N_PROCESSES * 2 
                    chunk_size = max(1, num_lats // num_chunks)
                    for i_start_lat in range(0, num_lats, chunk_size):
                        lat_chunk_indices = lat_indices_global[i_start_lat : min(i_start_lat + chunk_size, num_lats)]
                        if not lat_chunk_indices: continue
                        var_data_chunk = var_field_time_first[:, lat_chunk_indices[0]:lat_chunk_indices[-1]+1, :]
                        futures.append(executor.submit(AdvancedAnalyzer._parallel_regression_chunk,
                                                       jet_index_common, var_data_chunk,
                                                       lat_chunk_indices, lon_indices_global,
                                                       lat_dim_name, lon_dim_name))
                    for future in as_completed(futures):
                        try:
                            s_chunk, p_chunk, lats_processed, _ = future.result()
                            for i_chunk, lat_idx_global in enumerate(lats_processed):
                                slopes_map[lat_idx_global, :] = s_chunk[i_chunk, :]
                                p_values_map[lat_idx_global, :] = p_chunk[i_chunk, :]
                        except Exception as e_future:
                            logging.error(f"Error processing a jet impact regression chunk for {output_key}: {e_future}")
                
                # Store results
                lons_plot = var_field_common[lon_dim_name].values
                lats_plot = var_field_common[lat_dim_name].values
                if lons_plot.ndim == 1: lons_plot, lats_plot = np.meshgrid(lons_plot, lats_plot)

                impact_maps_for_season[season_to_analyze][output_key] = {
                    'slopes': slopes_map,
                    'p_values': p_values_map,
                    'lons': lons_plot,
                    'lats': lats_plot,
                    'common_years_count': len(common_years)
                }
                logging.info(f"  Successfully calculated jet impact map: {dataset_key_to_process} {season_to_analyze} - {output_key}")

            return impact_maps_for_season
        except Exception as e:
            logging.error(f"General error in calculate_jet_impact_maps for {dataset_key_to_process} ({season_to_analyze}): {e}")
            logging.error(traceback.format_exc())
            return {season_to_analyze: {}} # Return empty for this season on error

    @staticmethod
    def analyze_correlations(processed_datasets_dict, discharge_data_dict, jet_data_dict, dataset_key_to_process):
        """
        Analyze specific correlations (e.g., Discharge vs Jet Speed, PR vs Jet Lat) for a dataset.
        Args:
            processed_datasets_dict (dict): Contains processed seasonal box means for PR/TAS.
            discharge_data_dict (dict): Contains processed seasonal discharge metrics.
            jet_data_dict (dict): Contains processed jet indices and related data bundles.
            dataset_key_to_process (str): "ERA5" or "20CRv3".
        Returns:
            dict: Nested dictionary with correlation results.
                  e.g., {'winter': {'discharge_jet_speed': {r_value: ..., p_value: ...}},
                         'pr': {'winter': {'lat': {r_value: ..., p_value: ...}}}}
        """
        logging.info(f"Analyzing selected time series correlations for {dataset_key_to_process}...")

        if dataset_key_to_process not in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.warning(f"Invalid dataset key for correlations: {dataset_key_to_process}")
            return {}

        # Initialize structure for results
        all_correlations_for_dataset = {
            'winter': {}, 'summer': {},
            'pr': {'winter': {}, 'summer': {}},
            'tas': {'winter': {}, 'summer': {}}
        }

        # Jet data keys in jet_data_dict are prefixed with dataset_key, e.g., "ERA5_winter_speed_tas_data"
        # Discharge data keys are like "winter_discharge", "summer_extreme_flow"
        
        correlation_pairs = [
            # Season, Var1_source_dict, Var1_key, Var2_is_jet_type (speed/lat), Var2_impact_var (tas/pr for bundle key) Output_main_key, Output_sub_key
            # Winter Discharge vs Jet Speed
            ('Winter', discharge_data_dict, 'winter_discharge', 'speed', 'tas', 'winter', 'discharge_jet_speed'),
            ('Winter', discharge_data_dict, 'winter_extreme_flow', 'speed', 'tas', 'winter', 'extreme_flow_jet_speed'),
            # Winter PR vs Jet Latitude
            ('Winter', 'box_pr', None, 'lat', 'pr', 'pr', 'lat'), # 'box_pr' indicates using processed_datasets_dict
            # Winter TAS vs Jet Speed
            ('Winter', 'box_tas', None, 'speed', 'tas', 'tas', 'speed'),
            # Winter PR vs Jet Speed
            ('Winter', 'box_pr', None, 'speed', 'pr', 'pr', 'speed'),
            # Winter TAS vs Jet Latitude
            ('Winter', 'box_tas', None, 'lat', 'tas', 'tas', 'lat'),

            # Summer versions
            ('Summer', discharge_data_dict, 'summer_discharge', 'speed', 'tas', 'summer', 'discharge_jet_speed'),
            ('Summer', discharge_data_dict, 'summer_extreme_flow', 'speed', 'tas', 'summer', 'extreme_flow_jet_speed'),
            ('Summer', 'box_pr', None, 'lat', 'pr', 'pr', 'lat'),
            ('Summer', 'box_tas', None, 'speed', 'tas', 'tas', 'speed'),
            ('Summer', 'box_pr', None, 'speed', 'pr', 'pr', 'speed'),
            ('Summer', 'box_tas', None, 'lat', 'tas', 'tas', 'lat'),
        ]

        for season, src1_dict_or_key, key1, jet_type, jet_impact_var, out_main, out_sub in correlation_pairs:
            season_lower = season.lower()
            
            # Get Variable 1 (e.g., discharge, or PR/TAS box mean)
            var1_ts = None
            var1_label = key1 if key1 else f"{season} Box {src1_dict_or_key.split('_')[1].upper()}" # e.g. Winter Box PR
            
            if isinstance(src1_dict_or_key, dict): # For discharge data
                var1_ts = src1_dict_or_key.get(key1)
            elif src1_dict_or_key == 'box_pr':
                full_seasonal_pr = processed_datasets_dict.get(f'{dataset_key_to_process}_pr_box_mean')
                var1_ts = DataProcessor.filter_by_season(DataProcessor.detrend_data(full_seasonal_pr), season)
            elif src1_dict_or_key == 'box_tas':
                full_seasonal_tas = processed_datasets_dict.get(f'{dataset_key_to_process}_tas_box_mean')
                var1_ts = DataProcessor.filter_by_season(DataProcessor.detrend_data(full_seasonal_tas), season)

            # Get Variable 2 (Jet Index from the appropriate bundle in jet_data_dict)
            # The jet bundle key indicates which variable (tas or pr) was *originally* paired with the jet
            # for creating that specific bundle entry in `analyze_jet_indices`.
            # We just need the 'jet' timeseries from that bundle.
            jet_bundle_key = f"{dataset_key_to_process}_{season_lower}_{jet_type}_{jet_impact_var}_data"
            jet_bundle = jet_data_dict.get(jet_bundle_key)
            var2_ts = None
            var2_label = f"{dataset_key_to_process} {season} Jet {jet_type.capitalize()} Index (detrended)"
            
            if jet_bundle and 'jet' in jet_bundle:
                var2_ts = jet_bundle['jet'] # This is already detrended as per analyze_jet_indices logic

            if var1_ts is None or var1_ts.size == 0 or var2_ts is None or var2_ts.size == 0:
                logging.debug(f"  Skipping correlation {var1_label} vs {var2_label}: Missing data for one or both series.")
                continue

            # Ensure 'season_year' is the coordinate for alignment
            common_years = np.intersect1d(var1_ts.season_year.values, var2_ts.season_year.values)
            if len(common_years) < 5: # Need a few years for correlation
                logging.debug(f"  Skipping correlation {var1_label} vs {var2_label}: Not enough common years ({len(common_years)}).")
                continue
            
            var1_common = var1_ts.sel(season_year=common_years).data # Numpy
            var2_common = var2_ts.sel(season_year=common_years).data # Numpy

            slope, intercept, r_val, p_val, stderr = StatsAnalyzer.calculate_regression(var2_common, var1_common) # X=Jet, Y=Var1

            if not np.isnan(r_val):
                corr_result = {
                    'common_years_count': len(common_years),
                    'values1_on_common_years': var1_common, # For plotting later
                    'values2_on_common_years': var2_common, # For plotting later
                    'variable1_name': var1_label,
                    'variable2_name': var2_label,
                    'r_value': r_val, 'p_value': p_val,
                    'slope': slope, 'intercept': intercept, 'stderr': stderr
                }
                # Store in the nested dictionary
                if out_main in ['winter', 'summer']: # Direct season keys like 'discharge_jet_speed'
                    all_correlations_for_dataset[out_main][out_sub] = corr_result
                else: # Nested under 'pr' or 'tas'
                    all_correlations_for_dataset[out_main][season_lower][out_sub] = corr_result
                logging.info(f"  Correlation ({dataset_key_to_process}, {season}): {var1_label} vs {var2_label.split('(')[0]} -> r={r_val:.2f}, p={p_val:.3f}")

        return all_correlations_for_dataset

    @staticmethod
    def analyze_jet_indices(processed_datasets_dict, dataset_key_to_process):
        """
        Calculate and analyze jet indices (speed and latitude) for a specific dataset
        for Winter and Summer. Also prepares data bundles for correlation analysis.
        This method assumes that `processed_datasets_dict` contains `_ua850_seasonal`,
        `_pr_box_mean`, and `_tas_box_mean` for the given `dataset_key_to_process`.
        """
        logging.info(f"Analyzing jet stream indices and preparing data bundles for {dataset_key_to_process}...")

        if dataset_key_to_process not in [Config.DATASET_20CRV3, Config.DATASET_ERA5]:
            logging.warning(f"Invalid dataset key for jet index analysis: {dataset_key_to_process}")
            return {}

        try:
            ua850_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_ua850_seasonal')
            pr_box_mean_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_pr_box_mean')
            tas_box_mean_full_seasonal = processed_datasets_dict.get(f'{dataset_key_to_process}_tas_box_mean')

            if ua850_full_seasonal is None or pr_box_mean_full_seasonal is None or tas_box_mean_full_seasonal is None:
                logging.error(f"Missing base seasonal data (U850, PR_box, or TAS_box) for {dataset_key_to_process}.")
                return {}

            jet_data_results = {} # To store all results for this dataset

            for season in ['Winter', 'Summer']:
                season_lower = season.lower()
                
                # Filter U850 for the season
                ua850_season_filtered = DataProcessor.filter_by_season(ua850_full_seasonal, season)
                if ua850_season_filtered is None or ua850_season_filtered.size == 0 :
                    logging.warning(f"No U850 data for {season} in {dataset_key_to_process}. Skipping jet indices for this season.")
                    continue

                # Calculate original (non-detrended) jet indices
                jet_speed_orig = JetStreamAnalyzer.calculate_jet_speed_index(ua850_season_filtered)
                jet_lat_orig = JetStreamAnalyzer.calculate_jet_lat_index(ua850_season_filtered)
                
                # Store original indices
                jet_data_results[f'{dataset_key_to_process}_jet_speed_{season_lower}_orig'] = jet_speed_orig
                jet_data_results[f'{dataset_key_to_process}_jet_lat_{season_lower}_orig'] = jet_lat_orig

                # Detrend jet indices
                jet_speed_detrended = DataProcessor.detrend_data(jet_speed_orig)
                jet_lat_detrended = DataProcessor.detrend_data(jet_lat_orig)

                # Store detrended indices (these are the primary ones for correlation)
                jet_data_results[f'{dataset_key_to_process}_jet_speed_{season_lower}_detrended'] = jet_speed_detrended
                jet_data_results[f'{dataset_key_to_process}_jet_lat_{season_lower}_detrended'] = jet_lat_detrended

                # Prepare data bundles for PR and TAS correlations with these jet indices
                for var_type, var_box_mean_full_seasonal in [('pr', pr_box_mean_full_seasonal), ('tas', tas_box_mean_full_seasonal)]:
                    var_box_season_filtered = DataProcessor.filter_by_season(var_box_mean_full_seasonal, season)
                    var_box_season_detrended = DataProcessor.detrend_data(var_box_season_filtered)

                    if var_box_season_detrended is None or var_box_season_detrended.size == 0:
                        logging.warning(f"No detrended box mean for {var_type} in {season}, {dataset_key_to_process}. Bundles involving it will be incomplete.")
                        continue
                    
                    # Bundle with Jet Speed
                    if jet_speed_detrended is not None and jet_speed_detrended.size > 0:
                        common_years_speed = np.intersect1d(var_box_season_detrended.season_year.values, jet_speed_detrended.season_year.values)
                        if len(common_years_speed) > 2:
                            jet_data_results[f'{dataset_key_to_process}_{season_lower}_speed_{var_type}_data'] = {
                                'years': common_years_speed,
                                var_type: var_box_season_detrended.sel(season_year=common_years_speed),
                                'jet': jet_speed_detrended.sel(season_year=common_years_speed),
                                f'{var_type}_orig': var_box_season_filtered.sel(season_year=common_years_speed), # Original (non-detrended) var
                                'jet_orig': jet_speed_orig.sel(season_year=common_years_speed) if jet_speed_orig is not None else None
                            }
                    # Bundle with Jet Latitude
                    if jet_lat_detrended is not None and jet_lat_detrended.size > 0:
                        common_years_lat = np.intersect1d(var_box_season_detrended.season_year.values, jet_lat_detrended.season_year.values)
                        if len(common_years_lat) > 2:
                            jet_data_results[f'{dataset_key_to_process}_{season_lower}_lat_{var_type}_data'] = {
                                'years': common_years_lat,
                                var_type: var_box_season_detrended.sel(season_year=common_years_lat),
                                'jet': jet_lat_detrended.sel(season_year=common_years_lat),
                                f'{var_type}_orig': var_box_season_filtered.sel(season_year=common_years_lat),
                                'jet_orig': jet_lat_orig.sel(season_year=common_years_lat) if jet_lat_orig is not None else None
                            }
            
            # Add general coordinates for plotting if not already implicitly handled
            # This part was from original code, but lons/lats are usually part of the map data itself.
            # For timeseries plots, this isn't directly needed. For maps, they come with the map data.
            # lons = ua850_full_seasonal.lon.values
            # lats = ua850_full_seasonal.lat.values
            # if lons.ndim == 1 and lats.ndim == 1:
            #     lons_mesh, lats_mesh = np.meshgrid(lons, lats)
            # else:
            #     lons_mesh, lats_mesh = lons, lats # Assume already 2D
            # jet_data_results[f'{dataset_key_to_process}_lons_mesh'] = lons_mesh
            # jet_data_results[f'{dataset_key_to_process}_lats_mesh'] = lats_mesh
            
            return jet_data_results
        except Exception as e:
            logging.error(f"Error in analyze_jet_indices for {dataset_key_to_process}: {e}")
            logging.error(traceback.format_exc())
            return {}

    @staticmethod
    def analyze_amo_jet_correlations(jet_data_for_dataset, amo_data_dict, season_to_analyze, rolling_window_size=15):
        """
        Analyzes correlations between AMO and Jet indices (speed and latitude) for a specific dataset and season,
        using rolling means on detrended data.

        Args:
            jet_data_for_dataset (dict): Jet data bundles for ONE dataset (e.g., output of analyze_jet_indices for ERA5).
                                         Keys like 'ERA5_winter_speed_tas_data', containing 'jet' (detrended) and 'jet_orig'.
            amo_data_dict (dict): AMO data, keys like 'amo_winter_detrended', 'amo_summer_detrended'.
            season_to_analyze (str): "Winter" or "Summer".
            rolling_window_size (int): Window size for rolling mean.

        Returns:
            dict: Correlation results for the dataset and season.
                  e.g., {'ERA5': {'speed': {'r_value': ..., 'p_value': ...}, 'latitude': {...}}}
                  Returns an empty dict if critical data is missing.
        """
        logging.info(f"Analyzing AMO-Jet correlations for {season_to_analyze} ({rolling_window_size}-year rolling means)...")

        correlations_for_season_and_dataset = {} # Results for this specific dataset and season
        season_lower = season_to_analyze.lower()

        # Get detrended AMO data for the season
        amo_seasonal_detrended = amo_data_dict.get(f'amo_{season_lower}_detrended')
        amo_seasonal_original = amo_data_dict.get(f'amo_{season_lower}') # For raw value plotting

        if amo_seasonal_detrended is None or amo_seasonal_original is None:
            logging.error(f"Missing detrended or original AMO data for {season_to_analyze}.")
            return {}
        
        # Ensure AMO detrended is 1D
        if amo_seasonal_detrended.ndim > 1:
            amo_seasonal_detrended = amo_seasonal_detrended.squeeze(drop=True)
        if amo_seasonal_detrended.ndim != 1:
            logging.error(f"Detrended AMO for {season_to_analyze} is not 1D after squeeze. Dims: {amo_seasonal_detrended.dims}")
            return {}

        # Smooth the detrended AMO series
        amo_smooth_detrended = StatsAnalyzer.calculate_rolling_mean(amo_seasonal_detrended, window_size=rolling_window_size)
        if amo_smooth_detrended is None:
            logging.error(f"Failed to smooth detrended AMO for {season_to_analyze}.")
            return {}

        # Determine the dataset key from the jet_data_for_dataset dictionary (e.g., "ERA5", "20CRv3")
        # This is a bit heuristic; assumes keys in jet_data_for_dataset start with the dataset name.
        dataset_prefix = None
        first_key = next(iter(jet_data_for_dataset), None)
        if first_key:
            if first_key.startswith(Config.DATASET_ERA5): dataset_prefix = Config.DATASET_ERA5
            elif first_key.startswith(Config.DATASET_20CRV3): dataset_prefix = Config.DATASET_20CRV3
        
        if not dataset_prefix:
            logging.error("Could not determine dataset prefix from jet_data_for_dataset keys.")
            return {}
        
        correlations_for_dataset_key = {} # To store {'speed': result, 'latitude': result}

        # Correlate with Jet Speed and Jet Latitude
        for jet_type in ['speed', 'lat']:
            # Construct the key to find the appropriate jet data bundle.
            # The bundle often includes '_tas_data' or '_pr_data' in its name from analyze_jet_indices.
            # We try both as the exact variable paired initially doesn't matter for the jet index itself.
            jet_bundle = None
            for var_suffix in ['tas', 'pr']: # Try common variable pairings
                jet_bundle_key_attempt = f"{dataset_prefix}_{season_lower}_{jet_type}_{var_suffix}_data"
                jet_bundle_candidate = jet_data_for_dataset.get(jet_bundle_key_attempt)
                if jet_bundle_candidate and 'jet' in jet_bundle_candidate and 'jet_orig' in jet_bundle_candidate:
                    jet_bundle = jet_bundle_candidate
                    break # Found a suitable bundle

            if not jet_bundle:
                logging.warning(f"No suitable jet data bundle found for {dataset_prefix} {season_lower} Jet {jet_type.capitalize()}. Skipping AMO correlation.")
                continue

            jet_index_detrended = jet_bundle['jet'] # This is the detrended 1D jet index
            jet_index_original = jet_bundle['jet_orig'] # Original non-detrended jet index

            if jet_index_detrended is None or jet_index_original is None:
                logging.warning(f"Missing detrended or original jet index for {dataset_prefix} {season_lower} Jet {jet_type.capitalize()}.")
                continue
            
            # Ensure jet index is 1D
            if jet_index_detrended.ndim > 1: jet_index_detrended = jet_index_detrended.squeeze(drop=True)
            if jet_index_detrended.ndim != 1:
                logging.error(f"Detrended Jet {jet_type.capitalize()} for {dataset_prefix} {season_lower} is not 1D. Dims: {jet_index_detrended.dims}")
                continue

            jet_index_smooth_detrended = StatsAnalyzer.calculate_rolling_mean(jet_index_detrended, window_size=rolling_window_size)
            if jet_index_smooth_detrended is None:
                logging.error(f"Failed to smooth detrended Jet {jet_type.capitalize()} for {dataset_prefix} {season_lower}.")
                continue
            
            # Find common years for the smoothed, detrended series
            common_years = np.intersect1d(jet_index_smooth_detrended.season_year.values, amo_smooth_detrended.season_year.values)
            if len(common_years) < rolling_window_size // 2 + 1: # Need some overlap
                logging.warning(f"Not enough common years ({len(common_years)}) for AMO vs Jet {jet_type.capitalize()} ({dataset_prefix}, {season_lower}).")
                continue

            amo_vals_for_corr = amo_smooth_detrended.sel(season_year=common_years).data
            jet_vals_for_corr = jet_index_smooth_detrended.sel(season_year=common_years).data

            slope, intercept, r_val, p_val, stderr = StatsAnalyzer.calculate_regression(amo_vals_for_corr, jet_vals_for_corr) # X=AMO, Y=Jet

            if not np.isnan(r_val):
                correlations_for_dataset_key[jet_type] = {
                    'common_years': common_years,
                    'amo_values_smooth_detrended': amo_vals_for_corr,
                    'jet_values_smooth_detrended': jet_vals_for_corr,
                    'amo_values_detrended_raw': amo_seasonal_detrended.sel(season_year=common_years).data,
                    'jet_values_detrended_raw': jet_index_detrended.sel(season_year=common_years).data,
                    'amo_values_original': amo_seasonal_original.sel(season_year=common_years).data,
                    'jet_values_original': jet_index_original.sel(season_year=common_years).data,
                    'r_value': r_val, 'p_value': p_val, 'slope': slope,
                    'intercept': intercept, 'stderr': stderr,
                    'rolling_window_size': rolling_window_size
                }
        
        if correlations_for_dataset_key: # If any correlations (speed or lat) were successful
            correlations_for_season_and_dataset[dataset_prefix] = correlations_for_dataset_key
            
        return correlations_for_season_and_dataset


    @staticmethod
    def calculate_cmip6_regression_maps(cmip6_model_data_loaded, historical_period=(1995, 2014)):
        """
        Calculate regressions between U850 and PR/TAS box indices using CMIP6 ensemble data
        for a specified historical period. Calculates Multi-Model Mean (MMM) first.

        Args:
            cmip6_model_data_loaded (dict): Dict from StorylineAnalyzer._load_and_preprocess_model_data,
                                           e.g., {model_name: {'ua': da, 'pr': da, 'tas': da}}
            historical_period (tuple): Start and end year for the historical regression.

        Returns:
            dict: Regression results, e.g., {'Winter': {'slopes_pr':..., 'p_values_pr':...}, ...}
        """
        logging.info(f"Calculating CMIP6 regression maps (U850 vs Box Indices) for historical period {historical_period}...")

        models_available = list(cmip6_model_data_loaded.keys())
        if not models_available:
            logging.info("  No CMIP6 model data provided for regression maps.")
            return {}

        hist_start, hist_end = historical_period
        model_seasonal_means_hist = {'ua': {}, 'pr': {}, 'tas': {}} # {var: {model: seasonal_da}}

        # 1. Calculate historical seasonal means for each model
        for model_name in models_available:
            logging.info(f"  Processing historical seasonal means for CMIP6 model: {model_name}...")
            model_has_all_vars = True
            current_model_seasonal_data = {}

            for var in ['ua', 'pr', 'tas']:
                if var not in cmip6_model_data_loaded.get(model_name, {}):
                    logging.warning(f"    Warning: Missing variable '{var}' for model {model_name}.")
                    model_has_all_vars = False; break
                
                monthly_data_full_ts = cmip6_model_data_loaded[model_name][var]
                try:
                    # Select only the historical period from the full timeseries
                    monthly_data_hist_period = monthly_data_full_ts.sel(
                        time=slice(str(hist_start), str(hist_end))
                    )
                    if monthly_data_hist_period.time.size == 0:
                        min_yr = monthly_data_full_ts.time.dt.year.min().item()
                        max_yr = monthly_data_full_ts.time.dt.year.max().item()
                        logging.info(f"    Skipping {var} for {model_name}: Does not cover historical period "
                                     f"{hist_start}-{hist_end} (Full data range: {min_yr}-{max_yr}).")
                        model_has_all_vars = False; break
                    
                    seasonal_full_hist = DataProcessor.assign_season_to_dataarray(monthly_data_hist_period)
                    seasonal_mean_hist = DataProcessor.calculate_seasonal_means(seasonal_full_hist)
                    if seasonal_mean_hist is not None:
                        current_model_seasonal_data[var] = seasonal_mean_hist.load() # Load into memory
                    else:
                        logging.error(f"    Failed to calculate historical seasonal mean for {model_name}/{var}.")
                        model_has_all_vars = False; break
                except Exception as e_hist_proc:
                    logging.error(f"    Error processing historical data for {model_name}/{var}: {e_hist_proc}")
                    model_has_all_vars = False; break
            
            if model_has_all_vars:
                for var in ['ua', 'pr', 'tas']:
                    model_seasonal_means_hist[var][model_name] = current_model_seasonal_data[var]
        
        # 2. Calculate Multi-Model Mean (MMM) for the historical seasonal data
        mmm_seasonal_hist = {}
        for var in ['ua', 'pr', 'tas']:
            datasets_to_combine_for_var = []
            model_names_for_var = [] # Um die Modellnamen parallel zu den Datensätzen zu halten

            # Sicherstellen, dass model_seasonal_means_hist[var] existiert und ein Dict ist
            if var in model_seasonal_means_hist and isinstance(model_seasonal_means_hist[var], dict):
                for model, ds_array in model_seasonal_means_hist[var].items():
                    if ds_array is not None:
                        # Sicherstellen, dass ds_array ein DataArray ist und einen Namen hat
                        if isinstance(ds_array, xr.DataArray):
                            name_for_da = ds_array.name if ds_array.name else var
                            if ds_array.name != name_for_da: # Nur umbenennen, wenn nötig
                                datasets_to_combine_for_var.append(ds_array.rename(name_for_da))
                            else:
                                datasets_to_combine_for_var.append(ds_array)
                            model_names_for_var.append(model)
                        else:
                            logging.warning(f"    Item for model {model}, var {var} is not an xarray.DataArray. Skipping.")
            else:
                logging.warning(f"    No data or incorrect data structure in model_seasonal_means_hist for variable '{var}'. Skipping MMM for this var.")
                mmm_seasonal_hist[var] = None
                continue

            # Konvertiere DataArrays zu Datasets und füge Modelldimension hinzu
            datasets_with_model_dim = []
            if datasets_to_combine_for_var: # Nur wenn es Daten zum Kombinieren gibt
                for i, da_model in enumerate(datasets_to_combine_for_var):
                    model_nm = model_names_for_var[i]
                    # Der Name des DataArrays wird zum Variablennamen im Dataset
                    datasets_with_model_dim.append(da_model.to_dataset().expand_dims(model=[model_nm]))


            if len(datasets_with_model_dim) >= 3: # Mindestens 3 Modelle für MMM
                # --- START DEBUG LOGGING PRE-COMBINATION ---
                logging.info(f"  DEBUG: Pre-combination check for variable '{var}':")
                for i_debug, ds_item_expanded_debug in enumerate(datasets_with_model_dim):
                    # Der Modellname ist jetzt eine Koordinate in ds_item_expanded_debug
                    model_name_debug = ds_item_expanded_debug.model.item() 
                    try:
                        # Zugriff auf die Variable im Dataset; der Name sollte 'var' sein (z.B. 'ua')
                        # oder der ursprüngliche Name des DataArrays, falls er nicht 'var' war.
                        # Wir nehmen an, dass es nur eine Datenvariable pro Dataset in datasets_with_model_dim gibt.
                        data_var_name_in_ds = list(ds_item_expanded_debug.data_vars.keys())[0]
                        data_array_to_check = ds_item_expanded_debug[data_var_name_in_ds]

                        if 'lon' in data_array_to_check.coords:
                            is_mono_debug = data_array_to_check.indexes['lon'].is_monotonic_increasing or data_array_to_check.indexes['lon'].is_monotonic_decreasing
                            first_few_lons = data_array_to_check.lon.values[:5] if data_array_to_check.lon.size > 0 else "N/A"
                            last_few_lons = data_array_to_check.lon.values[-5:] if data_array_to_check.lon.size > 5 else "N/A"
                            logging.info(f"    Model {model_name_debug} ({data_var_name_in_ds}) lon monotonic: {is_mono_debug}. Len: {data_array_to_check.lon.size}. Lons: {first_few_lons}...{last_few_lons}")
                            if not is_mono_debug and data_array_to_check.lon.size > 0:
                                logging.error(f"      NON-MONOTONIC LON for {model_name_debug} ({data_var_name_in_ds}) PRE-COMBINE. All lons ({data_array_to_check.lon.size}): {data_array_to_check.lon.values}")
                                diffs = np.diff(data_array_to_check.lon.values)
                                non_mono_indices = np.where(diffs <= 0)[0] 
                                if 'lon' in data_array_to_check.indexes and data_array_to_check.indexes['lon'].is_monotonic_decreasing:
                                    non_mono_indices = np.where(diffs >=0)[0]
                                if len(non_mono_indices) > 0:
                                    problem_idx = non_mono_indices[0]
                                    context_slice = slice(max(0, problem_idx - 2), problem_idx + 3)
                                    logging.error(f"        Problematic segment around index {problem_idx}: ...{data_array_to_check.lon.values[context_slice]}...")
                        else:
                            logging.info(f"    Model {model_name_debug} ({data_var_name_in_ds}) no 'lon' coord.")
                    except Exception as e_debug_log:
                        logging.error(f"    Error in pre-combination debug log for {model_name_debug} (var: {var}): {e_debug_log}")
                # --- END DEBUG LOGGING PRE-COMBINATION ---

                combined_ds_for_var = None 
                _join_method_used = "unknown" 

                try:
                    logging.info(f"    Attempting xr.combine_by_coords for variable '{var}' with join='inner'...")
                    combined_ds_for_var_raw_inner = xr.combine_by_coords(
                        datasets_with_model_dim,
                        compat='override', 
                        join='inner',      
                        combine_attrs='drop_conflicts',
                        coords='minimal' 
                    )
                    logging.info(f"    SUCCESS: xr.combine_by_coords (join='inner') for '{var}' completed.")
                    _join_method_used = "inner"
                    
                    if 'lon' in combined_ds_for_var_raw_inner.coords:
                        lon_index_valid_inner = True
                        try:
                            _ = combined_ds_for_var_raw_inner.indexes['lon']
                        except Exception as e_index_inner:
                            lon_index_valid_inner = False
                            logging.error(f"    Could not create/access 'lon' index for 'combined_ds_for_var_raw_inner' ({var}, join='inner'): {e_index_inner}")

                        if lon_index_valid_inner:
                            is_mono_raw_inner = (combined_ds_for_var_raw_inner.indexes['lon'].is_monotonic_increasing or 
                                                combined_ds_for_var_raw_inner.indexes['lon'].is_monotonic_decreasing)
                            logging.info(f"    'combined_ds_for_var_raw_inner' ({var}, join='inner') - lon monotonic: {is_mono_raw_inner}. Len: {combined_ds_for_var_raw_inner.lon.size}. Lons: {combined_ds_for_var_raw_inner.lon.values[:10]}...{combined_ds_for_var_raw_inner.lon.values[-10:]}")
                            if not is_mono_raw_inner and combined_ds_for_var_raw_inner.lon.size > 0:
                                    logging.error(f"      RAW COMBINED (inner) LON NON-MONOTONIC for '{var}'. All lons ({combined_ds_for_var_raw_inner.lon.size}): {combined_ds_for_var_raw_inner.lon.values}")
                    else:
                        logging.warning(f"    'combined_ds_for_var_raw_inner' ({var}, join='inner') - 'lon' coordinate MISSING.")
                    
                    combined_ds_for_var = combined_ds_for_var_raw_inner

                except Exception as e_combine_inner:
                    logging.error(f"    ERROR during xr.combine_by_coords (join='inner') for {var}: {e_combine_inner}")
                    logging.error(traceback.format_exc()) 
                    logging.info(f"    Attempting xr.combine_by_coords for variable '{var}' with join='outer' as fallback...")
                    try:
                        combined_ds_for_var_raw_outer = xr.combine_by_coords(
                            datasets_with_model_dim,
                            compat='override', 
                            join='outer', 
                            combine_attrs='drop_conflicts',
                            coords='minimal',
                            fill_value=np.nan 
                        )
                        logging.info(f"    SUCCESS: xr.combine_by_coords (join='outer') for '{var}' completed.")
                        _join_method_used = "outer"

                        if 'lon' in combined_ds_for_var_raw_outer.coords:
                            lon_index_valid_outer = True
                            try:
                                _ = combined_ds_for_var_raw_outer.indexes['lon']
                            except Exception as e_index_outer:
                                lon_index_valid_outer = False
                                logging.error(f"    Could not create/access 'lon' index for 'combined_ds_for_var_raw_outer' ({var}, join='outer'): {e_index_outer}")
                            
                            if lon_index_valid_outer:
                                is_mono_raw_outer = (combined_ds_for_var_raw_outer.indexes['lon'].is_monotonic_increasing or
                                                     combined_ds_for_var_raw_outer.indexes['lon'].is_monotonic_decreasing)
                                logging.info(f"    'combined_ds_for_var_raw_outer' ({var}, join='outer') - lon monotonic: {is_mono_raw_outer}. Len: {combined_ds_for_var_raw_outer.lon.size}. Lons: {combined_ds_for_var_raw_outer.lon.values[:10]}...{combined_ds_for_var_raw_outer.lon.values[-10:]}")
                                if not is_mono_raw_outer and combined_ds_for_var_raw_outer.lon.size > 0:
                                    logging.error(f"      RAW COMBINED (outer) LON NON-MONOTONIC for '{var}'. All lons ({combined_ds_for_var_raw_outer.lon.size}): {combined_ds_for_var_raw_outer.lon.values}")
                        else:
                                logging.warning(f"    'combined_ds_for_var_raw_outer' ({var}, join='outer') - 'lon' coordinate MISSING.")
                        combined_ds_for_var = combined_ds_for_var_raw_outer
                    except Exception as e_combine_outer:
                        logging.error(f"    ERROR during xr.combine_by_coords (join='outer') fallback for {var}: {e_combine_outer}")
                        logging.error(traceback.format_exc())
                        mmm_seasonal_hist[var] = None
                        continue 

                if combined_ds_for_var is None:
                    logging.error(f"    Combined_ds_for_var is None for variable '{var}' after all combine attempts. Skipping MMM for this variable.")
                    mmm_seasonal_hist[var] = None
                    continue

                if 'lon' in combined_ds_for_var.coords and 'lon' in combined_ds_for_var.dims:
                    is_monotonic_before_sort = False
                    lon_index_valid_presort = True
                    try:
                        _ = combined_ds_for_var.indexes['lon']
                    except Exception:
                        lon_index_valid_presort = False
                    
                    if lon_index_valid_presort:
                        idx_before_sort = combined_ds_for_var.indexes['lon']
                        is_monotonic_before_sort = idx_before_sort.is_monotonic_increasing or idx_before_sort.is_monotonic_decreasing
                    else:
                         logging.warning(f"    'lon' index for combined_ds_for_var ({var}, join='{_join_method_used}') could not be accessed before sort. Attempting sort.")

                    if not is_monotonic_before_sort:
                        logging.info(f"    Sorting combined dataset for variable '{var}' (from join='{_join_method_used}') by 'lon' as it's not monotonic (is_monotonic_before_sort: {is_monotonic_before_sort}).")
                        try:
                            combined_ds_for_var = combined_ds_for_var.sortby('lon')
                            logging.info(f"    Successfully sorted combined_ds_for_var by 'lon' for '{var}'.")
                        except Exception as e_sort:
                            logging.error(f"    ERROR during sortby('lon') for {var}: {e_sort}")
                            logging.error(traceback.format_exc())
                            mmm_seasonal_hist[var] = None
                            continue 
                    else:
                        logging.info(f"    Combined dataset for '{var}' (using join='{_join_method_used}') is already monotonic before explicit sortby('lon').")
                    
                    # --- START DEBUG LOGGING POST-SORT ---
                    if 'lon' in combined_ds_for_var.coords and 'lon' in combined_ds_for_var.dims: 
                        is_monotonic_after_sort = False
                        lon_index_valid_postsort = True
                        try:
                            _ = combined_ds_for_var.indexes['lon']
                        except Exception:
                            lon_index_valid_postsort = False

                        if lon_index_valid_postsort:
                            idx_after_sort = combined_ds_for_var.indexes['lon']
                            is_monotonic_after_sort = idx_after_sort.is_monotonic_increasing or idx_after_sort.is_monotonic_decreasing
                        else:
                            logging.error(f"    'lon' index for combined_ds_for_var ({var}, join='{_join_method_used}') could not be accessed even AFTER sort.")
                        
                        first_few_lons_sorted = combined_ds_for_var.lon.values[:5] if combined_ds_for_var.lon.size > 0 else "N/A"
                        last_few_lons_sorted = combined_ds_for_var.lon.values[-5:] if combined_ds_for_var.lon.size > 5 else "N/A"
                        logging.info(f"    Combined DS for '{var}' (join='{_join_method_used}') AFTER explicit sortby('lon'), lon monotonic: {is_monotonic_after_sort}. Len: {combined_ds_for_var.lon.size}. Lons: {first_few_lons_sorted}...{last_few_lons_sorted}")
                        if not is_monotonic_after_sort and combined_ds_for_var.lon.size > 0:
                            logging.error(f"      LON STILL NON-MONOTONIC for '{var}' (join='{_join_method_used}') POST-SORT. All lons ({combined_ds_for_var.lon.size}): {combined_ds_for_var.lon.values}")
                    # --- END DEBUG LOGGING POST-SORT ---
                elif 'lon' not in combined_ds_for_var.coords and combined_ds_for_var is not None : 
                        logging.warning(f"    Cannot sort combined_ds_for_var by 'lon' for '{var}' (join='{_join_method_used}') - 'lon' is not a coordinate.")
                
                if combined_ds_for_var is not None:
                    # Prüfen, ob 'lon' eine Koordinate ist UND ob sie (nach Sortierung) monoton ist, ODER ob 'lon' gar nicht existiert (Oberflächenvariablen ohne lon?)
                    condition_for_mmm = False
                    if 'lon' not in combined_ds_for_var.coords: # Falls lon gar nicht da ist (z.B. globale Mittelwerte)
                        condition_for_mmm = True
                        logging.info(f"    Proceeding with MMM calculation for '{var}' as 'lon' coordinate is not present in combined_ds_for_var.")
                    elif 'lon' in combined_ds_for_var.indexes:
                        if combined_ds_for_var.indexes['lon'].is_monotonic_increasing or combined_ds_for_var.indexes['lon'].is_monotonic_decreasing:
                            condition_for_mmm = True
                        else:
                            logging.error(f"    Skipping MMM calculation for {var} (join='{_join_method_used}') because 'lon' was not monotonic after sort attempts.")
                    else: # lon ist Koordinate, aber kein Index (sollte nicht passieren nach sortby, wenn lon Dimension ist)
                        logging.error(f"    Skipping MMM calculation for {var} (join='{_join_method_used}') because 'lon' is a coordinate but not an index after sort attempts.")

                    if condition_for_mmm:
                        try:
                            mmm_ds_for_var = combined_ds_for_var.mean(dim='model', skipna=True)
                            # Der Name der Datenvariable im mmm_ds_for_var sollte dem Namen der ursprünglichen DataArrays entsprechen
                            # oder 'var', wenn sie umbenannt wurden.
                            data_var_name_in_mmm = var 
                            if not list(mmm_ds_for_var.data_vars.keys()): # Keine Datenvariablen
                                 logging.error(f"    ERROR: MMM Dataset for {var} (using join='{_join_method_used}') has NO data variables after .mean(dim='model').")
                                 mmm_seasonal_hist[var] = None
                            elif data_var_name_in_mmm not in mmm_ds_for_var.data_vars:
                                # Wenn der erwartete Name nicht da ist, aber nur eine Variable existiert, nimm diese.
                                if len(mmm_ds_for_var.data_vars) == 1:
                                    actual_var_name_in_mmm = list(mmm_ds_for_var.data_vars.keys())[0]
                                    logging.warning(f"    Variable '{data_var_name_in_mmm}' not found in MMM, using existing '{actual_var_name_in_mmm}'.")
                                    mmm_seasonal_hist[var] = mmm_ds_for_var[actual_var_name_in_mmm]
                                else:
                                    logging.error(f"    ERROR: Variable '{data_var_name_in_mmm}' not found in combined MMM Dataset for {var} (using join='{_join_method_used}'). Available: {list(mmm_ds_for_var.data_vars.keys())}")
                                    mmm_seasonal_hist[var] = None
                            else:
                                mmm_seasonal_hist[var] = mmm_ds_for_var[data_var_name_in_mmm]
                            
                            if mmm_seasonal_hist.get(var) is not None:
                                logging.info(f"    MMM for historical {var} (using join='{_join_method_used}') calculated using {len(datasets_with_model_dim)} models.")

                        except Exception as e_mmm_calc:
                            logging.error(f"    ERROR calculating MMM for {var} (using join='{_join_method_used}') after combination/sort: {e_mmm_calc}")
                            logging.error(traceback.format_exc())
                            mmm_seasonal_hist[var] = None
                    else: # condition_for_mmm war False
                        mmm_seasonal_hist[var] = None 
                else: # combined_ds_for_var war None
                    mmm_seasonal_hist[var] = None
            else: # Weniger als 3 Modelle
                logging.info(f"    Skipping MMM for historical {var}: Not enough valid model data ({len(datasets_with_model_dim)} models).")
                mmm_seasonal_hist[var] = None

        # Check if all necessary MMMs were calculated
        if mmm_seasonal_hist.get('ua') is None or mmm_seasonal_hist.get('pr') is None or mmm_seasonal_hist.get('tas') is None:
            logging.error("  ERROR: Cannot calculate CMIP6 regression maps due to missing MMM historical data for ua, pr, or tas.")
            return {}

        # 3. Calculate Box Means for MMM PR/TAS
        box_coords_tuple = (Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)
        mmm_pr_box_mean_hist = DataProcessor.calculate_spatial_mean(mmm_seasonal_hist['pr'], *box_coords_tuple)
        mmm_tas_box_mean_hist = DataProcessor.calculate_spatial_mean(mmm_seasonal_hist['tas'], *box_coords_tuple)

        # 4. Detrend MMM indices and U850 field
        mmm_pr_box_detrended_hist = DataProcessor.detrend_data(mmm_pr_box_mean_hist)
        mmm_tas_box_detrended_hist = DataProcessor.detrend_data(mmm_tas_box_mean_hist)
        mmm_ua_detrended_hist = DataProcessor.detrend_data(mmm_seasonal_hist['ua'])

        if mmm_pr_box_detrended_hist is None or mmm_tas_box_detrended_hist is None or mmm_ua_detrended_hist is None:
            logging.error("  ERROR: Detrending of CMIP6 MMM historical data failed.")
            return {}

        # 5. Perform regression analysis using the detrended MMM historical data (similar to reanalysis part)
        cmip6_regression_results = {}
        seasons_to_analyze = ['Winter', 'Summer']

        for season in seasons_to_analyze:
            logging.info(f"  - Processing {season} CMIP6 MMM regression maps...")
            pr_index_season = DataProcessor.filter_by_season(mmm_pr_box_detrended_hist, season)
            tas_index_season = DataProcessor.filter_by_season(mmm_tas_box_detrended_hist, season)
            ua850_field_season_detrended = DataProcessor.filter_by_season(mmm_ua_detrended_hist, season)
            # Original (non-detrended) MMM U850 for mean contours
            ua850_field_season_original_mmm = DataProcessor.filter_by_season(mmm_seasonal_hist['ua'], season)
            ua850_mean_contour_original_mmm = ua850_field_season_original_mmm.mean(dim='season_year', skipna=True).data if ua850_field_season_original_mmm is not None else np.nan

            if not all(da is not None and da.size > 0 for da in [pr_index_season, tas_index_season, ua850_field_season_detrended]):
                logging.warning(f"    Skipping {season} for CMIP6 MMM: Missing or empty seasonal data after filtering/detrending.")
                continue
            
            min_years_for_regression = 5 # Or Config.GWL_YEARS_WINDOW // 2
            if pr_index_season.season_year.size < min_years_for_regression or \
               tas_index_season.season_year.size < min_years_for_regression or \
               ua850_field_season_detrended.season_year.size < min_years_for_regression:
                logging.warning(f"    Skipping {season} for CMIP6 MMM: Not enough time points for regression "
                                f"(PR: {pr_index_season.season_year.size}, TAS: {tas_index_season.season_year.size}, U850: {ua850_field_season_detrended.season_year.size}).")
                continue
            
            # Regress PR vs U850 field
            common_years_pr = np.intersect1d(pr_index_season.season_year.values, ua850_field_season_detrended.season_year.values)
            pr_idx_vals = pr_index_season.sel(season_year=common_years_pr).data
            ua_common_pr = ua850_field_season_detrended.sel(season_year=common_years_pr)
            
            # Parallel processing setup
            lat_dim = next((d for d in ua_common_pr.dims if 'lat' in d.lower()), 'lat')
            lon_dim = next((d for d in ua_common_pr.dims if 'lon' in d.lower()), 'lon')
            time_dim = 'season_year'
            ua_time_first_pr = ua_common_pr.transpose(time_dim, lat_dim, lon_dim).data
            num_lats_pr, num_lons_pr = ua_common_pr.sizes[lat_dim], ua_common_pr.sizes[lon_dim]
            slopes_pr = np.full((num_lats_pr, num_lons_pr), np.nan)
            p_vals_pr = np.full((num_lats_pr, num_lons_pr), np.nan)
            lat_idx_g_pr = list(range(num_lats_pr)); lon_idx_g_pr = list(range(num_lons_pr))

            with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
                futures_pr = []
                chunk_size = max(1, num_lats_pr // (Config.N_PROCESSES * 2))
                for i_start in range(0, num_lats_pr, chunk_size):
                    lat_chunk_idx = lat_idx_g_pr[i_start : min(i_start + chunk_size, num_lats_pr)]
                    if not lat_chunk_idx: continue
                    ua_chunk = ua_time_first_pr[:, lat_chunk_idx[0]:lat_chunk_idx[-1]+1, :]
                    futures_pr.append(executor.submit(AdvancedAnalyzer._parallel_regression_chunk,
                                                     pr_idx_vals, ua_chunk, lat_chunk_idx, lon_idx_g_pr, lat_dim, lon_dim))
                for future in as_completed(futures_pr):
                    s_c, p_c, lats_p, _ = future.result()
                    for i_c, lat_idx in enumerate(lats_p): slopes_pr[lat_idx, :] = s_c[i_c, :]; p_vals_pr[lat_idx, :] = p_c[i_c, :]
            
            # Regress TAS vs U850 field (similar logic)
            common_years_tas = np.intersect1d(tas_index_season.season_year.values, ua850_field_season_detrended.season_year.values)
            tas_idx_vals = tas_index_season.sel(season_year=common_years_tas).data
            ua_common_tas = ua850_field_season_detrended.sel(season_year=common_years_tas)
            ua_time_first_tas = ua_common_tas.transpose(time_dim, lat_dim, lon_dim).data
            num_lats_tas, num_lons_tas = ua_common_tas.sizes[lat_dim], ua_common_tas.sizes[lon_dim]
            slopes_tas = np.full((num_lats_tas, num_lons_tas), np.nan)
            p_vals_tas = np.full((num_lats_tas, num_lons_tas), np.nan)
            lat_idx_g_tas = list(range(num_lats_tas)); lon_idx_g_tas = list(range(num_lons_tas))

            with ProcessPoolExecutor(max_workers=Config.N_PROCESSES) as executor:
                futures_tas = []
                chunk_size_tas = max(1, num_lats_tas // (Config.N_PROCESSES * 2))
                for i_start in range(0, num_lats_tas, chunk_size_tas):
                    lat_chunk_idx_tas = lat_idx_g_tas[i_start : min(i_start + chunk_size_tas, num_lats_tas)]
                    if not lat_chunk_idx_tas: continue
                    ua_chunk_tas = ua_time_first_tas[:, lat_chunk_idx_tas[0]:lat_chunk_idx_tas[-1]+1, :]
                    futures_tas.append(executor.submit(AdvancedAnalyzer._parallel_regression_chunk,
                                                       tas_idx_vals, ua_chunk_tas, lat_chunk_idx_tas, lon_idx_g_tas, lat_dim, lon_dim))
                for future in as_completed(futures_tas):
                    s_c, p_c, lats_p, _ = future.result()
                    for i_c, lat_idx in enumerate(lats_p): slopes_tas[lat_idx, :] = s_c[i_c, :]; p_vals_tas[lat_idx, :] = p_c[i_c, :]

            lons_p = ua_common_pr[lon_dim].values; lats_p = ua_common_pr[lat_dim].values
            if lons_p.ndim == 1: lons_p, lats_p = np.meshgrid(lons_p, lats_p)

            cmip6_regression_results[season] = {
                'slopes_pr': slopes_pr, 'p_values_pr': p_vals_pr,
                'slopes_tas': slopes_tas, 'p_values_tas': p_vals_tas,
                'ua850_mean_for_contours': ua850_mean_contour_original_mmm,
                'lons': lons_p, 'lats': lats_p
            }
        logging.info("Calculation of CMIP6 MMM regression maps completed.")
        return cmip6_regression_results

    @staticmethod
    def calculate_historical_slopes_comparison(beta_obs_slopes_from_reanalysis,
                                             cmip6_full_timeseries_data, # Output from StorylineAnalyzer._load_and_preprocess_model_data
                                             jet_data_reanalysis, # For reference, not directly used unless beta_obs empty
                                             historical_period_for_cmip6=(1981, 2010)):
        """
        Calculates regression slopes (jet vs impact variable) for each CMIP6 model
        over a specified historical period. This is to compare with beta_obs from reanalysis.

        Args:
            beta_obs_slopes_from_reanalysis (dict): Slopes from reanalysis (e.g., ERA5).
                                                    Keys like 'DJF_JetSpeed_vs_tas'.
            cmip6_full_timeseries_data (dict): Dict of {model: {var: DataArray_full_timeseries}}.
            jet_data_reanalysis (dict): Reanalysis jet data (mostly for context or if beta_obs is missing).
            historical_period_for_cmip6 (tuple): Start and end year for CMIP6 slope calculation.

        Returns:
            dict: {beta_key: [list_of_cmip6_model_slopes]}
        """
        logging.info(f"Calculating historical slopes for CMIP6 models ({historical_period_for_cmip6})...")
        cmip6_historical_slopes_results = {key: [] for key in beta_obs_slopes_from_reanalysis}
        hist_start, hist_end = historical_period_for_cmip6
        box_coords_tuple = (Config.BOX_LAT_MIN, Config.BOX_LAT_MAX, Config.BOX_LON_MIN, Config.BOX_LON_MAX)

        for model_name, model_vars_data in cmip6_full_timeseries_data.items():
            logging.info(f"  Processing historical slopes for CMIP6 model: {model_name}")
            try:
                # 1. Get full timeseries for ua, pr, tas for the model
                ua_monthly_full = model_vars_data.get('ua')
                pr_monthly_full = model_vars_data.get('pr')
                tas_monthly_full = model_vars_data.get('tas')

                if not all([ua_monthly_full, pr_monthly_full, tas_monthly_full]):
                    logging.info(f"    Skipping {model_name}: Missing ua, pr, or tas full timeseries data.")
                    continue

                # 2. Select the historical period
                ua_hist_period = ua_monthly_full.sel(time=slice(str(hist_start), str(hist_end)))
                pr_hist_period = pr_monthly_full.sel(time=slice(str(hist_start), str(hist_end)))
                tas_hist_period = tas_monthly_full.sel(time=slice(str(hist_start), str(hist_end)))

                if not all(da.time.size > 0 for da in [ua_hist_period, pr_hist_period, tas_hist_period]):
                    logging.info(f"    Skipping {model_name}: Data does not cover historical period {hist_start}-{hist_end} sufficiently after selection.")
                    continue
                
                # 3. Calculate seasonal means for the historical period
                ua_seas_full_hist = DataProcessor.assign_season_to_dataarray(ua_hist_period)
                pr_seas_full_hist = DataProcessor.assign_season_to_dataarray(pr_hist_period)
                tas_seas_full_hist = DataProcessor.assign_season_to_dataarray(tas_hist_period)

                ua_seas_mean_hist = DataProcessor.calculate_seasonal_means(ua_seas_full_hist)
                pr_seas_mean_hist = DataProcessor.calculate_seasonal_means(pr_seas_full_hist)
                tas_seas_mean_hist = DataProcessor.calculate_seasonal_means(tas_seas_full_hist)

                if not all([ua_seas_mean_hist, pr_seas_mean_hist, tas_seas_mean_hist]):
                    logging.info(f"    Skipping {model_name}: Failed to calculate seasonal means for historical period.")
                    continue

                # 4. Calculate Box Means (for PR, TAS) and Jet Indices (for UA) for the historical period
                pr_box_mean_hist = DataProcessor.calculate_spatial_mean(pr_seas_mean_hist, *box_coords_tuple)
                tas_box_mean_hist = DataProcessor.calculate_spatial_mean(tas_seas_mean_hist, *box_coords_tuple)
                
                ua_hist_winter = DataProcessor.filter_by_season(ua_seas_mean_hist, 'Winter')
                ua_hist_summer = DataProcessor.filter_by_season(ua_seas_mean_hist, 'Summer')
                
                model_jet_indices_hist = {
                    'DJF_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(ua_hist_winter),
                    'JJA_JetSpeed': JetStreamAnalyzer.calculate_jet_speed_index(ua_hist_summer),
                    'DJF_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(ua_hist_winter),
                    'JJA_JetLat': JetStreamAnalyzer.calculate_jet_lat_index(ua_hist_summer)
                }
                model_impact_vars_hist = {
                    'DJF_pr': DataProcessor.filter_by_season(pr_box_mean_hist, 'Winter') if pr_box_mean_hist is not None else None,
                    'JJA_pr': DataProcessor.filter_by_season(pr_box_mean_hist, 'Summer') if pr_box_mean_hist is not None else None,
                    'DJF_tas': DataProcessor.filter_by_season(tas_box_mean_hist, 'Winter') if tas_box_mean_hist is not None else None,
                    'JJA_tas': DataProcessor.filter_by_season(tas_box_mean_hist, 'Summer') if tas_box_mean_hist is not None else None
                }

                # 5. Detrend and Calculate Slopes for each relevant pair (matching beta_obs_slopes keys)
                for beta_key in beta_obs_slopes_from_reanalysis: # e.g., 'DJF_JetSpeed_vs_tas'
                    key_parts = beta_key.split('_vs_') # ['DJF_JetSpeed', 'tas']
                    jet_index_name_in_dict = key_parts[0]       # 'DJF_JetSpeed'
                    season_short = jet_index_name_in_dict.split('_')[0] # 'DJF'
                    impact_var_short_name = key_parts[1]      # 'tas'
                    impact_var_name_in_dict = f"{season_short}_{impact_var_short_name}" # 'DJF_tas'

                    logging.debug(f"      Processing beta_key for CMIP6 slope: {beta_key} (Jet: {jet_index_name_in_dict}, Impact: {impact_var_name_in_dict}) for {model_name}")

                    jet_ts_hist = model_jet_indices_hist.get(jet_index_name_in_dict)
                    impact_ts_hist = model_impact_vars_hist.get(impact_var_name_in_dict)

                    if jet_ts_hist is not None and impact_ts_hist is not None and \
                       jet_ts_hist.size > 0 and impact_ts_hist.size > 0:
                        
                        jet_ts_detrended = DataProcessor.detrend_data(jet_ts_hist)
                        impact_ts_detrended = DataProcessor.detrend_data(impact_ts_hist)

                        if jet_ts_detrended is None or impact_ts_detrended is None:
                            logging.warning(f"        {model_name} - {beta_key}: Detrending failed for historical jet or impact TS.")
                            continue

                        common_years = np.intersect1d(jet_ts_detrended.season_year.values, impact_ts_detrended.season_year.values)
                        if len(common_years) >= 5: # Need a few years for regression
                            jet_vals = jet_ts_detrended.sel(season_year=common_years).data
                            impact_vals = impact_ts_detrended.sel(season_year=common_years).data
                            
                            slope, _, _, _, _ = StatsAnalyzer.calculate_regression(jet_vals, impact_vals) # X=Jet, Y=Impact

                            if slope is not None and not np.isnan(slope):
                                cmip6_historical_slopes_results[beta_key].append(slope)
                                logging.info(f"        SUCCESS: Appended CMIP6 historical slope {slope:.3f} for {model_name} - {beta_key}")
                            else:
                                logging.warning(f"        {model_name} - {beta_key}: CMIP6 historical slope is None or NaN after regression.")
                        else:
                            logging.warning(f"        {model_name} - {beta_key}: Not enough common historical years ({len(common_years)}) after detrending for CMIP6 slope.")
                    else:
                        logging.warning(f"        {model_name} - {beta_key}: Jet or impact time series for CMIP6 historical period is None or empty.")
            except Exception as e_model_hist_slope:
                logging.error(f"    Error processing historical slopes for CMIP6 model {model_name}: {e_model_hist_slope}")
                logging.error(traceback.format_exc())

        logging.info("Finished calculating CMIP6 historical slopes for comparison.")
        for key, slopes_list in cmip6_historical_slopes_results.items():
            logging.info(f"  CMIP6 Historical slopes calculated for {key}: {len(slopes_list)} models")
        return cmip6_historical_slopes_results