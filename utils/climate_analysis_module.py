#!/usr/bin/env python3
"""
This module contains the main ClimateAnalysis class that orchestrates
the overall climate data analysis workflow.
"""
import logging
import pandas as pd
import numpy as np
import xarray as xr # Used by some methods if they return xarray objects directly
from functools import lru_cache
import json
import os # For path joining, used by Config
import traceback # For error logging

# Relative imports for other utility modules
from config_setup import Config
from data_utils import DataProcessor
from stats_utils import StatsAnalyzer # May be used by AdvancedAnalyzer or Visualizer
from jet_utils import JetStreamAnalyzer
from advanced_analysis_utils import AdvancedAnalyzer
from storyline_utils import StorylineAnalyzer
from plotting_utils import Visualizer


class ClimateAnalysis:
    """
    Main class to manage and execute the climate data analysis workflow.
    """

    def __init__(self, config_instance: Config):
        """
        Initializes the ClimateAnalysis class with a configuration instance.

        Args:
            config_instance (Config): An instance of the Config class.
        """
        self.config = config_instance
        # Instances of other analyzers can be created here if needed throughout the class
        # or created on-the-fly within methods.
        # For methods like process_discharge_data, static calls to DataProcessor are used.
        # The run_analysis method will instantiate or use static methods from other analyzers.

    @staticmethod
    def process_discharge_data(file_path: str):
        """
        Process discharge data from an Excel file and compute metrics.

        Args:
            file_path (str): Path to the Excel file containing discharge data.

        Returns:
            dict: A dictionary containing processed seasonal discharge metrics (detrended).
                  Returns an empty dict on error.
        """
        logging.info(f"Processing discharge data from {file_path}...")
        try:
            # Expected columns: 'year', 'month', and a column for discharge (e.g., 'Wien')
            # The original code used specific column names. Ensure these match your Excel file.
            # usecols='A,B,C,H' implies specific columns by letter. It's safer to use names if possible.
            # Assuming column H is 'Wien' as in original.
            excel_data = pd.read_excel(file_path, index_col=None, na_values=['NA'])
            
            # Adapt column names if necessary. Original used: 'year', 'month', 'Wien'
            # Check if typical column names exist
            year_col = next((col for col in ['year', 'Year', 'JAHR'] if col in excel_data.columns), None)
            month_col = next((col for col in ['month', 'Month', 'MONAT'] if col in excel_data.columns), None)
            discharge_col_name = next((col for col in ['Wien', 'Discharge', 'Value'] if col in excel_data.columns), None)

            if not all([year_col, month_col, discharge_col_name]):
                logging.error(f"Required columns ('year', 'month', discharge value) not found in {file_path}. Found: {excel_data.columns.tolist()}")
                return {}

            df = pd.DataFrame({
                'year': excel_data[year_col],
                'month': excel_data[month_col],
                'discharge': excel_data[discharge_col_name]
            }).dropna(subset=['year', 'month', 'discharge']) # Drop rows where essential data is missing

            if df.empty:
                logging.warning("Discharge data is empty after initial load and NaN drop.")
                return {}

            # Calculate high and low flow thresholds (e.g., 90th and 10th percentiles)
            high_flow_threshold = np.percentile(df['discharge'], 90)
            low_flow_threshold = np.percentile(df['discharge'], 10)
            df['is_high_flow'] = df['discharge'] > high_flow_threshold
            df['is_low_flow'] = df['discharge'] < low_flow_threshold

            # Create extreme flow index: +1 for high flow, -1 for low flow, 0 for normal
            df['extreme_flow_index'] = np.select(
                [df['is_high_flow'], df['is_low_flow']],
                [1, -1],
                default=0
            )

            # Assign season and season_year (vectorized)
            # This helper is defined in DataProcessor or can be a static method here too.
            # For this module, let's assume it's available via DataProcessor.
            df_seasonal_assignments = DataProcessor._assign_season_to_df(df[['year', 'month']].copy()) # Pass a copy
            df = df.merge(df_seasonal_assignments, on=['year', 'month'], how='left').dropna(subset=['season', 'season_year'])


            if df.empty:
                logging.warning("Discharge data is empty after season assignment.")
                return {}

            # Aggregate by season and season_year
            seasonal_aggregated_data = df.groupby(['season_year', 'season']).agg(
                mean_discharge=('discharge', 'mean'),
                high_flow_days=('is_high_flow', 'sum'), # Count of high flow days
                low_flow_days=('is_low_flow', 'sum'),   # Count of low flow days
                mean_extreme_flow_index=('extreme_flow_index', 'mean'),
                month_count=('month', 'count') # Number of months in the season (typically 3)
            ).reset_index()

            # Calculate frequencies
            seasonal_aggregated_data['high_flow_freq'] = seasonal_aggregated_data['high_flow_days'] / seasonal_aggregated_data['month_count']
            seasonal_aggregated_data['low_flow_freq'] = seasonal_aggregated_data['low_flow_days'] / seasonal_aggregated_data['month_count']

            processed_results = {}
            for season_name in ['Winter', 'Spring', 'Summer', 'Autumn']:
                season_subset = seasonal_aggregated_data[seasonal_aggregated_data['season'] == season_name]
                if season_subset.empty:
                    continue

                season_key_lower = season_name.lower()
                metrics_to_process = {
                    'discharge': 'mean_discharge',
                    'high_flow_freq': 'high_flow_freq',
                    'low_flow_freq': 'low_flow_freq',
                    'extreme_flow': 'mean_extreme_flow_index' # Using the mean of the +/-1/0 index
                }
                for output_metric_name, df_col_name in metrics_to_process.items():
                    result_key = f'{season_key_lower}_{output_metric_name}'
                    
                    # Create xarray DataArray for consistency and detrending
                    data_for_xr = season_subset[df_col_name].values
                    season_years_for_xr = season_subset['season_year'].values
                    
                    ts_data_array = xr.DataArray(
                        data=data_for_xr,
                        coords={'season_year': season_years_for_xr},
                        dims='season_year',
                        name=result_key
                    ).sortby('season_year') # Ensure sorted by year

                    # Detrend the time series
                    processed_results[result_key] = DataProcessor.detrend_data(ts_data_array)
            
            logging.info("Discharge data processing completed.")
            return processed_results
        except Exception as e:
            logging.error(f"Error processing discharge data from {file_path}: {e}")
            logging.error(traceback.format_exc())
            return {}

    @staticmethod
    @lru_cache(maxsize=1) # Cache results to avoid reprocessing if called multiple times
    def process_20crv3_data(config: Config):
        """Load and process 20CRv3 climate data."""
        logging.info("Loading and processing 20CRv3 climate data...")
        datasets_20crv3 = {}
        try:
            # Process climate data files using DataProcessor
            datasets_20crv3['20CRv3_pr_monthly'] = DataProcessor.process_ncfile(config.PR_FILE_20CRV3, 'pr')
            datasets_20crv3['20CRv3_tas_monthly'] = DataProcessor.process_ncfile(config.TAS_FILE_20CRV3, 'tas')
            datasets_20crv3['20CRv3_ua850_monthly'] = DataProcessor.process_ncfile(config.UA_FILE_20CRV3, 'ua', var_out_name='ua', level_to_select=config.WIND_LEVEL)

            # Check if any of the processed datasets are None (indicating a failure for that specific file)
            if any(value is None for value in datasets_20crv3.values()):
                logging.error("Error: One or more 20CRv3 data files could not be processed (resulted in None).")
                # Optional: Log which specific keys are None
                for key, value in datasets_20crv3.items():
                    if value is None:
                        logging.error(f"  - {key} processing failed.")
                return {}
            # You might also want to check if any DataArray is empty (e.g., .size == 0) if that's a failure condition
            if any(value.size == 0 for value in datasets_20crv3.values() if hasattr(value, 'size')):
                logging.error("Error: One or more processed 20CRv3 DataArrays are empty.")
                return {}

            logging.info("Calculating anomalies for 20CRv3...")
            # Anomalies are calculated on monthly data before seasonal aggregation
            pr_anom = DataProcessor.calculate_anomalies(datasets_20crv3['20CRv3_pr_monthly'], as_percentage=True)
            tas_anom = DataProcessor.calculate_anomalies(datasets_20crv3['20CRv3_tas_monthly'], as_percentage=False)
            # U850 anomalies often not taken, but original seasonal means used.
            
            logging.info("Assigning seasons and calculating seasonal means for 20CRv3...")
            datasets_20crv3['20CRv3_pr_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(pr_anom))
            datasets_20crv3['20CRv3_tas_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(tas_anom))
            datasets_20crv3['20CRv3_ua850_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(datasets_20crv3['20CRv3_ua850_monthly']))
            
            # Filter for specific seasons if needed for direct use (e.g., winter U850 for jet calcs)
            # This might be redundant if AdvancedAnalyzer.analyze_jet_indices takes full seasonal data
            # datasets_20crv3['20CRv3_ua850_winter'] = DataProcessor.filter_by_season(datasets_20crv3['20CRv3_ua850_seasonal'], "Winter")
            # datasets_20crv3['20CRv3_ua850_summer'] = DataProcessor.filter_by_season(datasets_20crv3['20CRv3_ua850_seasonal'], "Summer")

            logging.info("Calculating spatial means for 20CRv3 box region...")
            # Spatial means are typically on seasonal data
            datasets_20crv3['20CRv3_pr_box_mean'] = DataProcessor.calculate_spatial_mean(
                datasets_20crv3['20CRv3_pr_seasonal'],
                config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX
            )
            datasets_20crv3['20CRv3_tas_box_mean'] = DataProcessor.calculate_spatial_mean(
                datasets_20crv3['20CRv3_tas_seasonal'],
                config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX
            )
            logging.info("20CRv3 data processing completed.")
            return datasets_20crv3
        except Exception as e:
            logging.error(f"Error during 20CRv3 data processing pipeline: {e}")
            logging.error(traceback.format_exc())
            return {}

    @staticmethod
    @lru_cache(maxsize=1)
    def process_era5_data(config: Config):
        """Load and process ERA5 climate data."""
        logging.info("Loading and processing ERA5 climate data...")
        datasets_era5 = {}
        try:
            datasets_era5['ERA5_pr_monthly'] = DataProcessor.process_era5_file(config.ERA5_PR_FILE, 'pr')
            datasets_era5['ERA5_tas_monthly'] = DataProcessor.process_era5_file(config.ERA5_TAS_FILE, 'tas')
            datasets_era5['ERA5_ua850_monthly'] = DataProcessor.process_era5_file(config.ERA5_UA_FILE, 'u', var_out_name='ua', level_to_select=config.WIND_LEVEL)

            if any(value is None for value in datasets_era5.values()):
                logging.error("Error: One or more ERA5 data files could not be processed (resulted in None).")
                for key, value in datasets_era5.items():
                    if value is None:
                        logging.error(f"  - {key} processing failed.")
                return {}
            if any(value.size == 0 for value in datasets_era5.values() if hasattr(value, 'size')):
                logging.error("Error: One or more processed ERA5 DataArrays are empty.")
                return {}

            logging.info("Calculating anomalies for ERA5...")
            pr_anom_era5 = DataProcessor.calculate_anomalies(datasets_era5['ERA5_pr_monthly'], as_percentage=True)
            tas_anom_era5 = DataProcessor.calculate_anomalies(datasets_era5['ERA5_tas_monthly'], as_percentage=False)

            logging.info("Assigning seasons and calculating seasonal means for ERA5...")
            datasets_era5['ERA5_pr_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(pr_anom_era5))
            datasets_era5['ERA5_tas_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(tas_anom_era5))
            datasets_era5['ERA5_ua850_seasonal'] = DataProcessor.calculate_seasonal_means(DataProcessor.assign_season_to_dataarray(datasets_era5['ERA5_ua850_monthly']))
            
            logging.info("Calculating spatial means for ERA5 box region...")
            datasets_era5['ERA5_pr_box_mean'] = DataProcessor.calculate_spatial_mean(
                datasets_era5['ERA5_pr_seasonal'],
                config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX
            )
            datasets_era5['ERA5_tas_box_mean'] = DataProcessor.calculate_spatial_mean(
                datasets_era5['ERA5_tas_seasonal'],
                config.BOX_LAT_MIN, config.BOX_LAT_MAX, config.BOX_LON_MIN, config.BOX_LON_MAX
            )
            logging.info("ERA5 data processing completed.")
            return datasets_era5
        except Exception as e:
            logging.error(f"Error during ERA5 data processing pipeline: {e}")
            logging.error(traceback.format_exc())
            return {}

    @staticmethod
    def load_amo_index(file_path: str):
        """Loads and processes the AMO index from a CSV file for Winter and Summer."""
        logging.info(f"Loading AMO index from {file_path}...")
        try:
            amo_df_raw = pd.read_csv(file_path, sep=",", header=0) # Assuming standard CSV with header
            amo_df_raw.replace(-999, np.nan, inplace=True) # Replace common missing value indicator

            # Melt to long format: Year, Month_Name, AMO_Value
            amo_long_format = amo_df_raw.melt(id_vars="Year", var_name="Month_Name", value_name="AMO_Value")

            # Map month names to numbers
            month_name_to_number_map = {
                "Jan":1, "Feb":2, "Mar":3, "Apr":4, "May":5, "Jun":6,
                "Jul":7, "Aug":8, "Sep":9, "Oct":10, "Nov":11, "Dec":12
            }
            amo_long_format["month"] = amo_long_format["Month_Name"].map(month_name_to_number_map)
            amo_long_format.rename(columns={'Year': 'year'}, inplace=True) # Ensure 'year' column
            amo_long_format.dropna(subset=['month', 'AMO_Value'], inplace=True) # Drop if month mapping failed or AMO is NaN

            # Assign season and season_year
            amo_seasonal_assignments = DataProcessor._assign_season_to_df(amo_long_format[['year', 'month']].copy())
            amo_with_seasons = amo_long_format.merge(amo_seasonal_assignments, on=['year', 'month'], how='left').dropna(subset=['season', 'season_year'])

            amo_results = {}
            # Calculate annual mean AMO
            amo_annual_mean = amo_with_seasons.groupby("year")["AMO_Value"].mean().reset_index()
            amo_results['amo_annual_raw'] = xr.DataArray.from_series(amo_annual_mean.set_index('year')['AMO_Value'])
            
            # Calculate seasonal AMO means (Winter DJF, Summer JJA)
            for season_name_iter in ['Winter', 'Summer']:
                season_data_subset = amo_with_seasons[amo_with_seasons['season'] == season_name_iter]
                if season_data_subset.empty: continue

                seasonal_mean_amo = season_data_subset.groupby("season_year")["AMO_Value"].mean()
                
                season_amo_da = xr.DataArray(
                    data=seasonal_mean_amo.values,
                    coords={'season_year': seasonal_mean_amo.index.values.astype(int)},
                    dims='season_year',
                    name=f'AMO_{season_name_iter.lower()}'
                ).sortby('season_year')
                
                amo_results[f'amo_{season_name_iter.lower()}'] = season_amo_da # Original seasonal
                amo_results[f'amo_{season_name_iter.lower()}_detrended'] = DataProcessor.detrend_data(season_amo_da)
            
            logging.info("AMO index processing completed.")
            return amo_results
        except FileNotFoundError:
            logging.error(f"AMO index file not found at {file_path}")
            return None
        except Exception as e:
            logging.error(f"Error loading or processing AMO index: {e}")
            logging.error(traceback.format_exc())
            return None

    def run_full_analysis(self):
        """
        Executes the full climate analysis workflow, including Reanalysis, CMIP6, and Storylines.
        This method is a high-level orchestrator.
        """
        logging.info("\n====================================================================")
        logging.info("=== STARTING FULL CLIMATE ANALYSIS WORKFLOW ===")
        logging.info(f"=== Using Config: Plot Dir='{self.config.PLOT_DIR}', Results Dir='{self.config.RESULTS_DIR}' ===")
        logging.info("====================================================================\n")

        # Ensure output directories exist
        self.config.ensure_dir_exists(self.config.PLOT_DIR)
        self.config.ensure_dir_exists(self.config.RESULTS_DIR)

        # --- PART 1: REANALYSIS DATA PROCESSING & ANALYSIS ---
        logging.info("\n--- PHASE 1: REANALYSIS DATA PROCESSING & ANALYSIS ---")
        datasets_20crv3 = self.process_20crv3_data(self.config)
        datasets_era5 = self.process_era5_data(self.config)
        
        if not datasets_20crv3 or not datasets_era5:
            logging.critical("CRITICAL: Failed to process one or both reanalysis datasets. Halting analysis.")
            return {"error": "Reanalysis data processing failed."}
        
        # Combine all processed reanalysis data into one dictionary for easier access
        all_reanalysis_data_processed = {**datasets_20crv3, **datasets_era5}

        discharge_data_processed = self.process_discharge_data(self.config.DISCHARGE_FILE)
        amo_index_data_processed = self.load_amo_index(self.config.AMO_INDEX_FILE)

        # Store raw processed data (can be large, consider if needed for full return)
        analysis_results_store = {
            'reanalysis_datasets_processed': all_reanalysis_data_processed,
            'discharge_data_processed': discharge_data_processed,
            'amo_index_data_processed': amo_index_data_processed,
            'beta_obs_slopes_era5': {} # Will be populated
        }

        # Perform Reanalysis-specific calculations (Jet indices, Correlations, Maps)
        reanalysis_jet_data_all = {}
        reanalysis_correlations_all = {}
        reanalysis_jet_impact_maps_all = {}
        reanalysis_amo_jet_correlations_all = {'Winter': {}, 'Summer': {}}
        
        for dataset_id_key in [self.config.DATASET_ERA5, self.config.DATASET_20CRV3]: # Process ERA5 first for beta_obs
            logging.info(f"\n  Further analysis for reanalysis dataset: {dataset_id_key}")
            
            # 1. Jet Indices analysis (returns bundles for correlations)
            jet_data_for_current_ds = AdvancedAnalyzer.analyze_jet_indices(all_reanalysis_data_processed, dataset_id_key)
            reanalysis_jet_data_all.update(jet_data_for_current_ds) # Add to combined dict

            # 2. Time series correlations (e.g., discharge vs jet, PR box vs jet)
            correlations_for_current_ds = AdvancedAnalyzer.analyze_correlations(
                all_reanalysis_data_processed, discharge_data_processed,
                jet_data_for_current_ds, # Pass only the jet data for the current dataset
                dataset_id_key
            )
            reanalysis_correlations_all[dataset_id_key] = correlations_for_current_ds

            # Extract beta_obs slopes specifically from ERA5 results
            if dataset_id_key == self.config.DATASET_ERA5 and correlations_for_current_ds:
                logging.info(f"    Extracting Beta_obs slopes from {dataset_id_key} correlations...")
                # Define the specific slope keys needed for storylines
                # Format: (impact_var_category, season_lower, jet_dim_in_corr_dict, output_beta_key)
                beta_keys_to_extract = [
                    ('tas', 'winter', 'speed', 'DJF_JetSpeed_vs_tas'), ('tas', 'winter', 'lat', 'DJF_JetLat_vs_tas'),
                    ('pr',  'winter', 'speed', 'DJF_JetSpeed_vs_pr'),  ('pr',  'winter', 'lat', 'DJF_JetLat_vs_pr'),
                    ('tas', 'summer', 'speed', 'JJA_JetSpeed_vs_tas'), ('tas', 'summer', 'lat', 'JJA_JetLat_vs_tas'),
                    ('pr',  'summer', 'speed', 'JJA_JetSpeed_vs_pr'),  ('pr',  'summer', 'lat', 'JJA_JetLat_vs_pr')
                ]
                for var_cat, season_l, jet_d, beta_out_key in beta_keys_to_extract:
                    try:
                        slope_val = correlations_for_current_ds.get(var_cat, {}).get(season_l, {}).get(jet_d, {}).get('slope')
                        analysis_results_store['beta_obs_slopes_era5'][beta_out_key] = slope_val
                        if slope_val is None: logging.info(f"      - Slope for {beta_out_key} not found or is None in ERA5 correlations.")
                    except Exception as e_beta:
                        logging.error(f"      - Error extracting slope for {beta_out_key} from ERA5: {e_beta}")
                        analysis_results_store['beta_obs_slopes_era5'][beta_out_key] = None
            
            # 3. Jet Impact Maps (Regression of Jet Index onto PR/TAS fields)
            for season_str in ['Winter', 'Summer']:
                impact_maps_current_ds_season = AdvancedAnalyzer.calculate_jet_impact_maps(
                    all_reanalysis_data_processed, jet_data_for_current_ds,
                    dataset_id_key, season_str
                )
                if dataset_id_key not in reanalysis_jet_impact_maps_all: reanalysis_jet_impact_maps_all[dataset_id_key] = {}
                reanalysis_jet_impact_maps_all[dataset_id_key].update(impact_maps_current_ds_season) # Merge season results

            # 4. AMO - Jet Correlations
            if amo_index_data_processed:
                for season_str in ['Winter', 'Summer']:
                    amo_jet_corr_current_ds_season = AdvancedAnalyzer.analyze_amo_jet_correlations(
                        jet_data_for_current_ds, amo_index_data_processed, season_str
                    ) # Returns {dataset_id: {speed:{}, lat:{}}}
                    if dataset_id_key in amo_jet_corr_current_ds_season: # Check if results were produced for this dataset
                         reanalysis_amo_jet_correlations_all[season_str][dataset_id_key] = amo_jet_corr_current_ds_season[dataset_id_key]

        analysis_results_store['reanalysis_jet_indices_bundles'] = reanalysis_jet_data_all
        analysis_results_store['reanalysis_timeseries_correlations'] = reanalysis_correlations_all
        analysis_results_store['reanalysis_jet_impact_maps'] = reanalysis_jet_impact_maps_all
        analysis_results_store['reanalysis_amo_jet_correlations'] = reanalysis_amo_jet_correlations_all

        # 5. Reanalysis U850 vs. Box Index Regression Maps
        logging.info("\n  Calculating Reanalysis U850 vs. Box Index Regression Maps...")
        analysis_results_store['reanalysis_u850_box_regression_20crv3'] = AdvancedAnalyzer.calculate_regression_maps(all_reanalysis_data_processed, self.config.DATASET_20CRV3)
        analysis_results_store['reanalysis_u850_box_regression_era5'] = AdvancedAnalyzer.calculate_regression_maps(all_reanalysis_data_processed, self.config.DATASET_ERA5)

        # --- Reanalysis Visualizations ---
        logging.info("\n  Visualizing Reanalysis Results...")
        if analysis_results_store['reanalysis_jet_indices_bundles']:
            Visualizer.plot_jet_indices_timeseries(analysis_results_store['reanalysis_jet_indices_bundles'])
        
        if analysis_results_store['reanalysis_timeseries_correlations'].get(self.config.DATASET_ERA5) and \
           analysis_results_store['reanalysis_timeseries_correlations'].get(self.config.DATASET_20CRV3):
            Visualizer.plot_seasonal_correlation_matrix(
                analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_ERA5],
                analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_20CRV3], 'Winter'
            )
            Visualizer.plot_seasonal_correlation_matrix(
                analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_ERA5],
                analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_20CRV3], 'Summer'
            )
            # Plot timeseries for correlations (needs careful data passing from the nested dict)
            Visualizer.plot_seasonal_correlations_timeseries(
                 analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_ERA5],
                 analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_20CRV3], 'Winter'
            )
            Visualizer.plot_seasonal_correlations_timeseries(
                 analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_ERA5],
                 analysis_results_store['reanalysis_timeseries_correlations'][self.config.DATASET_20CRV3], 'Summer'
            )


        if analysis_results_store.get('reanalysis_u850_box_regression_20crv3'):
            Visualizer.plot_regression_analysis_figure(analysis_results_store['reanalysis_u850_box_regression_20crv3'], self.config.DATASET_20CRV3)
        if analysis_results_store.get('reanalysis_u850_box_regression_era5'):
            Visualizer.plot_regression_analysis_figure(analysis_results_store['reanalysis_u850_box_regression_era5'], self.config.DATASET_ERA5)

        if analysis_results_store['reanalysis_jet_impact_maps'].get(self.config.DATASET_ERA5) and \
           analysis_results_store['reanalysis_jet_impact_maps'].get(self.config.DATASET_20CRV3):
            # This plot function needs adaptation if Visualizer expects slightly different input structure
            # Visualizer.plot_jet_impact_maps(
            #     analysis_results_store['reanalysis_jet_impact_maps'][self.config.DATASET_20CRV3], # Data for 20CRv3
            #     analysis_results_store['reanalysis_jet_impact_maps'][self.config.DATASET_ERA5],   # Data for ERA5
            #     'Winter' # This implies the plot function internally loops seasons or is called per season
            # )
            # Visualizer.plot_jet_impact_maps(
            #     analysis_results_store['reanalysis_jet_impact_maps'][self.config.DATASET_20CRV3], 
            #     analysis_results_store['reanalysis_jet_impact_maps'][self.config.DATASET_ERA5],   
            #     'Summer'
            # )
            pass # Assuming plot_jet_impact_maps needs specific data structures from the dict

        if analysis_results_store['reanalysis_amo_jet_correlations'].get('Winter'):
            Visualizer.plot_amo_jet_correlations(analysis_results_store['reanalysis_amo_jet_correlations']['Winter'], 'Winter')
        if analysis_results_store['reanalysis_amo_jet_correlations'].get('Summer'):
            Visualizer.plot_amo_jet_correlations(analysis_results_store['reanalysis_amo_jet_correlations']['Summer'], 'Summer')
        
        logging.info("--- PHASE 1: REANALYSIS ANALYSIS COMPLETED ---")

        # --- PART 2: CMIP6 ANALYSIS & STORYLINES ---
        logging.info("\n\n--- PHASE 2: CMIP6 ANALYSIS & STORYLINES ---")
        storyline_module_analyzer = StorylineAnalyzer(self.config) # Create instance

        # This needs a list of CMIP6 models to process.
        # This list should ideally be generated by scanning available data or predefined.
        # For now, assuming it's passed or handled by storyline_module_analyzer.analyze_cmip6_changes_at_gwl
        # Ideally, main script would define: models_to_run_cmip6 = ['CESM2', 'MPI-ESM1-2-HR', ...]
        # If None, analyze_cmip6_changes_at_gwl might attempt to discover or use a default list.
        # For robustness, let's assume it needs an explicit list from config or elsewhere.
        cmip6_models_to_analyze = getattr(self.config, 'CMIP6_MODELS_TO_RUN_LIST', None) # Example: add this to Config
        if cmip6_models_to_analyze is None:
            logging.warning("Config.CMIP6_MODELS_TO_RUN_LIST not defined. CMIP6 analysis might not run for specific models.")
            # Fallback or error: depends on how analyze_cmip6_changes_at_gwl handles a None model list.
            # For this example, let's assume it will try to find some models if None, or we skip if truly None.
            # If it requires an explicit list and None is passed, it should handle it gracefully (e.g. return empty).
            # For now, we pass it as is.

        cmip6_gwl_results = storyline_module_analyzer.analyze_cmip6_changes_at_gwl(
            list_of_models_to_process=cmip6_models_to_analyze
        )
        analysis_results_store['cmip6_gwl_analysis_results'] = cmip6_gwl_results

        if cmip6_gwl_results and cmip6_gwl_results.get('cmip6_model_raw_data_loaded'):
            logging.info("\n  Calculating CMIP6 Historical Regression Maps (U850 vs Box Indices)...")
            cmip6_hist_reg_maps = AdvancedAnalyzer.calculate_cmip6_regression_maps(
                cmip6_gwl_results['cmip6_model_raw_data_loaded'],
                historical_period=(self.config.CMIP6_ANOMALY_REF_START, self.config.CMIP6_ANOMALY_REF_END)
            )
            analysis_results_store['cmip6_historical_regression_maps'] = cmip6_hist_reg_maps
            if cmip6_hist_reg_maps:
                Visualizer.plot_regression_analysis_figure(cmip6_hist_reg_maps, "CMIP6 MMM Hist", filename_suffix="cmip6_mmm_hist_regression_maps")

            # Beta_obs (from ERA5) vs CMIP6 historical slopes comparison
            if analysis_results_store['beta_obs_slopes_era5']:
                 logging.info("\n  Comparing ERA5 Observed Slopes vs. CMIP6 Historical Slopes...")
                 # Define a sensible comparison period
                 compare_start = max(self.config.BASE_PERIOD_START_YEAR, self.config.CMIP6_PRE_INDUSTRIAL_REF_START)
                 compare_end = min(self.config.BASE_PERIOD_END_YEAR, self.config.CMIP6_ANOMALY_REF_END)
                 
                 cmip6_hist_slopes_for_comp = AdvancedAnalyzer.calculate_historical_slopes_comparison(
                     analysis_results_store['beta_obs_slopes_era5'],
                     cmip6_gwl_results['cmip6_model_raw_data_loaded'], # Use the initially loaded full timeseries data
                     analysis_results_store['reanalysis_jet_indices_bundles'], # For context if needed
                     historical_period_for_cmip6=(compare_start, compare_end)
                 )
                 analysis_results_store['cmip6_historical_slopes_for_beta_comparison'] = cmip6_hist_slopes_for_comp
                 if cmip6_hist_slopes_for_comp:
                     Visualizer.plot_beta_obs_comparison(
                         analysis_results_store['beta_obs_slopes_era5'],
                         cmip6_hist_slopes_for_comp
                     )
            
            # Plot CMIP6 Jet Changes vs GWL (crucial for defining/validating storyline values in Config)
            Visualizer.plot_cmip6_jet_changes_vs_gwl(cmip6_gwl_results)
            logging.info("\n  >>> ACTION REQUIRED: Examine the plot 'cmip6_jet_changes_vs_gwl.png'. <<<")
            logging.info("  >>> Based on the plot, update the placeholder values in Config.STORYLINE_JET_CHANGES "
                         "with appropriate 'Core Mean/High' and 'Extreme Low/High' values for each GWL. <<<")

        # Calculate Storyline Impacts
        logging.info("\n  Calculating Storyline Impacts...")
        if cmip6_gwl_results and analysis_results_store['beta_obs_slopes_era5'] and self.config.STORYLINE_JET_CHANGES:
            # Ensure all necessary beta_obs slopes are available (not None/NaN)
            required_beta_keys_for_storylines = list(analysis_results_store['beta_obs_slopes_era5'].keys()) # Adapt if more specific keys needed
            if all(analysis_results_store['beta_obs_slopes_era5'].get(key) is not None and \
                   not np.isnan(analysis_results_store['beta_obs_slopes_era5'].get(key)) \
                   for key in required_beta_keys_for_storylines):
                
                storyline_impacts_calculated = storyline_module_analyzer.calculate_storyline_impacts(
                    cmip6_gwl_results,
                    analysis_results_store['beta_obs_slopes_era5']
                )
                analysis_results_store['storyline_impacts_calculated'] = storyline_impacts_calculated
                if storyline_impacts_calculated:
                    logging.info("\n  Calculated Storyline Impacts:")
                    logging.info(json.dumps(storyline_impacts_calculated, indent=2, default=lambda x: round(x, 2) if isinstance(x, (float, np.floating)) else str(x)))
                    Visualizer.plot_storyline_impacts(storyline_impacts_calculated)
                else:
                    logging.info("  Storyline impact calculation returned no results or failed.")
            else:
                logging.warning("  WARNING: Cannot calculate storyline impacts because some required beta_obs slopes from ERA5 are missing or NaN.")
                analysis_results_store['storyline_impacts_calculated'] = None
        else:
            logging.info("  Skipping storyline impact calculation: missing CMIP6 results, ERA5 beta_obs slopes, or storyline definitions in Config.")
            analysis_results_store['storyline_impacts_calculated'] = None

        logging.info("--- PHASE 2: CMIP6 ANALYSIS & STORYLINES COMPLETED ---")

        logging.info(f"\n\n====================================================================")
        logging.info(f"=== FULL CLIMATE ANALYSIS WORKFLOW COMPLETED ===")
        logging.info(f"====================================================================\n")
        logging.info(f"All plots should be saved to: {self.config.PLOT_DIR}")
        logging.info(f"Intermediate results might be in: {self.config.RESULTS_DIR}")
        
        return analysis_results_store