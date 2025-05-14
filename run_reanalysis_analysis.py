#!/usr/bin/env python3
"""
Main script for Reanalysis Data Processing and Analysis (ERA5 & 20CRv3).
This script processes reanalysis data, calculates various climate indices and
their relationships, generates relevant plots, and saves key results.
"""
import logging
import sys
import os

# --- Environment Setup ---
# Add the 'utils' directory to the Python path to import helper modules
# This assumes 'utils' is a subdirectory of the folder where this script is located.
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

import numpy as np
import traceback
import multiprocessing
import pickle # For saving/loading Python objects (like dictionaries of results)
import json # For saving simple structures like beta_obs_slopes

# Import utility modules
from config_setup import Config
from climate_analysis_module import ClimateAnalysis # Contains the main orchestration logic
from advanced_analysis_utils import AdvancedAnalyzer # For specific analysis steps
from plotting_utils import Visualizer # For generating plots

# --- Matplotlib Backend Configuration ---
# Set backend to 'Agg' for non-interactive plotting (e.g., on servers without a display)
# This must be done *before* importing matplotlib.pyplot
import matplotlib
matplotlib.use('Agg')

# --- Global Variables / Constants (if any, specific to this script) ---
# (None defined here, relying on Config)

# --- Logging Configuration ---
# (This can be more sophisticated, e.g., loaded from a config file)
def setup_logging(config_instance: Config, log_filename="reanalysis_analysis_log.log"):
    """Configures logging for the script."""
    log_file_path = os.path.join(config_instance.RESULTS_DIR, log_filename) # Save log in results dir
    config_instance.ensure_dir_exists(config_instance.RESULTS_DIR) # Ensure results dir exists

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'), # Overwrite log file each run
            logging.StreamHandler(sys.stdout) # Also print to console
        ]
    )
    logging.info("Logging initialized for Reanalysis Analysis script.")

# --- Helper functions for saving/loading results ---
def save_results(data_to_save, filepath, use_pickle=True):
    """Saves data to a file using pickle or json."""
    try:
        if use_pickle:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logging.info(f"Results successfully saved to (pickle): {filepath}")
        else: # Use JSON
            with open(filepath, 'w') as f:
                # Handle non-serializable numpy types for JSON
                def default_converter(o):
                    if isinstance(o, (np.integer, np.floating, np.bool_)):
                        return o.item()
                    elif isinstance(o, np.ndarray):
                        return o.tolist() # Convert arrays to lists
                    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
                json.dump(data_to_save, f, indent=4, default=default_converter)
            logging.info(f"Results successfully saved to (json): {filepath}")
    except Exception as e:
        logging.error(f"Error saving results to {filepath}: {e}")

def load_results(filepath, use_pickle=True):
    """Loads data from a file."""
    if not os.path.exists(filepath):
        logging.warning(f"File not found, cannot load results: {filepath}")
        return None
    try:
        if use_pickle:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
        else: # Use JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
        logging.info(f"Results successfully loaded from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading results from {filepath}: {e}")
        return None

# --- Main Analysis Function for Reanalysis ---
def run_reanalysis_phase(config: Config):
    """
    Orchestrates the processing and analysis of reanalysis data.
    """
    logging.info("--- STARTING REANALYSIS DATA PROCESSING AND ANALYSIS PHASE ---")

    # Initialize the main analysis class
    # The ClimateAnalysis class methods will handle calls to DataProcessor etc.
    climate_analyzer = ClimateAnalysis(config)

    # 1. Process core datasets (20CRv3, ERA5, Discharge, AMO)
    logging.info("Step 1.1: Processing 20CRv3 data...")
    datasets_20crv3 = climate_analyzer.process_20crv3_data(config)
    if not datasets_20crv3:
        logging.critical("CRITICAL: Failed to process 20CRv3 data. Check logs and data paths.")
        # Decide if to continue with ERA5 only or halt. For now, let's try to continue if ERA5 works.
    
    logging.info("Step 1.2: Processing ERA5 data...")
    datasets_era5 = climate_analyzer.process_era5_data(config)
    if not datasets_era5:
        logging.critical("CRITICAL: Failed to process ERA5 data. Check logs and data paths.")
        # If ERA5 fails, beta_obs_slopes cannot be calculated, impacting storylines.
        # Depending on the project, this might be a hard stop.

    # Combine processed reanalysis datasets if both are available
    all_reanalysis_data_processed = {}
    if datasets_20crv3: all_reanalysis_data_processed.update(datasets_20crv3)
    if datasets_era5: all_reanalysis_data_processed.update(datasets_era5)

    if not all_reanalysis_data_processed:
        logging.critical("No reanalysis datasets were successfully processed. Halting reanalysis phase.")
        return None # Indicate failure

    logging.info("Step 1.3: Processing Discharge data...")
    discharge_data_processed = climate_analyzer.process_discharge_data(config.DISCHARGE_FILE)
    
    logging.info("Step 1.4: Processing AMO Index data...")
    amo_index_data_processed = climate_analyzer.load_amo_index(config.AMO_INDEX_FILE)

    # Store initial processed data (optional, can be large)
    reanalysis_phase_results = {
        'reanalysis_datasets_processed': all_reanalysis_data_processed, # Contains monthly & seasonal data
        'discharge_data_processed': discharge_data_processed,
        'amo_index_data_processed': amo_index_data_processed,
        'beta_obs_slopes_era5': {} # To be populated specifically from ERA5
    }

    # 2. Perform further analyses on reanalysis data
    logging.info("\nStep 2: Performing detailed analyses on reanalysis datasets...")
    reanalysis_jet_data_all = {} # For jet indices and data bundles
    reanalysis_correlations_all = {} # For timeseries correlations
    reanalysis_jet_impact_maps_all = {} # For jet impact maps
    reanalysis_amo_jet_correlations_all = {'Winter': {}, 'Summer': {}} # For AMO-Jet correlations

    # Iterate through ERA5 first, then 20CRv3, to ensure beta_obs are from ERA5
    datasets_to_analyze_ordered = []
    if config.DATASET_ERA5 in all_reanalysis_data_processed: # Check if ERA5 data is actually present
        datasets_to_analyze_ordered.append(config.DATASET_ERA5)
    if config.DATASET_20CRV3 in all_reanalysis_data_processed: # Check if 20CRv3 data is actually present
         # Check if the keys for 20CRv3 exist, e.g. by looking for one essential key
        if f'{config.DATASET_20CRV3}_ua850_seasonal' in all_reanalysis_data_processed:
             datasets_to_analyze_ordered.append(config.DATASET_20CRV3)


    for dataset_id in datasets_to_analyze_ordered:
        logging.info(f"\n  Analyzing dataset: {dataset_id}")

        # 2.1 Jet Indices Analysis (provides data bundles for correlations)
        logging.info(f"    Analyzing Jet Indices for {dataset_id}...")
        # `analyze_jet_indices` needs the dict of *already processed* data for the current dataset_id
        jet_data_current_ds = AdvancedAnalyzer.analyze_jet_indices(all_reanalysis_data_processed, dataset_id)
        reanalysis_jet_data_all.update(jet_data_current_ds)

        # 2.2 Time Series Correlations
        logging.info(f"    Analyzing Time Series Correlations for {dataset_id}...")
        correlations_current_ds = AdvancedAnalyzer.analyze_correlations(
            all_reanalysis_data_processed,
            discharge_data_processed,
            jet_data_current_ds, # Pass the jet data specific to this dataset
            dataset_id
        )
        reanalysis_correlations_all[dataset_id] = correlations_current_ds

        # Extract beta_obs slopes if this is ERA5
        if dataset_id == Config.DATASET_ERA5 and correlations_current_ds:
            logging.info(f"    Extracting Beta_obs slopes from {dataset_id} correlations...")
            beta_keys_needed = [ # Define the specific slope keys required for storyline analysis
                ('tas', 'winter', 'speed', 'DJF_JetSpeed_vs_tas'), ('tas', 'winter', 'lat', 'DJF_JetLat_vs_tas'),
                ('pr',  'winter', 'speed', 'DJF_JetSpeed_vs_pr'),  ('pr',  'winter', 'lat', 'DJF_JetLat_vs_pr'),
                ('tas', 'summer', 'speed', 'JJA_JetSpeed_vs_tas'), ('tas', 'summer', 'lat', 'JJA_JetLat_vs_tas'),
                ('pr',  'summer', 'speed', 'JJA_JetSpeed_vs_pr'),  ('pr',  'summer', 'lat', 'JJA_JetLat_vs_pr')
            ]
            for var_cat, season_lwr, jet_dim_key, beta_output_key in beta_keys_needed:
                try:
                    # Structure from analyze_correlations: {dataset: {var_cat: {season_lwr: {jet_dim_key: {slope: VAL}}}}}
                    slope_value = correlations_current_ds.get(var_cat, {}).get(season_lwr, {}).get(jet_dim_key, {}).get('slope')
                    reanalysis_phase_results['beta_obs_slopes_era5'][beta_output_key] = slope_value
                    if slope_value is None or np.isnan(slope_value):
                        logging.warning(f"      - Beta_obs slope for '{beta_output_key}' is None/NaN in ERA5 results.")
                except Exception as e_beta_extract:
                    logging.error(f"      - Error extracting Beta_obs slope for '{beta_output_key}': {e_beta_extract}")
                    reanalysis_phase_results['beta_obs_slopes_era5'][beta_output_key] = None
        
        # 2.3 Jet Impact Maps
        logging.info(f"    Calculating Jet Impact Maps for {dataset_id}...")
        if dataset_id not in reanalysis_jet_impact_maps_all: reanalysis_jet_impact_maps_all[dataset_id] = {}
        for season_name in ['Winter', 'Summer']:
            impact_maps_ds_season = AdvancedAnalyzer.calculate_jet_impact_maps(
                all_reanalysis_data_processed,
                jet_data_current_ds, # Jet data for this specific dataset
                dataset_id,
                season_name
            ) # Returns {season_name: {impact_key: data}}
            reanalysis_jet_impact_maps_all[dataset_id].update(impact_maps_ds_season)

        # 2.4 AMO-Jet Correlations
        if amo_index_data_processed:
            logging.info(f"    Analyzing AMO-Jet Correlations for {dataset_id}...")
            for season_name in ['Winter', 'Summer']:
                # analyze_amo_jet_correlations expects jet data for a single dataset
                amo_jet_corr_ds_season = AdvancedAnalyzer.analyze_amo_jet_correlations(
                    jet_data_current_ds, # Pass jet data for the current dataset_id
                    amo_index_data_processed,
                    season_name,
                    rolling_window_size=15 # Example window size
                ) # Returns {dataset_id: {speed: {}, lat: {}}} if successful for this dataset_id
                if dataset_id in amo_jet_corr_ds_season: # Check if current dataset_id results are there
                    if dataset_id not in reanalysis_amo_jet_correlations_all[season_name]:
                         reanalysis_amo_jet_correlations_all[season_name][dataset_id] = {}
                    reanalysis_amo_jet_correlations_all[season_name][dataset_id] = amo_jet_corr_ds_season[dataset_id]


    # Store detailed analysis results
    reanalysis_phase_results['reanalysis_jet_indices_bundles'] = reanalysis_jet_data_all
    reanalysis_phase_results['reanalysis_timeseries_correlations'] = reanalysis_correlations_all
    reanalysis_phase_results['reanalysis_jet_impact_maps'] = reanalysis_jet_impact_maps_all
    reanalysis_phase_results['reanalysis_amo_jet_correlations'] = reanalysis_amo_jet_correlations_all

    # 2.5 U850 vs. Box Index Regression Maps (Field Regressions)
    logging.info("\n  Calculating Reanalysis U850 vs. Box Index Regression Maps...")
    if Config.DATASET_20CRV3 in datasets_to_analyze_ordered:
        reanalysis_phase_results['reanalysis_u850_box_regression_20crv3'] = AdvancedAnalyzer.calculate_regression_maps(
            all_reanalysis_data_processed, Config.DATASET_20CRV3
        )
    if Config.DATASET_ERA5 in datasets_to_analyze_ordered:
        reanalysis_phase_results['reanalysis_u850_box_regression_era5'] = AdvancedAnalyzer.calculate_regression_maps(
            all_reanalysis_data_processed, Config.DATASET_ERA5
        )

    # 3. Visualizations for Reanalysis Phase
    logging.info("\nStep 3: Visualizing Reanalysis Results...")
    try:
        if reanalysis_phase_results['reanalysis_jet_indices_bundles']:
            Visualizer.plot_jet_indices_timeseries(reanalysis_phase_results['reanalysis_jet_indices_bundles'])
        
        corrs_era5 = reanalysis_phase_results['reanalysis_timeseries_correlations'].get(Config.DATASET_ERA5)
        corrs_20cr = reanalysis_phase_results['reanalysis_timeseries_correlations'].get(Config.DATASET_20CRV3)
        if corrs_era5 and corrs_20cr:
            Visualizer.plot_seasonal_correlation_matrix(corrs_era5, corrs_20cr, 'Winter')
            Visualizer.plot_seasonal_correlation_matrix(corrs_era5, corrs_20cr, 'Summer')
            Visualizer.plot_seasonal_correlations_timeseries(corrs_era5, corrs_20cr, 'Winter')
            Visualizer.plot_seasonal_correlations_timeseries(corrs_era5, corrs_20cr, 'Summer')

        if reanalysis_phase_results.get('reanalysis_u850_box_regression_20crv3'):
            Visualizer.plot_regression_analysis_figure(
                reanalysis_phase_results['reanalysis_u850_box_regression_20crv3'], Config.DATASET_20CRV3)
        if reanalysis_phase_results.get('reanalysis_u850_box_regression_era5'):
            Visualizer.plot_regression_analysis_figure(
                reanalysis_phase_results['reanalysis_u850_box_regression_era5'], Config.DATASET_ERA5)
        
        # Plot jet impact maps (requires Visualizer.plot_jet_impact_maps to be adapted)
        # Example for Winter, assuming plot_jet_impact_maps takes data for both datasets for one season
        # if reanalysis_phase_results['reanalysis_jet_impact_maps'].get(Config.DATASET_ERA5) and \
        #    reanalysis_phase_results['reanalysis_jet_impact_maps'].get(Config.DATASET_20CRV3):
        #     Visualizer.plot_jet_impact_maps(
        #         reanalysis_phase_results['reanalysis_jet_impact_maps'][Config.DATASET_20CRV3],
        #         reanalysis_phase_results['reanalysis_jet_impact_maps'][Config.DATASET_ERA5],
        #         'Winter' 
        #     ) # This call needs to align with how plot_jet_impact_maps is structured in Visualizer

        if reanalysis_phase_results['reanalysis_amo_jet_correlations'].get('Winter'):
            Visualizer.plot_amo_jet_correlations(reanalysis_phase_results['reanalysis_amo_jet_correlations']['Winter'], 'Winter')
        if reanalysis_phase_results['reanalysis_amo_jet_correlations'].get('Summer'):
            Visualizer.plot_amo_jet_correlations(reanalysis_phase_results['reanalysis_amo_jet_correlations']['Summer'], 'Summer')
        
        logging.info("Reanalysis visualizations completed.")
    except Exception as e_viz:
        logging.error(f"Error during reanalysis visualization phase: {e_viz}")
        logging.error(traceback.format_exc())


    # 4. Save key reanalysis results for subsequent scripts
    logging.info("\nStep 4: Saving key reanalysis results...")
    # Save beta_obs_slopes (important for storylines)
    beta_slopes_filepath = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    save_results(reanalysis_phase_results['beta_obs_slopes_era5'], beta_slopes_filepath, use_pickle=False) # Save as JSON

    # Save other processed data (e.g., for quick loading if re-running parts, or for CMIP6 comparison)
    # Example: Save the entire reanalysis_phase_results dictionary (can be large)
    full_reanalysis_results_filepath = os.path.join(config.RESULTS_DIR, "reanalysis_phase_full_results.pkl")
    save_results(reanalysis_phase_results, full_reanalysis_results_filepath, use_pickle=True)
    
    logging.info("--- REANALYSIS DATA PROCESSING AND ANALYSIS PHASE COMPLETED ---")
    return reanalysis_phase_results


# --- Script Execution ---
if __name__ == "__main__":
    # 1. Create Config instance
    config = Config()

    # 2. Setup Logging
    setup_logging(config) # Passes the config instance for RESULT_DIR path

    # 3. Print some initial info
    logging.info(f"Script: {os.path.basename(__file__)}")
    logging.info(f"Project Base Directory (from Config): {Config.PROJECT_BASE_DIR}")
    logging.info(f"Plot Output Directory: {Config.PLOT_DIR}")
    logging.info(f"Intermediate Results Directory: {Config.RESULTS_DIR}")
    logging.info(f"Number of CPUs available: {multiprocessing.cpu_count()}")
    logging.info(f"Using N_PROCESSES = {Config.N_PROCESSES} for parallel tasks.")

    # 4. Run the reanalysis phase
    logging.info("Starting reanalysis processing and analysis...")
    results = run_reanalysis_phase(config)

    if results:
        logging.info("Reanalysis phase completed successfully.")
        # Optionally, print summary of where key results are saved
        logging.info(f"  ERA5 Beta_obs slopes saved to: {os.path.join(config.RESULTS_DIR, 'era5_beta_obs_slopes.json')}")
        logging.info(f"  Full reanalysis phase results dictionary saved to: {os.path.join(config.RESULTS_DIR, 'reanalysis_phase_full_results.pkl')}")
    else:
        logging.error("Reanalysis phase encountered errors or did not complete.")

    logging.info("Script execution finished.")