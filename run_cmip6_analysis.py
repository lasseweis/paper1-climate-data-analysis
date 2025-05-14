#!/usr/bin/env python3
"""
Main script for CMIP6 Data Processing and Analysis.
This script processes CMIP6 model data, calculates Global Warming Levels (GWLs),
analyzes changes at these GWLs, performs historical comparisons,
generates relevant plots, and saves key results for storyline synthesis.
"""
import logging
import sys
import os
import multiprocessing
import pickle # For saving/loading Python objects
import json   # For saving/loading simple structures like beta_obs_slopes

# --- Environment Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import utility modules
from config_setup import Config
# ClimateAnalysis class itself might not be directly used here if we call StorylineAnalyzer directly
# from climate_analysis_module import ClimateAnalysis
from storyline_utils import StorylineAnalyzer # Handles CMIP6 GWL processing logic
from advanced_analysis_utils import AdvancedAnalyzer # For CMIP6 regression maps, historical slopes
from plotting_utils import Visualizer # For generating CMIP6 plots

import matplotlib
matplotlib.use('Agg') # For non-interactive plotting

# --- Logging Configuration ---
def setup_logging(config_instance: Config, log_filename="cmip6_analysis_log.log"):
    """Configures logging for this script."""
    log_file_path = os.path.join(config_instance.RESULTS_DIR, log_filename)
    config_instance.ensure_dir_exists(config_instance.RESULTS_DIR)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized for CMIP6 Analysis script.")

# --- Helper functions for saving/loading results (can be moved to a shared io_utils.py later) ---
def save_results(data_to_save, filepath, use_pickle=True):
    """Saves data to a file using pickle or json."""
    try:
        if use_pickle:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logging.info(f"Results successfully saved to (pickle): {filepath}")
        else: # Use JSON
            import numpy as np # Needed for default_converter
            def default_converter(o):
                if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
                elif isinstance(o, np.ndarray): return o.tolist()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
            with open(filepath, 'w') as f:
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
            with open(filepath, 'rb') as f: data = pickle.load(f)
        else: # Use JSON
            with open(filepath, 'r') as f: data = json.load(f)
        logging.info(f"Results successfully loaded from: {filepath}")
        return data
    except Exception as e:
        logging.error(f"Error loading results from {filepath}: {e}")
        return None

# --- Main Analysis Function for CMIP6 ---
def run_cmip6_phase(config: Config):
    """
    Orchestrates the processing and analysis of CMIP6 data.
    """
    logging.info("--- STARTING CMIP6 DATA PROCESSING AND ANALYSIS PHASE ---")

    storyline_analyzer = StorylineAnalyzer(config) # Handles core CMIP6 GWL logic

    # 1. CMIP6 Core GWL Analysis
    # This method requires a list of CMIP6 models to process.
    # Get this list from Config or define it here.
    # Example: models_to_process_cmip6 = ['CESM2', 'MPI-ESM1-2-HR', 'UKESM1-0-LL']
    # It's best to define this in config_setup.py as Config.CMIP6_MODELS_TO_RUN_LIST
    
    cmip6_models_to_process = getattr(config, 'CMIP6_MODELS_TO_RUN_LIST', None)
    if cmip6_models_to_process is None:
        logging.warning("Config.CMIP6_MODELS_TO_RUN_LIST is not defined. "
                        "Attempting to run CMIP6 analysis without a specific model list "
                        "might be slow or process unwanted models if discovery is broad. "
                        "For now, this script requires an explicit list.")
        # You might want to implement a model discovery mechanism here or in StorylineAnalyzer
        # if an explicit list isn't always provided.
        # For this example, if no list is provided, we'll skip this intensive step or use a small default.
        # For demonstration, let's assume an empty list means skip, or define a small default for testing.
        # models_to_process_cmip6 = ['ACCESS-CM2'] # Example default if none in Config
        logging.error("CRITICAL: No CMIP6 models specified for analysis in Config.CMIP6_MODELS_TO_RUN_LIST. Halting CMIP6 phase.")
        return None


    logging.info(f"Step 1: Running CMIP6 GWL analysis for models: {cmip6_models_to_process}...")
    cmip6_gwl_analysis_results = storyline_analyzer.analyze_cmip6_changes_at_gwl(
        list_of_models_to_process=cmip6_models_to_process
    )
    if not cmip6_gwl_analysis_results or 'cmip6_model_raw_data_loaded' not in cmip6_gwl_analysis_results:
        logging.error("CMIP6 GWL analysis failed or returned incomplete results. Halting CMIP6 phase.")
        return {'cmip6_gwl_analysis_results': cmip6_gwl_analysis_results} # Return partial if needed

    # 2. CMIP6 Historical Regression Maps (U850 vs. Box Indices for MMM)
    logging.info("\nStep 2: Calculating CMIP6 MMM Historical Regression Maps...")
    cmip6_mmm_hist_regression_maps = {}
    if cmip6_gwl_analysis_results.get('cmip6_model_raw_data_loaded'):
        cmip6_mmm_hist_regression_maps = AdvancedAnalyzer.calculate_cmip6_regression_maps(
            cmip6_gwl_analysis_results['cmip6_model_raw_data_loaded'], # This dict contains {model: {var: data}}
            historical_period=(config.CMIP6_ANOMALY_REF_START, config.CMIP6_ANOMALY_REF_END)
        )
    else:
        logging.warning("Skipping CMIP6 MMM historical regression maps: Raw loaded CMIP6 data not available.")

    # 3. Compare ERA5 Observed Slopes (beta_obs) with CMIP6 Historical Slopes
    logging.info("\nStep 3: Comparing ERA5 Beta_obs Slopes with CMIP6 Historical Slopes...")
    # Load beta_obs_slopes from reanalysis phase
    beta_slopes_filepath = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    beta_obs_slopes_era5 = load_results(beta_slopes_filepath, use_pickle=False)
    cmip6_historical_slopes_for_comparison = {}

    if beta_obs_slopes_era5 and cmip6_gwl_analysis_results.get('cmip6_model_raw_data_loaded'):
        # Define a sensible comparison period, e.g., common overlap or specific historical window
        compare_start_year = max(config.BASE_PERIOD_START_YEAR, config.CMIP6_PRE_INDUSTRIAL_REF_START)
        compare_end_year = min(config.BASE_PERIOD_END_YEAR, config.CMIP6_ANOMALY_REF_END)
        
        cmip6_historical_slopes_for_comparison = AdvancedAnalyzer.calculate_historical_slopes_comparison(
            beta_obs_slopes_era5,
            cmip6_gwl_analysis_results['cmip6_model_raw_data_loaded'],
            {}, # jet_data_reanalysis (not strictly needed here if beta_obs_slopes_era5 is provided)
            historical_period_for_cmip6=(compare_start_year, compare_end_year)
        )
    else:
        logging.warning("Skipping Beta_obs vs CMIP6 historical slopes comparison: "
                        "Missing ERA5 beta_obs slopes or raw loaded CMIP6 data.")

    # 4. Visualizations for CMIP6 Phase
    logging.info("\nStep 4: Visualizing CMIP6 Analysis Results...")
    try:
        if cmip6_gwl_analysis_results:
            Visualizer.plot_cmip6_jet_changes_vs_gwl(cmip6_gwl_analysis_results)
            logging.info("  ACTION REQUIRED: Examine 'cmip6_jet_changes_vs_gwl.png'. "
                         "Update Config.STORYLINE_JET_CHANGES with appropriate values.")

        if cmip6_mmm_hist_regression_maps:
            Visualizer.plot_regression_analysis_figure(
                cmip6_mmm_hist_regression_maps, "CMIP6 MMM Hist",
                filename_suffix="cmip6_mmm_hist_regression_maps"
            )
        
        if beta_obs_slopes_era5 and cmip6_historical_slopes_for_comparison:
            Visualizer.plot_beta_obs_comparison(
                beta_obs_slopes_era5,
                cmip6_historical_slopes_for_comparison
            )
    except Exception as e_viz_cmip6:
        logging.error(f"Error during CMIP6 visualization phase: {e_viz_cmip6}")
        import traceback
        logging.error(traceback.format_exc())

    # 5. Save key CMIP6 analysis results
    logging.info("\nStep 5: Saving key CMIP6 analysis results...")
    # Save the main CMIP6 GWL analysis results (can be large)
    cmip6_results_filepath = os.path.join(config.RESULTS_DIR, "cmip6_gwl_analysis_full_results.pkl")
    save_results(cmip6_gwl_analysis_results, cmip6_results_filepath, use_pickle=True)

    # Save CMIP6 historical slopes if calculated
    if cmip6_historical_slopes_for_comparison:
        cmip6_hist_slopes_filepath = os.path.join(config.RESULTS_DIR, "cmip6_historical_slopes_for_beta_comparison.pkl")
        save_results(cmip6_historical_slopes_for_comparison, cmip6_hist_slopes_filepath, use_pickle=True)
    
    logging.info("--- CMIP6 DATA PROCESSING AND ANALYSIS PHASE COMPLETED ---")
    
    # Return a dictionary of key results from this phase
    return {
        'cmip6_gwl_analysis_results': cmip6_gwl_analysis_results,
        'cmip6_mmm_hist_regression_maps': cmip6_mmm_hist_regression_maps,
        'cmip6_historical_slopes_for_comparison': cmip6_historical_slopes_for_comparison,
        'loaded_era5_beta_obs_slopes': beta_obs_slopes_era5
    }

# --- Script Execution ---
if __name__ == "__main__":
    config = Config()
    setup_logging(config, log_filename="cmip6_analysis_log.log") # Specific log file

    logging.info(f"Script: {os.path.basename(__file__)}")
    logging.info(f"Using Config: Plot Dir='{config.PLOT_DIR}', Results Dir='{config.RESULTS_DIR}'")
    logging.info(f"Number of CPUs available: {multiprocessing.cpu_count()}, Using N_PROCESSES = {config.N_PROCESSES}")

    # Check if required input from reanalysis phase exists
    beta_slopes_filepath = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    if not os.path.exists(beta_slopes_filepath):
        logging.error(f"CRITICAL: Required input file '{beta_slopes_filepath}' not found. "
                      "Run reanalysis phase script first.")
        sys.exit(1) # Exit if crucial input is missing

    logging.info("Starting CMIP6 processing and analysis...")
    cmip6_results_summary = run_cmip6_phase(config)

    if cmip6_results_summary:
        logging.info("CMIP6 analysis phase completed successfully.")
        logging.info(f"  Full CMIP6 GWL analysis results saved to: {os.path.join(config.RESULTS_DIR, 'cmip6_gwl_analysis_full_results.pkl')}")
    else:
        logging.error("CMIP6 analysis phase encountered errors or did not complete fully.")

    logging.info("Script execution finished.")