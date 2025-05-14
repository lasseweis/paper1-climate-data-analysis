#!/usr/bin/env python3
"""
Main script for Storyline Synthesis.
This script loads results from the reanalysis and CMIP6 analysis phases,
applies defined storyline assumptions to calculate impacts on climate variables,
generates relevant plots, and saves the storyline impact results.
"""
import logging
import sys
import os
import pickle # For loading Python objects
import json   # For loading simple structures like beta_obs_slopes

# --- Environment Setup ---
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

# Import utility modules
from config_setup import Config
from storyline_utils import StorylineAnalyzer # Handles storyline calculation logic
from plotting_utils import Visualizer     # For generating storyline plots

import matplotlib
matplotlib.use('Agg') # For non-interactive plotting

# --- Logging Configuration ---
def setup_logging(config_instance: Config, log_filename="storyline_synthesis_log.log"):
    """Configures logging for this script."""
    log_file_path = os.path.join(config_instance.RESULTS_DIR, log_filename)
    config_instance.ensure_dir_exists(config_instance.RESULTS_DIR) # Ensure results dir exists

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging initialized for Storyline Synthesis script.")

# --- Helper functions for loading results (can be moved to a shared io_utils.py later) ---
def load_results(filepath, use_pickle=True):
    """Loads data from a file."""
    if not os.path.exists(filepath):
        logging.error(f"CRITICAL: Input file not found, cannot load: {filepath}")
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

def save_results(data_to_save, filepath, use_pickle=True):
    """Saves data to a file using pickle or json."""
    try:
        if use_pickle:
            with open(filepath, 'wb') as f:
                pickle.dump(data_to_save, f)
            logging.info(f"Storyline results successfully saved to (pickle): {filepath}")
        else: # Use JSON
            import numpy as np # Needed for default_converter
            def default_converter(o):
                if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
                elif isinstance(o, np.ndarray): return o.tolist()
                raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
            with open(filepath, 'w') as f:
                json.dump(data_to_save, f, indent=4, default=default_converter)
            logging.info(f"Storyline results successfully saved to (json): {filepath}")
    except Exception as e:
        logging.error(f"Error saving storyline results to {filepath}: {e}")


# --- Main Function for Storyline Synthesis ---
def run_storyline_phase(config: Config):
    """
    Orchestrates the calculation and visualization of storyline impacts.
    """
    logging.info("--- STARTING STORYLINE SYNTHESIS PHASE ---")

    storyline_analyzer = StorylineAnalyzer(config) # Needs config for STORYLINE_JET_CHANGES

    # 1. Load necessary inputs from previous phases
    logging.info("Step 1: Loading prerequisite data from reanalysis and CMIP6 phases...")

    # Load ERA5 beta_obs slopes
    beta_slopes_filepath = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    beta_obs_slopes_era5 = load_results(beta_slopes_filepath, use_pickle=False)
    if beta_obs_slopes_era5 is None:
        logging.critical("Halting storyline phase: Could not load ERA5 beta_obs slopes.")
        return None

    # Load CMIP6 GWL analysis results
    cmip6_results_filepath = os.path.join(config.RESULTS_DIR, "cmip6_gwl_analysis_full_results.pkl")
    cmip6_gwl_analysis_results = load_results(cmip6_results_filepath, use_pickle=True)
    if cmip6_gwl_analysis_results is None or 'cmip6_mmm_changes_at_gwl' not in cmip6_gwl_analysis_results:
        logging.critical("Halting storyline phase: Could not load CMIP6 GWL analysis results or MMM changes are missing.")
        return None

    # 2. Crucial User Check: Verify Config.STORYLINE_JET_CHANGES
    logging.info("\nStep 2: Verifying Storyline Jet Change Definitions in Config...")
    if not config.STORYLINE_JET_CHANGES:
        logging.critical("CRITICAL: Config.STORYLINE_JET_CHANGES is empty. "
                         "Please define storyline assumptions based on CMIP6 jet change plots.")
        return None
    
    # Check for placeholder values (simple check based on one example value)
    # This is a heuristic and might need to be more robust depending on your placeholder values.
    try:
        example_placeholder_val = 0.4 # From original DJF_JetSpeed 2.0 Core Mean
        if config.STORYLINE_JET_CHANGES.get('DJF_JetSpeed', {}).get(2.0, {}).get('Core Mean') == example_placeholder_val:
             logging.warning("WARNING: Config.STORYLINE_JET_CHANGES might still contain placeholder values. "
                             "Ensure these have been updated based on the 'cmip6_jet_changes_vs_gwl.png' plot "
                             "from the CMIP6 analysis phase for meaningful storyline results.")
    except Exception:
        pass # Could not perform placeholder check, proceed with caution.


    # 3. Calculate Storyline Impacts
    logging.info("\nStep 3: Calculating Storyline Impacts...")
    storyline_impacts_calculated = storyline_analyzer.calculate_storyline_impacts(
        cmip6_gwl_analysis_results,
        beta_obs_slopes_era5
    )

    if storyline_impacts_calculated is None or not any(storyline_impacts_calculated.values()):
        logging.error("Storyline impact calculation failed or returned no results.")
    else:
        logging.info("Storyline impacts calculated successfully.")
        # Log the calculated impacts for review
        logging.info("\nCalculated Storyline Impacts Summary:")
        try:
            # Pretty print JSON (handles numpy types via converter in save_results)
            def default_converter_log(o):
                import numpy as np
                if isinstance(o, (np.integer, np.floating, np.bool_)): return o.item()
                elif isinstance(o, np.ndarray): return o.tolist() # Convert arrays to lists for logging
                if isinstance(o, float) and (np.isnan(o) or np.isinf(o)): return str(o) # Represent NaN/inf as strings
                return repr(o) # Fallback for other types
            logging.info(json.dumps(storyline_impacts_calculated, indent=2, default=default_converter_log))
        except Exception as e_log_json:
            logging.warning(f"Could not log full storyline impacts as JSON due to: {e_log_json}. Logging raw dict.")
            logging.info(str(storyline_impacts_calculated))


    # 4. Visualize Storyline Impacts
    logging.info("\nStep 4: Visualizing Storyline Impacts...")
    if storyline_impacts_calculated and any(storyline_impacts_calculated.values()):
        try:
            Visualizer.plot_storyline_impacts(storyline_impacts_calculated)
        except Exception as e_viz_story:
            logging.error(f"Error during storyline impact visualization: {e_viz_story}")
            import traceback
            logging.error(traceback.format_exc())
    else:
        logging.info("Skipping storyline impact visualization: No impacts were calculated.")

    # 5. Save Storyline Impact Results
    logging.info("\nStep 5: Saving Storyline Impact Results...")
    if storyline_impacts_calculated:
        storyline_output_filepath = os.path.join(config.RESULTS_DIR, config.STORYLINE_IMPACTS_FILE) # Use path from Config
        # Decide format: JSON is more readable for this kind of nested dict data.
        save_results(storyline_impacts_calculated, storyline_output_filepath, use_pickle=False)
    
    logging.info("--- STORYLINE SYNTHESIS PHASE COMPLETED ---")
    
    return {'storyline_impacts': storyline_impacts_calculated}


# --- Script Execution ---
if __name__ == "__main__":
    config = Config()
    setup_logging(config, log_filename="storyline_synthesis_log.log")

    logging.info(f"Script: {os.path.basename(__file__)}")
    logging.info(f"Using Config: Plot Dir='{config.PLOT_DIR}', Results Dir='{config.RESULTS_DIR}'")

    # Check for required input files from previous phases
    beta_slopes_file = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    cmip6_results_file = os.path.join(config.RESULTS_DIR, "cmip6_gwl_analysis_full_results.pkl")

    if not os.path.exists(beta_slopes_file):
        logging.critical(f"CRITICAL: Input file '{beta_slopes_file}' not found. Run reanalysis script first.")
        sys.exit(1)
    if not os.path.exists(cmip6_results_file):
        logging.critical(f"CRITICAL: Input file '{cmip6_results_file}' not found. Run CMIP6 analysis script first.")
        sys.exit(1)

    logging.info("Starting storyline synthesis...")
    storyline_results = run_storyline_phase(config)

    if storyline_results and storyline_results.get('storyline_impacts'):
        logging.info("Storyline synthesis phase completed successfully.")
        logging.info(f"  Storyline impact results saved to: {os.path.join(config.RESULTS_DIR, config.STORYLINE_IMPACTS_FILE)}")
    else:
        logging.error("Storyline synthesis phase encountered errors or produced no impact results.")

    logging.info("Script execution finished.")