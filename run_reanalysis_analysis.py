#!/usr/bin/env python3
"""
Main script for Reanalysis Data Processing and Analysis.

This script orchestrates the loading, processing, and analysis of reanalysis
data (ERA5, 20CRv3). It calculates jet stream indices, performs regression
analysis between climate variables and jet indices, analyzes correlations,
and generates relevant plots and intermediate results for further use.
"""

import logging
import sys
import os
import multiprocessing
import pickle # For saving/loading Python objects
import json   # For saving/loading simple structures like beta_obs_slopes

# --- Environment Setup ---
# Hinzufügen des 'utils'-Verzeichnisses zum sys.path
# Dies ermöglicht Importe wie 'from config_setup import Config' direkt
# sowohl in diesem Skript als auch potenziell in den Modulen innerhalb von utils,
# wenn sie andere Module aus utils laden.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))


# Import utility modules (jetzt sollten sie gefunden werden, da 'utils' im Pfad ist)
from config_setup import Config
from climate_analysis_module import ClimateAnalysis # Handles data loading and initial processing
from advanced_analysis_utils import AdvancedAnalyzer # For regressions, jet index analysis, etc.
from plotting_utils import Visualizer # For generating plots

import matplotlib
matplotlib.use('Agg') # For non-interactive plotting to generate files

# --- Logging Configuration ---
def setup_logging(config_instance: Config, log_filename="reanalysis_analysis_log.log"):
    """Configures logging for this script."""
    log_file_path = os.path.join(config_instance.RESULTS_DIR, log_filename)
    config_instance.ensure_dir_exists(config_instance.RESULTS_DIR) # Ensure results dir exists

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='w'),
            logging.StreamHandler(sys.stdout)
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
            import numpy as np 
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

# --- Main Analysis Function ---
def run_reanalysis_phase(config: Config):
    """
    Orchestrates the processing and analysis of reanalysis data.
    """
    logging.info("--- STARTING REANALYSIS DATA PROCESSING AND ANALYSIS PHASE ---")

    reanalysis_phase_results = {}
    climate_analyzer = ClimateAnalysis(config) # Benötigt Config, das jetzt importiert sein sollte

    # climate_analyzer = ClimateAnalysis(config) # Kann bleiben, falls andere Methoden genutzt werden

    logging.info("\nStep 1a: Loading ERA5 Data...")
    # Rufen Sie die statische Methode aus ClimateAnalysis auf, um ERA5-Daten zu laden.
    # Diese Methode gibt ein Dictionary mit den verarbeiteten ERA5-Datensätzen zurück.
    era5_processed_datasets = ClimateAnalysis.process_era5_data(config) # [1]

    era5_jet_data = {}
    # Die Struktur für 'era5_data_loaded', wie sie im Rest des Skripts erwartet wird,
    # ist ein Dictionary, bei dem der Schlüssel der Dataset-Name ist.
    era5_data_loaded_for_results = {}

    if era5_processed_datasets:
        # Bereiten Sie era5_data_loaded für die Speicherung und weitere Verwendung vor
        era5_data_loaded_for_results = {Config.DATASET_ERA5: era5_processed_datasets}
        
        logging.info("Calculating ERA5 Jet Indices...")
        # AdvancedAnalyzer.analyze_jet_indices erwartet die direkt verarbeiteten Datensätze
        # und den Namen des Datasets.
        era5_jet_data = AdvancedAnalyzer.analyze_jet_indices(era5_processed_datasets, Config.DATASET_ERA5) # [1] (basierend auf der Logik in climate_analysis_module.py)
    else:
        logging.error(f"CRITICAL: ERA5 data loading failed for {Config.DATASET_ERA5}. Cannot calculate jet indices.")
        # Stellen Sie sicher, dass era5_data_loaded_for_results leer bleibt oder entsprechend behandelt wird,
        # um Fehler weiter unten zu vermeiden.
        era5_data_loaded_for_results = {Config.DATASET_ERA5: None}


    reanalysis_phase_results['era5_data_loaded'] = era5_data_loaded_for_results
    reanalysis_phase_results['era5_jet_data'] = era5_jet_data

    # Die nachfolgenden Überprüfungen und Verwendungen von era5_data_loaded
    # sollten mit der neuen Struktur era5_data_loaded_for_results[Config.DATASET_ERA5] arbeiten.
    # Beispiel:
    # if not era5_data_loaded_for_results.get(Config.DATASET_ERA5) or not era5_jet_data:
    #     logging.error(f"CRITICAL: ERA5 data loading or jet index calculation failed for {Config.DATASET_ERA5}. Check logs.")
    # else:
    #     logging.info(f"ERA5 data and jet indices processed for {Config.DATASET_ERA5}.")
    #     logging.debug(f"  ERA5 loaded data keys: {list(era5_data_loaded_for_results.get(Config.DATASET_ERA5, {}).keys())}")
    #     logging.debug(f"  ERA5 jet data keys: {list(era5_jet_data.keys())}")

    # Anpassungen für beta_obs_slopes_era5:
    beta_obs_slopes_era5 = {}
    # Verwenden Sie era5_processed_datasets direkt, wenn es erfolgreich geladen wurde
    if era5_processed_datasets: # Dies prüft, ob die Daten erfolgreich geladen wurden
        logging.info("\n  Calculating Reanalysis Beta_obs SLOPES (U850 Box vs. Climate Box Indices)...")
        # era5_datasets_for_beta_calc sollte das Dictionary der Datasets für ERA5 sein
        era5_datasets_for_beta_calc = era5_processed_datasets 

        if era5_datasets_for_beta_calc and \
           'ua850' in era5_datasets_for_beta_calc and \
           'pr' in era5_datasets_for_beta_calc and \
           'tas' in era5_datasets_for_beta_calc:
            
            beta_obs_slopes_era5 = AdvancedAnalyzer.calculate_beta_obs_slopes_for_era5(
                era5_datasets_for_beta_calc, 
                Config.DATASET_ERA5
            )
        else:
            logging.warning(f"Skipping ERA5 Beta_obs SLOPES calculation: Missing one or more required datasets (ua850, pr, tas) within loaded ERA5 data.")
    else:
        logging.warning(f"Skipping ERA5 Beta_obs SLOPES calculation: Missing loaded ERA5 data altogether.")
    # ...

    # Anpassungen für era5_regression_maps:
    # Die Funktion AdvancedAnalyzer.calculate_regression_maps erwartet laut climate_analysis_module.py
    # die verarbeiteten Datasets und den Dataset-Namen, nicht die Jet-Daten.
    # Wenn Ihre Version von calculate_regression_maps in run_reanalysis_analysis.py anders ist (d.h. Jet-Daten benötigt),
    # dann ist die ursprüngliche Struktur dort korrekt. Hier gehe ich von der Konsistenz mit climate_analysis_module.py aus.
    era5_regression_maps = {}
    if era5_processed_datasets: # Verwenden Sie die direkt geladenen Daten
        logging.info("\n  Calculating Reanalysis U850 vs. Box Index Regression MAPS...")
        # Annahme: calculate_regression_maps benötigt die geladenen Datensätze und nicht die Jet-Indizes als separates Argument
        # Falls Ihre Version era5_jet_data erwartet, passen Sie dies an.
        # Die Signatur in climate_analysis_module.py für AdvancedAnalyzer.calculate_regression_maps ist (all_reanalysis_data_processed, dataset_id_key)
        # all_reanalysis_data_processed enthält direkt die Datensätze (z.B. 'ERA5_pr_seasonal')
        era5_regression_maps = AdvancedAnalyzer.calculate_regression_maps(
            era5_processed_datasets, # Die direkt geladenen Daten für ERA5
            Config.DATASET_ERA5 
        )
    else:
        logging.warning("Skipping ERA5 Regression MAPS calculation: Missing loaded ERA5 data.")
    reanalysis_phase_results['era5_regression_maps'] = era5_regression_maps

    if not era5_data_loaded.get(Config.DATASET_ERA5) or not era5_jet_data:
        logging.error(f"CRITICAL: ERA5 data loading or jet index calculation failed for {Config.DATASET_ERA5}. Check logs from ClimateAnalysis.")
    else:
        logging.info(f"ERA5 data and jet indices processed for {Config.DATASET_ERA5}.")
        logging.debug(f"  ERA5 loaded data keys: {list(era5_data_loaded.get(Config.DATASET_ERA5, {}).keys())}")
        logging.debug(f"  ERA5 jet data keys: {list(era5_jet_data.keys())}")

    beta_obs_slopes_era5 = {} 
    if era5_data_loaded.get(Config.DATASET_ERA5):
        logging.info("\n  Calculating Reanalysis Beta_obs SLOPES (U850 Box vs. Climate Box Indices)...")
        era5_datasets_for_beta_calc = era5_data_loaded.get(Config.DATASET_ERA5)

        if era5_datasets_for_beta_calc and \
           'ua850' in era5_datasets_for_beta_calc and \
           'pr' in era5_datasets_for_beta_calc and \
           'tas' in era5_datasets_for_beta_calc:
            
            beta_obs_slopes_era5 = AdvancedAnalyzer.calculate_beta_obs_slopes_for_era5( # Diese Funktion muss in advanced_analysis_utils.py existieren
                era5_datasets_for_beta_calc, 
                Config.DATASET_ERA5
            )
        else:
            logging.warning(f"Skipping ERA5 Beta_obs SLOPES calculation: Missing one or more required datasets (ua850, pr, tas) within era5_data_loaded['{Config.DATASET_ERA5}'].")
            beta_obs_slopes_era5 = {} 
    else:
        logging.warning(f"Skipping ERA5 Beta_obs SLOPES calculation: Missing loaded ERA5 data (era5_data_loaded['{Config.DATASET_ERA5}']) altogether.")
        beta_obs_slopes_era5 = {} 
    
    reanalysis_phase_results['beta_obs_slopes_era5'] = beta_obs_slopes_era5

    era5_regression_maps = {}
    if era5_data_loaded.get(Config.DATASET_ERA5) and era5_jet_data:
        logging.info("\n  Calculating Reanalysis U850 vs. Box Index Regression MAPS...")
        era5_regression_maps = AdvancedAnalyzer.calculate_regression_maps(
            era5_data_loaded, 
            era5_jet_data,
            Config.DATASET_ERA5 
        )
    else:
        logging.warning("Skipping ERA5 Regression MAPS calculation: Missing loaded ERA5 data or ERA5 jet data.")
    reanalysis_phase_results['era5_regression_maps'] = era5_regression_maps

    logging.info("\nStep 3: Visualizing Reanalysis Results...")
    try:
        if era5_regression_maps:
            Visualizer.plot_regression_analysis_figure(
                era5_regression_maps, Config.DATASET_ERA5,
                filename_suffix=f"reanalysis_regression_maps_{Config.DATASET_ERA5.lower()}"
            )
    except Exception as e_viz:
        logging.error(f"Error during reanalysis visualization phase: {e_viz}", exc_info=True)
    logging.info("Reanalysis visualizations completed.")

    logging.info("\nStep 4: Saving key reanalysis results...")
    beta_slopes_filepath = os.path.join(config.RESULTS_DIR, "era5_beta_obs_slopes.json")
    save_results(beta_obs_slopes_era5, beta_slopes_filepath, use_pickle=False)

    full_results_filepath = os.path.join(config.RESULTS_DIR, "reanalysis_phase_full_results.pkl")
    save_results(reanalysis_phase_results, full_results_filepath, use_pickle=True)
    
    logging.info("--- REANALYSIS DATA PROCESSING AND ANALYSIS PHASE COMPLETED ---")
    return reanalysis_phase_results

# --- Script Execution ---
if __name__ == "__main__":
    # Config Instanz muss VOR setup_logging erstellt werden, wenn setup_logging sie verwendet
    config = Config() 
    setup_logging(config, log_filename="reanalysis_analysis_log.log")

    logging.info(f"Script: {os.path.basename(__file__)}")
    logging.info(f"Project Base Directory (from Config): {config.PROJECT_BASE_DIR}") # PROJECT_BASE_DIR wird von Config gelesen
    logging.info(f"Plot Output Directory: {config.PLOT_DIR}")
    logging.info(f"Intermediate Results Directory: {config.RESULTS_DIR}")
    logging.info(f"Number of CPUs available: {multiprocessing.cpu_count()}")
    logging.info(f"Using N_PROCESSES = {config.N_PROCESSES} for parallel tasks.")

    results_summary = run_reanalysis_phase(config)

    if results_summary:
        logging.info("Reanalysis analysis phase completed successfully.")
        logging.info(f"  ERA5 Beta_obs slopes saved to: {os.path.join(config.RESULTS_DIR, 'era5_beta_obs_slopes.json')}")
        logging.info(f"  Full reanalysis phase results dictionary saved to: {os.path.join(config.RESULTS_DIR, 'reanalysis_phase_full_results.pkl')}")
        if results_summary.get('beta_obs_slopes_era5'):
            logging.info(f"  Calculated beta_obs_slopes_era5: {results_summary['beta_obs_slopes_era5']}")
        else:
            logging.warning("  beta_obs_slopes_era5 was empty or not calculated.")
    else:
        logging.error("Reanalysis analysis phase encountered errors or did not complete fully.")

    logging.info("Script execution finished.")