#!/usr/bin/env python3
"""
Configuration settings for the climate analysis project.
"""
import os
import multiprocessing
import logging

class Config:
    """Configuration parameters for the analysis."""

    # Define the absolute path to the main project folder
    # **** USER: Please verify this path is correct ****
    PROJECT_BASE_DIR = '/nas/home/vlw/Desktop/STREAM/Paper1-Project-Folder'

    # --- Relative paths based on PROJECT_BASE_DIR ---
    # Assuming 'STREAM' data folder is a sibling to PROJECT_BASE_DIR
    _STREAM_DATA_SIBLING_DIR = os.path.join(os.path.dirname(PROJECT_BASE_DIR), 'STREAM')

    # NetCDF data paths for 20CRv3 (Example: place actual data accordingly or update paths)
    DATA_BASE_PATH_20CRV3 = os.path.join(_STREAM_DATA_SIBLING_DIR) # Adjust if your 20CRv3 data is elsewhere
    UA_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'uwnd.mon.mean.nc')
    PR_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'prate.mon.mean.nc')
    TAS_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'air.2m.mon.mean.nc')

    # Discharge data path (Example)
    DISCHARGE_FILE = os.path.join(DATA_BASE_PATH_20CRV3, 'danube_discharge_monthly_1893-2021.xlsx')

    # Plot output directory (subdirectory within the project)
    PLOT_DIR = os.path.join(PROJECT_BASE_DIR, 'output_plots_paper1')

    # --- Potentially different file systems / Absolute paths ---
    # **** USER: Please verify these paths if your data is on separate mounts/locations ****
    ERA5_BASE_PATH = '/data/reloclim/normal/ERA5_daily/' # KEEPING ABSOLUTE - LIKELY DIFFERENT FILESYSTEM
    ERA5_UA_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5_day_UA_19580101-20221231_lon80W-40E_lat0-90N.nc')
    ERA5_PR_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_PR_19500101-20221231.nc')
    ERA5_TAS_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_TAS_19500101-20221231.nc')

    # AMO index path (Example using a subfolder 'data' in project dir, or adjust)
    _PROJECT_DATA_DIR = os.path.join(PROJECT_BASE_DIR, 'data') # Example for project-specific data
    AMO_INDEX_FILE = os.path.join(os.path.dirname(PROJECT_BASE_DIR), 'CMIP6-datasets', 'amo_index.csv')

    # CMIP6 Data Paths (KEEPING ABSOLUTE - LIKELY DIFFERENT FILESYSTEM)
    # **** USER: Please verify this path ****
    CMIP6_DATA_BASE_PATH = '/data/users/vlw/paper1-cmip-data'
    CMIP6_VAR_PATH = os.path.join(CMIP6_DATA_BASE_PATH, '{variable}') # Subfolder per variable
    CMIP6_FILE_PATTERN = '{variable}_Amon_{model}_{experiment}_{member}_{grid}_*.nc'


    # --- Analysis Parameters ---
    ANALYSIS_START_YEAR, ANALYSIS_END_YEAR = 1850, 2021
    WIND_LEVEL = 850  # hPa pressure level

    BOX_LAT_MIN, BOX_LAT_MAX = 46.0, 51.0
    BOX_LON_MIN, BOX_LON_MAX = 8.0, 18.0

    JET_SPEED_BOX_LAT_MIN, JET_SPEED_BOX_LAT_MAX = 40.0, 60.0
    JET_SPEED_BOX_LON_MIN, JET_SPEED_BOX_LON_MAX = -20.0, 20.0
    JET_LAT_BOX_LAT_MIN, JET_LAT_BOX_LAT_MAX = 30.0, 70.0
    JET_LAT_BOX_LON_MIN, JET_LAT_BOX_LON_MAX = -20.0, 0.0

    MAP_EXTENT = [-80, 40, 0, 90] # Plotting map extent [lon_min, lon_max, lat_min, lat_max]
    BASE_PERIOD_START_YEAR = 1981
    BASE_PERIOD_END_YEAR = 2010

    # --- Plotting Parameters ---
    COLORMAP = 'RdBu_r'
    COLORBAR_LEVELS = 21

    # --- Dataset Identifiers ---
    DATASET_20CRV3 = "20CRv3"
    DATASET_ERA5 = "ERA5"

    # --- Computation Parameters ---
    N_PROCESSES = max(1, multiprocessing.cpu_count() - 4) # Number of processes for parallel tasks

    # --- CMIP6 Analysis Parameters ---
    CMIP6_SCENARIOS = ['ssp585']
    CMIP6_HISTORICAL_EXPERIMENT_NAME = 'historical'
    CMIP6_MEMBER_ID = 'r1i1p1f1' # Standard ensemble member
    CMIP6_VARIABLES_TO_LOAD = ['ua', 'pr', 'tas'] # Regional vars + ua for Jets
    CMIP6_GLOBAL_TAS_VAR = 'tas' # Variable for global temperature
    CMIP6_LEVEL = 850 # For U850
    CMIP6_PRE_INDUSTRIAL_REF_START = 1850
    CMIP6_PRE_INDUSTRIAL_REF_END = 1900
    CMIP6_ANOMALY_REF_START = 1995 # Standard IPCC AR6 reference
    CMIP6_ANOMALY_REF_END = 2014
    GWL_YEARS_WINDOW = 20 # Window for GWL means (as in Harvey et al., 2023)
    GWL_TEMP_SMOOTHING_WINDOW = 20 # Window for moving average of global temperature
    GLOBAL_WARMING_LEVELS = [2.0, 3.0, 4.0] # GWLs in °C to investigate

    # --- Storyline Definitions (PLACEHOLDER!) ---
    # These values MUST be adjusted after analyzing CMIP6 jet changes!
    # Format: {'IndexName': {GWL: {'StorylineType': Value}}}
    STORYLINE_JET_CHANGES = {
        'DJF_JetSpeed': {
            2.0: {'Core Mean': 0.4, 'Core High': 0.7, 'Extreme Low': -0.8, 'Extreme High': 1.5}, # Example values!
            3.0: {'Core Mean': 0.5, 'Core High': 1.0, 'Extreme Low': -0.6, 'Extreme High': 1.7}, # Example values!
            4.0: {'Core Mean': 0.6, 'Core High': 1.3, 'Extreme Low': -0.4, 'Extreme High': 1.9}, # Example values!
        },
        'JJA_JetLat': {
            2.0: {'Core Mean': 0.5, 'Core High': 2.0, 'Extreme Low': -1.0, 'Extreme High': 3.0}, # Example values!
            3.0: {'Core Mean': 1.0, 'Core High': 2.2, 'Extreme Low': -0.5, 'Extreme High': 3.3}, # Example values!
            4.0: {'Core Mean': 1.2, 'Core High': 2.4, 'Extreme Low':  0.0, 'Extreme High': 3.6}, # Example values!
        }
    }

    # --- Output file naming/paths for intermediate results ---
    # This section can be expanded as needed by the main scripts
    RESULTS_DIR = os.path.join(PROJECT_BASE_DIR, 'results_intermediate')
    REANALYSIS_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reanalysis_processed_data.pkl') # Example for pickling dicts
    ERA5_BETA_SLOPES_FILE = os.path.join(RESULTS_DIR, 'era5_beta_slopes.json')
    CMIP6_GWL_RESULTS_FILE = os.path.join(RESULTS_DIR, 'cmip6_gwl_analysis_results.pkl')
    STORYLINE_IMPACTS_FILE = os.path.join(RESULTS_DIR, 'storyline_impacts.json')

    @staticmethod
    def ensure_dir_exists(path):
        """Ensure a directory exists, creating it if necessary."""
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logging.info(f"Created directory: {path}")

    def __init__(self):
        """Ensure essential directories exist when Config is instantiated."""
        # This is useful if main scripts create a Config() object early.
        # However, for purely static access, this init might not always run.
        # Consider calling ensure_dir_exists explicitly in main scripts for PLOT_DIR and RESULTS_DIR.
        import logging # Local import for this method if not already module-level
        self.ensure_dir_exists(self.PLOT_DIR)
        self.ensure_dir_exists(self.RESULTS_DIR)