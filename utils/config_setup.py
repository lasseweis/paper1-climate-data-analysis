#!/usr/bin/env python3
"""
Configuration settings for the climate analysis project.
"""
import os
import multiprocessing
import logging
import numpy as np # Beibehalten, falls für andere Config-Teile benötigt

class Config:
    """Configuration parameters for the analysis."""

    # Define the absolute path to the main project folder
    # **** USER: Please verify this path is correct ****
    PROJECT_BASE_DIR = '/nas/home/vlw/Desktop/STREAM/Paper1-Project-Folder' #

    # --- Relative paths based on PROJECT_BASE_DIR ---
    # Assuming 'STREAM' data folder is a sibling to PROJECT_BASE_DIR
    _STREAM_DATA_SIBLING_DIR = os.path.join(os.path.dirname(PROJECT_BASE_DIR), 'STREAM') #

    # NetCDF data paths for 20CRv3 (Example: place actual data accordingly or update paths)
    DATA_BASE_PATH_20CRV3 = os.path.join(_STREAM_DATA_SIBLING_DIR) #
    UA_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'uwnd.mon.mean.nc') #
    PR_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'prate.mon.mean.nc') #
    TAS_FILE_20CRV3 = os.path.join(DATA_BASE_PATH_20CRV3, 'air.2m.mon.mean.nc') #

    # Discharge data path (Example)
    DISCHARGE_FILE = os.path.join(DATA_BASE_PATH_20CRV3, 'danube_discharge_monthly_1893-2021.xlsx') #

    # Plot output directory (subdirectory within the project)
    PLOT_DIR = os.path.join(PROJECT_BASE_DIR, 'output_plots_paper1') #

    # --- Potentially different file systems / Absolute paths ---
    ERA5_BASE_PATH = '/data/reloclim/normal/ERA5_daily/' #
    ERA5_UA_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5_day_UA_19580101-20221231_lon80W-40E_lat0-90N.nc') #
    ERA5_PR_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_PR_19500101-20221231.nc') #
    ERA5_TAS_FILE = os.path.join(ERA5_BASE_PATH, 'ERA5_2p5cdo_day_TAS_19500101-20221231.nc') #

    # AMO index path
    AMO_INDEX_FILE = os.path.join(os.path.dirname(PROJECT_BASE_DIR), 'CMIP6-datasets', 'amo_index.csv') #

    # CMIP6 Data Paths 
    # **** WICHTIG: Diese Pfade zeigen jetzt auf die mit CDO VORVERARBEITETEN (regriddeten) Daten ****
    CMIP6_DATA_BASE_PATH = '/data/users/vlw/paper1-cmip-data' # Basisverzeichnis bleibt gleich #
    
    # Pfad zu den Unterordnern mit den regriddeten Daten (z.B. tas_regrid, pr_regrid, ua_regrid)
    # {variable} wird durch 'tas', 'pr', 'ua' ersetzt.
    CMIP6_VAR_PATH = os.path.join(CMIP6_DATA_BASE_PATH, '{variable}_regrid') #

    # Dateimuster für die regriddeten Dateien. 
    # Passen Sie dies an, WIE IHRE DATEIEN NACH DEM CDO REGRIDDING HEISSEN.
    # Beispiel 1: Wenn Sie das Suffix "_regridded" hinzugefügt haben:
    CMIP6_FILE_PATTERN = '{variable}_Amon_{model}_{experiment}_{member}_{grid}_*_regridded.nc' #
    # Beispiel 2: Wenn die Dateien im Ordner "_regrid" den Originalnamen behalten haben (ohne Suffix):
    # CMIP6_FILE_PATTERN = '{variable}_Amon_{model}_{experiment}_{member}_{grid}_*.nc'
    # Beispiel 3: Wenn Sie ein einfaches Muster für alle .nc-Dateien im Ordner wollen:
    # CMIP6_FILE_PATTERN = '*.nc' # Dies ist sehr generisch, spezifischer ist besser.


    CMIP6_MODELS_TO_RUN_LIST = [ #
        'ACCESS-CM2',
        'ACCESS-ESM1-5',
        'AWI-CM-1-1-MR',
        'AWI-ESM-1-1-LR',
        'BCC-CSM2-MR',
        'BCC-ESM1',
        'CAMS-CSM1-0',
        'CAS-ESM2-0',
        'CESM2-WACCM',
        'MPI-ESM1-2-LR' # Sicherstellen, dass Ihr Referenzmodell hier ist, falls es verarbeitet werden soll
    ]


    # --- CMIP6 Regridding Parameters (werden nicht mehr für xesmf benötigt, wenn CDO verwendet wird) ---
    REGRID_TO_REFERENCE = False # WICHTIG: Deaktiviert das xesmf-Regridding im Python-Skript #
    # Die folgenden können auskommentiert oder entfernt werden, da CDO das Regridding übernimmt:
    # REFERENCE_MODEL_FOR_GRID = 'MPI-ESM1-2-LR' 
    # TARGET_GRID_LON = np.arange(-179.5, 180.0, 1.0)
    # TARGET_GRID_LAT = np.arange(-89.5, 90.0, 1.0)  
    # REGRIDDING_METHODS = {
    #     'tas': 'bilinear',
    #     'ua': 'bilinear',
    #     'pr': 'conservative',
    #     'tas_global': 'bilinear' 
    # }


    # --- Analysis Parameters ---
    ANALYSIS_START_YEAR, ANALYSIS_END_YEAR = 1850, 2021 # Oder Ihre gewünschten Jahre #
    WIND_LEVEL = 850  # hPa #

    BOX_LAT_MIN, BOX_LAT_MAX = 46.0, 51.0 #
    BOX_LON_MIN, BOX_LON_MAX = 8.0, 18.0 #

    JET_SPEED_BOX_LAT_MIN, JET_SPEED_BOX_LAT_MAX = 40.0, 60.0 #
    JET_SPEED_BOX_LON_MIN, JET_SPEED_BOX_LON_MAX = -20.0, 20.0 #
    JET_LAT_BOX_LAT_MIN, JET_LAT_BOX_LAT_MAX = 30.0, 70.0 #
    JET_LAT_BOX_LON_MIN, JET_LAT_BOX_LON_MAX = -20.0, 0.0 #

    MAP_EXTENT = [-80, 40, 0, 90] #
    BASE_PERIOD_START_YEAR = 1981 #
    BASE_PERIOD_END_YEAR = 2010 #

    DETREND_JET_INDICES = True  # Steuert, ob Jet-Indizes detrended werden

    # --- Plotting Parameters ---
    COLORMAP = 'RdBu_r' #
    COLORBAR_LEVELS = 21 #

    # --- Dataset Identifiers ---
    DATASET_20CRV3 = "20CRv3" #
    DATASET_ERA5 = "ERA5" #

    # --- Computation Parameters ---
    N_PROCESSES = 1 #

    # --- CMIP6 Analysis Parameters ---
    CMIP6_SCENARIOS = ['ssp585'] # Szenarien, die geladen werden sollen #
    CMIP6_HISTORICAL_EXPERIMENT_NAME = 'historical' #
    CMIP6_MEMBER_ID = 'r1i1p1f1' #
    CMIP6_VARIABLES_TO_LOAD = ['ua', 'pr', 'tas'] #
    CMIP6_GLOBAL_TAS_VAR = 'tas_global' # Wird intern aus 'tas' berechnet; kein direkter Dateiname #
    CMIP6_LEVEL = 850 #
    CMIP6_PRE_INDUSTRIAL_REF_START = 1850 #
    CMIP6_PRE_INDUSTRIAL_REF_END = 1900 #
    CMIP6_ANOMALY_REF_START = 1995 #
    CMIP6_ANOMALY_REF_END = 2014 #
    GWL_YEARS_WINDOW = 20 #
    GWL_TEMP_SMOOTHING_WINDOW = 20 #
    GWL_MAX_YEAR_PROC = 2300 # Maximales Jahr, bis zu dem Daten verarbeitet werden (für lange Szenarien) #
    GLOBAL_WARMING_LEVELS = [2.0, 3.0, 4.0] #
    MIN_MODELS_FOR_MMM = 3 # Mindestanzahl Modelle für einen validen MMM #

    # --- Storyline Definitions ---
    STORYLINE_JET_CHANGES = { #
        'DJF_JetSpeed': { #
            2.0: {'Core Mean': 0.4, 'Core High': 0.7, 'Extreme Low': -0.8, 'Extreme High': 1.5}, #
            3.0: {'Core Mean': 0.5, 'Core High': 1.0, 'Extreme Low': -0.6, 'Extreme High': 1.7}, #
            4.0: {'Core Mean': 0.6, 'Core High': 1.3, 'Extreme Low': -0.4, 'Extreme High': 1.9}, #
        },
        'JJA_JetLat': { #
            2.0: {'Core Mean': 0.5, 'Core High': 2.0, 'Extreme Low': -1.0, 'Extreme High': 3.0}, #
            3.0: {'Core Mean': 1.0, 'Core High': 2.2, 'Extreme Low': -0.5, 'Extreme High': 3.3}, #
            4.0: {'Core Mean': 1.2, 'Core High': 2.4, 'Extreme Low':  0.0, 'Extreme High': 3.6}, #
        }
    }

    # --- Output file naming/paths for intermediate results ---
    RESULTS_DIR = os.path.join(PROJECT_BASE_DIR, 'results_intermediate') #
    REANALYSIS_RESULTS_FILE = os.path.join(RESULTS_DIR, 'reanalysis_processed_data.pkl') #
    ERA5_BETA_SLOPES_FILE = os.path.join(RESULTS_DIR, 'era5_beta_obs_slopes.json') # Beibehalten, da aus Reanalyse #
    CMIP6_GWL_RESULTS_FILE = os.path.join(RESULTS_DIR, 'cmip6_gwl_analysis_full_results.pkl') #
    STORYLINE_IMPACTS_FILE = os.path.join(RESULTS_DIR, 'storyline_impacts.json') #

    @staticmethod
    def ensure_dir_exists(path): #
        if not os.path.exists(path): #
            os.makedirs(path, exist_ok=True) #
            logging.info(f"Created directory: {path}") #

    def __init__(self): #
        self.ensure_dir_exists(self.PLOT_DIR) #
        self.ensure_dir_exists(self.RESULTS_DIR) #