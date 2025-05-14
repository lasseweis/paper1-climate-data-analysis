#!/usr/bin/env python3
"""
Utilities for CMIP6 data analysis, Global Warming Level (GWL) calculations,
and storyline impact assessments.
"""
import logging
import os
import glob
# from fnmatch import fnmatch # fnmatch was in original, but glob handles patterns. Not strictly needed if using glob.
import numpy as np
import xarray as xr
import json # For logging storyline definitions
import traceback

# Relative imports for utility modules
from config_setup import Config
from data_utils import DataProcessor, select_level_preprocess # select_level_preprocess is used in a lambda
from jet_utils import JetStreamAnalyzer
# StatsAnalyzer might be indirectly used if DataProcessor.detrend_data calls it,
# but not directly by StorylineAnalyzer methods here.

class StorylineAnalyzer:
    """Class for analyzing CMIP6 data and calculating storyline impacts."""

    def __init__(self, config_instance: Config):
        """
        Initialize the StorylineAnalyzer.

        Args:
            config_instance (Config): An instance of the Config class.
        """
        self.config = config_instance
        # Initialize other analyzers if their methods are directly called by this class's instances.
        # Based on the provided code, DataProcessor and JetStreamAnalyzer methods are called as static
        # or instance methods but not necessarily needing instances stored in self here,
        # unless specific state from them is required across methods.
        # For now, we'll call them statically or create instances as needed locally if that's simpler.
        # However, the original ClimateAnalysis class instantiated them, so for consistency:
        self.data_processor = DataProcessor() # If methods are static, this instance isn't strictly needed
        self.jet_analyzer = JetStreamAnalyzer() # Same as above

    def _find_cmip_files(self, variable_name, experiment_name, model_name='*',
                         member_id='*', grid_name='*', base_path_override=None):
        """
        Finds CMIP6 files matching the criteria, searching in variable-specific subdirectories.

        Args:
            variable_name (str): The CMIP6 variable name (e.g., 'tas', 'pr', 'ua').
                                 If 'tas_global', it uses 'tas' for path and pattern.
            experiment_name (str): The experiment name (e.g., 'historical', 'ssp585').
            model_name (str, optional): Model name pattern. Defaults to '*'.
            member_id (str, optional): Ensemble member ID. Defaults to '*'.
            grid_name (str, optional): Grid name pattern. Defaults to '*'.
            base_path_override (str, optional): Override the base path for searching.

        Returns:
            list: A sorted list of unique file paths.
        """
        actual_var_for_path = 'tas' if variable_name == 'tas_global' else variable_name

        if base_path_override is None:
            # Path to the variable-specific subdirectory
            var_specific_path = self.config.CMIP6_VAR_PATH.format(variable=actual_var_for_path)
            # Fallback for 'tas' if not in its specific subdirectory (e.g., if all tas files are in root CMIP6_DATA_BASE_PATH)
            if actual_var_for_path == 'tas' and not os.path.exists(var_specific_path):
                logging.debug(f"Path {var_specific_path} not found for 'tas', trying CMIP6_DATA_BASE_PATH directly for 'tas' files.")
                search_base_path = self.config.CMIP6_DATA_BASE_PATH
            else:
                search_base_path = var_specific_path
        else:
            search_base_path = base_path_override

        file_pattern_template = self.config.CMIP6_FILE_PATTERN
        constructed_file_pattern = file_pattern_template.format(
            variable=actual_var_for_path, model=model_name, experiment=experiment_name,
            member=member_id, grid=grid_name # Removed start_date, end_date as they are usually part of '*'
        )

        search_glob_pattern = os.path.join(search_base_path, constructed_file_pattern)
        logging.debug(f"Searching for CMIP6 files with glob pattern: {search_glob_pattern}")
        found_files = glob.glob(search_glob_pattern)

        # Additional filtering if a specific member (not wildcard) was requested
        if member_id != '*':
            found_files = [f for f in found_files if f"_{member_id}_" in os.path.basename(f)]
        
        # Deduplicate and sort
        unique_sorted_files = sorted(list(set(found_files)))
        logging.debug(f"Found {len(unique_sorted_files)} files for {variable_name}, {experiment_name}, {model_name}.")
        return unique_sorted_files

    def _load_and_preprocess_model_data(self, model_name_str, scenarios_to_include,
                                        variable_name_str, target_level_hpa=None):
        """
        Loads, concatenates historical and scenario data, and preprocesses
        for a single CMIP6 model and variable.

        Args:
            model_name_str (str): The name of the CMIP6 model.
            scenarios_to_include (list): List of scenario names (e.g., ['ssp585']).
            variable_name_str (str): The variable to load ('ua', 'pr', 'tas', 'tas_global').
            target_level_hpa (int, optional): Target pressure level in hPa for 'ua'.

        Returns:
            xr.DataArray: Preprocessed DataArray for the model and variable, or None on failure.
        """
        logging.info(f"Loading CMIP6 data for variable '{variable_name_str}', model '{model_name_str}' "
                     f"(historical + scenarios: {scenarios_to_include})...")
        ensemble_member = self.config.CMIP6_MEMBER_ID
        all_model_files = []

        # Find historical files
        hist_files_list = self._find_cmip_files(variable_name_str, self.config.CMIP6_HISTORICAL_EXPERIMENT_NAME,
                                                model_name=model_name_str, member_id=ensemble_member)
        if not hist_files_list:
            logging.warning(f"  No historical files found for {variable_name_str}, {model_name_str}. GWL calculations might fail.")
        all_model_files.extend(hist_files_list)

        # Find scenario files
        for scenario_name in scenarios_to_include:
            scenario_files_list = self._find_cmip_files(variable_name_str, scenario_name,
                                                       model_name=model_name_str, member_id=ensemble_member)
            if not scenario_files_list:
                logging.warning(f"  No {scenario_name} files found for {variable_name_str}, {model_name_str}.")
            all_model_files.extend(scenario_files_list)

        if not all_model_files:
            logging.warning(f"--> No files found for {variable_name_str}, {model_name_str}. Skipping.")
            return None
        
        unique_files = sorted(list(set(all_model_files)))
        logging.info(f"  Found {len(unique_files)} unique files to load for {variable_name_str}, {model_name_str}.")

        combined_ds = None
        try:
            preprocess_func = None
            if variable_name_str == 'ua' and target_level_hpa is not None:
                preprocess_func = lambda ds_chunk: select_level_preprocess(ds_chunk, level_hpa=target_level_hpa)
                logging.info(f"  Using preprocessing to select level {target_level_hpa} hPa for 'ua'.")

            combined_ds = xr.open_mfdataset(
                unique_files,
                preprocess=preprocess_func,
                combine='by_coords', 
                parallel=False, 
                engine='netcdf4',
                decode_times=True,
                use_cftime=True,
                coords='minimal', 
                data_vars='minimal', 
                compat='override', 
                chunks={'time': 120} 
            )

            actual_var_in_ds = None
            if variable_name_str == 'tas_global': actual_var_in_ds = 'tas'
            elif variable_name_str == 'ua':
                actual_var_in_ds = next((name for name in ['ua', 'u', 'uwnd'] if name in combined_ds.data_vars), None)
            else: 
                actual_var_in_ds = variable_name_str
            
            if not actual_var_in_ds or actual_var_in_ds not in combined_ds.data_vars:
                alt_names_map = {'pr': ['prate'], 'tas': ['t2m', 'air']}
                if variable_name_str in alt_names_map:
                    for alt_name in alt_names_map[variable_name_str]:
                        if alt_name in combined_ds.data_vars:
                            logging.info(f"  Found variable as '{alt_name}', renaming to '{variable_name_str}'.")
                            combined_ds = combined_ds.rename({alt_name: variable_name_str})
                            actual_var_in_ds = variable_name_str
                            break
                if not actual_var_in_ds or actual_var_in_ds not in combined_ds.data_vars:
                    raise ValueError(f"Variable '{variable_name_str}' (or its aliases) not found in combined dataset for {model_name_str}.")
            
            data_array_var = combined_ds[actual_var_in_ds]

            coord_renames = {'latitude': 'lat', 'longitude': 'lon', 'plev': 'lev'}
            current_renames = {k: v for k, v in coord_renames.items() if k in data_array_var.dims or k in data_array_var.coords}
            if current_renames:
                data_array_var = data_array_var.rename(current_renames)

            if 'time' not in data_array_var.coords:
                raise ValueError("  Critical: 'time' coordinate missing after loading.")
            if not data_array_var.indexes['time'].is_unique:
                logging.warning(f"  Time coordinate for {model_name_str}/{variable_name_str} contains duplicates. Removing duplicates.")
                _, unique_indices = np.unique(data_array_var['time'], return_index=True)
                data_array_var = data_array_var.isel(time=unique_indices)
            if not data_array_var.indexes['time'].is_monotonic_increasing:
                logging.info(f"  Sorting time coordinate for {model_name_str}/{variable_name_str}.")
                data_array_var = data_array_var.sortby('time')
            
            data_array_var = data_array_var.assign_coords(
                year=("time", data_array_var.time.dt.year.values),
                month=("time", data_array_var.time.dt.month.values)
            )

            if variable_name_str == 'pr' and 'units' in data_array_var.attrs:
                if 'kg' in data_array_var.attrs['units'].lower() and 's-1' in data_array_var.attrs['units'].lower():
                    logging.info(f"  Converting precipitation units from {data_array_var.attrs['units']} to mm/day.")
                    data_array_var = data_array_var * 86400.0
                    data_array_var.attrs['units'] = 'mm/day'
            if variable_name_str in ['tas', 'tas_global'] and 'units' in data_array_var.attrs:
                if data_array_var.attrs['units'].lower() in ['k', 'kelvin']:
                    logging.info(f"  Converting {variable_name_str} units from K to °C.")
                    data_array_var = data_array_var - 273.15
                    data_array_var.attrs['units'] = '°C'

            if variable_name_str in ['pr', 'tas', 'ua']: 
                if 'lon' in data_array_var.coords and np.any(data_array_var['lon'] > 180):
                    logging.info(f"  Normalizing longitude from 0-360 to -180-180 for {model_name_str}/{variable_name_str}.")
                    data_array_var = data_array_var.assign_coords(lon=(((data_array_var.lon + 180) % 360) - 180)).sortby('lon')
            
            # ----- START MODIFICATION FOR tas_global -----
            if variable_name_str == 'tas_global':
                if 'lat' in data_array_var.dims and 'lon' in data_array_var.dims:
                    logging.info(f"  Calculating global mean for tas_global ({model_name_str})...")
                    weights = np.cos(np.deg2rad(data_array_var.lat))
                    weights.name = "weights"
                    global_mean_tas = data_array_var.weighted(weights).mean(("lon", "lat"), skipna=True)
                    
                    logging.debug(f"  DEBUG (_load_and_preprocess_model_data, {model_name_str}): Global mean tas_global before explicit processing: {global_mean_tas.dims}, {global_mean_tas.shape}")

                    target_dims_for_global_mean = ['time']
                    current_dims_of_global_mean = list(global_mean_tas.dims)
                    dims_to_explicitly_remove = [d for d in current_dims_of_global_mean if d not in target_dims_for_global_mean]

                    if dims_to_explicitly_remove:
                        logging.info(f"  Explicitly removing dimensions {dims_to_explicitly_remove} and their coordinates from global_mean_tas for {model_name_str}.")
                        coords_to_drop_with_dims = [coord for coord in global_mean_tas.coords if coord in dims_to_explicitly_remove and coord != 'time']
                        if coords_to_drop_with_dims:
                            try:
                                global_mean_tas = global_mean_tas.drop_vars(coords_to_drop_with_dims, errors='ignore')
                                logging.debug(f"    Dropped coordinates: {coords_to_drop_with_dims}")
                            except Exception as e_drop_vars:
                                logging.warning(f"    Could not drop coordinates {coords_to_drop_with_dims}: {e_drop_vars}")
                        
                        global_mean_tas = global_mean_tas.squeeze(dim=dims_to_explicitly_remove, drop=True)
                        logging.debug(f"  DEBUG (_load_and_preprocess_model_data, {model_name_str}): Global mean tas_global after explicit processing: {global_mean_tas.dims}, {global_mean_tas.shape}")

                    if set(global_mean_tas.dims) != set(target_dims_for_global_mean):
                         raise ValueError(f"Global mean 'tas_global' for {model_name_str} could not be reduced to 1D time series. Dims: {global_mean_tas.dims}")
                    data_array_var = global_mean_tas
                elif not ('lat' in data_array_var.dims and 'lon' in data_array_var.dims): # If already (seemingly) globally averaged
                    logging.warning(f"  'tas_global' for {model_name_str} already seems globally averaged or is missing lat/lon dims. "
                                    f"Current dims: {data_array_var.dims}. Ensuring it's 1D (time).")
                    non_time_dims = [dim for dim in data_array_var.dims if dim != 'time']
                    if non_time_dims:
                        # Drop associated coordinates first if they are not dimensions anymore
                        coords_to_drop = [coord for coord in data_array_var.coords if coord in non_time_dims and coord != 'time']
                        if coords_to_drop:
                            data_array_var = data_array_var.drop_vars(coords_to_drop, errors='ignore')
                        data_array_var = data_array_var.squeeze(dim=non_time_dims, drop=True)
                    if set(data_array_var.dims) != {'time'}:
                        raise ValueError(f"Pre-averaged 'tas_global' for {model_name_str} could not be reduced to 1D. Dims: {data_array_var.dims}")
            # ----- END MODIFICATION FOR tas_global -----
            
            max_year_to_keep = 2300 
            if 'time' in data_array_var.coords:
                original_time_size = data_array_var.time.size
                years_present = data_array_var.time.dt.year
                mask_upto_max_year = (years_present <= max_year_to_keep)
                if mask_upto_max_year.any():
                    data_array_var = data_array_var.sel(time=data_array_var.time[mask_upto_max_year])
                    if data_array_var.time.size < original_time_size:
                        logging.info(f"  Filtered timeseries up to year {max_year_to_keep}.")
                else:
                    logging.warning(f"  No data found up to year {max_year_to_keep} for {model_name_str}/{variable_name_str}.")
            
            if data_array_var is not None and data_array_var.size > 0:
                # ----- START MODIFICATION: ADD model_name ATTRIBUTE -----
                data_array_var.attrs['model_name'] = model_name_str
                # ----- END MODIFICATION: ADD model_name ATTRIBUTE -----
                logging.info(f"  Successfully loaded and preprocessed {variable_name_str} for {model_name_str}. Time range: "
                             f"{data_array_var.time.min().dt.strftime('%Y-%m').item()} to {data_array_var.time.max().dt.strftime('%Y-%m').item()}")
                return data_array_var.load() 
            else:
                logging.warning(f"  No data after preprocessing for {variable_name_str}, {model_name_str}.")
                return None

        except Exception as e:
            logging.error(f"  FATAL ERROR loading/preprocessing {variable_name_str} for {model_name_str}: {e}")
            logging.error(traceback.format_exc())
            return None
        finally:
            if combined_ds is not None:
                combined_ds.close()


    @staticmethod
    def calculate_gwl_thresholds(model_global_mean_tas_da: xr.DataArray,
                                 pre_industrial_period_tuple: tuple,
                                 smoothing_window_years: int,
                                 global_warming_levels_list: list):
        """
        Calculates the first year when specific global warming levels (GWLs) are exceeded.

        Args:
            model_global_mean_tas_da (xr.DataArray): 1D DataArray of global mean temperature anomaly
                                                     (or absolute temperature) for a single model, with 'time' or 'year' coordinate.
            pre_industrial_period_tuple (tuple): (start_year, end_year) for pre-industrial reference.
            smoothing_window_years (int): Window size in years for rolling mean smoothing.
            global_warming_levels_list (list): List of GWLs to check (e.g., [1.5, 2.0, 3.0]).

        Returns:
            dict: {gwl_value (float): first_year_exceeded (int) or None if not exceeded}.
                  Returns None if calculation fails.
        """
        gwl_crossing_years = {gwl: None for gwl in global_warming_levels_list}
        
        if model_global_mean_tas_da is None or model_global_mean_tas_da.size == 0:
            logging.error("  GWL Thresholds: Input global mean temperature DataArray is None or empty.")
            return None

        # ----- START MODIFICATION: ADD model_name AND SAFETY CHECK FOR SPATIAL MEAN -----
        model_name_for_log = model_global_mean_tas_da.attrs.get('model_name', 'Unknown Model')
        logging.info(f"  GWL Thresholds: Processing model '{model_name_for_log}'") # Logging des Modellnamens
        
        global_tas_for_gwl = model_global_mean_tas_da.copy() # Arbeite mit einer Kopie

        # Zusätzlicher Sicherheitscheck für räumliche Mittelung (inspiriert von paper1.py)
        spatial_dims_found = [dim for dim in ['lat', 'lon', 'latitude', 'longitude'] if dim in global_tas_for_gwl.dims]
        if spatial_dims_found:
            logging.warning(f"  GWL Thresholds ({model_name_for_log}): Input 'model_global_mean_tas_da' hat noch räumliche Dimensionen: {spatial_dims_found}. Versuche erneute Mittelung.")
            try:
                # Versuche, die Standard-Koordinatennamen zu verwenden
                lat_coord_name = 'lat' if 'lat' in global_tas_for_gwl.coords else 'latitude' if 'latitude' in global_tas_for_gwl.coords else None
                lon_coord_name = 'lon' if 'lon' in global_tas_for_gwl.coords else 'longitude' if 'longitude' in global_tas_for_gwl.coords else None

                if lat_coord_name and lon_coord_name and lat_coord_name in global_tas_for_gwl.dims and lon_coord_name in global_tas_for_gwl.dims:
                    weights_check = np.cos(np.deg2rad(global_tas_for_gwl[lat_coord_name]))
                    weights_check.name = "weights"
                    # Stelle sicher, dass Gewichte nur von der Lat-Dimension abhängen
                    if set(weights_check.dims) == {lat_coord_name}:
                        global_tas_mean_check = global_tas_for_gwl.weighted(weights_check).mean(dim=[lon_coord_name, lat_coord_name], skipna=True)
                    else:
                        logging.warning(f"  GWL Thresholds ({model_name_for_log}): Gewichte haben unerwartete Dimensionen {weights_check.dims}. Nutze ungewichteten Mittelwert.")
                        global_tas_mean_check = global_tas_for_gwl.mean(dim=[lon_coord_name, lat_coord_name], skipna=True)
                else: # Generischer Fall, wenn lat/lon nicht als Koordinaten standardisiert sind oder nicht Dimensionen sind
                    logging.warning(f"  GWL Thresholds ({model_name_for_log}): Standard lat/lon Koordinaten/Dimensionen nicht für gewichteten Mittelwert gefunden. Nutze generischen Mittelwert über {spatial_dims_found}.")
                    global_tas_mean_check = global_tas_for_gwl.mean(dim=spatial_dims_found, skipna=True)
                
                global_tas_for_gwl = global_tas_mean_check
                logging.debug(f"  GWL Thresholds ({model_name_for_log}): Nach erneutem Mitteln, Dimensionen: {global_tas_for_gwl.dims}")
            except Exception as e_remiddle:
                logging.error(f"  GWL Thresholds ({model_name_for_log}): Fehler beim erneuten Mitteln: {e_remiddle}. Fahre mit potenziell ungemittelten Daten fort.")
        
        # Verwende global_tas_for_gwl für die weitere Berechnung
        # ----- END MODIFICATION -----

        # --- Beginn der angepassten Logik zur Erstellung von annual_mean_tas ---
        # global_tas_for_gwl kommt von _load_and_preprocess_model_data und ist monatlich
        # mit einer 'time'-Dimension und 'year' als zugewiesener Koordinate.
        # ODER wurde gerade oben nochmal gemittelt.

        if 'time' in global_tas_for_gwl.dims:
            logging.debug(f"  GWL Thresholds: Input data for model '{model_name_for_log}' has 'time' dimension. Calculating annual means using 'time.dt.year'.")
            try:
                annual_mean_tas = global_tas_for_gwl.groupby(global_tas_for_gwl.time.dt.year).mean(dim='time', skipna=True)
                
                if 'year' not in annual_mean_tas.dims:
                    if 'year' in annual_mean_tas.coords: 
                        logging.warning(f"  GWL Thresholds: 'year' is a coordinate but not a dimension after groupby for model '{model_name_for_log}'. Attempting to set_index.")
                        annual_mean_tas = annual_mean_tas.set_index(year='year') 
                    else:
                        logging.error(f"  GWL Thresholds: 'year' dimension could not be established after annual mean calculation for model '{model_name_for_log}'. Dims: {annual_mean_tas.dims}")
                        return None 
                elif 'year' not in annual_mean_tas.coords: 
                     logging.warning(f"  GWL Thresholds: 'year' is a dimension but not a coordinate after groupby for model '{model_name_for_log}'. Attempting to assign coordinate.")
                     annual_mean_tas = annual_mean_tas.assign_coords(year=annual_mean_tas.year)

            except Exception as e_annual:
                logging.error(f"  GWL Thresholds: Failed to calculate annual mean from time series for model '{model_name_for_log}': {e_annual}")
                return None
        elif 'year' in global_tas_for_gwl.dims:
            logging.debug(f"  GWL Thresholds: Input data for model '{model_name_for_log}' already has 'year' dimension. Assuming it's annual.")
            annual_mean_tas = global_tas_for_gwl
            if 'year' not in annual_mean_tas.coords:
                logging.warning(f"  GWL Thresholds: 'year' is a dimension but not a coordinate for model '{model_name_for_log}'. Assigning coordinate from dimension.")
                annual_mean_tas = annual_mean_tas.assign_coords(year=annual_mean_tas.year)
        else:
            logging.error(f"  GWL Thresholds: Input TAS DataArray for model '{model_name_for_log}' needs a 'time' or 'year' dimension. Dims: {global_tas_for_gwl.dims}")
            return None
        # --- Ende der angepassten Logik ---

        ref_start, ref_end = pre_industrial_period_tuple
        try:
            tas_pre_industrial_slice = annual_mean_tas.sel(year=slice(ref_start, ref_end))
            if tas_pre_industrial_slice.year.size == 0:
                min_yr_data, max_yr_data = annual_mean_tas.year.min().item(), annual_mean_tas.year.max().item()
                logging.error(f"  GWL Thresholds: No data in pre-industrial reference period ({ref_start}-{ref_end}) for model '{model_name_for_log}'. "
                              f"Data available: {min_yr_data}-{max_yr_data}.")
                return None
            
            # ----- START MODIFICATION: DEBUG BEFORE .item() -----
            baseline_mean_da = tas_pre_industrial_slice.mean(dim='year', skipna=True)
            logging.debug(f"  GWL DEBUG ({model_name_for_log}): Baseline mean DataArray before .item(): {baseline_mean_da}")
            logging.debug(f"  GWL DEBUG ({model_name_for_log}): Dimensions: {baseline_mean_da.dims}, Shape: {baseline_mean_da.shape}, Size: {baseline_mean_da.size}")
            
            if baseline_mean_da.size != 1:
                logging.error(f"  GWL ERROR ({model_name_for_log}): Baseline mean is not a scalar before .item()! Size: {baseline_mean_da.size}. Data: {baseline_mean_da.data}")
                # Option: Versuchen, trotzdem einen Wert zu nehmen, wenn es nur eine nicht-Zeit Dimension ist, die Singleton ist
                if baseline_mean_da.ndim == 1 and baseline_mean_da.size > 0 : # Z.B. wenn eine `member` Dimension übrig blieb
                    logging.warning(f"  GWL WARNING ({model_name_for_log}): Baseline mean is 1D with size {baseline_mean_da.size}. Taking the first element.")
                    pre_industrial_baseline_temp = baseline_mean_da.data[0] # Vorsichtiger Zugriff
                else:
                    return None # Sicherer Ausstieg
            else:
                pre_industrial_baseline_temp = baseline_mean_da.item()
            # ----- END MODIFICATION: DEBUG BEFORE .item() -----

        except Exception as e_baseline:
            logging.error(f"  GWL Thresholds: Error calculating pre-industrial baseline for model '{model_name_for_log}': {e_baseline}")
            logging.error(traceback.format_exc()) # Hinzugefügt für mehr Details
            return None

        temperature_anomaly = annual_mean_tas - pre_industrial_baseline_temp
        
        if 'year' not in temperature_anomaly.dims:
            logging.error(f"  GWL Thresholds: 'year' is not a dimension in temperature_anomaly for model '{model_name_for_log}'. Cannot apply rolling mean. Dims: {temperature_anomaly.dims}")
            return gwl_crossing_years 

        smoothed_anomaly = temperature_anomaly.rolling(year=smoothing_window_years, center=True).mean().dropna(dim='year')
        if smoothed_anomaly.size == 0:
            logging.warning(f"  GWL Thresholds: Smoothed anomaly series is empty for model '{model_name_for_log}' (e.g., not enough data for rolling window).")
            return gwl_crossing_years 

        try:
            for gwl_target in global_warming_levels_list:
                if 'year' not in smoothed_anomaly.coords:
                    logging.error(f"  GWL Thresholds: 'year' coordinate missing in smoothed_anomaly for model '{model_name_for_log}'.")
                    continue 

                years_exceeding_gwl = smoothed_anomaly.where(smoothed_anomaly > gwl_target, drop=True).year
                if years_exceeding_gwl.size > 0:
                    first_year = int(years_exceeding_gwl.min().item())
                    gwl_crossing_years[gwl_target] = first_year
                    logging.info(f"      GWL {gwl_target}°C first exceeded in year: {first_year} for model '{model_name_for_log}'")
                else:
                    logging.info(f"      GWL {gwl_target}°C not exceeded in the smoothed timeseries for model '{model_name_for_log}'.")
            return gwl_crossing_years
        except Exception as e_find_gwl:
            logging.error(f"  GWL Thresholds: Error finding GWL crossing years for model '{model_name_for_log}': {e_find_gwl}")
            logging.error(traceback.format_exc()) # Hinzugefügt für mehr Details
            return None


    def _extract_gwl_period_mean(self, index_timeseries_da, model_gwl_threshold_years,
                                 gwl_value, years_window_for_mean):
        """
        Extracts the N-year mean of an index_timeseries centered around a GWL threshold year.

        Args:
            index_timeseries_da (xr.DataArray): 1D DataArray with 'season_year' coordinate.
            model_gwl_threshold_years (dict): {gwl: year} for the specific model.
            gwl_value (float): The Global Warming Level.
            years_window_for_mean (int): Number of years for the averaging window.

        Returns:
            float: Mean value over the GWL window, or np.nan if calculation fails.
        """
        crossing_year = model_gwl_threshold_years.get(gwl_value)
        if crossing_year is None or index_timeseries_da is None or index_timeseries_da.size == 0:
            # logging.debug(f"      _extract_gwl_period_mean: GWL {gwl_value}°C not reached or index timeseries missing.")
            return np.nan

        if 'season_year' not in index_timeseries_da.coords:
            logging.error(f"      _extract_gwl_period_mean: 'season_year' coordinate missing in index_timeseries for GWL {gwl_value}.")
            return np.nan

        # Ensure index_timeseries_da is 1D
        if index_timeseries_da.ndim > 1:
            dims_to_squeeze = [d for d in index_timeseries_da.dims if d != 'season_year']
            if dims_to_squeeze:
                index_timeseries_da = index_timeseries_da.squeeze(dims_to_squeeze, drop=True)
            if index_timeseries_da.ndim > 1:
                logging.error(f"      _extract_gwl_period_mean: Index timeseries could not be reduced to 1D ('season_year'). Dims: {index_timeseries_da.dims}")
                return np.nan
        
        start_avg_year = crossing_year - years_window_for_mean // 2
        end_avg_year = crossing_year + (years_window_for_mean - 1) // 2 # Inclusive end year

        try:
            mean_slice = index_timeseries_da.sel(season_year=slice(start_avg_year, end_avg_year))
            if mean_slice.season_year.size == 0:
                logging.warning(f"      Warning (_extract_gwl_period_mean): No data points in window [{start_avg_year}-{end_avg_year}] "
                                f"for GWL {gwl_value}. Index: {index_timeseries_da.name if index_timeseries_da.name else 'Unnamed'}. "
                                f"Model GWL year: {crossing_year}.")
                return np.nan
            
            # Warn if window is not fully populated
            if mean_slice.season_year.size < years_window_for_mean * 0.5: # Less than half the expected points
                 logging.warning(f"      Warning (_extract_gwl_period_mean): Only {mean_slice.season_year.size}/{years_window_for_mean} "
                                 f"data points in window [{start_avg_year}-{end_avg_year}] for GWL {gwl_value}. "
                                 f"Index: {index_timeseries_da.name if index_timeseries_da.name else 'Unnamed'}.")

            mean_at_gwl = mean_slice.mean(dim='season_year', skipna=True).item()
            return mean_at_gwl
        except Exception as e:
            index_name = index_timeseries_da.name if index_timeseries_da.name else 'Unnamed_Index'
            logging.error(f"      Error extracting mean for GWL {gwl_value} ({start_avg_year}-{end_avg_year}) "
                          f"from index '{index_name}': {e}")
            return np.nan


    def analyze_cmip6_changes_at_gwl(self, list_of_models_to_process=None):
        """
        Analyzes CMIP6 changes (Jet indices, Pr, Tas) at specified GWLs.
        This is a high-level orchestrator for CMIP6 GWL analysis.

        Args:
            list_of_models_to_process (list, optional): Specific models to analyze.
                                                        If None, attempts all found models.
        Returns:
            dict: Comprehensive results including GWL thresholds, MMM changes,
                  individual model changes, and loaded raw data.
        """
        logging.info("\n--- Starting CMIP6 Analysis at Global Warming Levels (GWLs) ---")
        
        # --- 1. Find and load data for all required variables and models ---
        logging.info("Step 1: Loading and preprocessing CMIP6 model data...")
        all_vars_to_load = self.config.CMIP6_VARIABLES_TO_LOAD + [self.config.CMIP6_GLOBAL_TAS_VAR]
        # Ensure unique variables if global_tas_var is already in variables_to_load
        all_vars_to_load = sorted(list(set(all_vars_to_load)))

        # Determine which models to process
        # This part can be complex: find models that have ALL required variables.
        # For simplicity here, we assume _load_and_preprocess_model_data will return None if a var is missing for a model.
        # A more robust initial scan for models could be implemented if needed.
        if list_of_models_to_process is None:
            # Dynamically find potential models (e.g., by listing subdirs in CMIP6_DATA_BASE_PATH or parsing filenames)
            # This is a placeholder for actual model discovery logic.
            # For now, we rely on trying to load data for a predefined list or all known good models.
            # Let's assume a simple discovery based on the first variable's path
            first_var_path_template = self.config.CMIP6_VAR_PATH.format(variable=all_vars_to_load[0])
            if os.path.exists(first_var_path_template):
                 # This example assumes model names are not part of CMIP6_VAR_PATH but are found by _find_cmip_files
                 # This part needs a robust way to get a list of all potential model names.
                 # For now, we'll rely on the main script to pass a list or handle model discovery.
                 logging.warning("No specific model list provided to analyze_cmip6_changes_at_gwl. "
                                 "Full model discovery logic is complex and not fully implemented here. "
                                 "Ensure the calling script provides a model list or handles discovery.")
                 # As a fallback, if _find_cmip_files with model='*' can return all models, that would be used.
                 # Let's assume it can for now for the sake of proceeding.
                 pass # Proceed, _load_and_preprocess will try for models found by _find_cmip_files if model_name='*'

        loaded_model_data = {} # {model_name: {variable_name: xr.DataArray}}
        global_tas_data_per_model = {} # {model_name: xr.DataArray_global_tas}

        # If no specific models are given, _load_and_preprocess will try to find files for model='*'
        # which might be slow or not what's intended if _find_cmip_files isn't designed for it.
        # It's better if the calling routine (e.g. main script) determines the models to run.
        # For this utility, if list_of_models_to_process is None, it implies a broader search.

        # For this method to be robust, it should iterate over a list of known/discoverable model names.
        # Let's assume the main analysis script will provide this list. If not, this step is weak.
        # Here, we'll simulate it by attempting to load for a dummy model if none provided, just to show flow.
        if list_of_models_to_process is None:
            logging.warning("No model list provided to analyze_cmip6_changes_at_gwl. This function works best with an explicit list.")
            # Placeholder: if you had a way to list all model directories, you'd use that.
            # For now, we'll rely on the _find_cmip_files to handle model='*' if this is the case.
            # This is NOT ideal. The main script should manage the model list.
            # Let's assume for now the user intends to run for all discoverable models if None.
            # The current _find_cmip_files takes model='*' and returns files from all models matching.
            # _load_and_preprocess_model_data is per-model, so it needs a model name.

            # This logic needs to be in the calling script or main_workflow.
            # This method should receive the list of models to process.
            # For now, if list_of_models_to_process is None, we cannot proceed effectively.
            logging.error("CRITICAL: analyze_cmip6_changes_at_gwl requires a list of models to process.")
            return {"error": "No models specified for CMIP6 GWL analysis."}


        for model_name in list_of_models_to_process:
            current_model_data = {}
            all_vars_successfully_loaded_for_model = True

            # Load global TAS first
            global_tas = self._load_and_preprocess_model_data(
                model_name, self.config.CMIP6_SCENARIOS, self.config.CMIP6_GLOBAL_TAS_VAR
            )
            if global_tas is None:
                logging.warning(f"  --> Failed to load critical global tas for model {model_name}. Skipping this model.")
                continue # Skip to next model
            global_tas_data_per_model[model_name] = global_tas

            # Load other regional variables
            for var_name in self.config.CMIP6_VARIABLES_TO_LOAD:
                var_data = self._load_and_preprocess_model_data(
                    model_name, self.config.CMIP6_SCENARIOS, var_name,
                    target_level_hpa=(self.config.CMIP6_LEVEL if var_name == 'ua' else None)
                )
                if var_data is None:
                    logging.warning(f"  --> Failed to load regional variable '{var_name}' for model {model_name}. This model will be excluded from some analyses.")
                    all_vars_successfully_loaded_for_model = False
                    # Decide if to break or just mark this var as missing for the model
                    # break # If any regional var fails, model might be unusable for full analysis
                current_model_data[var_name] = var_data
            
            if all_vars_successfully_loaded_for_model: # Or some more nuanced check
                loaded_model_data[model_name] = current_model_data
            else: # Clean up if model is incomplete
                if model_name in global_tas_data_per_model:
                    del global_tas_data_per_model[model_name]


        analyzed_model_names = list(loaded_model_data.keys())
        if not analyzed_model_names:
            logging.error("CMIP6 GWL Analysis: No CMIP6 models could be fully processed (global TAS + regional vars). Stopping.")
            return {}
        logging.info(f"Step 1 Completed: Successfully loaded data for {len(analyzed_model_names)} models: {analyzed_model_names}")

        # --- 2. Calculate GWL thresholds for each model ---
        logging.info("\nStep 2: Calculating GWL thresholds per model...")
        gwl_thresholds_all_models = {}
        for model_name_iter, gtas_data_iter in global_tas_data_per_model.items():
            logging.info(f"  Calculating GWL thresholds for model: {model_name_iter}")
            thresholds = self.calculate_gwl_thresholds(
                gtas_data_iter,
                (self.config.CMIP6_PRE_INDUSTRIAL_REF_START, self.config.CMIP6_PRE_INDUSTRIAL_REF_END),
                self.config.GWL_TEMP_SMOOTHING_WINDOW,
                self.config.GLOBAL_WARMING_LEVELS
            )
            if thresholds is not None:
                gwl_thresholds_all_models[model_name_iter] = thresholds
            else:
                logging.warning(f"    Could not determine GWL thresholds for model: {model_name_iter}. This model will be excluded from GWL-specific stats.")
        
        # Filter models to only those for which GWL thresholds could be calculated
        models_with_gwl_thresholds = [m for m in analyzed_model_names if m in gwl_thresholds_all_models]
        if not models_with_gwl_thresholds:
            logging.error("CMIP6 GWL Analysis: No models have valid GWL thresholds. Cannot proceed with GWL-specific analysis.")
            return {'cmip6_model_data_loaded_raw': loaded_model_data, 'gwl_threshold_years': gwl_thresholds_all_models}


        # --- 3. Calculate time series of all required metrics for each model ---
        logging.info("\nStep 3: Calculating full time series of metrics for models with GWL thresholds...")
        model_metrics_timeseries_all = {} # {model: {metric_name: DataArray_timeseries}}
        regional_box_coords = (self.config.BOX_LAT_MIN, self.config.BOX_LAT_MAX,
                               self.config.BOX_LON_MIN, self.config.BOX_LON_MAX)

        for model_name_iter in models_with_gwl_thresholds:
            logging.info(f"  Calculating metric time series for: {model_name_iter}")
            model_metrics_timeseries_all[model_name_iter] = {}
            model_regional_data = loaded_model_data[model_name_iter] # Data for 'ua', 'pr', 'tas'

            try:
                ua_monthly_ts = model_regional_data.get('ua')
                pr_monthly_ts = model_regional_data.get('pr')
                tas_monthly_ts = model_regional_data.get('tas')

                if ua_monthly_ts is None or ua_monthly_ts.size == 0 or \
                   pr_monthly_ts is None or pr_monthly_ts.size == 0 or \
                   tas_monthly_ts is None or tas_monthly_ts.size == 0:
                    logging.warning(f"    Skipping metric calculation for {model_name_iter}: Missing or empty regional variable timeseries (ua, pr, or tas).")
                    continue

                # Convert monthly to seasonal means (time series over all years)
                ua_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(ua_monthly_ts))
                pr_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(pr_monthly_ts))
                tas_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(tas_monthly_ts))

                if ua_seas_mean_fullts is None or ua_seas_mean_fullts.size == 0 or \
                   pr_seas_mean_fullts is None or pr_seas_mean_fullts.size == 0 or \
                   tas_seas_mean_fullts is None or tas_seas_mean_fullts.size == 0:
                    logging.warning(f"    Skipping metric calculation for {model_name_iter}: Failed to calculate or result is empty for one or more base seasonal mean time series.")
                    continue
                
                # Calculate Box Means (time series over all years)
                pr_box_mean_fullts = self.data_processor.calculate_spatial_mean(pr_seas_mean_fullts, *regional_box_coords)
                tas_box_mean_fullts = self.data_processor.calculate_spatial_mean(tas_seas_mean_fullts, *regional_box_coords)

                # Calculate Jet Indices (time series over all years)
                ua_winter_fullts = self.data_processor.filter_by_season(ua_seas_mean_fullts, 'Winter')
                ua_summer_fullts = self.data_processor.filter_by_season(ua_seas_mean_fullts, 'Summer')
                
                model_metrics_timeseries_all[model_name_iter]['DJF_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_winter_fullts)
                model_metrics_timeseries_all[model_name_iter]['JJA_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_summer_fullts)
                model_metrics_timeseries_all[model_name_iter]['DJF_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_winter_fullts)
                model_metrics_timeseries_all[model_name_iter]['JJA_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_summer_fullts)
                
                # Extract seasonal box means (time series over all years)
                model_metrics_timeseries_all[model_name_iter]['DJF_pr']  = self.data_processor.filter_by_season(pr_box_mean_fullts, 'Winter') if pr_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['JJA_pr']  = self.data_processor.filter_by_season(pr_box_mean_fullts, 'Summer') if pr_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['DJF_tas'] = self.data_processor.filter_by_season(tas_box_mean_fullts, 'Winter') if tas_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['JJA_tas'] = self.data_processor.filter_by_season(tas_box_mean_fullts, 'Summer') if tas_box_mean_fullts is not None else None
                
                # Log if any metric is None (problem in calculation)
                for metric_key, metric_val_ts in model_metrics_timeseries_all[model_name_iter].items():
                    if metric_val_ts is None or metric_val_ts.size == 0:
                        logging.warning(f"    Metric time series for '{metric_key}' in model {model_name_iter} is None or empty.")
            except Exception as e_metric_calc:
                logging.error(f"    ERROR calculating full metric time series for {model_name_iter}: {e_metric_calc}")
                if model_name_iter in model_metrics_timeseries_all: # Remove partial data for this model
                    del model_metrics_timeseries_all[model_name_iter]


        # --- 4. Calculate means at GWLs and reference period using the metric time series ---
        logging.info("\nStep 4: Calculating metric means at GWLs and CMIP6 reference period...")
        model_metric_values_at_gwl_and_ref = {} # {model: {'ref': {metric: val}, GWL1: {metric: val}, GWL2: ...}}
        cmip6_reference_period_means = {}    # {model: {metric: val}}
        
        metrics_for_gwl_avg = ['DJF_JetSpeed', 'JJA_JetSpeed', 'DJF_JetLat', 'JJA_JetLat',
                               'DJF_pr', 'JJA_pr', 'DJF_tas', 'JJA_tas']
        cmip6_ref_start = self.config.CMIP6_ANOMALY_REF_START
        cmip6_ref_end = self.config.CMIP6_ANOMALY_REF_END

        for model_name_iter in models_with_gwl_thresholds:
            if model_name_iter not in model_metrics_timeseries_all:
                logging.info(f"  Skipping GWL/Ref means for {model_name_iter}: No full metric time series available.")
                continue
            logging.info(f"  Processing GWL/Ref means for model: {model_name_iter}")

            current_model_results = {'ref': {}} # For CMIP6 reference period
            for gwl in self.config.GLOBAL_WARMING_LEVELS:
                current_model_results[gwl] = {} # For each GWL period

            # Calculate CMIP6 Reference Period Means
            all_ref_metrics_valid_for_model = True
            for metric_name in metrics_for_gwl_avg:
                metric_full_ts = model_metrics_timeseries_all[model_name_iter].get(metric_name)
                if metric_full_ts is None or 'season_year' not in metric_full_ts.coords:
                    current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid_for_model = False; continue
                try:
                    ref_slice = metric_full_ts.sel(season_year=slice(cmip6_ref_start, cmip6_ref_end))
                    if ref_slice.season_year.size == 0: current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid_for_model = False; continue
                    current_model_results['ref'][metric_name] = ref_slice.mean(dim='season_year', skipna=True).item()
                except Exception: current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid_for_model = False;
            
            if all_ref_metrics_valid_for_model and not any(np.isnan(v) for v in current_model_results['ref'].values()):
                cmip6_reference_period_means[model_name_iter] = current_model_results['ref']
            else:
                logging.warning(f"    Reference period means for {model_name_iter} are incomplete or contain NaN.")
                # current_model_results['ref'] will contain NaNs or be partially filled

            # Calculate GWL Period Means
            for gwl_target_val in self.config.GLOBAL_WARMING_LEVELS:
                all_metrics_valid_for_this_gwl = True
                for metric_name in metrics_for_gwl_avg:
                    metric_full_ts = model_metrics_timeseries_all[model_name_iter].get(metric_name)
                    gwl_mean_val = self._extract_gwl_period_mean(
                        metric_full_ts,
                        gwl_thresholds_all_models[model_name_iter],
                        gwl_target_val,
                        self.config.GWL_YEARS_WINDOW
                    )
                    current_model_results[gwl_target_val][metric_name] = gwl_mean_val
                    if np.isnan(gwl_mean_val): all_metrics_valid_for_this_gwl = False
                if not all_metrics_valid_for_this_gwl:
                     logging.warning(f"    GWL {gwl_target_val}°C means for {model_name_iter} are incomplete or contain NaN.")
            
            model_metric_values_at_gwl_and_ref[model_name_iter] = current_model_results
        
        # --- 5. Calculate Multi-Model Mean (MMM) changes ---
        logging.info("\nStep 5: Calculating CMIP6 Multi-Model Mean (MMM) changes at GWLs...")
        mmm_gwl_changes = {gwl: {} for gwl in self.config.GLOBAL_WARMING_LEVELS}
        
        for gwl_target_val in self.config.GLOBAL_WARMING_LEVELS:
            all_models_gwl_metric_data = {metric: [] for metric in metrics_for_gwl_avg}
            all_models_ref_metric_data = {metric: [] for metric in metrics_for_gwl_avg}
            contributing_models_for_gwl_mmm = []

            for model_name_iter in models_with_gwl_thresholds:
                if model_name_iter in model_metric_values_at_gwl_and_ref and \
                   model_name_iter in cmip6_reference_period_means: # Ensure model has valid ref period means
                    
                    model_specific_ref_means = cmip6_reference_period_means[model_name_iter]
                    model_specific_gwl_data = model_metric_values_at_gwl_and_ref[model_name_iter].get(gwl_target_val, {})
                    
                    # Check if all metrics are non-NaN for this model, for this ref and this GWL
                    if all(m_name in model_specific_ref_means and m_name in model_specific_gwl_data and \
                           not np.isnan(model_specific_ref_means[m_name]) and not np.isnan(model_specific_gwl_data[m_name]) \
                           for m_name in metrics_for_gwl_avg):
                        contributing_models_for_gwl_mmm.append(model_name_iter)
                        for metric_name in metrics_for_gwl_avg:
                            all_models_gwl_metric_data[metric_name].append(model_specific_gwl_data[metric_name])
                            all_models_ref_metric_data[metric_name].append(model_specific_ref_means[metric_name])
            
            num_contrib_models = len(contributing_models_for_gwl_mmm)
            if num_contrib_models >= 3: # Threshold for robust MMM
                logging.info(f"  Calculating MMM for GWL {gwl_target_val}°C based on {num_contrib_models} models: {contributing_models_for_gwl_mmm}")
                mmm_gwl_changes[gwl_target_val]['contributing_models'] = contributing_models_for_gwl_mmm
                mmm_gwl_changes[gwl_target_val]['model_count'] = num_contrib_models

                for metric_name in metrics_for_gwl_avg:
                    mmm_value_at_gwl = np.mean(all_models_gwl_metric_data[metric_name])
                    mmm_value_at_ref = np.mean(all_models_ref_metric_data[metric_name])
                    
                    delta_change_mmm = np.nan
                    if metric_name.endswith('_pr'): # Precipitation change as percentage
                        if abs(mmm_value_at_ref) > 1e-9: # Avoid division by zero
                            delta_change_mmm = ((mmm_value_at_gwl - mmm_value_at_ref) / mmm_value_at_ref) * 100.0
                    else: # Absolute change for other variables (TAS, Jet indices)
                        delta_change_mmm = mmm_value_at_gwl - mmm_value_at_ref
                    
                    mmm_gwl_changes[gwl_target_val][metric_name] = delta_change_mmm # Store the change
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_mean_at_ref_mmm"] = mmm_value_at_ref
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_mean_at_gwl_mmm"] = mmm_value_at_gwl
                    
                    # Store all individual model deltas for spread calculation if needed later
                    ref_array_all = np.array(all_models_ref_metric_data[metric_name])
                    gwl_array_all = np.array(all_models_gwl_metric_data[metric_name])
                    deltas_all_indiv_models = np.full_like(ref_array_all, np.nan)
                    if metric_name.endswith('_pr'):
                        with np.errstate(divide='ignore', invalid='ignore'): # Handle potential division by zero for individual models
                            deltas_all_indiv_models = np.where(np.abs(ref_array_all) > 1e-9, ((gwl_array_all - ref_array_all) / ref_array_all) * 100.0, np.nan)
                    else:
                        deltas_all_indiv_models = gwl_array_all - ref_array_all
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_all_model_deltas"] = deltas_all_indiv_models
            else:
                logging.info(f"  Skipping MMM calculation for GWL {gwl_target_val}°C: Only {num_contrib_models} valid models found (minimum 3 required).")
                mmm_gwl_changes[gwl_target_val] = None # Indicate MMM calculation failed for this GWL
        
        # --- Compile final results ---
        final_cmip6_results = {
            'cmip6_model_raw_data_loaded': loaded_model_data, # Full timeseries data loaded initially
            'cmip6_global_tas_per_model': global_tas_data_per_model, # Global TAS for each model
            'gwl_threshold_years_per_model': gwl_thresholds_all_models, # {model: {gwl: year}}
            'cmip6_metric_timeseries_per_model': model_metrics_timeseries_all, # {model: {metric: full_ts}}
            'cmip6_metric_values_at_gwl_and_ref_per_model': model_metric_values_at_gwl_and_ref, # {model: {'ref': {met:val}, gwl: {met:val}}}
            'cmip6_mmm_changes_at_gwl': mmm_gwl_changes # {gwl: {metric: change_val, ...}}
        }
        logging.info("\n--- CMIP6 Analysis at GWLs Completed ---")
        return final_cmip6_results


    def calculate_storyline_impacts(self, cmip6_gwl_analysis_results, observed_beta_slopes):
        """
        Calculates the impact changes for each defined storyline using Eq. 1 from Harvey et al. (2023).
        Delta_Impact_storyline = Delta_Impact_MMM + beta_obs * (Delta_Jet_storyline - Delta_Jet_MMM)

        Args:
            cmip6_gwl_analysis_results (dict): Output from analyze_cmip6_changes_at_gwl.
                                              Needs 'cmip6_mmm_changes_at_gwl'.
            observed_beta_slopes (dict): Dictionary of observed regression slopes (from reanalysis).
                                         Keys like 'DJF_JetSpeed_vs_tas'.

        Returns:
            dict: Impacts per storyline, GWL, and impact variable.
                  e.g., {2.0: {'DJF_pr': {'Core Mean': -5.0, 'Core High': -8.2, ...}, ...}}
        """
        logging.info("\n--- Calculating Storyline Impacts ---")
        if not cmip6_gwl_analysis_results or 'cmip6_mmm_changes_at_gwl' not in cmip6_gwl_analysis_results:
            logging.error("ERROR: Cannot calculate storyline impacts. Missing CMIP6 MMM change results.")
            return None
        if not observed_beta_slopes:
            logging.error("ERROR: Cannot calculate storyline impacts. Missing observed_beta_slopes from reanalysis.")
            return None
        if not self.config.STORYLINE_JET_CHANGES: # Storyline definitions from Config
            logging.error("ERROR: Cannot calculate storyline impacts. Config.STORYLINE_JET_CHANGES is empty.")
            return None

        storyline_impact_results = {gwl: {} for gwl in self.config.GLOBAL_WARMING_LEVELS}
        storyline_jet_definitions = self.config.STORYLINE_JET_CHANGES # e.g. {'DJF_JetSpeed': {2.0: {'Core Mean': val}}}
        cmip6_mmm_changes = cmip6_gwl_analysis_results['cmip6_mmm_changes_at_gwl']

        # Define which impact variables are driven by which jet indices and their corresponding beta_obs keys.
        # This mapping is crucial for applying the correct storyline adjustments.
        impact_variable_to_driver_map = {
            # Winter Impacts
            'DJF_pr':  {'driving_jet_index': 'DJF_JetSpeed', 'beta_obs_key': 'DJF_JetSpeed_vs_pr'},
            'DJF_tas': {'driving_jet_index': 'DJF_JetSpeed', 'beta_obs_key': 'DJF_JetSpeed_vs_tas'},
            # Summer Impacts
            'JJA_pr':  {'driving_jet_index': 'JJA_JetLat',   'beta_obs_key': 'JJA_JetLat_vs_pr'},
            'JJA_tas': {'driving_jet_index': 'JJA_JetLat',   'beta_obs_key': 'JJA_JetLat_vs_tas'},
            # Additional optional mappings if storylines for other jet drivers are defined:
            # 'DJF_pr':  {'driving_jet_index': 'DJF_JetLat', 'beta_obs_key': 'DJF_JetLat_vs_pr'},
            # 'JJA_tas': {'driving_jet_index': 'JJA_JetSpeed', 'beta_obs_key': 'JJA_JetSpeed_vs_tas'},
        }
        # User should ensure Config.STORYLINE_JET_CHANGES has entries for the 'driving_jet_index' names used above.
        # User should ensure observed_beta_slopes has entries for the 'beta_obs_key' names used above.

        logging.info("Using the following Beta_obs slopes (from reanalysis):")
        for bk, bv in observed_beta_slopes.items(): logging.info(f"  {bk}: {bv:.3f}" if bv is not None and not np.isnan(bv) else f"  {bk}: None/NaN")
        logging.info("Using the following Storyline Jet Change definitions from Config:")
        logging.info(json.dumps(storyline_jet_definitions, indent=2))


        for gwl_value in self.config.GLOBAL_WARMING_LEVELS:
            if cmip6_mmm_changes.get(gwl_value) is None: # Check if MMM results exist for this GWL
                logging.info(f"  Skipping storyline impact calculation for GWL {gwl_value}°C: No valid CMIP6 MMM changes.")
                continue
            logging.info(f"\n  Processing Storyline Impacts for GWL {gwl_value}°C...")

            for impact_var_name, driver_info in impact_variable_to_driver_map.items():
                driving_jet_index_name = driver_info['driving_jet_index'] # e.g., 'DJF_JetSpeed'
                beta_obs_key_for_driver = driver_info['beta_obs_key']   # e.g., 'DJF_JetSpeed_vs_pr'

                # Get CMIP6 MMM change for the current impact variable (e.g., Delta_PR_MMM)
                delta_impact_var_mmm = cmip6_mmm_changes[gwl_value].get(impact_var_name)
                # Get CMIP6 MMM change for the driving jet index (e.g., Delta_JetSpeed_MMM)
                delta_driving_jet_mmm = cmip6_mmm_changes[gwl_value].get(driving_jet_index_name)
                
                # Get the observed beta slope (sensitivity)
                beta_slope_obs = observed_beta_slopes.get(beta_obs_key_for_driver)

                # --- Input Sanity Checks ---
                inputs_are_valid = True
                if delta_impact_var_mmm is None or np.isnan(delta_impact_var_mmm):
                    logging.warning(f"    - Skipping {impact_var_name}: Missing or NaN CMIP6 MMM change for this impact variable at GWL {gwl_value}°C.")
                    inputs_are_valid = False
                if delta_driving_jet_mmm is None or np.isnan(delta_driving_jet_mmm):
                    logging.warning(f"    - Skipping {impact_var_name}: Missing or NaN CMIP6 MMM change for driving jet '{driving_jet_index_name}' at GWL {gwl_value}°C.")
                    inputs_are_valid = False
                if beta_slope_obs is None or np.isnan(beta_slope_obs):
                    logging.warning(f"    - Skipping {impact_var_name}: Missing or NaN observed beta slope for key '{beta_obs_key_for_driver}'.")
                    inputs_are_valid = False
                
                # Check if storyline definitions exist for this driving jet index and GWL
                if driving_jet_index_name not in storyline_jet_definitions or \
                   gwl_value not in storyline_jet_definitions[driving_jet_index_name]:
                    logging.warning(f"    - Skipping {impact_var_name}: Storyline definitions missing in Config for "
                                   f"driving jet '{driving_jet_index_name}' at GWL {gwl_value}°C.")
                    inputs_are_valid = False
                
                if not inputs_are_valid:
                    continue # Move to the next impact variable or GWL

                # --- Calculate impact for each storyline type (Core Mean, Core High, etc.) ---
                logging.info(f"    Calculating storyline impacts for '{impact_var_name}' (driven by '{driving_jet_index_name}')")
                storyline_impact_results[gwl_value][impact_var_name] = {} # Initialize dict for this impact var

                for storyline_type_name, delta_jet_storyline_value in storyline_jet_definitions[driving_jet_index_name][gwl_value].items():
                    # Equation: Delta_Impact_story = Delta_Impact_MMM + beta_obs * (Delta_Jet_story - Delta_Jet_MMM)
                    impact_adjustment_term = beta_slope_obs * (delta_jet_storyline_value - delta_driving_jet_mmm)
                    final_storyline_impact_change = delta_impact_var_mmm + impact_adjustment_term

                    storyline_impact_results[gwl_value][impact_var_name][storyline_type_name] = final_storyline_impact_change
                    logging.info(f"      {storyline_type_name:<12}: "
                                 f"Imp_MMM={delta_impact_var_mmm:+.2f}, "
                                 f"beta={beta_slope_obs:+.2f} * (Jet_story={delta_jet_storyline_value:+.2f} - Jet_MMM={delta_driving_jet_mmm:+.2f}) "
                                 f"= Adj_Term={impact_adjustment_term:+.2f} -> Total_Impact={final_storyline_impact_change:+.2f}")
            
            if not storyline_impact_results[gwl_value]: # If no impacts were calculated for this GWL (e.g. all skipped)
                del storyline_impact_results[gwl_value]


        logging.info("\n--- Storyline Impact Calculation Completed ---")
        # Filter out GWLs for which no impacts were calculated at all
        final_storyline_impacts_filtered = {gwl: impacts for gwl, impacts in storyline_impact_results.items() if impacts}
        return final_storyline_impacts_filtered