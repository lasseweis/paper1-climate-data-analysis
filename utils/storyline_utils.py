#!/usr/bin/env python3
"""
Utilities for CMIP6 data analysis, Global Warming Level (GWL) calculations,
and storyline impact assessments.
"""
import logging
import os
import glob
import numpy as np
import xarray as xr
import json 
import traceback

# Relative imports for utility modules
from config_setup import Config
from data_utils import DataProcessor, select_level_preprocess 
from jet_utils import JetStreamAnalyzer

# xesmf und die globalen Regridding-Variablen/Funktionen werden nicht mehr benötigt,
# da das Regridding extern mit CDO erfolgen soll.

class StorylineAnalyzer:
    """Class for analyzing CMIP6 data and calculating storyline impacts."""

    def __init__(self, config_instance: Config):
        """
        Initialize the StorylineAnalyzer.

        Args:
            config_instance (Config): An instance of the Config class.
        """
        self.config = config_instance
        self.data_processor = DataProcessor() 
        self.jet_analyzer = JetStreamAnalyzer() 

    def _find_cmip_files(self, variable_name, experiment_name, model_name='*',
                         member_id='*', grid_name='*', base_path_override=None):
        """
        Finds CMIP6 files matching the criteria.
        This function now assumes it's looking for pre-regridded files if Config is set accordingly.
        """
        actual_var_for_path = 'tas' if variable_name == 'tas_global' else variable_name

        if base_path_override is None:
            # CMIP6_VAR_PATH in Config sollte auf die _regrid Ordner zeigen
            search_base_path = self.config.CMIP6_VAR_PATH.format(variable=actual_var_for_path)
            # Fallback, falls der spezifische _regrid Ordner nicht existiert (sollte aber)
            if not os.path.exists(search_base_path):
                 logging.warning(f"Path {search_base_path} not found. Trying CMIP6_DATA_BASE_PATH directly.")
                 search_base_path = self.config.CMIP6_DATA_BASE_PATH
        else:
            search_base_path = base_path_override

        # CMIP6_FILE_PATTERN in Config sollte zu den Namen der regriddeten Dateien passen
        # (z.B. mit einem Suffix wie *_regridded.nc)
        file_pattern_template = self.config.CMIP6_FILE_PATTERN 
        constructed_file_pattern = file_pattern_template.format(
            variable=actual_var_for_path, model=model_name, experiment=experiment_name,
            member=member_id, grid=grid_name 
        )

        search_glob_pattern = os.path.join(search_base_path, constructed_file_pattern)
        logging.debug(f"Searching for PRE-REGRIDDED CMIP6 files with glob pattern: {search_glob_pattern}")
        found_files = sorted(list(set(glob.glob(search_glob_pattern)))) 

        # Zusätzliche Filterung, falls nötig (z.B. wenn member_id nicht Teil des glob-Musters war)
        if member_id != '*' and member_id not in self.config.CMIP6_FILE_PATTERN: # Wenn member_id nicht im Muster ist
            found_files = [f for f in found_files if f"_{member_id}_" in os.path.basename(f)]
        
        logging.debug(f"Found {len(found_files)} pre-regridded files for {variable_name}, {experiment_name}, {model_name}.")
        return found_files

    def _load_and_preprocess_model_data(self, model_name_str, scenarios_to_include,
                                        variable_name_str, target_level_hpa=None):
        """
        Loads, concatenates historical and scenario data from PRE-REGRIDDED files.
        """
        logging.info(f"Loading PRE-REGRIDDED CMIP6 data for variable '{variable_name_str}', model '{model_name_str}' "
                     f"(historical + scenarios: {scenarios_to_include})...")
        ensemble_member = self.config.CMIP6_MEMBER_ID
        all_model_files = []

        # Dateisuche für historische Daten (im regriddeten Verzeichnis)
        hist_files_list = self._find_cmip_files(variable_name_str, self.config.CMIP6_HISTORICAL_EXPERIMENT_NAME,
                                                model_name=model_name_str, member_id=ensemble_member)
        if not hist_files_list:
            logging.warning(f"  No historical pre-regridded files found for {variable_name_str}, {model_name_str}.")
        all_model_files.extend(hist_files_list)

        # Dateisuche für Szenariodaten (im regriddeten Verzeichnis)
        for scenario_name in scenarios_to_include:
            scenario_files_list = self._find_cmip_files(variable_name_str, scenario_name,
                                                       model_name=model_name_str, member_id=ensemble_member)
            if not scenario_files_list:
                logging.warning(f"  No {scenario_name} pre-regridded files found for {variable_name_str}, {model_name_str}.")
            all_model_files.extend(scenario_files_list)

        if not all_model_files:
            logging.warning(f"--> No pre-regridded files found for {variable_name_str}, {model_name_str}. Skipping.")
            return None
        
        unique_files = sorted(list(set(all_model_files)))
        logging.info(f"  Found {len(unique_files)} unique pre-regridded files to load for {variable_name_str}, {model_name_str}.")

        combined_ds = None
        data_array_var = None
        try:
            preprocess_func = None
            if variable_name_str == 'ua' and target_level_hpa is not None:
                preprocess_func = lambda ds_chunk: select_level_preprocess(ds_chunk, level_hpa=target_level_hpa)
            
            combined_ds = xr.open_mfdataset(
                unique_files, preprocess=preprocess_func, combine='by_coords', 
                parallel=self.config.N_PROCESSES > 1, engine='netcdf4', decode_times=True, use_cftime=True,
                coords='minimal', data_vars='minimal', compat='override', chunks={'time': 120}
            )

            actual_var_in_ds = variable_name_str if variable_name_str != 'tas_global' else 'tas'
            if variable_name_str == 'ua': 
                actual_var_in_ds = next((name for name in ['ua', 'u', 'uwnd'] if name in combined_ds.data_vars), actual_var_in_ds)

            if actual_var_in_ds not in combined_ds.data_vars:
                 alt_names_map = {'pr': ['prate'], 'tas': ['t2m', 'air']}
                 if variable_name_str in alt_names_map:
                     for alt_name in alt_names_map[variable_name_str]:
                         if alt_name in combined_ds.data_vars:
                             logging.info(f"  Found variable as '{alt_name}', renaming to '{variable_name_str}'.")
                             combined_ds = combined_ds.rename({alt_name: variable_name_str})
                             actual_var_in_ds = variable_name_str
                             break
                 if actual_var_in_ds not in combined_ds.data_vars:
                    raise ValueError(f"Variable '{actual_var_in_ds}' (mapped from '{variable_name_str}') not found in pre-regridded files for {model_name_str}.")

            data_array_var = combined_ds[actual_var_in_ds]
            data_array_var.attrs['model_name'] = model_name_str
            data_array_var.attrs['grid_info'] = "Assumed pre-regridded by CDO to reference grid"

            # Standardisierung der Koordinatennamen
            # CDO benennt Koordinaten normalerweise konsistent (lon, lat), aber eine Prüfung ist gut.
            coord_renames = {}
            if 'longitude' in data_array_var.coords and 'lon' not in data_array_var.coords : coord_renames['longitude'] = 'lon'
            if 'latitude' in data_array_var.coords and 'lat' not in data_array_var.coords: coord_renames['latitude'] = 'lat'
            if 'plev' in data_array_var.coords and 'lev' not in data_array_var.coords: coord_renames['plev'] = 'lev'
            if coord_renames:
                data_array_var = data_array_var.rename(coord_renames)

            # Zeitliche Verarbeitung
            if 'time' not in data_array_var.coords: raise ValueError("Time coordinate missing.")
            if not data_array_var.indexes['time'].is_unique:
                logging.warning(f"  Time coordinate for {model_name_str}/{variable_name_str} (pre-regridded) contains duplicates. Removing.")
                _, unique_indices = np.unique(data_array_var['time'], return_index=True)
                data_array_var = data_array_var.isel(time=unique_indices)
            if not data_array_var.indexes['time'].is_monotonic_increasing:
                logging.info(f"  Sorting time coordinate for {model_name_str}/{variable_name_str} (pre-regridded).")
                data_array_var = data_array_var.sortby('time')
            
            data_array_var = data_array_var.assign_coords(
                year=("time", data_array_var.time.dt.year.values),
                month=("time", data_array_var.time.dt.month.values)
            )

            # Einheitenkonvertierung
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

            # Längengrad-Normalisierung und -Sortierung:
            # Da die Daten mit CDO auf ein Referenzgitter regridded wurden, sollten die Längengrade
            # bereits im Bereich -180 bis 180 (oder 0-360, je nach CDO-Zielgitter) und sortiert sein.
            # Ein Sicherheitscheck und eine eventuelle Normalisierung/Sortierung sind dennoch gut.
            if 'lon' in data_array_var.coords:
                lon_values = data_array_var.lon.values
                needs_lon_transform = False
                if np.any(lon_values > 180.1) or np.any(lon_values < -180.1): # Toleranz für leichte Ungenauigkeiten
                    logging.info(f"  Normalizing longitude for pre-regridded {model_name_str}/{variable_name_str} from 0-360 to -180-180 range (safety check).")
                    data_array_var = data_array_var.assign_coords(lon=(((data_array_var.lon + 180) % 360) - 180))
                    needs_lon_transform = True # Nach Transformation immer sortieren

                if needs_lon_transform or not data_array_var.indexes['lon'].is_monotonic_increasing:
                    logging.info(f"  Sorting 'lon' for pre-regridded {model_name_str}/{variable_name_str} (safety check).")
                    data_array_var = data_array_var.sortby('lon')
            
            if 'lat' in data_array_var.coords and not (data_array_var.indexes['lat'].is_monotonic_increasing or data_array_var.indexes['lat'].is_monotonic_decreasing):
                logging.info(f"  Sorting 'lat' for pre-regridded {model_name_str}/{variable_name_str} (safety check).")
                data_array_var = data_array_var.sortby('lat')

            # Finale Monotonie-Prüfung im Log
            if 'lon' in data_array_var.coords:
                is_mono_final_check = data_array_var.indexes['lon'].is_monotonic_increasing or data_array_var.indexes['lon'].is_monotonic_decreasing
                logging.info(f"  Final monotonicity check for {model_name_str}/{variable_name_str} (pre-regridded): lon monotonic: {is_mono_final_check}. Lons: {data_array_var.lon.values[:3]}...{data_array_var.lon.values[-3:]}")
                if not is_mono_final_check: 
                    logging.error(f"    LON IS STILL NON-MONOTONIC for {model_name_str}/{variable_name_str} after loading pre-regridded data!")

            # Verarbeitung von tas_global (arbeitet auf den nun regriddeten 'tas' Daten)
            if variable_name_str == 'tas_global':
                if 'lat' in data_array_var.dims and 'lon' in data_array_var.dims:
                    logging.info(f"  Calculating global mean for tas_global ({model_name_str}) from pre-regridded data.")
                    weights = np.cos(np.deg2rad(data_array_var.lat))
                    weights.name = "weights"
                    global_mean_tas = data_array_var.weighted(weights).mean(("lon", "lat"), skipna=True)
                    
                    target_dims_for_global_mean = ['time']
                    current_dims_of_global_mean = list(global_mean_tas.dims)
                    dims_to_explicitly_remove = [d for d in current_dims_of_global_mean if d not in target_dims_for_global_mean]

                    if dims_to_explicitly_remove:
                        coords_to_drop_with_dims = [coord for coord in global_mean_tas.coords if coord in dims_to_explicitly_remove and coord != 'time']
                        if coords_to_drop_with_dims:
                            try: global_mean_tas = global_mean_tas.drop_vars(coords_to_drop_with_dims, errors='ignore')
                            except Exception : pass # Ignoriere Fehler beim Droppen von Koordinaten
                        global_mean_tas = global_mean_tas.squeeze(dim=dims_to_explicitly_remove, drop=True)
                    
                    if set(global_mean_tas.dims) != set(target_dims_for_global_mean):
                         raise ValueError(f"Global mean 'tas_global' for {model_name_str} could not be reduced to 1D. Dims: {global_mean_tas.dims}")
                    data_array_var = global_mean_tas
                elif not ('lat' in data_array_var.dims and 'lon' in data_array_var.dims): 
                    logging.warning(f"  'tas_global' for {model_name_str} (pre-regridded) already seems globally averaged or is missing lat/lon dims. Dims: {data_array_var.dims}.")
                    non_time_dims = [dim for dim in data_array_var.dims if dim != 'time']
                    if non_time_dims:
                        coords_to_drop = [coord for coord in data_array_var.coords if coord in non_time_dims and coord != 'time']
                        if coords_to_drop: data_array_var = data_array_var.drop_vars(coords_to_drop, errors='ignore')
                        data_array_var = data_array_var.squeeze(dim=non_time_dims, drop=True)
                    if set(data_array_var.dims) != {'time'}:
                        raise ValueError(f"Pre-averaged 'tas_global' for {model_name_str} (pre-regridded) not 1D. Dims: {data_array_var.dims}")

            # Zeitfilterung
            max_year_to_keep = self.config.GWL_MAX_YEAR_PROC 
            if 'time' in data_array_var.coords:
                original_time_size = data_array_var.time.size
                data_array_var = data_array_var.sel(time=data_array_var.time.dt.year <= max_year_to_keep)
                if data_array_var.time.size < original_time_size:
                    logging.info(f"  Filtered timeseries up to year {max_year_to_keep}.")
                if data_array_var.time.size == 0 :
                    logging.warning(f"  No data remaining for {model_name_str}/{variable_name_str} after filtering up to year {max_year_to_keep}.")
                    return None # Wichtig, um leere Arrays zu vermeiden
            
            if data_array_var is not None and data_array_var.size > 0:
                logging.info(f"  Successfully loaded and preprocessed pre-regridded {variable_name_str} for {model_name_str}. Time range: "
                             f"{data_array_var.time.min().dt.strftime('%Y-%m').item()} to {data_array_var.time.max().dt.strftime('%Y-%m').item()}")
                return data_array_var.load() 
            else:
                logging.warning(f"  No data after all processing for pre-regridded {variable_name_str}, {model_name_str}.")
                return None
        except Exception as e:
            logging.error(f"  FATAL ERROR in _load_and_preprocess_model_data for pre-regridded {variable_name_str}, {model_name_str}: {e}")
            logging.error(traceback.format_exc())
            return None
        finally:
            if combined_ds is not None: combined_ds.close()

    def calculate_gwl_thresholds(self, model_global_mean_tas_da: xr.DataArray,
                                 pre_industrial_period_tuple: tuple,
                                 smoothing_window_years: int,
                                 global_warming_levels_list: list):
        gwl_crossing_years = {gwl: None for gwl in global_warming_levels_list}
        if model_global_mean_tas_da is None or model_global_mean_tas_da.size == 0:
            logging.error("  GWL Thresholds: Input global mean temperature DataArray is None or empty.")
            return None

        model_name_for_log = model_global_mean_tas_da.attrs.get('model_name', 'Unknown Model')
        logging.info(f"  GWL Thresholds: Processing model '{model_name_for_log}'")
        
        global_tas_for_gwl = model_global_mean_tas_da.copy()

        spatial_dims_found = [dim for dim in ['lat', 'lon', 'latitude', 'longitude'] if dim in global_tas_for_gwl.dims]
        if spatial_dims_found:
            logging.warning(f"  GWL Thresholds ({model_name_for_log}): Input 'model_global_mean_tas_da' still has spatial dimensions: {spatial_dims_found}. This should have been handled in _load_and_preprocess_model_data for 'tas_global'.")
            try:
                lat_coord_name = next((c for c in ['lat', 'latitude'] if c in global_tas_for_gwl.coords and c in global_tas_for_gwl.dims), None)
                lon_coord_name = next((c for c in ['lon', 'longitude'] if c in global_tas_for_gwl.coords and c in global_tas_for_gwl.dims), None)
                if lat_coord_name and lon_coord_name:
                    weights_check = np.cos(np.deg2rad(global_tas_for_gwl[lat_coord_name]))
                    weights_check.name = "weights"
                    global_tas_for_gwl = global_tas_for_gwl.weighted(weights_check).mean(dim=[lon_coord_name, lat_coord_name], skipna=True)
                else:
                    global_tas_for_gwl = global_tas_for_gwl.mean(dim=spatial_dims_found, skipna=True)
                logging.debug(f"  GWL Thresholds ({model_name_for_log}): After emergency re-middling, dimensions: {global_tas_for_gwl.dims}")
            except Exception as e_remiddle:
                logging.error(f"  GWL Thresholds ({model_name_for_log}): Error during emergency re-middling: {e_remiddle}. GWL calculation likely to fail.")
                return None

        if 'time' in global_tas_for_gwl.dims:
            annual_mean_tas = global_tas_for_gwl.groupby(global_tas_for_gwl.time.dt.year).mean(dim='time', skipna=True)
            if 'year' not in annual_mean_tas.dims and 'year' in annual_mean_tas.coords:
                 annual_mean_tas = annual_mean_tas.set_index(year='year')
            elif 'year' not in annual_mean_tas.coords and 'year' in annual_mean_tas.dims: 
                 # xarray kann 'year_group' oder ähnliches erstellen, wenn die Jahreskoordinate nicht direkt 'year' heißt
                 # Versuche, die korrekte Jahreskoordinate zu finden und als Dimension zu setzen
                 year_coord_name_after_group = next((c for c in annual_mean_tas.coords if 'year' in c), None)
                 if year_coord_name_after_group and year_coord_name_after_group != 'year':
                     annual_mean_tas = annual_mean_tas.rename({year_coord_name_after_group: 'year'})
                 if 'year' in annual_mean_tas.coords and 'year' not in annual_mean_tas.dims:
                     annual_mean_tas = annual_mean_tas.set_index(year='year')
                 elif 'year' not in annual_mean_tas.coords and 'year' in annual_mean_tas.dims: # Fallback, wenn es eine Dimension aber keine Koordinate ist
                     annual_mean_tas = annual_mean_tas.assign_coords(year=annual_mean_tas.year)


        elif 'year' in global_tas_for_gwl.dims: 
            annual_mean_tas = global_tas_for_gwl
            if 'year' not in annual_mean_tas.coords : annual_mean_tas = annual_mean_tas.assign_coords(year=annual_mean_tas.year)
        else:
            logging.error(f"  GWL Thresholds: Input TAS for model '{model_name_for_log}' needs 'time' or 'year' dimension. Dims: {global_tas_for_gwl.dims}")
            return None
        
        if 'year' not in annual_mean_tas.dims : 
             logging.error(f"  GWL Thresholds: 'year' dimension could not be established for model '{model_name_for_log}'. Dims: {annual_mean_tas.dims}")
             return None

        ref_start, ref_end = pre_industrial_period_tuple
        try:
            tas_pre_industrial_slice = annual_mean_tas.sel(year=slice(ref_start, ref_end))
            if tas_pre_industrial_slice.year.size == 0:
                min_yr_data, max_yr_data = annual_mean_tas.year.min().item(), annual_mean_tas.year.max().item()
                logging.error(f"  GWL Thresholds: No data in pre-industrial reference period ({ref_start}-{ref_end}) for model '{model_name_for_log}'. Data available: {min_yr_data}-{max_yr_data}.")
                return None
            
            baseline_mean_da = tas_pre_industrial_slice.mean(dim='year', skipna=True)
            if baseline_mean_da.size != 1: 
                logging.error(f"  GWL ERROR ({model_name_for_log}): Baseline mean is not scalar! Size: {baseline_mean_da.size}. Data: {baseline_mean_da.data}")
                # Versuch, nicht-skalare Werte zu reduzieren, falls es sich um eine Singleton-Dimension handelt
                squeezable_dims = [dim for dim, size in baseline_mean_da.sizes.items() if size == 1 and dim != 'year'] # 'year' sollte schon weg sein
                if squeezable_dims: baseline_mean_da = baseline_mean_da.squeeze(dim=squeezable_dims, drop=True)
                if baseline_mean_da.size !=1 : return None 
            pre_industrial_baseline_temp = baseline_mean_da.item()

        except Exception as e_baseline:
            logging.error(f"  GWL Thresholds: Error calculating pre-industrial baseline for model '{model_name_for_log}': {e_baseline}")
            logging.error(traceback.format_exc())
            return None

        temperature_anomaly = annual_mean_tas - pre_industrial_baseline_temp
        
        if 'year' not in temperature_anomaly.dims: 
            logging.error(f"  GWL Thresholds: 'year' is not a dimension in temperature_anomaly for model '{model_name_for_log}'.")
            return gwl_crossing_years 

        min_periods_for_smoothing = max(1, smoothing_window_years // 2) # Erfordert mindestens die Hälfte der Fenstergröße
        smoothed_anomaly = temperature_anomaly.rolling(year=smoothing_window_years, center=True, min_periods=min_periods_for_smoothing).mean().dropna(dim='year')
        
        if smoothed_anomaly.size == 0:
            logging.warning(f"  GWL Thresholds: Smoothed anomaly series empty for model '{model_name_for_log}'. Not enough data for window {smoothing_window_years} with min_periods {min_periods_for_smoothing}.")
            return gwl_crossing_years 

        try:
            for gwl_target in global_warming_levels_list:
                if 'year' not in smoothed_anomaly.coords: 
                    logging.warning(f"   GWL Thresholds: 'year' coordinate missing in smoothed_anomaly for {model_name_for_log}. Skipping GWL {gwl_target}.")
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
            logging.error(traceback.format_exc())
            return None

    def _extract_gwl_period_mean(self, index_timeseries_da, model_gwl_threshold_years,
                                 gwl_value, years_window_for_mean):
        crossing_year = model_gwl_threshold_years.get(gwl_value)
        if crossing_year is None or index_timeseries_da is None or index_timeseries_da.size == 0:
            return np.nan

        if 'season_year' not in index_timeseries_da.coords:
            logging.error(f"      _extract_gwl_period_mean: 'season_year' coordinate missing for GWL {gwl_value}.")
            return np.nan

        if index_timeseries_da.ndim > 1: 
            dims_to_squeeze = [d for d in index_timeseries_da.dims if d != 'season_year' and index_timeseries_da.sizes[d] == 1]
            if dims_to_squeeze: index_timeseries_da = index_timeseries_da.squeeze(dims_to_squeeze, drop=True)
            if index_timeseries_da.ndim > 1: 
                logging.error(f"      _extract_gwl_period_mean: Index TS for GWL {gwl_value} not 1D after squeeze. Dims: {index_timeseries_da.dims}")
                return np.nan
        
        start_avg_year = crossing_year - years_window_for_mean // 2
        end_avg_year = crossing_year + (years_window_for_mean - 1) // 2

        try:
            mean_slice = index_timeseries_da.sel(season_year=slice(start_avg_year, end_avg_year))
            if mean_slice.season_year.size == 0: 
                logging.warning(f"      Warning (_extract_gwl_period_mean): No data in window [{start_avg_year}-{end_avg_year}] for GWL {gwl_value}.")
                return np.nan
            
            min_points_in_window = max(1, years_window_for_mean // 2)
            if mean_slice.season_year.size < min_points_in_window: 
                 logging.warning(f"      Warning (_extract_gwl_period_mean): Only {mean_slice.season_year.size}/{years_window_for_mean} "
                                 f"points in window [{start_avg_year}-{end_avg_year}] for GWL {gwl_value} (min_points: {min_points_in_window}).")
            
            return mean_slice.mean(dim='season_year', skipna=True).item()
        except Exception as e:
            logging.error(f"      Error extracting mean for GWL {gwl_value} ({start_avg_year}-{end_avg_year}): {e}")
            return np.nan

    def analyze_cmip6_changes_at_gwl(self, list_of_models_to_process=None):
        logging.info("\n--- Starting CMIP6 Analysis at Global Warming Levels (GWLs) ---")
        
        logging.info("Step 1: Loading and preprocessing CMIP6 model data (with CDO pre-regridding assumed)...")
        all_vars_to_load = self.config.CMIP6_VARIABLES_TO_LOAD + [self.config.CMIP6_GLOBAL_TAS_VAR]
        all_vars_to_load = sorted(list(set(all_vars_to_load)))

        if list_of_models_to_process is None:
            logging.error("CRITICAL: analyze_cmip6_changes_at_gwl requires a list of models to process.")
            return {"error": "No models specified for CMIP6 GWL analysis."}

        loaded_model_data = {} 
        global_tas_data_per_model = {} 

        for model_name in list_of_models_to_process:
            current_model_data = {}
            all_vars_ok_for_model = True
            
            # Lade globale TAS zuerst
            global_tas = self._load_and_preprocess_model_data(
                model_name, self.config.CMIP6_SCENARIOS, self.config.CMIP6_GLOBAL_TAS_VAR 
            )
            if global_tas is None:
                logging.warning(f"  --> Failed to load critical global tas for model {model_name}. Skipping this model for GWL analysis.")
                continue 
            global_tas_data_per_model[model_name] = global_tas

            for var_name in self.config.CMIP6_VARIABLES_TO_LOAD: # ua, pr, tas
                var_data = self._load_and_preprocess_model_data(
                    model_name, self.config.CMIP6_SCENARIOS, var_name,
                    target_level_hpa=(self.config.CMIP6_LEVEL if var_name == 'ua' else None)
                )
                if var_data is None:
                    logging.warning(f"  --> Failed to load regional var '{var_name}' for model {model_name}. Model might be excluded from some stats.")
                    all_vars_ok_for_model = False; break 
                current_model_data[var_name] = var_data
            
            if all_vars_ok_for_model:
                loaded_model_data[model_name] = current_model_data
            else: 
                if model_name in global_tas_data_per_model: del global_tas_data_per_model[model_name]
        
        analyzed_model_names = list(loaded_model_data.keys())
        if not analyzed_model_names:
            logging.error("CMIP6 GWL Analysis: No models could be fully processed. Stopping.")
            return {'cmip6_model_data_loaded_raw': loaded_model_data}

        logging.info(f"Step 1 Completed: Data loaded for {len(analyzed_model_names)} models: {analyzed_model_names}")

        logging.info("\nStep 2: Calculating GWL thresholds per model...")
        gwl_thresholds_all_models = {}
        valid_models_for_gwl_stats = []
        for model_name_iter, gtas_data_iter in global_tas_data_per_model.items():
            if model_name_iter not in analyzed_model_names: continue
            
            thresholds = self.calculate_gwl_thresholds(
                gtas_data_iter,
                (self.config.CMIP6_PRE_INDUSTRIAL_REF_START, self.config.CMIP6_PRE_INDUSTRIAL_REF_END),
                self.config.GWL_TEMP_SMOOTHING_WINDOW,
                self.config.GLOBAL_WARMING_LEVELS
            )
            if thresholds is not None and any(t is not None for t in thresholds.values()):
                gwl_thresholds_all_models[model_name_iter] = thresholds
                valid_models_for_gwl_stats.append(model_name_iter)
            else:
                logging.warning(f"    Could not determine valid GWL thresholds for model: {model_name_iter}. Excluding from GWL stats.")
        
        if not valid_models_for_gwl_stats:
            logging.error("CMIP6 GWL Analysis: No models have valid GWL thresholds. Cannot proceed.")
            return {
                'cmip6_model_raw_data_loaded': loaded_model_data, 
                'cmip6_global_tas_per_model': global_tas_data_per_model,
                'gwl_threshold_years_per_model': gwl_thresholds_all_models
            }

        logging.info("\nStep 3: Calculating full time series of metrics for models with GWL thresholds...")
        model_metrics_timeseries_all = {} 
        regional_box_coords_tuple = (self.config.BOX_LAT_MIN, self.config.BOX_LAT_MAX,
                                     self.config.BOX_LON_MIN, self.config.BOX_LON_MAX)

        for model_name_iter in valid_models_for_gwl_stats:
            logging.info(f"  Calculating metric time series for: {model_name_iter}")
            model_metrics_timeseries_all[model_name_iter] = {}
            model_regional_data = loaded_model_data[model_name_iter]

            try:
                ua_monthly_ts = model_regional_data.get('ua')
                pr_monthly_ts = model_regional_data.get('pr')
                tas_monthly_ts = model_regional_data.get('tas')

                if not all(ts is not None and ts.size > 0 for ts in [ua_monthly_ts, pr_monthly_ts, tas_monthly_ts]):
                    logging.warning(f"    Skipping metric calculation for {model_name_iter}: Missing/empty base timeseries.")
                    del model_metrics_timeseries_all[model_name_iter]; continue

                ua_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(ua_monthly_ts))
                pr_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(pr_monthly_ts))
                tas_seas_mean_fullts = self.data_processor.calculate_seasonal_means(self.data_processor.assign_season_to_dataarray(tas_monthly_ts))
                
                if not all(ts is not None and ts.size > 0 for ts in [ua_seas_mean_fullts, pr_seas_mean_fullts, tas_seas_mean_fullts]):
                     logging.warning(f"    Skipping metric calc for {model_name_iter}: Failed to get seasonal means.")
                     del model_metrics_timeseries_all[model_name_iter]; continue
                
                pr_box_mean_fullts = self.data_processor.calculate_spatial_mean(pr_seas_mean_fullts, *regional_box_coords_tuple)
                tas_box_mean_fullts = self.data_processor.calculate_spatial_mean(tas_seas_mean_fullts, *regional_box_coords_tuple)

                ua_winter_fullts = self.data_processor.filter_by_season(ua_seas_mean_fullts, 'Winter')
                ua_summer_fullts = self.data_processor.filter_by_season(ua_seas_mean_fullts, 'Summer')
                
                model_metrics_timeseries_all[model_name_iter]['DJF_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_winter_fullts)
                model_metrics_timeseries_all[model_name_iter]['JJA_JetSpeed'] = self.jet_analyzer.calculate_jet_speed_index(ua_summer_fullts)
                model_metrics_timeseries_all[model_name_iter]['DJF_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_winter_fullts)
                model_metrics_timeseries_all[model_name_iter]['JJA_JetLat'] = self.jet_analyzer.calculate_jet_lat_index(ua_summer_fullts)
                
                model_metrics_timeseries_all[model_name_iter]['DJF_pr']  = self.data_processor.filter_by_season(pr_box_mean_fullts, 'Winter') if pr_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['JJA_pr']  = self.data_processor.filter_by_season(pr_box_mean_fullts, 'Summer') if pr_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['DJF_tas'] = self.data_processor.filter_by_season(tas_box_mean_fullts, 'Winter') if tas_box_mean_fullts is not None else None
                model_metrics_timeseries_all[model_name_iter]['JJA_tas'] = self.data_processor.filter_by_season(tas_box_mean_fullts, 'Summer') if tas_box_mean_fullts is not None else None
                
                if any(val is None or (hasattr(val, 'size') and val.size == 0) for val in model_metrics_timeseries_all[model_name_iter].values()):
                     logging.warning(f"    One or more metric TS for {model_name_iter} are None/empty.")

            except Exception as e_metric_calc:
                logging.error(f"    ERROR calculating full metric time series for {model_name_iter}: {e_metric_calc}")
                if model_name_iter in model_metrics_timeseries_all: del model_metrics_timeseries_all[model_name_iter]
        
        final_models_for_mmm = [m for m in valid_models_for_gwl_stats if m in model_metrics_timeseries_all and 
                                model_metrics_timeseries_all[m] and 
                                not any(v is None or (hasattr(v,'size') and v.size==0) for v in model_metrics_timeseries_all[m].values())]
        
        if not final_models_for_mmm:
            logging.error("No models remaining with complete metric timeseries for MMM calculation.")
            return {
                'cmip6_model_raw_data_loaded': loaded_model_data,
                'cmip6_global_tas_per_model': global_tas_data_per_model,
                'gwl_threshold_years_per_model': gwl_thresholds_all_models,
                'cmip6_metric_timeseries_per_model': model_metrics_timeseries_all,
            }

        logging.info("\nStep 4: Calculating metric means at GWLs and CMIP6 reference period...")
        model_metric_values_at_gwl_and_ref = {} 
        cmip6_reference_period_means = {}    
        
        metrics_for_gwl_avg = ['DJF_JetSpeed', 'JJA_JetSpeed', 'DJF_JetLat', 'JJA_JetLat',
                               'DJF_pr', 'JJA_pr', 'DJF_tas', 'JJA_tas']
        cmip6_ref_start_year = self.config.CMIP6_ANOMALY_REF_START
        cmip6_ref_end_year = self.config.CMIP6_ANOMALY_REF_END

        for model_name_iter in final_models_for_mmm:
            logging.info(f"  Processing GWL/Ref means for model: {model_name_iter}")
            current_model_results = {'ref': {}}
            for gwl in self.config.GLOBAL_WARMING_LEVELS: current_model_results[gwl] = {}

            all_ref_metrics_valid = True
            for metric_name in metrics_for_gwl_avg:
                metric_full_ts = model_metrics_timeseries_all[model_name_iter].get(metric_name)
                if metric_full_ts is None or 'season_year' not in metric_full_ts.coords:
                    current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid = False; continue
                try:
                    ref_slice = metric_full_ts.sel(season_year=slice(cmip6_ref_start_year, cmip6_ref_end_year))
                    if ref_slice.season_year.size == 0: current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid = False; continue
                    current_model_results['ref'][metric_name] = ref_slice.mean(dim='season_year', skipna=True).item()
                except Exception: current_model_results['ref'][metric_name] = np.nan; all_ref_metrics_valid = False
            
            if all_ref_metrics_valid and not any(np.isnan(v) for v in current_model_results['ref'].values()):
                cmip6_reference_period_means[model_name_iter] = current_model_results['ref']
            else:
                 logging.warning(f"    Reference period means for {model_name_iter} are incomplete or NaN.")

            for gwl_target_val in self.config.GLOBAL_WARMING_LEVELS:
                all_metrics_valid_this_gwl = True
                for metric_name in metrics_for_gwl_avg:
                    metric_full_ts = model_metrics_timeseries_all[model_name_iter].get(metric_name)
                    # Sicherstellen, dass model_gwl_threshold_years für das aktuelle Modell existiert
                    model_gwl_years = gwl_thresholds_all_models.get(model_name_iter, {})

                    gwl_mean_val = self._extract_gwl_period_mean(
                        metric_full_ts, model_gwl_years, # Verwende die spezifischen Jahre des Modells
                        gwl_target_val, self.config.GWL_YEARS_WINDOW
                    )
                    current_model_results[gwl_target_val][metric_name] = gwl_mean_val
                    if np.isnan(gwl_mean_val): all_metrics_valid_this_gwl = False
                if not all_metrics_valid_this_gwl:
                     logging.warning(f"    GWL {gwl_target_val}°C means for {model_name_iter} are incomplete or NaN.")
            model_metric_values_at_gwl_and_ref[model_name_iter] = current_model_results
        
        logging.info("\nStep 5: Calculating CMIP6 Multi-Model Mean (MMM) changes at GWLs...")
        mmm_gwl_changes = {gwl: {} for gwl in self.config.GLOBAL_WARMING_LEVELS}
        
        for gwl_target_val in self.config.GLOBAL_WARMING_LEVELS:
            all_models_gwl_metric_data = {metric: [] for metric in metrics_for_gwl_avg}
            all_models_ref_metric_data = {metric: [] for metric in metrics_for_gwl_avg}
            contrib_models_this_gwl_mmm = []

            for model_name_iter in final_models_for_mmm: 
                if model_name_iter not in model_metric_values_at_gwl_and_ref or \
                   model_name_iter not in cmip6_reference_period_means: continue 
                    
                model_ref_means = cmip6_reference_period_means[model_name_iter]
                model_gwl_data = model_metric_values_at_gwl_and_ref[model_name_iter].get(gwl_target_val, {})
                
                # Prüfen ob GWL-Daten für dieses Modell überhaupt existieren und nicht leer sind
                if not model_gwl_data: continue

                if all(m_name in model_ref_means and m_name in model_gwl_data and \
                       not np.isnan(model_ref_means[m_name]) and not np.isnan(model_gwl_data[m_name]) \
                       for m_name in metrics_for_gwl_avg):
                    contrib_models_this_gwl_mmm.append(model_name_iter)
                    for metric_name in metrics_for_gwl_avg:
                        all_models_gwl_metric_data[metric_name].append(model_gwl_data[metric_name])
                        all_models_ref_metric_data[metric_name].append(model_ref_means[metric_name])
            
            num_contrib = len(contrib_models_this_gwl_mmm)
            if num_contrib >= self.config.MIN_MODELS_FOR_MMM:
                logging.info(f"  Calculating MMM for GWL {gwl_target_val}°C based on {num_contrib} models: {contrib_models_this_gwl_mmm}")
                mmm_gwl_changes[gwl_target_val]['contributing_models'] = contrib_models_this_gwl_mmm
                mmm_gwl_changes[gwl_target_val]['model_count'] = num_contrib

                for metric_name in metrics_for_gwl_avg:
                    mmm_val_gwl = np.mean(all_models_gwl_metric_data[metric_name])
                    mmm_val_ref = np.mean(all_models_ref_metric_data[metric_name])
                    
                    delta_mmm = np.nan
                    if metric_name.endswith('_pr'): 
                        if abs(mmm_val_ref) > 1e-9: delta_mmm = ((mmm_val_gwl - mmm_val_ref) / mmm_val_ref) * 100.0
                    else: 
                        delta_mmm = mmm_val_gwl - mmm_val_ref
                    
                    mmm_gwl_changes[gwl_target_val][metric_name] = delta_mmm
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_mean_at_ref_mmm"] = mmm_val_ref
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_mean_at_gwl_mmm"] = mmm_val_gwl
                    
                    ref_arr = np.array(all_models_ref_metric_data[metric_name])
                    gwl_arr = np.array(all_models_gwl_metric_data[metric_name])
                    deltas_indiv = np.full_like(ref_arr, np.nan)
                    if metric_name.endswith('_pr'):
                        with np.errstate(divide='ignore', invalid='ignore'): 
                            deltas_indiv = np.where(np.abs(ref_arr) > 1e-9, ((gwl_arr - ref_arr) / ref_arr) * 100.0, np.nan)
                    else:
                        deltas_indiv = gwl_arr - ref_arr
                    mmm_gwl_changes[gwl_target_val][f"{metric_name}_all_model_deltas"] = deltas_indiv
            else:
                logging.info(f"  Skipping MMM for GWL {gwl_target_val}°C: Only {num_contrib} valid models (min {self.config.MIN_MODELS_FOR_MMM} required).")
                mmm_gwl_changes[gwl_target_val] = None 
        
        final_cmip6_results = {
            'cmip6_model_raw_data_loaded': loaded_model_data, 
            'cmip6_global_tas_per_model': global_tas_data_per_model, 
            'gwl_threshold_years_per_model': gwl_thresholds_all_models, 
            'cmip6_metric_timeseries_per_model': model_metrics_timeseries_all, 
            'cmip6_metric_values_at_gwl_and_ref_per_model': model_metric_values_at_gwl_and_ref, 
            'cmip6_mmm_changes_at_gwl': mmm_gwl_changes 
        }
        logging.info("\n--- CMIP6 Analysis at GWLs Completed ---")
        return final_cmip6_results

    def calculate_storyline_impacts(self, cmip6_gwl_analysis_results, observed_beta_slopes):
        logging.info("\n--- Calculating Storyline Impacts ---")
        if not cmip6_gwl_analysis_results or 'cmip6_mmm_changes_at_gwl' not in cmip6_gwl_analysis_results:
            logging.error("ERROR: Cannot calculate storyline impacts. Missing CMIP6 MMM change results.")
            return None
        if not observed_beta_slopes: 
            logging.error("ERROR: Cannot calculate storyline impacts. Missing observed_beta_slopes from reanalysis.")
            return None
        if not self.config.STORYLINE_JET_CHANGES: 
            logging.error("ERROR: Cannot calculate storyline impacts. Config.STORYLINE_JET_CHANGES is empty.")
            return None

        storyline_impact_results = {gwl: {} for gwl in self.config.GLOBAL_WARMING_LEVELS}
        storyline_jet_definitions = self.config.STORYLINE_JET_CHANGES 
        cmip6_mmm_changes = cmip6_gwl_analysis_results['cmip6_mmm_changes_at_gwl']

        impact_variable_to_driver_map = {
            'DJF_pr':  {'driving_jet_index': 'DJF_JetSpeed', 'beta_obs_key': 'DJF_JetSpeed_vs_pr'},
            'DJF_tas': {'driving_jet_index': 'DJF_JetSpeed', 'beta_obs_key': 'DJF_JetSpeed_vs_tas'},
            'JJA_pr':  {'driving_jet_index': 'JJA_JetLat',   'beta_obs_key': 'JJA_JetLat_vs_pr'},
            'JJA_tas': {'driving_jet_index': 'JJA_JetLat',   'beta_obs_key': 'JJA_JetLat_vs_tas'},
        }

        logging.info("Using Beta_obs slopes: %s", json.dumps(observed_beta_slopes, indent=2, default=lambda x: round(x,3) if isinstance(x, (float, np.floating)) else str(x)))
        logging.info("Using Storyline Jet Change definitions: %s", json.dumps(storyline_jet_definitions, indent=2))

        for gwl_value in self.config.GLOBAL_WARMING_LEVELS:
            if cmip6_mmm_changes.get(gwl_value) is None: 
                logging.info(f"  Skipping storyline impacts for GWL {gwl_value}°C: No CMIP6 MMM changes.")
                continue
            logging.info(f"\n  Processing Storyline Impacts for GWL {gwl_value}°C...")

            for impact_var_name, driver_info in impact_variable_to_driver_map.items():
                driving_jet_index_name = driver_info['driving_jet_index'] 
                beta_obs_key_for_driver = driver_info['beta_obs_key']   

                delta_impact_var_mmm = cmip6_mmm_changes[gwl_value].get(impact_var_name)
                delta_driving_jet_mmm = cmip6_mmm_changes[gwl_value].get(driving_jet_index_name)
                beta_slope_obs = observed_beta_slopes.get(beta_obs_key_for_driver)

                inputs_valid = True
                if delta_impact_var_mmm is None or np.isnan(delta_impact_var_mmm): inputs_valid = False; logging.warning(f"    - Skip {impact_var_name}: Missing MMM impact for GWL {gwl_value}°C.")
                if delta_driving_jet_mmm is None or np.isnan(delta_driving_jet_mmm): inputs_valid = False; logging.warning(f"    - Skip {impact_var_name}: Missing MMM driving jet '{driving_jet_index_name}' for GWL {gwl_value}°C.")
                if beta_slope_obs is None or np.isnan(beta_slope_obs): inputs_valid = False; logging.warning(f"    - Skip {impact_var_name}: Missing beta_obs slope for key '{beta_obs_key_for_driver}'.")
                
                if driving_jet_index_name not in storyline_jet_definitions or \
                   gwl_value not in storyline_jet_definitions[driving_jet_index_name]:
                    inputs_valid = False; logging.warning(f"    - Skip {impact_var_name}: Storyline defs missing for '{driving_jet_index_name}' at GWL {gwl_value}°C.")
                
                if not inputs_valid: continue

                logging.info(f"    Calculating impacts for '{impact_var_name}' (driven by '{driving_jet_index_name}')")
                storyline_impact_results[gwl_value][impact_var_name] = {} 

                for storyline_type_name, delta_jet_storyline_value in storyline_jet_definitions[driving_jet_index_name][gwl_value].items():
                    adjustment_term = beta_slope_obs * (delta_jet_storyline_value - delta_driving_jet_mmm)
                    final_storyline_impact = delta_impact_var_mmm + adjustment_term
                    storyline_impact_results[gwl_value][impact_var_name][storyline_type_name] = final_storyline_impact
                    logging.info(f"      {storyline_type_name:<12}: Imp_MMM={delta_impact_var_mmm:+.2f}, beta={beta_slope_obs:+.2f} * (Jet_story={delta_jet_storyline_value:+.2f} - Jet_MMM={delta_driving_jet_mmm:+.2f}) = Adj={adjustment_term:+.2f} -> Total_Impact={final_storyline_impact:+.2f}")
            
            if not storyline_impact_results[gwl_value]: del storyline_impact_results[gwl_value]

        logging.info("\n--- Storyline Impact Calculation Completed ---")
        return {gwl: impacts for gwl, impacts in storyline_impact_results.items() if impacts}