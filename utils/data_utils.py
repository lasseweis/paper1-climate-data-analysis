#!/usr/bin/env python3
"""
Utility functions and class for general climate data processing.
"""
import pandas as pd
import numpy as np
import xarray as xr
import logging
import os
import traceback
from functools import lru_cache

# Relative import for Config, assuming both files are in the 'utils' package
from config_setup import Config

def select_level_preprocess(ds, level_hpa=850):
    """
    Preprocessing function for xarray.open_mfdataset to select a specific pressure level.
    Handles different units (hPa or Pa) and potential errors gracefully.

    Args:
        ds (xr.Dataset): The dataset chunk to preprocess.
        level_hpa (int, optional): The target pressure level in hPa. Defaults to 850.

    Returns:
        xr.Dataset: The dataset chunk, possibly with a level selected, or the original if selection fails.
    """
    var_name = None
    potential_names = ['ua', 'u', 'uwnd'] # Common names for zonal wind
    for name in potential_names:
        if name in ds.data_vars:
            var_name = name
            break

    if var_name is None:
        logging.warning("      Warning (preprocess): Could not find a wind variable (ua/u/uwnd) in the dataset.")
        return ds

    level_coord_name = None
    if 'lev' in ds.coords:
        level_coord_name = 'lev'
    elif 'plev' in ds.coords:
        ds = ds.rename({'plev': 'lev'}) # Standardize to 'lev'
        level_coord_name = 'lev'
    elif 'lev' in ds.dims and 'lev' not in ds.coords:
         logging.warning(f"      Warning (preprocess): Dimension 'lev' exists but is not a coordinate variable in {ds.attrs.get('tracking_id', 'dataset')}. Cannot select level.")
         return ds
    elif 'level' in ds.coords: # Another common name
         ds = ds.rename({'level': 'lev'}) # Standardize to 'lev'
         level_coord_name = 'lev'

    if level_coord_name:
        try:
            target_plev_pa = level_hpa * 100  # Target in Pascal
            level_coord = ds[level_coord_name]
            lev_units = level_coord.attrs.get('units', '').lower()

            target_plev_in_file_units = level_hpa # Default to hPa
            if 'pa' in lev_units and 'hpa' not in lev_units: # Clearly Pascal
                target_plev_in_file_units = target_plev_pa
            elif not lev_units: # No units, try to infer
                # Heuristic: if max level > 50000, likely Pa, else hPa
                if level_coord.max() > 50000:
                    target_plev_in_file_units = target_plev_pa
                logging.warning(f"      Warning (preprocess): Units for level coordinate '{level_coord_name}' are missing or unclear. Assuming {'Pa' if target_plev_in_file_units == target_plev_pa else 'hPa'}. Max value: {level_coord.max().item()}")
            # If 'hpa' in lev_units or units are ambiguous but small values, target_plev_in_file_units remains level_hpa

            ds_sel = ds.sel({level_coord_name: target_plev_in_file_units}, method='nearest')
            # actual_selected_level = ds_sel[level_coord_name].item()
            # logging.info(f"      Preprocessing: Selected level {actual_selected_level} {lev_units if lev_units else 'assumed units'} for target {level_hpa} hPa.")
            return ds_sel
        except Exception as e:
            filename = os.path.basename(ds.encoding.get("source", "Unknown file")) # Get filename if available
            logging.error(f"      ERROR (preprocess): Level selection failed for file '{filename}'. Target: {level_hpa}hPa. Error: {e}")
            return ds # Return original dataset on error to allow pipeline to continue if possible
    else:
        # No level coordinate found (e.g., for surface variables like tas, pr)
        return ds

class DataProcessor:
    """Class for processing climate data."""

    @staticmethod
    def _assign_season_to_df(df_time):
        """Helper to assign season and season_year to a DataFrame with 'month' and 'year' columns."""
        month_map = {
            1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Autumn', 10: 'Autumn',
            11: 'Autumn', 12: 'Winter'
        }
        df_time['season'] = df_time['month'].map(month_map)
        df_time['season_year'] = df_time['year']
        # For Winter (December), the season_year is the next year
        dec_mask = df_time['month'] == 12
        df_time.loc[dec_mask, 'season_year'] = df_time.loc[dec_mask, 'year'] + 1
        return df_time.dropna(subset=['season', 'season_year'])

    @staticmethod
    @lru_cache(maxsize=16) # Cache results of file processing
    def process_ncfile(file_path, var_name_requested, var_out_name=None, level_to_select=None):
        """
        Process a single NetCDF file (typically for 20CRv3).
        Results are cached.
        """
        logging.info(f"Processing NetCDF data from {file_path} for variable '{var_name_requested}'...")
        try:
            ds = xr.open_dataset(file_path, decode_times=True, chunks={'time': 'auto'}, use_cftime=True)

            # Map common requested variable names to typical names in 20CRv3 files
            var_mapping_20crv3 = {'pr': 'prate', 'tas': 'air', 'ua': 'uwnd'}
            actual_var_in_file = var_mapping_20crv3.get(var_name_requested, var_name_requested)

            if actual_var_in_file not in ds.data_vars:
                logging.error(f"Variable '{actual_var_in_file}' (mapped from '{var_name_requested}') not found in {file_path}")
                ds.close()
                return None

            # Standardize coordinate names (longitude to lon, latitude to lat)
            rename_coords = {}
            if 'longitude' in ds.dims: rename_coords['longitude'] = 'lon'
            if 'latitude' in ds.dims: rename_coords['latitude'] = 'lat'
            if 'longitude' in ds.coords and 'longitude' not in ds.dims: rename_coords['longitude'] = 'lon'
            if 'latitude' in ds.coords and 'latitude' not in ds.dims: rename_coords['latitude'] = 'lat'
            if rename_coords:
                ds = ds.rename(rename_coords)

            # Normalize longitudes from 0-360 to -180-180 if necessary
            if 'lon' in ds.coords and np.any(ds.lon > 180):
                logging.info(f"  Normalizing longitudes for {file_path}")
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                ds = ds.sortby('lon') # Sort by new longitude values

            # Select pressure level if requested (e.g., for 'ua')
            if level_to_select is not None:
                level_dim_name = next((name for name in ['level', 'plev', 'lev'] if name in ds.dims), None)
                if level_dim_name:
                    try:
                        level_units = ds[level_dim_name].attrs.get('units', '').lower()
                        target_level_val = level_to_select
                        if 'pa' in level_units and 'hpa' not in level_units: # Units are Pa
                            target_level_val = level_to_select * 100
                        
                        ds_level_sel = ds.sel({level_dim_name: target_level_val}, method='nearest')
                        # actual_sel_lev_val = ds_level_sel[level_dim_name].item()
                        logging.info(f"  Selected level {level_to_select} hPa (target in file: {target_level_val} {level_units}) from dim '{level_dim_name}'.")
                        ds = ds_level_sel
                    except Exception as e_level:
                        logging.error(f"  Error selecting level {level_to_select} hPa from dim '{level_dim_name}': {e_level}")
                        # Decide if to proceed without level selection or fail
                else:
                    logging.warning(f"  Level selection requested ({level_to_select} hPa) but no level dimension found.")


            # Rename variable to output name if specified
            final_var_name = actual_var_in_file
            if var_out_name and var_out_name != actual_var_in_file:
                if actual_var_in_file in ds.data_vars:
                    ds = ds.rename({actual_var_in_file: var_out_name})
                    final_var_name = var_out_name
                else:
                    logging.warning(f"  Cannot rename '{actual_var_in_file}' to '{var_out_name}', variable not found after potential level selection.")

            data_array = ds[final_var_name] # Work with DataArray from here

            # Ensure time coordinate is correctly parsed and add year/month
            if 'time' not in data_array.coords:
                logging.error(f"Time coordinate not found in variable {final_var_name} from {file_path}")
                ds.close(); return None
            
            # Convert cftime objects or other time representations to pandas datetime if possible and safe
            # For robust cftime handling, xarray's dt accessor is preferred.
            try:
                time_accessor = data_array.time.dt
                data_array = data_array.assign_coords(
                    year=("time", time_accessor.year.values),
                    month=("time", time_accessor.month.values)
                )
            except Exception as e_time_coords:
                logging.error(f"Failed to assign year/month coordinates using .dt accessor: {e_time_coords}")
                ds.close(); return None


            # Filter by analysis period
            data_array_filtered = data_array.sel(
                time=((data_array.time.dt.year >= Config.ANALYSIS_START_YEAR) &
                      (data_array.time.dt.year <= Config.ANALYSIS_END_YEAR))
            )
            if data_array_filtered.time.size == 0:
                logging.warning(f"No data found within analysis period {Config.ANALYSIS_START_YEAR}-{Config.ANALYSIS_END_YEAR} for {file_path}.")
                ds.close(); return None
            
            ds.close() # Close the main dataset file handle

            # Convert precipitation units if necessary (kg/m^2/s to mm/day)
            if final_var_name == 'prate' or var_name_requested == 'pr': # Check based on common names
                units = data_array_filtered.attrs.get('units', '').lower()
                if 'kg' in units and ('m-2' in units or 'm^2' in units) and ('s-1' in units or '/s' in units): # common for kg/m2/s
                    logging.info(f"  Converting precipitation units from '{units}' to 'mm/day'.")
                    data_array_filtered = data_array_filtered * 86400.0  # s/day
                    data_array_filtered.attrs['units'] = 'mm/day'
                elif 'mm' in units and ('day' in units or 'd' in units):
                    logging.info(f"  Precipitation units already appear to be '{units}'. No conversion applied by script.")
                else:
                    logging.warning(f"  Precipitation units are '{units}'. Not recognized as 'kg/m^2/s' or 'mm/day'. No conversion applied by script.")

            data_array_filtered.attrs['dataset_source'] = Config.DATASET_20CRV3 # Add source identifier
            return data_array_filtered.load() # Load into memory

        except FileNotFoundError:
            logging.error(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            logging.error(f"General error processing {file_path}: {e}")
            logging.error(traceback.format_exc())
            if 'ds' in locals() and ds is not None: ds.close()
            return None

    @staticmethod
    @lru_cache(maxsize=16)
    def process_era5_file(file_path, var_in_file_req, var_out_name=None, level_to_select=None):
        """
        Process a single ERA5 NetCDF file (typically daily data to monthly means).
        Results are cached.
        """
        logging.info(f"Processing ERA5 data from {file_path} for variable '{var_in_file_req}'...")
        try:
            ds = xr.open_dataset(file_path, decode_times=True, use_cftime=True)

            # Handle different variable naming conventions in ERA5 files
            potential_var_names_map = {
                'pr': ['tp', 'precipitation', 'precip', 'PR'], # Total precipitation
                'tas': ['t2m', 'temperature', 'temp', 'TAS'],    # 2m temperature
                'u': ['u', 'ua', 'u_component', 'U']          # Zonal wind
            }
            actual_var_in_file = var_in_file_req # Default
            for req_name, potential_names in potential_var_names_map.items():
                if var_in_file_req.lower() == req_name:
                    actual_var_in_file = next((v for v in potential_names if v in ds.data_vars), None)
                    if actual_var_in_file: break
            if not actual_var_in_file or actual_var_in_file not in ds.data_vars:
                 actual_var_in_file = var_in_file_req # Fallback to requested if map fails
                 if actual_var_in_file not in ds.data_vars:
                    logging.error(f"Variable for '{var_in_file_req}' not found in {file_path}. Available: {list(ds.data_vars.keys())}")
                    ds.close(); return None

            # Standardize coordinate names
            rename_coords = {}
            if 'latitude' in ds.dims: rename_coords['latitude'] = 'lat'
            if 'longitude' in ds.dims: rename_coords['longitude'] = 'lon'
            if 'latitude' in ds.coords and 'latitude' not in ds.dims: rename_coords['latitude'] = 'lat'
            if 'longitude' in ds.coords and 'longitude' not in ds.dims: rename_coords['longitude'] = 'lon'
            if rename_coords:
                ds = ds.rename(rename_coords)

            # Normalize longitudes if necessary
            if 'lon' in ds.coords and np.any(ds.lon > 180):
                logging.info(f"  Normalizing longitudes for {file_path}")
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
                ds = ds.sortby('lon')

            # Select pressure level if requested
            if level_to_select is not None and ('level' in ds.dims or 'lev' in ds.dims):
                level_dim_name = 'level' if 'level' in ds.dims else 'lev'
                try:
                    ds = ds.sel({level_dim_name: level_to_select}, method='nearest') # ERA5 levels usually in hPa
                    logging.info(f"  Selected level {level_to_select} hPa from dim '{level_dim_name}'.")
                except Exception as e_level:
                    logging.error(f"  Error selecting level {level_to_select} hPa for ERA5: {e_level}")

            # Rename variable to output name
            final_var_name = actual_var_in_file
            if var_out_name and var_out_name != actual_var_in_file:
                ds = ds.rename({actual_var_in_file: var_out_name})
                final_var_name = var_out_name
            
            data_array = ds[final_var_name]

            # Calculate monthly means from daily data
            # Ensure time coordinate is valid for resampling
            if 'time' not in data_array.coords or data_array.time.size == 0:
                logging.error("Time coordinate missing or empty, cannot resample to monthly.")
                ds.close(); return None

            monthly_data = data_array.resample(time='1MS').mean(skipna=True) # '1MS' for start of month
            
            # Add year and month coordinates to the monthly data
            monthly_data = monthly_data.assign_coords(
                year=("time", monthly_data.time.dt.year.values),
                month=("time", monthly_data.time.dt.month.values)
            )
            
            # Filter by analysis period
            monthly_filtered = monthly_data.sel(
                time=((monthly_data.time.dt.year >= Config.ANALYSIS_START_YEAR) &
                      (monthly_data.time.dt.year <= Config.ANALYSIS_END_YEAR))
            )
            if monthly_filtered.time.size == 0:
                logging.warning(f"No ERA5 data found within analysis period {Config.ANALYSIS_START_YEAR}-{Config.ANALYSIS_END_YEAR} for {file_path} after monthly resampling.")
                ds.close(); return None
                
            ds.close()

            # Precipitation unit conversion for ERA5 (often in m, convert to mm/day)
            # ERA5 'tp' (total precipitation) is often accumulated (e.g., m of water equivalent).
            # If daily file, and values are accumulation over day, unit might be 'm'.
            # If it's 'm' per day, then multiply by 1000 for 'mm/day'.
            # If it's mean rate in 'm s**-1', then multiply by 1000 * 86400.
            # The original script's ERA5 PR file "ERA5_2p5cdo_day_PR_19500101-20221231.nc" suggests daily.
            # Let's assume the input data for 'pr' (tp) is daily total accumulation in 'm'.
            if final_var_name == 'tp' or var_in_file_req == 'pr':
                units = monthly_filtered.attrs.get('units', '').lower()
                # If units are 'm' and it's daily data resampled to monthly mean of daily totals:
                if units == 'm':
                    # If the values represent daily totals in 'm', and we took a monthly mean of these daily totals,
                    # then the mean is also in 'm/day equivalent'. Convert to 'mm/day'.
                    logging.info(f"  Assuming ERA5 precipitation '{final_var_name}' units '{units}' represent mean daily total in meters. Converting to 'mm/day'.")
                    monthly_filtered = monthly_filtered * 1000.0
                    monthly_filtered.attrs['units'] = 'mm/day'
                elif 'kg m-2 s-1' == units or 'kg/m^2/s' == units: # If it's a rate
                    logging.info(f"  Converting ERA5 precipitation units from '{units}' to 'mm/day'.")
                    monthly_filtered = monthly_filtered * 86400.0
                    monthly_filtered.attrs['units'] = 'mm/day'
                elif 'mm' in units and ('day' in units or 'd' in units):
                     logging.info(f"  ERA5 Precipitation units already appear to be '{units}'. No conversion by script.")
                else:
                    logging.warning(f"  Units for ERA5 precipitation '{final_var_name}' are '{units}'. Conversion logic might be needed if not mm/day.")


            monthly_filtered.attrs['dataset_source'] = Config.DATASET_ERA5
            return monthly_filtered.load()

        except Exception as e:
            logging.error(f"General error processing ERA5 file {file_path}: {e}")
            logging.error(traceback.format_exc())
            if 'ds' in locals() and ds is not None: ds.close()
            return None

    @staticmethod
    def assign_season_to_dataarray(data_array):
        """
        Assigns 'season' and 'season_year' coordinates to a DataArray.
        Input DataArray must have a 'time' coordinate with 'month' and 'year' info.
        """
        if data_array is None:
            return None
        if 'time' not in data_array.coords:
            logging.error("assign_season_to_dataarray: 'time' coordinate missing.")
            return None
        
        try:
            # Ensure year and month are available as coordinates on the time dimension
            if not all(c in data_array.time.coords for c in ['year', 'month']):
                 data_array = data_array.assign_coords(
                     year=("time", data_array.time.dt.year.values),
                     month=("time", data_array.time.dt.month.values)
                 )

            # Create a DataFrame for easier season assignment
            df_time = pd.DataFrame({
                'time_idx': data_array.time.to_index(), # Keep original time for selection
                'year': data_array.year.values,
                'month': data_array.month.values
            })
            
            df_time_seasonal = DataProcessor._assign_season_to_df(df_time.copy()) # Use helper
            
            if df_time_seasonal.empty:
                logging.warning("No valid seasonal data after assignment.")
                return None # Or handle as appropriate

            # Select the original data_array times that have valid season/season_year
            # and assign the new coordinates.
            # This ensures we only keep time steps that fall into a complete season.
            data_array_sel = data_array.sel(time=df_time_seasonal['time_idx'].values)
            
            data_array_with_seasons = data_array_sel.assign_coords(
                season=("time", df_time_seasonal['season'].values),
                season_year=("time", df_time_seasonal['season_year'].astype(int).values) # Ensure int
            )
            
            # Create a 'season_key' for easier groupby operations if needed later
            # Example: "2000-Winter"
            season_keys = [f"{sy}-{s}" for sy, s in zip(data_array_with_seasons['season_year'].values, data_array_with_seasons['season'].values)]
            return data_array_with_seasons.assign_coords(season_key=("time", season_keys))

        except Exception as e:
            logging.error(f"Error in assign_season_to_dataarray: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def calculate_seasonal_means(data_array_with_seasons):
        """
        Calculate seasonal means from a DataArray that has 'season' and 'season_year' coordinates.
        The output will have 'season_year' and 'season' as dimensions.
        """
        if data_array_with_seasons is None:
            return None
        
        required_coords = ['season', 'season_year']
        if not all(coord in data_array_with_seasons.coords for coord in required_coords):
            logging.error(f"calculate_seasonal_means: Missing one or more required coordinates: {required_coords}.")
            return None

        try:
            # Group by season_year and season, then take the mean over the 'time' dimension.
            # This works if 'season' and 'season_year' are coordinates along the 'time' dimension.
            seasonal_mean = data_array_with_seasons.groupby(['season_year', 'season']).mean(dim='time', skipna=True)
            
            # The result of groupby might need unstacking if you want 'season' as a separate dimension
            # from a MultiIndex. If 'season' is already a separate dimension after mean, this might not be needed.
            # If 'season_year' and 'season' form a MultiIndex on a new dimension (e.g., 'stacked_season_season_year'),
            # then unstack it.
            # However, xarray often handles this well if 'season' and 'season_year' were coordinates.
            # If the output is (lat, lon, season_year, season), it's already in the desired format.
            # If it's (lat, lon, group_dim) where group_dim is a multi-index of (season_year, season), then unstack.

            # For simplicity, let's assume groupby results in (..., season_year, season) or can be unstacked to it.
            # If `groupby(['season_year', 'season'])` creates a multi-index, it needs unstacking.
            # A more robust way if `season_key` was created by `assign_season_to_dataarray`:
            if 'season_key' in data_array_with_seasons.coords:
                # This path was more developed in your original code
                seasonal_mean_by_key = data_array_with_seasons.groupby("season_key").mean(dim="time", skipna=True)
                if seasonal_mean_by_key.season_key.size == 0:
                    logging.warning("No valid season_keys after groupby in calculate_seasonal_means.")
                    return None

                season_key_values = seasonal_mean_by_key['season_key'].values
                years_list = [int(key.split('-')[0]) for key in season_key_values if isinstance(key, str) and '-' in key]
                seasons_list = [key.split('-')[1] for key in season_key_values if isinstance(key, str) and '-' in key]
                
                if not years_list or len(years_list) != len(season_key_values): # Check parsing success
                    logging.error("Failed to parse all season_keys into year and season.")
                    return None

                seasonal_mean_by_key = seasonal_mean_by_key.assign_coords(
                    temp_season_year=("season_key", years_list),
                    temp_season_str=("season_key", seasons_list)
                )
                # Unstack to get 'season_year' and 'season' as dimensions
                seasonal_mean = seasonal_mean_by_key.set_index(season_key=['temp_season_year', 'temp_season_str']) \
                                                    .unstack('season_key') \
                                                    .rename({'temp_season_year': 'season_year', 'temp_season_str': 'season'})
            
            if 'dataset_source' in data_array_with_seasons.attrs: # Preserve source
                seasonal_mean.attrs['dataset_source'] = data_array_with_seasons.attrs['dataset_source']
            
            logging.debug(f"calculate_seasonal_means output dims: {seasonal_mean.dims}")
            return seasonal_mean

        except Exception as e:
            logging.error(f"Error in calculate_seasonal_means: {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def calculate_anomalies(data_array, base_period_start=None, base_period_end=None, as_percentage=False):
        """
        Calculate anomalies relative to a climatological period (monthly).
        """
        if data_array is None:
            return None
        
        # Use Config defaults if not provided
        base_start = base_period_start if base_period_start is not None else Config.BASE_PERIOD_START_YEAR
        base_end = base_period_end if base_period_end is not None else Config.BASE_PERIOD_END_YEAR

        if 'time' not in data_array.coords or 'month' not in data_array.coords:
            logging.error("calculate_anomalies: 'time' or 'month' coordinate missing from input DataArray.")
            # Attempt to add month if only time exists
            if 'time' in data_array.coords and 'month' not in data_array.coords:
                try:
                    data_array = data_array.assign_coords(month=("time", data_array.time.dt.month.values))
                except Exception as e:
                    logging.error(f"Could not assign month coordinate: {e}")
                    return None
            else:
                return None # Cannot proceed

        try:
            # Select the base period for climatology calculation
            climatology_period_data = data_array.sel(
                time=((data_array.time.dt.year >= base_start) &
                      (data_array.time.dt.year <= base_end))
            )
            if climatology_period_data.time.size == 0:
                logging.warning(f"No data in base period [{base_start}-{base_end}] for anomaly calculation.")
                return xr.full_like(data_array, np.nan) # Return NaNs if no base period data

            # Calculate monthly climatology
            monthly_climatology = climatology_period_data.groupby('month').mean(dim='time', skipna=True)
            
            if as_percentage:
                # (X - X_clim) / X_clim * 100
                # Avoid division by zero or by very small climatology values if they are effectively zero
                # Add a small epsilon to climatology denominator if it's exactly zero and makes sense for the variable
                # For precipitation, zero climatology means any rain is infinite percent anomaly.
                # A more robust way is to check if climatology is effectively zero.
                anomalies = data_array.groupby('month').apply(
                    lambda x, clim: ((x - clim) / clim.where(abs(clim) > 1e-9, np.nan)) * 100 if clim.notnull().any() else xr.full_like(x, np.nan),
                    clim=monthly_climatology
                )
            else:
                # Absolute anomalies: X - X_clim
                anomalies = data_array.groupby('month') - monthly_climatology
            
            if 'dataset_source' in data_array.attrs:
                anomalies.attrs['dataset_source'] = data_array.attrs['dataset_source']
            anomalies.attrs['anomaly_base_period'] = f"{base_start}-{base_end}"
                
            return anomalies
        except Exception as e:
            logging.error(f"Error in calculate_anomalies: {e}")
            traceback.print_exc()
            return xr.full_like(data_array, np.nan)

    @staticmethod
    def calculate_spatial_mean(data_array, lat_min, lat_max, lon_min, lon_max):
        """
        Calculate area-weighted spatial mean within a defined box.
        Preserves non-spatial coordinates.
        """
        if data_array is None:
            return None

        original_attrs = data_array.attrs.copy()
        original_name = data_array.name

        try:
            # Identify spatial dimensions (lat, lon)
            spatial_dims = [d for d in ['lat', 'lon'] if d in data_array.dims]
            if not spatial_dims:
                 logging.warning(f"Warning: No spatial dimensions ('lat', 'lon') found for spatial mean in DataArray '{original_name}'. Returning input as is.")
                 return data_array

            # Select the domain
            # Handle latitude order (ascending/descending)
            lat_slice = slice(lat_min, lat_max)
            if 'lat' in data_array.coords and data_array.lat.size > 1:
                if data_array.lat.values[0] > data_array.lat.values[-1]: # Descending latitudes
                    lat_slice = slice(lat_max, lat_min)
            
            domain_selected = data_array.sel(lat=lat_slice, lon=slice(lon_min, lon_max))

            if domain_selected.sizes.get('lat', 0) == 0 or domain_selected.sizes.get('lon', 0) == 0:
                 logging.warning(f"Warning: Spatial domain selection for '{original_name}' resulted in zero size for lat/lon. Box: {lat_min}-{lat_max}, {lon_min}-{lon_max}. Returning NaN.")
                 # Create a NaN result that preserves non-spatial dimensions
                 non_spatial_dims = [d for d in data_array.dims if d not in spatial_dims]
                 if not non_spatial_dims: return xr.DataArray(np.nan, name=original_name, attrs=original_attrs)
                 
                 output_coords = {name: data_array[name] for name in non_spatial_dims if name in data_array.coords}
                 output_shape = tuple(data_array.sizes[d] for d in non_spatial_dims)
                 return xr.DataArray(np.full(output_shape, np.nan), coords=output_coords, dims=non_spatial_dims, name=original_name, attrs=original_attrs)

            # Calculate area weights (cosine of latitude)
            if 'lat' in domain_selected.coords:
                weights = np.cos(np.deg2rad(domain_selected.lat))
                weights = weights.where(weights > 0) # Ensure weights are positive
                weights.name = "weights" # Important for xarray's weighted operations

                # Perform weighted mean over spatial dimensions
                weighted_mean_result = domain_selected.weighted(weights).mean(dim=spatial_dims, skipna=True)
            else:
                 logging.warning(f"Warning: No 'lat' coordinate found for weighting in DataArray '{original_name}'. Using unweighted mean.")
                 weighted_mean_result = domain_selected.mean(dim=spatial_dims, skipna=True)

            weighted_mean_result.attrs = original_attrs
            weighted_mean_result.name = original_name
            return weighted_mean_result

        except Exception as e:
            logging.error(f"Error in calculate_spatial_mean for DataArray '{original_name}': {e}")
            traceback.print_exc()
            # Attempt to return NaN with correct non-spatial dimensions
            non_spatial_dims = [d for d in data_array.dims if d not in ['lat', 'lon']]
            if not non_spatial_dims: return xr.DataArray(np.nan, name=original_name, attrs=original_attrs)
            output_coords = {name: data_array[name] for name in non_spatial_dims if name in data_array.coords}
            output_shape = tuple(data_array.sizes[d] for d in non_spatial_dims)
            return xr.DataArray(np.full(output_shape, np.nan), coords=output_coords, dims=non_spatial_dims, name=original_name, attrs=original_attrs)

    @staticmethod
    def filter_by_season(data_array_seasonal, season_name):
        """
        Filter a DataArray (that has 'season' and 'season_year' dimensions/coordinates)
        for a specific season.
        """
        if data_array_seasonal is None:
            logging.warning("filter_by_season: Input DataArray is None.")
            return None
        if 'season' not in data_array_seasonal.coords and 'season' not in data_array_seasonal.dims:
            logging.warning(f"filter_by_season: 'season' coordinate or dimension not found in DataArray.")
            return None

        try:
            # If 'season' is a dimension, select directly
            if 'season' in data_array_seasonal.dims:
                filtered_da = data_array_seasonal.sel(season=season_name)
            # If 'season' is a coordinate (e.g., along 'time' or 'season_year' before grouping)
            elif 'season' in data_array_seasonal.coords:
                # This case is more complex if 'season' is not the primary dimension.
                # Assuming 'season_year' is the primary time-like dimension after seasonal means.
                if 'season_year' in data_array_seasonal.dims:
                     # This scenario implies data_array_seasonal might be shaped (..., season_year)
                     # and 'season' is a coordinate aligned with 'season_year'.
                     # This is less common for the output of `calculate_seasonal_means` which should have 'season' as a dim.
                     # This part might need adjustment based on actual data structure if 'season' is not a dim.
                     # For now, we assume `calculate_seasonal_means` produces 'season' as a dimension.
                     logging.warning("filter_by_season: 'season' is a coordinate but not a dimension. This case might not be handled correctly for all inputs. Expected 'season' as a dimension.")
                     # Attempt a where clause, but this changes structure
                     filtered_da = data_array_seasonal.where(data_array_seasonal.season == season_name, drop=True)

                else: # Fallback if 'season' is just a coordinate without clear dimensional context
                    logging.error("filter_by_season: 'season' is a coordinate, but 'season_year' is not a dimension. Unclear how to filter.")
                    return None
            else: # Should not happen due to initial checks
                return None


            # Sort by 'season_year' if it's a dimension
            if 'season_year' in filtered_da.dims:
                filtered_da = filtered_da.sortby('season_year')
            
            # Preserve attributes
            if 'dataset_source' in data_array_seasonal.attrs:
                filtered_da.attrs['dataset_source'] = data_array_seasonal.attrs['dataset_source']

            if filtered_da.size == 0:
                logging.warning(f"filter_by_season: No data remaining after filtering for season '{season_name}'.")
            return filtered_da
            
        except Exception as e:
            logging.error(f"Error in filter_by_season for season '{season_name}': {e}")
            traceback.print_exc()
            return None

    @staticmethod
    def detrend_data(data_array):
        """
        Detrend a time series or a spatial DataArray along the 'season_year' dimension.
        Uses simple linear detrending (subtracts linear fit).
        """
        if data_array is None:
            return None

        try:
            # Identify the time dimension (should be 'season_year' after seasonal processing)
            time_dim_name = 'season_year'
            if time_dim_name not in data_array.dims:
                potential_time_dims = [d for d in data_array.dims if 'year' in d.lower() or 'time' in d.lower()]
                if potential_time_dims:
                    time_dim_name = potential_time_dims[0]
                    logging.debug(f"detrend_data: Using dimension '{time_dim_name}' as the time axis.")
                else:
                    logging.error(f"detrend_data: No suitable time dimension (e.g., 'season_year') found. Dims: {data_array.dims}")
                    return data_array # Return original if no time dimension

            if data_array[time_dim_name].size < 2:
                logging.debug("detrend_data: Not enough time points (<2) to detrend.")
                return data_array # Cannot detrend with fewer than 2 points

            # Use xarray's polyfit and polyval for detrending if possible (requires dask an numpy >= 1.22 for polyfit on xarray)
            # For a simpler, more universally compatible approach, use scipy.signal.detrend or manual lstsq.
            # Here, a manual approach using np.linalg.lstsq for broader compatibility:
            
            years = data_array[time_dim_name].values
            if np.isnan(years).any():
                 logging.warning(f"detrend_data: NaNs found in time coordinate '{time_dim_name}'. Detrending may fail or be inaccurate.")
                 # Decide whether to return original or proceed with caution
                 # return data_array

            # Prepare the design matrix for linear regression (time, constant)
            X = np.vstack([years, np.ones(len(years))]).T
            
            # Apply detrending along the time dimension
            # This can be done by iterating over other dimensions if present.
            # xarray.apply_ufunc can make this cleaner for multi-dimensional arrays.

            original_values = data_array.data # Get numpy array
            # Move time axis to the first position for easier processing
            time_axis_num = data_array.get_axis_num(time_dim_name)
            values_time_first = np.moveaxis(original_values, time_axis_num, 0)
            
            # Reshape to 2D: (time, other_dims_combined)
            original_shape_time_first = values_time_first.shape
            reshaped_values = values_time_first.reshape(original_shape_time_first[0], -1)
            
            detrended_reshaped_values = np.full_like(reshaped_values, np.nan)

            for i in range(reshaped_values.shape[1]): # Iterate over each spatial point/other dim combination
                y_series = reshaped_values[:, i]
                valid_mask = np.isfinite(y_series) & np.isfinite(X[:,0]) # Also check X for safety
                
                if np.sum(valid_mask) >= 2: # Need at least 2 points for lstsq
                    X_valid = X[valid_mask]
                    y_valid = y_series[valid_mask]
                    
                    # Avoid issues if X_valid or y_valid has no variance
                    if np.var(X_valid[:, 0]) < 1e-10 or np.var(y_valid) < 1e-10:
                        detrended_reshaped_values[:, i] = y_series # Trend is effectively zero or undefined
                        continue
                    
                    try:
                        slope, intercept = np.linalg.lstsq(X_valid, y_valid, rcond=None)[0]
                        trend = slope * years + intercept
                        # Subtract trend from original series (NaNs will propagate correctly if y_series has them)
                        detrended_reshaped_values[:, i] = y_series - trend 
                    except np.linalg.LinAlgError:
                        logging.warning(f"detrend_data: Linear algebra error for a series. Keeping original values for this series.")
                        detrended_reshaped_values[:, i] = y_series # Fallback to original if lstsq fails
                else: # Not enough valid points
                    detrended_reshaped_values[:, i] = y_series # Keep original

            # Reshape back to (time, other_dims...) and then to original dimension order
            detrended_values_time_first = detrended_reshaped_values.reshape(original_shape_time_first)
            detrended_original_order = np.moveaxis(detrended_values_time_first, 0, time_axis_num)

            # Create new DataArray with detrended values
            detrended_da = xr.DataArray(
                data=detrended_original_order,
                coords=data_array.coords,
                dims=data_array.dims,
                name=data_array.name + "_detrended" if data_array.name else "detrended_data",
                attrs=data_array.attrs.copy()
            )
            detrended_da.attrs['detrended'] = 'linear'
            
            if 'dataset_source' in data_array.attrs: # Preserve source
                detrended_da.attrs['dataset_source'] = data_array.attrs['dataset_source']
            return detrended_da

        except Exception as e:
            logging.error(f"General error in detrend_data: {e}")
            traceback.print_exc()
            return data_array # Return original on error