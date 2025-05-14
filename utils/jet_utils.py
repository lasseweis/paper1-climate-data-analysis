#!/usr/bin/env python3
"""
Utilities for jet stream analysis.
"""
import numpy as np
import xarray as xr
import logging
import traceback

# Relative import for Config, assuming both files are in the 'utils' package
from config_setup import Config

class JetStreamAnalyzer:
    """Class for jet stream-specific analyses."""

    @staticmethod
    def calculate_jet_speed_index(seasonal_u_wind_da):
        """
        Calculate jet speed index from a seasonal U850 DataArray.
        The index is the area-weighted mean zonal wind speed within a defined box.
        Ensures 1D output (time series).

        Args:
            seasonal_u_wind_da (xr.DataArray): Seasonal mean of zonal wind (e.g., U850).
                                               Expected dimensions typically (lat, lon, season_year)
                                               or (season_year, lat, lon).

        Returns:
            xr.DataArray: 1D DataArray of the jet speed index, or None if calculation fails.
        """
        if seasonal_u_wind_da is None:
            logging.debug("JetSpeed: Input seasonal_u_wind_da is None.")
            return None

        try:
            lat_min, lat_max = Config.JET_SPEED_BOX_LAT_MIN, Config.JET_SPEED_BOX_LAT_MAX
            lon_min, lon_max = Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LON_MAX

            # Check for necessary coordinates and non-empty data
            if not (hasattr(seasonal_u_wind_da, 'lat') and hasattr(seasonal_u_wind_da, 'lon') and
                    seasonal_u_wind_da.lat.size > 0 and seasonal_u_wind_da.lon.size > 0):
                logging.warning(f"JetSpeed: seasonal_u_wind_da (attrs: {seasonal_u_wind_da.attrs.get('dataset_source', 'N/A')}) "
                                f"lacks lat/lon coordinates or they are empty. Lat size: {seasonal_u_wind_da.sizes.get('lat', 0)}, "
                                f"Lon size: {seasonal_u_wind_da.sizes.get('lon', 0)}")
                # Attempt to return a NaN DataArray with correct time dimensions if possible
                time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                if time_coord_name:
                    return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                        coords={time_coord_name: seasonal_u_wind_da[time_coord_name]},
                                        dims=[time_coord_name], name="jet_speed_index")
                return xr.DataArray(np.nan, name="jet_speed_index") # Fallback

            # Determine latitude slice direction
            lat_ascending = seasonal_u_wind_da.lat.values[0] < seasonal_u_wind_da.lat.values[-1]
            lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)

            # Select domain for jet speed calculation
            selection_dict = {'lon': slice(lon_min, lon_max), 'lat': lat_slice}
            domain = seasonal_u_wind_da.sel(**selection_dict)

            if domain.sizes.get('lat', 0) > 0 and domain.sizes.get('lon', 0) > 0:
                # Calculate area weights (cosine of latitude)
                weights = np.cos(np.deg2rad(domain.lat))
                weights.name = "weights" # Important for xarray's weighted operations

                # Perform weighted mean over spatial dimensions
                try:
                    weighted_mean_index = domain.weighted(weights).mean(dim=["lat", "lon"], skipna=True)
                except Exception as e_weighted_mean:
                    logging.warning(f"JetSpeed: xarray weighted().mean() failed ({e_weighted_mean}), attempting manual calculation.")
                    # Manual fallback (ensure spatial_dims_to_sum are actually in domain.dims)
                    spatial_dims_present = [d for d in ["lat", "lon"] if d in domain.dims]
                    if not spatial_dims_present:
                        logging.error("JetSpeed: No spatial dimensions in the selected domain for manual mean.")
                        time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                        if time_coord_name:
                            return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                                coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_speed_index")
                        return xr.DataArray(np.nan, name="jet_speed_index")

                    weighted_sum_manual = (domain * weights).sum(dim=spatial_dims_present, skipna=True)
                    valid_domain_weights = weights.where(domain.notnull()) # Weights only where data is valid
                    total_weight_manual = valid_domain_weights.sum(dim=spatial_dims_present, skipna=True)
                    weighted_mean_index = xr.where(abs(total_weight_manual) > 1e-9, weighted_sum_manual / total_weight_manual, np.nan)

                # Ensure the index is 1D (time dimension, e.g., 'season_year')
                if weighted_mean_index.ndim > 1:
                    time_dim = next((d for d in ['season_year', 'year', 'time'] if d in weighted_mean_index.dims), None)
                    if time_dim:
                        dims_to_squeeze = [d for d in weighted_mean_index.dims if d != time_dim and weighted_mean_index.sizes[d] == 1]
                        if dims_to_squeeze:
                            logging.debug(f"JetSpeed: Squeezing dimensions {dims_to_squeeze} from index. Original Dims: {weighted_mean_index.dims}")
                            weighted_mean_index = weighted_mean_index.squeeze(dim=dims_to_squeeze, drop=True)
                    else: # If no clear time dimension, just try a general squeeze
                        weighted_mean_index = weighted_mean_index.squeeze(drop=True)

                if weighted_mean_index.ndim > 1: # Still not 1D
                    logging.error(f"JetSpeed Index is still >1D after squeeze: Dims {weighted_mean_index.dims}. Index should only depend on time.")
                    # Attempt more aggressive squeeze or specific dimension removal if a known singleton like 'level' exists
                    if 'level' in weighted_mean_index.dims and weighted_mean_index.sizes['level'] == 1:
                        weighted_mean_index = weighted_mean_index.isel(level=0, drop=True)
                        logging.warning("JetSpeed: Aggressively removed 'level' dimension.")
                    if weighted_mean_index.ndim > 1: # Final check
                        logging.error(f"JetSpeed: Could not reduce index to 1D. Current Dims: {weighted_mean_index.dims}")
                        # Return NaN array matching original time dimension if possible
                        time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                        if time_coord_name:
                            return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                                coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_speed_index")
                        return xr.DataArray(np.nan, name="jet_speed_index")


                if 'dataset_source' in seasonal_u_wind_da.attrs:
                    weighted_mean_index.attrs['dataset_source'] = seasonal_u_wind_da.attrs['dataset_source']
                weighted_mean_index.name = "jet_speed_index"
                return weighted_mean_index
            else: # Domain selection resulted in empty lat or lon
                logging.warning(f"JetSpeed: Domain for index calculation is empty for dataset {seasonal_u_wind_da.attrs.get('dataset_source', 'N/A')}.")
                time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                if time_coord_name:
                    return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                        coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_speed_index")
                return xr.DataArray(np.nan, name="jet_speed_index")

        except Exception as e:
            logging.error(f"Error in calculate_jet_speed_index: {e}")
            logging.error(traceback.format_exc())
            # Fallback to NaN array with original time dimension if possible
            time_coord_name = next((d for d in ['season_year', 'year', 'time'] if hasattr(seasonal_u_wind_da, 'coords') and d in seasonal_u_wind_da.coords), None)
            if time_coord_name:
                return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                    coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_speed_index")
            return xr.DataArray(np.nan, name="jet_speed_index")

    @staticmethod
    def calculate_jet_lat_index(seasonal_u_wind_da):
        """
        Calculate jet latitude index from a seasonal U850 DataArray.
        The index is the latitude of the maximum zonally averaged wind, weighted by positive wind speeds.
        Ensures 1D output (time series).

        Args:
            seasonal_u_wind_da (xr.DataArray): Seasonal mean of zonal wind (e.g., U850).

        Returns:
            xr.DataArray: 1D DataArray of the jet latitude index, or None if calculation fails.
        """
        if seasonal_u_wind_da is None:
            logging.debug("JetLat: Input seasonal_u_wind_da is None.")
            return None

        try:
            lat_min, lat_max = Config.JET_LAT_BOX_LAT_MIN, Config.JET_LAT_BOX_LAT_MAX
            lon_min, lon_max = Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LON_MAX

            if not (hasattr(seasonal_u_wind_da, 'lat') and hasattr(seasonal_u_wind_da, 'lon') and
                    seasonal_u_wind_da.lat.size > 0 and seasonal_u_wind_da.lon.size > 0):
                logging.warning(f"JetLat: seasonal_u_wind_da (attrs: {seasonal_u_wind_da.attrs.get('dataset_source', 'N/A')}) "
                                f"lacks lat/lon coordinates or they are empty. Lat size: {seasonal_u_wind_da.sizes.get('lat', 0)}, "
                                f"Lon size: {seasonal_u_wind_da.sizes.get('lon', 0)}")
                time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                if time_coord_name:
                    return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                        coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                return xr.DataArray(np.nan, name="jet_lat_index")

            lat_ascending = seasonal_u_wind_da.lat.values[0] < seasonal_u_wind_da.lat.values[-1]
            lat_slice = slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)

            selection_dict = {'lon': slice(lon_min, lon_max), 'lat': lat_slice}
            domain = seasonal_u_wind_da.sel(**selection_dict)

            if domain.sizes.get('lat', 0) > 0 and domain.sizes.get('lon', 0) > 0:
                lon_dim_name = next((d for d in ["lon", "longitude"] if d in domain.dims), None)
                if not lon_dim_name:
                    logging.error("JetLat: Longitude dimension not found in the selected domain.")
                    time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                    if time_coord_name:
                        return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                            coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                    return xr.DataArray(np.nan, name="jet_lat_index")

                zonal_avg_u = domain.mean(dim=lon_dim_name, skipna=True) # Average over longitudes
                positive_u_winds = zonal_avg_u.where(zonal_avg_u > 0) # Keep only positive wind speeds

                if positive_u_winds.notnull().any(): # Check if there are any non-NaN positive winds
                    lat_dim_name_pu = next((d for d in ["lat", "latitude"] if d in positive_u_winds.dims), None)
                    if not lat_dim_name_pu:
                        logging.error("JetLat: Latitude dimension not found in positive_u_winds.")
                        time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in positive_u_winds.coords), None)
                        if time_coord_name:
                             return xr.DataArray(np.full(positive_u_winds[time_coord_name].shape, np.nan),
                                                 coords={time_coord_name: positive_u_winds[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                        return xr.DataArray(np.nan, name="jet_lat_index")


                    # Calculate sum of positive U winds (denominator for weighted average)
                    sum_positive_u = positive_u_winds.sum(dim=lat_dim_name_pu, skipna=True)

                    # Calculate jet latitude: sum(U_pos * lat) / sum(U_pos)
                    # Ensure lat coordinate is available on positive_u_winds for multiplication
                    if lat_dim_name_pu not in positive_u_winds.coords: # Should be a coordinate if it's a dimension
                        positive_u_winds = positive_u_winds.assign_coords({lat_dim_name_pu: domain[lat_dim_name_pu]})


                    jet_latitude_index = xr.where(
                        abs(sum_positive_u) > 1e-9, # Avoid division by zero
                        (positive_u_winds * positive_u_winds[lat_dim_name_pu]).sum(dim=lat_dim_name_pu, skipna=True) / sum_positive_u,
                        np.nan # Assign NaN if sum of positive winds is effectively zero
                    )

                    # Ensure 1D output
                    if jet_latitude_index.ndim > 1:
                        time_dim = next((d for d in ['season_year', 'year', 'time'] if d in jet_latitude_index.dims), None)
                        if time_dim:
                            dims_to_squeeze = [d for d in jet_latitude_index.dims if d != time_dim and jet_latitude_index.sizes[d] == 1]
                            if dims_to_squeeze:
                                jet_latitude_index = jet_latitude_index.squeeze(dim=dims_to_squeeze, drop=True)
                        else:
                            jet_latitude_index = jet_latitude_index.squeeze(drop=True)
                    
                    if jet_latitude_index.ndim > 1: # Still not 1D
                        logging.error(f"JetLat Index is still >1D after squeeze: Dims {jet_latitude_index.dims}.")
                        if 'level' in jet_latitude_index.dims and jet_latitude_index.sizes['level'] == 1:
                             jet_latitude_index = jet_latitude_index.isel(level=0, drop=True)
                             logging.warning("JetLat: Aggressively removed 'level' dimension.")
                        if jet_latitude_index.ndim > 1:
                             logging.error(f"JetLat: Could not reduce index to 1D. Current Dims: {jet_latitude_index.dims}")
                             time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                             if time_coord_name:
                                 return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                                     coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                             return xr.DataArray(np.nan, name="jet_lat_index")


                    if 'dataset_source' in seasonal_u_wind_da.attrs:
                        jet_latitude_index.attrs['dataset_source'] = seasonal_u_wind_da.attrs['dataset_source']
                    jet_latitude_index.name = "jet_lat_index"
                    return jet_latitude_index
                else: # No positive U-wind values found in the domain
                    logging.info(f"JetLat: No positive U-wind values found in the domain for {seasonal_u_wind_da.attrs.get('dataset_source', 'N/A')}.")
                    time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in zonal_avg_u.coords), None)
                    if time_coord_name:
                        return xr.DataArray(np.full(zonal_avg_u[time_coord_name].shape, np.nan),
                                            coords={time_coord_name: zonal_avg_u[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                    return xr.DataArray(np.nan, name="jet_lat_index")
            else: # Domain selection resulted in empty lat or lon
                logging.warning(f"JetLat: Domain for index calculation is empty for dataset {seasonal_u_wind_da.attrs.get('dataset_source', 'N/A')}.")
                time_coord_name = next((d for d in ['season_year', 'year', 'time'] if d in seasonal_u_wind_da.coords), None)
                if time_coord_name:
                    return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                        coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
                return xr.DataArray(np.nan, name="jet_lat_index")

        except Exception as e:
            logging.error(f"Error in calculate_jet_lat_index: {e}")
            logging.error(traceback.format_exc())
            time_coord_name = next((d for d in ['season_year', 'year', 'time'] if hasattr(seasonal_u_wind_da, 'coords') and d in seasonal_u_wind_da.coords), None)
            if time_coord_name:
                return xr.DataArray(np.full(seasonal_u_wind_da[time_coord_name].shape, np.nan),
                                    coords={time_coord_name: seasonal_u_wind_da[time_coord_name]}, dims=[time_coord_name], name="jet_lat_index")
            return xr.DataArray(np.nan, name="jet_lat_index")