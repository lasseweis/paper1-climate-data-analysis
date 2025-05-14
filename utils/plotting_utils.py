#!/usr/bin/env python3
"""
Utility functions for creating various plots for climate analysis results.
"""
import logging
import os
import numpy as np
import pandas as pd # Used in plot_seasonal_correlation_matrix
import xarray as xr # Though not directly creating, it often handles data passed to plots

import matplotlib
# Important: matplotlib.use('Agg') should be called in the *main executable script*
# BEFORE importing pyplot, if running in a non-GUI environment.
# Do not call it here as this is a utility module.
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata # For regridding in plot_jet_impact_maps

# Relative imports for utility modules
from config_setup import Config
from stats_utils import StatsAnalyzer # For normalization within plotting functions

class Visualizer:
    """Visualization methods for climate analysis results."""

    @staticmethod
    def ensure_plot_dir_exists():
        """Ensure the plot directory defined in Config exists."""
        # Config.ensure_dir_exists(Config.PLOT_DIR) # Config class now handles this in __init__ or statically
        # For safety, ensure it here too if Config isn't instantiated or if PLOT_DIR might change.
        if not os.path.exists(Config.PLOT_DIR):
            os.makedirs(Config.PLOT_DIR, exist_ok=True)
            logging.info(f"Created plot directory: {Config.PLOT_DIR}")

    @staticmethod
    def plot_regression_map(ax, slopes_data, p_values_data, lons_mesh, lats_mesh,
                            map_title, analysis_box_coords, season_label_str,
                            variable_type_str, u_wind_mean_for_contours=None,
                            show_jet_info_boxes=False, significance_level_alpha=0.05,
                            custom_slope_levels=None):
        """
        Creates a single regression map on a given Matplotlib Axes object.

        Args:
            ax (matplotlib.axes.Axes): The axes object to plot on.
            slopes_data (np.ndarray): 2D array of regression slopes.
            p_values_data (np.ndarray): 2D array of p-values for significance.
            lons_mesh (np.ndarray): 2D array of longitudes.
            lats_mesh (np.ndarray): 2D array of latitudes.
            map_title (str): Title for the map.
            analysis_box_coords (list/tuple): [lon_min, lon_max, lat_min, lat_max] for the analysis box.
            season_label_str (str): Season label (e.g., "DJF", "JJA").
            variable_type_str (str): Type of variable being regressed ('pr' for precipitation, 'tas' for temperature).
            u_wind_mean_for_contours (np.ndarray, optional): 2D array of mean U-wind for contours. Defaults to None.
            show_jet_info_boxes (bool, optional): Whether to show jet index boxes. Defaults to False.
            significance_level_alpha (float, optional): P-value threshold for significance. Defaults to 0.05.
            custom_slope_levels (np.array, optional): Custom levels for the slope colorbar.
        Returns:
            matplotlib.collections.QuadMesh or ContourSet, str: The plot object (e.g., from pcolormesh) and colorbar label.
        """
        # Define color limits and labels based on variable type
        if variable_type_str == 'pr':
            # Assuming slope represents change in U850 (m/s) per % change in box precipitation
            default_levels = np.linspace(-0.15, 0.15, Config.COLORBAR_LEVELS) if Config.COLORBAR_LEVELS > 1 else np.array([-0.15, 0, 0.15])
            colorbar_label = 'U850 Slope (m/s per % change in box precip)'
        elif variable_type_str == 'tas':
            # Assuming slope represents change in U850 (m/s) per °C change in box temperature
            default_levels = np.linspace(-2.5, 2.5, Config.COLORBAR_LEVELS) if Config.COLORBAR_LEVELS > 1 else np.array([-2.5, 0, 2.5])
            colorbar_label = 'U850 Slope (m/s per °C change in box temp)'
        else: # Fallback for unknown variable type
            valid_slopes = slopes_data[np.isfinite(slopes_data)]
            vmin = np.percentile(valid_slopes, 5) if len(valid_slopes) > 0 else -1
            vmax = np.percentile(valid_slopes, 95) if len(valid_slopes) > 0 else 1
            default_levels = np.linspace(vmin, vmax, Config.COLORBAR_LEVELS) if Config.COLORBAR_LEVELS > 1 else np.array([vmin, (vmin+vmax)/2, vmax])
            colorbar_label = 'Regression Slope Value'

        levels_to_use = custom_slope_levels if custom_slope_levels is not None else default_levels
        vmin_plot, vmax_plot = levels_to_use[0], levels_to_use[-1]

        ax.set_extent(Config.MAP_EXTENT, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5, zorder=1)
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
        gl.top_labels = gl.right_labels = False # Show labels only on left and bottom
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}

        # Use pcolormesh for filled contours; robust for potentially non-uniform grids
        plot_obj = ax.pcolormesh(lons_mesh, lats_mesh, slopes_data, shading='auto',
                                 cmap=Config.COLORMAP, vmin=vmin_plot, vmax=vmax_plot,
                                 transform=ccrs.PlateCarree())

        # Add contours of mean U-wind if provided
        if u_wind_mean_for_contours is not None and np.any(np.isfinite(u_wind_mean_for_contours)):
            u_max_abs = np.ceil(np.nanmax(np.abs(u_wind_mean_for_contours)))
            contour_step = 2 # m/s
            wind_levels = np.arange(-u_max_abs, u_max_abs + contour_step, contour_step)
            wind_levels = wind_levels[wind_levels != 0] # Exclude zero contour usually
            if len(wind_levels) > 1:
                cs_wind = ax.contour(lons_mesh, lats_mesh, u_wind_mean_for_contours,
                                     levels=wind_levels, colors='black', linewidths=0.8,
                                     transform=ccrs.PlateCarree())
                ax.clabel(cs_wind, inline=True, fontsize=7, fmt='%d')

        # Add significance hatching or stippling
        significant_mask = (p_values_data < significance_level_alpha) & np.isfinite(slopes_data)
        if np.any(significant_mask):
            # Using small dots for stippling
            ax.scatter(lons_mesh[significant_mask], lats_mesh[significant_mask],
                       s=0.5, color='black', marker='.', alpha=0.7, # Smaller, less intrusive dots
                       transform=ccrs.PlateCarree(), label=f'p < {significance_level_alpha}')

        # Add analysis box
        lon_min_box, lon_max_box, lat_min_box, lat_max_box = analysis_box_coords
        rect_patch = mpatches.Rectangle((lon_min_box, lat_min_box), lon_max_box - lon_min_box, lat_max_box - lat_min_box,
                                        fill=False, edgecolor='lime', linewidth=1.5, zorder=10,
                                        transform=ccrs.PlateCarree(), label='Analysis Box')
        ax.add_patch(rect_patch)

        # Add jet information boxes if requested
        if show_jet_info_boxes:
            jet_speed_box_patch = mpatches.Rectangle(
                (Config.JET_SPEED_BOX_LON_MIN, Config.JET_SPEED_BOX_LAT_MIN),
                Config.JET_SPEED_BOX_LON_MAX - Config.JET_SPEED_BOX_LON_MIN,
                Config.JET_SPEED_BOX_LAT_MAX - Config.JET_SPEED_BOX_LAT_MIN,
                fill=False, edgecolor='blue', linewidth=1, linestyle='--', zorder=10,
                transform=ccrs.PlateCarree(), label='Jet Speed Box')
            ax.add_patch(jet_speed_box_patch)

            jet_lat_box_patch = mpatches.Rectangle(
                (Config.JET_LAT_BOX_LON_MIN, Config.JET_LAT_BOX_LAT_MIN),
                Config.JET_LAT_BOX_LON_MAX - Config.JET_LAT_BOX_LON_MIN,
                Config.JET_LAT_BOX_LAT_MAX - Config.JET_LAT_BOX_LAT_MIN,
                fill=False, edgecolor='red', linewidth=1, linestyle='--', zorder=10,
                transform=ccrs.PlateCarree(), label='Jet Latitude Box')
            ax.add_patch(jet_lat_box_patch)

        ax.set_title(f"{map_title}\n{season_label_str}", fontsize=10)
        
        # Add legend if there are labeled items
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='lower right', fontsize=7, frameon=False)

        return plot_obj, colorbar_label, levels_to_use


    @staticmethod
    def plot_regression_analysis_figure(seasonal_regression_data_dict, dataset_id_str, filename_suffix="regression_maps"):
        """
        Creates a figure with regression maps for PR and TAS, for Winter and Summer.

        Args:
            seasonal_regression_data_dict (dict): Data from AdvancedAnalyzer.calculate_regression_maps.
                                                  e.g., {'Winter': {'slopes_pr':..., 'p_values_pr':...}, 'Summer':{...}}
            dataset_id_str (str): Identifier for the dataset (e.g., "ERA5", "20CRv3", "CMIP6 MMM").
            filename_suffix (str): Suffix for the output plot filename.
        """
        logging.info(f"Plotting U850 vs Box Index regression maps for {dataset_id_str}...")
        Visualizer.ensure_plot_dir_exists()

        if not isinstance(seasonal_regression_data_dict, dict) or \
           not all(s in seasonal_regression_data_dict for s in ['Winter', 'Summer']):
            logging.warning(f"Warning: Missing 'Winter' or 'Summer' data in regression results for {dataset_id_str}. Skipping plot.")
            return

        fig = plt.figure(figsize=(14, 10)) # Adjusted for potentially better layout
        # GridSpec: 2 rows (variables: PR, TAS) x 2 cols (seasons: Winter, Summer) + 1 col for colorbars
        gs = gridspec.GridSpec(2, 3, width_ratios=[10, 10, 1.2], height_ratios=[1, 1],
                               wspace=0.05, hspace=0.25) # Adjusted spacing
        
        analysis_box_definition = [Config.BOX_LON_MIN, Config.BOX_LON_MAX, Config.BOX_LAT_MIN, Config.BOX_LAT_MAX]
        
        plot_objects_for_colorbar = {'pr': None, 'tas': None}
        colorbar_labels = {'pr': "", 'tas': ""}
        colorbar_levels = {'pr': None, 'tas': None}


        for i, var_type in enumerate(['pr', 'tas']): # Rows for PR, then TAS
            for j, season in enumerate(['Winter', 'Summer']): # Columns for Winter, then Summer
                ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
                season_data = seasonal_regression_data_dict.get(season, {})
                
                slopes = season_data.get(f'slopes_{var_type}')
                p_values = season_data.get(f'p_values_{var_type}')
                lons = season_data.get('lons')
                lats = season_data.get('lats')
                u_mean = season_data.get('ua850_mean_for_contours')

                if all(d is not None for d in [slopes, p_values, lons, lats]):
                    plot_obj, cb_label, cb_levels = Visualizer.plot_regression_map(
                        ax, slopes, p_values, lons, lats,
                        f"{dataset_id_str}: U850 vs {var_type.upper()} Box Index",
                        analysis_box_definition, season.upper(), var_type,
                        ua_seasonal_mean=u_mean, show_jet_info_boxes=True
                    )
                    if plot_objects_for_colorbar[var_type] is None: # Store for colorbar creation
                        plot_objects_for_colorbar[var_type] = plot_obj
                        colorbar_labels[var_type] = cb_label
                        colorbar_levels[var_type] = cb_levels
                else:
                    ax.text(0.5, 0.5, "Data Missing", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                    ax.set_title(f"{dataset_id_str}: {var_type.upper()} vs U850 ({season.upper()})\nData Missing", fontsize=10)

        # Add Colorbars
        if plot_objects_for_colorbar['pr'] and colorbar_levels['pr'] is not None:
            cax_pr = fig.add_subplot(gs[0, 2]) # Colorbar for PR in the first row, third column
            cbar_pr = fig.colorbar(plot_objects_for_colorbar['pr'], cax=cax_pr, extend='both', ticks=colorbar_levels['pr'][::max(1, len(colorbar_levels['pr'])//5)]) # Reduce ticks if too many
            cbar_pr.set_label(colorbar_labels['pr'], fontsize=9)
            cbar_pr.ax.tick_params(labelsize=8)
        
        if plot_objects_for_colorbar['tas'] and colorbar_levels['tas'] is not None:
            cax_tas = fig.add_subplot(gs[1, 2]) # Colorbar for TAS in the second row, third column
            cbar_tas = fig.colorbar(plot_objects_for_colorbar['tas'], cax=cax_tas, extend='both', ticks=colorbar_levels['tas'][::max(1, len(colorbar_levels['tas'])//5)])
            cbar_tas.set_label(colorbar_labels['tas'], fontsize=9)
            cbar_tas.ax.tick_params(labelsize=8)

        plt.suptitle(f"{dataset_id_str}: U850 Regression onto Box Climate Indices (Detrended)", fontsize=14, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.95, 0.96]) # Adjust rect to make space for suptitle and colorbars
        
        filename = os.path.join(Config.PLOT_DIR, f'{filename_suffix}_{dataset_id_str.lower().replace(" ", "_")}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved regression analysis maps for {dataset_id_str} to {filename}")
        plt.close(fig)


    @staticmethod
    def plot_jet_indices_timeseries(jet_data_all_datasets: dict, filename="jet_indices_comparison_seasonal_detrended.png"):
        """
        Plot timeseries of jet speed and latitude indices for Winter and Summer, comparing datasets.
        Assumes jet_data_all_datasets contains keys like 'ERA5_winter_speed_tas_data' (from AdvancedAnalyzer.analyze_jet_indices)
        which then contains the 'jet' (detrended) and 'years' DataArrays.
        """
        logging.info("Plotting comparative jet indices timeseries (detrended)...")
        Visualizer.ensure_plot_dir_exists()

        fig, axs = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
        datasets_to_plot = [Config.DATASET_ERA5, Config.DATASET_20CRV3] # Order of plotting
        colors = {Config.DATASET_ERA5: 'red', Config.DATASET_20CRV3: 'blue'}

        plot_configs = [
            {'ax_coords': (0, 0), 'season': 'Winter', 'index_type': 'speed', 'ylabel': 'Wind Speed (m/s)'},
            {'ax_coords': (0, 1), 'season': 'Summer', 'index_type': 'speed', 'ylabel': 'Wind Speed (m/s)'},
            {'ax_coords': (1, 0), 'season': 'Winter', 'index_type': 'lat',   'ylabel': 'Latitude (°N)'},
            {'ax_coords': (1, 1), 'season': 'Summer', 'index_type': 'lat',   'ylabel': 'Latitude (°N)'},
        ]
        
        all_years_union = []

        for config in plot_configs:
            ax = axs[config['ax_coords']]
            season_l = config['season'].lower()
            title = f'{config["season"]} Jet {config["index_type"].capitalize()} Index (Detrended)'
            ax.set_title(title, fontsize=11)
            ax.set_ylabel(config['ylabel'], fontsize=10)
            if config['ax_coords'][0] == 1: # Bottom row
                ax.set_xlabel('Year', fontsize=10)
            ax.grid(True, alpha=0.5, linestyle=':')
            ax.tick_params(axis='both', which='major', labelsize=9)
            
            legend_handles_subplot = []

            for dataset_id in datasets_to_plot:
                # Construct the key for the detrended jet index from jet_data_all_datasets
                # The analyze_jet_indices stores detrended jet as, e.g., 'ERA5_jet_speed_winter_detrended'
                detrended_jet_key = f"{dataset_id}_jet_{config['index_type']}_{season_l}_detrended"
                
                index_data_array = jet_data_all_datasets.get(detrended_jet_key)

                if index_data_array is not None and index_data_array.size > 0 and 'season_year' in index_data_array.coords:
                    years = index_data_array.season_year.values
                    values = index_data_array.data
                    all_years_union.extend(years)
                    line, = ax.plot(years, values, '-', color=colors.get(dataset_id, 'gray'), linewidth=1.5, label=dataset_id)
                    legend_handles_subplot.append(line)
                else:
                    logging.warning(f"  Plotting: Detrended jet data not found or empty for {detrended_jet_key}.")
            
            if legend_handles_subplot:
                ax.legend(handles=legend_handles_subplot, fontsize=8)
        
        if all_years_union:
            fig_title_time_period = f"({int(min(all_years_union))}-{int(max(all_years_union))})"
        else:
            fig_title_time_period = "(Time period N/A)"

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for main title
        plt.suptitle(f"Jet Stream Indices Comparison (Detrended): {', '.join(datasets_to_plot)} {fig_title_time_period}",
                     fontsize=14, weight='bold')
        
        output_filename = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved seasonal detrended jet indices comparison to {output_filename}")
        plt.close(fig)

    @staticmethod
    def plot_seasonal_correlation_matrix(correlations_data_era5: dict, correlations_data_20crv3: dict,
                                         season: str, filename_suffix="correlation_matrix_comparison_detrended"):
        """
        Plot a comprehensive correlation matrix comparing ERA5 and 20CRv3 correlations for a given season.
        Args:
            correlations_data_era5 (dict): Output from AdvancedAnalyzer.analyze_correlations for ERA5.
            correlations_data_20crv3 (dict): Output from AdvancedAnalyzer.analyze_correlations for 20CRv3.
            season (str): "Winter" or "Summer".
        """
        season_lower = season.lower()
        logging.info(f"Plotting comprehensive {season} correlation matrix for ERA5 vs 20CRv3...")
        Visualizer.ensure_plot_dir_exists()

        plot_data_list = []
        # Structure of correlations_data: {'winter': {'discharge_jet_speed': res}, 'pr': {'winter': {'lat': res}}}
        for dataset_id, dataset_corr_full in [("ERA5", correlations_data_era5), ("20CRv3", correlations_data_20crv3)]:
            if not dataset_corr_full: continue

            # Direct seasonal correlations (e.g., discharge)
            season_specific_direct_corrs = dataset_corr_full.get(season_lower, {})
            for key, corr_data_item in season_specific_direct_corrs.items():
                if 'r_value' in corr_data_item and 'p_value' in corr_data_item:
                    plot_data_list.append({
                        'correlation': corr_data_item['r_value'],
                        'p_value': corr_data_item['p_value'],
                        'label': f"{corr_data_item['variable1_name']} vs {corr_data_item['variable2_name'].replace(f'{dataset_id} ', '').replace(' (detrended)','')}",
                        'base_label': f"{corr_data_item['variable1_name']} vs {corr_data_item['variable2_name'].split('(')[0].replace(f'{dataset_id} ', '').strip()}", # For grouping
                        'category': 'discharge_flow' if 'discharge' in key or 'flow' in key else 'other',
                        'dataset': dataset_id
                    })
            
            # Nested PR/TAS correlations
            for var_category in ['pr', 'tas']:
                var_season_corrs = dataset_corr_full.get(var_category, {}).get(season_lower, {})
                for jet_dim_key, corr_data_item in var_season_corrs.items(): # jet_dim_key is 'speed' or 'lat'
                     if 'r_value' in corr_data_item and 'p_value' in corr_data_item:
                        plot_data_list.append({
                            'correlation': corr_data_item['r_value'],
                            'p_value': corr_data_item['p_value'],
                            'label': f"{corr_data_item['variable1_name']} vs {corr_data_item['variable2_name'].replace(f'{dataset_id} ', '').replace(' (detrended)','')}",
                            'base_label': f"{corr_data_item['variable1_name']} vs {corr_data_item['variable2_name'].split('(')[0].replace(f'{dataset_id} ', '').strip()}",
                            'category': 'precipitation' if var_category == 'pr' else 'temperature',
                            'dataset': dataset_id
                        })
        
        if not plot_data_list:
            logging.info(f"No {season} correlation data found for matrix plot.")
            return

        df_plot = pd.DataFrame(plot_data_list)
        df_plot['abs_correlation'] = df_plot['correlation'].abs()
        df_plot.dropna(subset=['correlation'], inplace=True)
        if df_plot.empty:
            logging.info(f"No valid correlation data after processing for {season} matrix plot.")
            return

        # Sort by mean absolute correlation of base_label, then by dataset for consistent ordering
        grouped_mean_abs_corr = df_plot.groupby('base_label')['abs_correlation'].mean().sort_values(ascending=False)
        df_plot['sort_order'] = df_plot['base_label'].map(grouped_mean_abs_corr) # Higher mean abs corr first
        df_plot.sort_values(by=['sort_order', 'dataset'], ascending=[False, True], inplace=True) # Sort by mean, then dataset

        fig, ax = plt.subplots(figsize=(10, max(5, len(df_plot) * 0.3))) # Dynamic height
        cmap_used = plt.cm.RdBu_r
        norm_used = Normalize(vmin=-1, vmax=1)
        
        y_positions = np.arange(len(df_plot))
        dataset_markers = {Config.DATASET_ERA5: 's', Config.DATASET_20CRV3: 'o'}
        dataset_colors_bar = {Config.DATASET_ERA5: 'salmon', Config.DATASET_20CRV3: 'lightskyblue'} # For bars
        dataset_colors_marker = {Config.DATASET_ERA5: 'red', Config.DATASET_20CRV3: 'blue'} # For markers

        for i, row_data in df_plot.iterrows():
            bar_color_val = cmap_used(norm_used(row_data['correlation']))
            # Use distinct fixed colors for bars to differentiate datasets better than just cmap
            bar_color_ds = dataset_colors_bar[row_data['dataset']]

            ax.barh(y_positions[df_plot.index.get_loc(i)], row_data['correlation'], 
                    align='center', color=bar_color_ds, height=0.4, alpha=0.8) # Thinner bars
            
            # Add significance text
            corr_val_text = f"{row_data['correlation']:.2f}"
            if row_data['p_value'] < 0.001: corr_val_text += "***"
            elif row_data['p_value'] < 0.01: corr_val_text += "**"
            elif row_data['p_value'] < 0.05: corr_val_text += "*"
            
            text_x_pos = row_data['correlation'] + (0.02 * np.sign(row_data['correlation'])) if row_data['correlation'] != 0 else 0.02
            ha_align = 'left' if row_data['correlation'] >= 0 else 'right'
            ax.text(text_x_pos, y_positions[df_plot.index.get_loc(i)], corr_val_text,
                    ha=ha_align, va='center', fontsize=7, color='black')

        # Y-axis: Use unique base_labels for ticks
        unique_base_labels = df_plot['base_label'].unique()
        # Find mid-point y_position for each unique_base_label for cleaner y-ticks
        ytick_positions_final = []
        ytick_labels_final = []
        for base_lbl in unique_base_labels:
            indices = df_plot[df_plot['base_label'] == base_lbl].index
            y_pos_for_label = y_positions[df_plot.index.get_loc(indices.mean())] # approx mid y-pos
            ytick_positions_final.append(y_pos_for_label)
            ytick_labels_final.append(base_lbl.replace(f"{season} ", "")) # Shorten label


        ax.set_yticks(ytick_positions_final)
        ax.set_yticklabels(ytick_labels_final, fontsize=8)

        ax.invert_yaxis() # Highest abs correlation at top
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.7, alpha=0.6)
        ax.set_xlabel('Correlation Coefficient (r)', fontsize=10)
        ax.set_title(f'{season} Correlation Analysis: ERA5 vs 20CRv3 (Detrended)', fontsize=12, weight='bold')
        ax.grid(True, axis='x', alpha=0.4, linestyle=':')
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlim([-1, 1])


        # Custom legend for datasets
        legend_elements = [
            mpatches.Patch(facecolor=dataset_colors_bar[Config.DATASET_ERA5], label='ERA5', alpha=0.8),
            mpatches.Patch(facecolor=dataset_colors_bar[Config.DATASET_20CRV3], label='20CRv3', alpha=0.8)
        ]
        ax.legend(handles=legend_elements, loc='lower right', title="Dataset", fontsize=8, title_fontsize=9)
        plt.figtext(0.1, 0.01, "* p<0.05, ** p<0.01, *** p<0.001", ha="left", fontsize=8)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        filename = os.path.join(Config.PLOT_DIR, f'{filename_suffix}_{season_lower}.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved {season} detrended comparative correlation matrix to {filename}")
        plt.close(fig)


    # Add other plotting methods from your original Visualizer class here,
    # adapting them to use data passed as arguments rather than from a shared 'results' dict.
    # Examples: plot_seasonal_correlations_timeseries, plot_jet_impact_maps,
    # plot_amo_jet_correlations, plot_cmip6_jet_changes_vs_gwl, plot_storyline_impacts
    
    @staticmethod
    def plot_seasonal_correlations_timeseries(correlations_data_era5: dict, correlations_data_20crv3: dict, 
                                              season: str, filename_suffix="correlations_comparison_detrended_ts"):
        """
        Plot selected seasonal correlation timeseries for ERA5 and 20CRv3.
        Data comes from AdvancedAnalyzer.analyze_correlations.
        """
        season_lower = season.lower()
        logging.info(f"Plotting {season} correlation timeseries comparison (ERA5 vs 20CRv3)...")
        Visualizer.ensure_plot_dir_exists()

        # Define which specific correlations to plot as timeseries
        # Format: (var_category_in_dict, season_key_in_dict, jet_dim_key_in_dict_if_pr_tas, output_title_key_part1, output_title_key_part2)
        # For direct keys like 'discharge_jet_speed': (season_lower, 'discharge_jet_speed', None, 'Discharge', 'Jet Speed')
        plot_configurations = [
            ('tas', season_lower, 'speed', f'{season} Temperature', 'Jet Speed'),
            ('pr',  season_lower, 'lat',   f'{season} Precipitation', 'Jet Latitude'),
            ('tas', season_lower, 'lat',   f'{season} Temperature', 'Jet Latitude'),
            (season_lower, 'discharge_jet_speed', None, f'{season} Discharge', 'Jet Speed'),
            (season_lower, 'extreme_flow_jet_speed', None, f'{season} Extreme Flow', 'Jet Speed'),
            ('pr',  season_lower, 'speed', f'{season} Precipitation', 'Jet Speed'),
        ]
        
        available_plots_configs = []
        for config_tuple in plot_configurations:
            cat, skey, jkey, _, _ = config_tuple
            # Check if data exists in at least one dataset for this configuration
            found_in_any = False
            for ds_corr_data in [correlations_data_era5, correlations_data_20crv3]:
                if not ds_corr_data: continue
                data_item = None
                if jkey is None: # Direct key like 'discharge_jet_speed'
                    data_item = ds_corr_data.get(cat, {}).get(skey)
                else: # Nested like pr/tas -> winter/summer -> speed/lat
                    data_item = ds_corr_data.get(cat, {}).get(skey, {}).get(jkey)
                if data_item and 'values1_on_common_years' in data_item:
                    found_in_any = True; break
            if found_in_any:
                available_plots_configs.append(config_tuple)
        
        if not available_plots_configs:
            logging.info(f"No data available to plot {season} correlation timeseries comparison.")
            return

        n_plots = len(available_plots_configs)
        n_cols = min(3, n_plots) # Max 3 plots per row
        n_rows = (n_plots + n_cols - 1) // n_cols
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.5 * n_rows), squeeze=False, sharex=True)
        axs_flat = axs.flatten()

        dataset_colors = {Config.DATASET_ERA5: 'red', Config.DATASET_20CRV3: 'blue'}

        for i, (cat_key, season_dict_key, jet_dim_key, title_var1, title_var2) in enumerate(available_plots_configs):
            ax = axs_flat[i]
            plot_title = f"{title_var1} vs {title_var2}"
            legend_items_ax = []
            # Store R-values to add to legend or text later
            r_values_text = []


            for dataset_id, color in dataset_colors.items():
                dataset_corr_data_full = correlations_data_era5 if dataset_id == Config.DATASET_ERA5 else correlations_data_20crv3
                if not dataset_corr_data_full: continue

                correlation_data_item = None
                if jet_dim_key is None: # Direct key (e.g., for discharge)
                    correlation_data_item = dataset_corr_data_full.get(cat_key, {}).get(season_dict_key)
                else: # Nested (for pr/tas)
                    correlation_data_item = dataset_corr_data_full.get(cat_key, {}).get(season_dict_key, {}).get(jet_dim_key)

                if correlation_data_item and 'values1_on_common_years' in correlation_data_item and 'values2_on_common_years' in correlation_data_item:
                    # Retrieve common years from the data item itself (should be there from analyze_correlations)
                    # Assuming analyze_correlations stores actual year values if possible, or range.
                    # For plotting, we need the actual year values for x-axis.
                    # This part requires analyze_correlations to store 'common_year_values' or similar.
                    # For now, let's assume 'values1_on_common_years' implies an x-axis of its length
                    # This needs careful handling of the X-axis (years). Let's assume the *length* implies common period.
                    
                    # For simplicity, we will use the years from the *first* valid dataset that provides them for this plot.
                    # This requires analyze_correlations to store `common_year_values` if they are not directly indices.
                    # Let's assume for now that years can be reconstructed or are implicitly handled by plotting index vs index.
                    # A better way: `analyze_correlations` should return `common_year_coords` in its result dict.
                    # Fallback: create dummy years if not available, or get from first series.

                    vals1_norm = StatsAnalyzer.normalize_series(correlation_data_item['values1_on_common_years'])
                    vals2_norm = StatsAnalyzer.normalize_series(correlation_data_item['values2_on_common_years'])
                    
                    # Determine x-axis (years). This is tricky without explicit year coordinates in the results.
                    # Assuming the stored values are numpy arrays. We need corresponding year coordinates.
                    # This part needs `analyze_correlations` to also return the `common_year_coords`
                    # For now, if we assume vals1_norm and vals2_norm are aligned and cover the same period:
                    
                    # Create a dummy x-axis if no year info (e.g. if only values were stored)
                    num_points = len(vals1_norm)
                    x_axis_years = np.arange(num_points) # Placeholder. Replace with actual years if available.
                    # TODO: Modify analyze_correlations to return the actual 'common_year_coords'

                    # Plot normalized, detrended time series
                    # Label for var1 (e.g. Temperature) - use a generic approach
                    var1_label_short = title_var1.replace(f"{season} ", "")
                    var2_label_short = title_var2.replace("Jet ", "J.") # Abbreviate

                    line1, = ax.plot(x_axis_years, vals1_norm, '-', color=color, linewidth=1.2, alpha=0.7, label=f"{dataset_id} {var1_label_short}")
                    line2, = ax.plot(x_axis_years, vals2_norm, '--', color=color, linewidth=1.5, label=f"{dataset_id} {var2_label_short}")
                    
                    # Collect r-value for text annotation
                    r_val = correlation_data_item.get('r_value', np.nan)
                    p_val = correlation_data_item.get('p_value', np.nan)
                    sig_stars = ""
                    if p_val < 0.001: sig_stars = "***"
                    elif p_val < 0.01: sig_stars = "**"
                    elif p_val < 0.05: sig_stars = "*"
                    r_values_text.append(f"{dataset_id}: r={r_val:.2f}{sig_stars}")

            ax.set_title(plot_title, fontsize=10)
            ax.grid(True, alpha=0.4, linestyle=':')
            if i % n_cols == 0: ax.set_ylabel('Normalized Value (Detrended)', fontsize=9)
            if i // n_cols == n_rows - 1: ax.set_xlabel('Year Index (Requires actual years)', fontsize=9) # TODO: Fix X-axis
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add R-value text annotations
            for k, r_text in enumerate(r_values_text):
                 ax.text(0.03, 0.95 - (k * 0.07), r_text, transform=ax.transAxes, 
                         color=list(dataset_colors.values())[k % len(dataset_colors)], # Cycle through colors
                         fontweight='bold', fontsize=7, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

            ax.legend(loc='best', fontsize=7, ncol=2)


        # Remove empty subplots if layout is not full
        for j_empty in range(n_plots, n_rows * n_cols):
            fig.delaxes(axs_flat[j_empty])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(f"{season} Detrended Time Series Correlations: ERA5 vs 20CRv3", fontsize=14, weight='bold')
        
        output_filename = os.path.join(Config.PLOT_DIR, f'{filename_suffix}_{season_lower}.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved {season} correlation timeseries comparison to {output_filename}")
        plt.close(fig)

    # ... (Other plot methods like plot_jet_impact_maps, plot_amo_jet_correlations,
    #      plot_cmip6_jet_changes_vs_gwl, plot_storyline_impacts would go here)
    # These will be substantial and require careful adaptation of the original logic.

    @staticmethod
    def plot_cmip6_jet_changes_vs_gwl(cmip6_gwl_analysis_results: dict, filename="cmip6_jet_changes_vs_gwl.png"):
        """
        Plots CMIP6 jet index changes vs. Global Warming Level (GWL).
        Similar to Figure 7 in Harvey et al. (2023).

        Args:
            cmip6_gwl_analysis_results (dict): Output from StorylineAnalyzer.analyze_cmip6_changes_at_gwl.
                                             Needs 'cmip6_metric_values_at_gwl_and_ref_per_model'
                                             and 'cmip6_mmm_changes_at_gwl'.
            filename (str): Name of the output plot file.
        """
        logging.info("Plotting CMIP6 Jet Changes vs. Global Warming Level...")
        Visualizer.ensure_plot_dir_exists()

        if not cmip6_gwl_analysis_results or \
           'cmip6_metric_values_at_gwl_and_ref_per_model' not in cmip6_gwl_analysis_results or \
           'cmip6_mmm_changes_at_gwl' not in cmip6_gwl_analysis_results:
            logging.info("  Cannot plot Jet Changes vs GWL: Missing required CMIP6 analysis results.")
            return

        model_metric_values = cmip6_gwl_analysis_results['cmip6_metric_values_at_gwl_and_ref_per_model']
        mmm_gwl_changes = cmip6_gwl_analysis_results['cmip6_mmm_changes_at_gwl']
        
        # GWLs to plot are those for which MMM data exists
        gwls_to_plot = sorted([gwl for gwl in Config.GLOBAL_WARMING_LEVELS if mmm_gwl_changes.get(gwl) is not None])
        if not gwls_to_plot:
            logging.info("  Cannot plot Jet Changes vs GWL: No valid GWL data in MMM results.")
            return

        # Jet indices to plot are typically defined by storyline configurations
        # These should be keys like 'DJF_JetSpeed', 'JJA_JetLat'
        jet_indices_for_plot = list(Config.STORYLINE_JET_CHANGES.keys())
        num_indices = len(jet_indices_for_plot)
        if num_indices == 0:
            logging.info("  Cannot plot Jet Changes vs GWL: No jet indices defined in Config.STORYLINE_JET_CHANGES.")
            return

        fig, axs = plt.subplots(1, num_indices, figsize=(6 * num_indices, 5.5), sharey=False, squeeze=False)
        axs_flat = axs.flatten() # Ensure axs_flat is always an array

        for i, jet_idx_key_name in enumerate(jet_indices_for_plot):
            ax = axs_flat[i]
            
            # Plot individual model lines (change relative to their own reference period mean)
            for model_name, model_data in model_metric_values.items():
                if 'ref' not in model_data or jet_idx_key_name not in model_data['ref']:
                    continue # Skip model if no reference value for this jet index
                
                ref_period_value = model_data['ref'][jet_idx_key_name]
                if np.isnan(ref_period_value): continue

                model_gwl_points_x = [] # GWL values
                model_jet_changes_y = [] # Change in jet index

                for gwl_val in gwls_to_plot:
                    if gwl_val in model_data and jet_idx_key_name in model_data[gwl_val]:
                        gwl_period_value = model_data[gwl_val][jet_idx_key_name]
                        if not np.isnan(gwl_period_value):
                            model_gwl_points_x.append(gwl_val)
                            model_jet_changes_y.append(gwl_period_value - ref_period_value)
                
                if model_gwl_points_x: # Only plot if there's data for this model and jet index
                     # Limit plotting individual lines if too many models, e.g. > 20
                    if len(model_metric_values) <= 20 :
                         ax.plot(model_gwl_points_x, model_jet_changes_y, marker='.', linestyle='-',
                                 color='grey', alpha=0.35, markersize=4, linewidth=0.7)
            
            if len(model_metric_values) > 20:
                 ax.text(0.97, 0.03, f"{len(model_metric_values)} models\n(indiv. lines hidden)",
                         transform=ax.transAxes, ha='right', va='bottom', fontsize=7, color='grey')


            # Plot MMM changes (already calculated as Delta relative to MMM reference)
            mmm_jet_changes = [mmm_gwl_changes[gwl].get(jet_idx_key_name) for gwl in gwls_to_plot]
            ax.plot(gwls_to_plot, mmm_jet_changes, marker='o', linestyle='-', color='black',
                    linewidth=2.0, markersize=6, label='Multi-Model Mean Change')

            # Plot spread (e.g., 10-90th percentile of individual model changes)
            # This uses the '_all_model_deltas' stored during MMM calculation
            p10_values = []
            p90_values = []
            for gwl_val in gwls_to_plot:
                delta_values_all_models = mmm_gwl_changes[gwl_val].get(f"{jet_idx_key_name}_all_model_deltas", [])
                valid_deltas = delta_values_all_models[np.isfinite(delta_values_all_models)]
                if len(valid_deltas) >= 5: # Need a few models for meaningful percentiles
                    p10_values.append(np.percentile(valid_deltas, 10))
                    p90_values.append(np.percentile(valid_deltas, 90))
                else:
                    p10_values.append(np.nan)
                    p90_values.append(np.nan)
            
            if not all(np.isnan(p) for p in p10_values): # Only plot if data exists
                ax.fill_between(gwls_to_plot, p10_values, p90_values, color='lightcoral', alpha=0.5, label='10-90th Percentile Spread')

            # Plot defined Storyline Jet Change values from Config
            storyline_marker_styles = {'Core Mean': 's', 'Core High': 'D', 'Extreme Low': '^', 'Extreme High': 'v'}
            storyline_colors = {'Core Mean': 'blue', 'Core High': 'darkblue', 'Extreme Low': 'purple', 'Extreme High': 'magenta'}
            legend_handles_storylines = {}

            if jet_idx_key_name in Config.STORYLINE_JET_CHANGES:
                for gwl_story_def, storyline_types_at_gwl in Config.STORYLINE_JET_CHANGES[jet_idx_key_name].items():
                    if gwl_story_def in gwls_to_plot: # Only plot if this GWL is in our main list
                        for storyline_type, storyline_jet_change_value in storyline_types_at_gwl.items():
                            handle = ax.plot(gwl_story_def, storyline_jet_change_value,
                                             marker=storyline_marker_styles.get(storyline_type, 'x'),
                                             color=storyline_colors.get(storyline_type, 'orange'),
                                             linestyle='none', markersize=8, label=storyline_type if storyline_type not in legend_handles_storylines else "")
                            if storyline_type not in legend_handles_storylines:
                                legend_handles_storylines[storyline_type] = handle[0]


            # Formatting
            ax.set_xlabel('Global Warming Level (°C)', fontsize=10)
            y_axis_label = f'Change in Jet Latitude (°N)' if 'Lat' in jet_idx_key_name else f'Change in Jet Speed (m/s)'
            ax.set_ylabel(y_axis_label, fontsize=10)
            ax.set_title(f'Projected Change in {jet_idx_key_name.replace("_", " ")}', fontsize=11)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.axhline(0, color='grey', linewidth=0.8, linestyle='--')
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Create a combined legend
            handles, labels = ax.get_legend_handles_labels() # Get all handles/labels from plot
            # Filter for unique labels to avoid duplicates, prioritize certain items
            unique_legend_items = {}
            for h, l in zip(handles, labels):
                if l not in unique_legend_items : # Prioritize first encountered (e.g. MMM, spread)
                     if l in ['Multi-Model Mean Change', '10-90th Percentile Spread'] or l in storyline_marker_styles:
                          unique_legend_items[l] = h
            
            if unique_legend_items:
                ax.legend(unique_legend_items.values(), unique_legend_items.keys(), fontsize=8, loc='best')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle("CMIP6 Projected Jet Index Changes vs. Global Warming Level", fontsize=14, weight='bold')
        
        output_filename = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved CMIP6 jet changes vs. GWL plot to {output_filename}")
        plt.close(fig)

    @staticmethod
    def plot_storyline_impacts(storyline_impact_data: dict, filename="storyline_impacts.png"):
        """
        Plots the calculated storyline impacts using bar charts.

        Args:
            storyline_impact_data (dict): Output from StorylineAnalyzer.calculate_storyline_impacts.
                                         Format: {GWL: {impact_var: {storyline_type: value}}}
            filename (str): Name of the output plot file.
        """
        logging.info("Plotting storyline impacts...")
        Visualizer.ensure_plot_dir_exists()

        if not storyline_impact_data or not any(storyline_impact_data.values()):
            logging.info("  Cannot plot storyline impacts: No data available.")
            return

        gwl_levels_present = sorted([gwl for gwl in storyline_impact_data.keys() if storyline_impact_data[gwl]])
        if not gwl_levels_present:
            logging.info("  No GWL levels with data to plot for storylines.")
            return

        # Dynamically determine impact variables and storyline types present in the data
        first_gwl_with_data = gwl_levels_present[0]
        impact_vars_present = sorted(list(storyline_impact_data[first_gwl_with_data].keys()))
        if not impact_vars_present:
            logging.info("  No impact variables found in the storyline data for plotting.")
            return
        
        # Infer storyline types from the first valid data point, maintaining a preferred order
        storyline_types_ordered = ['Extreme Low', 'Core Mean', 'Core High', 'Extreme High']
        actual_storyline_types_in_data = []
        try:
            first_impact_var_data = storyline_impact_data[first_gwl_with_data][impact_vars_present[0]]
            # Get types present in data, then order them
            types_in_first_point = list(first_impact_var_data.keys())
            actual_storyline_types_in_data = [st for st in storyline_types_ordered if st in types_in_first_point] + \
                                             [st for st in types_in_first_point if st not in storyline_types_ordered]
        except Exception:
            logging.warning("Could not robustly determine storyline types from data for plotting order.")
            actual_storyline_types_in_data = storyline_types_ordered # Fallback

        if not actual_storyline_types_in_data:
            logging.info("  No storyline types found to plot.")
            return

        num_rows = len(impact_vars_present)
        num_cols = len(gwl_levels_present)
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(5.5 * num_cols, 4.5 * num_rows), 
                                squeeze=False, sharey='row') # Share Y-axis per row

        storyline_bar_colors = {'Core Mean': 'tab:blue', 'Core High': 'tab:cyan', 
                                'Extreme Low': 'tab:purple', 'Extreme High': 'tab:pink', 'Other': 'tab:gray'}

        for r_idx, impact_var_key in enumerate(impact_vars_present):
            # Determine units for Y-axis label
            y_axis_unit = "Change"
            if '_pr' in impact_var_key.lower(): y_axis_unit = "% Change"
            elif '_tas' in impact_var_key.lower(): y_axis_unit = "°C Change"
            elif 'speed' in impact_var_key.lower(): y_axis_unit = "m/s Change"
            elif 'lat' in impact_var_key.lower(): y_axis_unit = "°Lat Change"
            
            if num_cols > 0: # Set Y-label only for the first column in a row
                axs[r_idx, 0].set_ylabel(y_axis_unit, fontsize=10)

            for c_idx, gwl_val in enumerate(gwl_levels_present):
                ax = axs[r_idx, c_idx]
                
                impact_data_for_gwl_var = storyline_impact_data.get(gwl_val, {}).get(impact_var_key, {})
                
                if not impact_data_for_gwl_var:
                    ax.text(0.5, 0.5, "No Data", ha='center', va='center', transform=ax.transAxes, alpha=0.5)
                    ax.set_title(f"{impact_var_key.replace('_', ' ')}\n@ {gwl_val}°C GWL", fontsize=10)
                    ax.set_xticks([]) # No x-ticks if no data
                    continue

                bar_values = [impact_data_for_gwl_var.get(st_type, np.nan) for st_type in actual_storyline_types_in_data]
                bar_colors_list = [storyline_bar_colors.get(st_type, 'gray') for st_type in actual_storyline_types_in_data]

                bars = ax.bar(actual_storyline_types_in_data, bar_values, color=bar_colors_list, width=0.7)
                ax.set_title(f"{impact_var_key.replace('_', ' ')}\n@ {gwl_val}°C GWL", fontsize=10)
                ax.tick_params(axis='x', rotation=45, labelsize=8, pad=-2) # Rotate and adjust padding
                ax.grid(True, axis='y', linestyle=':', alpha=0.7)
                ax.axhline(0, color='black', linewidth=0.8)
                ax.tick_params(axis='y', labelsize=9)

                # Add value labels on bars
                for bar_item in bars:
                    y_val = bar_item.get_height()
                    if not np.isnan(y_val):
                        # Adjust text position based on bar height (positive/negative)
                        vertical_alignment = 'bottom' if y_val >= 0 else 'top'
                        text_offset = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.015 # Small offset
                        text_y_position = y_val + text_offset if y_val >= 0 else y_val - text_offset
                        ax.text(bar_item.get_x() + bar_item.get_width() / 2.0, text_y_position,
                                f'{y_val:.1f}', va=vertical_alignment, ha='center', fontsize=7, weight='bold')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for main title
        plt.suptitle("Storyline Impacts on Regional Climate Variables", fontsize=15, weight='bold')
        
        output_filename = os.path.join(Config.PLOT_DIR, filename)
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        logging.info(f"Saved storyline impacts plot to {output_filename}")
        plt.close(fig)