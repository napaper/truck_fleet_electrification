"""
Energy simulation plotting functions for truck fleet electrification analysis.

This module provides visualization functions for energy simulation results, including
altitude profiles, speed profiles, and combined track visualizations. These plots
can be activated during track creation and energy simulation processes.

Activation locations:
- plot_alt_profile:               M31_CreateTracks
- plot_alt_and_speed_profile:     M31_CreateTracks

The module generates publication-ready PDF plots with TUM styling and saves them
to appropriate output directories for further analysis and documentation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt


# =============================================================================
# ALTITUDE PROFILE VISUALIZATION
# =============================================================================

def plot_alt_profile(scenario, track, altitude_profile, altitude_profile_filter, track_id_map):
    """
    Generate and save altitude profile visualization for a single track.
    
    This function creates a plot comparing original and filtered altitude profiles
    over time, with proper labeling and formatting for publication use.
    
    Args:
        scenario (dict): Scenario configuration containing track_id information
        track (pd.DataFrame): Track data with time index
        altitude_profile (np.array): Original altitude values
        altitude_profile_filter (np.array): Filtered/processed altitude values
        track_id_map (np.array): Mapping between internal and external track IDs
        
    Returns:
        None: Saves plot to 'output/figures/working_on/altitude_profile_{track_id}.pdf'
        
    Note:
        This function can be activated in M31_CreateTracks for track analysis.
    """
    # Create figure with appropriate dimensions
    plt.figure(figsize=(10, 6))
    
    # Plot both original and filtered altitude profiles
    plt.plot(track.index, altitude_profile, label='Original Altitude')
    plt.plot(track.index, altitude_profile_filter, label='Filtered Altitude', linestyle='--')
    
    # Configure axis labels and title
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    
    # Extract track ID and map to external identifier
    track_id = scenario['track_id'][0]
    mapped_id = track_id_map[track_id_map[:, 0] == track_id, 1]
    
    # Set title based on whether track ID mapping is available
    if mapped_id.size > 0:
        plt.title(f'Altitude Profile of Track: {mapped_id[0]}')
    else:
        plt.title('Altitude Profile Track Unknown')
    
    # Add legend and configure grid
    plt.legend()
    ax = plt.gca()
    ax.grid(True)
    
    # Configure x-axis formatting for time display
    # Uncomment the following line to set major tick intervals to 2 hours
    # ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    # Save the figure as PDF with appropriate filename
    if mapped_id.size > 0:
        output_path = f'output/figures/working_on/altitude_profile_{mapped_id[0]}.pdf'
        plt.savefig(output_path, format='pdf')
    else:
        output_path = 'output/figures/working_on/altitude_profile_unknown.pdf'
        plt.savefig(output_path, format='pdf')

    # Uncomment to display plot interactively
    # plt.show()


# =============================================================================
# COMBINED ALTITUDE AND SPEED PROFILE VISUALIZATION
# =============================================================================

def plot_alt_and_speed_profile(scenario, track, altitude_profile, altitude_profile_filter, 
                              track_id_map, speed_profile):
    """
    Generate and save combined altitude and speed profile visualization.
    
    This function creates a dual-axis plot showing both altitude and speed profiles
    over time, allowing for comprehensive track analysis and comparison.
    
    Args:
        scenario (dict): Scenario configuration containing track_id information
        track (pd.DataFrame): Track data with time index
        altitude_profile (np.array): Original altitude values
        altitude_profile_filter (np.array): Filtered/processed altitude values
        track_id_map (np.array): Mapping between internal and external track IDs
        speed_profile (np.array): Speed values for the track
        
    Returns:
        None: Saves plot to 'output/figures/speed-altitude_profiles/altitude_profile_{track_id}.pdf'
        
    Note:
        This function can be activated in M31_CreateTracks for comprehensive track analysis.
        It creates a publication-ready dual-axis plot with proper TUM styling.
    """
    # Create figure with dual y-axes for altitude and speed
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot altitude profile on primary y-axis (left)
    ax1.plot(track.index, altitude_profile, label='Original Altitude', color='blue')
    ax1.plot(track.index, altitude_profile_filter, label='Filtered Altitude', 
             linestyle='--', color='darkblue')
    
    # Configure primary y-axis (altitude)
    ax1.set_ylabel('Altitude (m)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True)
    
    # Extract track ID and map to external identifier
    track_id = scenario['track_id'][0]
    mapped_id = track_id_map[track_id_map[:, 0] == track_id, 1]
    
    # Set title based on whether track ID mapping is available
    if mapped_id.size > 0:
        ax1.set_title(f'Altitude and Speed Profile of Track: {mapped_id[0]}')
    else:
        ax1.set_title('Altitude and Speed Profile of Unknown Track')
    
    # Create secondary y-axis for speed profile (right)
    ax2 = ax1.twinx()
    ax2.plot(track.index, speed_profile, label='Speed', color='green', alpha=0.6)
    ax2.set_ylabel('Speed (m/s)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Configure x-axis formatting and labels
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel('Time')
    
    # Create combined legend for both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Optimize layout to prevent overlap
    plt.tight_layout()

    # Save the figure as PDF with appropriate filename
    if mapped_id.size > 0:
        output_path = f'output/figures/speed-altitude_profiles/altitude_profile_{mapped_id[0]}.pdf'
        plt.savefig(output_path, format='pdf')
    else:
        # Note: This path appears to be incomplete in the original code
        # Consider adding proper filename for unknown tracks
        output_path = 'output/figures/speed-altitude_profiles/altitude_profile_unknown.pdf'
        plt.savefig(output_path, format='pdf')

    # Display plot interactively
    plt.show()
