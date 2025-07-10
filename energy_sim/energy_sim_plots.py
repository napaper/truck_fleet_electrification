"""
Plots during the energy simulation. 

Locations in which the plots can be activated:
plot_alt_profile:               M31_CreateTracks
plot_alt_and_speed_profile:     M31_CreateTracks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime as dt


def plot_alt_profile(scenario, track, altitude_profile, altitude_profile_filter, track_id_map):
    # Can be avtivated in M31_CreateTracks

    # Plot altitude profile
    plt.figure(figsize=(10, 6))
    plt.plot(track.index, altitude_profile, label='Original Altitude')
    plt.plot(track.index, altitude_profile_filter, label='Filtered Altitude', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Altitude')
    track_id = scenario['track_id'][0]
    mapped_id = track_id_map[track_id_map[:, 0] == track_id, 1]
    if mapped_id.size > 0:
        plt.title(f'Altitude Profile of Track: {mapped_id[0]}')
    else:
        plt.title('Altitude Profile Track Unknown')
    plt.legend()
    ax = plt.gca()
    #ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    
    plt.grid(True)

    # Save the figure as PDF
    if mapped_id.size > 0:
        plt.savefig(f'data/output/figures/working_on/altitude_profile_{mapped_id[0]}.pdf', format='pdf')
    else:
        plt.savefig('data/output/figures/working_on/altitude_profile_unknown.pdf', format='pdf')

    #plt.show()


def plot_alt_and_speed_profile(scenario, track, altitude_profile, altitude_profile_filter, track_id_map, speed_profile):
    # Create a single figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot altitude profile on primary axis
    ax1.plot(track.index, altitude_profile, label='Original Altitude', color='blue')
    ax1.plot(track.index, altitude_profile_filter, label='Filtered Altitude', linestyle='--', color='darkblue')
    ax1.set_ylabel('Altitude (m)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    track_id = scenario['track_id'][0]
    mapped_id = track_id_map[track_id_map[:, 0] == track_id, 1]
    if mapped_id.size > 0:
        ax1.set_title(f'Altitude and Speed Profile of Track: {mapped_id[0]}')
    else:
        ax1.set_title('Altitude and Speed Profile of Unknown Track')
    ax1.grid(True)
    
    # Create secondary y-axis for speed profile
    ax2 = ax1.twinx()
    ax2.plot(track.index, speed_profile, label='Speed', color='green', alpha=0.6)
    ax2.set_ylabel('Speed (m/s)', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Format x-axis
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.set_xlabel('Time')
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()

    # Save the figure as PDF
    if mapped_id.size > 0:
        plt.savefig(f'data/output/figures/speed-altitude_profiles/altitude_profile_{mapped_id[0]}.pdf', format='pdf')
    else:
        plt.savefig('data/output/figures/speed-altitude_profiles', format='pdf')

    plt.show()
